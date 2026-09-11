from __future__ import annotations

import json
import mido
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.musicxml_standardize import (
    MusicXMLStandardizationError,
    WorkerEvent,
    WorkerKeySignature,
    WorkerMeasure,
    WorkerPart,
    WorkerPayload,
    WorkerPickup,
    WorkerTempo,
    WorkerTimeSignature,
    _normalize_worker_key,
    _key_sharps,
    _repair_fine_score_events,
    standardize_musicxml_payload,
    standardize_musicxml,
    write_standardized_score,
)
from backend.jianpu_score.musescore_import import (
    MUSESCORE_ORIGIN_SENTINEL_NAME,
    MuseScoreImportError,
    convert_performance_midi,
)
import backend.jianpu_score.musescore_import as musescore_import
from backend.jianpu_score.quantize import JianpuSerializationError, _validate_explicit_ties, score_to_jianpu
from backend.jianpu_score.render import render_score
from backend.jianpu_score.high_accuracy import (
    MUSESCORE_IMPORT_PROFILE_EXPECTED,
    MUSESCORE_IMPORT_PROFILE_SHA256,
    MUSESCORE_VOCAL_IMPORT_PROFILE_EXPECTED,
    MUSESCORE_VOCAL_IMPORT_PROFILE_PATH,
    MUSESCORE_VOCAL_IMPORT_PROFILE_SHA256,
    resolve_musescore,
    resolve_notation_python,
    validate_musescore_import_profile,
)
from backend.jianpu_score.musicxml_standardize import run_musicxml_worker
from scripts.musicxml_score_worker import _duration_details


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json"
PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"
EXTERNAL_READY = (
    resolve_musescore() is not None
    and resolve_notation_python().is_file()
    and PROFILE.is_file()
    and MUSESCORE_VOCAL_IMPORT_PROFILE_PATH.is_file()
)


def _write_test_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(path)


def _fake_musescore_xml() -> str:
    return (
        "<score-partwise version='3.1'><part-list>"
        "<score-part id='P1'><part-name>source</part-name></score-part>"
        f"<score-part id='P2'><part-name>Grand Piano, {MUSESCORE_ORIGIN_SENTINEL_NAME}</part-name></score-part>"
        "</part-list><part id='P1'>"
        "<!-- fixture output --><!-- fixture output --><!-- fixture output -->"
        "<!-- fixture output --><!-- fixture output --><!-- fixture output -->"
        "</part><part id='P2'></part></score-partwise>"
    )


def test_musescore_profile_pins_exact_48_tpq_tuplet_policy() -> None:
    details = validate_musescore_import_profile(PROFILE, expected_sha256=MUSESCORE_IMPORT_PROFILE_SHA256)

    assert details["options"] == MUSESCORE_IMPORT_PROFILE_EXPECTED
    assert details["tuplets"] == {
        "Duplets": True,
        "Triplets": True,
        "Quadruplets": True,
        "Quintuplets": False,
        "Septuplets": False,
        "Nonuplets": False,
    }


def test_musescore_vocal_profile_pins_tempo_preserving_policy() -> None:
    details = validate_musescore_import_profile(
        MUSESCORE_VOCAL_IMPORT_PROFILE_PATH,
        expected_sha256=MUSESCORE_VOCAL_IMPORT_PROFILE_SHA256,
        expected_options=MUSESCORE_VOCAL_IMPORT_PROFILE_EXPECTED,
    )

    assert details["policy"] == "vocal-tempo-preserving"
    assert details["options"]["HumanPerformance"] == "false"
    assert details["options"]["SimplifyDurations"] == "true"
    assert details["tuplets"]["Triplets"] is True
    assert details["tuplets"]["Quintuplets"] is False


@pytest.mark.parametrize("boundary", ["start", "stop", "continue", None])
def test_music21_worker_preserves_explicit_tuplet_boundary(boundary: str | None) -> None:
    tuplet = SimpleNamespace(
        numberNotesActual=3,
        numberNotesNormal=2,
        type=boundary,
    )
    duration = SimpleNamespace(
        quarterLength=1 / 6,
        dots=0,
        tuplets=[tuplet],
        isGrace=False,
    )

    details = _duration_details(SimpleNamespace(duration=duration))

    assert details["tuplet_actual"] == 3
    assert details["tuplet_normal"] == 2
    assert details["tuplet_type"] == boundary


def _musicxml_actual_tuplet_ratios(path: Path) -> set[tuple[int, int]]:
    ratios: set[tuple[int, int]] = set()
    for modification in ET.parse(path).getroot().findall(".//time-modification"):
        actual = modification.findtext("actual-notes")
        normal = modification.findtext("normal-notes")
        if actual and normal:
            ratios.add((int(actual), int(normal)))
    return ratios


def _manual_payload() -> WorkerPayload:
    events = [
        WorkerEvent(
            event_id="dotted",
            kind="note",
            offset_quarter=0,
            duration_quarter=1.5,
            pitches=[60],
            dots=1,
            voice="1",
        ),
        WorkerEvent(
            event_id="short",
            kind="note",
            offset_quarter=1.5,
            duration_quarter=0.125,
            pitches=[62],
            voice="1",
        ),
        WorkerEvent(
            event_id="triplet",
            kind="note",
            offset_quarter=0,
            duration_quarter=1 / 3,
            pitches=[64],
            tuplet_actual=3,
            tuplet_normal=2,
            voice="2",
        ),
        WorkerEvent(
            event_id="chord",
            kind="chord",
            offset_quarter=0,
            duration_quarter=0.5,
            pitches=[67, 71, 74],
            voice="3",
        ),
        WorkerEvent(
            event_id="tied",
            kind="note",
            offset_quarter=0,
            duration_quarter=1,
            pitches=[76],
            tie="start",
            tie_types=["start"],
            voice="4",
        ),
        WorkerEvent(
            event_id="rest",
            kind="rest",
            offset_quarter=0,
            duration_quarter=4,
            voice="5",
        ),
    ]
    return WorkerPayload(
        schema_version="1.0",
        worker="music21",
        music21_version="9.9.2",
        source_path="manual.musicxml",
        title="Manual notation fixture",
        highest_time_quarter=4,
        parts=[
            WorkerPart(
                part_id="P1",
                name="Piano",
                instrument="Acoustic Piano",
                highest_time_quarter=4,
                events=events,
                measures=[
                    WorkerMeasure(
                        part_index=0,
                        number=1,
                        start_quarter=0,
                        duration_quarter=4,
                        end_quarter=4,
                        time_signature="4/4",
                    )
                ],
            )
        ],
        measures=[
            WorkerMeasure(
                part_index=0,
                number=1,
                start_quarter=0,
                duration_quarter=4,
                end_quarter=4,
                time_signature="4/4",
            )
        ],
        tempo_events=[WorkerTempo(offset_quarter=0, bpm=120), WorkerTempo(offset_quarter=2, bpm=96)],
        time_signature_events=[WorkerTimeSignature(offset_quarter=0, ratio="4/4", numerator=4, denominator=4)],
        key_signature_events=[WorkerKeySignature(offset_quarter=0, key="D", sharps=2)],
    )


def _tuplet_marker_payload(events: list[WorkerEvent]) -> WorkerPayload:
    payload = _manual_payload()
    payload.highest_time_quarter = 4
    payload.parts[0].highest_time_quarter = 4
    payload.parts[0].events = events
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=4,
            end_quarter=4,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)
    return payload


def test_score_normalizer_preserves_complete_same_voice_tuplet_markers() -> None:
    payload = _tuplet_marker_payload(
        [
            WorkerEvent(
                event_id="same-start",
                kind="note",
                offset_quarter=0,
                duration_quarter=1 / 6,
                pitches=[60],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="start",
                voice="1",
            ),
            WorkerEvent(
                event_id="same-middle",
                kind="note",
                offset_quarter=1 / 6,
                duration_quarter=1 / 6,
                pitches=[62],
                tuplet_actual=3,
                tuplet_normal=2,
                voice="1",
            ),
            WorkerEvent(
                event_id="same-stop",
                kind="note",
                offset_quarter=1 / 3,
                duration_quarter=1 / 6,
                pitches=[64],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="stop",
                voice="1",
            ),
            WorkerEvent(event_id="same-tail", kind="rest", offset_quarter=0.5, duration_quarter=3.5, voice="1"),
        ]
    )

    score, report = standardize_musicxml_payload(payload)

    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert [(event.start_tick, event.duration_tick, event.tuplet_type) for event in notes] == [
        (0, 8, "start"),
        (8, 8, None),
        (16, 8, "stop"),
    ]
    assert report["tuplet_marker_repairs"] == []
    assert score_to_jianpu(score)


@pytest.mark.parametrize("boundary", ["start", "stop"])
def test_score_normalizer_clears_serializable_isolated_tuplet_boundary(boundary: str) -> None:
    payload = _tuplet_marker_payload(
        [
            WorkerEvent(
                event_id="orphan",
                kind="note",
                offset_quarter=0,
                duration_quarter=0.25,
                pitches=[60],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type=boundary,
                voice="1",
            ),
            WorkerEvent(event_id="orphan-tail", kind="rest", offset_quarter=0.25, duration_quarter=3.75, voice="1"),
        ]
    )

    score, report = standardize_musicxml_payload(payload)

    note = next(event for voice in score.voices for event in voice.events if event.midi == 60)
    assert note.tuplet_actual is None
    assert note.tuplet_normal is None
    assert note.tuplet_type is None
    assert [item["reason"] for item in report["tuplet_marker_repairs"]] == ["orphan_tuplet_marker_cleared"]
    assert report["tuplet_marker_repairs"][0]["original_marker"]["tuplet_type"] == boundary
    assert score_to_jianpu(score)


def test_score_normalizer_rejoins_cross_voice_tuplet_fragment_without_moving_timing() -> None:
    payload = _tuplet_marker_payload(
        [
            WorkerEvent(
                event_id="cross-start",
                kind="note",
                offset_quarter=0,
                duration_quarter=1 / 6,
                pitches=[60],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="start",
                voice="1",
            ),
            WorkerEvent(
                event_id="cross-stop",
                kind="note",
                offset_quarter=1 / 6,
                duration_quarter=1 / 12,
                pitches=[62],
                tie="start",
                tie_types=["start"],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="stop",
                voice="2",
            ),
            WorkerEvent(
                event_id="cross-successor",
                kind="note",
                offset_quarter=1 / 4,
                duration_quarter=1 / 4,
                pitches=[62],
                tie="stop",
                tie_types=["stop"],
                voice="2",
            ),
            WorkerEvent(event_id="cross-tail", kind="rest", offset_quarter=0.5, duration_quarter=3.5, voice="2"),
            # An unrelated complete same-voice group must not be mistaken for
            # the stop belonging to cross-start merely because it shares the
            # part/staff/ratio context.
            WorkerEvent(
                event_id="later-start",
                kind="note",
                offset_quarter=1,
                duration_quarter=1 / 6,
                pitches=[64],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="start",
                voice="1",
            ),
            WorkerEvent(
                event_id="later-middle",
                kind="note",
                offset_quarter=7 / 6,
                duration_quarter=1 / 6,
                pitches=[65],
                tuplet_actual=3,
                tuplet_normal=2,
                voice="1",
            ),
            WorkerEvent(
                event_id="later-stop",
                kind="note",
                offset_quarter=4 / 3,
                duration_quarter=1 / 6,
                pitches=[67],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="stop",
                voice="1",
            ),
        ]
    )

    score, report = standardize_musicxml_payload(payload)

    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert [(event.midi, event.start_tick, event.end_tick) for event in notes] == [
        (60, 0, 8),
        (62, 8, 12),
        (62, 12, 24),
        (64, 48, 56),
        (65, 56, 64),
        (67, 64, 72),
    ]
    assert len({event.voice_id for event in notes}) == 1
    repair = report["tuplet_marker_repairs"][0]
    assert repair["reason"] == "cross_voice_tuplet_marker_reassigned"
    assert repair["voice"] == "2"
    assert repair["target_voice"] == "1"
    assert repair["start_tick"] == 8
    assert repair["end_tick"] == 12
    assert score_to_jianpu(score)


def test_score_normalizer_reassembles_complete_tuplet_split_by_tied_chord_voice_repair() -> None:
    payload = _tuplet_marker_payload(
        [
            WorkerEvent(
                event_id="lead-chord",
                kind="chord",
                offset_quarter=0,
                duration_quarter=3 / 8,
                pitches=[60, 80],
                tie_types=["start", None],
                voice="1",
            ),
            WorkerEvent(
                event_id="bridge-chord",
                kind="chord",
                offset_quarter=3 / 8,
                duration_quarter=1 / 8,
                pitches=[60, 81],
                tie_types=["continue", None],
                voice="2",
            ),
            WorkerEvent(
                event_id="source-tuplet-start",
                kind="note",
                offset_quarter=1 / 2,
                duration_quarter=1 / 6,
                pitches=[60],
                tie="continue",
                tie_types=["continue"],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="start",
                voice="2",
            ),
            WorkerEvent(
                event_id="source-tuplet-middle",
                kind="chord",
                offset_quarter=2 / 3,
                duration_quarter=1 / 6,
                pitches=[60, 64],
                tie_types=["stop", None],
                tuplet_actual=3,
                tuplet_normal=2,
                voice="2",
            ),
            WorkerEvent(
                event_id="source-tuplet-stop",
                kind="note",
                offset_quarter=5 / 6,
                duration_quarter=1 / 6,
                pitches=[65],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="stop",
                voice="2",
            ),
            WorkerEvent(event_id="tail", kind="rest", offset_quarter=1, duration_quarter=3, voice="1"),
        ]
    )

    score, report = standardize_musicxml_payload(payload)

    repair = next(
        item for item in report["tuplet_marker_repairs"] if item["reason"] == "cross_voice_tuplet_marker_reassembled"
    )
    assert repair["voice"] == "2"
    assert repair["target_voice"] == "1"
    assert (repair["start_tick"], repair["end_tick"]) == (24, 48)
    assert (repair["actual_ticks"], repair["nominal_ticks"]) == (24, 36)
    assert repair["source_event_ids"] == [
        "source-tuplet-start",
        "source-tuplet-middle",
        "source-tuplet-stop",
    ]
    notes = [
        event
        for voice in score.voices
        for event in voice.events
        if event.metadata.get("musicxml_event_id", "").startswith("source-tuplet")
    ]
    assert [(event.start_tick, event.end_tick, event.chord_pitches, event.tuplet_type) for event in notes] == [
        (24, 32, [60], "start"),
        (32, 40, [60, 64], None),
        (40, 48, [65], "stop"),
    ]
    assert score_to_jianpu(score)


def test_score_normalizer_does_not_hide_same_voice_tuplet_gap() -> None:
    payload = _tuplet_marker_payload(
        [
            WorkerEvent(
                event_id="gap-start",
                kind="note",
                offset_quarter=0,
                duration_quarter=1 / 6,
                pitches=[60],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="start",
                voice="1",
            ),
            WorkerEvent(
                event_id="gap-stop",
                kind="note",
                offset_quarter=0.25,
                duration_quarter=1 / 12,
                pitches=[62],
                tuplet_actual=3,
                tuplet_normal=2,
                tuplet_type="stop",
                voice="1",
            ),
        ]
    )

    score, report = standardize_musicxml_payload(payload)

    assert report["tuplet_marker_repairs"] == []
    with pytest.raises(JianpuSerializationError, match="gap or inconsistent ratio"):
        score_to_jianpu(score)


def test_score_normalizer_preserves_notation_fields_and_more_than_four_voices() -> None:
    score, report = standardize_musicxml_payload(_manual_payload())

    assert score.quarter_ticks == 48
    assert score.total_ticks == 192
    assert score.key == "D"
    assert score.time_signature == "4/4"
    assert [event.start_tick for event in score.tempo_events] == [0, 96]
    assert len(score.voices) == 5
    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert any(event.dots == 1 and event.duration_tick == 72 for event in notes)
    assert not any(item["reason"].startswith("explicit_dots_") for item in report["notation_grid_repairs"])
    assert any(event.tuplet_actual == 3 and event.tuplet_normal == 2 and event.duration_tick == 16 for event in notes)
    chord = next(event for event in notes if event.chord_pitches == [67, 71, 74])
    assert chord.midi == 67
    tied = next(event for event in notes if event.midi == 76)
    assert tied.tie == "start"
    assert tied.tie_types == ["start"]
    assert any(event.is_rest for voice in score.voices for event in voice.events)
    assert report["score_voice_count"] == 5
    assert score.metadata["measure_total_ticks"] == score.total_ticks
    assert score.metadata["measure_duration_total_ticks"] == score.total_ticks


def test_score_normalizer_rejoins_tie_fragments_exposed_in_different_music21_voices() -> None:
    payload = _manual_payload()
    payload.parts[0].events = [
        WorkerEvent(
            event_id="tie-start",
            kind="note",
            offset_quarter=0,
            duration_quarter=1,
            pitches=[60],
            tie="start",
            tie_types=["start"],
            voice="2",
        ),
        WorkerEvent(
            event_id="voice-two-rest",
            kind="rest",
            offset_quarter=1,
            duration_quarter=3,
            voice="2",
        ),
        WorkerEvent(
            event_id="tie-stop",
            kind="note",
            offset_quarter=1,
            duration_quarter=0.25,
            pitches=[60],
            tie="stop",
            tie_types=["stop"],
            voice="1",
        ),
        WorkerEvent(
            event_id="voice-one-rest",
            kind="rest",
            offset_quarter=0,
            duration_quarter=4,
            voice="1",
        ),
    ]

    score, report = standardize_musicxml_payload(payload)

    tied = [
        (voice.voice_id, event)
        for voice in score.voices
        for event in voice.events
        if event.midi == 60 and event.tie_types
    ]
    assert [(event.tie, event.start_tick, event.end_tick) for _voice, event in tied] == [
        ("start", 0, 48),
        ("stop", 48, 60),
    ]
    assert len({voice_id for voice_id, _event in tied}) == 1
    repairs = [item for item in report["repairs"] if item.get("reason") == "tie_chain_voice_reassigned"]
    assert repairs == [
        {
            "reason": "tie_chain_voice_reassigned",
            "musicxml_event_id": "tie-stop",
            "source_voice": "1",
            "source_staff": 1,
            "target_voice": "2",
            "target_staff": 1,
            "pitches": [60],
        }
    ]
    assert report["tie_voice_repairs"] == repairs
    assert "tie fragments were normalized" in " ".join(score.warnings)
    jianpu = score_to_jianpu(score)
    assert "~" in jianpu


def test_score_normalizer_keeps_repaired_tie_stop_ahead_of_overlapping_filler() -> None:
    """An incoming tie must claim its contiguous lane before a same-tick rest."""

    payload = _manual_payload()
    payload.parts[0].events = [
        WorkerEvent(
            event_id="tie-start",
            kind="note",
            offset_quarter=1,
            duration_quarter=1,
            pitches=[64],
            tie="start",
            tie_types=["start"],
            voice="5",
        ),
        WorkerEvent(
            event_id="tie-stop",
            kind="note",
            offset_quarter=2,
            duration_quarter=1,
            pitches=[64],
            tie="stop",
            tie_types=["stop"],
            voice="6",
        ),
        WorkerEvent(
            event_id="overlapping-filler",
            kind="rest",
            offset_quarter=2,
            duration_quarter=0.5,
            voice="5",
        ),
        WorkerEvent(
            event_id="following-note",
            kind="note",
            offset_quarter=2.5,
            duration_quarter=0.5,
            pitches=[55],
            voice="5",
        ),
        WorkerEvent(
            event_id="tail",
            kind="rest",
            offset_quarter=3,
            duration_quarter=1,
            voice="5",
        ),
    ]

    score, report = standardize_musicxml_payload(payload)

    tied = [
        (voice, event)
        for voice in score.voices
        for event in voice.events
        if event.midi == 64 and event.tie_types
    ]
    assert len(tied) == 2
    assert len({voice.voice_id for voice, _event in tied}) == 1
    tie_voice, _ = tied[0]
    assert tie_voice.staff == 1
    assert [(event.midi, event.start_tick, event.end_tick, event.tie) for _voice, event in tied] == [
        (64, 48, 96, "start"),
        (64, 96, 144, "stop"),
    ]
    assert report["tie_voice_repairs"] == [
        {
            "reason": "tie_chain_voice_reassigned",
            "musicxml_event_id": "tie-stop",
            "source_voice": "6",
            "source_staff": 1,
            "target_voice": "5",
            "target_staff": 1,
            "pitches": [64],
        }
    ]
    _validate_explicit_ties(tie_voice)
    assert "~" in score_to_jianpu(score)


def test_score_normalizer_keeps_a_normal_cross_measure_tie_in_one_voice() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 8
    payload.parts[0].highest_time_quarter = 8
    measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=4,
            end_quarter=4,
            time_signature="4/4",
        ),
        WorkerMeasure(
            part_index=0,
            number=2,
            start_quarter=4,
            duration_quarter=4,
            end_quarter=8,
            time_signature=None,
        ),
    ]
    payload.measures = measures
    payload.parts[0].measures = measures
    payload.parts[0].events = [
        WorkerEvent(
            event_id="tie-start",
            kind="note",
            offset_quarter=3.5,
            duration_quarter=0.5,
            pitches=[60],
            tie="start",
            tie_types=["start"],
            voice="1",
        ),
        WorkerEvent(
            event_id="tie-stop",
            kind="note",
            offset_quarter=4,
            duration_quarter=1,
            pitches=[60],
            tie="stop",
            tie_types=["stop"],
            voice="1",
        ),
        WorkerEvent(
            event_id="rest",
            kind="rest",
            offset_quarter=5,
            duration_quarter=3,
            voice="1",
        ),
    ]

    score, report = standardize_musicxml_payload(payload)

    assert report["tie_voice_repairs"] == []
    assert "~" in score_to_jianpu(score)


def test_score_normalizer_restores_same_pitch_overlap_into_another_voice() -> None:
    payload = _manual_payload()
    payload.parts[0].events = [
        WorkerEvent(event_id="first", kind="note", offset_quarter=0, duration_quarter=1, pitches=[60]),
        WorkerEvent(event_id="second", kind="note", offset_quarter=1, duration_quarter=2, pitches=[60]),
    ]
    payload.highest_time_quarter = 4
    source = {
        "notes": [
            {"index": 0, "midi": 60, "start_tick": 0, "end_tick": 960},
            {"index": 1, "midi": 60, "start_tick": 480, "end_tick": 1440},
        ]
    }
    score, report = standardize_musicxml_payload(payload, performance_metadata=source)

    note_voices = [
        voice
        for voice in score.voices
        if any(event.midi == 60 for event in voice.events)
    ]
    assert len(note_voices) == 2
    assert sorted(
        (event.start_tick, event.duration_tick)
        for voice in note_voices
        for event in voice.events
        if event.midi == 60
    ) == [(0, 96), (48, 96)]
    assert any(item["reason"] == "musescore_truncated_source_span_restored" for item in report["source_to_score"])
    assert any(item["reason"] == "overlapping_events_allocated_to_additional_score_voice" for item in report["repairs"])


def test_alignment_keeps_normal_musicxml_adaptive_timing() -> None:
    payload = _manual_payload()
    payload.parts[0].events = [
        WorkerEvent(
            event_id="adaptive",
            kind="note",
            offset_quarter=0.0625,
            duration_quarter=0.5,
            pitches=[60],
        )
    ]
    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={
            "notes": [{"index": 0, "midi": 60, "start_tick": 0, "end_tick": 240}]
        },
    )

    note = next(event for voice in score.voices for event in voice.events if event.midi == 60)
    alignment = report["source_to_score"][0]
    assert (note.start_tick, note.duration_tick) == (3, 24)
    assert alignment["reason"] == "matched_musicxml_event"
    assert alignment["source_to_score_movement_start_ticks"] == 3
    assert alignment["source_to_score_movement_end_ticks"] == 3
    assert alignment["musicxml_to_score_movement_start_ticks"] == 0
    assert alignment["musicxml_to_score_movement_end_ticks"] == 0


def test_standardizer_preserves_source_score_origin_and_undetermined_warning() -> None:
    payload = _manual_payload()
    source_origin = {
        "strategy": "downbeat_phase_undetermined",
        "downbeat_status": "undetermined",
        "origin_shift_beats": 0.0,
        "timeline_offset_beats": 0.25,
    }
    source_warning = "无法确认弱起；保留共享 BeatNet 原点"
    score, _report = standardize_musicxml_payload(
        payload,
        performance_metadata={
            "score_origin": source_origin,
            "score_timeline_offset_beats": 0.25,
            "downbeat_status": "undetermined",
            "downbeat_warning": source_warning,
        },
    )

    assert score.metadata["score_origin"] == source_origin
    assert score.metadata["source_score_origin"] == source_origin
    assert score.metadata["score_timeline_offset_beats"] == pytest.approx(0.25)
    assert score.metadata["source_score_timeline_offset_beats"] == pytest.approx(0.25)
    assert score.metadata["downbeat_status"] == "undetermined"
    assert score.metadata["downbeat_warning"] == source_warning
    assert source_warning in score.warnings


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("C", 0),
        ("C#", 7),
        ("Db", -5),
        ("D", 2),
        ("Eb", -3),
        ("E", 4),
        ("F", -1),
        ("F#", 6),
        ("Gb", -6),
        ("G", 1),
        ("Ab", -4),
        ("A", 3),
        ("Bb", -2),
        ("B", 5),
        ("Cm", -3),
        ("C#m", 4),
        ("Dbm", 4),
        ("Dm", -1),
        ("Ebm", -6),
        ("Em", 1),
        ("Fm", -4),
        ("F#m", 3),
        ("Gbm", 3),
        ("Gm", -2),
        ("Abm", -7),
        ("Am", 0),
        ("Bbm", -5),
        ("Bm", 2),
    ],
)
def test_key_signature_sharps_cover_all_application_keys(key: str, expected: int) -> None:
    assert _key_sharps(key) == expected


def test_key_signature_uses_midi_emitted_enharmonic_key_without_changing_display() -> None:
    assert _key_sharps("Dbm", emitted_key="C#m") == 4
    assert _key_sharps("Gbm", emitted_key="F#m") == 3
    assert _key_sharps("F#m") == 3
    assert _key_sharps("Cm") == -3
    assert _key_sharps("Am") == 0


@pytest.mark.parametrize(
    ("worker_key", "expected"),
    [
        ("D-", "Db"),
        ("A-", "Ab"),
        ("E-", "Eb"),
        ("B-", "Bb"),
        ("G-", "Gb"),
        ("D- major", "Db"),
        ("b- minor", "Bbm"),
        ("F# minor", "F#m"),
        ("a major", "A"),
        ("Bbm", "Bbm"),
    ],
)
def test_normalize_worker_key_accepts_music21_flat_and_mode_aliases(worker_key: str, expected: str) -> None:
    assert _normalize_worker_key(worker_key) == expected


@pytest.mark.parametrize("worker_key", ["C-", "F-", "D--", "D- mystery", "D-flat"])
def test_normalize_worker_key_does_not_widen_unsupported_or_malformed_keys(worker_key: str) -> None:
    with pytest.raises(MusicXMLStandardizationError, match="unsupported MusicXML key signature"):
        _normalize_worker_key(worker_key)


def test_music21_flat_key_is_reconciled_to_source_bbm_and_keeps_negative_fifths() -> None:
    payload = _manual_payload()
    payload.key_signature_events = [WorkerKeySignature(offset_quarter=0, key="D-", sharps=-5)]
    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={"key": "Bbm", "time_signature": "4/4"},
    )

    assert score.key == "Bbm"
    assert score.metadata["key_signature_events"][0] == {"start_tick": 0, "key": "Bbm", "sharps": -5}
    reconciliation = next(item for item in report["conductor_reconciliation"] if item["field"] == "key")
    assert reconciliation["musicxml_value"] == "D-"
    assert reconciliation["production_value"] == "Bbm"
    assert reconciliation["final_value"] == "Bbm"


def test_production_emitted_key_only_controls_signature_number() -> None:
    score, report = standardize_musicxml_payload(
        _manual_payload(),
        performance_metadata={
            "key": "Dbm",
            "emitted_key": "C#m",
            "time_signature": "4/4",
        },
    )

    assert score.key == "Dbm"
    assert score.metadata["key_signature_events"][0]["key"] == "Dbm"
    assert score.metadata["key_signature_events"][0]["sharps"] == 4
    assert any(
        item.get("emitted_value") == "C#m"
        for item in report["conductor_reconciliation"]
        if item["field"] == "key"
    )


def test_unmatched_non_overlapping_source_note_is_an_explicit_error() -> None:
    with pytest.raises(
        MusicXMLStandardizationError,
        match=r"count=1; index=17,midi=127",
    ):
        standardize_musicxml_payload(
            _manual_payload(),
            performance_metadata={
                "notes": [
                    {"index": 17, "midi": 127, "start_tick": 0, "end_tick": 480}
                ]
            },
        )


def test_early_musicxml_end_is_not_treated_as_overlap_truncation() -> None:
    payload = _manual_payload()
    payload.parts[0].events = [
        WorkerEvent(event_id="early", kind="note", offset_quarter=0, duration_quarter=0.5, pitches=[60]),
        WorkerEvent(event_id="next", kind="note", offset_quarter=1, duration_quarter=1, pitches=[60]),
    ]
    source = {
        "notes": [
            {"index": 0, "midi": 60, "start_tick": 0, "end_tick": 960},
            {"index": 1, "midi": 60, "start_tick": 480, "end_tick": 1440},
        ]
    }
    score, report = standardize_musicxml_payload(payload, performance_metadata=source)

    first = next(
        event
        for voice in score.voices
        for event in voice.events
        if event.midi == 60 and event.start_tick == 0
    )
    first_alignment = next(item for item in report["source_to_score"] if item["source_index"] == 0)
    assert first.duration_tick == 24
    assert first_alignment["reason"] == "matched_musicxml_event"
    assert first_alignment["source_to_score_movement_end_ticks"] == -72


@pytest.mark.parametrize("second_start", [1, 3])
def test_timeline_measure_invariant_rejects_overlap_or_gap(second_start: int) -> None:
    payload = _manual_payload()
    payload.measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=2,
            end_quarter=2,
            time_signature="4/4",
        ),
        WorkerMeasure(
            part_index=0,
            number=2,
            start_quarter=second_start,
            duration_quarter=4 - second_start,
            end_quarter=4,
            time_signature=None,
        ),
    ]
    with pytest.raises(MusicXMLStandardizationError, match="illegal (overlap|gap)"):
        standardize_musicxml_payload(payload)


def test_non_exact_48_tpq_duration_is_rejected_explicitly() -> None:
    payload = _manual_payload()
    payload.parts[0].events = [
        WorkerEvent(event_id="unsupported-tuplet", kind="note", offset_quarter=0, duration_quarter=0.2, pitches=[60])
    ]
    with pytest.raises(MusicXMLStandardizationError, match="cannot be represented exactly"):
        standardize_musicxml_payload(payload)


def test_score_normalizer_clears_inconsistent_dots_before_bounded_fragment_repair() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 24
    payload.parts[0].highest_time_quarter = 24
    payload.parts[0].events = [
        WorkerEvent(
            event_id="logical-start",
            kind="note",
            offset_quarter=21,
            duration_quarter=1,
            pitches=[72],
            tie="start",
            tie_types=["start"],
        ),
        WorkerEvent(
            event_id="rounded-dotted-fragment",
            kind="note",
            offset_quarter=22,
            duration_quarter=0.09375,
            pitches=[72],
            tie="stop",
            tie_types=["stop"],
            dots=1,
        ),
        WorkerEvent(
            event_id="following-fine-fragment",
            kind="note",
            offset_quarter=22.09375,
            duration_quarter=0.03125,
            pitches=[72],
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=24,
            end_quarter=24,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)

    score, report = standardize_musicxml_payload(payload)

    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert [(event.midi, event.start_tick, event.end_tick, event.dots, event.tie) for event in notes] == [
        (72, 1008, 1056, 0, "start"),
        (72, 1056, 1060, 0, "stop"),
        (72, 1060, 1062, 0, None),
    ]
    reasons = [item["reason"] for item in report["notation_grid_repairs"]]
    assert reasons == [
        "explicit_dots_cleared_after_duration_validation",
        "fine_grid_singleton_encoded_as_explicit_tuplet",
        "fine_grid_singleton_encoded_as_explicit_tuplet",
    ]
    dot_repair = report["notation_grid_repairs"][0]
    assert dot_repair["musicxml_event_id"] == "rounded-dotted-fragment"
    assert dot_repair["original_dots"] == 1
    assert dot_repair["repaired_dots"] == 0
    assert dot_repair["notated_duration_ticks"] == 4
    assert "explicit dot hints" in " ".join(score.warnings)
    assert "Independent 1/2/4/5-tick" in " ".join(score.warnings)
    for voice in score.voices:
        _validate_explicit_ties(voice)
    assert score_to_jianpu(score)


def test_tiny_fragment_after_tie_repair_closes_one_tick_gap_before_tuplet_inference() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 24
    payload.parts[0].highest_time_quarter = 24
    payload.parts[0].events = [
        WorkerEvent(
            event_id="leading-tie-start",
            kind="note",
            offset_quarter=16,
            duration_quarter=2,
            pitches=[76],
            tie="start",
            tie_types=["start"],
            voice="6",
        ),
        WorkerEvent(
            event_id="long-tie-continue",
            kind="note",
            offset_quarter=18,
            duration_quarter=2,
            pitches=[76],
            tie="continue",
            tie_types=["continue"],
            voice="6",
        ),
        WorkerEvent(
            event_id="rounded-tie-stop",
            kind="note",
            offset_quarter=20,
            duration_quarter=0.09375,
            pitches=[76],
            tie="stop",
            tie_types=["stop"],
            dots=1,
            voice="6",
        ),
        WorkerEvent(
            event_id="following-tiny-chord",
            kind="chord",
            offset_quarter=20.09375,
            duration_quarter=0.03125,
            pitches=[40, 53, 57],
            voice="6",
        ),
        WorkerEvent(
            event_id="following-rest",
            kind="rest",
            offset_quarter=20.125,
            duration_quarter=0.125,
            voice="6",
        ),
        WorkerEvent(
            event_id="following-rest-2",
            kind="rest",
            offset_quarter=20.25,
            duration_quarter=0.125,
            voice="6",
        ),
        WorkerEvent(
            event_id="tail",
            kind="rest",
            offset_quarter=20.375,
            duration_quarter=3.625,
            voice="6",
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=24,
            end_quarter=24,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)

    score, report = standardize_musicxml_payload(payload)
    voice = next(voice for voice in score.voices if voice.source_voice == "6")
    assert all(left.end_tick == right.start_tick for left, right in zip(voice.events, voice.events[1:]))
    tiny = next(event for event in voice.events if event.metadata.get("musicxml_event_id") == "following-tiny-chord")
    assert (tiny.start_tick, tiny.end_tick) == (964, 966)
    repairs = report["notation_grid_repairs"]
    assert [item["reason"] for item in repairs] == [
        "explicit_dots_cleared_after_duration_validation",
        "fine_grid_singleton_encoded_as_explicit_tuplet",
        "fine_grid_fragment_encoded_as_explicit_tuplet",
    ]
    singleton = repairs[1]
    assert singleton["musicxml_event_ids"] == ["rounded-tie-stop"]
    assert singleton["movement_ticks"] == 0
    assert singleton["action"] == "annotate_exact_fine_grid_singleton_tuplet"
    inferred = repairs[2]
    assert inferred["musicxml_event_ids"] == ["following-tiny-chord", "following-rest", "following-rest-2"]
    assert inferred["movement_ticks"] == 0
    assert score_to_jianpu(score)


def test_finer_binary_musescore_fragment_is_bounded_to_48_tpq_with_alignment_diagnostic() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 16
    payload.parts[0].highest_time_quarter = 16
    payload.parts[0].events = [
        WorkerEvent(
            event_id="supported-32nd",
            kind="note",
            offset_quarter=15.75,
            duration_quarter=0.1875,
            pitches=[60],
            tie="start",
            tie_types=["start"],
        ),
        WorkerEvent(
            event_id="finer-fragment-1",
            kind="note",
            offset_quarter=15.9375,
            duration_quarter=0.03125,
            pitches=[60],
            tie="stop",
            tie_types=["stop"],
        ),
        WorkerEvent(
            event_id="finer-fragment-2",
            kind="note",
            offset_quarter=15.96875,
            duration_quarter=0.03125,
            pitches=[67],
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=16,
            end_quarter=16,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)
    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={
            "notes": [
                {"index": 0, "midi": 60, "start_tick": 7560, "end_tick": 7660},
                {"index": 1, "midi": 67, "start_tick": 7660, "end_tick": 7680},
            ]
        },
    )
    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert [(event.midi, event.start_tick, event.end_tick) for event in notes] == [
        (60, 756, 765),
        (60, 765, 766),
        (67, 766, 768),
    ]
    assert notes[0].tie == "start"
    assert notes[0].tie_types == ["start"]
    for voice in score.voices:
        _validate_explicit_ties(voice)
    assert score_to_jianpu(score)
    repairs = report["fine_grid_quantization"]
    assert len(repairs) == 2
    assert {item["original_quarter"] for item in repairs} == {15.96875}
    assert all(abs(float(item["movement_ticks"])) <= 0.5 for item in repairs)
    notation_repairs = report["notation_grid_repairs"]
    assert [item["reason"] for item in notation_repairs] == [
        "fine_grid_singleton_encoded_as_explicit_tuplet",
        "fine_grid_singleton_encoded_as_explicit_tuplet",
    ]
    assert all(abs(int(item["movement_ticks"])) <= 2 for item in notation_repairs)
    assert [(event.start_tick, event.end_tick) for event in notes] == [(756, 765), (765, 766), (766, 768)]
    assert report["source_to_score"][0]["score_end_tick"] == 766
    assert report["source_to_score"][0]["musicxml_to_score_movement_end_ticks"] == 0
    assert report["source_to_score"][1]["score_start_tick"] == 766
    assert report["source_to_score"][1]["musicxml_to_score_movement_start_ticks"] == 0
    assert "Finer binary MusicXML fragments" in " ".join(score.warnings)
    assert "bounded jianpu atom repairs" in " ".join(score.warnings)


def test_finer_binary_three_fragment_tie_keeps_stop_for_existing_chain() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 16
    payload.parts[0].highest_time_quarter = 16
    payload.parts[0].events = [
        WorkerEvent(
            event_id="tie-start",
            kind="note",
            offset_quarter=15,
            duration_quarter=0.75,
            pitches=[60],
            tie="start",
            tie_types=["start"],
        ),
        WorkerEvent(
            event_id="tie-continue",
            kind="note",
            offset_quarter=15.75,
            duration_quarter=0.1875,
            pitches=[60],
            tie="continue",
            tie_types=["continue"],
        ),
        WorkerEvent(
            event_id="tie-stop-fine",
            kind="note",
            offset_quarter=15.9375,
            duration_quarter=0.03125,
            pitches=[60],
            tie="stop",
            tie_types=["stop"],
        ),
        WorkerEvent(
            event_id="following-fine",
            kind="note",
            offset_quarter=15.96875,
            duration_quarter=0.03125,
            pitches=[55],
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=16,
            end_quarter=16,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)

    score, report = standardize_musicxml_payload(payload)

    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert [(event.midi, event.start_tick, event.end_tick, event.tie, event.tie_types) for event in notes] == [
        (60, 720, 756, "start", ["start"]),
        (60, 756, 765, "continue", ["continue"]),
        (60, 765, 766, "stop", ["stop"]),
        (55, 766, 768, None, []),
    ]
    assert [item["reason"] for item in report["notation_grid_repairs"]] == [
        "fine_grid_singleton_encoded_as_explicit_tuplet",
        "fine_grid_singleton_encoded_as_explicit_tuplet",
    ]
    for voice in score.voices:
        _validate_explicit_ties(voice)
    assert score_to_jianpu(score)


def test_explicit_tuplet_tie_fragment_keeps_boundaries_and_following_events() -> None:
    """A tied 2-tick tuplet member must not be merged into its predecessor."""

    payload = _manual_payload()
    payload.highest_time_quarter = 24
    payload.parts[0].highest_time_quarter = 24
    payload.parts[0].events = [
        WorkerEvent(
            event_id="tuple-tie-start",
            kind="note",
            offset_quarter=33 / 8,
            duration_quarter=1 / 8,
            pitches=[75],
            tie="start",
            tie_types=["start"],
            voice="2",
        ),
        WorkerEvent(
            event_id="tuple-tie-stop",
            kind="note",
            offset_quarter=17 / 4,
            duration_quarter=1 / 24,
            pitches=[75],
            tie="stop",
            tie_types=["stop"],
            tuplet_actual=3,
            tuplet_normal=2,
            tuplet_type="start",
            voice="2",
        ),
        WorkerEvent(
            event_id="tuple-rest",
            kind="rest",
            offset_quarter=103 / 24,
            duration_quarter=1 / 24,
            tuplet_actual=3,
            tuplet_normal=2,
            voice="2",
        ),
        WorkerEvent(
            event_id="tuple-note",
            kind="note",
            offset_quarter=13 / 3,
            duration_quarter=1 / 8,
            pitches=[68],
            tuplet_actual=3,
            tuplet_normal=2,
            voice="2",
        ),
        WorkerEvent(
            event_id="tuple-stop",
            kind="note",
            offset_quarter=107 / 24,
            duration_quarter=1 / 24,
            pitches=[72],
            tie="start",
            tie_types=["start"],
            tuplet_actual=3,
            tuplet_normal=2,
            tuplet_type="stop",
            voice="2",
        ),
        WorkerEvent(
            event_id="following-tie-stop",
            kind="note",
            offset_quarter=9 / 2,
            duration_quarter=1 / 8,
            pitches=[72],
            tie="stop",
            tie_types=["stop"],
            voice="2",
        ),
    ]
    measure = WorkerMeasure(
        part_index=0,
        number=1,
        start_quarter=0,
        duration_quarter=24,
        end_quarter=24,
        time_signature="4/4",
    )
    payload.parts[0].measures = [measure]
    payload.measures = [measure]

    score, report = standardize_musicxml_payload(payload)

    voice = next(voice for voice in score.voices if voice.source_voice == "2")
    events = [
        event
        for event in voice.events
        if event.metadata.get("musicxml_event_id") in {
            "tuple-tie-start",
            "tuple-tie-stop",
            "tuple-rest",
            "tuple-note",
            "tuple-stop",
            "following-tie-stop",
        }
    ]
    assert [(event.start_tick, event.end_tick) for event in events] == [
        (198, 204),
        (204, 206),
        (206, 208),
        (208, 214),
        (214, 216),
        (216, 222),
    ]
    assert [(event.tie, event.tie_types) for event in events if event.midi is not None] == [
        ("start", ["start"]),
        ("stop", ["stop"]),
        (None, []),
        ("start", ["start"]),
        ("stop", ["stop"]),
    ]
    assert [(event.tuplet_type, event.tuplet_actual, event.tuplet_normal) for event in events[1:5]] == [
        ("start", 3, 2),
        (None, 3, 2),
        (None, 3, 2),
        ("stop", 3, 2),
    ]
    assert all(left.end_tick == right.start_tick for left, right in zip(voice.events, voice.events[1:]))
    assert report["notation_grid_repairs"] == []
    assert score_to_jianpu(score)


def test_finer_binary_standalone_note_uses_exact_singleton_tuplet() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 8
    payload.parts[0].highest_time_quarter = 8
    payload.parts[0].events = [
        WorkerEvent(
            event_id="standalone-fine-note",
            kind="note",
            offset_quarter=0,
            duration_quarter=0.03125,
            pitches=[67],
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=8,
            end_quarter=8,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)

    score, report = standardize_musicxml_payload(payload)

    note = next(event for voice in score.voices for event in voice.events if event.midi == 67)
    assert (note.start_tick, note.duration_tick, note.end_tick) == (0, 2, 2)
    assert (note.tuplet_actual, note.tuplet_normal, note.tuplet_type) == (3, 1, "start")
    repair = next(item for item in report["notation_grid_repairs"] if item["musicxml_event_ids"] == ["standalone-fine-note"])
    assert repair["reason"] == "fine_grid_singleton_encoded_as_explicit_tuplet"
    assert repair["nominal_duration_ticks"] == [6]
    assert repair["movement_ticks"] == 0
    assert repair["timing_preserved"] is True
    assert repair["original_start_tick"] == 0
    assert repair["original_end_tick"] == 2
    for voice in score.voices:
        _validate_explicit_ties(voice)


@pytest.mark.parametrize(
    ("duration_tick", "midi"),
    [(1, 60), (2, None), (3, 61), (4, 62), (5, None)],
)
def test_fine_grid_singleton_note_and_rest_durations_are_exact(
    duration_tick: int,
    midi: int | None,
) -> None:
    event_id = f"fine-single-{duration_tick}"
    voice = ScoreVoice(
        voice_id="fine-singleton",
        source_voice="1",
        events=[
            ScoreNote(
                start_tick=0,
                duration_tick=duration_tick,
                midi=midi,
                metadata={"musicxml_event_id": event_id},
            )
        ],
    )
    voices, repairs = _repair_fine_score_events([voice], [], total_ticks=duration_tick)
    repaired = voices[0].events[0]
    assert (repaired.start_tick, repaired.duration_tick, repaired.end_tick) == (0, duration_tick, duration_tick)

    score = Score(
        title="fine-grid singleton",
        bpm=120,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=duration_tick,
        voices=voices,
        metadata={
            "timeline_measures": [
                {
                    "start_tick": 0,
                    "duration_tick": duration_tick,
                    "end_tick": duration_tick,
                    "time_signature": "4/4",
                }
            ]
        },
    )
    serialized = score_to_jianpu(score)
    if duration_tick in {1, 2, 4, 5}:
        assert (repaired.tuplet_actual, repaired.tuplet_normal, repaired.tuplet_type) == (3, 1, "start")
        assert "3:1[" in serialized
        assert [item["musicxml_event_ids"] for item in repairs] == [[event_id]]
        assert repairs[0]["duration_ticks"] == [duration_tick]
        assert repairs[0]["nominal_duration_ticks"] == [duration_tick * 3]
        assert repairs[0]["movement_ticks"] == 0
    else:
        assert repaired.tuplet_actual is None
        assert repaired.tuplet_normal is None
        assert "3:1[" not in serialized
        assert repairs == []


@pytest.mark.skipif(
    not (ROOT / "vendor" / "jianpu-ly" / "jianpu-ly.py").is_file()
    or not (ROOT / "tools" / "lilypond-2.24.4" / "bin" / "lilypond.exe").is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)
def test_fine_grid_singletons_render_exact_note_intervals(tmp_path: Path) -> None:
    """The visible 3:1 brackets must survive the complete LilyPond MIDI path."""

    def event(start_tick: int, duration_tick: int, midi: int | None) -> ScoreNote:
        metadata: dict[str, object] = {}
        kwargs: dict[str, object] = {}
        if duration_tick in {1, 2, 4, 5}:
            kwargs.update(tuplet_actual=3, tuplet_normal=1, tuplet_type="start")
            metadata.update(fine_grid_tuplet=True, fine_grid_tuplet_single=True)
        return ScoreNote(
            start_tick=start_tick,
            duration_tick=duration_tick,
            midi=midi,
            metadata=metadata,
            **kwargs,
        )

    events: list[ScoreNote] = []
    cursor = 0
    for duration_tick, midi in ((1, None), (2, 60), (3, None), (4, 62), (5, None)):
        events.append(event(cursor, duration_tick, midi))
        cursor += duration_tick
    for duration_tick, midi in ((144, 64), (24, None), (9, 65)):
        events.append(ScoreNote(start_tick=cursor, duration_tick=duration_tick, midi=midi))
        cursor += duration_tick

    score = Score(
        title="fine-grid singleton render smoke",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=cursor,
        voices=[ScoreVoice(voice_id="fine-grid", events=events)],
    )
    serialized = score_to_jianpu(score)
    assert serialized.count("3:1[") == 4

    artifacts = render_score(score, tmp_path, basename="fine-grid-singletons")
    assert artifacts.svg_paths
    midi = mido.MidiFile(artifacts.midi_path)
    scale = midi.ticks_per_beat // score.quarter_ticks
    expected = [(60, 1, 3), (62, 6, 10), (64, 15, 159), (65, 183, 192)]
    intervals: list[tuple[int, int, int]] = []
    note_track_end: int | None = None
    for track in midi.tracks:
        absolute = 0
        active: dict[int, list[int]] = {}
        track_intervals: list[tuple[int, int, int]] = []
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity:
                active.setdefault(message.note, []).append(absolute)
            elif message.type in {"note_off", "note_on"} and not message.velocity:
                starts = active.get(message.note, [])
                if starts:
                    track_intervals.append((message.note, starts.pop(0), absolute))
        if track_intervals:
            intervals.extend(track_intervals)
            note_track_end = absolute

    assert intervals == [(pitch, start * scale, end * scale) for pitch, start, end in expected]
    assert note_track_end == score.total_ticks * scale


def test_fine_grid_singleton_at_tie_boundary_does_not_consume_tied_neighbor() -> None:
    voice = ScoreVoice(
        voice_id="fine-tie-boundary",
        source_voice="1",
        events=[
            ScoreNote(
                start_tick=0,
                duration_tick=2,
                midi=None,
                metadata={"musicxml_event_id": "boundary-rest"},
            ),
            ScoreNote(
                start_tick=2,
                duration_tick=4,
                midi=60,
                tie="start",
                tie_types=["start"],
                metadata={"musicxml_event_id": "tie-start"},
            ),
            ScoreNote(
                start_tick=6,
                duration_tick=3,
                midi=60,
                tie="stop",
                tie_types=["stop"],
                metadata={"musicxml_event_id": "tie-stop"},
            ),
        ],
    )
    voices, repairs = _repair_fine_score_events([voice], [], total_ticks=9)
    events = voices[0].events
    assert [event.metadata["musicxml_event_id"] for event in events] == ["boundary-rest", "tie-start", "tie-stop"]
    assert [(event.start_tick, event.end_tick) for event in events] == [(0, 2), (2, 6), (6, 9)]
    assert events[0].metadata["fine_grid_tuplet_single"] is True
    assert events[1].metadata["fine_grid_tuplet_single"] is True
    assert events[1].tie == "start" and events[2].tie == "stop"
    assert [item["musicxml_event_ids"] for item in repairs] == [["boundary-rest"], ["tie-start"]]

    score = Score(
        title="fine-grid tie boundary",
        bpm=120,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=9,
        voices=voices,
        metadata={
            "timeline_measures": [{"start_tick": 0, "duration_tick": 9, "end_tick": 9, "time_signature": "4/4"}]
        },
    )
    assert "3:1[" in score_to_jianpu(score)


def _fine_grid_tuplet_fixture_payload() -> WorkerPayload:
    payload = _manual_payload()
    payload.highest_time_quarter = 16
    payload.parts[0].highest_time_quarter = 16
    # MuseScore's 1/32-quarter fragments round to 2 and 1 ticks at 48 TPQ.
    # The first two events are an exact 3:2 fragment pair (6+2 ticks); the
    # final 1+3 pair needs the explicit fine-grid 3:1 encoding.
    payload.parts[0].events = [
        WorkerEvent(
            event_id="triplet-rest-6",
            kind="rest",
            offset_quarter=15.75,
            duration_quarter=0.125,
            pitches=[],
        ),
        WorkerEvent(
            event_id="triplet-rest-2",
            kind="rest",
            offset_quarter=15.875,
            duration_quarter=0.03125,
            pitches=[],
        ),
        WorkerEvent(
            event_id="fine-chord-1",
            kind="chord",
            offset_quarter=15.90625,
            duration_quarter=0.03125,
            pitches=[57, 60, 65],
        ),
        WorkerEvent(
            event_id="fine-rest-3",
            kind="rest",
            offset_quarter=15.9375,
            duration_quarter=0.0625,
            pitches=[],
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=16,
            end_quarter=16,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)
    return payload


def test_fine_grid_fragments_use_exact_tuplet_serialization_without_timing_move() -> None:
    score, report = standardize_musicxml_payload(_fine_grid_tuplet_fixture_payload())
    voice = next(voice for voice in score.voices if voice.source_voice == "1")
    events = [
        event
        for event in voice.events
        if event.start_tick >= 750 and (event.midi is not None or event.is_rest)
    ]
    events = [event for event in events if event.start_tick < 768]
    assert [(event.start_tick, event.duration_tick, event.midi, event.tuplet_actual, event.tuplet_normal, event.tuplet_type) for event in events] == [
        (756, 6, None, 3, 2, "start"),
        (762, 2, None, 3, 2, "stop"),
        (764, 1, 57, 3, 1, "start"),
        (765, 3, None, 3, 1, "stop"),
    ]
    repairs = [item for item in report["notation_grid_repairs"] if item["reason"] == "fine_grid_fragment_encoded_as_explicit_tuplet"]
    assert [item["tuplet_actual"] for item in repairs] == [3, 3]
    assert all(item["timing_preserved"] and item["movement_ticks"] == 0 for item in repairs)
    assert score.total_ticks == 768
    serialized = score_to_jianpu(score)
    assert "3[" in serialized
    assert "3:1[" in serialized


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore/music21/jianpu-ly/LilyPond toolchain unavailable")
def test_fine_grid_tuplet_fixture_renders_through_jianpu_and_lilypond(tmp_path: Path) -> None:
    score, _report = standardize_musicxml_payload(_fine_grid_tuplet_fixture_payload())
    artifacts = render_score(score, tmp_path, basename="fine-grid-tuplets")
    assert Path(artifacts.jly_path).is_file()
    assert Path(artifacts.lilypond_path).is_file()
    assert artifacts.svg_paths and all(Path(path).is_file() for path in artifacts.svg_paths)


def test_fine_grid_tuplet_repair_keeps_adjacent_group_closed_and_non_overlapping() -> None:
    # The middle 1-tick note has a following 2-tick rest.  Before the guard,
    # the ordinary atom repair consumed that rest after the 3:1 group had been
    # inferred, deleting its stop member and making the next group appear
    # nested at the same tick.
    source = [
        ("e78", 0, 12, None),
        ("e79", 12, 2, 76),
        ("e80", 14, 1, 75),
        ("e81", 15, 1, 74),
        ("e82", 16, 2, None),
        ("e83", 18, 2, 73),
        ("e84", 20, 1, 72),
        ("e85", 21, 1, 83),
    ]
    voice = ScoreVoice(
        voice_id="fine-grid-regression",
        source_voice="1",
        events=[
            ScoreNote(
                start_tick=start,
                duration_tick=duration,
                midi=midi,
                metadata={"musicxml_event_id": event_id},
            )
            for event_id, start, duration, midi in source
        ],
    )

    voices, repairs = _repair_fine_score_events([voice], [], total_ticks=22)
    events = voices[0].events
    assert [event.metadata["musicxml_event_id"] for event in events] == [item[0] for item in source]
    assert [(event.start_tick, event.end_tick, event.tuplet_type) for event in events] == [
        (0, 12, "start"),
        (12, 14, "stop"),
        (14, 15, "start"),
        (15, 16, None),
        (16, 18, "stop"),
        (18, 20, "start"),
        (20, 21, None),
        (21, 22, "stop"),
    ]
    group_ids = [event.metadata["fine_grid_tuplet_group_id"] for event in events]
    assert len(set(group_ids)) == 3
    assert [item["musicxml_event_ids"] for item in repairs] == [
        ["e78", "e79"],
        ["e80", "e81", "e82"],
        ["e83", "e84", "e85"],
    ]

    score = Score(
        title="fine-grid regression",
        bpm=120,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=22,
        voices=voices,
        metadata={
            "timeline_measures": [
                {"start_tick": 0, "duration_tick": 22, "end_tick": 22, "time_signature": "4/4"}
            ]
        },
    )
    assert "3:1[" in score_to_jianpu(score)


def test_finer_binary_chord_tie_slots_clear_only_merged_pitches() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 16
    payload.parts[0].highest_time_quarter = 16
    payload.parts[0].events = [
        WorkerEvent(
            event_id="chord-start",
            kind="chord",
            offset_quarter=15.75,
            duration_quarter=0.1875,
            pitches=[60, 64],
            tie_types=["start", "start"],
        ),
        WorkerEvent(
            event_id="chord-stop-fine",
            kind="chord",
            offset_quarter=15.9375,
            duration_quarter=0.03125,
            pitches=[60, 64],
            tie_types=["stop", "stop"],
        ),
        WorkerEvent(
            event_id="following-fine",
            kind="note",
            offset_quarter=15.96875,
            duration_quarter=0.03125,
            pitches=[55],
        ),
    ]
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=16,
            end_quarter=16,
            time_signature="4/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)

    score, _report = standardize_musicxml_payload(payload)

    notes = [event for voice in score.voices for event in voice.events if event.midi is not None]
    assert notes[0].chord_pitches == [60, 64]
    assert [(event.midi, event.start_tick, event.end_tick, event.tie, event.tie_types) for event in notes] == [
        (60, 756, 765, "start", ["start", "start"]),
        (60, 765, 766, "stop", ["stop", "stop"]),
        (55, 766, 768, None, []),
    ]
    for voice in score.voices:
        _validate_explicit_ties(voice)
    assert score_to_jianpu(score)


def test_score_normalizer_retains_pickup_and_meter_key_tempo_changes_in_metadata() -> None:
    payload = _manual_payload()
    payload.highest_time_quarter = 6
    payload.parts[0].highest_time_quarter = 6
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=0,
            start_quarter=0,
            duration_quarter=2,
            end_quarter=2,
            time_signature="4/4",
            is_pickup=True,
        ),
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=2,
            duration_quarter=4,
            end_quarter=6,
            time_signature="3/4",
        ),
    ]
    payload.measures = list(payload.parts[0].measures)
    payload.pickup = WorkerPickup(is_pickup=True, duration_quarter=2, measure_number=0)
    payload.time_signature_events = [
        WorkerTimeSignature(offset_quarter=0, ratio="4/4", numerator=4, denominator=4),
        WorkerTimeSignature(offset_quarter=2, ratio="3/4", numerator=3, denominator=4),
    ]
    payload.key_signature_events = [
        WorkerKeySignature(offset_quarter=0, key="D", sharps=2),
        WorkerKeySignature(offset_quarter=2, key="G", sharps=1),
    ]
    score, _report = standardize_musicxml_payload(payload)

    assert score.total_ticks == 288
    assert score.metadata["pickup"] == {"is_pickup": True, "duration_tick": 96, "measure_number": 0}
    assert [(item["start_tick"], item["time_signature"]) for item in score.metadata["time_signature_events"]] == [
        (0, "4/4"),
        (96, "3/4"),
    ]
    assert [(item["start_tick"], item["key"]) for item in score.metadata["key_signature_events"]] == [
        (0, "D"),
        (96, "G"),
    ]
    assert score.metadata["key_signature_events"][0]["sharps"] == 2
    assert score.metadata["measure_total_ticks"] == 288


def test_production_conductor_metadata_backfills_initial_values_only() -> None:
    payload = _manual_payload()
    payload.time_signature_events = [
        WorkerTimeSignature(offset_quarter=0, ratio="4/4", numerator=4, denominator=4),
        WorkerTimeSignature(offset_quarter=2, ratio="3/4", numerator=3, denominator=4),
    ]
    payload.key_signature_events = [
        WorkerKeySignature(offset_quarter=0, key="C", sharps=0),
        WorkerKeySignature(offset_quarter=2, key="G", sharps=1),
    ]
    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={
            "key": "D",
            "time_signature": "4/4",
            "tempo_points": [{"tick": 0, "bpm": 96}],
        },
    )

    assert score.key == "D"
    assert [(item["start_tick"], item["key"]) for item in score.metadata["key_signature_events"]] == [
        (0, "D"),
        (96, "G"),
    ]
    assert [(item["start_tick"], item["time_signature"]) for item in score.metadata["time_signature_events"]] == [
        (0, "4/4"),
        (96, "3/4"),
    ]
    assert score.bpm == pytest.approx(96)
    assert any(item["field"] == "key" for item in report["conductor_reconciliation"])


def test_production_meter_hint_rebars_conflicting_imported_timeline() -> None:
    payload = _tuplet_marker_payload(
        [WorkerEvent(event_id="rest", kind="rest", offset_quarter=0, duration_quarter=3, voice="1")]
    )
    payload.highest_time_quarter = 3
    payload.parts[0].highest_time_quarter = 3
    payload.parts[0].measures = [
        WorkerMeasure(
            part_index=0,
            number=1,
            start_quarter=0,
            duration_quarter=3,
            end_quarter=3,
            time_signature="3/4",
        )
    ]
    payload.measures = list(payload.parts[0].measures)
    payload.time_signature_events = [WorkerTimeSignature(offset_quarter=0, ratio="3/4", numerator=3, denominator=4)]
    payload.pickup = WorkerPickup(is_pickup=False, duration_quarter=3, measure_number=1)

    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={"time_signature": "4/4"},
    )

    assert score.time_signature == "4/4"
    assert score.metadata["time_signature_events"] == [{
        "start_tick": 0,
        "time_signature": "4/4",
        "numerator": 4,
        "denominator": 4,
    }]
    assert any(
        item["reason"] == "production_metadata_replaced_changed_initial_time_signature"
        and item["production_value"] == "4/4"
        and item["final_value"] == "4/4"
        for item in report["conductor_reconciliation"]
    )
    assert report["meter_rebar"]["applied"] is True
    assert report["meter_rebar"]["imported_timeline"][0]["time_signature"] == "3/4"
    assert report["meter_rebar"]["final_timeline"][0]["time_signature"] == "4/4"
    assert score.metadata["pickup"]["duration_tick"] == 0
    assert score.total_ticks == 192
    assert report["meter_rebar"]["terminal_padding"]["duration_tick"] == 48
    assert score_to_jianpu(score)


def test_production_meter_rebar_splits_cross_measure_chord_and_preserves_ties() -> None:
    payload = _tuplet_marker_payload(
        [
            WorkerEvent(
                event_id="crossing-chord",
                kind="chord",
                offset_quarter=1,
                duration_quarter=4,
                pitches=[60, 64],
                voice="1",
            ),
            WorkerEvent(event_id="leading-rest", kind="rest", offset_quarter=0, duration_quarter=1, voice="1"),
            WorkerEvent(event_id="trailing-rest", kind="rest", offset_quarter=5, duration_quarter=3, voice="1"),
        ]
    )
    payload.highest_time_quarter = 8
    payload.parts[0].highest_time_quarter = 8
    payload.parts[0].measures = [
        WorkerMeasure(part_index=0, number=1, start_quarter=0, duration_quarter=3, end_quarter=3, time_signature="3/4"),
        WorkerMeasure(part_index=0, number=2, start_quarter=3, duration_quarter=3, end_quarter=6, time_signature="3/4"),
        WorkerMeasure(part_index=0, number=3, start_quarter=6, duration_quarter=2, end_quarter=8, time_signature="3/4"),
    ]
    payload.measures = list(payload.parts[0].measures)
    payload.time_signature_events = [WorkerTimeSignature(offset_quarter=0, ratio="3/4", numerator=3, denominator=4)]

    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={"time_signature": "4/4", "manual_time_signature_override": True},
    )

    voice = next(voice for voice in score.voices if voice.source_voice == "1")
    chord_segments = [event for event in voice.events if event.chord_pitches == [60, 64]]
    assert [(event.start_tick, event.duration_tick, event.tie_types) for event in chord_segments] == [
        (48, 144, ["start", "start"]),
        (192, 48, ["stop", "stop"]),
    ]
    assert sum(event.duration_tick for event in chord_segments) == 192
    _validate_explicit_ties(voice)
    assert report["meter_rebar"]["event_split_count"] == 2
    assert score_to_jianpu(score)


@pytest.mark.parametrize(
    ("meter", "expected_starts"),
    [
        ("2/4", [0, 96, 192, 288]),
        ("3/4", [0, 144, 288]),
        ("4/4", [0, 192]),
        ("6/8", [0, 144, 288]),
    ],
)
def test_production_manual_meter_rebar_supports_supported_meters(
    meter: str, expected_starts: list[int]
) -> None:
    payload = _tuplet_marker_payload([WorkerEvent(event_id="rest", kind="rest", offset_quarter=0, duration_quarter=8, voice="1")])
    payload.highest_time_quarter = 8
    payload.parts[0].highest_time_quarter = 8
    payload.parts[0].measures = [
        WorkerMeasure(part_index=0, number=1, start_quarter=0, duration_quarter=4, end_quarter=4, time_signature="4/4"),
        WorkerMeasure(part_index=0, number=2, start_quarter=4, duration_quarter=4, end_quarter=8, time_signature="4/4"),
    ]
    payload.measures = list(payload.parts[0].measures)
    payload.time_signature_events = [WorkerTimeSignature(offset_quarter=0, ratio="4/4", numerator=4, denominator=4)]

    score, _report = standardize_musicxml_payload(
        payload,
        performance_metadata={"time_signature": meter, "manual_time_signature_override": True},
    )

    assert score.time_signature == meter
    assert [item["start_tick"] for item in score.metadata["timeline_measures"]] == expected_starts
    assert all(item["time_signature"] == meter for item in score.metadata["timeline_measures"])
    assert score_to_jianpu(score)


def test_production_meter_change_inside_nominal_bar_rebars_and_splits_event() -> None:
    payload = _tuplet_marker_payload(
        [WorkerEvent(event_id="long-note", kind="note", offset_quarter=0, duration_quarter=4, pitches=[60], voice="1")]
    )
    payload.highest_time_quarter = 4
    payload.parts[0].highest_time_quarter = 4
    payload.parts[0].measures = [
        WorkerMeasure(part_index=0, number=1, start_quarter=0, duration_quarter=4, end_quarter=4, time_signature="4/4")
    ]
    payload.measures = list(payload.parts[0].measures)
    payload.time_signature_events = [
        WorkerTimeSignature(offset_quarter=0, ratio="4/4", numerator=4, denominator=4),
        WorkerTimeSignature(offset_quarter=2, ratio="3/4", numerator=3, denominator=4),
    ]

    score, report = standardize_musicxml_payload(payload, performance_metadata={"time_signature": "4/4"})

    assert [(item["start_tick"], item["end_tick"], item["time_signature"]) for item in score.metadata["timeline_measures"]] == [
        (0, 96, "4/4"),
        (96, 240, "3/4"),
    ]
    note_segments = [event for voice in score.voices for event in voice.events if event.midi == 60]
    assert [(event.start_tick, event.duration_tick, event.tie) for event in note_segments] == [
        (0, 96, "start"),
        (96, 96, "stop"),
    ]
    assert report["meter_rebar"]["final_timeline"][0]["rebar_reason"] == "meter_change_inside_nominal_bar"
    assert report["meter_rebar"]["event_split_count"] == 2
    for voice in score.voices:
        _validate_explicit_ties(voice)
    assert score_to_jianpu(score)


def test_musescore_adapter_reports_missing_pinned_executable_without_fallback(tmp_path: Path) -> None:
    midi = tmp_path / "source.mid"
    _write_test_midi(midi)
    with pytest.raises(MuseScoreImportError, match="executable is unavailable"):
        convert_performance_midi(
            midi,
            tmp_path / "output.musicxml",
            musescore_path=tmp_path / "missing-MuseScore4.exe",
        )


def test_musescore_cli_calls_are_serialized_within_one_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "MuseScore4.exe"
    profile = tmp_path / "profile.xml"
    source_a = tmp_path / "a.mid"
    source_b = tmp_path / "b.mid"
    executable.write_bytes(b"stub")
    profile.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    _write_test_midi(source_a)
    _write_test_midi(source_b)
    xml = _fake_musescore_xml()
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()
    intervals: list[tuple[int, int]] = []

    def fake_run(command, **_kwargs):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            started = active
        assert musescore_import.MUSESCORE_CLI_LOCK.locked()
        try:
            destination = Path(command[command.index("-o") + 1])
            destination.write_text(xml, encoding="utf-8")
        finally:
            with state_lock:
                active -= 1
                intervals.append((started, active))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(musescore_import.subprocess, "run", fake_run)
    barrier = threading.Barrier(2)

    def convert(source: Path, destination: Path) -> object:
        barrier.wait(timeout=2)
        return convert_performance_midi(
            source,
            destination,
            instrument_id=source.stem,
            musescore_path=executable,
            profile_path=profile,
            overwrite=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(convert, source_a, tmp_path / "a.musicxml"),
            pool.submit(convert, source_b, tmp_path / "b.musicxml"),
        ]
        artifacts = [future.result(timeout=5) for future in futures]

    assert len(artifacts) == 2
    assert maximum_active == 1
    assert len(intervals) == 2
    assert intervals == [(1, 0), (1, 0)]


def test_musescore_transient_crash_is_retried_once_with_fresh_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "MuseScore4.exe"
    profile = tmp_path / "profile.xml"
    source = tmp_path / "source.mid"
    destination = tmp_path / "result.musicxml"
    executable.write_bytes(b"stub")
    profile.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    _write_test_midi(source)
    xml = _fake_musescore_xml()
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 3221225477, stdout="", stderr="Crashpad")
        Path(command[command.index("-o") + 1]).write_text(xml, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(musescore_import.subprocess, "run", fake_run)
    monkeypatch.setattr(musescore_import.time, "sleep", lambda _seconds: None)

    artifact = convert_performance_midi(
        source,
        destination,
        instrument_id="transient",
        musescore_path=executable,
        profile_path=profile,
    )

    assert destination.is_file()
    assert len(calls) == 2
    assert [returncode for _, returncode in artifact.attempts] == [3221225477, 0]
    assert artifact.command[artifact.command.index("-o") + 1] == str(destination.resolve())
    assert all(Path(command[command.index("-o") + 1]) != destination for command in calls)
    assert not list(tmp_path.glob(".result.musescore-*.musicxml"))


def test_musescore_transient_crash_after_retry_is_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "MuseScore4.exe"
    profile = tmp_path / "profile.xml"
    source = tmp_path / "source.mid"
    destination = tmp_path / "result.musicxml"
    executable.write_bytes(b"stub")
    profile.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    _write_test_midi(source)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 3221225477, stdout="", stderr="Crashpad")

    monkeypatch.setattr(musescore_import.subprocess, "run", fake_run)
    monkeypatch.setattr(musescore_import.time, "sleep", lambda _seconds: None)

    with pytest.raises(MuseScoreImportError, match=r"attempts:.*returncode=3221225477"):
        convert_performance_midi(
            source,
            destination,
            instrument_id="transient-failure",
            musescore_path=executable,
            profile_path=profile,
        )

    assert len(calls) == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".result.musescore-*.musicxml"))


def test_musescore_nontransient_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "MuseScore4.exe"
    profile = tmp_path / "profile.xml"
    source = tmp_path / "source.mid"
    destination = tmp_path / "result.musicxml"
    executable.write_bytes(b"stub")
    profile.write_text(PROFILE.read_text(encoding="utf-8"), encoding="utf-8")
    _write_test_midi(source)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="ordinary failure")

    monkeypatch.setattr(musescore_import.subprocess, "run", fake_run)

    with pytest.raises(MuseScoreImportError, match=r"failed \(1\)"):
        convert_performance_midi(
            source,
            destination,
            instrument_id="ordinary-failure",
            musescore_path=executable,
            profile_path=profile,
        )

    assert len(calls) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".result.musescore-*.musicxml"))


def test_drum_performance_is_explicitly_midi_only(tmp_path: Path) -> None:
    with pytest.raises(MusicXMLStandardizationError, match="MIDI-only"):
        standardize_musicxml(
            tmp_path / "drums.musicxml",
            performance_metadata={"is_drum": True, "drum_jianpu_policy": "midi_only"},
        )


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore and notation environment are unavailable")
def test_vocal_profile_preserves_sparse_performance_positions(tmp_path: Path) -> None:
    """HumanPerformance reflows a sparse line; the vocal profile preserves its beat map."""

    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("time_signature", numerator=2, denominator=4, time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60), time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(conductor)
    notes = [
        (1140, 1589, 53),
        (1590, 1990, 65),
        (1990, 2073, 63),
        (2073, 2291, 62),
        (2291, 2356, 57),
        (2356, 2434, 58),
        (2434, 2596, 60),
        (2596, 2694, 62),
    ]
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="vocal", time=0))
    previous = 0
    for start, end, pitch in notes:
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=start - previous))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=end - start))
        previous = end
    midi.tracks.append(track)
    midi_path = tmp_path / "sparse-vocal.performance.mid"
    midi.save(midi_path)

    outputs: dict[str, list[float]] = {}
    for label, profile in (
        ("human", PROFILE),
        ("vocal", MUSESCORE_VOCAL_IMPORT_PROFILE_PATH),
    ):
        musicxml = tmp_path / f"{label}.musicxml"
        convert_performance_midi(
            midi_path,
            musicxml,
            instrument_id=f"sparse-{label}",
            profile_path=profile,
        )
        payload = run_musicxml_worker(musicxml)
        outputs[label] = [
            event.offset_quarter
            for part in payload.parts
            for event in part.events
            if event.pitches
        ]

    # 1140/480 = 2.375.  Human-performance mode starts on a new inferred
    # quarter, while the vocal profile keeps the performance onset and lets
    # the normalizer use the production tempo map later.
    assert outputs["vocal"][0] == pytest.approx(2.375)
    assert outputs["human"][0] != pytest.approx(2.375)
    assert outputs["human"][0] == pytest.approx(2.0)


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore and notation environment are unavailable")
def test_real_musescore_music21_fixture_preserves_triplet_tie_chord_and_staff(tmp_path: Path) -> None:
    from scripts.generate_stage56_fixture import build_production_fixture

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    midi_bytes, performance_metadata = build_production_fixture(payload)
    midi = tmp_path / "stage56.performance.mid"
    musicxml = tmp_path / "stage56.musicxml"
    score_json = tmp_path / "stage56.score.json"
    alignment_json = tmp_path / "stage56.alignment.json"
    midi.write_bytes(midi_bytes)
    converted = convert_performance_midi(midi, musicxml, instrument_id="stage56")
    assert converted.musicxml_path == musicxml.resolve()
    score, report = standardize_musicxml(
        musicxml,
        performance_metadata=performance_metadata,
        title="Stage 56 fixture",
    )
    artifact = write_standardized_score(
        musicxml,
        score_json,
        alignment_report_path=alignment_json,
        performance_metadata=performance_metadata,
        title="Stage 56 fixture",
    )

    assert musicxml.stat().st_size > 200
    assert score.quarter_ticks == 48
    assert score.bpm == pytest.approx(96.0)
    assert score.key == "D"
    assert score.time_signature == "6/8"
    assert score.total_ticks == 288
    assert {voice.staff for voice in score.voices} >= {1, 2}
    source_pitches = {int(note["midi"]) for note in payload["notes"]}
    score_pitches = {
        pitch
        for voice in score.voices
        for event in voice.events
        for pitch in (event.chord_pitches or ([event.midi] if event.midi is not None else []))
    }
    assert source_pitches <= score_pitches
    assert any(event.tuplet_actual == 3 and event.tuplet_normal == 2 for voice in score.voices for event in voice.events)
    assert any(event.chord_pitches == [76, 78] for voice in score.voices for event in voice.events)
    assert any(event.tie_types for voice in score.voices for event in voice.events)
    assert report["musicxml_event_count"] >= 18
    assert report["source_note_count"] == len(payload["notes"])
    assert any(item["field"] == "key" for item in report["conductor_reconciliation"])
    assert artifact.score.model_dump(mode="json") == json.loads(score_json.read_text(encoding="utf-8"))
    assert json.loads(alignment_json.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert score.metadata["measure_total_ticks"] == score.total_ticks


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore and notation environment are unavailable")
def test_stage56_cli_smoke_writes_independent_outputs(tmp_path: Path) -> None:
    from scripts.generate_stage56_fixture import build_production_fixture

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    midi_bytes, performance_metadata = build_production_fixture(payload)
    midi = tmp_path / "cli.performance.mid"
    musicxml = tmp_path / "cli.musicxml"
    score_json = tmp_path / "cli.score.json"
    alignment_json = tmp_path / "cli.alignment.json"
    metadata_json = tmp_path / "cli.performance.metadata.json"
    midi.write_bytes(midi_bytes)
    metadata_json.write_text(json.dumps(performance_metadata), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "musicxml_to_score.py"),
            "--midi",
            str(midi),
            "--musicxml",
            str(musicxml),
            "--score-json",
            str(score_json),
            "--alignment-json",
            str(alignment_json),
            "--performance-metadata",
            str(metadata_json),
            "--instrument-id",
            "cli-fixture",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["score_ticks_per_quarter"] == 48
    assert musicxml.is_file() and score_json.is_file() and alignment_json.is_file()
    ratios = _musicxml_actual_tuplet_ratios(musicxml)
    assert (3, 2) in ratios
    assert all(actual not in {5, 7, 9} for actual, _normal in ratios)


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore and notation environment are unavailable")
def test_musescore_profile_disables_nonrepresentable_tuplets_causally(tmp_path: Path) -> None:
    from scripts.high_accuracy_fixture_smoke import (
        _convert,
        _musicxml_tuplet_ratios,
        _write_enabled_unsupported_tuplet_profile,
        _write_tuplet_stress_midi,
    )

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    stress_midi = tmp_path / "tuplet-policy-stress.mid"
    disabled_musicxml = tmp_path / "disabled.musicxml"
    _write_tuplet_stress_midi(stress_midi, fixture["tuplet_stress"])
    _convert(resolve_musescore(), PROFILE, stress_midi, disabled_musicxml, tmp_path)
    disabled_ratios = _musicxml_tuplet_ratios(disabled_musicxml)
    assert all(actual not in {5, 7, 9} for actual, _normal in disabled_ratios)

    enabled_profile = tmp_path / "unsupported-enabled.xml"
    enabled_musicxml = tmp_path / "enabled.musicxml"
    _write_enabled_unsupported_tuplet_profile(PROFILE, enabled_profile)
    _convert(resolve_musescore(), enabled_profile, stress_midi, enabled_musicxml, tmp_path)
    enabled_ratios = _musicxml_tuplet_ratios(enabled_musicxml)
    expected_enabled = {tuple(item) for item in fixture["tuplet_stress"]["expected_enabled_ratios"]}
    assert expected_enabled.issubset(enabled_ratios)
