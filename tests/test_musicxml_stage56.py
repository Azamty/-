from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

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
    standardize_musicxml_payload,
    standardize_musicxml,
    write_standardized_score,
)
from backend.jianpu_score.musescore_import import MuseScoreImportError, convert_performance_midi
from backend.jianpu_score.high_accuracy import resolve_musescore, resolve_notation_python


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json"
PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"
EXTERNAL_READY = resolve_musescore() is not None and resolve_notation_python().is_file() and PROFILE.is_file()


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
    assert score.metadata["measure_total_ticks"] == 288


def test_musescore_adapter_reports_missing_pinned_executable_without_fallback(tmp_path: Path) -> None:
    midi = tmp_path / "source.mid"
    midi.write_bytes(b"MThd")
    with pytest.raises(MuseScoreImportError, match="executable is unavailable"):
        convert_performance_midi(
            midi,
            tmp_path / "output.musicxml",
            musescore_path=tmp_path / "missing-MuseScore4.exe",
        )


def test_drum_performance_is_explicitly_midi_only(tmp_path: Path) -> None:
    with pytest.raises(MusicXMLStandardizationError, match="MIDI-only"):
        standardize_musicxml(
            tmp_path / "drums.musicxml",
            performance_metadata={"is_drum": True, "drum_jianpu_policy": "midi_only"},
        )


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore and notation environment are unavailable")
def test_real_musescore_music21_fixture_preserves_triplet_tie_chord_and_staff(tmp_path: Path) -> None:
    from scripts.high_accuracy_fixture_smoke import _write_fixture_midi

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    midi = tmp_path / "stage56.performance.mid"
    musicxml = tmp_path / "stage56.musicxml"
    score_json = tmp_path / "stage56.score.json"
    alignment_json = tmp_path / "stage56.alignment.json"
    _write_fixture_midi(midi, payload)
    converted = convert_performance_midi(midi, musicxml, instrument_id="stage56")
    assert converted.musicxml_path == musicxml.resolve()
    score, report = standardize_musicxml(musicxml, title="Stage 56 fixture")
    artifact = write_standardized_score(
        musicxml,
        score_json,
        alignment_report_path=alignment_json,
        title="Stage 56 fixture",
    )

    assert musicxml.stat().st_size > 200
    assert score.quarter_ticks == 48
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
    assert artifact.score.model_dump(mode="json") == json.loads(score_json.read_text(encoding="utf-8"))
    assert json.loads(alignment_json.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    assert score.metadata["measure_total_ticks"] == score.total_ticks


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned MuseScore and notation environment are unavailable")
def test_stage56_cli_smoke_writes_independent_outputs(tmp_path: Path) -> None:
    from scripts.high_accuracy_fixture_smoke import _write_fixture_midi

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    midi = tmp_path / "cli.performance.mid"
    musicxml = tmp_path / "cli.musicxml"
    score_json = tmp_path / "cli.score.json"
    alignment_json = tmp_path / "cli.alignment.json"
    _write_fixture_midi(midi, payload)
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
            "--instrument-id",
            "cli-fixture",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["score_ticks_per_quarter"] == 48
    assert musicxml.is_file() and score_json.is_file() and alignment_json.is_file()
