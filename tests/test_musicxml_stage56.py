from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

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
    _key_sharps,
    standardize_musicxml_payload,
    standardize_musicxml,
    write_standardized_score,
)
from backend.jianpu_score.musescore_import import MuseScoreImportError, convert_performance_midi
import backend.jianpu_score.musescore_import as musescore_import
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


def test_musescore_adapter_reports_missing_pinned_executable_without_fallback(tmp_path: Path) -> None:
    midi = tmp_path / "source.mid"
    midi.write_bytes(b"MThd")
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
    profile.write_text("<MidiOptions />", encoding="utf-8")
    source_a.write_bytes(b"MThd")
    source_b.write_bytes(b"MThd")
    xml = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<score-partwise version='3.1'><part-list></part-list><part id='P1'>"
        "<!-- fixture output --><!-- fixture output --><!-- fixture output -->"
        "<!-- fixture output --><!-- fixture output --><!-- fixture output -->"
        "</part></score-partwise>"
    )
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


def test_drum_performance_is_explicitly_midi_only(tmp_path: Path) -> None:
    with pytest.raises(MusicXMLStandardizationError, match="MIDI-only"):
        standardize_musicxml(
            tmp_path / "drums.musicxml",
            performance_metadata={"is_drum": True, "drum_jianpu_policy": "midi_only"},
        )


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
