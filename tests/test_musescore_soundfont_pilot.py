from __future__ import annotations

from pathlib import Path

import mido

from scripts.pilot_musescore_soundfont import (
    DEFAULT_CASE_IDS,
    TIMING_TOLERANCE_SEC,
    compare_imported_midi,
    midi_notes,
)


def _write_midi(path: Path, *, shortened: bool = False, missing: bool = False) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    midi.tracks.append(conductor)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    track.append(
        mido.Message(
            "note_off",
            channel=0,
            note=60,
            velocity=0,
            time=240 if shortened else 480,
        )
    )
    if not missing:
        track.append(mido.Message("note_on", channel=0, note=64, velocity=80, time=0))
        track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(path)


def test_pilot_case_set_is_fixed_before_model_execution() -> None:
    assert DEFAULT_CASE_IDS == (
        "synthetic-bass-02",
        "special-triplet",
        "special-complex-chord",
        "maestro-midi-07",
        "synthetic-piano-01",
        "maestro-midi-01",
    )
    assert TIMING_TOLERANCE_SEC == 0.010


def test_musescore_import_comparison_accepts_exact_source_timing(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    imported = tmp_path / "imported.mid"
    _write_midi(source)
    _write_midi(imported)

    comparison = compare_imported_midi(source, imported)

    assert comparison["pitch_multiset_equal"] is True
    assert comparison["unmatched_source_count"] == 0
    assert comparison["unmatched_imported_count"] == 0
    assert comparison["timing_ok"] is True
    assert midi_notes(source)[1] == midi_notes(imported)[1]


def test_musescore_import_comparison_rejects_duration_loss_and_missing_note(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    shortened = tmp_path / "shortened.mid"
    missing = tmp_path / "missing.mid"
    _write_midi(source)
    _write_midi(shortened, shortened=True)
    _write_midi(missing, missing=True)

    shortened_result = compare_imported_midi(source, shortened)
    missing_result = compare_imported_midi(source, missing)

    assert shortened_result["pitch_multiset_equal"] is True
    assert shortened_result["timing_ok"] is False
    assert shortened_result["max_abs_end_delta_sec"] > TIMING_TOLERANCE_SEC
    assert missing_result["pitch_multiset_equal"] is False
    assert missing_result["unmatched_source_count"] == 1
    assert missing_result["timing_ok"] is False
