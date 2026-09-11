from __future__ import annotations

import pytest

from backend.v2_job_manager import V2JobService


def _note(track: str, pitch: int, start: float, end: float) -> dict[str, object]:
    return {
        "track_id": track,
        "instrument_group": track,
        "is_drum": False,
        "pitch": pitch,
        "start_sec": start,
        "end_sec": end,
        "velocity": None,
        "confidence": None,
    }


def _select(
    notes: list[dict[str, object]], selected_ids: set[str] | None = None
) -> tuple[list[int], dict[str, object]]:
    selected, audit = V2JobService._select_main_melody_notes(
        notes, selected_ids or {"upper", "lower", "piano"}
    )
    return [int(note["pitch"]) for note in selected], audit


def test_held_upper_recovery_skips_overlapping_lower_insert() -> None:
    pitches, audit = _select(
        [
            _note("upper", 82, 0.0, 0.9),
            _note("lower", 66, 0.5, 1.1),
            _note("upper", 77, 0.9, 1.2),
        ]
    )

    assert pitches == [82, 77]
    assert audit["skipped"][0]["reason"] == "held_high_recovery_lower_insert"
    assert audit["skipped"][0]["pitch"] == 66


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        (
            [
                _note("piano", 82, 0.0, 0.9),
                _note("piano", 70, 0.5, 1.1),
                _note("piano", 72, 0.75, 1.3),
                _note("piano", 74, 1.0, 1.5),
            ],
            [82, 70, 72, 74],
        ),
        (
            [
                _note("piano", 84, 0.0, 0.3),
                _note("piano", 72, 0.125, 0.425),
                _note("piano", 84, 0.25, 0.55),
                _note("piano", 72, 0.375, 0.675),
            ],
            [84, 72, 84, 72],
        ),
        (
            [
                _note("piano", 82, 0.0, 0.4),
                _note("piano", 66, 0.4, 0.7),
                _note("piano", 77, 0.7, 1.0),
            ],
            [82, 66, 77],
        ),
    ],
)
def test_overlapping_or_adjacent_low_notes_are_not_deleted_without_hold_evidence(
    notes: list[dict[str, object]], expected: list[int]
) -> None:
    pitches, audit = _select(notes)

    assert pitches == expected
    assert audit["skipped"] == []


@pytest.mark.parametrize("ioi", [0.125, 0.1])
def test_fast_single_voice_is_not_removed_by_absolute_duration_rule(ioi: float) -> None:
    source = [72, 74, 76, 77, 79, 77, 76, 74] * 2
    notes = [_note("piano", pitch, index * ioi, index * ioi + 0.9 * ioi) for index, pitch in enumerate(source)]

    pitches, audit = _select(notes)

    assert pitches == source
    assert audit["selected_note_count"] == len(source)


def test_sustained_cross_stem_accompaniment_does_not_delete_low_phrase() -> None:
    notes = [
        _note("lower", 60, 0.0, 0.45),
        _note("upper", 84, 0.0, 2.0),
        _note("lower", 62, 0.5, 1.0),
        _note("lower", 64, 1.0, 1.5),
    ]

    pitches, audit = _select(notes, {"upper", "lower"})

    assert pitches == [84, 62, 64]
    assert audit["skipped"] == []
    assert audit["selected_track_ids"] == ["lower", "upper"]


def test_sparse_timing_keeps_normal_high_reentry_over_same_onset_lower_note() -> None:
    notes = [
        _note("piano", 75, 0.0, 0.35),
        _note("piano", 82, 1.8, 2.16),
        _note("piano", 65, 1.8, 3.0),
        _note("piano", 80, 2.16, 2.52),
    ]

    pitches, audit = _select(notes)

    assert pitches == [75, 82, 80]
    assert audit["skipped"] == []


def test_derived_note_end_is_clipped_without_losing_source_end_audit() -> None:
    notes = [
        _note("piano", 82, 0.0, 0.9),
        _note("piano", 70, 0.5, 1.1),
    ]

    selected, audit = V2JobService._select_main_melody_notes(notes, {"piano"})

    assert [note["end_sec"] for note in selected] == [0.5, 1.1]
    assert audit["selected_notes"][0]["end_sec"] == 0.9
    assert audit["derived_notes"][0]["source_end_sec"] == 0.9
    assert audit["derived_notes"][0]["derived_end_sec"] == 0.5
    assert audit["derived_notes"][0]["clipped_to_next_onset"] is True


def test_missing_confidence_and_playback_metadata_are_not_selector_evidence() -> None:
    notes = [_note("piano", 60, 0.0, 0.2), _note("piano", 62, 0.2, 0.4)]
    notes[0]["metadata"] = {"playback_default": 80}
    notes[1]["metadata"] = {"playback_default": 1}

    pitches, audit = _select(notes)

    assert pitches == [60, 62]
    assert audit["policy"]["confidence"].startswith("missing confidence is neutral")
