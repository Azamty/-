from __future__ import annotations

import pytest

from backend.jianpu_score.domain import NoteEvent
from backend.jianpu_score.quantize import select_melody_path


def _note(
    start: float,
    end: float,
    midi: int,
    *,
    stem: str | None = None,
    confidence: float | None = None,
) -> NoteEvent:
    return NoteEvent(
        start_sec=start,
        end_sec=end,
        midi=midi,
        confidence=confidence,
        source="fixture",
        stem_id=stem,
    )


def test_overlapping_tail_does_not_block_a_real_reonset() -> None:
    events = [
        _note(0.0, 2.0, 36, stem="bass"),
        _note(0.0, 0.45, 72, stem="lead"),
        _note(0.44, 0.90, 74, stem="lead"),
        _note(0.90, 1.35, 76, stem="lead"),
    ]

    result = select_melody_path(events)

    assert [event.midi for event in result.selected] == [72, 74, 76]
    assert result.audit["transitions"][0]["overlap_sec"] == pytest.approx(0.01)
    assert result.audit["confidence_policy"].startswith("missing confidence is neutral")


def test_short_overlapping_ornament_can_be_skipped_between_melody_onsets() -> None:
    events = [
        _note(0.0, 0.40, 72, stem="lead"),
        _note(0.20, 0.25, 64, stem="accompaniment"),
        _note(0.40, 0.80, 74, stem="lead"),
    ]

    result = select_melody_path(events)

    assert [event.midi for event in result.selected] == [72, 74]
    assert result.audit["skipped_group_ids"] == [1]


def test_repeated_same_pitch_reonsets_are_retained() -> None:
    events = [
        _note(0.00, 0.18, 67, stem="lead"),
        _note(0.19, 0.37, 67, stem="lead"),
        _note(0.38, 0.56, 69, stem="lead"),
    ]

    result = select_melody_path(events)

    assert [(event.start_sec, event.midi) for event in result.selected] == [
        (0.00, 67),
        (0.19, 67),
        (0.38, 69),
    ]
    assert result.audit["transitions"][0]["same_pitch"] is True


def test_melody_can_change_stems_without_stem_hard_constraint() -> None:
    events = [
        _note(0.0, 0.35, 72, stem="piano-upper"),
        _note(0.4, 0.75, 74, stem="piano-lower"),
        _note(0.8, 1.15, 76, stem="piano-upper"),
    ]

    result = select_melody_path(events)

    assert [event.midi for event in result.selected] == [72, 74, 76]
    assert [event.stem_id for event in result.selected] == [
        "piano-upper",
        "piano-lower",
        "piano-upper",
    ]
