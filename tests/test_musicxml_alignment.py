from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.jianpu_score.musicxml_standardize import (
    MusicXMLStandardizationError,
    _RawEvent,
    _align_source_notes,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "high_accuracy" / "alignment_cases.json"


def _event(value: dict[str, object]) -> _RawEvent:
    pitches = [int(pitch) for pitch in value["pitches"]]  # type: ignore[index]
    tie_types = list(value.get("tie_types", [None] * len(pitches)))  # type: ignore[union-attr]
    return _RawEvent(
        event_id=str(value["event_id"]),
        part_group="p1",
        part_id="p1",
        staff=1,
        voice="1",
        start_tick=int(value["start_tick"]),
        end_tick=int(value["end_tick"]),
        pitches=pitches,
        kind="chord" if len(pitches) > 1 else "note",
        tie=None,
        tie_types=tie_types,
        tuplet_actual=None,
        tuplet_normal=None,
        dots=0,
        measure_number=1,
        metadata={},
    )


def _source(value: dict[str, object]) -> dict[str, int]:
    start = int(value["start_tick"])
    end = int(value["end_tick"])
    return {
        "source_index": int(value["source_index"]),
        "midi": int(value["midi"]),
        "start_tick_480": start * 10,
        "end_tick_480": end * 10,
        "start_tick": start,
        "end_tick": end,
    }


def _case(name: str) -> tuple[list[_RawEvent], list[dict[str, int]]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value = next(item for item in fixture["cases"] if item["name"] == name)
    return [_event(item) for item in value["events"]], [_source(item) for item in value["sources"]]


@pytest.mark.parametrize(
    "name",
    [
        "adaptive_shift_over_four_ticks",
        "logical_tie_chain",
        "dense_chord_quantized_group",
        "same_pitch_overlap",
        "duplicate_source_merge",
    ],
)
def test_anonymous_alignment_cases_account_every_source_note(name: str) -> None:
    events, sources = _case(name)
    report = _align_source_notes(events, sources)

    assert len(report) == len(sources)
    assert {item["source_index"] for item in report} == {item["source_index"] for item in sources}
    assert all(item["reason"] and item["accounting_category"] != "unresolved" for item in report)
    assert all(item["musicxml_event_ids"] for item in report)


def test_anonymous_alignment_keeps_quantized_xml_timing_and_tie_chain() -> None:
    events, sources = _case("adaptive_shift_over_four_ticks")
    report = _align_source_notes(events, sources)
    assert report[0]["score_start_tick"] == 12
    assert report[0]["score_end_tick"] == 60
    assert report[0]["source_to_score_movement_start_ticks"] == 12
    assert report[0]["matching_evidence"] == "adaptive_quantization_window"

    events, sources = _case("logical_tie_chain")
    report = _align_source_notes(events, sources)
    assert report[0]["reason"] == "matched_musicxml_tie_chain"
    assert report[0]["musicxml_event_ids"] == ["tie-start", "tie-stop"]


def test_anonymous_alignment_marks_duplicate_source_as_merged() -> None:
    events, sources = _case("duplicate_source_merge")
    report = _align_source_notes(events, sources)
    merged = [item for item in report if item["accounting_category"] == "merged"]
    assert len(merged) == 1
    assert merged[0]["reason"] == "merged_overlapping_duplicate_source_note"
    assert merged[0]["merged_into_source_index"] == 0


def test_anonymous_alignment_rejects_distant_same_pitch_without_evidence() -> None:
    events, sources = _case("distant_same_pitch_is_unresolved")
    with pytest.raises(MusicXMLStandardizationError, match=r"count=1; index=0,midi=60"):
        _align_source_notes(events, sources)
