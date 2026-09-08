from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.jianpu_score.musicxml_standardize import (
    MusicXMLStandardizationError,
    _RawEvent,
    _align_source_notes,
    _estimate_source_alignment,
    WorkerEvent,
    WorkerKeySignature,
    WorkerMeasure,
    WorkerPart,
    WorkerPayload,
    WorkerTempo,
    WorkerTimeSignature,
    standardize_musicxml_payload,
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


def _timed_source(source_index: int, midi: int, start_tick: int, end_tick: int) -> dict[str, int]:
    return {
        "source_index": source_index,
        "midi": midi,
        "start_tick_480": start_tick * 10,
        "end_tick_480": end_tick * 10,
        "start_tick": start_tick,
        "end_tick": end_tick,
    }


def _timed_event(event_id: str, midi: int, start_tick: int, end_tick: int) -> _RawEvent:
    return _RawEvent(
        event_id=event_id,
        part_group="p1",
        part_id="p1",
        staff=1,
        voice="1",
        start_tick=start_tick,
        end_tick=end_tick,
        pitches=[midi],
        kind="note",
        tie=None,
        tie_types=[None],
        tuplet_actual=None,
        tuplet_normal=None,
        dots=0,
        measure_number=1,
        metadata={},
    )


def _parted_timed_event(
    event_id: str,
    midi: int,
    start_tick: int,
    end_tick: int,
    *,
    part_id: str,
    part_group: str,
    staff: int,
) -> _RawEvent:
    return _RawEvent(
        event_id=event_id,
        part_group=part_group,
        part_id=part_id,
        staff=staff,
        voice="1",
        start_tick=start_tick,
        end_tick=end_tick,
        pitches=[midi],
        kind="note",
        tie=None,
        tie_types=[None],
        tuplet_actual=None,
        tuplet_normal=None,
        dots=0,
        measure_number=1,
        metadata={},
    )


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


def test_scattered_chord_does_not_supply_long_distance_support() -> None:
    events, sources = _case("scattered_chord_does_not_support_remote_pitch")
    with pytest.raises(MusicXMLStandardizationError, match=r"count=1; index=0,midi=60"):
        _align_source_notes(events, sources)


def test_source_coordinate_reconciliation_requires_complete_affine_evidence() -> None:
    # The source ticks are in a different, but exact, coordinate system:
    # MusicXML = source - 100.  Three distinct pitches make that transform
    # independently identifiable; matching is then locked to the audited
    # logical-unit ids rather than widening a nearest-neighbour window.
    events = [
        _timed_event("xml-60", 60, 0, 24),
        _timed_event("xml-62", 62, 48, 72),
        _timed_event("xml-64", 64, 96, 120),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 62, 148, 172),
        _timed_source(2, 64, 196, 220),
    ]

    hints, audit = _estimate_source_alignment(events, sources)
    assert audit["applied"] is True
    assert audit["method"] == "monotonic_pitch_assignment_affine_staff_models"
    assert audit["max_raw_start_difference_ticks"] == 100
    assert audit["max_raw_end_difference_ticks"] == 100
    assert audit["movement_bound_ticks"] == 384
    assert audit["models"][0]["source_indices"] == [0, 1, 2]
    assert audit["models"][0]["musicxml_unit_ids"] == [0, 1, 2]

    for source in sources:
        source["_alignment_hint"] = hints[int(source["source_index"])]
    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert [item["musicxml_event_id"] for item in report] == ["xml-60", "xml-62", "xml-64"]
    assert all(item["reason"] == "matched_musicxml_affine_source_alignment" for item in report)
    assert all(item["matching_evidence"] == "monotonic_pitch_affine_alignment" for item in report)
    assert [item["source_alignment_musicxml_unit_id"] for item in report] == [0, 1, 2]
    assert all(item["source_alignment_start_residual_ticks"] == 0 for item in report)


def test_cross_part_source_reconciliation_uses_global_one_to_one_model() -> None:
    # Each synthetic MuseScore part has only one anchor.  The complete pitch
    # assignment is still unique, so a global model can reconcile the parts
    # without changing the imported MusicXML timings.
    events = [
        _parted_timed_event("xml-60", 60, 10, 34, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-62", 62, 58, 82, part_id="P1-Staff2", part_group="P1", staff=2),
        _parted_timed_event("xml-64", 64, 106, 130, part_id="P2", part_group="P2", staff=1),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 62, 148, 172),
        _timed_source(2, 64, 196, 220),
    ]

    hints, audit = _estimate_source_alignment(events, sources)
    assert audit["applied"] is True
    assert audit["method"] == "monotonic_pitch_assignment_cross_part_affine_model"
    assert audit["one_to_one"] is True
    assert audit["models"][0]["source_indices"] == [0, 1, 2]
    assert audit["models"][0]["musicxml_unit_ids"] == [0, 1, 2]
    assert audit["strict_start_residual_bound_ticks"] == 64

    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert [item["musicxml_event_id"] for item in report] == ["xml-60", "xml-62", "xml-64"]
    assert {item["accounting_category"] for item in report} == {"matched"}
    assert all(item["matching_evidence"] == "monotonic_pitch_affine_alignment" for item in report)


def test_source_coordinate_reconciliation_rejects_pitch_multiset_changes() -> None:
    events = [
        _timed_event("xml-60", 60, 0, 24),
        _timed_event("xml-62", 62, 48, 72),
        _timed_event("xml-64", 64, 96, 120),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 62, 148, 172),
        _timed_source(2, 65, 196, 220),
    ]

    hints, audit = _estimate_source_alignment(events, sources)
    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "source_and_musicxml_pitch_sets_differ"
    with pytest.raises(MusicXMLStandardizationError, match=r"count=3;.*index=2,midi=65"):
        _align_source_notes(events, sources, alignment_hints=hints)


def test_source_coordinate_reconciliation_rejects_single_note_offset_without_anchors() -> None:
    events = [_timed_event("xml-60", 60, 0, 24)]
    sources = [_timed_source(0, 60, 100, 124)]

    hints, audit = _estimate_source_alignment(events, sources)
    assert hints == {}
    assert audit == {
        "applied": False,
        "reason": "insufficient_alignment_model_points",
        "group": {"part_group": "p1", "staff": 1},
        "candidate_count": 1,
        "minimum_model_points": 3,
    }
    with pytest.raises(MusicXMLStandardizationError, match=r"count=1; index=0,midi=60"):
        _align_source_notes(events, sources, alignment_hints=hints)


def test_source_coordinate_reconciliation_rejects_ambiguous_same_pitch_timing() -> None:
    events = [
        _timed_event("xml-a", 60, 0, 24),
        _timed_event("xml-b", 60, 0, 24),
        _timed_event("xml-c", 62, 48, 72),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 60, 100, 124),
        _timed_source(2, 62, 148, 172),
    ]

    hints, audit = _estimate_source_alignment(events, sources)
    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "provisional_alignment_not_unique_same_pitch_timing"
    assert audit["pitch"] == 60


def test_source_coordinate_reconciliation_rejects_global_order_reversal() -> None:
    events = [
        _timed_event("xml-62", 62, 0, 24),
        _timed_event("xml-60", 60, 48, 72),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 62, 148, 172),
    ]

    hints, audit = _estimate_source_alignment(events, sources)
    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "provisional_alignment_global_order_reversed"


def test_partial_chord_match_reports_unreferenced_units_by_unit_id() -> None:
    event = WorkerEvent(
        event_id="partial-chord",
        kind="chord",
        offset_quarter=0,
        duration_quarter=1,
        pitches=[60, 64, 67],
        tie_types=[None, None, None],
    )
    measure = WorkerMeasure(
        part_index=0,
        number=1,
        start_quarter=0,
        duration_quarter=4,
        end_quarter=4,
        time_signature="4/4",
    )
    payload = WorkerPayload(
        schema_version="1.0",
        worker="music21",
        music21_version="9.9.2",
        source_path="anonymous.xml",
        title="anonymous",
        highest_time_quarter=4,
        parts=[
            WorkerPart(
                part_id="p1",
                name="anonymous",
                highest_time_quarter=4,
                events=[event],
                measures=[measure],
            )
        ],
        measures=[measure],
        tempo_events=[WorkerTempo(offset_quarter=0, bpm=120)],
        time_signature_events=[WorkerTimeSignature(offset_quarter=0, ratio="4/4", numerator=4, denominator=4)],
        key_signature_events=[WorkerKeySignature(offset_quarter=0, key="C", sharps=0)],
    )
    score, report = standardize_musicxml_payload(
        payload,
        performance_metadata={
            "notes": [{"index": 0, "midi": 60, "start_tick": 0, "end_tick": 480}]
        },
    )

    assert score is not None
    assert report["musicxml_logical_unit_count"] == 3
    assert report["musicxml_matched_logical_unit_count"] == 1
    assert report["musicxml_extra_count"] == 2
    assert {item["pitch"] for item in report["musicxml_extras"]} == {64, 67}
    assert report["source_to_score"][0]["musicxml_unit_id"] not in {
        item["unit_id"] for item in report["musicxml_extras"]
    }
