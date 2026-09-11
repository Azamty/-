from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.jianpu_score.musicxml_standardize import (
    MusicXMLStandardizationError,
    _RawEvent,
    _align_source_notes,
    _estimate_source_alignment,
    _extend_timeline_for_tempo_tail,
    _logical_pitch_units,
    _map_source_tempo_values_to_score,
    _split_source_retrigger_events,
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


def test_source_coordinate_reconciliation_allows_long_movement_proven_by_affine_anchors() -> None:
    # A MuseScore import can rescale a long performance timeline.  The final
    # raw displacement exceeds 384 ticks here, but three exact anchors prove
    # the 0.5 scale; the accepted movement bound must come from that model and
    # its checked residual limit rather than from a broad matching window.
    events = [
        _timed_event("xml-60", 60, 0, 48),
        _timed_event("xml-62", 62, 480, 528),
        _timed_event("xml-64", 64, 960, 1008),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
        _timed_source(2, 64, 1920, 2016),
    ]

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    assert audit["classification"] == "global_affine_scale_and_offset"
    assert audit["max_raw_end_difference_ticks"] == 1008
    assert audit["movement_bound_ticks"] == 1072
    assert audit["models"][0]["movement_bound_ticks"] == 1072

    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert [item["musicxml_event_id"] for item in report] == ["xml-60", "xml-62", "xml-64"]
    assert all(item["reason"] == "matched_musicxml_affine_source_alignment" for item in report)


def test_source_coordinate_reconciliation_keeps_unproven_long_offset_fail_closed() -> None:
    events = [
        _timed_event("xml-60", 60, 0, 48),
        _timed_event("xml-62", 62, 480, 528),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
    ]

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "insufficient_alignment_model_points"


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


def test_cross_part_source_reconciliation_allows_bounded_overlap_reordering() -> None:
    events = [
        _parted_timed_event("xml-60", 60, 0, 38, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-62", 62, 96, 134, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-64", 64, 192, 230, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-65", 65, 180, 277, part_id="P1-Staff2", part_group="P1", staff=2),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 120, 168),
        _timed_source(2, 64, 240, 288),
        _timed_source(3, 65, 250, 346),
    ]

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    assert audit["global_order_preserved"] is False
    assert audit["pairing"].endswith("bounded_cross_part_overlap_reordering")
    assert audit["order_reconciliation"]["reordered_pair_count"] == 1
    assert audit["order_reconciliation"]["pairs"][0]["source_gap_ticks"] == -38
    assert audit["order_reconciliation"]["pairs"][0]["musicxml_score_overlap_ticks"] == 38
    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert [item["musicxml_event_id"] for item in report] == [
        "xml-60",
        "xml-62",
        "xml-64",
        "xml-65",
    ]


def test_cross_part_source_reconciliation_rejects_disjoint_order_reordering() -> None:
    events = [
        _parted_timed_event("xml-60", 60, 0, 38, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-62", 62, 96, 134, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-64", 64, 192, 230, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("xml-65", 65, 144, 192, part_id="P1-Staff2", part_group="P1", staff=2),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 120, 168),
        _timed_source(2, 64, 240, 288),
        _timed_source(3, 65, 250, 346),
    ]

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "provisional_alignment_global_order_reversed"


def test_exact_source_timing_proves_identity_when_model_has_one_anchor() -> None:
    events = [_timed_event("xml-60", 60, 0, 48)]
    sources = [_timed_source(0, 60, 0, 48)]

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit == {
        "applied": False,
        "reason": "existing_source_alignment_within_strict_window",
        "provisional_pair_count": 1,
        "identity_proven": True,
        "pitch_multiset_equal": True,
        "one_to_one": True,
        "global_order_preserved": True,
    }


def test_midi_lane_identity_reconciles_duplicate_pitch_across_imported_parts() -> None:
    events = [
        _parted_timed_event("p1-60", 60, 10, 34, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("p1-62", 62, 58, 82, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("p1-64", 64, 106, 130, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event(
            "p2-60",
            60,
            154,
            178,
            part_id="Piano, lane two",
            part_group="Piano, lane two",
            staff=1,
        ),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 62, 148, 172),
        _timed_source(2, 64, 196, 220),
        _timed_source(3, 60, 244, 268),
    ]
    for source in sources:
        source["midi_lane"] = 0 if source["source_index"] < 3 else 1
        source["midi_track_name"] = "lane one" if source["midi_lane"] == 0 else "lane two"

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    assert audit["method"] == "midi_lane_identity_affine_models"
    assert audit["one_to_one"] is True
    assert audit["mapped_groups"]["Piano, lane two"] == 1
    assert len(audit["models"]) == 2
    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert {item["accounting_category"] for item in report} == {"matched"}
    assert {item["musicxml_event_id"] for item in report} == {"p1-60", "p1-62", "p1-64", "p2-60"}


def test_midi_lane_identity_accepts_complete_named_partition_without_fallback() -> None:
    events = [
        _parted_timed_event("p1-60", 60, 10, 34, part_id="P1", part_group="Piano, lane one", staff=1),
        _parted_timed_event("p1-62", 62, 58, 82, part_id="P1", part_group="Piano, lane one", staff=1),
        _parted_timed_event("p1-64", 64, 106, 130, part_id="P1", part_group="Piano, lane one", staff=1),
        _parted_timed_event("p2-65", 65, 154, 178, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("p2-67", 67, 202, 226, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("p2-69", 69, 250, 274, part_id="P2", part_group="Piano, lane two", staff=1),
    ]
    sources = [
        _timed_source(0, 60, 100, 124),
        _timed_source(1, 62, 148, 172),
        _timed_source(2, 64, 196, 220),
        _timed_source(3, 65, 244, 268),
        _timed_source(4, 67, 292, 316),
        _timed_source(5, 69, 340, 364),
    ]
    for source in sources:
        source["midi_lane"] = 0 if source["source_index"] < 3 else 1
        source["midi_track_name"] = "lane one" if source["midi_lane"] == 0 else "lane two"

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    assert audit["mapped_groups"] == {"Piano, lane one": 0, "Piano, lane two": 1}
    assert audit["unassigned_lanes"] == []
    assert audit["unassigned_groups"] == []
    assert audit["one_to_one"] is True
    assert len(audit["models"]) == 2
    assert len(hints) == len(sources)


@pytest.mark.parametrize(
    ("base_name", "display_prefix"),
    [
        ("synthetic-multitrack-02", "Piano"),
        ("asap-v11-04", "Piano"),
    ],
    ids=["synthetic", "asap"],
)
def test_midi_lane_identity_uses_display_names_for_multistaff_groups(
    base_name: str,
    display_prefix: str,
) -> None:
    lane_two_name = f"{base_name} voice 2"
    events = [
        _parted_timed_event("p1-60", 60, 0, 48, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("p1-62", 62, 480, 528, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("p1-64", 64, 960, 1008, part_id="P1-Staff2", part_group="P1", staff=2),
        _parted_timed_event("p2-65", 65, 0, 48, part_id="P2-Staff1", part_group="P2", staff=1),
        _parted_timed_event("p2-67", 67, 480, 528, part_id="P2-Staff1", part_group="P2", staff=1),
        _parted_timed_event("p2-69", 69, 960, 1008, part_id="P2-Staff2", part_group="P2", staff=2),
    ]
    for event in events[:3]:
        event.metadata["musicxml_part_name"] = f"{display_prefix}, {base_name}"
    for event in events[3:]:
        event.metadata["musicxml_part_name"] = f"{display_prefix}, {lane_two_name}"
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 480, 528),
        _timed_source(2, 64, 960, 1008),
        _timed_source(3, 65, 0, 48),
        _timed_source(4, 67, 480, 528),
        _timed_source(5, 69, 960, 1008),
    ]
    for source in sources[:3]:
        source.update(midi_lane=0, midi_track_name=base_name, midi_channel=1, midi_track_index=1)
    for source in sources[3:]:
        source.update(midi_lane=1, midi_track_name=lane_two_name, midi_channel=2, midi_track_index=2)

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    assert audit["mapped_groups"] == {"P1": 0, "P2": 1}
    assert audit["group_part_names"] == {
        "P1": f"{display_prefix}, {base_name}",
        "P2": f"{display_prefix}, {lane_two_name}",
    }
    assert audit["unassigned_lanes"] == []
    assert audit["unassigned_groups"] == []
    assert audit["candidate_matches"]["P2"] == [
        {
            "lane": 1,
            "match_length": len(lane_two_name),
            "musicxml_part_name": f"{display_prefix}, {lane_two_name}",
            "source_track_name": lane_two_name,
        },
        {
            "lane": 0,
            "match_length": len(base_name),
            "musicxml_part_name": f"{display_prefix}, {lane_two_name}",
            "source_track_name": base_name,
        },
    ]
    assert audit["source_lane_metadata"] == {
        "0": {"midi_channels": [1], "midi_track_indices": [1], "track_names": [base_name]},
        "1": {"midi_channels": [2], "midi_track_indices": [2], "track_names": [lane_two_name]},
    }
    assert audit["pitch_multiset_equal"] is True
    assert audit["one_to_one"] is True
    assert len(hints) == len(sources)


def test_midi_lane_identity_missing_display_names_fails_closed_before_global_match() -> None:
    events = [
        _parted_timed_event("p1-60", 60, 0, 48, part_id="P1-Staff1", part_group="P1", staff=1),
        _parted_timed_event("p2-60", 60, 0, 48, part_id="P2-Staff1", part_group="P2", staff=1),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 60, 0, 48),
    ]
    sources[0].update(midi_lane=0, midi_track_name="lane one", midi_channel=1, midi_track_index=1)
    sources[1].update(midi_lane=1, midi_track_name="lane two", midi_channel=2, midi_track_index=2)

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "source_midi_lane_part_identity_missing"
    assert audit["mapped_groups"] == {}
    assert audit["unassigned_lanes"] == [0, 1]
    assert audit["unassigned_groups"] == ["P1", "P2"]
    assert audit["candidate_matches"] == {"P1": [], "P2": []}


def test_midi_lane_identity_rejects_non_bijective_longest_matches() -> None:
    events = [
        _parted_timed_event("p1-60", 60, 0, 48, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("p2-62", 62, 480, 528, part_id="P2", part_group="P2", staff=1),
    ]
    events[0].metadata["musicxml_part_name"] = "Piano, lane two"
    events[1].metadata["musicxml_part_name"] = "Piano, lane two"
    sources = [_timed_source(0, 60, 0, 48), _timed_source(1, 62, 480, 528)]
    sources[0].update(midi_lane=0, midi_track_name="lane", midi_channel=1, midi_track_index=1)
    sources[1].update(midi_lane=1, midi_track_name="lane two", midi_channel=2, midi_track_index=2)

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "source_midi_lane_part_identity_not_bijective"
    assert audit["duplicate_lane_groups"] == {"1": ["P1", "P2"]}


def test_midi_lane_singleton_reuses_unique_shared_affine_model() -> None:
    events = [
        _parted_timed_event("lane1-60", 60, 0, 48, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-62", 62, 480, 528, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-64", 64, 960, 1008, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event(
            "lane2-65",
            65,
            1200,
            1248,
            part_id="Piano, lane two",
            part_group="Piano, lane two",
            staff=1,
        ),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
        _timed_source(2, 64, 1920, 2016),
        _timed_source(3, 65, 2400, 2496),
    ]
    for source in sources:
        source["midi_lane"] = 0 if source["source_index"] < 3 else 1
        source["midi_track_name"] = "lane one" if source["midi_lane"] == 0 else "lane two"

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    singleton = next(model for model in audit["models"] if model["lane"] == 1)
    assert singleton["method"] == "midi_lane_shared_affine_alignment"
    assert singleton["shared_anchor_lane"] == 0
    assert singleton["shared_anchor_pair_count"] == 3
    assert hints[3]["scale"] == pytest.approx(0.5)
    assert hints[3]["musicxml_unit_id"] == 3
    assert singleton["tempo_eligible"] is False
    assert singleton["coordinate_scope"] == "note_alignment_only"


def test_midi_lane_two_anchors_reuses_unique_shared_affine_model() -> None:
    events = [
        _parted_timed_event("lane1-60", 60, 0, 24, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-62", 62, 480, 528, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-64", 64, 960, 1008, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event(
            "lane2-65",
            65,
            1200,
            1248,
            part_id="Piano, lane two",
            part_group="Piano, lane two",
            staff=1,
        ),
        _parted_timed_event(
            "lane2-67",
            67,
            1440,
            1488,
            part_id="Piano, lane two",
            part_group="Piano, lane two",
            staff=1,
        ),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
        _timed_source(2, 64, 1920, 2016),
        _timed_source(3, 65, 2400, 2496),
        _timed_source(4, 67, 2880, 2976),
    ]
    for source in sources:
        source["midi_lane"] = 0 if source["source_index"] < 3 else 1
        source["midi_track_name"] = "lane one" if source["midi_lane"] == 0 else "lane two"

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    assert audit["one_to_one"] is True
    short_model = next(model for model in audit["models"] if model["lane"] == 1)
    assert short_model["method"] == "midi_lane_shared_affine_alignment"
    assert short_model["pair_count"] == 2
    assert short_model["shared_anchor_lane"] == 0
    assert short_model["shared_anchor_pair_count"] == 3
    assert hints[3]["scale"] == pytest.approx(0.5)
    assert hints[4]["scale"] == pytest.approx(0.5)
    assert hints[3]["start_residual_ticks"] == pytest.approx(0)
    assert hints[4]["end_residual_ticks"] == pytest.approx(0)


def test_midi_lane_short_partition_rejects_multiple_distinct_shared_models() -> None:
    events = [
        _parted_timed_event("lane1-60", 60, 0, 24, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-62", 62, 480, 528, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-64", 64, 960, 1008, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane2-65", 65, 5, 53, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("lane2-67", 67, 485, 533, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("lane2-69", 69, 965, 1013, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("lane3-71", 71, 1000, 1024, part_id="P3", part_group="Piano, lane three", staff=1),
        _parted_timed_event("lane3-73", 73, 1240, 1264, part_id="P3", part_group="Piano, lane three", staff=1),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
        _timed_source(2, 64, 1920, 2016),
        _timed_source(3, 65, 0, 48),
        _timed_source(4, 67, 480, 528),
        _timed_source(5, 69, 960, 1008),
        _timed_source(6, 71, 2000, 2048),
        _timed_source(7, 73, 2480, 2528),
    ]
    for source in sources:
        lane = 0 if source["source_index"] < 3 else 1 if source["source_index"] < 6 else 2
        source["midi_lane"] = lane
        source["midi_track_name"] = {0: "lane one", 1: "lane two", 2: "lane three"}[lane]

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "source_midi_lane_alignment_ambiguous_shared_models"
    assert audit["lane"] == 2
    assert audit["pair_count"] == 2
    assert audit["shared_model_count"] == 2
    assert audit["shared_model_lanes"] == [0, 1]


def test_midi_lane_singleton_accounts_identity_without_selecting_shared_model() -> None:
    events = [
        _parted_timed_event("lane1-60", 60, 0, 48, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-62", 62, 480, 528, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-64", 64, 960, 1008, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane2-65", 65, 5, 53, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("lane2-67", 67, 485, 533, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("lane2-69", 69, 965, 1013, part_id="P2", part_group="Piano, lane two", staff=1),
        _parted_timed_event("lane3-71", 71, 1000, 1048, part_id="P3", part_group="Piano, lane three", staff=1),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
        _timed_source(2, 64, 1920, 2016),
        _timed_source(3, 65, 0, 48),
        _timed_source(4, 67, 480, 528),
        _timed_source(5, 69, 960, 1008),
        _timed_source(6, 71, 1600, 1648),
    ]
    for source in sources:
        lane = 0 if source["source_index"] < 3 else 1 if source["source_index"] < 6 else 2
        source["midi_lane"] = lane
        source["midi_track_name"] = {0: "lane one", 1: "lane two", 2: "lane three"}[lane]

    hints, audit = _estimate_source_alignment(events, sources)

    assert audit["applied"] is True
    singleton = next(model for model in audit["models"] if model["lane"] == 2)
    assert singleton["method"] == "midi_lane_identity_singleton_offset_alignment"
    assert singleton["candidate_shared_model_count"] == 2
    assert singleton["tempo_eligible"] is False
    assert singleton["coordinate_scope"] == "note_alignment_only"
    assert "shared_anchor_lane" not in singleton
    assert hints[6]["musicxml_unit_id"] == 6
    assert hints[6]["coordinate_scope"] == "note_alignment_only"
    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert report[-1]["musicxml_event_id"] == "lane3-71"


def test_midi_lane_singleton_shared_model_keeps_duration_residual_fail_closed() -> None:
    events = [
        _parted_timed_event("lane1-60", 60, 0, 48, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-62", 62, 480, 528, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event("lane1-64", 64, 960, 1008, part_id="P1", part_group="P1", staff=1),
        _parted_timed_event(
            "lane2-65",
            65,
            1200,
            1313,
            part_id="Piano, lane two",
            part_group="Piano, lane two",
            staff=1,
        ),
    ]
    sources = [
        _timed_source(0, 60, 0, 48),
        _timed_source(1, 62, 960, 1056),
        _timed_source(2, 64, 1920, 2016),
        _timed_source(3, 65, 2400, 2496),
    ]
    for source in sources:
        source["midi_lane"] = 0 if source["source_index"] < 3 else 1
        source["midi_track_name"] = "lane one" if source["midi_lane"] == 0 else "lane two"

    hints, audit = _estimate_source_alignment(events, sources)

    assert hints == {}
    assert audit["applied"] is False
    assert audit["reason"] == "source_midi_lane_alignment_residual_exceeds_bound"
    assert audit["lane"] == 1


def test_source_retrigger_split_preserves_each_nonduplicate_source_event() -> None:
    events = [
        _timed_event("anchor-40", 40, 0, 6),
        _timed_event("anchor-41", 41, 10, 16),
        _timed_event("anchor-42", 42, 20, 26),
        _RawEvent(
            event_id="imported-chord",
            part_group="p1",
            part_id="p1",
            staff=1,
            voice="1",
            start_tick=30,
            end_tick=36,
            pitches=[48, 60],
            kind="chord",
            tie=None,
            tie_types=[None, None],
            tuplet_actual=None,
            tuplet_normal=None,
            tuplet_type=None,
            dots=0,
            measure_number=1,
            metadata={},
        ),
    ]
    sources = [
        _timed_source(0, 40, 0, 6),
        _timed_source(1, 41, 10, 16),
        _timed_source(2, 42, 20, 26),
        _timed_source(3, 48, 30, 32),
        _timed_source(4, 48, 32, 36),
        _timed_source(5, 60, 32, 36),
    ]

    repairs = _split_source_retrigger_events(events, sources)

    assert repairs
    assert any(item["reason"] == "musicxml_event_split_for_source_retriggers" for item in repairs)
    units = _logical_pitch_units(events)
    assert len(units) == len(sources)
    hints, audit = _estimate_source_alignment(events, sources)
    assert audit["applied"] is False
    assert audit["reason"] == "existing_source_alignment_within_strict_window"
    report = _align_source_notes(events, sources, alignment_hints=hints)
    assert len(report) == len(sources)
    assert {item["accounting_category"] for item in report} == {"matched"}
    assert {item["source_index"] for item in report if item["source_midi"] == 48} == {3, 4}
    assert len({item["musicxml_unit_id"] for item in report if item["source_midi"] == 48}) == 2


def test_source_retrigger_split_rejects_a_gap_between_same_pitch_events() -> None:
    events = [
        _timed_event("anchor-40", 40, 0, 6),
        _timed_event("anchor-41", 41, 10, 16),
        _timed_event("anchor-42", 42, 20, 26),
        _timed_event("imported-48", 48, 30, 36),
    ]
    sources = [
        _timed_source(0, 40, 0, 6),
        _timed_source(1, 41, 10, 16),
        _timed_source(2, 42, 20, 26),
        _timed_source(3, 48, 30, 31),
        _timed_source(4, 48, 34, 36),
    ]

    assert _split_source_retrigger_events(events, sources) == []
    with pytest.raises(MusicXMLStandardizationError, match=r"count=2; index=3,midi=48; index=4,midi=48"):
        _align_source_notes(events, sources)


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


def test_source_tempo_points_map_through_proven_alignment_and_dedupe_same_tick() -> None:
    conductor = {
        "tempo_values": [
            {"offset_quarter": 0.0, "bpm": 120.0},
            {"offset_quarter": 0.5, "bpm": 110.0},
            {"offset_quarter": 1.0, "bpm": 110.0},
        ]
    }
    mapped, repairs, required_total_ticks = _map_source_tempo_values_to_score(
        conductor,
        performance_metadata={
            "tempo_points": [
                {"tick": 0, "bpm": 120.0},
                {"tick": 480, "bpm": 110.0},
            ]
        },
        source_alignment_report={
            "applied": True,
            "method": "test_affine",
            "models": [{"scale": 0.5, "offset": 0.0, "pair_count": 3}],
        },
        total_ticks=96,
    )

    assert mapped == [
        {"offset_quarter": 0.0, "bpm": 120.0},
        {"offset_quarter": 0.5, "bpm": 110.0},
    ]
    assert any(item["reason"] == "duplicate_tempo_event_removed_at_same_score_tick" for item in repairs)
    source_repair = next(item for item in repairs if item.get("source_offset_quarter") == 1.0)
    assert source_repair["source_tick"] == 48.0
    assert source_repair["mapped_tick"] == 24
    assert required_total_ticks == 96


def test_source_tempo_mapping_ignores_identity_only_singleton_models() -> None:
    mapped, repairs, required_total_ticks = _map_source_tempo_values_to_score(
        {
            "tempo_values": [
                {"offset_quarter": 0.0, "bpm": 120.0},
                {"offset_quarter": 1.0, "bpm": 110.0},
            ]
        },
        performance_metadata={
            "tempo_points": [
                {"tick": 0, "bpm": 120.0},
                {"tick": 480, "bpm": 110.0},
            ]
        },
        source_alignment_report={
            "applied": True,
            "models": [
                {"scale": 0.5, "offset": 0.0, "pair_count": 3, "tempo_eligible": True},
                {
                    "scale": 1.0,
                    "offset": 200.0,
                    "pair_count": 1,
                    "tempo_eligible": False,
                    "coordinate_scope": "note_alignment_only",
                },
            ],
        },
        total_ticks=96,
    )

    assert mapped == [
        {"offset_quarter": 0.0, "bpm": 120.0},
        {"offset_quarter": 0.5, "bpm": 110.0},
    ]
    assert required_total_ticks == 96
    assert all(item["alignment_model"]["scale"] == pytest.approx(0.5) for item in repairs)


def test_source_tempo_mapping_fails_when_only_identity_only_models_exist() -> None:
    with pytest.raises(MusicXMLStandardizationError, match="require a proven source-to-score alignment model"):
        _map_source_tempo_values_to_score(
            {"tempo_values": [{"offset_quarter": 0.5, "bpm": 110.0}]},
            performance_metadata={"tempo_points": [{"tick": 480, "bpm": 110.0}]},
            source_alignment_report={
                "applied": True,
                "models": [
                    {
                        "scale": 1.0,
                        "offset": 2.0,
                        "pair_count": 1,
                        "tempo_eligible": False,
                        "coordinate_scope": "note_alignment_only",
                    }
                ],
            },
            total_ticks=96,
        )


def test_source_tempo_points_use_proven_strict_identity_without_repair() -> None:
    mapped, repairs, required_total_ticks = _map_source_tempo_values_to_score(
        {"tempo_values": [{"offset_quarter": 0.0, "bpm": 96.0}, {"offset_quarter": 2.0, "bpm": 104.0}]},
        performance_metadata={
            "tempo_points": [
                {"tick": 0, "bpm": 96.0},
                {"tick": 960, "bpm": 104.0},
            ]
        },
        source_alignment_report={
            "applied": False,
            "reason": "existing_source_alignment_within_strict_window",
            "identity_proven": True,
            "pitch_multiset_equal": True,
            "one_to_one": True,
            "global_order_preserved": True,
            "provisional_pair_count": 4,
        },
        total_ticks=96,
    )

    assert mapped == [
        {"offset_quarter": 0.0, "bpm": 96.0},
        {"offset_quarter": 2.0, "bpm": 104.0},
    ]
    assert required_total_ticks == 96
    assert all(item["alignment_model"]["scale"] == 1.0 for item in repairs)
    assert all(item["alignment_method"] == "strict_identity_source_alignment" for item in repairs)


def test_source_tempo_mapping_rejects_conflict_and_out_of_range_without_clamp() -> None:
    with pytest.raises(MusicXMLStandardizationError, match="conflicting tempo events"):
        _map_source_tempo_values_to_score(
            {"tempo_values": [{"offset_quarter": 0.5, "bpm": 100.0}, {"offset_quarter": 1.0, "bpm": 110.0}]},
            performance_metadata={"tempo_points": [{"tick": 480, "bpm": 110.0}]},
            source_alignment_report={"applied": True, "models": [{"scale": 0.5, "offset": 0.0, "pair_count": 3}]},
            total_ticks=96,
        )
    mapped, _repairs, required_total_ticks = _map_source_tempo_values_to_score(
        {"tempo_values": [{"offset_quarter": 3.0, "bpm": 110.0}]},
        performance_metadata={"tempo_points": [{"tick": 1440, "bpm": 110.0}]},
        source_alignment_report={"applied": True, "models": [{"scale": 1.0, "offset": 0.0, "pair_count": 3}]},
        total_ticks=96,
    )
    assert mapped == [{"offset_quarter": 3.0, "bpm": 110.0}]
    assert required_total_ticks == 144

    timeline, final_total, audit = _extend_timeline_for_tempo_tail(
        [
            {
                "start_tick": 0,
                "duration_tick": 144,
                "end_tick": 144,
                "time_signature": "3/4",
                "is_pickup": False,
                "number": 1,
            }
        ],
        total_ticks=144,
        required_total_ticks=200,
    )
    assert final_total == 288
    assert timeline[-1]["start_tick"] == 144
    assert audit["reason"] == "extended_terminal_bars_to_retain_source_tempo_tail"


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
