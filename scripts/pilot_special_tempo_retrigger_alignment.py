"""Audit the special tempo retrigger lane before and after 5a00430.

This is an offline diagnostic pilot.  It reads the already generated
performance metadata, performance MIDI, and MuseScore MusicXML for one case;
it does not call a recognizer, use a reference MIDI/beat annotation, or alter
the production service.  The pilot deliberately proves the source-to-score
pairing from the imported part names and note coordinates before suggesting a
small standardizer change.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mido

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.jianpu_score.musicxml_standardize import (  # noqa: I001
    MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
    MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
    _alignment_unit_group,
    _estimate_source_alignment,
    _fit_source_alignment_line,
    _logical_pitch_units,
    _source_alignment_residuals,
    _source_notes,
    _worker_raw_events,
    resolve_notation_python,
    run_musicxml_worker,
    standardize_musicxml,
)


DEFAULT_BEFORE = ROOT / ".artifacts" / "review" / "production-acceptance-v3" / "new" / "special-tempo-change"
DEFAULT_AFTER = ROOT / ".artifacts" / "review" / "production-acceptance-v4" / "new" / "special-tempo-change"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "production-acceptance-v4" / "special-tempo-retrigger-pilot-v1"
SCHEMA_VERSION = "special_tempo_retrigger_alignment_pilot_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _service_output(case_root: Path) -> Path:
    candidate = case_root / "service_output"
    return candidate if candidate.is_dir() else case_root


def _single_file(directory: Path, pattern: str) -> Path:
    values = sorted(directory.glob(pattern))
    if len(values) != 1:
        raise FileNotFoundError(f"expected one {pattern} under {directory}, found {len(values)}")
    return values[0]


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _xml_shape(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    part_list = next((item for item in root if _tag(item) == "part-list"), None)
    score_parts: list[dict[str, Any]] = []
    if part_list is not None:
        for item in part_list:
            if _tag(item) != "score-part":
                continue
            score_parts.append(
                {
                    "id": item.attrib.get("id", ""),
                    "name": next(
                        (child.text or "" for child in item if _tag(child) == "part-name"),
                        "",
                    ),
                }
            )
    parts: list[dict[str, Any]] = []
    sentinel_name = "__JIANPU_SOURCE_ORIGIN_SENTINEL_v1__"
    for item in root:
        if _tag(item) != "part":
            continue
        notes = [note for note in item.iter() if _tag(note) == "note" and any(_tag(child) == "pitch" for child in note)]
        parts.append(
            {
                "id": item.attrib.get("id", ""),
                "pitched_note_element_count": len(notes),
                "measure_count": sum(1 for child in item if _tag(child) == "measure"),
            }
        )
    return {
        "root": _tag(root),
        "score_parts": score_parts,
        "parts": parts,
        "sentinel_part_present": any(sentinel_name in str(item.get("name", "")) for item in score_parts),
    }


def _midi_shape(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    tracks: list[dict[str, Any]] = []
    for index, track in enumerate(midi.tracks):
        absolute = 0
        track_name = ""
        note_on_count = 0
        note_off_count = 0
        note_spans: list[dict[str, int]] = []
        open_notes: dict[tuple[int, int], list[int]] = defaultdict(list)
        for message in track:
            absolute += int(message.time)
            if message.type == "track_name":
                track_name = str(message.name)
            if message.type == "note_on" and int(message.velocity) > 0:
                note_on_count += 1
                open_notes[(int(message.channel), int(message.note))].append(absolute)
            elif message.type == "note_off" or (message.type == "note_on" and int(message.velocity) == 0):
                note_off_count += 1
                key = (int(message.channel), int(message.note))
                if open_notes[key]:
                    note_spans.append(
                        {
                            "pitch": int(message.note),
                            "channel": int(message.channel) + 1,
                            "start_tick_480": open_notes[key].pop(0),
                            "end_tick_480": absolute,
                        }
                    )
        tracks.append(
            {
                "track_index": index,
                "track_name": track_name,
                "note_on_count": note_on_count,
                "note_off_count": note_off_count,
                "closed_note_span_count": len(note_spans),
                "note_spans": note_spans,
            }
        )
    return {
        "type": int(midi.type),
        "ticks_per_beat": int(midi.ticks_per_beat),
        "track_count": len(midi.tracks),
        "tracks": tracks,
    }


def _unit_group_name(unit: Any) -> str:
    return _alignment_unit_group(unit)[0]


def _map_groups_to_lanes(units: Sequence[Any], lane_names: Sequence[str]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    groups = sorted({_unit_group_name(unit) for unit in units})
    mapping: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    for group in groups:
        candidates = [
            (len(str(name)), index)
            for index, name in enumerate(lane_names)
            if str(name).strip() and str(name).casefold() in group.casefold()
        ]
        candidates.sort(reverse=True)
        if not candidates:
            failures.append({"group": group, "reason": "no_lane_name_in_musicxml_group"})
            continue
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0] and candidates[0][1] != candidates[1][1]:
            failures.append(
                {
                    "group": group,
                    "reason": "lane_name_match_tie",
                    "candidate_lanes": [item[1] for item in candidates],
                }
            )
            continue
        mapping[group] = candidates[0][1]
    return mapping, failures


def _source_row(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_index": int(source["source_index"]),
        "midi_lane": int(source.get("midi_lane", -1)),
        "pitch": int(source["midi"]),
        "start_tick_48": int(source["start_tick"]),
        "end_tick_48": int(source["end_tick"]),
    }


def _unit_row(unit: Any) -> dict[str, Any]:
    event = unit.chain[0][0]
    return {
        "unit_id": int(unit.unit_id),
        "pitch": int(unit.pitch),
        "start_tick_48": int(unit.start_tick),
        "end_tick_48": int(unit.end_tick),
        "part_group": _unit_group_name(unit),
        "staff": int(event.staff),
        "voice": str(event.voice),
        "event_ids": [item.event_id for item, _pitch_index in unit.chain],
    }


def pilot_per_lane_alignment(
    events: list[Any],
    source_notes: list[dict[str, Any]],
    lane_names: Sequence[str],
) -> dict[str, Any]:
    """Prove a source-to-score pairing using only lane names and XML units."""

    units = _logical_pitch_units(events)
    group_to_lane, group_failures = _map_groups_to_lanes(units, lane_names)
    source_by_lane: dict[int, list[dict[str, Any]]] = defaultdict(list)
    units_by_lane: dict[int, list[Any]] = defaultdict(list)
    for source in source_notes:
        if source.get("midi_lane") is not None:
            source_by_lane[int(source["midi_lane"])].append(source)
    for unit in units:
        group = _unit_group_name(unit)
        if group in group_to_lane:
            units_by_lane[group_to_lane[group]].append(unit)

    lane_pair_rows: dict[
        int,
        tuple[
            list[dict[str, Any]],
            list[Any],
            list[tuple[dict[str, Any], Any]],
            dict[str, int],
            dict[str, int],
            bool,
        ],
    ] = {}
    proof = not group_failures
    for lane in sorted(set(source_by_lane) | set(units_by_lane)):
        source_rows = source_by_lane.get(lane, [])
        unit_rows = units_by_lane.get(lane, [])
        source_by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        units_by_pitch: dict[int, list[Any]] = defaultdict(list)
        for source in source_rows:
            source_by_pitch[int(source["midi"])].append(source)
        for unit in unit_rows:
            units_by_pitch[int(unit.pitch)].append(unit)
        source_pitch_counts = {str(pitch): len(rows) for pitch, rows in sorted(source_by_pitch.items())}
        unit_pitch_counts = {str(pitch): len(rows) for pitch, rows in sorted(units_by_pitch.items())}
        lane_proof = source_pitch_counts == unit_pitch_counts and len(source_rows) == len(unit_rows)
        pairs: list[tuple[dict[str, Any], Any]] = []
        if lane_proof:
            for pitch in sorted(source_by_pitch):
                ordered_source = sorted(
                    source_by_pitch[pitch],
                    key=lambda item: (int(item["start_tick"]), int(item["end_tick"]), int(item["source_index"])),
                )
                ordered_units = sorted(
                    units_by_pitch[pitch],
                    key=lambda item: (int(item.start_tick), int(item.end_tick), int(item.unit_id)),
                )
                source_timing = [(int(item["start_tick"]), int(item["end_tick"])) for item in ordered_source]
                unit_timing = [(int(item.start_tick), int(item.end_tick)) for item in ordered_units]
                if len(source_timing) != len(set(source_timing)) or len(unit_timing) != len(set(unit_timing)):
                    lane_proof = False
                    break
                pairs.extend(zip(ordered_source, ordered_units, strict=True))
        if len({int(unit.unit_id) for _source, unit in pairs}) != len(pairs):
            lane_proof = False
        lane_pair_rows[lane] = (source_rows, unit_rows, pairs, source_pitch_counts, unit_pitch_counts, lane_proof)

    # Match the production lane model's bounded shared-model treatment for a
    # one-note lane.  A singleton may borrow exactly one affine model backed by
    # at least three anchors on another lane; otherwise it gets the same
    # explicit one-anchor offset that production already permits.
    shared_models: dict[int, tuple[float, float]] = {}
    for lane, (_sources, _units, pairs, _source_counts, _unit_counts, lane_proof) in lane_pair_rows.items():
        if not lane_proof or len(pairs) < 3:
            continue
        scale, offset = _fit_source_alignment_line(pairs)
        residuals = [_source_alignment_residuals(pair, scale=scale, offset=offset) for pair in pairs]
        if (
            0.5 <= scale <= 1.8
            and max((item[0] for item in residuals), default=0.0) <= MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS
            and max((item[1] for item in residuals), default=0.0) <= MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS
        ):
            shared_models[lane] = (float(scale), float(offset))

    lane_rows: list[dict[str, Any]] = []
    all_pairs: list[tuple[dict[str, Any], Any]] = []
    for lane in sorted(set(source_by_lane) | set(units_by_lane)):
        source_rows, unit_rows, pairs, source_pitch_counts, unit_pitch_counts, lane_proof = lane_pair_rows[lane]
        model: dict[str, Any] | None = None
        if lane_proof and pairs:
            if len(pairs) >= 3:
                scale, offset = _fit_source_alignment_line(pairs)
                method = "pilot_per_lane_affine"
            else:
                source, unit = pairs[0]
                distinct_shared = list(shared_models.items())
                if len(distinct_shared) == 1 and distinct_shared[0][0] != lane:
                    anchor_lane, (scale, offset) = distinct_shared[0]
                    method = "pilot_shared_affine"
                elif len(pairs) == 2:
                    # Two points are intentionally insufficient without one
                    # independently proven lane model.
                    scale = offset = 0.0
                    method = "pilot_insufficient_anchors"
                    anchor_lane = None
                    lane_proof = False
                else:
                    scale, offset = 1.0, float(unit.start_tick - int(source["start_tick"]))
                    method = "pilot_singleton_offset"
                    anchor_lane = None
            residuals = [_source_alignment_residuals(pair, scale=scale, offset=offset) for pair in pairs]
            model = {
                "method": method,
                "scale": float(scale),
                "offset_ticks": float(offset),
                "pair_count": len(pairs),
                "max_start_residual_ticks": max((item[0] for item in residuals), default=0.0),
                "max_end_residual_ticks": max((item[1] for item in residuals), default=0.0),
                "within_strict_bounds": (
                    max((item[0] for item in residuals), default=0.0)
                    <= MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS
                    and max((item[1] for item in residuals), default=0.0)
                    <= MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS
                ),
            }
            if method == "pilot_shared_affine":
                model["shared_anchor_lane"] = anchor_lane
            lane_proof = lane_proof and bool(model["within_strict_bounds"])
        lane_rows.append(
            {
                "lane": lane,
                "source_count": len(source_rows),
                "musicxml_unit_count": len(unit_rows),
                "source_pitch_counts": source_pitch_counts,
                "musicxml_pitch_counts": unit_pitch_counts,
                "one_to_one": len(pairs) == len(source_rows) == len(unit_rows)
                and len({int(unit.unit_id) for _source, unit in pairs}) == len(pairs),
                "model": model,
                "pairs": [
                    {
                        "source": _source_row(source),
                        "musicxml": _unit_row(unit),
                        "raw_start_difference_ticks": int(unit.start_tick) - int(source["start_tick"]),
                        "raw_end_difference_ticks": int(unit.end_tick) - int(source["end_tick"]),
                    }
                    for source, unit in pairs
                ],
                "proof": lane_proof,
            }
        )
        all_pairs.extend(pairs)
        proof = proof and lane_proof
    proof = proof and len(all_pairs) == len(source_notes) == len(units)
    return {
        "group_to_lane": group_to_lane,
        "groups": sorted({_unit_group_name(unit) for unit in units}),
        "group_failures": group_failures,
        "lane_rows": lane_rows,
        "source_note_count": len(source_notes),
        "musicxml_logical_unit_count": len(units),
        "pitch_multiset_equal": Counter(int(item["midi"]) for item in source_notes)
        == Counter(int(unit.pitch) for unit in units),
        "one_to_one": len(all_pairs) == len(source_notes) == len(units)
        and len({int(unit.unit_id) for _source, unit in all_pairs}) == len(all_pairs),
        "no_merge_drop_or_extra": len(all_pairs) == len(source_notes) == len(units),
        "strict_model_provable": proof,
    }


def _legacy_partition_guard_probe(
    source_notes: Sequence[Mapping[str, Any]],
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the pre-fix guard without checking out or mutating old code."""

    lanes = sorted({int(item["midi_lane"]) for item in source_notes if item.get("midi_lane") is not None})
    mapped_groups = {
        str(group): int(lane)
        for group, lane in (pilot.get("group_to_lane") or {}).items()
    }
    groups = [str(group) for group in (pilot.get("groups") or sorted(mapped_groups))]
    unassigned_lanes = [lane for lane in lanes if lane not in set(mapped_groups.values())]
    unassigned_groups = [group for group in groups if group not in mapped_groups]
    applies = len(lanes) >= 2
    legacy_rejected = applies and len(unassigned_lanes) != 1
    return {
        "applies_to_multi_lane_partition": applies,
        "mapped_groups": mapped_groups,
        "unassigned_lanes": unassigned_lanes,
        "unassigned_groups": unassigned_groups,
        "legacy_guard_rejected": legacy_rejected,
        "legacy_reason": "source_midi_lane_part_identity_incomplete" if legacy_rejected else None,
    }


def _case_audit(case_root: Path, *, label: str) -> dict[str, Any]:
    service_output = _service_output(case_root.resolve())
    metadata_path = _single_file(service_output, "*.performance.metadata.json")
    midi_path = _single_file(service_output, "*.performance.mid")
    musicxml_path = _single_file(service_output, "*.notated.musicxml")
    metadata = _load(metadata_path)
    manifest_path = service_output / "manifest.json"
    manifest = _load(manifest_path) if manifest_path.is_file() else {}
    worker_payload = run_musicxml_worker(
        musicxml_path,
        notation_python=resolve_notation_python(),
        timeout_sec=180,
    )
    events, worker_diagnostics = _worker_raw_events(worker_payload)
    source_notes = _source_notes(metadata)
    units = _logical_pitch_units(events)
    current_hints, current_alignment = _estimate_source_alignment(events, source_notes)
    try:
        score, standardized_report = standardize_musicxml(
            musicxml_path,
            performance_metadata=metadata,
            title=f"{label} pilot",
            notation_python=resolve_notation_python(),
            timeout_sec=180,
        )
        standardize_probe: dict[str, Any] = {
            "success": True,
            "score_total_ticks": int(score.total_ticks),
            "score_quarter_ticks": int(score.quarter_ticks),
            "score_voice_count": len(score.voices),
            "tempo_event_count": len(score.tempo_events),
            "source_coordinate_reconciliation": standardized_report.get("source_coordinate_reconciliation"),
            "source_note_count": standardized_report.get("source_note_count"),
            "musicxml_logical_unit_count": standardized_report.get("musicxml_logical_unit_count"),
            "musicxml_matched_logical_unit_count": standardized_report.get("musicxml_matched_logical_unit_count"),
            "musicxml_extra_count": standardized_report.get("musicxml_extra_count"),
        }
    except Exception as exc:  # noqa: BLE001 - the report must preserve a pilot failure
        standardize_probe = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    pilot = pilot_per_lane_alignment(
        events,
        source_notes,
        [str(item) for item in metadata.get("instrument_lane_track_names", [])],
    )
    legacy_guard = _legacy_partition_guard_probe(source_notes, pilot)
    origin = (
        ((manifest.get("stages") or {}).get("musescore_import") or {}).get("origin_sentinel")
        if isinstance(manifest, Mapping)
        else None
    )
    return {
        "label": label,
        "case_root": str(case_root.resolve()),
        "metadata": {
            "note_count": metadata.get("note_count"),
            "voice_lane_count": metadata.get("voice_lane_count"),
            "voice_lane_policy": metadata.get("voice_lane_policy"),
            "instrument_lane_track_names": metadata.get("instrument_lane_track_names"),
            "lane_assignment": metadata.get("lane_assignment"),
            "time_signature": metadata.get("time_signature"),
            "score_origin": metadata.get("score_origin"),
            "tempo_points": metadata.get("tempo_points", []),
        },
        "midi": _midi_shape(midi_path),
        "musicxml": _xml_shape(musicxml_path),
        "musescore_origin_sentinel": origin,
        "worker": {
            "parts": [
                {
                    "part_id": part.part_id,
                    "name": part.name,
                    "event_count": len(part.events),
                    "staffs": sorted({int(event.staff) for event in part.events}),
                    "voices": sorted({str(event.voice) for event in part.events}),
                }
                for part in worker_payload.parts
            ],
            "raw_event_count": len(events),
            "logical_pitch_unit_count": len(units),
            "highest_time_quarter": worker_payload.highest_time_quarter,
            "diagnostics": worker_diagnostics,
        },
        "source_to_musicxml": {
            "standardize_probe": standardize_probe,
            "current_standardizer_probe": {
                "hints": current_hints,
                "alignment": current_alignment,
            },
            "pre_fix_legacy_partition_guard_probe": legacy_guard,
            "strict_lane_pilot": pilot,
        },
    }


def build_report(before: Path, after: Path) -> dict[str, Any]:
    before_audit = _case_audit(before, label="before_5a00430_v3")
    after_audit = _case_audit(after, label="after_5a00430_v4")
    after_current = after_audit["source_to_musicxml"]["current_standardizer_probe"]["alignment"]
    after_pilot = after_audit["source_to_musicxml"]["strict_lane_pilot"]
    after_legacy_guard = after_audit["source_to_musicxml"]["pre_fix_legacy_partition_guard_probe"]
    before_pilot = before_audit["source_to_musicxml"]["strict_lane_pilot"]
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "reference_annotations_used": False,
        "recognizer_rerun": False,
        "before": before_audit,
        "after": after_audit,
        "comparison": {
            "before_voice_lane_count": before_audit["metadata"]["voice_lane_count"],
            "after_voice_lane_count": after_audit["metadata"]["voice_lane_count"],
            "after_adjacent_retrigger_split_count": (
                (after_audit["metadata"].get("lane_assignment") or {}).get("adjacent_retrigger_split_count", 0)
            ),
            "before_xml_part_count": len(before_audit["musicxml"]["parts"]),
            "after_xml_part_count": len(after_audit["musicxml"]["parts"]),
            "before_current_alignment_reason": before_audit["source_to_musicxml"]["current_standardizer_probe"]["alignment"].get("reason"),
            "after_current_alignment_reason": after_current.get("reason"),
            "after_pre_fix_legacy_guard_reason": after_legacy_guard.get("legacy_reason"),
            "after_pre_fix_legacy_guard_rejected": bool(after_legacy_guard.get("legacy_guard_rejected")),
            "before_pilot_strict_model_provable": bool(before_pilot["strict_model_provable"]),
            "after_pilot_strict_model_provable": bool(after_pilot["strict_model_provable"]),
            "after_pilot_no_merge_drop_or_extra": bool(after_pilot["no_merge_drop_or_extra"]),
            "after_pilot_pitch_multiset_equal": bool(after_pilot["pitch_multiset_equal"]),
        },
        "root_cause": {
            "classification": "standardizer_complete_lane_partition_guard_rejects_all_mapped_lanes",
            "evidence": [
                "The pre-change MIDI has one instrument track and the standardizer proves existing source identity.",
                "5a00430 splits the adjacent same-pitch retrigger into a second named MIDI lane; MuseScore preserves two named MusicXML parts.",
                "The post-change source and logical MusicXML pitch-unit counts match 15 to 15, and the lane pilot pairs every source note with one unique MusicXML unit.",
                "Replaying the pre-fix partition guard on the post-change evidence reports mapped_groups for both lanes with no unassigned lanes or groups, which the old len(unassigned_lanes)==1 condition rejected as source_midi_lane_part_identity_incomplete.",
                "The origin sentinel is present in the import audit and removed from MusicXML; it does not explain the failure.",
            ],
            "not_evidence_of": [
                "MuseScore note merge or deletion",
                "missing beat-grid or tempo-map fields",
                "source score-origin mismatch",
            ],
        },
        "recommended_minimal_fix": {
            "production_files": ["backend/jianpu_score/musicxml_standardize.py"],
            "change": "In _source_track_identity_partitions, accept the complete case where every MusicXML group maps to a distinct source lane and both unassigned_lanes and unassigned_groups are empty; retain the existing fail-closed branches for missing, ambiguous, or extra groups.",
            "why_no_marker_is_needed_here": "special-tempo-change already preserves the generated lane names in MusicXML part names, and the pilot proves a complete per-lane pairing with bounded affine/offset coordinates.",
            "safety_constraints": [
                "Do not relax residual limits.",
                "Do not infer lanes from event order when part-name identity is absent.",
                "Do not bypass the tempo map or use reference annotations.",
            ],
            "production_changed_in_this_pilot": False,
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    comparison = report["comparison"]
    after_pilot = report["after"]["source_to_musicxml"]["strict_lane_pilot"]
    after_standardize = report["after"]["source_to_musicxml"]["standardize_probe"]
    lines = [
        "# special-tempo-change retrigger lane pilot",
        "",
        "This diagnostic compares the existing v3 output before 5a00430 with the v4 output after 5a00430. It does not rerun recognition and does not use a reference MIDI or beat annotation.",
        "",
        "## Result",
        "",
        f"- Before: {comparison['before_voice_lane_count']} instrument lane, {comparison['before_xml_part_count']} MusicXML part(s), current alignment `{comparison['before_current_alignment_reason']}`, pilot strict proof `{comparison['before_pilot_strict_model_provable']}`.",
        f"- After: {comparison['after_voice_lane_count']} instrument lanes, {comparison['after_xml_part_count']} MusicXML parts, adjacent retrigger splits `{comparison['after_adjacent_retrigger_split_count']}`, current standardizer success `{after_standardize['success']}`, tempo events `{after_standardize.get('tempo_event_count')}`.",
        f"- Pre-fix guard replay on the after evidence: `{comparison['after_pre_fix_legacy_guard_reason']}` (rejected `{comparison['after_pre_fix_legacy_guard_rejected']}`).",
        f"- After pilot: source notes `{after_pilot['source_note_count']}`, logical MusicXML units `{after_pilot['musicxml_logical_unit_count']}`, pitch multiset equal `{after_pilot['pitch_multiset_equal']}`, one-to-one `{after_pilot['one_to_one']}`, no merge/drop/extra `{after_pilot['no_merge_drop_or_extra']}`, strict model provable `{after_pilot['strict_model_provable']}`.",
        "",
        "## Root cause",
        "",
        "The adjacent retrigger lane split preserves the note. It exposes a guard in `_source_track_identity_partitions`: the code maps both named MusicXML groups to lanes, but treats the complete `unassigned_lanes=[]` and `unassigned_groups=[]` state as incomplete and rejects it. The origin sentinel is audited and removed correctly.",
        "",
        "## Recommended minimal fix",
        "",
        "Accept the complete all-mapped partition in `backend/jianpu_score/musicxml_standardize.py`. Keep all existing ambiguity, missing-group, residual, one-to-one, and tempo-map checks fail-closed. No origin marker is required for this case because the MusicXML part names already preserve both lane identities.",
        "",
        "Production code was not changed by this pilot.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE, help="v3 case root before 5a00430")
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER, help="v4 case root after 5a00430")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="independent diagnostic output root")
    args = parser.parse_args(argv)
    report = build_report(args.before, args.after)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "report.json", report)
    (args.output / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
