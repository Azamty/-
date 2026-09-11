"""Reference-free integer downbeat-phase diagnostic for Beat Transformer.

The official Beat Transformer pilot already contains the only model outputs used
by this diagnostic.  This module keeps those beat events byte-for-byte and only
tests an integer phase over the fixed DBN period.  Audio observations are used
to decide whether a phase is stable enough to change; annotations are opened
only by :func:`score_development` after ``raw-decision-freeze.json`` exists.

This is a review artifact, not a production routing helper.  In particular it
does not search tempo, offsets, meters, beat times, or model parameters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These helpers are intentionally shared with the earlier reference-free
# refinement pilot.  _select_downbeats is not called: that pilot searches
# meters and adds a meter prior, both of which are forbidden here.
from scripts.pilot_beatnet_observable_refinement import (  # noqa: E402
    _audio_features,
    _contrast,
    _sample_envelope,
)


BT_ROOT = ROOT / ".artifacts" / "review" / "beat-transformer-official-pilot-v1"
DEFAULT_INPUT_MANIFEST = BT_ROOT / "frozen-input-manifest.json"
DEFAULT_RAW_DIR = BT_ROOT / "raw"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beat-transformer-integer-phase-pilot-v1"
DEFAULT_EVALUATION = BT_ROOT / "evaluation.json"

PERIOD_BY_CASE: dict[str, int] = {"special-6-8": 3}
DEFAULT_PERIOD = 4
PHASE_WEIGHTS: dict[str, float] = {
    "onset_contrast": 0.40,
    "low_frequency_contrast": 0.25,
    "bass_alignment": 0.20,
    "bar_stability": 0.15,
}
MIN_COMPLETE_BARS = 3
CROP_MIN_COMPLETE_BARS = 2
MIN_MARGIN = 0.10
TOLERANCE_SEC = 0.07
SCHEMA_VERSION = "bt_integer_phase_diagnostic_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _local_contrast(values: Sequence[float], selected: Sequence[int], segment: Sequence[int]) -> float:
    """Use the earlier pilot's contrast definition on one complete-bar span."""

    segment_set = set(int(index) for index in segment)
    selected_local = {int(index) for index in selected if int(index) in segment_set}
    if not selected_local:
        return 0.0
    local_values = [float(values[index]) for index in segment]
    local_positions = {position for position, index in enumerate(segment) if index in selected_local}
    return float(_contrast(local_values, local_positions))


def _median_interval(beats: Sequence[float]) -> float:
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    return statistics.median(intervals) if intervals else 0.5


def _bass_alignment(
    downbeat_times: Sequence[float], bass_onsets: Sequence[float], interval: float
) -> float:
    """Return the fixed-pilot alignment score without reweighting missing bass."""

    if not downbeat_times or not bass_onsets or interval <= 0:
        return 0.0
    distances = [min(abs(float(onset) - float(beat)) for beat in downbeat_times) / interval for onset in bass_onsets]
    return _clamp(1.0 - statistics.fmean(min(value, 1.0) for value in distances), 0.0, 1.0)


def _complete_bar_starts(length: int, period: int, phase: int) -> list[int]:
    return [start for start in range(int(phase), length, int(period)) if start + int(period) < length]


def _bar_span(starts: Sequence[int], period: int, length: int) -> list[int]:
    if not starts:
        return []
    first = int(starts[0])
    last = min(length, int(starts[-1]) + int(period))
    return list(range(first, last))


def _phase_record(
    *,
    phase: int,
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    bar_numbers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Score one integer phase over the requested complete bars."""

    starts_all = _complete_bar_starts(len(beats), period, phase)
    if bar_numbers is None:
        starts = starts_all
    else:
        starts = [starts_all[index] for index in bar_numbers if 0 <= int(index) < len(starts_all)]
    span = _bar_span(starts, period, len(beats))
    selected = list(starts)
    interval = _median_interval(beats)
    accents = [float(onset_values[index]) + 0.5 * float(low_values[index]) for index in starts]
    components = {
        "onset_contrast": _local_contrast(onset_values, selected, span),
        "low_frequency_contrast": _local_contrast(low_values, selected, span),
        "bass_alignment": _bass_alignment([beats[index] for index in starts], bass_onsets, interval),
        "bar_stability": max(0.0, 1.0 - statistics.pstdev(accents)) if len(accents) > 1 else 0.5,
    }
    score = sum(PHASE_WEIGHTS[key] * float(components[key]) for key in PHASE_WEIGHTS)
    return {
        "phase_index": int(phase),
        "complete_bar_count": len(starts_all),
        "scored_bar_count": len(starts),
        "bar_start_indices": [int(index) for index in starts],
        "components": {key: float(value) for key, value in components.items()},
        "score_higher_is_better": float(score),
    }


def _unique_winner(records: Sequence[Mapping[str, Any]]) -> tuple[int | None, float | None, float | None]:
    if not records:
        return None, None, None
    ordered = sorted(records, key=lambda item: (-float(item["score_higher_is_better"]), int(item["phase_index"])))
    top = float(ordered[0]["score_higher_is_better"])
    second = float(ordered[1]["score_higher_is_better"]) if len(ordered) > 1 else None
    if second is not None and math.isclose(top, second, rel_tol=0.0, abs_tol=1e-12):
        return None, top, second
    return int(ordered[0]["phase_index"]), top, second


def _eligible_phase_records(
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    minimum_complete_bars: int,
) -> list[dict[str, Any]]:
    records = [
        _phase_record(
            phase=phase,
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=period,
        )
        for phase in range(period)
    ]
    return [record for record in records if record["complete_bar_count"] >= minimum_complete_bars]


def _crop_equivariance_check(
    *,
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    original_phase: int,
    selected_phase: int,
) -> dict[str, Any]:
    """Check phase covariance after removing a leading beat.

    A leading crop translates a phase by ``-crop_start`` modulo the fixed
    period.  The crop gate is deliberately weaker on the short fixtures (two
    complete bars) because the full decision still requires three; it prevents
    the check from being vacuous for the 13-beat cases.
    """

    crop_start = 1
    crop_end = len(beats)
    if len(beats) - crop_start < (CROP_MIN_COMPLETE_BARS * period + 1):
        return {"available": False, "passed": False, "reason": "insufficient_beats_for_crop"}
    crop_beats = list(beats[crop_start:crop_end])
    crop_onset = list(onset_values[crop_start:crop_end])
    crop_low = list(low_values[crop_start:crop_end])
    crop_original = (int(original_phase) - crop_start) % period
    crop_result = _select_integer_phase(
        beats=crop_beats,
        onset_values=crop_onset,
        low_values=crop_low,
        bass_onsets=bass_onsets,
        period=period,
        original_phase=crop_original,
        minimum_complete_bars=CROP_MIN_COMPLETE_BARS,
        check_crop=False,
    )
    expected = (int(selected_phase) - crop_start) % period
    passed = crop_result["selected_phase_index"] == expected and crop_result["decision"] != "abstain"
    return {
        "available": True,
        "passed": bool(passed),
        "crop_start_beat_index": crop_start,
        "crop_end_beat_index_exclusive": crop_end,
        "expected_phase_index": expected,
        "cropped_original_phase_index": crop_original,
        "cropped_selected_phase_index": crop_result["selected_phase_index"],
        "cropped_decision": crop_result["decision"],
        "cropped_reason": crop_result["reason"],
    }


def _select_integer_phase(
    *,
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    original_phase: int,
    minimum_complete_bars: int = MIN_COMPLETE_BARS,
    check_crop: bool = True,
) -> dict[str, Any]:
    """Select one fixed-period integer phase, or abstain to the DBN phase."""

    if len(beats) != len(onset_values) or len(beats) != len(low_values):
        raise ValueError("beats and all audio feature samples must have equal length")
    eligible = _eligible_phase_records(
        beats, onset_values, low_values, bass_onsets, period, minimum_complete_bars
    )
    all_records = [
        _phase_record(
            phase=phase,
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=period,
        )
        for phase in range(period)
    ]
    winner, top_score, second_score = _unique_winner(eligible)
    margin = (top_score - second_score) if top_score is not None and second_score is not None else None

    half_records: dict[str, list[dict[str, Any]]] = {"first": [], "last": []}
    for record in eligible:
        phase = int(record["phase_index"])
        bars = _complete_bar_starts(len(beats), period, phase)
        half_size = max(1, math.ceil(len(bars) / 2))
        half_records["first"].append(
            _phase_record(
                phase=phase,
                beats=beats,
                onset_values=onset_values,
                low_values=low_values,
                bass_onsets=bass_onsets,
                period=period,
                bar_numbers=list(range(half_size)),
            )
        )
        half_records["last"].append(
            _phase_record(
                phase=phase,
                beats=beats,
                onset_values=onset_values,
                low_values=low_values,
                bass_onsets=bass_onsets,
                period=period,
                bar_numbers=list(range(max(0, len(bars) - half_size), len(bars))),
            )
        )
    first_winner, first_top, first_second = _unique_winner(half_records["first"])
    last_winner, last_top, last_second = _unique_winner(half_records["last"])

    reason = "selected"
    proposed_phase: int | None = winner
    if original_phase is None:
        proposed_phase = None
        reason = "missing_original_phase"
    elif winner is None:
        proposed_phase = None
        reason = "no_unique_global_winner"
    elif second_score is None:
        proposed_phase = None
        reason = "insufficient_competing_phases"
    elif margin < MIN_MARGIN:
        proposed_phase = None
        reason = "winner_margin_below_0_10"
    elif first_winner is None or last_winner is None or first_winner != last_winner or first_winner != winner:
        proposed_phase = None
        reason = "first_last_half_winner_disagreement"

    crop = {"available": False, "passed": None, "reason": "not_run"}
    if proposed_phase is not None and check_crop:
        crop = _crop_equivariance_check(
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=period,
            original_phase=int(original_phase),
            selected_phase=int(proposed_phase),
        )
        if not crop["passed"]:
            proposed_phase = None
            reason = "crop_equivariance_failed"

    if proposed_phase is None:
        selected_phase = int(original_phase)
        decision = "abstain"
    elif int(proposed_phase) == int(original_phase):
        selected_phase = int(proposed_phase)
        decision = "keep_original"
    else:
        selected_phase = int(proposed_phase)
        decision = "changed"

    return {
        "period_beats": int(period),
        "original_phase_index": int(original_phase),
        "selected_phase_index": selected_phase,
        "proposed_phase_index": proposed_phase,
        "decision": decision,
        "reason": reason,
        "minimum_complete_bars": int(minimum_complete_bars),
        "eligible_phase_count": len(eligible),
        "phase_records": all_records,
        "winner_phase_index": winner,
        "winner_score": top_score,
        "runner_up_score": second_score,
        "winner_margin": margin,
        "half_winners": {
            "first": first_winner,
            "last": last_winner,
            "first_score": first_top,
            "first_runner_up_score": first_second,
            "last_score": last_top,
            "last_runner_up_score": last_second,
        },
        "crop_equivariance": crop,
    }


def _raw_case(raw_path: Path, case_id: str, period: int) -> dict[str, Any]:
    with np.load(raw_path, allow_pickle=False) as raw:
        beats = np.asarray(raw["beat_events"], dtype=np.float64).copy()
        downbeats = np.asarray(raw["downbeat_events"], dtype=np.float64).copy()
    if beats.ndim != 1 or downbeats.ndim != 2 or downbeats.shape[1] < 2:
        raise ValueError(f"unexpected Beat Transformer raw schema for {case_id}")
    positions = downbeats[:, 1]
    downbeat_rows = np.flatnonzero(np.isclose(positions, 1.0, rtol=0.0, atol=1e-9))
    if len(downbeat_rows) == 0:
        raise ValueError(f"raw DBN output has no position-1 row for {case_id}")
    original_phase = int(downbeat_rows[0] % period)
    return {
        "case_id": case_id,
        "period_beats": int(period),
        "beat_events": beats,
        "downbeat_events": downbeats,
        "original_downbeat_times": [float(downbeats[index, 0]) for index in downbeat_rows],
        "original_phase_index": original_phase,
        "beat_event_count": int(len(beats)),
        "downbeat_event_count": int(len(downbeats)),
        "beat_events_sha256": _sha256_bytes(beats.tobytes()),
        "downbeat_events_sha256": _sha256_bytes(downbeats.tobytes()),
    }


def _input_case_manifest(
    case: Mapping[str, Any], raw_dir: Path, period: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(case["id"])
    audio_path = Path(str(case["path"]))
    raw_path = raw_dir / f"{case_id}.npz"
    raw_json_path = raw_dir / f"{case_id}.json"
    if not audio_path.is_file() or not raw_path.is_file() or not raw_json_path.is_file():
        raise FileNotFoundError(f"missing frozen input/raw for {case_id}")
    raw = _raw_case(raw_path, case_id, period)
    manifest = {
        "id": case_id,
        "category": str(case.get("category", "unknown")),
        "audio_path": str(audio_path),
        "audio_bytes": audio_path.stat().st_size,
        "audio_sha256": _sha256(audio_path),
        "raw_npz_path": str(raw_path),
        "raw_npz_bytes": raw_path.stat().st_size,
        "raw_npz_sha256": _sha256(raw_path),
        "raw_metadata_path": str(raw_json_path),
        "raw_metadata_bytes": raw_json_path.stat().st_size,
        "raw_metadata_sha256": _sha256(raw_json_path),
        "period_beats": int(period),
        "beat_event_count": raw["beat_event_count"],
        "downbeat_event_count": raw["downbeat_event_count"],
        "beat_events_sha256": raw["beat_events_sha256"],
        "downbeat_events_sha256": raw["downbeat_events_sha256"],
        "original_phase_index": raw["original_phase_index"],
    }
    return manifest, raw


def _reference_f1(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    used: set[int] = set()
    errors: list[float] = []
    for value in predicted:
        candidates = [
            (abs(float(value) - float(item)), index)
            for index, item in enumerate(reference)
            if index not in used and abs(float(value) - float(item)) <= TOLERANCE_SEC
        ]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            errors.append(error)
    tp = len(errors)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": len(predicted),
        "reference": len(reference),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_error_sec": statistics.fmean(errors) if errors else None,
    }


def _annotation_times(path: Path) -> tuple[list[float], list[float]]:
    payload = _load_json(path)
    records = list(payload["beat_grid"]["beats"])
    beats = [float(item["time_sec"]) for item in records]
    downbeats = [float(item["time_sec"]) for item in records if bool(item.get("downbeat"))]
    return beats, downbeats


def _candidate_downbeats(raw: Mapping[str, Any], decision: Mapping[str, Any]) -> list[float]:
    if decision["decision"] in {"abstain", "keep_original"}:
        return [float(value) for value in raw["original_downbeat_times"]]
    period = int(decision["period_beats"])
    phase = int(decision["selected_phase_index"])
    beats = np.asarray(raw["beat_events"], dtype=np.float64)
    return [float(value) for value in beats[phase::period]]


def _score_development(
    *,
    evaluation_path: Path,
    input_cases: Sequence[Mapping[str, Any]],
    raw_cases: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Open labels only after the raw decision freeze has been written."""

    evaluation = _load_json(evaluation_path)
    evaluation_by_id = {str(item["id"]): item for item in evaluation["cases"]}
    rows: list[dict[str, Any]] = []
    for input_case in input_cases:
        case_id = str(input_case["id"])
        entry = evaluation_by_id[case_id]
        annotation_path = Path(str(entry["annotation_path"]))
        reference_beats, reference_downbeats = _annotation_times(annotation_path)
        raw = raw_cases[case_id]
        decision = decisions[case_id]
        predicted_beats = [float(value) for value in raw["beat_events"]]
        predicted_downbeats = _candidate_downbeats(raw, decision)
        original_downbeats = [float(value) for value in raw["original_downbeat_times"]]
        original_beat_metric = _reference_f1(reference_beats, predicted_beats)
        original_downbeat_metric = _reference_f1(reference_downbeats, original_downbeats)
        candidate_downbeat_metric = _reference_f1(reference_downbeats, predicted_downbeats)
        rows.append(
            {
                "id": case_id,
                "category": input_case.get("category"),
                "annotation_path": str(annotation_path),
                "annotation_sha256": _sha256(annotation_path),
                "reference_meter": entry.get("reference_meter"),
                "original": {
                    "beat": original_beat_metric,
                    "downbeat": original_downbeat_metric,
                },
                "candidate": {
                    "downbeat": candidate_downbeat_metric,
                    "predicted_downbeat_times": predicted_downbeats,
                },
                "beat_preserved": predicted_beats == [float(value) for value in raw["beat_events"]],
                "beat_count_preserved": len(predicted_beats) == int(raw["beat_event_count"]),
                "decision": decision["decision"],
            }
        )
    candidate_db = [float(row["candidate"]["downbeat"]["f1"]) for row in rows]
    non_asap = [row for row in rows if not row["id"].startswith("asap-")]
    original_perfect = [row for row in rows if math.isclose(float(row["original"]["downbeat"]["f1"]), 1.0, rel_tol=0.0, abs_tol=1e-12)]
    original_perfect_preserved = all(
        float(row["candidate"]["downbeat"]["f1"]) + 1e-12 >= float(row["original"]["downbeat"]["f1"])
        for row in original_perfect
    )
    all_beats_exact = all(row["beat_preserved"] and row["beat_count_preserved"] for row in rows)
    summary = {
        "case_count": len(rows),
        "downbeat_f1_macro": statistics.fmean(candidate_db) if candidate_db else None,
        "non_asap_downbeat_f1_macro": statistics.fmean(
            float(row["candidate"]["downbeat"]["f1"]) for row in non_asap
        ) if non_asap else None,
        "original_downbeat_f1_macro": statistics.fmean(
            float(row["original"]["downbeat"]["f1"]) for row in rows
        ) if rows else None,
        "original_downbeat_f1_equals_1_count": len(original_perfect),
        "original_perfect_cases": [row["id"] for row in original_perfect],
        "original_perfect_not_decreased": original_perfect_preserved,
        "all_beats_float_and_count_exact": all_beats_exact,
    }
    gate = {
        "downbeat_macro_at_least_0_75": summary["downbeat_f1_macro"] is not None and summary["downbeat_f1_macro"] >= 0.75,
        "non_asap_downbeat_macro_at_least_0_80": summary["non_asap_downbeat_f1_macro"] is not None and summary["non_asap_downbeat_f1_macro"] >= 0.80,
        "exactly_two_original_downbeat_f1_equals_1": summary["original_downbeat_f1_equals_1_count"] == 2,
        "original_f1_equals_1_not_decreased": bool(original_perfect_preserved),
        "beat_float_and_count_unchanged": all_beats_exact,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "labels_read_after_raw_decision_freeze": True,
        "tolerance_sec": TOLERANCE_SEC,
        "cases": rows,
        "summary": summary,
        "development_gate": gate,
        "development_gate_passed": all(gate.values()),
        "decision": "eligible_for_heldout8" if all(gate.values()) else "stop_after_development8",
    }


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        files.append({"path": str(path.relative_to(output_root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    tracked_sources: list[dict[str, Any]] = []
    for source in (ROOT / "scripts" / "pilot_bt_integer_phase_diagnostic.py", ROOT / "tests" / "test_pilot_bt_integer_phase_diagnostic.py"):
        if source.is_file():
            tracked_sources.append({"path": str(source.relative_to(ROOT)).replace("\\", "/"), "bytes": source.stat().st_size, "sha256": _sha256(source)})
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(output_root),
        "files": files,
        "tracked_sources": tracked_sources,
    }


def run(
    *,
    input_manifest_path: Path = DEFAULT_INPUT_MANIFEST,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_root: Path = DEFAULT_OUTPUT,
    evaluation_path: Path = DEFAULT_EVALUATION,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    input_source = _load_json(input_manifest_path)
    source_cases = list(input_source["cases"])
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen_before_annotation_access",
        "selection": {
            "case_ids": [str(case["id"]) for case in source_cases],
            "source": str(input_manifest_path),
            "source_sha256": _sha256(input_manifest_path),
            "raw_dir": str(raw_dir),
        },
        "phase_policy": {
            "period_beats": "4 for seven fixed cases; 3 for special-6-8",
            "enumeration": "integer phases 0..period-1 only",
            "weights": PHASE_WEIGHTS,
            "minimum_complete_bars": MIN_COMPLETE_BARS,
            "first_last_half_same_winner": True,
            "winner_margin_at_least": MIN_MARGIN,
            "crop_equivariance": "leading one-beat crop; crop-only check may use two complete bars on short fixtures",
        },
        "forbidden_searches": ["beat_time", "beat_count", "tempo", "offset_sec", "meter", "segment_phase", "velocity", "drum", "chroma", "oracle", "reference_labels"],
        "observable_sources": {
            "audio": "raw frozen WAV clips",
            "beat_events": "Beat Transformer official raw NPZ, unchanged",
            "downbeat_period": "fixed prior from official joint DBN output and preregistration",
            "muscriptor_game_notes": "unavailable for these eight raw artifacts; no bass renormalization",
            "bass_alignment": "zero component when no raw model bass onsets are supplied",
        },
        "post_hoc_labels": "opened only after raw-decision-freeze.json is written",
    }
    _write_json(output_root / "preregistration.json", preregistration)

    input_cases: list[dict[str, Any]] = []
    raw_cases: dict[str, dict[str, Any]] = {}
    for case in source_cases:
        case_id = str(case["id"])
        period = int(PERIOD_BY_CASE.get(case_id, DEFAULT_PERIOD))
        manifest, raw = _input_case_manifest(case, raw_dir, period)
        input_cases.append(manifest)
        raw_cases[case_id] = raw
    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "annotation_accessed": False,
        "source_manifest_path": str(input_manifest_path),
        "source_manifest_sha256": _sha256(input_manifest_path),
        "raw_directory": str(raw_dir),
        "raw_freeze_source": str(BT_ROOT / "raw-freeze-manifest.json"),
        "raw_freeze_source_sha256": _sha256(BT_ROOT / "raw-freeze-manifest.json"),
        "cases": input_cases,
    }
    _write_json(output_root / "input-manifest.json", input_manifest)

    decisions: dict[str, dict[str, Any]] = {}
    raw_case_records: list[dict[str, Any]] = []
    for input_case in input_cases:
        case_id = str(input_case["id"])
        raw = raw_cases[case_id]
        features = _audio_features(Path(str(input_case["audio_path"])))
        beats = [float(value) for value in raw["beat_events"]]
        onset_values = _sample_envelope(features["onset"], beats)
        low_values = _sample_envelope(features["low_onset"], beats)
        bass_onsets: list[float] = []
        decision = _select_integer_phase(
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=int(input_case["period_beats"]),
            original_phase=int(raw["original_phase_index"]),
        )
        decision = {
            **decision,
            "audio_observations": {
                "duration_sec": float(features["duration_sec"]),
                "onset_sample_count": len(onset_values),
                "low_frequency_sample_count": len(low_values),
                "bass_onsets_available": False,
                "bass_onset_count": 0,
                "bass_reweighted": False,
                "velocity_used": False,
                "drum_used": False,
                "chroma_used": False,
            },
            "beat_output": {
                "input_count": len(beats),
                "output_count": len(beats),
                "input_sha256": raw["beat_events_sha256"],
                "output_sha256": _sha256_bytes(np.asarray(beats, dtype=np.float64).tobytes()),
                "float_values_unchanged": True,
            },
            "raw_sources": {
                "raw_npz_sha256": input_case["raw_npz_sha256"],
                "raw_metadata_sha256": input_case["raw_metadata_sha256"],
                "audio_sha256": input_case["audio_sha256"],
            },
        }
        decisions[case_id] = decision
        raw_case_records.append(
            {
                "id": case_id,
                "raw_npz_sha256": input_case["raw_npz_sha256"],
                "audio_sha256": input_case["audio_sha256"],
                "decision_sha256": _sha256_bytes(_canonical_bytes(decision)),
                "beat_count": len(beats),
                "beat_events_sha256": raw["beat_events_sha256"],
            }
        )

    raw_decision_freeze = {
        "schema_version": SCHEMA_VERSION,
        "annotation_accessed_before_freeze": False,
        "freeze_created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": _sha256(output_root / "preregistration.json"),
        "input_manifest_sha256": _sha256(output_root / "input-manifest.json"),
        "source_raw_freeze_manifest_sha256": _sha256(BT_ROOT / "raw-freeze-manifest.json"),
        "case_count": len(raw_case_records),
        "cases": raw_case_records,
        "decisions": decisions,
        "decision_semantics": "only this frozen decision may be scored; no annotation/oracle value participated",
    }
    _write_json(output_root / "raw-decision-freeze.json", raw_decision_freeze)

    # This is the first point where this diagnostic reads any annotation path
    # or label content.  The freeze above is durable before this call starts.
    evaluation = _score_development(
        evaluation_path=evaluation_path,
        input_cases=input_cases,
        raw_cases=raw_cases,
        decisions=decisions,
    )
    _write_json(output_root / "evaluation.json", evaluation)

    decision = evaluation["decision"]
    if evaluation["development_gate_passed"]:
        heldout_status = {
            "attempted": False,
            "status": "blocked_pending_strict_asap_discovery_implementation",
            "reason": "The development gate unexpectedly passed; no heldout labels or model run were started by this bounded diagnostic.",
        }
    else:
        heldout_status = {
            "attempted": False,
            "status": "stopped_after_development_failure",
            "reason": "The preregistered development gate failed; no new ASAP works were opened or selected.",
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "development_decision": decision,
        "heldout8": heldout_status,
        "provenance": {
            "labels_already_present_before_experiment": True,
            "weights_and_labels_are_theoretical_diagnostic_only": True,
            "production_modified": False,
        },
        "evaluation_summary": evaluation["summary"],
        "gate": evaluation["development_gate"],
    }
    _write_json(output_root / "report.json", report)
    _write_json(output_root / "artifact-manifest.json", _artifact_manifest(output_root))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    report = run(
        input_manifest_path=args.input_manifest,
        raw_dir=args.raw_dir,
        output_root=args.output,
        evaluation_path=args.evaluation,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
