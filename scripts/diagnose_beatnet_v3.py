"""Attribute BeatNet errors in the immutable production-acceptance-v3 batch.

This is an offline diagnostic.  Reference annotations are used only for the
explicit oracle measurements; the onset candidate scorer never reads them.
Nothing produced here is consumed by the production pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beatnet-v3-error-attribution"
TOLERANCE_SEC = 0.07
# Reference annotations express positions in quarter-note beats.  A 6/8 bar is
# therefore three quarter notes here; its compound 2-beat feel is not encoded
# in the timestamp-only F1 metric.
METERS = {"2/4": 2, "3/4": 3, "4/4": 4, "6/8": 3}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f1(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    used: set[int] = set()
    errors: list[float] = []
    for value in predicted:
        candidates = [
            (abs(value - item), index)
            for index, item in enumerate(reference)
            if index not in used and abs(value - item) <= TOLERANCE_SEC
        ]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            errors.append(error)
    tp = len(errors)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": len(predicted),
        "reference": len(reference),
        "precision": precision,
        "recall": recall,
        "f1": score,
        "mean_matched_error_sec": statistics.fmean(errors) if errors else None,
    }


def _best_shift(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    if not reference or not predicted:
        return {"shift_sec": 0.0, "metrics": _f1(reference, predicted)}
    differences = {0.0}
    for ref in reference:
        for pred in predicted:
            difference = ref - pred
            if abs(difference) <= 2.0:
                differences.add(round(difference, 6))
    best = None
    for shift in sorted(differences):
        metrics = _f1(reference, [value + shift for value in predicted])
        rank = (metrics["f1"], metrics["true_positive"], -(metrics["mean_matched_error_sec"] or 9), -abs(shift))
        if best is None or rank > best[0]:
            best = (rank, shift, metrics)
    assert best is not None
    return {"shift_sec": best[1], "metrics": best[2]}


def _linear_regression(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 1e-12:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    return slope, mean_y - slope * mean_x


def _best_affine(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    """Search plausible monotone index alignments, then fit one scale and offset."""
    baseline = {"scale": 1.0, "offset_sec": 0.0, "metrics": _f1(reference, predicted)}
    if len(reference) < 2 or len(predicted) < 2:
        return baseline
    shift = _best_shift(reference, predicted)["shift_sec"]
    candidates: list[tuple[float, float]] = [(1.0, 0.0), (1.0, shift)]
    # Quantile-aligned windows cover missing leading/trailing tracker events and
    # octave errors without granting local time warping to this oracle.
    for pred_start in range(min(5, len(predicted) - 1)):
        for ref_start in range(min(5, len(reference) - 1)):
            length = min(len(predicted) - pred_start, len(reference) - ref_start)
            if length < 2:
                continue
            sample_count = min(12, length)
            indices = sorted({round(i * (length - 1) / (sample_count - 1)) for i in range(sample_count)})
            xs = [predicted[pred_start + i] for i in indices]
            ys = [reference[ref_start + i] for i in indices]
            fit = _linear_regression(xs, ys)
            if fit and 0.45 <= fit[0] <= 2.2 and abs(fit[1]) <= 4.0:
                candidates.append(fit)
    best = None
    for scale, offset in candidates:
        transformed = [scale * value + offset for value in predicted]
        metrics = _f1(reference, transformed)
        rank = (metrics["f1"], metrics["true_positive"], -(metrics["mean_matched_error_sec"] or 9), -abs(math.log2(scale)))
        if best is None or rank > best[0]:
            best = (rank, scale, offset, metrics)
    assert best is not None
    return {"scale": best[1], "offset_sec": best[2], "metrics": best[3]}


def _piecewise_upper(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    """Count-constrained upper bound for arbitrary monotone local time warping."""
    matched = min(len(reference), len(predicted))
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "description": "monotone_piecewise_time_warp_count_upper_bound",
        "true_positive_upper_bound": matched,
        "predicted": len(predicted),
        "reference": len(reference),
        "f1_upper_bound": score,
    }


def _meter_oracle(reference_downbeats: Sequence[float], beats: Sequence[float], reference_meter: str | None = None) -> dict[str, Any]:
    per_meter: dict[str, Any] = {}
    best = None
    for label, period in METERS.items():
        meter_best = None
        for phase in range(period):
            predicted = list(beats[phase::period])
            metrics = _f1(reference_downbeats, predicted)
            rank = (metrics["f1"], metrics["true_positive"], -phase)
            if meter_best is None or rank > meter_best[0]:
                meter_best = (rank, phase, metrics)
        assert meter_best is not None
        per_meter[label] = {"phase_index": meter_best[1], "metrics": meter_best[2]}
        reference_tie_break = int(label == reference_meter)
        rank = (meter_best[2]["f1"], meter_best[2]["true_positive"], reference_tie_break, -list(METERS).index(label))
        if best is None or rank > best[0]:
            best = (rank, label, meter_best[1], meter_best[2])
    assert best is not None
    return {"meters": per_meter, "best_meter": best[1], "best_phase_index": best[2], "metrics": best[3]}


def _note_groups(notes: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    groups = {"all": [], "bass": [], "drum": []}
    bass_words = ("bass", "contrabass", "tuba", "cello")
    for note in notes:
        try:
            onset = float(note["start_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        groups["all"].append(onset)
        name = " ".join(str(note.get(key, "")).lower() for key in ("instrument_group", "voice_id", "source"))
        if bool(note.get("is_drum")) or "drum" in name or "percussion" in name:
            groups["drum"].append(onset)
        if any(word in name for word in bass_words) or int(note.get("midi", 60)) < 48:
            groups["bass"].append(onset)
    return {key: sorted(set(round(value, 6) for value in values)) for key, values in groups.items()}


def _normalized_deviation(onsets: Sequence[float], beats: Sequence[float]) -> tuple[float | None, float | None]:
    if not onsets or len(beats) < 2:
        return None, None
    interval = statistics.median(right - left for left, right in zip(beats, beats[1:]) if right > left)
    distances = [min(abs(onset - beat) for beat in beats) / interval for onset in onsets]
    return statistics.fmean(min(value, 1.0) for value in distances), sum(value <= 0.18 for value in distances) / len(distances)


def _candidate_evidence(beats: Sequence[float], groups: Mapping[str, Sequence[float]], factor: float) -> dict[str, Any]:
    source_weights = {"all": 1.0, "bass": 2.0, "drum": 3.0}
    sources: dict[str, Any] = {}
    weighted_deviation = weighted_miss = weight_total = 0.0
    for source, weight in source_weights.items():
        deviation, coverage = _normalized_deviation(groups[source], beats)
        sources[source] = {"onset_count": len(groups[source]), "mean_normalized_deviation": deviation, "coverage_within_0_18_beat": coverage}
        if deviation is not None and coverage is not None:
            weighted_deviation += weight * deviation
            weighted_miss += weight * (1.0 - coverage)
            weight_total += weight
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    mean_interval = statistics.fmean(intervals) if intervals else 0.0
    stability = statistics.pstdev(intervals) / mean_interval if len(intervals) > 1 and mean_interval else 1.0
    smoothness = (
        statistics.fmean(abs(c - 2 * b + a) / mean_interval for a, b, c in zip(beats, beats[1:], beats[2:]))
        if len(beats) >= 3 and mean_interval else 1.0
    )
    if weight_total:
        weighted_deviation /= weight_total
        weighted_miss /= weight_total
    else:
        weighted_deviation = weighted_miss = 1.0
    octave_prior = 0.12 * abs(math.log2(factor)) if factor > 0 else 1.0
    score = weighted_deviation + 0.35 * weighted_miss + 0.20 * stability + 0.15 * smoothness + octave_prior
    return {
        "sources": sources,
        "weighted_onset_deviation": weighted_deviation,
        "weighted_miss_rate": weighted_miss,
        "bar_interval_cv": stability,
        "tempo_change_smoothness": smoothness,
        "octave_prior": octave_prior,
        "score_lower_is_better": score,
    }


def _times(records: Iterable[Mapping[str, Any]], *, downbeats: bool = False) -> list[float]:
    return [float(item["time_sec"]) for item in records if not downbeats or bool(item.get("downbeat"))]


def _candidate_rows(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    window = grid.get("context", {}).get("window", {})
    absolute_start = window.get("absolute_start_sec")
    absolute_end = window.get("absolute_end_sec")
    for item in grid.get("tempo", {}).get("candidates", []):
        source_times = [float(value) for value in item.get("beat_times", [])]
        if absolute_start is not None and absolute_end is not None:
            beat_times = [value - float(absolute_start) for value in source_times if float(absolute_start) <= value <= float(absolute_end)]
        else:
            beat_times = source_times
        rows.append({
            "label": str(item.get("label")),
            "factor": float(item.get("factor", 1.0)),
            "bpm": item.get("bpm"),
            "beat_times": beat_times,
            "source_beat_times": source_times,
            "source_time_basis": "full_track_absolute" if absolute_start is not None else "case_local",
            "runtime_selected": bool(item.get("selected")),
        })
    if not rows:
        times = _times(grid.get("beats", []))
        rows.append({"label": "original", "factor": 1.0, "bpm": None, "beat_times": times, "source_beat_times": times, "source_time_basis": "case_local", "runtime_selected": True})
    return rows


def _classify(current: float, best_candidate: Mapping[str, Any], phase: Mapping[str, Any], affine: Mapping[str, Any], piecewise: Mapping[str, Any]) -> str:
    candidate_f1 = float(best_candidate["direct_metrics"]["f1"])
    phase_f1 = float(phase["metrics"]["f1"])
    affine_f1 = float(affine["metrics"]["f1"])
    piecewise_f1 = float(piecewise["f1_upper_bound"])
    if piecewise_f1 < 0.85:
        return "raw_tracker_failure"
    if best_candidate["factor"] != 1.0 and candidate_f1 >= max(0.85, current + 0.10):
        return "octave"
    if phase_f1 >= max(0.85, current + 0.10):
        return "phase"
    if affine_f1 >= max(0.85, phase_f1 + 0.05):
        return "global_tempo_drift"
    if piecewise_f1 >= max(0.85, affine_f1 + 0.05):
        return "local_drift"
    return "raw_tracker_failure" if current < 0.85 else "none"


def diagnose(batch_root: Path, output_root: Path) -> dict[str, Any]:
    selection = _load(batch_root / "raw-selection.json")
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    eval_cases = {item["id"]: item for item in evaluator["cases"]}
    cases: list[dict[str, Any]] = []
    for selected in selection["selected"]:
        case_id = selected["case_id"]
        raw_root = batch_root / case_id / "raw"
        recognition_path = raw_root / "recognition.json"
        grid_path = raw_root / "beat_grid.json"
        recognition, grid = _load(recognition_path), _load(grid_path)
        registry_case = eval_cases[case_id]
        annotation_path = Path(registry_case["beat_annotation"]["path"])
        audio_path = Path(registry_case["input"]["path"])
        annotation = _load(annotation_path)["beat_grid"]
        reference_beats = _times(annotation.get("beats", []))
        reference_downbeats = _times(annotation.get("beats", []), downbeats=True)
        current_beats = _times(grid.get("beats", []))
        current_downbeats = _times(grid.get("beats", []), downbeats=True)
        current_metrics = _f1(reference_beats, current_beats)
        current_downbeat_metrics = _f1(reference_downbeats, current_downbeats)
        evaluator_beat = registry_case["metrics"]["beat_f1"]
        evaluator_downbeat = registry_case["metrics"]["downbeat_f1"]
        groups = _note_groups(recognition.get("notes", []))
        candidates = _candidate_rows(grid)
        for candidate in candidates:
            candidate["direct_metrics"] = _f1(reference_beats, candidate["beat_times"])
            candidate["no_reference_evidence"] = _candidate_evidence(candidate["beat_times"], groups, candidate["factor"])
            affine_oracle = _best_affine(reference_beats, candidate["beat_times"])
            affine_beats = [affine_oracle["scale"] * value + affine_oracle["offset_sec"] for value in candidate["beat_times"]]
            candidate["reference_oracles"] = {
                "fixed_phase_shift": _best_shift(reference_beats, candidate["beat_times"]),
                "single_affine_tempo_offset": affine_oracle,
                "local_piecewise_upper_bound": _piecewise_upper(reference_beats, candidate["beat_times"]),
                "meter_downbeat": _meter_oracle(reference_downbeats, candidate["beat_times"], annotation.get("time_signature")),
                "affine_aligned_meter_downbeat": _meter_oracle(reference_downbeats, affine_beats, annotation.get("time_signature")),
            }
        oracle_candidate = max(candidates, key=lambda item: (item["direct_metrics"]["f1"], item["direct_metrics"]["true_positive"], -abs(math.log2(item["factor"]))))
        evidence_candidate = min(candidates, key=lambda item: (item["no_reference_evidence"]["score_lower_is_better"], abs(math.log2(item["factor"]))))
        raw_original = next((item for item in candidates if item["label"] == "original"), candidates[0])
        phase_candidate = max(candidates, key=lambda item: item["reference_oracles"]["fixed_phase_shift"]["metrics"]["f1"])
        affine_candidate = max(candidates, key=lambda item: item["reference_oracles"]["single_affine_tempo_offset"]["metrics"]["f1"])
        piecewise_candidate = max(candidates, key=lambda item: item["reference_oracles"]["local_piecewise_upper_bound"]["f1_upper_bound"])
        meter_candidate = max(candidates, key=lambda item: item["reference_oracles"]["meter_downbeat"]["metrics"]["f1"])
        affine_meter_candidate = max(candidates, key=lambda item: item["reference_oracles"]["affine_aligned_meter_downbeat"]["metrics"]["f1"])
        phase = {"candidate_label": phase_candidate["label"], "candidate_factor": phase_candidate["factor"], **phase_candidate["reference_oracles"]["fixed_phase_shift"]}
        affine = {"candidate_label": affine_candidate["label"], "candidate_factor": affine_candidate["factor"], **affine_candidate["reference_oracles"]["single_affine_tempo_offset"]}
        piecewise = {"candidate_label": piecewise_candidate["label"], "candidate_factor": piecewise_candidate["factor"], **piecewise_candidate["reference_oracles"]["local_piecewise_upper_bound"]}
        meter = {"candidate_label": meter_candidate["label"], "candidate_factor": meter_candidate["factor"], **meter_candidate["reference_oracles"]["meter_downbeat"]}
        affine_meter = {"candidate_label": affine_meter_candidate["label"], "candidate_factor": affine_meter_candidate["factor"], **affine_meter_candidate["reference_oracles"]["affine_aligned_meter_downbeat"]}
        primary = _classify(current_metrics["f1"], oracle_candidate, phase, affine, piecewise)
        meter_failure = current_downbeat_metrics["f1"] < 0.75 and meter["metrics"]["f1"] >= max(0.75, current_downbeat_metrics["f1"] + 0.10)
        cases.append({
            "case_id": case_id,
            "category": registry_case.get("category"),
            "paths": {"audio": str(audio_path), "raw_recognition": str(recognition_path), "raw_beat_grid": str(grid_path), "reference_annotation": str(annotation_path)},
            "immutable_validation": {
                "recognition_sha256": _sha256(recognition_path),
                "recognition_matches_selection": _sha256(recognition_path) == selected["raw_recognition_sha256"],
                "beat_grid_sha256": _sha256(grid_path),
                "beat_grid_matches_selection": _sha256(grid_path) == selected["source_beat_grid_sha256"],
                "audio_sha256": _sha256(audio_path),
                "audio_matches_registry": _sha256(audio_path) == registry_case["input"]["sha256"],
                "reference_annotation_sha256": _sha256(annotation_path),
                "reference_annotation_matches_registry": _sha256(annotation_path) == registry_case["beat_annotation"]["sha256"],
            },
            "reference": {"beat_count": len(reference_beats), "downbeat_count": len(reference_downbeats), "meter": annotation.get("time_signature")},
            "current": {
                "beat_metrics": current_metrics, "downbeat_metrics": current_downbeat_metrics,
                "matches_v3_evaluator": abs(current_metrics["f1"] - evaluator_beat["f1"]) < 1e-12 and abs(current_downbeat_metrics["f1"] - evaluator_downbeat["f1"]) < 1e-12,
                "selected_factor": grid.get("tempo", {}).get("selected_factor"), "selected_meter": grid.get("time_signature", {}).get("selected"),
            },
            "recoverable_raw_beatnet": {
                "label": raw_original["label"], "factor": raw_original["factor"],
                "source_time_basis": raw_original["source_time_basis"],
                "source_beat_count": len(raw_original["source_beat_times"]), "source_beat_times": raw_original["source_beat_times"],
                "evaluation_window_beat_count": len(raw_original["beat_times"]), "evaluation_window_beat_times": raw_original["beat_times"],
            },
            "tempo_candidates": candidates,
            "candidate_oracle": {"label": oracle_candidate["label"], "factor": oracle_candidate["factor"], "metrics": oracle_candidate["direct_metrics"]},
            "no_reference_selection": {
                "label": evidence_candidate["label"], "factor": evidence_candidate["factor"],
                "agrees_with_candidate_oracle": evidence_candidate["label"] == oracle_candidate["label"],
                "realized_reference_metrics_for_diagnostic_only": evidence_candidate["direct_metrics"],
            },
            "oracles_reference_only": {"fixed_phase_shift": phase, "single_affine_tempo_offset": affine, "local_piecewise_upper_bound": piecewise, "meter_downbeat": meter, "affine_aligned_meter_downbeat": affine_meter},
            "attribution": {"primary_beat_error": primary, "meter_error": meter_failure},
        })

    counts: dict[str, int] = {}
    counts_by_category: dict[str, dict[str, int]] = {}
    for case in cases:
        key = case["attribution"]["primary_beat_error"]
        counts[key] = counts.get(key, 0) + 1
        category = str(case["category"])
        category_counts = counts_by_category.setdefault(category, {})
        category_counts[key] = category_counts.get(key, 0) + 1
    agreement = sum(case["no_reference_selection"]["agrees_with_candidate_oracle"] for case in cases) / len(cases)
    evidence_realized_f1 = statistics.fmean(case["no_reference_selection"]["realized_reference_metrics_for_diagnostic_only"]["f1"] for case in cases)
    candidate_only_f1 = statistics.fmean(case["candidate_oracle"]["metrics"]["f1"] for case in cases)
    phase_f1 = statistics.fmean(case["oracles_reference_only"]["fixed_phase_shift"]["metrics"]["f1"] for case in cases)
    affine_f1 = statistics.fmean(case["oracles_reference_only"]["single_affine_tempo_offset"]["metrics"]["f1"] for case in cases)
    piecewise_f1 = statistics.fmean(case["oracles_reference_only"]["local_piecewise_upper_bound"]["f1_upper_bound"] for case in cases)
    meter_f1 = statistics.fmean(case["oracles_reference_only"]["meter_downbeat"]["metrics"]["f1"] for case in cases)
    affine_meter_f1 = statistics.fmean(case["oracles_reference_only"]["affine_aligned_meter_downbeat"]["metrics"]["f1"] for case in cases)
    report = {
        "schema_version": "beatnet_v3_error_attribution_1",
        "diagnostic_only": True,
        "runtime_consumed": False,
        "tolerance_sec": TOLERANCE_SEC,
        "case_count": len(cases),
        "source_batch": str(batch_root),
        "summary": {
            "current_mean_beat_f1": statistics.fmean(case["current"]["beat_metrics"]["f1"] for case in cases),
            "current_mean_downbeat_f1": statistics.fmean(case["current"]["downbeat_metrics"]["f1"] for case in cases),
            "candidate_oracle_mean_beat_f1": candidate_only_f1,
            "fixed_phase_oracle_mean_beat_f1": phase_f1,
            "single_affine_oracle_mean_beat_f1": affine_f1,
            "piecewise_count_upper_mean_beat_f1": piecewise_f1,
            "meter_oracle_mean_downbeat_f1": meter_f1,
            "affine_aligned_meter_oracle_mean_downbeat_f1": affine_meter_f1,
            "no_reference_candidate_agreement_rate": agreement,
            "no_reference_selection_realized_mean_beat_f1": evidence_realized_f1,
            "primary_error_counts": counts,
            "primary_error_counts_by_category": counts_by_category,
            "onset_source_case_counts": {
                source: sum(bool(case["tempo_candidates"][0]["no_reference_evidence"]["sources"][source]["onset_count"]) for case in cases)
                for source in ("all", "bass", "drum")
            },
            "immutable_inputs_verified_case_count": sum(
                all(value for key, value in case["immutable_validation"].items() if key.endswith("_matches_selection") or key.endswith("_matches_registry"))
                for case in cases
            ),
            "current_metrics_match_v3_evaluator_case_count": sum(case["current"]["matches_v3_evaluator"] for case in cases),
            "meter_error_case_count": sum(case["attribution"]["meter_error"] for case in cases),
            "candidate_selection_alone_can_reach_beat_0_85": candidate_only_f1 >= 0.85,
            "candidate_and_meter_selection_can_reach_downbeat_0_75": meter_f1 >= 0.75,
        },
        "limitations": [
            "All oracle fields read reference annotations and are forbidden from runtime use.",
            "The piecewise result is a count-constrained upper bound, not an implementable tracker score.",
            "The no-reference selector uses only immutable MuScriptor all/bass/drum onsets and candidate-grid regularity; it does not inspect reference annotations.",
            "CCMusic windows share one source song and therefore are not statistically independent recordings.",
        ],
        "cases": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "attribution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, output_root / "attribution.md")
    return report


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# BeatNet v3 error attribution",
        "",
        "This report is offline diagnostic evidence. Oracle values use reference annotations and are not available to production runtime.",
        "",
        "## Aggregate",
        "",
        "| measurement | value |",
        "|---|---:|",
        f"| cases | {report['case_count']} |",
        f"| current beat F1 | {summary['current_mean_beat_f1']:.6f} |",
        f"| current downbeat F1 | {summary['current_mean_downbeat_f1']:.6f} |",
        f"| half/original/double candidate oracle beat F1 | {summary['candidate_oracle_mean_beat_f1']:.6f} |",
        f"| fixed phase oracle beat F1 | {summary['fixed_phase_oracle_mean_beat_f1']:.6f} |",
        f"| single affine oracle beat F1 | {summary['single_affine_oracle_mean_beat_f1']:.6f} |",
        f"| local piecewise count upper bound | {summary['piecewise_count_upper_mean_beat_f1']:.6f} |",
        f"| meter/phase oracle downbeat F1 | {summary['meter_oracle_mean_downbeat_f1']:.6f} |",
        f"| affine-aligned meter/phase oracle downbeat F1 | {summary['affine_aligned_meter_oracle_mean_downbeat_f1']:.6f} |",
        f"| no-reference selector agrees with candidate oracle | {summary['no_reference_candidate_agreement_rate']:.1%} |",
        f"| no-reference selector realized beat F1 (post-hoc evaluation) | {summary['no_reference_selection_realized_mean_beat_f1']:.6f} |",
        "",
        "Primary beat attribution counts: " + ", ".join(f"`{key}`={value}" for key, value in sorted(summary["primary_error_counts"].items())) + ".",
        "Attribution by registry category: " + "; ".join(f"`{category}` " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) for category, counts in sorted(summary["primary_error_counts_by_category"].items())) + ".",
        "Onset evidence availability: " + ", ".join(f"{source}={count}/{report['case_count']}" for source, count in summary["onset_source_case_counts"].items()) + ".",
        f"Meter errors recoverable by meter/phase oracle: {summary['meter_error_case_count']} cases.",
        "",
        "## Per case",
        "",
        "| case | current beat | candidate | phase | affine | piecewise upper | current downbeat | meter oracle | attribution | no-ref agrees |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for case in report["cases"]:
        oracle = case["oracles_reference_only"]
        lines.append(
            "| {id} | {current:.3f} | {candidate:.3f} ({factor:g}x) | {phase:.3f} | {affine:.3f} | {piecewise:.3f} | {down:.3f} | {meter:.3f} ({meter_name}) | {kind}{meter_flag} | {agree} |".format(
                id=case["case_id"], current=case["current"]["beat_metrics"]["f1"], candidate=case["candidate_oracle"]["metrics"]["f1"], factor=case["candidate_oracle"]["factor"],
                phase=oracle["fixed_phase_shift"]["metrics"]["f1"], affine=oracle["single_affine_tempo_offset"]["metrics"]["f1"], piecewise=oracle["local_piecewise_upper_bound"]["f1_upper_bound"],
                down=case["current"]["downbeat_metrics"]["f1"], meter=oracle["meter_downbeat"]["metrics"]["f1"], meter_name=f"{oracle['meter_downbeat']['best_meter']} @{oracle['meter_downbeat']['candidate_factor']:g}x",
                kind=case["attribution"]["primary_beat_error"], meter_flag=" + meter" if case["attribution"]["meter_error"] else "", agree="yes" if case["no_reference_selection"]["agrees_with_candidate_oracle"] else "no",
            )
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        f"Candidate selection alone reaches mean beat F1 0.85: **{str(summary['candidate_selection_alone_can_reach_beat_0_85']).lower()}**.",
        f"An oracle meter/phase choice reaches mean downbeat F1 0.75: **{str(summary['candidate_and_meter_selection_can_reach_downbeat_0_75']).lower()}**.",
        "The latter is an upper-bound attribution result and does not establish that a reference-free meter selector can attain it.",
        "Meter labels that produce the same downbeat timestamps at different tempo factors are indistinguishable under timestamp F1; the JSON retains both the factor and meter instead of claiming semantic meter identity.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = diagnose(args.batch_root.resolve(), args.output_root.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
