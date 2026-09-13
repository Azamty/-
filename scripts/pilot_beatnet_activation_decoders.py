"""Pilot fixed global activation ensembles and official madmom DBN settings.

The script reuses saved BeatNet activations. All candidate decodes are written
before any reference annotation is opened; references are post-hoc evaluation
only. Nothing here is imported by the production pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_ACTIVATIONS = ROOT / ".artifacts" / "review" / "beatnet-official-config-pilot-v1" / "raw"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beatnet-activation-decoder-pilot-v1"
TOLERANCE_SEC = 0.07

DBN_PROFILES: tuple[dict[str, Any], ...] = (
    {"id": "default_234", "beats_per_bar": [2, 3, 4]},
    {"id": "default_2346", "beats_per_bar": [2, 3, 4, 6]},
    {"id": "wide_tempo", "beats_per_bar": [2, 3, 4, 6], "min_bpm": 35.0, "max_bpm": 260.0, "num_tempi": 80},
    {"id": "flexible_tempo", "beats_per_bar": [2, 3, 4, 6], "min_bpm": 40.0, "max_bpm": 240.0, "num_tempi": 80, "transition_lambda": 60},
    {"id": "stable_tempo", "beats_per_bar": [2, 3, 4, 6], "min_bpm": 40.0, "max_bpm": 240.0, "num_tempi": 80, "transition_lambda": 200},
    {"id": "broad_observation", "beats_per_bar": [2, 3, 4, 6], "observation_lambda": 8},
    {"id": "narrow_observation", "beats_per_bar": [2, 3, 4, 6], "observation_lambda": 32},
    {"id": "sensitive", "beats_per_bar": [2, 3, 4, 6], "threshold": 0.02},
    {"id": "strict", "beats_per_bar": [2, 3, 4, 6], "threshold": 0.10},
    {"id": "wide_default_meter", "beats_per_bar": [2, 3, 4], "min_bpm": 35.0, "max_bpm": 260.0, "num_tempi": 80},
)
SOURCE_IDS = ("model1_control", "three_model_equal", "three_model_robust_global")
FIXED_GLOBAL_CANDIDATE = "three_model_equal__wide_tempo"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _f1(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    used: set[int] = set()
    errors = []
    for value in predicted:
        candidates = [(abs(value - item), index) for index, item in enumerate(reference) if index not in used and abs(value - item) <= TOLERANCE_SEC]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            errors.append(error)
    tp = len(errors)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "predicted": len(predicted), "reference": len(reference), "precision": precision, "recall": recall, "f1": score, "mean_error_sec": statistics.fmean(errors) if errors else None}


def _records(output: np.ndarray) -> list[dict[str, Any]]:
    array = np.asarray(output, dtype=float).reshape(-1, 2)
    return [{"time_sec": float(time_sec), "beat_number": int(round(number)), "downbeat": int(round(number)) == 1} for time_sec, number in array]


def _annotation(path: Path, downbeats: bool = False) -> list[float]:
    records = _load(path)["beat_grid"]["beats"]
    return [float(item["time_sec"]) for item in records if not downbeats or item.get("downbeat")]


def _activation_contrast(activation: np.ndarray) -> float:
    values = np.max(np.asarray(activation, dtype=float), axis=1)
    return max(1e-6, float(np.quantile(values, 0.95) - np.quantile(values, 0.50)))


def _robust_global_weights(case_activations: Mapping[str, Mapping[int, np.ndarray]]) -> dict[int, float]:
    contrasts = {model: statistics.median(_activation_contrast(values[model]) for values in case_activations.values()) for model in (1, 2, 3)}
    median = statistics.median(contrasts.values())
    clipped = {model: min(2.0 * median, max(0.5 * median, value)) for model, value in contrasts.items()}
    total = sum(clipped.values())
    return {model: value / total for model, value in clipped.items()}


def _ensemble(values: Mapping[int, np.ndarray], source: str, robust_weights: Mapping[int, float]) -> np.ndarray:
    shapes = {tuple(value.shape) for value in values.values()}
    if len(shapes) != 1:
        raise ValueError(f"activation shapes differ: {sorted(shapes)}")
    if source == "model1_control":
        return np.asarray(values[1], dtype=np.float32)
    if source == "three_model_equal":
        return np.mean(np.stack([values[index] for index in (1, 2, 3)]), axis=0, dtype=np.float32)
    if source == "three_model_robust_global":
        return sum(np.asarray(values[index], dtype=np.float32) * robust_weights[index] for index in (1, 2, 3))
    raise ValueError(source)


def _candidate_id(source: str, profile: Mapping[str, Any]) -> str:
    return f"{source}__{profile['id']}"


def _decode(activation: np.ndarray, profile: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    from madmom.features import DBNDownBeatTrackingProcessor

    parameters = {key: value for key, value in profile.items() if key != "id"}
    started = time.perf_counter()
    output = DBNDownBeatTrackingProcessor(fps=50, **parameters)(activation)
    return output, time.perf_counter() - started


def _aggregate(candidate_id: str, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [case["candidates"][candidate_id] for case in cases if case["candidates"][candidate_id]["status"] == "ok"]
    return {
        "candidate_id": candidate_id, "case_count": len(values), "failure_count": len(cases) - len(values),
        "mean_beat_f1": statistics.fmean(item["beat_metrics"]["f1"] for item in values) if values else None,
        "mean_downbeat_f1": statistics.fmean(item["downbeat_metrics"]["f1"] for item in values) if values else None,
        "total_decode_seconds": sum(item["decode_seconds"] for item in values),
    }


def _rank(row: Mapping[str, Any]) -> tuple[float, bool]:
    if row["mean_beat_f1"] is None or row["mean_downbeat_f1"] is None:
        return -1.0, False
    return (float(row["mean_beat_f1"]) + float(row["mean_downbeat_f1"])) / 2.0, row["candidate_id"] == FIXED_GLOBAL_CANDIDATE


def run(batch_root: Path, activation_root: Path, output_root: Path) -> dict[str, Any]:
    selection = _load(batch_root / "raw-selection.json")
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    registry = {item["id"]: item for item in evaluator["cases"]}
    case_activations: dict[str, dict[int, np.ndarray]] = {}
    activation_records = {}
    for selected in selection["selected"]:
        case_id = selected["case_id"]
        values = {}
        records = {}
        for model in (1, 2, 3):
            path = activation_root / case_id / f"model-{model}" / "activations.npz"
            with np.load(path) as archive:
                values[model] = np.asarray(archive["activations"], dtype=np.float32)
            records[str(model)] = {"path": str(path), "sha256": _sha256(path), "shape": list(values[model].shape)}
        case_activations[case_id] = values
        activation_records[case_id] = records
    robust_weights = _robust_global_weights(case_activations)
    candidate_ids = [_candidate_id(source, profile) for source in SOURCE_IDS for profile in DBN_PROFILES]
    decoded_cases = []
    raw_root = output_root / "raw"
    for selected in selection["selected"]:
        case_id = selected["case_id"]
        decoded = {}
        errors = {}
        for source in SOURCE_IDS:
            activation = _ensemble(case_activations[case_id], source, robust_weights)
            for profile in DBN_PROFILES:
                candidate_id = _candidate_id(source, profile)
                try:
                    output, elapsed = _decode(activation, profile)
                    records = _records(output)
                    path = raw_root / case_id / f"{candidate_id}.json"
                    _write(path, {"schema_version": "activation_decoder_raw_1", "case_id": case_id, "candidate_id": candidate_id, "activation_source": source, "robust_global_weights": robust_weights if source == "three_model_robust_global" else None, "dbn_profile": profile, "fps": 50, "records": records})
                    decoded[candidate_id] = {"status": "decoded", "source": source, "profile": profile, "decode_seconds": elapsed, "path": str(path), "sha256": _sha256(path)}
                except Exception as exc:
                    errors[candidate_id] = f"{type(exc).__name__}: {exc}"
        decoded_cases.append({"case_id": case_id, "decoded": decoded, "errors": errors})

    # Reference access begins only after all 900 raw candidate outputs have
    # been attempted and saved. This makes reference-free generation auditable
    # across the whole batch, rather than merely within each case.
    cases = []
    for decoded_case in decoded_cases:
        case_id = decoded_case["case_id"]
        decoded = decoded_case["decoded"]
        errors = decoded_case["errors"]
        item = registry[case_id]
        annotation_path = Path(item["beat_annotation"]["path"])
        reference_beats = _annotation(annotation_path)
        reference_downbeats = _annotation(annotation_path, True)
        candidates = {}
        for candidate_id in candidate_ids:
            if candidate_id in errors:
                candidates[candidate_id] = {"status": "failed", "error": errors[candidate_id]}
                continue
            value = decoded[candidate_id]
            records = _load(Path(value["path"]))["records"]
            beats = [record["time_sec"] for record in records]
            downbeats = [record["time_sec"] for record in records if record["downbeat"]]
            value.update({"status": "ok", "beat_metrics": _f1(reference_beats, beats), "downbeat_metrics": _f1(reference_downbeats, downbeats)})
            candidates[candidate_id] = value
        cases.append({"case_id": case_id, "category": item["category"], "activation_inputs": activation_records[case_id], "reference_annotation": {"path": str(annotation_path), "sha256": _sha256(annotation_path)}, "candidates": candidates})

    aggregates = [_aggregate(candidate_id, cases) for candidate_id in candidate_ids]
    categories = sorted({case["category"] for case in cases})
    by_category = {category: [_aggregate(candidate_id, [case for case in cases if case["category"] == category]) for candidate_id in candidate_ids] for category in categories}
    loco = []
    for held_out in categories:
        train = [case for case in cases if case["category"] != held_out]
        test = [case for case in cases if case["category"] == held_out]
        train_rows = [_aggregate(candidate_id, train) for candidate_id in candidate_ids]
        selected_row = max(train_rows, key=_rank)
        loco.append({"held_out_category": held_out, "selected_on_other_categories": selected_row, "held_out_result": _aggregate(selected_row["candidate_id"], test)})
    fixed = next(row for row in aggregates if row["candidate_id"] == FIXED_GLOBAL_CANDIDATE)
    best = max(aggregates, key=_rank)
    loco_counts = {candidate_id: sum(fold["selected_on_other_categories"]["candidate_id"] == candidate_id for fold in loco) for candidate_id in candidate_ids if any(fold["selected_on_other_categories"]["candidate_id"] == candidate_id for fold in loco)}
    report = {
        "schema_version": "beatnet_activation_decoder_pilot_1", "diagnostic_only": True, "runtime_consumed": False,
        "activation_source": {"root": str(activation_root), "tree_sha256": _tree_hash(activation_root), "inference_rerun": False, "activation_count": 90},
        "candidate_policy": {
            "candidate_count": len(candidate_ids), "activation_sources": list(SOURCE_IDS), "dbn_profiles": list(DBN_PROFILES),
            "robust_weight_rule": "Across all 30 cases without references: per model median(p95(max activation)-p50(max activation)); clip to [0.5,2.0] times cross-model median; normalize once globally.",
            "robust_global_weights": {str(key): value for key, value in robust_weights.items()},
            "fixed_global_candidate": FIXED_GLOBAL_CANDIDATE,
            "fixed_reason": "predeclared equal three-model ensemble with one broad 35-260 BPM official madmom DBN and meter 2/3/4/6",
            "stop_rule": "Do not expand the grid when no fixed candidate clearly approaches beat 0.85 and downbeat 0.75 across all categories.",
            "reference_used_for_candidate_selection": False,
        },
        "summary": {
            "case_count": len(cases), "candidate_count": len(candidate_ids), "fixed_global_result": fixed, "best_aggregate_diagnostic_only": best,
            "any_candidate_target_met": any(row["mean_beat_f1"] >= 0.85 and row["mean_downbeat_f1"] >= 0.75 for row in aggregates),
            "loco_selected_candidate_counts": loco_counts, "loco_selection_stable": len(loco_counts) == 1,
            "total_decode_seconds": sum(row["total_decode_seconds"] for row in aggregates),
        },
        "aggregates": aggregates, "by_category": by_category, "leave_one_category_out": loco, "cases": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path, md_path = output_root / "pilot-report.json", output_root / "pilot-report.md"
    _write(report_path, report)
    _write_markdown(report, md_path)
    manifest = {"schema_version": "diagnostic_artifact_manifest_1", "artifacts": [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in (report_path, md_path)], "raw_tree": {"file_count": sum(1 for path in raw_root.rglob("*") if path.is_file()), "sha256": _tree_hash(raw_root)}}
    _write(output_root / "artifact-manifest.json", manifest)
    return report


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    fixed, best = summary["fixed_global_result"], summary["best_aggregate_diagnostic_only"]
    lines = ["# BeatNet activation decoder pilot v1", "", "This diagnostic reused 90 saved activations and did not rerun BeatNet inference. Every raw decode was saved before reference evaluation.", "", "## Candidate policy", "", "```json", json.dumps(report["candidate_policy"], ensure_ascii=False, indent=2), "```", "", "## Result", "", f"Fixed `{fixed['candidate_id']}`: beat/downbeat **{fixed['mean_beat_f1']:.6f}/{fixed['mean_downbeat_f1']:.6f}**, failures {fixed['failure_count']}.", f"Diagnostic aggregate leader `{best['candidate_id']}`: **{best['mean_beat_f1']:.6f}/{best['mean_downbeat_f1']:.6f}**.", f"Any candidate met 0.85/0.75: **{str(summary['any_candidate_target_met']).lower()}**. LOCO stable: **{str(summary['loco_selection_stable']).lower()}**.", "", "## All candidates", "", "| candidate | n | failures | beat | downbeat | decode s |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['candidate_id']} | {row['case_count']} | {row['failure_count']} | {row['mean_beat_f1']:.6f} | {row['mean_downbeat_f1']:.6f} | {row['total_decode_seconds']:.3f} |")
    lines += ["", "## Leave one category out", "", "| held out | selected elsewhere | beat | downbeat |", "|---|---|---:|---:|"]
    for fold in report["leave_one_category_out"]:
        result = fold["held_out_result"]
        lines.append(f"| {fold['held_out_category']} | {result['candidate_id']} | {result['mean_beat_f1']:.6f} | {result['mean_downbeat_f1']:.6f} |")
    lines += ["", "## Per-category leaders (diagnostic only)", "", "| category | candidate | beat | downbeat |", "|---|---|---:|---:|"]
    for category, rows in report["by_category"].items():
        leader = max(rows, key=_rank)
        lines.append(f"| {category} | {leader['candidate_id']} | {leader['mean_beat_f1']:.6f} | {leader['mean_downbeat_f1']:.6f} |")
    lines += ["", "## Decision", "", "The grid stops here if the fixed global result and category robustness remain far from 0.85/0.75. Aggregate or category winners are reference diagnostics and cannot be routed per case in production.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.batch_root.resolve(), args.activation_root.resolve(), args.output_root.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
