"""Evaluate locally installed BeatNet 1.1.3 official models and DBN meters.

This script is diagnostic-only and must be run with .venv-model-beatnet.  It
saves model activations and every decoded output before loading references.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beatnet-official-config-pilot-v1"
MODEL_IDS = (1, 2, 3)
METER_CONFIGS = ((2, 3, 4), (2, 3, 4, 6), (2,), (3,), (4,), (6,))
FIXED_GLOBAL_CONFIG = "model-1_meter-2-3-4"
TOLERANCE_SEC = 0.07


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _f1(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    used: set[int] = set()
    errors = []
    for value in predicted:
        matches = [(abs(value - item), index) for index, item in enumerate(reference) if index not in used and abs(value - item) <= TOLERANCE_SEC]
        if matches:
            error, index = min(matches)
            used.add(index)
            errors.append(error)
    true_positive = len(errors)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": true_positive, "predicted": len(predicted), "reference": len(reference), "precision": precision, "recall": recall, "f1": score, "mean_error_sec": statistics.fmean(errors) if errors else None}


def _annotation_times(path: Path, *, downbeats: bool = False) -> list[float]:
    grid = _load(path)["beat_grid"]
    return [float(item["time_sec"]) for item in grid["beats"] if not downbeats or item.get("downbeat")]


def _decoded_records(output: np.ndarray) -> list[dict[str, Any]]:
    array = np.asarray(output, dtype=float).reshape(-1, 2)
    return [{"time_sec": float(time_sec), "beat_number": int(round(beat_number)), "downbeat": int(round(beat_number)) == 1} for time_sec, beat_number in array]


def _config_id(model_id: int, meters: Sequence[int]) -> str:
    return f"model-{model_id}_meter-{'-'.join(str(value) for value in meters)}"


def _installed_capabilities() -> dict[str, Any]:
    import importlib.metadata

    from BeatNet.BeatNet import BeatNet
    from madmom.features import DBNDownBeatTrackingProcessor

    package_root = Path(inspect.getsourcefile(BeatNet)).parent
    source = Path(inspect.getsourcefile(BeatNet))
    model_assets = []
    for model_id in MODEL_IDS:
        path = package_root / "models" / f"model_{model_id}_weights.pt"
        model_assets.append({"model_id": model_id, "path": str(path), "available": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None, "sha256": _sha256(path) if path.is_file() else None})
    return {
        "beatnet_version": importlib.metadata.version("BeatNet"),
        "madmom_version": importlib.metadata.version("madmom"),
        "beatnet_source": str(source),
        "beatnet_source_sha256": _sha256(source),
        "constructor_signature": str(inspect.signature(BeatNet)),
        "supported_model_ids_from_source": list(MODEL_IDS),
        "model_descriptions_from_source_comments": {"1": "GTZAN out trained model", "2": "Ballroom out trained model", "3": "Rock_corpus out trained model"},
        "model_assets": model_assets,
        "offline_contract": {"mode": "offline", "required_inference_model": "DBN", "fps": 50},
        "beatnet_default_dbn": {"beats_per_bar": [2, 3, 4], "fps": 50},
        "madmom_dbn_signature": str(inspect.signature(DBNDownBeatTrackingProcessor)),
        "madmom_dbn_defaults": {"min_bpm": 55.0, "max_bpm": 215.0, "num_tempi": 60, "transition_lambda": 100, "observation_lambda": 16, "threshold": 0.05, "correct": True},
        "activation_export": {"supported": callable(getattr(BeatNet, "activation_extractor_online", None)), "public_method": "activation_extractor_online", "columns": ["beat_probability", "downbeat_probability"], "fps": 50},
        "tested_dbn_meter_configs": [list(value) for value in METER_CONFIGS],
    }


def _run_model(audio_path: Path, model_id: int, case_root: Path) -> dict[str, Any]:
    from BeatNet.BeatNet import BeatNet
    from madmom.features import DBNDownBeatTrackingProcessor

    started = time.perf_counter()
    beatnet = BeatNet(model_id, mode="offline", inference_model="DBN", device="cpu")
    load_seconds = time.perf_counter() - started
    activation_started = time.perf_counter()
    activations = beatnet.activation_extractor_online(str(audio_path))
    activation_seconds = time.perf_counter() - activation_started
    model_root = case_root / f"model-{model_id}"
    model_root.mkdir(parents=True, exist_ok=True)
    activation_path = model_root / "activations.npz"
    np.savez_compressed(activation_path, activations=np.asarray(activations, dtype=np.float32), fps=np.asarray([50], dtype=np.int32))
    decodes = {}
    for meters in METER_CONFIGS:
        decode_started = time.perf_counter()
        decoder = DBNDownBeatTrackingProcessor(beats_per_bar=list(meters), fps=50)
        output = decoder(activations)
        decode_seconds = time.perf_counter() - decode_started
        records = _decoded_records(output)
        config_id = _config_id(model_id, meters)
        output_path = model_root / f"{config_id}.json"
        _json(output_path, {"schema_version": "beatnet_official_raw_output_1", "model_id": model_id, "mode": "offline", "inference": "DBN", "dbn_parameters": {"beats_per_bar": list(meters), "fps": 50, "all_other_parameters": "madmom_0.16.1_defaults"}, "activation_path": str(activation_path), "records": records})
        decodes[config_id] = {
            "meters": list(meters), "records": records,
            "model_load_seconds": load_seconds,
            "activation_seconds": activation_seconds,
            "decode_seconds": decode_seconds,
            "end_to_end_seconds": load_seconds + activation_seconds + decode_seconds,
            "path": str(output_path), "sha256": _sha256(output_path),
        }
    return {"load_seconds": load_seconds, "activation_seconds": activation_seconds, "activation_shape": list(activations.shape), "activation_path": str(activation_path), "activation_sha256": _sha256(activation_path), "decodes": decodes}


def _aggregate(config_id: str, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [item["configs"][config_id] for item in cases if item["configs"][config_id]["status"] == "ok"]
    return {
        "config_id": config_id,
        "model_id": valid[0]["model_id"] if valid else int(config_id.split("_")[0].split("-")[1]),
        "meters": valid[0]["meters"] if valid else [],
        "case_count": len(valid),
        "failure_count": len(cases) - len(valid),
        "mean_beat_f1": statistics.fmean(item["beat_metrics"]["f1"] for item in valid) if valid else None,
        "mean_downbeat_f1": statistics.fmean(item["downbeat_metrics"]["f1"] for item in valid) if valid else None,
        "total_model_load_seconds": sum(item["model_load_seconds"] for item in valid),
        "total_activation_seconds": sum(item["activation_seconds"] for item in valid),
        "total_decode_seconds": sum(item["decode_seconds"] for item in valid),
        "total_end_to_end_seconds": sum(item["end_to_end_seconds"] for item in valid),
    }


def _score(row: Mapping[str, Any]) -> float:
    return (float(row["mean_beat_f1"]) + float(row["mean_downbeat_f1"])) / 2.0 if row["mean_beat_f1"] is not None and row["mean_downbeat_f1"] is not None else -1.0


def run(batch_root: Path, output_root: Path) -> dict[str, Any]:
    capabilities = _installed_capabilities()
    if capabilities["beatnet_version"] != "1.1.3":
        raise RuntimeError(f"expected BeatNet 1.1.3, found {capabilities['beatnet_version']}")
    selection = _load(batch_root / "raw-selection.json")
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    registry = {item["id"]: item for item in evaluator["cases"]}
    output_root.mkdir(parents=True, exist_ok=True)
    _json(output_root / "capabilities.json", capabilities)
    cases = []
    for selected in selection["selected"]:
        case_id = selected["case_id"]
        item = registry[case_id]
        audio_path = Path(item["input"]["path"])
        case_root = output_root / "raw" / case_id
        configs = {}
        model_runs = {}
        for model_id in MODEL_IDS:
            try:
                model_run = _run_model(audio_path, model_id, case_root)
                model_runs[str(model_id)] = {key: value for key, value in model_run.items() if key != "decodes"}
                for config_id, decode in model_run["decodes"].items():
                    configs[config_id] = {"status": "decoded", "model_id": model_id, **decode}
            except Exception as exc:
                model_runs[str(model_id)] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                for meters in METER_CONFIGS:
                    configs[_config_id(model_id, meters)] = {"status": "failed", "model_id": model_id, "meters": list(meters), "error": f"{type(exc).__name__}: {exc}"}
        # Reference access starts only after every raw activation and decode is saved.
        annotation_path = Path(item["beat_annotation"]["path"])
        reference_beats = _annotation_times(annotation_path)
        reference_downbeats = _annotation_times(annotation_path, downbeats=True)
        for config in configs.values():
            if config["status"] != "decoded":
                continue
            records = config.pop("records")
            beats = [record["time_sec"] for record in records]
            downbeats = [record["time_sec"] for record in records if record["downbeat"]]
            config.update({"status": "ok", "beat_metrics": _f1(reference_beats, beats), "downbeat_metrics": _f1(reference_downbeats, downbeats)})
        cases.append({"case_id": case_id, "category": item["category"], "audio": {"path": str(audio_path), "sha256": _sha256(audio_path)}, "reference_annotation": {"path": str(annotation_path), "sha256": _sha256(annotation_path)}, "model_runs": model_runs, "configs": configs})

    config_ids = [_config_id(model_id, meters) for model_id in MODEL_IDS for meters in METER_CONFIGS]
    aggregates = [_aggregate(config_id, cases) for config_id in config_ids]
    categories = sorted({item["category"] for item in cases})
    by_category = {}
    for category in categories:
        subset = [item for item in cases if item["category"] == category]
        by_category[category] = [_aggregate(config_id, subset) for config_id in config_ids]
    loco = []
    for held_out in categories:
        training = [item for item in cases if item["category"] != held_out]
        heldout = [item for item in cases if item["category"] == held_out]
        training_rows = [_aggregate(config_id, training) for config_id in config_ids]
        selected_row = max(training_rows, key=lambda row: (_score(row), row["config_id"] == FIXED_GLOBAL_CONFIG))
        heldout_row = _aggregate(selected_row["config_id"], heldout)
        loco.append({"held_out_category": held_out, "selected_on_other_categories": selected_row, "held_out_result": heldout_row})
    fixed = next(row for row in aggregates if row["config_id"] == FIXED_GLOBAL_CONFIG)
    best_diagnostic = max(aggregates, key=lambda row: (_score(row), row["config_id"] == FIXED_GLOBAL_CONFIG))
    actual_load_seconds = sum(float(model["load_seconds"]) for case in cases for model in case["model_runs"].values() if "load_seconds" in model)
    actual_activation_seconds = sum(float(model["activation_seconds"]) for case in cases for model in case["model_runs"].values() if "activation_seconds" in model)
    actual_decode_seconds = sum(float(config["decode_seconds"]) for case in cases for config in case["configs"].values() if config["status"] == "ok")
    loco_counts = {config_id: sum(item["selected_on_other_categories"]["config_id"] == config_id for item in loco) for config_id in config_ids if any(item["selected_on_other_categories"]["config_id"] == config_id for item in loco)}
    report = {
        "schema_version": "beatnet_official_config_pilot_1",
        "diagnostic_only": True,
        "runtime_consumed": False,
        "capabilities": capabilities,
        "selection_policy": {"fixed_global_config": FIXED_GLOBAL_CONFIG, "reason": "predeclared current official BeatNet default; no case or reference-dependent routing", "best_aggregate_is_diagnostic_only": True},
        "evaluation_input_policy": {"audio_count": 30, "one_decode_batch_per_registered_case_audio": True, "ccmusic_note": "The five CCMusic case WAV windows are decoded independently in this pilot; production-v3 used one full-song decode cropped into five windows."},
        "summary": {
            "case_count": len(cases), "config_count": len(config_ids), "fixed_global_result": fixed, "best_aggregate_diagnostic_only": best_diagnostic,
            "fixed_target_met": fixed["mean_beat_f1"] >= 0.85 and fixed["mean_downbeat_f1"] >= 0.75,
            "any_config_target_met": any(row["mean_beat_f1"] >= 0.85 and row["mean_downbeat_f1"] >= 0.75 for row in aggregates),
            "actual_batch_runtime_seconds": {"model_load": actual_load_seconds, "activation": actual_activation_seconds, "all_dbn_decodes": actual_decode_seconds, "total": actual_load_seconds + actual_activation_seconds + actual_decode_seconds},
            "loco_selected_config_counts": loco_counts,
            "loco_unique_selected_config_count": len(loco_counts),
            "loco_selection_stable": len(loco_counts) == 1,
        },
        "aggregates": aggregates,
        "by_category": by_category,
        "leave_one_category_out": loco,
        "cases": cases,
    }
    report_path, md_path = output_root / "pilot-report.json", output_root / "pilot-report.md"
    _json(report_path, report)
    _write_markdown(report, md_path)
    artifact_paths = [output_root / "capabilities.json", report_path, md_path]
    manifest = {"schema_version": "diagnostic_artifact_manifest_1", "artifacts": [{"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in artifact_paths], "raw_tree": {"file_count": sum(1 for path in (output_root / "raw").rglob("*") if path.is_file()), "sha256": _tree_hash(output_root / "raw")}}
    _json(output_root / "artifact-manifest.json", manifest)
    return report


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((path for path in root.rglob("*") if path.is_file()), key=lambda value: value.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    fixed = summary["fixed_global_result"]
    best = summary["best_aggregate_diagnostic_only"]
    lines = ["# BeatNet 1.1.3 official configuration pilot", "", "All raw activations and official DBN outputs were saved before reference evaluation. The production pipeline was not changed.", "", "The five CCMusic case WAV windows were decoded independently because this pilot evaluates the 30 registered audio inputs. Production-v3 instead decoded their shared full song once and cropped its grid, so the fixed model-1 score is not expected to equal the v3 snapshot exactly.", "", "## Installed capabilities", "", "```json", json.dumps(report["capabilities"], ensure_ascii=False, indent=2), "```", "", "## Result", "", f"The predeclared global configuration `{fixed['config_id']}` scored beat/downbeat F1 **{fixed['mean_beat_f1']:.6f}/{fixed['mean_downbeat_f1']:.6f}** with {fixed['failure_count']} failures.", f"The diagnostic aggregate leader `{best['config_id']}` scored **{best['mean_beat_f1']:.6f}/{best['mean_downbeat_f1']:.6f}**. It was not selected for runtime.", f"Any configuration met 0.85/0.75: **{str(summary['any_config_target_met']).lower()}**.", f"LOCO selected {summary['loco_unique_selected_config_count']} distinct configurations across {len(report['leave_one_category_out'])} folds; stable: **{str(summary['loco_selection_stable']).lower()}**.", f"Actual sequential batch time was {summary['actual_batch_runtime_seconds']['total']:.3f}s: load {summary['actual_batch_runtime_seconds']['model_load']:.3f}s, activation {summary['actual_batch_runtime_seconds']['activation']:.3f}s, and all DBN decodes {summary['actual_batch_runtime_seconds']['all_dbn_decodes']:.3f}s.", "", "End-to-end time for one configuration is model construction/weight loading plus shared neural activation extraction plus that DBN decode. The component totals are retained to make the shared activation cost explicit.", "", "## All configurations", "", "| config | n | failures | beat F1 | downbeat F1 | load s | activation s | decode s | end-to-end s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["aggregates"]:
        lines.append(f"| {row['config_id']} | {row['case_count']} | {row['failure_count']} | {row['mean_beat_f1']:.6f} | {row['mean_downbeat_f1']:.6f} | {row['total_model_load_seconds']:.3f} | {row['total_activation_seconds']:.3f} | {row['total_decode_seconds']:.3f} | {row['total_end_to_end_seconds']:.3f} |")
    lines += ["", "## Leave one category out", "", "| held out | selected on other categories | held-out beat | held-out downbeat |", "|---|---|---:|---:|"]
    for item in report["leave_one_category_out"]:
        result = item["held_out_result"]
        lines.append(f"| {item['held_out_category']} | {result['config_id']} | {result['mean_beat_f1']:.6f} | {result['mean_downbeat_f1']:.6f} |")
    lines += ["", "## Category detail", ""]
    for category, rows in report["by_category"].items():
        lines += [f"### {category}", "", "| config | beat | downbeat |", "|---|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['config_id']} | {row['mean_beat_f1']:.6f} | {row['mean_downbeat_f1']:.6f} |")
        lines.append("")
    lines += ["## Decision", "", "No model or meter configuration is eligible for production unless one fixed global configuration clearly approaches the 0.85 beat and 0.75 downbeat targets across the complete batch and categories.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.batch_root.resolve(), args.output_root.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
