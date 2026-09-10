"""Compare an offline madmom downbeat ensemble on the v3 audio batch.

This is a diagnostic-only runner.  The official madmom RNN ensemble and every
DBN decode are written under the requested review artifact directory before
any reference annotation is opened.  No production module imports this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beat-tracker-comparator-v1"
TOLERANCE_SEC = 0.07
TARGET_BEAT_F1 = 0.85
TARGET_DOWNBEAT_F1 = 0.75

# The first profile is the predeclared fixed global choice.  The second profile
# is retained as a diagnostic because the batch contains a 6/8 fixture.  No
# profile is selected per case.
DBN_PROFILES: tuple[dict[str, Any], ...] = (
    {"id": "official_234", "beats_per_bar": [2, 3, 4]},
    {"id": "official_2346", "beats_per_bar": [2, 3, 4, 6]},
)
FIXED_CONFIG_ID = "madmom_official_ensemble__official_234"


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
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
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
    true_positive = len(errors)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(reference) if reference else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": len(predicted),
        "reference": len(reference),
        "precision": precision,
        "recall": recall,
        "f1": score,
        "mean_error_sec": statistics.fmean(errors) if errors else None,
    }


def _annotation_times(path: Path, *, downbeats: bool = False) -> list[float]:
    records = _load(path)["beat_grid"]["beats"]
    return [float(item["time_sec"]) for item in records if not downbeats or bool(item.get("downbeat"))]


def _records(output: Any) -> list[dict[str, Any]]:
    import numpy as np

    array = np.asarray(output, dtype=float).reshape(-1, 2)
    return [
        {
            "time_sec": float(time_sec),
            "beat_number": int(round(beat_number)),
            "downbeat": int(round(beat_number)) == 1,
        }
        for time_sec, beat_number in array
    ]


def _candidate_id(profile: Mapping[str, Any]) -> str:
    return f"madmom_official_ensemble__{profile['id']}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _asset_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "available": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _madmom_capabilities() -> dict[str, Any]:
    """Inventory the installed official ensemble without touching references."""

    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
    from madmom.models import DOWNBEATS_BLSTM

    processor_source = Path(inspect.getsourcefile(RNNDownBeatProcessor) or "")
    assets = [_asset_record(Path(path)) for path in DOWNBEATS_BLSTM]
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "madmom": _package_version("madmom"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "torch": _package_version("torch"),
            "BeatNet": _package_version("BeatNet"),
            "beat-this": _package_version("beat-this"),
        },
        "madmom_source": str(processor_source),
        "madmom_source_sha256": _sha256(processor_source),
        "rnn_processor": {
            "class": "madmom.features.downbeats.RNNDownBeatProcessor",
            "constructor_signature": str(inspect.signature(RNNDownBeatProcessor)),
            "official_ensemble_assets": assets,
            "asset_count": len(assets),
            "all_assets_available": all(item["available"] for item in assets),
            "activation_fps": 100,
            "activation_columns": ["beat_probability", "downbeat_probability"],
        },
        "dbn_processor": {
            "class": "madmom.features.downbeats.DBNDownBeatTrackingProcessor",
            "constructor_signature": str(inspect.signature(DBNDownBeatTrackingProcessor)),
            "fps": 100,
            "defaults": {
                "min_bpm": 55.0,
                "max_bpm": 215.0,
                "num_tempi": 60,
                "transition_lambda": 100,
                "observation_lambda": 16,
                "threshold": 0.05,
                "correct": True,
            },
        },
        "offline_contract": {
            "network_access_required_at_run_time": False,
            "model_download_required_at_run_time": False,
            "device": "cpu",
        },
        "beat_this_inventory": {
            "module_available": importlib.util.find_spec("beat_this") is not None,
            "package_version": _package_version("beat-this"),
            "evaluated": False,
        },
    }


def _aggregate(candidate_id: str, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [case["candidates"][candidate_id] for case in cases if case["candidates"][candidate_id]["status"] == "ok"]
    return {
        "candidate_id": candidate_id,
        "case_count": len(values),
        "failure_count": len(cases) - len(values),
        "mean_beat_f1": statistics.fmean(item["beat_metrics"]["f1"] for item in values) if values else None,
        "mean_downbeat_f1": statistics.fmean(item["downbeat_metrics"]["f1"] for item in values) if values else None,
        "total_activation_seconds": sum(float(item["activation_seconds"]) for item in values),
        "total_decode_seconds": sum(float(item["decode_seconds"]) for item in values),
        "total_end_to_end_seconds": sum(float(item["end_to_end_seconds"]) for item in values),
    }


def _rank(row: Mapping[str, Any]) -> tuple[float, bool]:
    beat = row.get("mean_beat_f1")
    downbeat = row.get("mean_downbeat_f1")
    if beat is None or downbeat is None:
        return -1.0, False
    return (float(beat) + float(downbeat)) / 2.0, row["candidate_id"] == FIXED_CONFIG_ID


def _target_result(row: Mapping[str, Any]) -> dict[str, Any]:
    beat = float(row["mean_beat_f1"]) if row["mean_beat_f1"] is not None else 0.0
    downbeat = float(row["mean_downbeat_f1"]) if row["mean_downbeat_f1"] is not None else 0.0
    return {
        "beat_f1": beat,
        "downbeat_f1": downbeat,
        "beat_target": TARGET_BEAT_F1,
        "downbeat_target": TARGET_DOWNBEAT_F1,
        "beat_gap": beat - TARGET_BEAT_F1,
        "downbeat_gap": downbeat - TARGET_DOWNBEAT_F1,
        "target_met": beat >= TARGET_BEAT_F1 and downbeat >= TARGET_DOWNBEAT_F1,
    }


def _run_case(processor: Any, audio_path: Path, case_root: Path) -> dict[str, Any]:
    """Save activations and all raw decodes; do not open a reference here."""

    import numpy as np

    case_root.mkdir(parents=True, exist_ok=True)
    activation_path = case_root / "activations.npz"
    activation_started = time.perf_counter()
    activation = np.asarray(processor(str(audio_path)), dtype=np.float32)
    activation_seconds = time.perf_counter() - activation_started
    np.savez_compressed(activation_path, activations=activation, fps=np.asarray([100], dtype=np.int32))

    candidates: dict[str, dict[str, Any]] = {}
    for profile in DBN_PROFILES:
        candidate_id = _candidate_id(profile)
        decode_started = time.perf_counter()
        try:
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor

            output = DBNDownBeatTrackingProcessor(fps=100, **{key: value for key, value in profile.items() if key != "id"})(activation)
            records = _records(output)
            decode_seconds = time.perf_counter() - decode_started
            raw_path = case_root / f"{candidate_id}.json"
            _write(
                raw_path,
                {
                    "schema_version": "beat_tracker_comparator_raw_1",
                    "case_id": case_root.name,
                    "candidate_id": candidate_id,
                    "model": "madmom_official_rnn_downbeat_ensemble",
                    "processor": "RNNDownBeatProcessor",
                    "dbn_profile": profile,
                    "fps": 100,
                    "activation_path": str(activation_path),
                    "activation_sha256": _sha256(activation_path),
                    "records": records,
                },
            )
            candidates[candidate_id] = {
                "status": "decoded",
                "profile": profile,
                "activation_seconds": activation_seconds,
                "decode_seconds": decode_seconds,
                "end_to_end_seconds": activation_seconds + decode_seconds,
                "path": str(raw_path),
                "sha256": _sha256(raw_path),
                "record_count": len(records),
            }
        except Exception as exc:  # pragma: no cover - depends on native madmom runtime
            error = f"{type(exc).__name__}: {exc}"
            failure_path = case_root / f"{candidate_id}.failure.json"
            _write(failure_path, {"schema_version": "beat_tracker_comparator_failure_1", "candidate_id": candidate_id, "error": error})
            candidates[candidate_id] = {
                "status": "failed",
                "profile": profile,
                "activation_seconds": activation_seconds,
                "decode_seconds": time.perf_counter() - decode_started,
                "end_to_end_seconds": activation_seconds + time.perf_counter() - decode_started,
                "path": str(failure_path),
                "sha256": _sha256(failure_path),
                "error": error,
            }

    _write(
        case_root / "run-manifest.json",
        {
            "schema_version": "beat_tracker_comparator_case_manifest_1",
            "audio_path": str(audio_path),
            "audio_sha256": _sha256(audio_path),
            "activation_path": str(activation_path),
            "activation_sha256": _sha256(activation_path),
            "activation_shape": list(activation.shape),
            "activation_seconds": activation_seconds,
            "candidates": candidates,
        },
    )
    return {
        "audio": {"path": str(audio_path), "sha256": _sha256(audio_path)},
        "activation": {
            "path": str(activation_path),
            "sha256": _sha256(activation_path),
            "shape": list(activation.shape),
            "seconds": activation_seconds,
        },
        "candidates": candidates,
    }


def _evaluate_cases(raw_cases: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read references only after the complete raw generation phase."""

    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        case_id = raw_case["case_id"]
        annotation_path = Path(raw_case["reference_annotation_path"])
        reference_beats = _annotation_times(annotation_path)
        reference_downbeats = _annotation_times(annotation_path, downbeats=True)
        candidates: dict[str, dict[str, Any]] = {}
        for candidate_id, candidate in raw_case["candidates"].items():
            if candidate["status"] != "decoded":
                candidates[candidate_id] = dict(candidate)
                continue
            records = _load(Path(candidate["path"]))["records"]
            beats = [float(item["time_sec"]) for item in records]
            downbeats = [float(item["time_sec"]) for item in records if bool(item.get("downbeat"))]
            scored = dict(candidate)
            scored.update(
                {
                    "status": "ok",
                    "beat_metrics": _f1(reference_beats, beats),
                    "downbeat_metrics": _f1(reference_downbeats, downbeats),
                }
            )
            candidates[candidate_id] = scored
        registry_case = registry[case_id]
        cases.append(
            {
                "case_id": case_id,
                "category": registry_case["category"],
                "audio": raw_case["audio"],
                "reference_annotation": {
                    "path": str(annotation_path),
                    "sha256": _sha256(annotation_path),
                    "beat_count": len(reference_beats),
                    "downbeat_count": len(reference_downbeats),
                },
                "candidates": candidates,
            }
        )
    return cases


def run(batch_root: Path, output_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    capabilities = _madmom_capabilities()
    from madmom.features.downbeats import RNNDownBeatProcessor

    processor_started = time.perf_counter()
    processor = RNNDownBeatProcessor()
    processor_load_seconds = time.perf_counter() - processor_started
    capabilities["preflight"] = {
        "status": "ok",
        "processor_construction_seconds": processor_load_seconds,
        "message": "RNNDownBeatProcessor constructed with local official weights; no network access was needed.",
    }

    selection = _load(batch_root / "raw-selection.json")
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    registry = {item["id"]: item for item in evaluator["cases"]}
    selected = selection["selected"][:limit] if limit is not None else selection["selected"]
    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "capabilities.json", capabilities)

    raw_cases: list[dict[str, Any]] = []
    generation_failures: list[dict[str, Any]] = []
    for selected_case in selected:
        case_id = selected_case["case_id"]
        registry_case = registry[case_id]
        audio_path = Path(registry_case["input"]["path"])
        case_root = output_root / "raw" / case_id
        try:
            raw = _run_case(processor, audio_path, case_root)
        except Exception as exc:  # pragma: no cover - depends on native madmom runtime
            error = f"{type(exc).__name__}: {exc}"
            failure_path = case_root / "run-failure.json"
            _write(failure_path, {"schema_version": "beat_tracker_comparator_failure_1", "case_id": case_id, "error": error})
            generation_failures.append({"case_id": case_id, "error": error, "path": str(failure_path)})
            raw = {
                "audio": {"path": str(audio_path), "sha256": _sha256(audio_path)},
                "activation": None,
                "candidates": {
                    _candidate_id(profile): {
                        "status": "failed",
                        "profile": profile,
                        "activation_seconds": 0.0,
                        "decode_seconds": 0.0,
                        "end_to_end_seconds": 0.0,
                        "path": str(failure_path),
                        "sha256": _sha256(failure_path),
                        "error": error,
                    }
                    for profile in DBN_PROFILES
                },
            }
        raw_cases.append(
            {
                "case_id": case_id,
                "category": registry_case["category"],
                "audio": raw["audio"],
                "activation": raw["activation"],
                "candidates": raw["candidates"],
                "reference_annotation_path": str(registry_case["beat_annotation"]["path"]),
            }
        )

    # This is the phase boundary required by the diagnostic contract: only now
    # do we read annotations and calculate reference metrics.
    cases = _evaluate_cases(raw_cases, registry)
    candidate_ids = [_candidate_id(profile) for profile in DBN_PROFILES]
    aggregates = [_aggregate(candidate_id, cases) for candidate_id in candidate_ids]
    categories = sorted({case["category"] for case in cases})
    by_category = {
        category: [_aggregate(candidate_id, [case for case in cases if case["category"] == category]) for candidate_id in candidate_ids]
        for category in categories
    }
    failures = [
        {"case_id": case["case_id"], "category": case["category"], "candidate_id": candidate_id, "error": candidate["error"], "path": candidate.get("path")}
        for case in cases
        for candidate_id, candidate in case["candidates"].items()
        if candidate["status"] == "failed"
    ]
    fixed = next(row for row in aggregates if row["candidate_id"] == FIXED_CONFIG_ID)
    best = max(aggregates, key=_rank)
    baseline_cases = [registry[item["case_id"]] for item in selected]
    baseline_beat = statistics.fmean(float(item["metrics"]["beat_f1"]["f1"]) for item in baseline_cases) if baseline_cases else None
    baseline_downbeat = statistics.fmean(float(item["metrics"]["downbeat_f1"]["f1"]) for item in baseline_cases) if baseline_cases else None
    summary = {
        "case_count": len(cases),
        "candidate_count": len(candidate_ids),
        "fixed_global_result": fixed,
        "fixed_global_target": _target_result(fixed),
        "best_aggregate_diagnostic_only": best,
        "best_aggregate_target": _target_result(best),
        "any_candidate_target_met": any(_target_result(row)["target_met"] for row in aggregates),
        "baseline_v3_evaluator_mean": {"beat_f1": baseline_beat, "downbeat_f1": baseline_downbeat},
        "runtime_seconds": {
            "processor_load": processor_load_seconds,
            "activation": sum(float(case["activation"]["seconds"]) for case in raw_cases if case["activation"]),
            "dbn_decode": sum(float(candidate["decode_seconds"]) for case in cases for candidate in case["candidates"].values()),
            "wall_clock": time.perf_counter() - started,
        },
        "generation_failure_count": len(generation_failures),
        "decode_failure_count": len(failures),
    }
    report = {
        "schema_version": "beat_tracker_comparator_v1",
        "diagnostic_only": True,
        "runtime_consumed": False,
        "capabilities": capabilities,
        "selection_policy": {
            "fixed_global_candidate": FIXED_CONFIG_ID,
            "fixed_reason": "predeclared official madmom RNN ensemble with the documented 2/3/4 meter family",
            "diagnostic_candidate": "madmom_official_ensemble__official_2346",
            "reference_used_for_candidate_selection": False,
            "per_case_reference_selection": False,
            "raw_generation_completed_before_reference_scoring": True,
        },
        "evaluation_input_policy": {
            "batch_root": str(batch_root),
            "selected_case_count": len(selected),
            "same_production_acceptance_v3_wavs": True,
            "fps": 100,
            "tolerance_sec": TOLERANCE_SEC,
        },
        "beat_this_follow_up": {
            "module_available": capabilities["beat_this_inventory"]["module_available"],
            "evaluated": False,
            "isolated_environment_created": False,
            "decision": "stopped_after_madmom_pilot",
            "reason": "madmom was locally runnable; this minimum official-ensemble pilot is retained as the gating result before any Beat This installation.",
        },
        "summary": summary,
        "aggregates": aggregates,
        "by_category": by_category,
        "failures": generation_failures + failures,
        "cases": cases,
    }
    report_path = output_root / "comparator-report.json"
    markdown_path = output_root / "comparator-report.md"
    _write(report_path, report)
    _write_markdown(report, markdown_path)
    raw_root = output_root / "raw"
    manifest = {
        "schema_version": "diagnostic_artifact_manifest_1",
        "artifacts": [
            {"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (output_root / "capabilities.json", report_path, markdown_path)
        ],
        "raw_tree": {"file_count": sum(1 for path in raw_root.rglob("*") if path.is_file()), "sha256": _tree_hash(raw_root)},
    }
    _write(output_root / "artifact-manifest.json", manifest)
    return report


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    fixed = summary["fixed_global_result"]
    best = summary["best_aggregate_diagnostic_only"]
    fixed_target = summary["fixed_global_target"]
    best_target = summary["best_aggregate_target"]
    lines = [
        "# Offline beat tracker comparator v1",
        "",
        "This diagnostic ran the locally installed official madmom RNN downbeat ensemble on the same production-acceptance-v3 WAV selection. Activations and DBN outputs were saved before reference annotations were read. Production routing was not changed.",
        "",
        "## Capability and policy",
        "",
        f"The preflight status is `{report['capabilities']['preflight']['status']}` with {report['capabilities']['rnn_processor']['asset_count']} official BLSTM assets. Runtime network/model download was not required.",
        f"The fixed global candidate is `{fixed['candidate_id']}`. The `{best['candidate_id']}` row is an aggregate diagnostic only and was not selected per case.",
        f"Beat This! evaluated: **{str(report['beat_this_follow_up']['evaluated']).lower()}**; decision: `{report['beat_this_follow_up']['decision']}`.",
        "",
        "## Aggregate result",
        "",
        "| candidate | cases | failures | beat F1 | downbeat F1 | beat target | downbeat target |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {fixed['candidate_id']} (fixed) | {fixed['case_count']} | {fixed['failure_count']} | {fixed_target['beat_f1']:.6f} | {fixed_target['downbeat_f1']:.6f} | {str(fixed_target['beat_f1'] >= TARGET_BEAT_F1).lower()} | {str(fixed_target['downbeat_f1'] >= TARGET_DOWNBEAT_F1).lower()} |",
        f"| {best['candidate_id']} (diagnostic) | {best['case_count']} | {best['failure_count']} | {best_target['beat_f1']:.6f} | {best_target['downbeat_f1']:.6f} | {str(best_target['beat_f1'] >= TARGET_BEAT_F1).lower()} | {str(best_target['downbeat_f1'] >= TARGET_DOWNBEAT_F1).lower()} |",
        "",
        f"The fixed candidate gap to the 0.85/0.75 target is {fixed_target['beat_gap']:+.6f}/{fixed_target['downbeat_gap']:+.6f}. The v3 evaluator snapshot mean is {_format_metric(summary['baseline_v3_evaluator_mean']['beat_f1'])}/{_format_metric(summary['baseline_v3_evaluator_mean']['downbeat_f1'])}.",
        "",
        "## By category",
        "",
        "| category | candidate | cases | failures | beat F1 | downbeat F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for category, rows in report["by_category"].items():
        for row in rows:
            lines.append(f"| {category} | {row['candidate_id']} | {row['case_count']} | {row['failure_count']} | {_format_metric(row['mean_beat_f1'])} | {_format_metric(row['mean_downbeat_f1'])} |")
    lines += ["", "## Runtime", "", f"Processor construction: {summary['runtime_seconds']['processor_load']:.3f}s; activation: {summary['runtime_seconds']['activation']:.3f}s; DBN decode: {summary['runtime_seconds']['dbn_decode']:.3f}s; wall clock: {summary['runtime_seconds']['wall_clock']:.3f}s.", "", "## Failures", ""]
    if report["failures"]:
        lines += ["| case | candidate | error |", "|---|---|---|"]
        for failure in report["failures"]:
            lines.append(f"| {failure['case_id']} | {failure.get('candidate_id', 'case-run')} | {failure['error']} |")
    else:
        lines.append("No generation or decode failures.")
    lines += ["", "## Decision", "", f"Fixed target met: **{str(fixed_target['target_met']).lower()}**. Any fixed global candidate target met: **{str(summary['any_candidate_target_met']).lower()}**.", "", "The pilot stops at the official madmom ensemble because the result is retained as the minimum comparator gate before considering an isolated Beat This environment. No production route consumes these artifacts.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected cases for a preflight pilot.")
    args = parser.parse_args()
    report = run(args.batch_root.resolve(), args.output_root.resolve(), limit=args.limit)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
