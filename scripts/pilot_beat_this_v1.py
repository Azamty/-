"""Run a small, fixed Beat This! feasibility pilot.

The pilot deliberately uses one predeclared official ``final0`` checkpoint and
one case per production-v3 category.  Raw logits and decoded times are saved
before reference annotations are opened.  This script is diagnostic-only and
does not change the production route.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beat-tracker-comparator-v1" / "beat-this-pilot"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "beat_this-final0.ckpt"
OFFICIAL_REPOSITORY = "https://github.com/CPJKU/beat_this"
OFFICIAL_TAG = "v1.1.0"
OFFICIAL_COMMIT = "ad7974846029835307ba19a3d5cefbf40b243041"
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
TOLERANCE_SEC = 0.07
TARGET_BEAT_F1 = 0.85
TARGET_DOWNBEAT_F1 = 0.75
CLOSE_BEAT_F1 = 0.80
CLOSE_DOWNBEAT_F1 = 0.70


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


def _first_case_per_category(selected: Sequence[Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    by_category: dict[str, str] = {}
    for item in selected:
        case_id = str(item["case_id"])
        category = str(registry[case_id]["category"])
        by_category.setdefault(category, case_id)
    return [by_category[category] for category in sorted(by_category)]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _inventory(checkpoint_path: Path) -> dict[str, Any]:
    import beat_this
    import torch
    import torchaudio

    package_root = Path(beat_this.__file__).parent
    direct_url = package_root.parent / "beat_this-1.1.0.dist-info" / "direct_url.json"
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_prefix": sys.prefix,
        "environment_setup": {
            "install_spec": f"beat-this @ git+{OFFICIAL_REPOSITORY}.git@{OFFICIAL_COMMIT}",
            "dependency_bridge": str(Path(sys.prefix).parent / ".venv-model-muscriptor" / "Lib" / "site-packages"),
            "dependency_bridge_reason": "The dedicated environment owns the official Beat This package; its already validated torch stack is shared through a .pth bridge after a separate torch wheel download hit a TLS error.",
            "main_venv_untouched": True,
        },
        "packages": {
            "beat-this": _package_version("beat-this"),
            "torch": _package_version("torch"),
            "torchaudio": _package_version("torchaudio"),
            "numpy": _package_version("numpy"),
            "einops": _package_version("einops"),
            "rotary-embedding-torch": _package_version("rotary-embedding-torch"),
            "soxr": _package_version("soxr"),
        },
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "tag": OFFICIAL_TAG,
            "commit": OFFICIAL_COMMIT,
            "direct_url_metadata": _load(direct_url) if direct_url.is_file() else None,
            "package_root": str(package_root),
            "package_tree_sha256": _tree_hash(package_root),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "available": checkpoint_path.is_file(),
            "bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else None,
            "sha256": _sha256(checkpoint_path) if checkpoint_path.is_file() else None,
            "official_url": CHECKPOINT_URL,
        },
        "inference_contract": {
            "model": "final0",
            "device": "cpu",
            "dbn": False,
            "frame_rate": 50,
            "network_access_at_inference": False,
            "cuda_available_but_disabled": bool(torch.cuda.is_available()),
            "torchaudio_version": torchaudio.__version__,
        },
    }


def _run_case(model: Any, postprocessor: Any, audio_path: Path, case_root: Path) -> dict[str, Any]:
    import numpy as np
    from beat_this.inference import load_audio

    case_root.mkdir(parents=True, exist_ok=True)
    signal, sample_rate = load_audio(audio_path)
    started = time.perf_counter()
    beat_logits, downbeat_logits = model(signal, sample_rate)
    inference_seconds = time.perf_counter() - started
    logits_path = case_root / "logits.npz"
    np.savez_compressed(
        logits_path,
        beat_logits=beat_logits.detach().cpu().numpy().astype(np.float32),
        downbeat_logits=downbeat_logits.detach().cpu().numpy().astype(np.float32),
        fps=np.asarray([50], dtype=np.int32),
        sample_rate=np.asarray([sample_rate], dtype=np.int32),
    )
    decode_started = time.perf_counter()
    beats, downbeats = postprocessor(beat_logits, downbeat_logits)
    decode_seconds = time.perf_counter() - decode_started
    raw_path = case_root / "final0-minimal.json"
    _write(
        raw_path,
        {
            "schema_version": "beat_this_raw_output_1",
            "model": "final0",
            "postprocessor": "minimal",
            "fps": 50,
            "audio_path": str(audio_path),
            "logits_path": str(logits_path),
            "logits_sha256": _sha256(logits_path),
            "beats": [float(value) for value in beats],
            "downbeats": [float(value) for value in downbeats],
        },
    )
    run_manifest = {
        "schema_version": "beat_this_case_manifest_1",
        "audio_path": str(audio_path),
        "audio_sha256": _sha256(audio_path),
        "logits_path": str(logits_path),
        "logits_sha256": _sha256(logits_path),
        "raw_output_path": str(raw_path),
        "raw_output_sha256": _sha256(raw_path),
        "inference_seconds": inference_seconds,
        "decode_seconds": decode_seconds,
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
    }
    _write(case_root / "run-manifest.json", run_manifest)
    return {
        "audio": {"path": str(audio_path), "sha256": _sha256(audio_path)},
        "raw_output": {"path": str(raw_path), "sha256": _sha256(raw_path)},
        "logits": {"path": str(logits_path), "sha256": _sha256(logits_path)},
        "inference_seconds": inference_seconds,
        "decode_seconds": decode_seconds,
        "end_to_end_seconds": inference_seconds + decode_seconds,
    }


def run(batch_root: Path, output_root: Path, checkpoint_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    inventory = _inventory(checkpoint_path)
    if inventory["packages"]["beat-this"] != "1.1.0":
        raise RuntimeError(f"expected beat-this 1.1.0, found {inventory['packages']['beat-this']}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"official checkpoint is required for the offline pilot: {checkpoint_path}")

    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor

    model_started = time.perf_counter()
    model = Audio2Frames(checkpoint_path=str(checkpoint_path), device="cpu", float16=False)
    model_load_seconds = time.perf_counter() - model_started
    postprocessor = Postprocessor(type="minimal", fps=50)
    inventory["model_load_seconds"] = model_load_seconds
    inventory["preflight"] = {"status": "ok", "message": "Official final0 checkpoint loaded on CPU without network access."}
    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "environment.json", inventory)

    selection = _load(batch_root / "raw-selection.json")
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    registry = {item["id"]: item for item in evaluator["cases"]}
    case_ids = _first_case_per_category(selection["selected"], registry)
    if limit is not None:
        case_ids = case_ids[:limit]

    raw_cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_id in case_ids:
        registry_case = registry[case_id]
        audio_path = Path(registry_case["input"]["path"])
        case_root = output_root / "raw" / case_id
        try:
            raw = _run_case(model, postprocessor, audio_path, case_root)
        except Exception as exc:  # pragma: no cover - depends on native torch runtime
            error = f"{type(exc).__name__}: {exc}"
            failure_path = case_root / "run-failure.json"
            _write(failure_path, {"schema_version": "beat_this_failure_1", "case_id": case_id, "error": error})
            failures.append({"case_id": case_id, "category": registry_case["category"], "error": error, "path": str(failure_path)})
            raw = {"audio": {"path": str(audio_path), "sha256": _sha256(audio_path)}, "raw_output": None, "logits": None, "inference_seconds": 0.0, "decode_seconds": 0.0, "end_to_end_seconds": 0.0}
        raw_cases.append(
            {
                "case_id": case_id,
                "category": registry_case["category"],
                "audio": raw["audio"],
                "raw_output": raw["raw_output"],
                "logits": raw["logits"],
                "inference_seconds": raw["inference_seconds"],
                "decode_seconds": raw["decode_seconds"],
                "end_to_end_seconds": raw["end_to_end_seconds"],
                "reference_annotation_path": str(registry_case["beat_annotation"]["path"]),
            }
        )

    # Reference access begins only after every selected case has raw logits and
    # decoded output on disk (or an explicit failure record).
    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        case_id = raw_case["case_id"]
        registry_case = registry[case_id]
        if raw_case["raw_output"] is None:
            cases.append({"case_id": case_id, "category": raw_case["category"], "status": "failed"})
            continue
        annotation_path = Path(raw_case["reference_annotation_path"])
        reference_beats = _annotation_times(annotation_path)
        reference_downbeats = _annotation_times(annotation_path, downbeats=True)
        output = _load(Path(raw_case["raw_output"]["path"]))
        beat_metrics = _f1(reference_beats, output["beats"])
        downbeat_metrics = _f1(reference_downbeats, output["downbeats"])
        cases.append(
            {
                "case_id": case_id,
                "category": registry_case["category"],
                "status": "ok",
                "audio": raw_case["audio"],
                "reference_annotation": {
                    "path": str(annotation_path),
                    "sha256": _sha256(annotation_path),
                    "beat_count": len(reference_beats),
                    "downbeat_count": len(reference_downbeats),
                },
                "raw_output": raw_case["raw_output"],
                "logits": raw_case["logits"],
                "inference_seconds": raw_case["inference_seconds"],
                "decode_seconds": raw_case["decode_seconds"],
                "end_to_end_seconds": raw_case["end_to_end_seconds"],
                "beat_metrics": beat_metrics,
                "downbeat_metrics": downbeat_metrics,
            }
        )

    valid = [case for case in cases if case["status"] == "ok"]
    mean_beat = statistics.fmean(case["beat_metrics"]["f1"] for case in valid) if valid else 0.0
    mean_downbeat = statistics.fmean(case["downbeat_metrics"]["f1"] for case in valid) if valid else 0.0
    category_metrics: dict[str, Any] = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [case for case in valid if case["category"] == category]
        category_metrics[category] = {
            "case_count": len(subset),
            "failure_count": sum(case["category"] == category and case["status"] != "ok" for case in cases),
            "mean_beat_f1": statistics.fmean(case["beat_metrics"]["f1"] for case in subset) if subset else None,
            "mean_downbeat_f1": statistics.fmean(case["downbeat_metrics"]["f1"] for case in subset) if subset else None,
        }
    close = mean_beat >= CLOSE_BEAT_F1 and mean_downbeat >= CLOSE_DOWNBEAT_F1
    summary = {
        "case_count": len(cases),
        "successful_case_count": len(valid),
        "failure_count": len(cases) - len(valid),
        "selected_case_ids": case_ids,
        "mean_beat_f1": mean_beat,
        "mean_downbeat_f1": mean_downbeat,
        "target": {"beat_f1": TARGET_BEAT_F1, "downbeat_f1": TARGET_DOWNBEAT_F1, "met": mean_beat >= TARGET_BEAT_F1 and mean_downbeat >= TARGET_DOWNBEAT_F1},
        "close_to_target": {"beat_f1": CLOSE_BEAT_F1, "downbeat_f1": CLOSE_DOWNBEAT_F1, "met": close},
        "runtime_seconds": {
            "model_load": model_load_seconds,
            "inference": sum(float(case["inference_seconds"]) for case in raw_cases),
            "decode": sum(float(case["decode_seconds"]) for case in raw_cases),
            "wall_clock": time.perf_counter() - started,
        },
        "decision": "expand_to_full_batch" if close else "stop_after_minimum_pilot",
    }
    report = {
        "schema_version": "beat_this_feasibility_pilot_v1",
        "diagnostic_only": True,
        "runtime_consumed": False,
        "selection_policy": {
            "model": "final0",
            "postprocessor": "minimal",
            "case_rule": "first selected production-v3 case in each sorted category",
            "reference_used_for_case_selection": False,
            "per_case_model_selection": False,
            "raw_generation_completed_before_reference_scoring": True,
        },
        "environment": inventory,
        "summary": summary,
        "category_metrics": category_metrics,
        "failures": failures,
        "cases": cases,
    }
    report_path = output_root / "pilot-report.json"
    markdown_path = output_root / "pilot-report.md"
    _write(report_path, report)
    _write_markdown(report, markdown_path)
    raw_root = output_root / "raw"
    _write(
        output_root / "artifact-manifest.json",
        {
            "schema_version": "diagnostic_artifact_manifest_1",
            "artifacts": [
                {"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in (output_root / "environment.json", report_path, markdown_path)
            ],
            "raw_tree": {"file_count": sum(1 for path in raw_root.rglob("*") if path.is_file()), "sha256": _tree_hash(raw_root)},
        },
    )
    return report


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Beat This! feasibility pilot v1",
        "",
        "This is an isolated, diagnostic-only pilot of the official CPJKU/beat_this final0 checkpoint. One predeclared case per production-v3 category was used. Raw logits and decoded outputs were saved before reference annotations were read.",
        "",
        f"Source: `{report['environment']['official_source']['repository']}` tag `{report['environment']['official_source']['tag']}` commit `{report['environment']['official_source']['commit']}`.",
        f"Checkpoint SHA-256: `{report['environment']['checkpoint']['sha256']}`.",
        "",
        "## Result",
        "",
        "| cases | failures | beat F1 | downbeat F1 | target | close pilot gate | decision |",
        "|---:|---:|---:|---:|---|---|---|",
        f"| {summary['case_count']} | {summary['failure_count']} | {summary['mean_beat_f1']:.6f} | {summary['mean_downbeat_f1']:.6f} | {str(summary['target']['met']).lower()} | {str(summary['close_to_target']['met']).lower()} | `{summary['decision']}` |",
        "",
        "## By category",
        "",
        "| category | cases | failures | beat F1 | downbeat F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in report["category_metrics"].items():
        beat = "n/a" if values["mean_beat_f1"] is None else f"{values['mean_beat_f1']:.6f}"
        downbeat = "n/a" if values["mean_downbeat_f1"] is None else f"{values['mean_downbeat_f1']:.6f}"
        lines.append(f"| {category} | {values['case_count']} | {values['failure_count']} | {beat} | {downbeat} |")
    lines += ["", "## Runtime", "", f"Model load: {summary['runtime_seconds']['model_load']:.3f}s; inference: {summary['runtime_seconds']['inference']:.3f}s; decode: {summary['runtime_seconds']['decode']:.3f}s; wall clock: {summary['runtime_seconds']['wall_clock']:.3f}s.", "", "## Decision", "", f"The minimum pilot {'is close enough to justify a full batch' if summary['close_to_target']['met'] else 'is clearly below the requested target, so the expansion stops here'}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=None, help="Limit sorted category representatives for a smoke run.")
    args = parser.parse_args()
    report = run(args.batch_root.resolve(), args.output_root.resolve(), args.checkpoint_path.resolve(), limit=args.limit)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
