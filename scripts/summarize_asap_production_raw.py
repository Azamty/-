"""Audit and summarize the ten ASAP production raw-only benchmark runs.

This is an offline report writer.  It reads only the runner manifests and
production raw outputs; the ASAP reference MIDI and beat annotations are
never opened.  The registry WAV inputs were created by the pinned local
FluidSynth renderer from official ASAP performance MIDI.  That distinction is
deliberate: the report verifies that recognition used the registry WAV and the
original-mix BeatNet route.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_CASE_IDS = tuple(f"asap-v11-{index:02d}" for index in range(1, 11))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _is_pitched_note(item: Any) -> bool:
    if not isinstance(item, Mapping) or bool(item.get("is_drum", False)):
        return False
    value = item.get("midi", item.get("pitch"))
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _case_report(case: Mapping[str, Any], case_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    case_id = str(case.get("id", case_dir.name))
    errors: list[str] = []
    if case.get("production_gate_selected") is not True:
        errors.append("registry case is not selected for the production gate")
    if case.get("source_kind") != "official-score-performance-aligned":
        errors.append("registry case is not an official ASAP performance-aligned instrumental case")
    manifest_path = case_dir / "manifest.json"
    manifest: Mapping[str, Any] = {}
    if not manifest_path.is_file():
        errors.append("missing manifest.json")
    else:
        try:
            loaded = _read_json(manifest_path)
            if isinstance(loaded, Mapping):
                manifest = loaded
            else:
                errors.append("manifest.json is not an object")
        except (OSError, ValueError) as exc:
            errors.append(f"manifest.json unreadable: {type(exc).__name__}")

    raw_path = case_dir / "raw" / "recognition.json"
    manifest_raw = manifest.get("raw")
    if isinstance(manifest_raw, Mapping) and isinstance(manifest_raw.get("recognition"), str):
        candidate = case_dir / str(manifest_raw["recognition"])
        if candidate.is_file():
            raw_path = candidate
    raw: Mapping[str, Any] = {}
    if not raw_path.is_file():
        errors.append("missing raw recognition.json")
    else:
        try:
            loaded = _read_json(raw_path)
            if isinstance(loaded, Mapping):
                raw = loaded
            else:
                errors.append("raw recognition is not an object")
        except (OSError, ValueError) as exc:
            errors.append(f"raw recognition unreadable: {type(exc).__name__}")

    beat_path = case_dir / "raw" / "beat_grid.json"
    if isinstance(manifest_raw, Mapping) and isinstance(manifest_raw.get("beat_grid"), str):
        candidate = case_dir / str(manifest_raw["beat_grid"])
        if candidate.is_file():
            beat_path = candidate
    beat_grid: Mapping[str, Any] = {}
    if not beat_path.is_file():
        errors.append("missing raw beat_grid.json")
    else:
        try:
            loaded = _read_json(beat_path)
            if isinstance(loaded, Mapping):
                beat_grid = loaded
            else:
                errors.append("beat_grid is not an object")
        except (OSError, ValueError) as exc:
            errors.append(f"beat_grid unreadable: {type(exc).__name__}")

    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), Mapping) else {}
    identity = provenance.get("recognizer_identity") if isinstance(provenance.get("recognizer_identity"), Mapping) else {}
    route = provenance.get("route") if isinstance(provenance.get("route"), Mapping) else {}
    notes = raw.get("notes") if isinstance(raw.get("notes"), list) else []
    beats = beat_grid.get("beats") if isinstance(beat_grid.get("beats"), list) else []
    downbeats = [item for item in beats if isinstance(item, Mapping) and item.get("downbeat") is True]
    beatnet = beat_grid.get("beatnet") if isinstance(beat_grid.get("beatnet"), Mapping) else {}

    input_path_value = str(case.get("input") or "")
    input_path = _resolve_repo_path(input_path_value, repo_root=repo_root) if input_path_value else Path()
    input_exists = input_path.is_file() if input_path_value else False
    actual_input_sha256 = _sha256(input_path) if input_exists else None
    expected_input_sha256 = str(case.get("input_sha256") or "") or None
    input_hash_match = bool(input_exists and expected_input_sha256 and actual_input_sha256 == expected_input_sha256)
    if not input_exists:
        errors.append("input audio is unavailable")
    elif not input_hash_match:
        errors.append("input audio SHA-256 does not match registry")

    source_audio = str(provenance.get("source_audio") or "")
    source_audio_path = Path(source_audio).expanduser() if source_audio else None
    source_audio_match = bool(
        source_audio_path
        and input_exists
        and source_audio_path.resolve() == input_path.resolve()
    )
    if not source_audio_match:
        errors.append("production source_audio does not match registry input")

    model_output = raw.get("model_output") is True
    if not model_output:
        errors.append("raw payload is not marked model_output=true")
    production_route = (
        manifest.get("status") == "success"
        and manifest.get("raw_only") is True
        and manifest.get("recognizer_mode") == "production"
        and provenance.get("recognizer_mode") == "production"
        and identity.get("mode") == "production"
        and identity.get("version") == "1.2"
        and identity.get("implementation") == "MuScriptor+Demucs+GAME+BeatNet"
        and route.get("engine") == "muscriptor"
        and route.get("route_input") == "original_mix"
        and provenance.get("beat_engine") == "beatnet"
        and provenance.get("beat_source") == "original_mix"
        and provenance.get("beat_independent_of_reference") is True
        and beat_grid.get("engine") == "beatnet"
        and beat_grid.get("mode") == "offline"
        and beatnet.get("version") == "1.1.3"
        and beatnet.get("inference") == "DBN"
    )
    if not production_route:
        errors.append("production route provenance is incomplete or mismatched")

    no_score_pipelines = not (case_dir / "baseline").exists() and not (case_dir / "new").exists()
    if not no_score_pipelines:
        errors.append("raw-only case contains baseline/new pipeline output")

    production_log = case_dir / "raw" / "production-recognizer.log"
    worker_log = case_dir / "raw" / "muscriptor" / "worker.log"
    logs_present = production_log.is_file() and worker_log.is_file()
    if not logs_present:
        errors.append("production and MuScriptor logs are incomplete")

    fingerprint = str(provenance.get("recognizer_fingerprint") or manifest.get("recognizer_fingerprint") or "") or None
    pitched_count = sum(1 for item in notes if _is_pitched_note(item))
    if pitched_count == 0:
        errors.append("raw model output contains no pitched events")

    return {
        "case_id": case_id,
        "title": case.get("title"),
        "category": case.get("category"),
        "source_kind": case.get("source_kind"),
        "status": "success" if not errors else "failed",
        "runner_status": manifest.get("status"),
        "raw_only": manifest.get("raw_only") is True,
        "model_output": model_output,
        "input": {
            "path": input_path_value,
            "resolved_path": str(input_path) if input_path_value else None,
            "exists": input_exists,
            "expected_sha256": expected_input_sha256,
            "actual_sha256": actual_input_sha256,
            "sha256_match": input_hash_match,
            "source_audio": source_audio,
            "source_audio_matches_input": source_audio_match,
        },
        "recognizer": {
            "mode": provenance.get("recognizer_mode") or manifest.get("recognizer_mode"),
            "fingerprint": fingerprint,
            "identity": dict(identity),
            "route_input": route.get("route_input"),
            "engine": route.get("engine"),
        },
        "beat_grid": {
            "path": _relative(beat_path, case_dir) if beat_path.is_file() else "raw/beat_grid.json",
            "engine": beat_grid.get("engine"),
            "mode": beat_grid.get("mode"),
            "beatnet_version": beatnet.get("version"),
            "inference": beatnet.get("inference"),
            "beat_count": len(beats),
            "downbeat_count": len(downbeats),
            "time_signature": beat_grid.get("time_signature"),
            "tempo": beat_grid.get("tempo"),
            "source": provenance.get("beat_source"),
        },
        "events": {
            "total": len(notes),
            "pitched": pitched_count,
            "drum": sum(1 for item in notes if isinstance(item, Mapping) and bool(item.get("is_drum", False))),
        },
        "artifacts": {
            "recognition": _relative(raw_path, case_dir) if raw_path.is_file() else "raw/recognition.json",
            "recognition_sha256": _sha256(raw_path) if raw_path.is_file() else None,
            "beat_grid": _relative(beat_path, case_dir) if beat_path.is_file() else "raw/beat_grid.json",
            "beat_grid_sha256": _sha256(beat_path) if beat_path.is_file() else None,
            "log_present": production_log.is_file(),
            "muscriptor_log_present": worker_log.is_file(),
        },
        "reference_annotation_used": False,
        "errors": errors,
    }


def summarize(
    result_root: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    case_ids: Sequence[str] = DEFAULT_CASE_IDS,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    result_root = Path(result_root).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    registry = _read_json(registry_path)
    if not isinstance(registry, Mapping) or not isinstance(registry.get("cases"), list):
        raise ValueError("benchmark registry must contain cases[]")
    cases = {str(item.get("id")): item for item in registry["cases"] if isinstance(item, Mapping) and item.get("id")}
    requested = [str(case_id) for case_id in case_ids]
    missing = [case_id for case_id in requested if case_id not in cases]
    if missing:
        raise ValueError(f"registry is missing requested cases: {', '.join(missing)}")
    reports = [_case_report(cases[case_id], result_root / case_id, repo_root=repo_root) for case_id in requested]
    fingerprints = sorted({str(item["recognizer"]["fingerprint"]) for item in reports if item["recognizer"].get("fingerprint")})
    success_count = sum(item["status"] == "success" for item in reports)
    model_output_count = sum(item["model_output"] for item in reports)
    beat_grid_count = sum(item["beat_grid"]["beat_count"] > 0 for item in reports)
    return {
        "schema_version": "asap-production-raw-summary-1",
        "status": "success" if success_count == len(reports) else "partial" if success_count else "failed",
        "run_kind": "production_raw_only",
        "registry": _relative(registry_path, repo_root),
        "result_root": _relative(result_root, repo_root),
        "case_ids": requested,
        "source_policy": {
            "audio": "local pinned FluidSynth render of official ASAP performance MIDI; recognition input is the registry WAV",
            "instrumental_route": "MuScriptor 1.2 on original mix; BeatNet offline/DBN on original mix",
            "reference_isolation": False,
            "reference_midi_or_beat_annotation_read_by_recognizer": False,
            "score_pipelines_run": False,
        },
        "recognizer": {
            "mode": "production",
            "fingerprints": fingerprints,
            "single_fingerprint_across_cases": len(fingerprints) == 1,
            "identities": [item["recognizer"]["identity"] for item in reports if item["recognizer"].get("identity")],
        },
        "counts": {
            "requested": len(reports),
            "success": success_count,
            "failed": len(reports) - success_count,
            "model_output_true": model_output_count,
            "beat_grid_present": beat_grid_count,
            "pitched_events": sum(item["events"]["pitched"] for item in reports),
        },
        "cases": reports,
    }


def _artifact_records(result_root: Path, *, excluded: Iterable[Path] = ()) -> list[dict[str, Any]]:
    excluded_paths = {path.resolve() for path in excluded}
    records: list[dict[str, Any]] = []
    for path in sorted((candidate for candidate in result_root.rglob("*") if candidate.is_file()), key=lambda p: p.as_posix()):
        if path.resolve() in excluded_paths:
            continue
        records.append(
            {
                "path": path.relative_to(result_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def write_reports(summary: Mapping[str, Any], result_root: str | Path) -> tuple[Path, Path, Path]:
    result_root = Path(result_root).expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.json"
    markdown_path = result_root / "summary.md"
    artifact_manifest_path = result_root / "artifact_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ASAP production raw-only run",
        "",
        f"- Status: **{summary['status']}**",
        f"- Cases: {summary['counts']['success']}/{summary['counts']['requested']} successful",
        f"- Model outputs marked true: {summary['counts']['model_output_true']}/{summary['counts']['requested']}",
        f"- Total pitched events: {summary['counts']['pitched_events']}",
        f"- Recognizer fingerprints: {', '.join(summary['recognizer']['fingerprints']) or 'none'}",
        "",
        "The recognizer consumed the registry WAV inputs, which come from the local pinned FluidSynth render of official ASAP performance MIDI. Reference MIDI and beat annotations were not read by the recognizer; no reference-isolation, baseline, or new score pipeline was run.",
        "",
        "| Case | Status | Pitched events | Beats | Downbeats | Input SHA | Route | Errors |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in summary["cases"]:
        errors = "; ".join(item["errors"]) or "—"
        input_status = "match" if item["input"]["sha256_match"] else "mismatch"
        route = f"{item['recognizer'].get('engine')}/{item['recognizer'].get('route_input')}"
        lines.append(
            f"| {item['case_id']} | {item['status']} | {item['events']['pitched']} | {item['beat_grid']['beat_count']} | {item['beat_grid']['downbeat_count']} | {input_status} | {route} | {errors} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "artifact_manifest_1",
        "run_kind": summary["run_kind"],
        "root": str(result_root),
        "artifacts": _artifact_records(result_root, excluded=(artifact_manifest_path,)),
    }
    artifact_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path, markdown_path, artifact_manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids", help="repeat to override the default ASAP 10-case selection")
    args = parser.parse_args(argv)
    summary = summarize(args.result_root, args.manifest, case_ids=tuple(args.case_ids or DEFAULT_CASE_IDS), repo_root=ROOT)
    paths = write_reports(summary, args.result_root)
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "files": [str(path) for path in paths]}, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
