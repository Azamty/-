"""Write the exact 30-case production acceptance v3 report.

``high_accuracy_benchmark.py`` evaluates service manifests, while the batch
runner keeps the authoritative case failure state in the parent case
manifest.  This report joins those two explicit records so an absent new
pipeline manifest is reported as a crash rather than silently as an
unevaluated case.  It never discovers artifacts by recursive filename scan.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_ROOT = ROOT / ".artifacts" / "review" / "production-acceptance-v3"


def _load_benchmark_module():
    path = ROOT / "scripts" / "high_accuracy_benchmark.py"
    spec = importlib.util.spec_from_file_location("production_acceptance_v3_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_summary(evaluation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evaluation, Mapping):
        return {"pitch_f1": None, "chord_retention": None, "rhythm_error": None, "beat_f1": None, "downbeat_f1": None}
    metrics = evaluation.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return {
        "pitch_f1": metrics.get("pitch_f1"),
        "chord_retention": metrics.get("chord_retention"),
        "rhythm_error": metrics.get("rhythm_error"),
        "beat_f1": metrics.get("beat_f1"),
        "downbeat_f1": metrics.get("downbeat_f1"),
    }


def _runner_state(root: Path, case_id: str) -> Mapping[str, Any]:
    path = root / case_id / "manifest.json"
    if not path.is_file():
        return {"status": "missing", "error": {"stage": "runner", "message": "case runner manifest not found"}}
    payload = _json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid runner case manifest: {path}")
    return payload


def _selected_registry_cases(
    registry: Mapping[str, Any], selected_ids: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    """Index only the explicitly selected reliable production cases."""

    registry_cases = {
        str(case["id"]): case
        for case in registry.get("cases", [])
        if isinstance(case, Mapping) and "id" in case
    }
    reliable_ids = {
        case_id
        for case_id, case in registry_cases.items()
        if case.get("reference_midi_reliable") is True
    }
    selected_set = set(selected_ids)
    if len(selected_ids) != 30 or len(selected_set) != 30:
        raise ValueError("v3 selection IDs must contain exactly 30 unique cases")
    if selected_set != reliable_ids:
        raise ValueError("v3 selection IDs do not equal the registry reliable production set")
    return {case_id: registry_cases[case_id] for case_id in selected_ids}


def build_acceptance_report(
    registry: Mapping[str, Any],
    *,
    result_root: Path,
    selection_path: Path,
) -> dict[str, Any]:
    benchmark = _load_benchmark_module()
    selection = _json(selection_path)
    selected = selection.get("selected") if isinstance(selection, Mapping) else None
    if not isinstance(selected, list) or len(selected) != 30:
        raise ValueError("v3 selection manifest must contain exactly 30 selected cases")
    selected_ids = [str(item.get("case_id")) for item in selected if isinstance(item, Mapping)]
    if len(selected_ids) != 30 or len(set(selected_ids)) != 30:
        raise ValueError("v3 selection manifest has missing or duplicate case IDs")
    registry_cases = _selected_registry_cases(registry, selected_ids)
    new_evaluations = {
        case_id: benchmark.evaluate_case(registry_cases[case_id], result_root=result_root / "new")
        for case_id in selected_ids
    }
    baseline_evaluations = {
        case_id: benchmark.evaluate_case(registry_cases[case_id], result_root=result_root / "baseline")
        for case_id in selected_ids
    }
    evaluator = benchmark.build_report(
        registry,
        result_root=result_root / "new",
        baseline_root=result_root / "baseline",
    )
    records: list[dict[str, Any]] = []
    for selection_item in selected:
        case_id = str(selection_item["case_id"])
        case = registry_cases[case_id]
        runner = _runner_state(result_root, case_id)
        runner_status = str(runner.get("status") or "missing")
        new_evaluation = new_evaluations[case_id]
        baseline_evaluation = baseline_evaluations[case_id]
        error = runner.get("error") if isinstance(runner.get("error"), Mapping) else None
        records.append(
            {
                "id": case_id,
                "title": case.get("title", case_id),
                "source_batch": selection_item.get("source_batch"),
                "render_domain": case.get("render_domain"),
                "source_kind": case.get("source_kind"),
                "input_sha256": selection_item.get("registry_input_sha256"),
                "raw_recognition_sha256": selection_item.get("raw_recognition_sha256"),
                "raw_source_recognition_sha256": selection_item.get("source_recognition_sha256"),
                "pitched_note_count": selection_item.get("pitched_note_count"),
                "scope": selection_item.get("raw_effective_evaluation_scope"),
                "recognizer_fingerprint": selection_item.get("recognizer_fingerprint"),
                "new": {
                    "status": "success" if runner_status == "success" else "crashed",
                    "runner_status": runner_status,
                    "crash": runner_status != "success",
                    "error": error,
                    "metrics": _metric_summary(new_evaluation if new_evaluation.get("status") == "evaluated" else None),
                    "evaluation_status": new_evaluation.get("status"),
                    "result_midi": new_evaluation.get("result_midi"),
                },
                "baseline": {
                    "status": "success" if baseline_evaluation.get("status") == "evaluated" else "failed",
                    "crash": baseline_evaluation.get("status") != "evaluated",
                    "metrics": _metric_summary(baseline_evaluation if baseline_evaluation.get("status") == "evaluated" else None),
                    "evaluation_status": baseline_evaluation.get("status"),
                    "result_midi": baseline_evaluation.get("result_midi"),
                },
            }
        )
    new_success = sum(item["new"]["status"] == "success" for item in records)
    baseline_success = sum(item["baseline"]["status"] == "success" for item in records)
    gate = evaluator["accuracy_gate"]
    return {
        "schema_version": "production_acceptance_v3_report_1",
        "registry": str(DEFAULT_REGISTRY),
        "result_root": str(result_root.resolve()),
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_schema": selection.get("schema_version"),
        "recognizer_identity": selection.get("recognizer_identity"),
        "recognizer_fingerprint": selection.get("recognizer_fingerprint"),
        "registered_count": 30,
        "selected_count": 30,
        "evaluated_count": new_success,
        "crash_count": 30 - new_success,
        "baseline_evaluated_count": baseline_success,
        "accuracy_claim_ready": bool(gate.get("ready")),
        "accuracy_claim_reason": gate.get("reason"),
        "accuracy_gate": gate,
        "case_status_counts": {
            "new_success": new_success,
            "new_crashed": 30 - new_success,
            "baseline_success": baseline_success,
            "baseline_failed": 30 - baseline_success,
        },
        "cases": records,
        "evaluator_snapshot": {
            "registered_count": evaluator.get("registered_count"),
            "evaluated_count": evaluator.get("evaluated_count"),
            "accuracy_gate_scopes": evaluator.get("accuracy_gate_scopes"),
            "note": "The top-level cases are the exact 30 production gate cases; diagnostic-only registry cases are omitted here.",
        },
    }


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    gate = report.get("accuracy_gate") if isinstance(report.get("accuracy_gate"), Mapping) else {}
    lines = [
        "# Production acceptance v3",
        "",
        "This report covers exactly the 30 reliable production cases from the current registry. It reuses immutable MuScriptor/GAME/BeatNet raw payloads; no PJS, reference-isolation, or oscillator raw is included.",
        "",
        f"- New chain: **{report.get('evaluated_count')}/30 succeeded**, {report.get('crash_count')} crashed.",
        f"- Legacy baseline: **{report.get('baseline_evaluated_count')}/30 succeeded**.",
        f"- Recognizer fingerprint: `{report.get('recognizer_fingerprint')}`",
        f"- Gate: **{'PASS' if report.get('accuracy_claim_ready') else 'FAIL'}** — {report.get('accuracy_claim_reason')}",
        "",
        "The raw selection is 22 cases from `fluidsynth-production-raw-v1`, 3 updated special-context cases from `fluidsynth-special-context-production-raw-v1`, and 5 CCMusic full-track-context cases from `ccmusic-production-context-v1`.",
        "",
        "## Aggregate gate values",
        "",
        "| metric | new | baseline |",
        "|---|---:|---:|",
        f"| shared evaluated production cases | {gate.get('new_reliable_count', 0)}/30 | {gate.get('baseline_reliable_count', 0)}/30 |",
        f"| independent BeatNet cases | {gate.get('beat_cases_with_metrics', 0)}/30 | — |",
        f"| mean beat F1 | {_number(gate.get('new_mean_beat_f1'))} | — |",
        f"| mean downbeat F1 | {_number(gate.get('new_mean_downbeat_f1'))} | — |",
        f"| fixed-total rhythm error (quarter) | {_number(gate.get('new_mean_rhythm_error_quarter'))} | {_number(gate.get('baseline_mean_rhythm_error_quarter'))} |",
        f"| pitch F1 | {_number(gate.get('new_mean_pitch_f1'))} | {_number(gate.get('baseline_mean_pitch_f1'))} |",
        f"| chord retention | {_number(gate.get('new_mean_chord_retention'))} | {_number(gate.get('baseline_mean_chord_retention'))} |",
        "",
        "The formal gate requires 30 shared successful cases, 30 independent beat/downbeat metric cases, beat F1 ≥ 0.85, downbeat F1 ≥ 0.75, at least 20% lower fixed-total rhythm error, pitch F1 no more than 0.01 below baseline, and no chord-retention drop. A failed case remains in the denominator requirement and is not averaged away.",
        "",
        "## Case results",
        "",
        "| case | source | new | baseline | beat F1 | downbeat F1 | pitch F1 | rhythm new / baseline | chord new | failure |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in report.get("cases", []):
        new = case.get("new", {})
        baseline = case.get("baseline", {})
        metrics = new.get("metrics", {})
        baseline_metrics = baseline.get("metrics", {})
        error = new.get("error") or {}
        failure = ""
        if new.get("crash"):
            failure = f"{error.get('stage', 'unknown')}: {str(error.get('message', ''))[:180]}"
        rhythm = metrics.get("rhythm_error") or {}
        base_rhythm = baseline_metrics.get("rhythm_error") or {}
        lines.append(
            "| `{id}` | `{src}` | {new} | {base} | {beat} | {down} | {pitch} | {rhythm} / {base_rhythm} | {chord} | {failure} |".format(
                id=case.get("id"),
                src=case.get("source_batch"),
                new=new.get("status"),
                base=baseline.get("status"),
                beat=_number((metrics.get("beat_f1") or {}).get("f1")),
                down=_number((metrics.get("downbeat_f1") or {}).get("f1")),
                pitch=_number((metrics.get("pitch_f1") or {}).get("f1")),
                rhythm=_number(rhythm.get("mean_fixed_total_assignment_rhythm_error_quarter")),
                base_rhythm=_number(base_rhythm.get("mean_fixed_total_assignment_rhythm_error_quarter")),
                chord=_number((metrics.get("chord_retention") or {}).get("retention")),
                failure=failure.replace("|", "\\|")
            )
        )
    lines.extend(
        [
            "",
            "## Reproducibility and domain limits",
            "",
            "Each case directory contains the exact source raw copy, a source-file hash inventory, input hash, recognizer identity/fingerprint, and runner manifests. The 22 local MIDI-domain cases use the pinned direct FluidSynth/MS Basic renderer documented by the registry; this is a deterministic render domain and does not represent a live acoustic recording. CCMusic cases use BeatNet analysis from the complete original-song mix and crop the five absolute windows; their note events remain the immutable GAME raw payloads.",
            "",
            "The v2 acceptance snapshot remains historical at `docs/high-accuracy-production-acceptance-v2.md`; this v3 report uses the updated special-context raw and the full-track-context CCMusic raw selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    registry = _json(args.manifest.resolve())
    result_root = args.result_root.resolve()
    selection = (args.selection or result_root / "raw-selection.json").resolve()
    report = build_acceptance_report(registry, result_root=result_root, selection_path=selection)
    output = (args.output or result_root / "production-report-v3.json").resolve()
    markdown = (args.markdown or result_root / "production-report-v3.md").resolve()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown)
    print(json.dumps({"output": str(output), "markdown": str(markdown), "selected": report["selected_count"], "new_success": report["evaluated_count"], "new_crashed": report["crash_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
