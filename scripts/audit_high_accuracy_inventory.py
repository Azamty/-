"""Audit benchmark inputs, model raw provenance, and beat eligibility.

The audit deliberately treats a reference-isolation payload as a separate
class.  A case is counted as a production execution only when its raw payload
is marked ``model_output=true`` and contains a pitched event.  The registry's
30-case plan spans synthetic/local-render, public MIDI-render, mixed-song, and
specialized domains; the domain is disclosed separately from the production
role and never permits a reference payload to stand in for a model result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_SCAN_ROOTS = (ROOT / ".artifacts" / "review", ROOT / ".cache" / "high-accuracy-benchmarks")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _case_hint(path: Path, payload: Mapping[str, Any]) -> str | None:
    value = payload.get("case_id")
    if isinstance(value, str) and value:
        return value
    for part in reversed(path.parts):
        if part.startswith(("synthetic-", "maestro-", "pjs", "ccmusic-", "special-")):
            return part
    if path.name == "production_raw.json":
        candidate = path.parent.name
    elif len(path.parents) >= 2:
        candidate = path.parents[1].name
    else:
        candidate = ""
    if candidate:
        return candidate
    return None


def _source_audio_matches(payload: Mapping[str, Any], input_path: Path) -> bool:
    provenance = payload.get("provenance")
    source = payload.get("source_audio")
    if source is None and isinstance(provenance, Mapping):
        source = provenance.get("source_audio")
    if not source:
        return False
    try:
        return Path(str(source)).expanduser().resolve() == input_path.resolve()
    except OSError:
        return False


def _pitched_note_count(payload: Mapping[str, Any]) -> int:
    count = 0
    for item in payload.get("notes", []):
        if not isinstance(item, Mapping):
            continue
        channel = item.get("channel")
        try:
            is_channel9 = channel is not None and int(channel) == 9
        except (TypeError, ValueError):
            is_channel9 = False
        if bool(item.get("is_drum")) or is_channel9:
            continue
        count += 1
    return count


def _beat_info(case: Mapping[str, Any]) -> dict[str, Any]:
    annotation = str(case.get("beat_annotation") or "")
    path = (ROOT / annotation).resolve() if annotation and not Path(annotation).is_absolute() else Path(annotation)
    independent = case.get("beat_annotation_independent") is True
    return {
        "eligible": independent and path.is_file(),
        "independent_declared": independent,
        "annotation_exists": path.is_file(),
        "annotation_source": case.get("beat_annotation_source"),
        "annotation": annotation,
    }


def _case_role(case: Mapping[str, Any]) -> str:
    declared = str(case.get("benchmark_role") or "")
    if declared == "production_end_to_end":
        return "production_scope"
    if declared == "diagnostic_only":
        return "diagnostic_only"
    if declared == "manual_only":
        return "manual_only"
    scope = str(case.get("evaluation_scope") or "")
    if scope == "production_end_to_end":
        return "production_scope"
    if "quantizer" in scope:
        return "quantizer_fixture"
    if "diagnostic" in scope:
        return "diagnostic_only"
    return "unclassified"


def _production_gate_selected(case: Mapping[str, Any]) -> bool:
    marker = case.get("production_gate_selected")
    if marker is not None:
        return marker is True
    # Legacy ad-hoc registries had no marker and used production scope as the
    # selection signal.
    return str(case.get("evaluation_scope") or "") == "production_end_to_end"


def _raw_record(path: Path, payload: Mapping[str, Any], *, input_path: Path) -> dict[str, Any]:
    model_output = payload.get("model_output") is True
    reference_output = payload.get("model_output") is False or payload.get("source") == "reference_midi_quantizer_isolation"
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
    beat_grid = payload.get("beat_grid")
    beats = beat_grid.get("beats", []) if isinstance(beat_grid, Mapping) else []
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "model_output": model_output,
        "reference_derived": reference_output,
        "source": payload.get("source"),
        "recognizer_version": payload.get("recognizer_version"),
        "recognizer_mode": provenance.get("recognizer_mode"),
        "recognizer_fingerprint": provenance.get("recognizer_fingerprint"),
        "source_audio": payload.get("source_audio") or provenance.get("source_audio"),
        "source_audio_matches_registry_input": _source_audio_matches(payload, input_path),
        "note_count": len(payload.get("notes", [])) if isinstance(payload.get("notes"), list) else None,
        "pitched_note_count": _pitched_note_count(payload),
        "beat_count": len(beats) if isinstance(beats, list) else None,
        "downbeat_count": sum(1 for item in beats if isinstance(item, Mapping) and item.get("downbeat")),
        "evaluation_scope": provenance.get("evaluation_scope"),
        "beat_source": provenance.get("beat_source"),
        "beat_evidence_sources": (provenance.get("beat_onset_evidence") or {}).get("sources", []),
    }


def _scan_payloads(scan_roots: Sequence[Path], registry_cases: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs = {}
    for case in registry_cases:
        case_id = str(case.get("id"))
        raw_input = str(case.get("input") or "")
        inputs[case_id] = (ROOT / raw_input).resolve() if raw_input and not Path(raw_input).is_absolute() else Path(raw_input).resolve()
    seen: set[Path] = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        for pattern in ("production_raw.json", "recognition.json"):
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                payload = _load_json(path)
                if payload is None:
                    continue
                hint = _case_hint(path, payload)
                matches = [case_id for case_id, input_path in inputs.items() if _source_audio_matches(payload, input_path)]
                case_id = matches[0] if len(matches) == 1 else hint
                if case_id not in inputs:
                    continue
                by_case[case_id].append(_raw_record(path, payload, input_path=inputs[case_id]))
    for values in by_case.values():
        values.sort(key=lambda item: (not item["model_output"], item["path"]))
    return by_case


def _scan_raw_failures(scan_roots: Sequence[Path], registry_cases: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collect recorded raw-stage failures without treating them as raw output."""

    case_ids = {str(case.get("id")) for case in registry_cases}
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(case_id: Any, detail: Mapping[str, Any], path: Path) -> None:
        key = str(case_id)
        if key not in case_ids:
            return
        stage = detail.get("stage")
        message = detail.get("message") or detail.get("error")
        if stage != "raw" and not (isinstance(message, str) and "recognizer" in message.lower()):
            return
        failures[key].append({"path": str(path), "stage": stage or "raw", "message": str(message or "")})

    seen: set[Path] = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = _load_json(path)
            if payload is None:
                continue
            if payload.get("case_id") is not None and isinstance(payload.get("error"), Mapping):
                add(payload.get("case_id"), payload["error"], path)
            cases = payload.get("cases")
            if not isinstance(cases, list):
                continue
            for item in cases:
                if not isinstance(item, Mapping):
                    continue
                case_id = item.get("id", item.get("case_id"))
                failure = item.get("failure")
                if isinstance(failure, Mapping):
                    add(case_id, failure, path)
                raw = item.get("raw")
                if isinstance(raw, Mapping) and raw.get("status") == "failed":
                    add(case_id, {"stage": "raw", "message": "recorded raw status=failed"}, path)
    for case_id, values in failures.items():
        unique: list[dict[str, Any]] = []
        seen_details: set[tuple[str, str]] = set()
        for value in values:
            key = (str(value["stage"]), str(value["message"]))
            if key in seen_details:
                continue
            seen_details.add(key)
            unique.append(value)
        failures[case_id] = unique
    return failures


def build_inventory(registry_path: Path = DEFAULT_REGISTRY, *, scan_roots: Sequence[Path] = DEFAULT_SCAN_ROOTS) -> dict[str, Any]:
    registry = _load_json(registry_path)
    if registry is None or not isinstance(registry.get("cases"), list):
        raise ValueError(f"invalid benchmark registry: {registry_path}")
    cases = [case for case in registry["cases"] if isinstance(case, Mapping)]
    raw_by_case = _scan_payloads(scan_roots, cases)
    failure_by_case = _scan_raw_failures(scan_roots, cases)
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id"))
        raw_input = str(case.get("input") or "")
        input_path = (ROOT / raw_input).resolve() if raw_input and not Path(raw_input).is_absolute() else Path(raw_input).resolve()
        raws = raw_by_case.get(case_id, [])
        model_raws = [item for item in raws if item["model_output"]]
        reference_raws = [item for item in raws if item["reference_derived"]]
        successful_model = [item for item in model_raws if item["pitched_note_count"] > 0 and item["source_audio_matches_registry_input"]]
        beat = _beat_info(case)
        production_gate_selected = _production_gate_selected(case)
        records.append(
            {
                "id": case_id,
                "category": case.get("category"),
                "source_kind": case.get("source_kind"),
                "source_id": case.get("source_id"),
                "case_evaluation_scope": case.get("evaluation_scope"),
                "benchmark_role": case.get("benchmark_role"),
                "production_gate_selected": production_gate_selected,
                "render_domain": case.get("render_domain"),
                "case_role": _case_role(case),
                "input": raw_input,
                "input_exists": input_path.is_file(),
                "input_bytes": input_path.stat().st_size if input_path.is_file() else None,
                "input_sha256": _sha256(input_path) if input_path.is_file() else None,
                "beat": beat,
                "production_model_raw_count": len(model_raws),
                "production_success_count": len(successful_model),
                "production_raws": model_raws,
                "reference_derived_raw_count": len(reference_raws),
                "reference_derived_raws": reference_raws,
                "recorded_raw_failures": failure_by_case.get(case_id, []),
                "pitched_events_in_latest_production_raw": successful_model[0]["pitched_note_count"] if successful_model else None,
                "production_candidate": bool(
                    successful_model
                    and beat["eligible"]
                    and production_gate_selected
                    and _case_role(case) == "production_scope"
                ),
                "quantizer_fixture_model_smoke": bool(successful_model and beat["eligible"] and _case_role(case) == "quantizer_fixture"),
            }
        )
    categories = Counter(str(item["category"]) for item in records)
    production_candidates = [item["id"] for item in records if item["production_candidate"]]
    quantizer_smokes = [item["id"] for item in records if item["quantizer_fixture_model_smoke"]]
    return {
        "schema_version": "1.0",
        "registry": str(registry_path.resolve()),
        "scan_roots": [str(path.resolve()) for path in scan_roots],
        "case_count": len(records),
        "category_counts": dict(sorted(categories.items())),
        "production_candidate_count": len(production_candidates),
        "production_candidate_ids": production_candidates,
        "quantizer_fixture_model_smoke_count": len(quantizer_smokes),
        "quantizer_fixture_model_smoke_ids": quantizer_smokes,
        "production_cases_needed_for_30": max(0, 30 - len(production_candidates)),
        "cases": records,
    }


def _status(record: Mapping[str, Any]) -> str:
    if record.get("production_success_count", 0):
        return "production_model_raw"
    if record.get("production_model_raw_count", 0):
        return "production_raw_without_pitched_events"
    if record.get("reference_derived_raw_count", 0):
        return "reference_derived_only"
    if record.get("recorded_raw_failures"):
        return "production_model_failed"
    return "no_raw"


def render_markdown(inventory: Mapping[str, Any]) -> str:
    role_counts = Counter(str(item["case_role"]) for item in inventory["cases"])
    lines = [
        "# High accuracy benchmark inventory",
        "",
        "This report is generated from the registry and local result roots. `production_model_raw` requires `model_output=true`, a registry input hash/path match, and at least one non-drum event. Reference-isolation payloads remain diagnostic and never satisfy that status.",
        "",
        f"- Registry cases: **{inventory['case_count']}**",
        f"- Strict production-scope cases with independent beat annotation and successful model raw: **{inventory['production_candidate_count']}**",
        f"- Quantizer-fixture model smokes (diagnostic only): **{inventory['quantizer_fixture_model_smoke_count']}**",
        f"- Categories: `{inventory['category_counts']}`",
        f"- Case roles: `{dict(sorted(role_counts.items()))}`",
        f"- Production cases still needed to reach 30: **{inventory['production_cases_needed_for_30']}**",
        "",
        "| Case | Category | Render domain | Role | Input | Beat eligible | Raw status | Pitched events | Effective scopes seen |",
        "|---|---|---|---|---:|---:|---|---:|---|",
    ]
    for item in inventory["cases"]:
        raw_scopes = sorted({str(raw.get("evaluation_scope")) for raw in item["production_raws"] if raw.get("evaluation_scope")})
        lines.append(
            f"| `{item['id']}` | `{item['category']}` | `{item.get('render_domain') or '—'}` | `{item['case_role']}` | {'yes' if item['input_exists'] else 'no'} | "
            f"{'yes' if item['beat']['eligible'] else 'no'} | `{_status(item)}` | "
            f"{item['pitched_events_in_latest_production_raw'] if item['pitched_events_in_latest_production_raw'] is not None else '—'} | "
            f"{', '.join(raw_scopes) if raw_scopes else '—'} |"
        )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            "The registry keeps 40 cases with reliable reference MIDI and marks exactly 30 for the current production gate: 10 deterministic known-MIDI renders, 10 ASAP score/performance-aligned piano clips, 5 CCMusic mixed-song segments, and 5 specialized deterministic segments. A real `model_output=true` MuScriptor/GAME payload with a pitched event and an independent beat annotation is eligible for beat/downbeat scoring. The 10 MAESTRO MIDIs remain valid for pitch and event-time diagnostics, but their fixed transport ticks are not independent musical beat labels, so they are retained as `diagnostic_only` and excluded from the selected gate. A reference-isolation payload, empty model result, failed recognizer, or input mismatch never substitutes for a missing model result.",
            "The local MAESTRO archive/render cache is hash-verified by its selection manifest. Each selected case uses a deterministic source-MIDI transport-tick window with the original member hash, clip hash, and renderer provenance. Its generated 120 BPM tick grid is retained as diagnostic metadata only.",
            "",
            "## CCMusic context audit",
            "",
            "The five CCMusic production raw payloads in the scanned roots were produced from 12-second clip inputs, so their BeatNet calls were clip-context calls. The full aligned Yueding mix is available locally. The compliant reproducible path is one BeatNet call on that complete mix, retaining the absolute output times and hash, then a deterministic crop to each segment's `[audio_start_sec, audio_start_sec + duration_sec]` window with one boundary beat on each side for interpolation and a recorded absolute-to-local offset. The score-derived beat grid remains an evaluation annotation only; it must not be passed as a tempo, phase, or meter override. Existing full-window diagnostic output should be cited separately from the clip raw and must not be silently merged into it.",
            "",
            "## Failure semantics",
            "",
            "A model failure, empty pitched-event result, absent independent beat annotation, or input hash mismatch remains visible in the inventory. It is not converted into a successful case by copying a reference raw payload.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--scan-root", type=Path, action="append", dest="scan_roots")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    roots = tuple(path.resolve() for path in args.scan_roots) if args.scan_roots else DEFAULT_SCAN_ROOTS
    inventory = build_inventory(args.registry.resolve(), scan_roots=roots)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(inventory)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    print(json.dumps({"case_count": inventory["case_count"], "production_candidate_count": inventory["production_candidate_count"], "production_candidate_ids": inventory["production_candidate_ids"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
