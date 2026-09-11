"""Stage the exact immutable production raw payloads for acceptance v3.

The acceptance run intentionally has a small, explicit source map.  This
prevents a historical baseline/new directory or a reference-isolation result
from being selected by a recursive search.  The script copies the recognizer
payload exactly, keeps a complete source-file hash inventory beside it, and
fails closed on input, provenance, or case-count drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "production-acceptance-v3"

OLD_LOCAL_RAW_ROOT = ROOT / ".artifacts" / "review" / "fluidsynth-production-raw-v1"
SPECIAL_RAW_ROOT = ROOT / ".artifacts" / "review" / "fluidsynth-special-context-production-raw-v1"
CCMUSIC_RAW_ROOT = ROOT / ".artifacts" / "review" / "ccmusic-production-context-v1"

SPECIAL_CONTEXT_IDS = frozenset(
    {"special-pickup-3-4", "special-triplet", "special-complex-chord"}
)
CCMUSIC_IDS = frozenset(
    {
        "ccmusic-yueding-01",
        "ccmusic-yueding-02",
        "ccmusic-yueding-03",
        "ccmusic-yueding-04",
        "ccmusic-yueding-05",
    }
)
OLD_LOCAL_COUNT = 22
SPECIAL_CONTEXT_COUNT = 3
CCMUSIC_COUNT = 5
TOTAL_COUNT = OLD_LOCAL_COUNT + SPECIAL_CONTEXT_COUNT + CCMUSIC_COUNT
PRODUCTION_FINGERPRINT = "2d529aba92c9c9930945eaec5fe4e0ab46226fa68ce46d2956d6645049039ff3"
PRODUCTION_IDENTITY = {
    "mode": "production",
    "implementation": "MuScriptor+Demucs+GAME+BeatNet",
    "version": "1.2",
    "demucs_model": "htdemucs",
    "beat_route": "original_mix_once",
    "muscriptor_decode": "greedy_float32_deterministic",
    "muscriptor_seed": 20260907,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _production_gate_selected(case: Mapping[str, Any]) -> bool:
    """Use the explicit main-gate marker, with a legacy fallback."""

    marker = case.get("production_gate_selected")
    if marker is not None:
        return marker is True
    return case.get("reference_midi_reliable") is True and case.get("evaluation_policy") == "reference_metrics"


def _reliable_cases(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = registry.get("cases")
    if not isinstance(cases, list):
        raise ValueError("registry has no cases list")
    reliable = [
        case
        for case in cases
        if isinstance(case, Mapping)
        and _production_gate_selected(case)
        and case.get("reference_midi_reliable") is True
        and case.get("evaluation_policy") == "reference_metrics"
    ]
    ids = [str(case.get("id")) for case in reliable]
    if len(ids) != len(set(ids)):
        raise ValueError("registry reliable cases contain duplicate IDs")
    return reliable


def source_batch_for_case(case_id: str) -> tuple[str, Path]:
    """Return the only permitted raw batch for a reliable production case."""

    if case_id in CCMUSIC_IDS:
        return "ccmusic-production-context-v1", CCMUSIC_RAW_ROOT
    if case_id in SPECIAL_CONTEXT_IDS:
        return "fluidsynth-special-context-production-raw-v1", SPECIAL_RAW_ROOT
    return "fluidsynth-production-raw-v1", OLD_LOCAL_RAW_ROOT


def build_selection_plan(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build and validate the exact 22/3/5 case composition without I/O."""

    cases = _reliable_cases(registry)
    plan: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id or case_id.startswith("pjs") or case_id in {"luv-letter"}:
            raise ValueError(f"forbidden case in v3 production selection: {case_id!r}")
        batch, _ = source_batch_for_case(case_id)
        plan.append({"case_id": case_id, "source_batch": batch})
    if len(cases) != TOTAL_COUNT:
        raise ValueError(f"registry reliable production count is {len(cases)}, expected {TOTAL_COUNT}")
    counts = {batch: sum(item["source_batch"] == batch for item in plan) for batch in {
        "fluidsynth-production-raw-v1",
        "fluidsynth-special-context-production-raw-v1",
        "ccmusic-production-context-v1",
    }}
    expected = {
        "fluidsynth-production-raw-v1": OLD_LOCAL_COUNT,
        "fluidsynth-special-context-production-raw-v1": SPECIAL_CONTEXT_COUNT,
        "ccmusic-production-context-v1": CCMUSIC_COUNT,
    }
    if counts != expected:
        raise ValueError(f"v3 source composition is {counts}, expected {expected}")
    return plan


def _pitched_count(notes: Any) -> int:
    if not isinstance(notes, list):
        return 0
    count = 0
    for note in notes:
        if not isinstance(note, Mapping) or note.get("is_drum") is True:
            continue
        value = note.get("midi", note.get("pitch"))
        try:
            if 0 <= int(value) <= 127:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def _raw_files(source_case: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in source_case.rglob("*") if item.is_file()):
        records.append(
            {
                "relative_path": path.relative_to(source_case).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def _copy_source_case(source_case: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(item for item in source_case.rglob("*") if item.is_file()):
        relative = path.relative_to(source_case)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def stage_selection(
    registry: Mapping[str, Any],
    *,
    output_root: Path,
    registry_path: Path,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Copy exactly one immutable raw source per case and write the audit index."""

    plan = build_selection_plan(registry)
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(f"refusing to overwrite existing v3 root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    for item in plan:
        case_id = item["case_id"]
        case = next(case for case in _reliable_cases(registry) if str(case["id"]) == case_id)
        source_batch, source_root = source_batch_for_case(case_id)
        source_case = source_root / case_id
        source_recognition = source_case / "raw" / "recognition.json"
        source_beat = source_case / "raw" / "beat_grid.json"
        source_manifest = source_case / "manifest.json"
        for required in (source_recognition, source_beat, source_manifest):
            if not required.is_file():
                raise FileNotFoundError(f"{case_id}: missing source artifact {required}")
        raw = _load_json(source_recognition)
        provenance = raw.get("provenance")
        if raw.get("model_output") is not True or not isinstance(provenance, Mapping):
            raise ValueError(f"{case_id}: raw is not model output with provenance")
        if provenance.get("recognizer_mode") != "production" or provenance.get("recognizer_fingerprint") != PRODUCTION_FINGERPRINT:
            raise ValueError(f"{case_id}: raw recognizer identity/fingerprint is not production 1.2")
        if provenance.get("effective_evaluation_scope") != "production_end_to_end":
            raise ValueError(f"{case_id}: raw effective scope is not production_end_to_end")
        if _pitched_count(raw.get("notes")) <= 0:
            raise ValueError(f"{case_id}: raw has no pitched note event")
        if case_id not in CCMUSIC_IDS and case.get("renderer_version") != "fluidsynth_direct_ms_basic_v1":
            raise ValueError(f"{case_id}: registry input is not the current direct FluidSynth renderer")
        input_path = (ROOT / str(case["input"])).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"{case_id}: registry input is unavailable: {input_path}")
        input_sha = sha256(input_path)
        expected_input_sha = case.get("input_sha256")
        if expected_input_sha and input_sha != expected_input_sha:
            raise ValueError(f"{case_id}: registry input hash mismatch ({input_sha} != {expected_input_sha})")
        destination_case = output_root / case_id
        destination_raw = destination_case / "raw"
        destination_raw.mkdir(parents=True, exist_ok=True)
        destination_recognition = destination_raw / "recognition.json"
        shutil.copy2(source_recognition, destination_recognition)
        source_artifacts = destination_raw / "source_artifacts"
        _copy_source_case(source_case, source_artifacts)
        raw_files = _raw_files(source_case)
        source_record: dict[str, Any] = {
            "schema_version": "production_acceptance_v3_source_1",
            "case_id": case_id,
            "source_batch": source_batch,
            "source_case": str(source_case),
            "source_recognition": str(source_recognition),
            "source_recognition_sha256": sha256(source_recognition),
            "source_beat_grid": str(source_beat),
            "source_beat_grid_sha256": sha256(source_beat),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256(source_manifest),
            "source_files": raw_files,
            "exact_raw_immutable": True,
            "recognizer_identity": dict(PRODUCTION_IDENTITY),
            "recognizer_fingerprint": PRODUCTION_FINGERPRINT,
            "model_output": True,
            "pitched_note_count": _pitched_count(raw.get("notes")),
            "registry_input": str(input_path),
            "registry_input_sha256": input_sha,
            "registry_expected_input_sha256": expected_input_sha,
            "registry_case_evaluation_scope": case.get("evaluation_scope"),
            "raw_effective_evaluation_scope": provenance.get("effective_evaluation_scope"),
            "raw_recognition_sha256": sha256(destination_recognition),
        }
        (destination_raw / "source_provenance.json").write_text(
            json.dumps(source_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        selected.append(source_record)
    selection = {
        "schema_version": "production_acceptance_v3_selection_1",
        "registry": str(registry_path.resolve()),
        "recognizer_identity": dict(PRODUCTION_IDENTITY),
        "recognizer_fingerprint": PRODUCTION_FINGERPRINT,
        "selection_policy": {
            "old_local_cases": OLD_LOCAL_COUNT,
            "updated_special_context_cases": SPECIAL_CONTEXT_COUNT,
            "ccmusic_full_track_context_cases": CCMUSIC_COUNT,
            "reference_substitution_forbidden": True,
            "pjs_forbidden": True,
            "old_oscillator_forbidden": True,
        },
        "selected_count": len(selected),
        "selected": selected,
    }
    (output_root / "raw-selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection


def write_markdown(selection: Mapping[str, Any], path: Path) -> None:
    selected = selection.get("selected", [])
    lines = [
        "# Production acceptance v3 raw selection",
        "",
        "This manifest stages immutable model-output raw payloads; it never substitutes reference MIDI or reference-isolation output.",
        "",
        f"- Cases: **{selection.get('selected_count', 0)}**",
        f"- Recognizer fingerprint: `{selection.get('recognizer_fingerprint')}`",
        "- Composition: 22 old direct FluidSynth local MIDI cases, 3 updated special-context cases, 5 CCMusic full-track-context cases.",
        "",
        "| case | source batch | pitched notes | input SHA-256 | raw SHA-256 |",
        "|---|---|---:|---|---|",
    ]
    for item in selected:
        lines.append(
            f"| `{item['case_id']}` | `{item['source_batch']}` | {item['pitched_note_count']} | `{item['registry_input_sha256']}` | `{item['raw_recognition_sha256']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-existing", action="store_true", help="only allow an existing empty/identical staging root")
    args = parser.parse_args(argv)
    registry_path = args.manifest.resolve()
    registry = _load_json(registry_path)
    selection = stage_selection(
        registry,
        output_root=args.output_root,
        registry_path=registry_path,
        allow_existing=args.allow_existing,
    )
    write_markdown(selection, args.output_root.resolve() / "raw-selection.md")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "selected_count": selection["selected_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
