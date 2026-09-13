"""Stage the current 30-case production acceptance raw inputs.

The v4 acceptance run changes only the raw source selection: the ten ASAP
cases come from the completed raw-only production run and the other twenty
cases come from the immutable v3 staging root.  This script does not run a
recognizer and never opens a reference MIDI or beat annotation.  It copies
raw model output and its audit files into a fresh result root, then the batch
runner can execute baseline and new score chains against those same raw files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "production-acceptance-v4"
DEFAULT_V3_ROOT = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_ASAP_ROOT = ROOT / ".artifacts" / "review" / "asap-production-raw-v1"

ASAP_IDS = frozenset(f"asap-v11-{index:02d}" for index in range(1, 11))
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


def _resolve(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _pitched_count(notes: Any) -> int:
    if not isinstance(notes, list):
        return 0
    count = 0
    for note in notes:
        if not isinstance(note, Mapping) or note.get("is_drum") is True:
            continue
        try:
            pitch = int(note.get("midi", note.get("pitch")))
        except (TypeError, ValueError):
            continue
        if 0 <= pitch <= 127:
            count += 1
    return count


def _production_gate_selected(case: Mapping[str, Any]) -> bool:
    marker = case.get("production_gate_selected")
    if marker is not None:
        return marker is True
    return case.get("reference_midi_reliable") is True and case.get("evaluation_policy") == "reference_metrics"


def _selected_cases(registry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = [case for case in registry.get("cases", []) if isinstance(case, Mapping) and _production_gate_selected(case)]
    if len(cases) != 30:
        raise ValueError(f"registry production gate must contain exactly 30 cases, found {len(cases)}")
    ids = [str(case.get("id")) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("registry production gate contains duplicate case IDs")
    groups = {
        "synthetic": [case for case in cases if case.get("category") == "synthetic_rendered" and case.get("source_kind") == "synthetic"],
        "asap": [case for case in cases if str(case.get("id")) in ASAP_IDS],
        "vocal": [case for case in cases if case.get("category") == "vocal" and case.get("source_kind") == "vocal"],
        "specialized": [case for case in cases if case.get("category") == "specialized_fixture"],
    }
    expected = {"synthetic": 10, "asap": 10, "vocal": 5, "specialized": 5}
    counts = {name: len(items) for name, items in groups.items()}
    if counts != expected:
        raise ValueError(f"registry production gate composition is {counts}, expected {expected}")
    grouped_ids = [str(case["id"]) for items in groups.values() for case in items]
    if len(set(grouped_ids)) != 30 or set(grouped_ids) != set(ids):
        raise ValueError("registry production gate cases do not belong to exactly one v4 group")
    return cases


def build_selection_plan(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the explicit 10 synthetic/10 ASAP/5 vocal/5 special plan."""

    plan: list[dict[str, Any]] = []
    for case in _selected_cases(registry):
        case_id = str(case["id"])
        if case_id in ASAP_IDS:
            group, source_batch = "asap", "asap-production-raw-v1"
        elif case.get("category") == "vocal" and case.get("source_kind") == "vocal":
            group, source_batch = "vocal", "production-acceptance-v3"
        elif case.get("category") == "specialized_fixture":
            group, source_batch = "specialized", "production-acceptance-v3"
        else:
            group, source_batch = "synthetic", "production-acceptance-v3"
        plan.append({"case_id": case_id, "group": group, "source_batch": source_batch})
    return plan


def _source_root_for_batch(source_batch: str, *, v3_root: Path, asap_root: Path) -> Path:
    if source_batch == "asap-production-raw-v1":
        return asap_root.resolve()
    if source_batch == "production-acceptance-v3":
        return v3_root.resolve()
    raise ValueError(f"unsupported v4 source batch: {source_batch}")


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256(target) != sha256(path):
            raise ValueError(f"staged source artifact changed: {target}")
        if not target.exists():
            shutil.copy2(path, target)


def _copy_immutable(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256(source)
    if destination.is_file():
        if sha256(destination) != source_hash:
            raise ValueError(f"immutable staged artifact mismatch: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return source_hash


def _validate_source_case(
    case: Mapping[str, Any],
    source_case: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    case_id = str(case["id"])
    source_manifest_path = source_case / "manifest.json"
    source_raw_path = source_case / "raw" / "recognition.json"
    source_beat_path = source_case / "raw" / "beat_grid.json"
    for path in (source_manifest_path, source_raw_path, source_beat_path):
        if not path.is_file():
            raise FileNotFoundError(f"{case_id}: missing source artifact {path}")
    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get("status") != "success":
        raise ValueError(f"{case_id}: source runner manifest is not successful")
    raw = _load_json(source_raw_path)
    beat_grid = _load_json(source_beat_path)
    if raw.get("model_output") is not True:
        raise ValueError(f"{case_id}: source raw is not model output")
    if not isinstance(raw.get("notes"), list) or not isinstance(raw.get("beat_grid"), Mapping):
        raise ValueError(f"{case_id}: source raw has no notes[]/beat_grid object")
    if raw.get("beat_grid") != dict(beat_grid):
        raise ValueError(f"{case_id}: source beat_grid.json does not match recognition.json")
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{case_id}: source raw has no provenance")
    if provenance.get("recognizer_mode") != "production" or provenance.get("recognizer_fingerprint") != PRODUCTION_FINGERPRINT:
        raise ValueError(f"{case_id}: source recognizer identity/fingerprint mismatch")
    if provenance.get("effective_evaluation_scope") != "production_end_to_end":
        raise ValueError(f"{case_id}: source raw is not production_end_to_end")
    if provenance.get("reference_is_not_model_output") is True or raw.get("reference_derived") is True:
        raise ValueError(f"{case_id}: source raw is reference-derived")
    identity = provenance.get("recognizer_identity")
    if not isinstance(identity, Mapping) or dict(identity) != PRODUCTION_IDENTITY:
        raise ValueError(f"{case_id}: source recognizer identity details mismatch")
    if _pitched_count(raw.get("notes")) <= 0:
        raise ValueError(f"{case_id}: source raw has no pitched event")

    beatnet = beat_grid.get("beatnet") if isinstance(beat_grid.get("beatnet"), Mapping) else {}
    if (
        beat_grid.get("engine") != "beatnet"
        or beat_grid.get("mode") != "offline"
        or beatnet.get("version") != "1.1.3"
        or beatnet.get("inference") != "DBN"
    ):
        raise ValueError(f"{case_id}: source beat grid is not BeatNet 1.1.3 offline/DBN")

    input_path = _resolve(str(case["input"]), repo_root=repo_root)
    if not input_path.is_file():
        raise FileNotFoundError(f"{case_id}: registry input is unavailable: {input_path}")
    input_hash = sha256(input_path)
    expected_input_hash = case.get("input_sha256")
    if expected_input_hash and input_hash != str(expected_input_hash):
        raise ValueError(f"{case_id}: registry input hash mismatch ({input_hash} != {expected_input_hash})")
    source_audio_value = provenance.get("source_audio")
    source_audio = Path(str(source_audio_value)).expanduser() if source_audio_value else None
    if source_audio is None or source_audio.resolve() != input_path.resolve():
        raise ValueError(f"{case_id}: raw source_audio does not match registry input")

    route = provenance.get("route") if isinstance(provenance.get("route"), Mapping) else {}
    expected_route = ("game", "demucs_vocals") if case.get("source_kind") == "vocal" else ("muscriptor", "original_mix")
    if route.get("engine") != expected_route[0] or route.get("route_input") != expected_route[1]:
        raise ValueError(f"{case_id}: source model route is {route.get('engine')}/{route.get('route_input')}, expected {expected_route[0]}/{expected_route[1]}")

    return {
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_recognition_sha256": sha256(source_raw_path),
        "source_beat_grid_sha256": sha256(source_beat_path),
        "source_files": [
            {
                "relative_path": path.relative_to(source_case).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(item for item in source_case.rglob("*") if item.is_file())
        ],
        "registry_input": str(input_path),
        "registry_input_sha256": input_hash,
        "registry_expected_input_sha256": str(expected_input_hash) if expected_input_hash else None,
        "recognizer_identity": dict(identity),
        "recognizer_fingerprint": PRODUCTION_FINGERPRINT,
        "model_output": True,
        "pitched_note_count": _pitched_count(raw["notes"]),
        "beat_engine": beat_grid.get("engine"),
        "beatnet_version": beatnet.get("version"),
        "beatnet_inference": beatnet.get("inference"),
        "route": {"engine": route.get("engine"), "route_input": route.get("route_input")},
        "effective_evaluation_scope": provenance.get("effective_evaluation_scope"),
    }


def stage_selection(
    registry: Mapping[str, Any],
    *,
    output_root: Path,
    registry_path: Path,
    v3_root: Path = DEFAULT_V3_ROOT,
    asap_root: Path = DEFAULT_ASAP_ROOT,
    repo_root: Path = ROOT,
    allow_existing: bool = False,
) -> dict[str, Any]:
    plan = build_selection_plan(registry)
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(f"refusing to overwrite existing v4 root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    registry_cases = {str(case["id"]): case for case in _selected_cases(registry)}
    selected: list[dict[str, Any]] = []
    for item in plan:
        case_id = item["case_id"]
        case = registry_cases[case_id]
        source_root = _source_root_for_batch(item["source_batch"], v3_root=v3_root, asap_root=asap_root)
        source_case = source_root / case_id
        source_record = _validate_source_case(case, source_case, repo_root=repo_root.resolve())
        destination_case = output_root / case_id
        destination_raw = destination_case / "raw"
        destination_raw.mkdir(parents=True, exist_ok=True)
        staged_recognition = destination_raw / "recognition.json"
        staged_beat = destination_raw / "beat_grid.json"
        _copy_immutable(source_case / "raw" / "recognition.json", staged_recognition)
        _copy_immutable(source_case / "raw" / "beat_grid.json", staged_beat)
        _copy_tree(source_case, destination_raw / "source_artifacts")
        source_record.update(
            {
                "schema_version": "production_acceptance_v4_source_1",
                "case_id": case_id,
                "group": item["group"],
                "source_batch": item["source_batch"],
                "source_case": str(source_case.resolve()),
                "source_manifest": str((source_case / "manifest.json").resolve()),
                "source_recognition": str((source_case / "raw" / "recognition.json").resolve()),
                "source_beat_grid": str((source_case / "raw" / "beat_grid.json").resolve()),
                "staged_recognition_sha256": sha256(staged_recognition),
                "staged_beat_grid_sha256": sha256(staged_beat),
                "exact_raw_immutable": True,
                "reference_midi_or_beat_annotation_used_for_staging": False,
            }
        )
        (destination_raw / "source_provenance.json").write_text(
            json.dumps(source_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging_manifest = {
            "schema_version": "production_acceptance_v4_raw_manifest_1",
            "case_id": case_id,
            "status": "success",
            "stage": "raw-staging",
            "raw_only": True,
            "group": item["group"],
            "source_batch": item["source_batch"],
            "recognizer_mode": "production",
            "recognizer_fingerprint": PRODUCTION_FINGERPRINT,
            "recognizer_identity": dict(PRODUCTION_IDENTITY),
            "raw": {
                "recognition": "raw/recognition.json",
                "recognition_sha256": source_record["staged_recognition_sha256"],
                "beat_grid": "raw/beat_grid.json",
                "beat_grid_sha256": source_record["staged_beat_grid_sha256"],
                "immutable": True,
                "model_output": True,
                "evaluation_scope": "production_end_to_end",
            },
            "pipelines": {},
            "source_provenance": "raw/source_provenance.json",
            "error": None,
        }
        (destination_case / "manifest.json").write_text(
            json.dumps(staging_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        selected.append(source_record)
    selection = {
        "schema_version": "production_acceptance_v4_selection_1",
        "registry": str(registry_path.resolve()),
        "recognizer_identity": dict(PRODUCTION_IDENTITY),
        "recognizer_fingerprint": PRODUCTION_FINGERPRINT,
        "selection_policy": {
            "synthetic_cases": 10,
            "asap_cases": 10,
            "vocal_cases": 5,
            "specialized_cases": 5,
            "v3_reuse_cases": 20,
            "asap_raw_reuse_cases": 10,
            "reference_isolation_forbidden": True,
            "recognizer_rerun": False,
        },
        "selected_count": len(selected),
        "selected": selected,
    }
    (output_root / "raw-selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection


def write_markdown(selection: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Production acceptance v4 raw selection",
        "",
        "This selection stages immutable production model raw payloads only. It does not rerun recognition and does not read reference MIDI or beat annotations.",
        "",
        f"- Cases: **{selection.get('selected_count', 0)}**",
        "- Composition: 10 synthetic, 10 ASAP, 5 vocal, 5 specialized.",
        "- Raw sources: 20 cases reused from production-acceptance-v3; 10 ASAP cases reused from asap-production-raw-v1.",
        f"- Recognizer fingerprint: `{selection.get('recognizer_fingerprint')}`",
        "",
        "| case | group | source batch | route | pitched notes | input SHA-256 | raw SHA-256 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for item in selection.get("selected", []):
        route = item.get("route") or {}
        lines.append(
            f"| `{item.get('case_id')}` | `{item.get('group')}` | `{item.get('source_batch')}` | `{route.get('engine')}/{route.get('route_input')}` | {item.get('pitched_note_count')} | `{item.get('registry_input_sha256')}` | `{item.get('staged_recognition_sha256')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--asap-root", type=Path, default=DEFAULT_ASAP_ROOT)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)
    registry_path = args.manifest.resolve()
    registry = _load_json(registry_path)
    selection = stage_selection(
        registry,
        output_root=args.output_root,
        registry_path=registry_path,
        v3_root=args.v3_root,
        asap_root=args.asap_root,
        allow_existing=args.allow_existing,
    )
    write_markdown(selection, args.output_root.resolve() / "raw-selection.md")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "selected_count": selection["selected_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
