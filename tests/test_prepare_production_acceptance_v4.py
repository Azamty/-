from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import prepare_production_acceptance_v4 as prepare


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_registry(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    v3_root = repo_root / "v3"
    asap_root = repo_root / "asap"
    output_root = repo_root / "out"
    cases: list[dict[str, object]] = []
    ids = [f"synthetic-{index:02d}" for index in range(10)]
    ids.extend(f"asap-v11-{index:02d}" for index in range(1, 11))
    ids.extend(f"vocal-{index:02d}" for index in range(5))
    ids.extend(f"special-{index:02d}" for index in range(5))
    for case_id in ids:
        if case_id.startswith("asap-"):
            category, source_kind, source_root = "official_piano_aligned", "official-score-performance-aligned", asap_root
        elif case_id.startswith("vocal-"):
            category, source_kind, source_root = "vocal", "vocal", v3_root
        elif case_id.startswith("special-"):
            category, source_kind, source_root = "specialized_fixture", "synthetic", v3_root
        else:
            category, source_kind, source_root = "synthetic_rendered", "synthetic", v3_root
        audio = repo_root / "inputs" / f"{case_id}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(case_id.encode("utf-8"))
        audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        route = {"engine": "game", "route_input": "demucs_vocals"} if source_kind == "vocal" else {"engine": "muscriptor", "route_input": "original_mix"}
        raw = {
            "model_output": True,
            "source_kind": "vocal" if source_kind == "vocal" else "instrumental",
            "notes": [{"start_sec": 0.0, "end_sec": 0.5, "midi": 60, "is_drum": False}],
            "beat_grid": {
                "engine": "beatnet",
                "mode": "offline",
                "beatnet": {"version": "1.1.3", "inference": "DBN"},
                "beats": [{"time_sec": 0.0, "downbeat": True}],
            },
            "provenance": {
                "recognizer_mode": "production",
                "recognizer_fingerprint": prepare.PRODUCTION_FINGERPRINT,
                "recognizer_identity": prepare.PRODUCTION_IDENTITY,
                "effective_evaluation_scope": "production_end_to_end",
                "source_audio": str(audio.resolve()),
                "beat_engine": "beatnet",
                "beat_source": "original_mix",
                "beat_independent_of_reference": True,
                "route": route,
            },
        }
        source_case = source_root / case_id
        _write_json(source_case / "manifest.json", {"status": "success"})
        _write_json(source_case / "raw" / "recognition.json", raw)
        _write_json(source_case / "raw" / "beat_grid.json", raw["beat_grid"])
        cases.append(
            {
                "id": case_id,
                "category": category,
                "source_kind": source_kind,
                "input": str(audio.relative_to(repo_root)),
                "input_sha256": audio_hash,
                "production_gate_selected": True,
            }
        )
    return {"cases": cases}, repo_root / "registry.json", v3_root, asap_root, output_root


def test_v4_selection_plan_has_required_composition() -> None:
    registry = json.loads((prepare.ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    plan = prepare.build_selection_plan(registry)

    assert len(plan) == 30
    assert Counter(item["group"] for item in plan) == {"synthetic": 10, "asap": 10, "vocal": 5, "specialized": 5}
    assert Counter(item["source_batch"] for item in plan) == {"production-acceptance-v3": 20, "asap-production-raw-v1": 10}


def test_v4_staging_writes_per_case_raw_manifest_and_keeps_hashes(tmp_path: Path) -> None:
    registry, registry_path, v3_root, asap_root, output_root = _fixture_registry(tmp_path)
    _write_json(registry_path, registry)

    selection = prepare.stage_selection(
        registry,
        output_root=output_root,
        registry_path=registry_path,
        v3_root=v3_root,
        asap_root=asap_root,
        repo_root=registry_path.parent,
    )
    prepare.write_markdown(selection, output_root / "raw-selection.md")

    assert selection["selected_count"] == 30
    manifests = list(output_root.glob("*/manifest.json"))
    assert len(manifests) == 30
    for item in selection["selected"]:
        case_root = output_root / str(item["case_id"])
        assert (case_root / "raw" / "recognition.json").is_file()
        assert (case_root / "raw" / "beat_grid.json").is_file()
        manifest = json.loads((case_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "production_acceptance_v4_raw_manifest_1"
        assert manifest["stage"] == "raw-staging"
        assert manifest["raw_only"] is True
        assert manifest["pipelines"] == {}
        assert manifest["raw"]["recognition_sha256"] == item["staged_recognition_sha256"]
        assert item["staged_recognition_sha256"] == item["source_recognition_sha256"]
        assert item["staged_beat_grid_sha256"] == item["source_beat_grid_sha256"]
