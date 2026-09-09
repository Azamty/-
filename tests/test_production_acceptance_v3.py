from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_production_acceptance_v3",
    ROOT / "scripts" / "prepare_production_acceptance_v3.py",
)
assert SPEC and SPEC.loader
stager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stager)
REPORT_SPEC = importlib.util.spec_from_file_location(
    "write_production_acceptance_v3_report",
    ROOT / "scripts" / "write_production_acceptance_v3_report.py",
)
assert REPORT_SPEC and REPORT_SPEC.loader
reporter = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(reporter)


def test_v3_selection_plan_has_exact_22_3_5_composition() -> None:
    registry = json.loads((ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    plan = stager.build_selection_plan(registry)
    assert len(plan) == 30
    counts = {}
    for item in plan:
        counts[item["source_batch"]] = counts.get(item["source_batch"], 0) + 1
    assert counts == {
        "fluidsynth-production-raw-v1": 22,
        "fluidsynth-special-context-production-raw-v1": 3,
        "ccmusic-production-context-v1": 5,
    }
    assert {item["case_id"] for item in plan if item["source_batch"] == "fluidsynth-special-context-production-raw-v1"} == {
        "special-pickup-3-4",
        "special-triplet",
        "special-complex-chord",
    }


def test_v3_selection_plan_rejects_reference_only_case() -> None:
    registry = {
        "cases": [
            {
                "id": "pjs001",
                "evaluation_policy": "reference_metrics",
                "reference_midi_reliable": True,
            }
        ]
    }
    with pytest.raises(ValueError, match="forbidden case"):
        stager.build_selection_plan(registry)


def test_v3_report_selection_excludes_nonproduction_registry_cases() -> None:
    selected = [f"case-{index:02d}" for index in range(30)]
    registry = {
        "cases": [
            *[
                {"id": case_id, "reference_midi_reliable": True}
                for case_id in selected
            ],
            {"id": "pjs001", "reference_midi_reliable": False},
            {"id": "luv-letter", "reference_midi_reliable": False},
        ]
    }
    indexed = reporter._selected_registry_cases(registry, selected)
    assert list(indexed) == selected
    assert "pjs001" not in indexed
    assert "luv-letter" not in indexed
