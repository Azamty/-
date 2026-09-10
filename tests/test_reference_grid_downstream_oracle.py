from __future__ import annotations

import json
from fractions import Fraction

import pytest

from scripts.diagnose_reference_grid_downstream import (
    _aggregate,
    _metric_bundle,
    _reference_grid,
    _repair_existing_case_manifest,
)


def test_reference_grid_preserves_downbeat_phase_and_meter() -> None:
    raw = {"analysis": {"duration_sec": 2.2}}
    annotation = {
        "beats": [
            {"index": 0, "time_sec": 0.2, "downbeat": False},
            {"index": 1, "time_sec": 0.7, "downbeat": True},
            {"index": 2, "time_sec": 1.2, "downbeat": False},
        ],
        "downbeats": [{"index": 1, "time_sec": 0.7, "downbeat": True}],
        "time_signature": "3/4",
        "annotation_policy": "fixture",
    }
    grid = _reference_grid(annotation, raw)
    assert grid["schema_version"] == "reference_grid_downstream_oracle_v1"
    assert grid["time_signature"] == "3/4"
    assert grid["mapping"]["score_origin"]["downbeat_index"] == 1
    assert grid["beats"] == annotation["beats"]
    assert grid["downbeats"] == annotation["downbeats"]
    assert grid["tempo"]["reference_median_interval_sec"] == 0.5


def test_metric_bundle_uses_fixed_total_rhythm_and_keeps_chords() -> None:
    reference = [(60, Fraction(0), Fraction(1)), (64, Fraction(0), Fraction(1))]
    predicted = [(60, Fraction(0), Fraction(1)), (64, Fraction(1), Fraction(2))]
    metrics = _metric_bundle(reference, predicted)
    rhythm = metrics["rhythm_error"]
    assert rhythm["metric_schema"] == "fixed_total_assignment_v1"
    assert rhythm["mean_fixed_total_assignment_rhythm_error_quarter"] == 1.0
    assert metrics["chord_retention"]["retention"] == 0.0


def test_aggregate_reports_whether_oracle_reaches_twenty_percent_gate() -> None:
    def item(case_id: str, category: str, rhythm: float) -> dict[str, object]:
        return {
            "id": case_id,
            "category": category,
            "status": "success",
            "baseline": {"metrics": {"rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": 1.0}}},
            "continuous_mapping": {"rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": rhythm}},
            "performance_mapping": {"rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": rhythm}},
            "final_score": {
                "rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": rhythm},
                "pitch_f1": {"f1": 0.8},
                "chord_retention": {"retention": 0.5},
            },
            "notation_delta_performance_to_final": {"rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": 0.0}},
        }

    report = _aggregate([item("a", "synthetic", 0.7), item("b", "official", 0.7)], {"accuracy_gate": {}})
    assert report["final_score_improvement_vs_baseline_percent"] == pytest.approx(30.0)
    assert report["rhythm_20_percent_target_met"] is True


def test_aggregate_separates_strict_fallback_and_all_diagnostic_populations() -> None:
    def item(case_id: str, fallback: bool) -> dict[str, object]:
        metrics = {
            "rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": 0.5 if fallback else 0.6},
            "pitch_f1": {"f1": 0.8},
            "chord_retention": {"retention": 0.4},
        }
        return {
            "id": case_id,
            "category": "synthetic",
            "status": "success",
            "baseline": {"metrics": {"rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": 1.0}}},
            "continuous_mapping": metrics,
            "performance_mapping": metrics,
            "final_score": metrics,
            "notation_delta_performance_to_final": {"rhythm_error": {"mean_fixed_total_assignment_rhythm_error_quarter": 0.0}},
            "diagnostic_tempo_fallback": {"used": fallback},
        }

    report = _aggregate([item("strict", False), item("fallback", True)], {"accuracy_gate": {}})
    assert report["strict_production_compatible_count"] == 1
    assert report["diagnostic_tempo_fallback_count"] == 1
    assert report["all_diagnostic_count"] == 2
    assert report["groups"]["strict_production_compatible"]["case_count"] == 1
    assert report["groups"]["diagnostic_tempo_fallback"]["case_count"] == 1
    assert report["groups"]["all_diagnostic"]["case_count"] == 2
    assert report["strict_gate_status"] == "incomplete"
    assert report["all_diagnostic_gate_status"] == "provisional"
    assert report["reference_grid_sufficient_for_rhythm_target"] is None


def test_reused_fallback_manifest_points_to_oracle_manifest_and_keeps_failure(tmp_path) -> None:
    case_root = tmp_path / "fallback-case"
    service_root = case_root / "service_output"
    service_root.mkdir(parents=True)
    (service_root / "manifest.json").write_text('{"status":"failed","stage":"musicxml_standardize"}\n', encoding="utf-8")
    (service_root / "manifest.oracle.json").write_text('{"status":"completed","diagnostic_fallback":true}\n', encoding="utf-8")
    (case_root / "manifest.json").write_text(
        '{"result":{"manifest":"service_output/manifest.json"},"service_manifest":{"status":"failed"}}\n',
        encoding="utf-8",
    )
    record = {
        "id": "fallback-case",
        "status": "success",
        "diagnostic_tempo_fallback": {
            "used": True,
            "oracle_service_manifest": "service_output\\manifest.oracle.json",
        },
        "artifacts": {"service_manifest": "fallback-case/service_output/manifest.json"},
    }
    repaired = _repair_existing_case_manifest(tmp_path, record)
    root = json.loads((case_root / "manifest.json").read_text(encoding="utf-8"))
    assert root["service_manifest"] == "service_output/manifest.oracle.json"
    assert root["service_manifest_payload"]["status"] == "completed"
    assert root["result"]["manifest"] == "service_output/manifest.oracle.json"
    assert repaired["artifacts"]["service_manifest"] == "fallback-case/service_output/manifest.oracle.json"
    assert json.loads((service_root / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"
