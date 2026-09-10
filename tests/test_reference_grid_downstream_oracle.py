from __future__ import annotations

from fractions import Fraction

import pytest

from scripts.diagnose_reference_grid_downstream import (
    _aggregate,
    _metric_bundle,
    _reference_grid,
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
