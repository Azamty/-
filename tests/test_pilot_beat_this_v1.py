from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pilot_beat_this_v1", ROOT / "scripts" / "pilot_beat_this_v1.py")
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_representative_selection_is_one_case_per_sorted_category() -> None:
    selected = [{"case_id": "vocal-1"}, {"case_id": "special-1"}, {"case_id": "vocal-2"}, {"case_id": "synthetic-1"}]
    registry = {
        "vocal-1": {"category": "vocal"},
        "special-1": {"category": "specialized_fixture"},
        "vocal-2": {"category": "vocal"},
        "synthetic-1": {"category": "synthetic_rendered"},
    }
    assert pilot._first_case_per_category(selected, registry) == ["special-1", "synthetic-1", "vocal-1"]


def test_f1_matches_each_prediction_to_at_most_one_reference() -> None:
    result = pilot._f1([0.0, 1.0], [0.01, 0.02, 2.0])
    assert result["true_positive"] == 1
    assert result["f1"] == pytest.approx(0.4)


def test_pilot_targets_and_stop_thresholds_are_explicit() -> None:
    assert pilot.TARGET_BEAT_F1 == 0.85
    assert pilot.TARGET_DOWNBEAT_F1 == 0.75
    assert pilot.CLOSE_BEAT_F1 < pilot.TARGET_BEAT_F1
    assert pilot.CLOSE_DOWNBEAT_F1 < pilot.TARGET_DOWNBEAT_F1


def test_dbn_profiles_are_fixed_and_separate_from_minimal() -> None:
    assert pilot.OFFICIAL_DBN_ID == "final0_official_dbn_34"
    assert pilot.OFFICIAL_DBN_BEATS_PER_BAR == [3, 4]
    assert pilot.MADMOM_DBN_ID == "final0_madmom_dbn_2346"
    assert pilot.MADMOM_DBN_BEATS_PER_BAR == [2, 3, 4, 6]


def test_combined_dbn_activation_matches_official_probability_contract() -> None:
    activation = pilot._combined_dbn_activation(np.asarray([0.0, 4.0]), np.asarray([0.0, -4.0]))
    assert activation.shape == (2, 2)
    assert activation[0, 1] > 0.49
    assert activation[1, 0] > 0.9
    assert activation[1, 1] < 0.1


def test_official_dbn_gate_controls_expansion_decision() -> None:
    row = {"candidate_id": pilot.OFFICIAL_DBN_ID, "case_count": 4, "failure_count": 0, "mean_beat_f1": 0.80, "mean_downbeat_f1": 0.70}
    result = pilot._with_target_gate(row)
    assert result["close_to_target"]["met"] is True
    assert result["decision"] == "expand_to_full_batch"
