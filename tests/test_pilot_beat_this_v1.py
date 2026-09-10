from __future__ import annotations

import importlib.util
from pathlib import Path

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
