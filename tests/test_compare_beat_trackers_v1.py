from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_beat_trackers_v1", ROOT / "scripts" / "compare_beat_trackers_v1.py")
assert SPEC and SPEC.loader
comparator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparator)


def test_fixed_candidate_and_profiles_are_predeclared() -> None:
    assert comparator.FIXED_CONFIG_ID == "madmom_official_ensemble__official_234"
    assert [profile["id"] for profile in comparator.DBN_PROFILES] == ["official_234", "official_2346"]
    assert comparator._candidate_id(comparator.DBN_PROFILES[1]) == "madmom_official_ensemble__official_2346"


def test_f1_uses_one_to_one_tolerance_matching() -> None:
    result = comparator._f1([0.0, 1.0], [0.02, 0.03, 2.0])
    assert result["true_positive"] == 1
    assert result["predicted"] == 3
    assert result["reference"] == 2
    assert result["f1"] == pytest.approx(0.4)


def test_target_result_keeps_fixed_global_gate_explicit() -> None:
    row = {"mean_beat_f1": 0.85, "mean_downbeat_f1": 0.75}
    result = comparator._target_result(row)
    assert result["target_met"] is True
    assert result["beat_gap"] == pytest.approx(0.0)
    assert result["downbeat_gap"] == pytest.approx(0.0)


def test_package_inventory_does_not_claim_beat_this_evaluated() -> None:
    # The inventory is intentionally separate from the follow-up decision: a
    # package being importable must not imply that its outputs were scored.
    assert comparator._package_version("package-that-is-not-installed-for-this-test") is None
