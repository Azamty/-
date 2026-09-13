from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_maestro_timeline_v1", ROOT / "scripts" / "audit_maestro_timeline_v1.py")
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_interval_summary_reports_pulse_rate() -> None:
    result = audit._interval_summary([0.0, 0.5, 1.0, 1.5])
    assert result == {"count": 4, "median_interval_sec": 0.5, "median_bpm": 120.0}


def test_pairwise_checks_double_time_phase_without_reference_warp() -> None:
    fast = [index * 0.25 for index in range(9)]
    slow = [index * 0.5 for index in range(5)]
    result = audit._pairwise(fast, slow)
    assert result["best_variant"] == "a_every_second_phase_0"
    assert result["best_metrics"]["f1"] == 1.0


def test_tracker_times_accepts_records_and_explicit_arrays() -> None:
    records = {"records": [{"time_sec": 0.1, "downbeat": True}, {"time_sec": 0.6, "downbeat": False}]}
    assert audit._tracker_times(records) == [0.1, 0.6]
    assert audit._tracker_times(records, downbeats=True) == [0.1]
    assert audit._tracker_times({"beats": [0.2, 0.7]}) == [0.2, 0.7]
