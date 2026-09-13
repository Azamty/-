from __future__ import annotations

import pytest

from scripts.diagnose_beatnet_v3 import (
    _best_affine,
    _best_shift,
    _candidate_evidence,
    _f1,
    _meter_oracle,
    _piecewise_upper,
    _candidate_rows,
)


def test_oracles_separate_phase_affine_and_piecewise_limits() -> None:
    reference = [0.0, 0.5, 1.0, 1.5]
    shifted = [0.2, 0.7, 1.2, 1.7]
    stretched = [0.2, 0.8, 1.4, 2.0]
    assert _f1(reference, shifted)["f1"] == 0.0
    assert _best_shift(reference, shifted)["metrics"]["f1"] == 1.0
    assert _best_affine(reference, stretched)["metrics"]["f1"] == 1.0
    assert _piecewise_upper(reference, stretched + [2.6])["f1_upper_bound"] == pytest.approx(8 / 9)


def test_meter_oracle_searches_meter_and_phase() -> None:
    beats = [index * 0.5 for index in range(13)]
    downbeats = [0.5, 2.0, 3.5, 5.0]
    result = _meter_oracle(downbeats, beats, "3/4")
    assert result["best_meter"] == "3/4"
    assert result["best_phase_index"] == 1
    assert result["metrics"]["f1"] == 1.0


def test_no_reference_candidate_evidence_uses_onsets_only() -> None:
    groups = {"all": [0.0, 0.5, 1.0], "bass": [0.0, 1.0], "drum": []}
    aligned = _candidate_evidence([0.0, 0.5, 1.0], groups, 1.0)
    offset = _candidate_evidence([0.2, 0.7, 1.2], groups, 1.0)
    assert aligned["score_lower_is_better"] < offset["score_lower_is_better"]


def test_full_track_candidates_are_cropped_to_local_window() -> None:
    grid = {
        "context": {"window": {"absolute_start_sec": 10.0, "absolute_end_sec": 12.0}},
        "tempo": {"candidates": [{"label": "original", "factor": 1.0, "beat_times": [9.5, 10.5, 11.5, 12.5]}]},
    }
    candidate = _candidate_rows(grid)[0]
    assert candidate["source_beat_times"] == [9.5, 10.5, 11.5, 12.5]
    assert candidate["beat_times"] == [0.5, 1.5]
    assert candidate["source_time_basis"] == "full_track_absolute"
