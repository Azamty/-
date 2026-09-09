from __future__ import annotations

import numpy as np

from scripts.pilot_beatnet_observable_refinement import (
    POLICY,
    _candidate_grids,
    _current_downbeat_config,
    _score_beats,
    _seconds_to_beat,
    _select_downbeats,
)


def _features() -> dict[str, object]:
    onset = np.zeros(220)
    low = np.zeros(220)
    for time in (0.0, 0.5, 1.0, 1.5, 2.0):
        onset[round(time * 22050 / 512)] = 1.0
    for time in (0.0, 2.0):
        low[round(time * 22050 / 512)] = 1.0
    return {"duration_sec": 2.1, "onset": onset, "low_onset": low, "peak_times": [0.0, 0.5, 1.0, 1.5, 2.0], "tempo_peaks": [{"bpm": 120.0, "strength": 1.0}]}


def test_reference_free_score_prefers_audio_aligned_grid() -> None:
    features = _features()
    aligned = _score_beats([0.0, 0.5, 1.0, 1.5, 2.0], features, [0.0, 0.5, 1.0])
    offset = _score_beats([0.2, 0.7, 1.2, 1.7], features, [0.0, 0.5, 1.0])
    assert aligned["score_higher_is_better"] > offset["score_higher_is_better"]


def test_candidate_generation_includes_current_and_audio_affine() -> None:
    candidates = _candidate_grids([0.0, 1.0, 2.0], _features())
    sources = {item["source"] for item in candidates}
    assert "beatnet_current" in sources
    assert "beatnet_affine_audio_proposal" in sources


def test_downbeat_selection_uses_observable_accents() -> None:
    result = _select_downbeats([0.0, 0.5, 1.0, 1.5, 2.0], _features(), [0.0, 2.0])
    assert result["phase_index"] == 0
    assert result["period_beats"] == 4
    assert result["downbeats"] == [0.0, 2.0]


def test_piecewise_seconds_mapping_and_policy_are_fixed() -> None:
    assert _seconds_to_beat(0.75, [0.0, 0.5, 1.0]) == 1.5
    assert POLICY["weight_origin"] == "fixed_theory_before_post_hoc_reference_evaluation"


def test_current_downbeat_config_uses_decoded_meter_and_phase() -> None:
    grid = {"time_signature": {"selected": "3/4"}, "beats": [{"downbeat": False}, {"downbeat": True}]}
    assert _current_downbeat_config(grid) == {"period_beats": 3, "phase_index": 1}
