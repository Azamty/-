from __future__ import annotations

import numpy as np

from scripts.pilot_beatnet_activation_decoders import (
    DBN_PROFILES,
    _activation_contrast,
    _candidate_id,
    _ensemble,
    _robust_global_weights,
)


def test_candidate_grid_is_small_and_explicit() -> None:
    assert len(DBN_PROFILES) == 10
    assert _candidate_id("three_model_equal", DBN_PROFILES[2]) == "three_model_equal__wide_tempo"


def test_equal_ensemble_averages_all_models() -> None:
    values = {1: np.zeros((2, 2)), 2: np.ones((2, 2)), 3: np.full((2, 2), 2.0)}
    result = _ensemble(values, "three_model_equal", {1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
    np.testing.assert_allclose(result, 1.0)


def test_robust_weights_are_one_global_normalized_set() -> None:
    cases = {
        "a": {1: np.asarray([[0.0, 0.0], [1.0, 1.0]]), 2: np.asarray([[0.0, 0.0], [0.5, 0.5]]), 3: np.asarray([[0.0, 0.0], [0.2, 0.2]])},
        "b": {1: np.asarray([[0.0, 0.0], [1.0, 1.0]]), 2: np.asarray([[0.0, 0.0], [0.5, 0.5]]), 3: np.asarray([[0.0, 0.0], [0.2, 0.2]])},
    }
    weights = _robust_global_weights(cases)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights[1] > weights[2] > weights[3]


def test_activation_contrast_uses_signal_distribution() -> None:
    assert _activation_contrast(np.asarray([[0.0, 0.0], [1.0, 0.5], [0.2, 0.1]])) > 0
