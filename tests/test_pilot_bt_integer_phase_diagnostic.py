from __future__ import annotations

import numpy as np

from scripts.pilot_bt_integer_phase_diagnostic import (
    PHASE_WEIGHTS,
    _phase_record,
    _select_integer_phase,
)


def _phase_accent_fixture(*, length: int = 25, period: int = 4, winning_phase: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beats = np.arange(length, dtype=np.float64) * 0.5
    onset = np.full(length, 0.05, dtype=np.float64)
    low = np.full(length, 0.05, dtype=np.float64)
    onset[winning_phase::period] = 0.95
    low[winning_phase::period] = 0.95
    return beats, onset, low


def test_integer_phase_keeps_beat_values_and_passes_crop_equivariance() -> None:
    beats, onset, low = _phase_accent_fixture()
    before = beats.copy()
    result = _select_integer_phase(
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[],
        period=4,
        original_phase=1,
    )

    assert result["decision"] == "changed"
    assert result["selected_phase_index"] == 2
    assert result["crop_equivariance"]["passed"] is True
    assert result["crop_equivariance"]["expected_phase_index"] == 1
    assert len(beats) == len(before)
    assert np.array_equal(beats, before)


def test_missing_bass_is_zero_without_reweighting_and_weights_are_fixed() -> None:
    beats, onset, low = _phase_accent_fixture(length=25)
    record = _phase_record(
        phase=2,
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[],
        period=4,
    )

    assert PHASE_WEIGHTS == {
        "onset_contrast": 0.40,
        "low_frequency_contrast": 0.25,
        "bass_alignment": 0.20,
        "bar_stability": 0.15,
    }
    assert record["components"]["bass_alignment"] == 0.0
    expected = sum(PHASE_WEIGHTS[key] * record["components"][key] for key in PHASE_WEIGHTS)
    assert record["score_higher_is_better"] == expected


def test_phase_abstains_when_fewer_than_three_complete_bars_compete() -> None:
    beats, onset, low = _phase_accent_fixture(length=9)
    result = _select_integer_phase(
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[],
        period=4,
        original_phase=1,
    )

    assert result["decision"] == "abstain"
    assert result["selected_phase_index"] == 1
    assert result["proposed_phase_index"] is None
    assert all(record["complete_bar_count"] < 3 for record in result["phase_records"])


def test_phase_abstains_on_non_unique_or_low_margin_winner() -> None:
    beats = np.arange(25, dtype=np.float64) * 0.5
    onset = np.full(25, 0.3, dtype=np.float64)
    low = np.full(25, 0.3, dtype=np.float64)
    result = _select_integer_phase(
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[],
        period=4,
        original_phase=3,
    )

    assert result["decision"] == "abstain"
    assert result["reason"] in {"no_unique_global_winner", "winner_margin_below_0_10"}
    assert result["selected_phase_index"] == 3


def test_crop_phase_is_translated_instead_of_forced_to_zero() -> None:
    beats, onset, low = _phase_accent_fixture(length=25, winning_phase=3)
    result = _select_integer_phase(
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[],
        period=4,
        original_phase=1,
    )

    assert result["selected_phase_index"] == 3
    crop = result["crop_equivariance"]
    assert crop["passed"] is True
    assert crop["expected_phase_index"] == 2
    assert crop["cropped_selected_phase_index"] == 2

