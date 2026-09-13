from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.pilot_bt_integer_phase_diagnostic import (
    PHASE_WEIGHTS,
    _candidate_downbeats,
    _derive_joint_period,
    _extract_bass_onsets,
    _beat_output_invariant,
    _map_joint_downbeats_to_beat_phase,
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


def test_bass_extraction_uses_raw_note_fields_and_deduplicates_chords(tmp_path) -> None:
    path = tmp_path / "recognition.json"
    payload = {
        "notes": [
            {"start_sec": 1.0, "midi": 42, "instrument_group": "electric_bass", "voice_id": "electric_bass", "is_drum": False},
            {"start_sec": 1.0000004, "midi": 45, "instrument_group": "electric_bass", "voice_id": "electric_bass", "is_drum": False},
            {"start_sec": 2.0, "midi": 40, "instrument_group": "acoustic_piano", "voice_id": "piano", "is_drum": False},
            {"start_sec": 3.0, "midi": 60, "instrument_group": "drums", "voice_id": "drums", "is_drum": True},
        ],
        "beat_grid": {"must_not_be_read": True},
        "provenance": {"source_audio": str(tmp_path / "audio.wav")},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _extract_bass_onsets(path, tmp_path / "audio.wav")

    assert result["bass_onsets"] == [1.0, 2.0]
    assert result["bass_candidate_count"] == 3
    assert result["bass_onset_count"] == 2
    assert result["used_fields"] == ["start_sec", "midi", "instrument_group", "voice_id", "is_drum"]


def test_joint_period_is_derived_from_positions_and_invalid_cycle_rejected() -> None:
    downbeats = np.array([[0.0, 3], [1.0, 4], [2.0, 1], [3.0, 2], [4.0, 3]], dtype=np.float64)
    result = _derive_joint_period(downbeats, "misaligned")
    assert result["period_beats"] == 4
    assert result["candidate_periods"] == [4]

    with pytest.raises(ValueError):
        _derive_joint_period(np.array([[0.0, 1], [1.0, 3], [2.0, 2]], dtype=np.float64), "broken")


def test_joint_phase_maps_times_to_beat_indices_instead_of_joint_row_numbers() -> None:
    # The first joint row is a pickup event.  Position-1 rows are joint rows
    # 2 and 6, but nearest beat indices are 3 and 7; row-index modulo would
    # incorrectly return phase 2 instead of 3.
    beats = np.arange(-1.0, 8.0, dtype=np.float64)
    downbeats = np.array(
        [[0.0, 3], [1.0, 4], [2.0, 1], [3.0, 2], [4.0, 3], [5.0, 4], [6.0, 1]],
        dtype=np.float64,
    )
    result = _map_joint_downbeats_to_beat_phase(beats, downbeats, 4, "misaligned")

    assert result["phase_index"] == 3
    assert result["mapped_count"] == 2
    assert result["mapped_deviation_max_sec"] == 0.0
    assert [item["nearest_beat_index"] for item in result["mapped"]] == [3, 7]


def test_baseline_and_changed_outputs_have_explicit_asymmetric_sources() -> None:
    raw = {
        "original_downbeat_times": [0.1, 2.1],
        "beat_events": np.arange(8, dtype=np.float64),
    }
    baseline = _candidate_downbeats(raw, {"decision": "abstain", "period_beats": 4, "selected_phase_index": 0})
    changed = _candidate_downbeats(raw, {"decision": "changed", "period_beats": 4, "selected_phase_index": 1})

    assert baseline["source"] == "joint_dbn_downbeat_events"
    assert baseline["times"] == [0.1, 2.1]
    assert changed["source"] == "beat_events_integer_phase_slice"
    assert changed["times"] == [1.0, 5.0]
    assert baseline["times"] != changed["times"]


def test_beat_invariant_compares_actual_output_to_frozen_input_hash_and_values() -> None:
    frozen = [0.0, 1.0, 2.0]
    expected_sha = _beat_output_invariant(frozen, "placeholder", frozen)["actual_output_sha256"]
    passed = _beat_output_invariant(frozen, expected_sha, [0.0, 1.0, 2.0])
    failed = _beat_output_invariant(frozen, expected_sha, [0.0, 1.001, 2.0])

    assert passed["invariant_passed"] is True
    assert failed["invariant_passed"] is False
    assert failed["hash_equal_to_frozen_input"] is False
    assert failed["float_values_equal_to_frozen_input"] is False


def test_half_sets_are_disjoint_and_middle_strong_accent_cannot_hide_disagreement() -> None:
    beats = np.arange(21, dtype=np.float64) * 0.5
    onset = np.array([0.06, 1.13, 0.98, 0.13, 0.13, 0.04, 0.01, 0.08, 2.15, 0.16, 0.15, 0.02, 0.18, 0.16, 0.18, 0.1, 0.18, 0.83, 0.01, 0.0, 0.84])
    low = np.array([0.12, 0.03, 0.14, 0.0, 0.06, 0.19, 0.11, 0.16, 2.13, 0.12, 0.04, 0.11, 0.01, 0.16, 0.19, 0.17, 0.01, 0.07, 0.06, 0.02, 0.13])
    result = _select_integer_phase(
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[],
        period=4,
        original_phase=3,
        check_crop=False,
    )

    assert result["reason"] == "first_last_half_winner_disagreement"
    phase_zero_first = next(item for item in result["half_phase_records"]["first"] if item["phase_index"] == 0)
    phase_zero_last = next(item for item in result["half_phase_records"]["last"] if item["phase_index"] == 0)
    assert set(phase_zero_first["bar_start_indices"]).isdisjoint(phase_zero_last["bar_start_indices"])
    assert phase_zero_first["bar_start_indices"] == [0, 4]
    assert phase_zero_last["bar_start_indices"] == [12, 16]


def test_bass_alignment_is_limited_to_each_half_time_window() -> None:
    beats, onset, low = _phase_accent_fixture(length=25)
    record = _phase_record(
        phase=0,
        beats=beats,
        onset_values=onset,
        low_values=low,
        bass_onsets=[0.0, 1.0, 2.0, 4.0, 8.0, 20.0],
        period=4,
        bar_numbers=[0, 1],
    )

    assert record["bar_time_window_sec"] == [0.0, 4.0]
    assert record["bass_onset_count_in_window"] == 3
