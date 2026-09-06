from __future__ import annotations

import pytest

from backend.jianpu_score.beat_grid import (
    BeatGridError,
    build_beat_grid,
    choose_tempo_candidate,
    infer_time_signature,
    map_note_seconds,
    normalize_beat_observations,
    seconds_to_beat,
)
from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.quantize import quantize_events


def _observations(count: int, *, step: float = 0.5, beats_per_bar: int = 4) -> list[dict[str, object]]:
    return [
        {
            "time_sec": index * step,
            "beat_number": index % beats_per_bar + 1,
            "downbeat": index % beats_per_bar == 0,
        }
        for index in range(count)
    ]


def test_piecewise_mapping_interpolates_and_extrapolates_variable_tempo() -> None:
    beat_times = [0.25, 0.75, 1.5, 1.75]
    assert seconds_to_beat(0.25, beat_times) == pytest.approx(0.0)
    assert seconds_to_beat(1.125, beat_times) == pytest.approx(1.5)
    assert seconds_to_beat(0.0, beat_times) == pytest.approx(-0.5)
    assert seconds_to_beat(2.0, beat_times) == pytest.approx(4.0)
    assert map_note_seconds(0.5, 1.25, beat_times) == pytest.approx((0.5, 5 / 3))


def test_mapping_rejects_invalid_intervals() -> None:
    with pytest.raises(BeatGridError, match="strictly increasing"):
        seconds_to_beat(0.5, [0.0, 0.5, 0.5])
    with pytest.raises(BeatGridError, match="end_sec"):
        map_note_seconds(1.0, 1.0, [0.0, 0.5])


def test_tempo_candidates_use_onsets_and_keep_half_original_double_evidence() -> None:
    result = choose_tempo_candidate(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        source_onsets={"drums": [0.0, 0.5, 1.0, 1.5, 2.0], "bass": [0.0, 1.0, 2.0]},
    )
    assert result["selected_factor"] == 1.0
    assert [item["label"] for item in result["candidates"]] == ["half", "original", "double"]
    assert result["evidence_sources"] == ["drums", "bass"]
    assert any(item["selected"] for item in result["candidates"])


def test_manual_bpm_scales_local_beat_shape_without_discarding_phase() -> None:
    grid = build_beat_grid(
        _observations(9, step=0.5),
        duration_sec=4.5,
        manual_bpm=60.0,
        manual_time_signature="4/4",
    )
    assert grid["tempo"]["selected_bpm"] == pytest.approx(60.0)
    assert grid["tempo"]["manual_scale"] == pytest.approx(0.5)
    assert grid["mapping"]["first_beat_sec"] == pytest.approx(0.0)
    assert grid["mapping"]["beat_times"][:3] == pytest.approx([0.0, 0.5, 1.0])


def test_quantizer_consumes_manual_scale_and_retains_beatnet_phase() -> None:
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=60,
        beat_times=[0.25, 0.75, 1.25, 1.75],
        metadata={
            "beat_source": "beatnet",
            "beat_grid": {"mapping": {"manual_bpm_scale": 0.5}},
        },
    )
    score = quantize_events([NoteEvent(start_sec=0.75, end_sec=1.0, midi=60)], analysis, mode="monophonic")
    assert score.metadata["beat_scale"] == pytest.approx(0.5)
    assert score.tempo_events[0].bpm == pytest.approx(60.0)
    assert score.metadata["beat_offset_sec"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("beats_per_bar", "expected"),
    [(2, "2/4"), (3, "3/4"), (4, "4/4"), (6, "6/8")],
)
def test_meter_candidates_cover_supported_signatures(beats_per_bar: int, expected: str) -> None:
    observations = normalize_beat_observations(_observations(beats_per_bar * 3, beats_per_bar=beats_per_bar))
    result = infer_time_signature(observations)
    assert result["selected"] == expected
    if expected == "3/4":
        assert result["confidence"] < 0.65
    else:
        assert result["confidence"] >= 0.65


def test_six_eighths_without_compound_accent_remains_explicitly_ambiguous() -> None:
    observations = normalize_beat_observations(_observations(9, beats_per_bar=3))
    result = infer_time_signature(observations)
    assert result["selected"] == "3/4"
    assert result["confidence"] < 0.65
    assert result["warning"]


def test_compound_accent_can_select_six_eighths_from_three_beat_dbn_output() -> None:
    observations = _observations(9, beats_per_bar=3)
    result = infer_time_signature(
        normalize_beat_observations(observations),
        accent_times=[0.75, 2.25],
    )
    assert result["selected"] == "6/8"
    assert result["confidence"] >= 0.65
    assert result["warning"] is None


def test_grid_records_bars_local_bpm_candidates_and_low_confidence_warning() -> None:
    grid = build_beat_grid(_observations(9, beats_per_bar=3), duration_sec=4.5)
    assert grid["mode"] == "offline"
    assert grid["inference"] == "DBN"
    assert grid["bars"][0]["local_bpm"] == pytest.approx(120.0)
    assert {item["label"] for item in grid["tempo"]["candidates"]} == {"half", "original", "double"}
    assert any("拍号" in warning for warning in grid["warnings"])
