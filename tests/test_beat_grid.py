from __future__ import annotations

import pytest

from backend.jianpu_score.beat_grid import (
    BeatGridError,
    beat_unit_from_grid,
    build_beat_grid,
    choose_tempo_candidate,
    infer_time_signature,
    map_note_seconds,
    normalize_beat_observations,
    resolve_beat_unit,
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


def _state_observations(numbers: list[int]) -> list[dict[str, object]]:
    return [
        {"time_sec": index * 0.5, "beat_number": number, "downbeat": number == 1}
        for index, number in enumerate(numbers)
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


def test_dense_full_track_eighth_notes_do_not_force_double_tempo() -> None:
    result = choose_tempo_candidate(
        [index * 0.5 for index in range(17)],
        source_onsets={"full_track": [index * 0.25 for index in range(33)]},
    )
    assert result["selected_factor"] == 1.0
    assert "细分翻倍" in result["selection_reason"]
    candidates = {item["label"]: item for item in result["candidates"]}
    assert candidates["double"]["prior_penalty"] > 0
    assert candidates["double"]["grid_precision"] == pytest.approx(1.0)
    assert candidates["double"]["score"] > candidates["original"]["score"]


def test_sparse_drum_and_bass_evidence_can_select_half_tempo() -> None:
    result = choose_tempo_candidate(
        [index * 0.5 for index in range(17)],
        source_onsets={
            "drums": [index * 1.0 for index in range(9)],
            "bass": [index * 1.0 for index in range(9)],
        },
    )
    assert result["selected_factor"] == 0.5
    assert "鼓/贝斯" in result["selection_reason"]


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


def test_beatnet_score_origin_aligns_first_downbeat_to_measure_boundary() -> None:
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    beat_grid = {
        "beats": [
            {"index": index, "time_sec": time, "downbeat": index == 1}
            for index, time in enumerate(beat_times)
        ],
        "mapping": {
            "beat_times": beat_times,
            "manual_bpm_scale": 1.0,
            "score_origin": {"downbeat_index": 1, "downbeat_sec": 0.5},
        },
    }
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.5,
        bpm=120,
        time_signature="4/4",
        beat_times=beat_times,
        metadata={"beat_source": "beatnet", "beat_grid": beat_grid},
    )
    score = quantize_events([NoteEvent(start_sec=0.5, end_sec=0.75, midi=60)], analysis, mode="monophonic")
    origin = score.metadata["score_origin"]
    assert origin["strategy"] == "first_downbeat"
    assert origin["downbeat_score_beat"] == pytest.approx(0.0)
    assert origin["origin_shift_beats"] == pytest.approx(-1.0)
    assert score.metadata["downbeat_status"] == "aligned"
    note = next(event for event in score.voices[0].events if event.midi == 60)
    assert note.start_tick == 0


def test_beatnet_score_origin_records_pre_downbeat_pickup_candidate() -> None:
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    beat_grid = {
        "beats": [
            {"index": index, "time_sec": time, "downbeat": index == 1}
            for index, time in enumerate(beat_times)
        ],
        "mapping": {
            "beat_times": beat_times,
            "manual_bpm_scale": 1.0,
            "score_origin": {"downbeat_index": 1, "downbeat_sec": 0.5},
        },
    }
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        time_signature="4/4",
        beat_times=beat_times,
        metadata={"beat_source": "beatnet", "beat_grid": beat_grid},
    )
    score = quantize_events(
        [NoteEvent(start_sec=0.1, end_sec=0.25, midi=60), NoteEvent(start_sec=0.5, end_sec=0.75, midi=62)],
        analysis,
        mode="polyphonic",
    )
    origin = score.metadata["score_origin"]
    assert origin["strategy"] == "first_downbeat_with_pickup_candidate"
    assert origin["pickup_candidate"] is True
    assert origin["downbeat_score_beat"] == pytest.approx(4.0)
    assert origin["downbeat_score_beat"] % origin["downbeat_bar_beats"] == pytest.approx(0.0)
    assert score.metadata["downbeat_status"] == "aligned"


@pytest.mark.parametrize(
    ("beats_per_bar", "expected"),
    [(2, "2/4"), (3, "3/4"), (4, "4/4")],
)
def test_meter_candidates_cover_supported_signatures(beats_per_bar: int, expected: str) -> None:
    observations = normalize_beat_observations(_observations(beats_per_bar * 3, beats_per_bar=beats_per_bar))
    result = infer_time_signature(observations)
    assert result["selected"] == expected
    if expected == "3/4":
        assert result["confidence"] < 0.65
    assert result["confidence"] < 0.65


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
        independent_accent_times=[0.75, 2.25],
    )
    assert result["selected"] == "6/8"
    assert result["confidence"] >= 0.65
    assert result["warning"] is None


def test_six_eighths_candidate_needs_independent_compound_evidence() -> None:
    observations = normalize_beat_observations(_observations(18, step=0.5, beats_per_bar=6))
    result = infer_time_signature(observations)
    assert result["selected"] == "4/4"
    assert result["confidence"] < 0.65
    assert result["warning"]
    assert any(item["value"] == "6/8" for item in result["candidates"])


def test_six_eighths_can_be_selected_when_compound_accents_are_stable() -> None:
    observations = normalize_beat_observations(_observations(18, step=0.5, beats_per_bar=6))
    result = infer_time_signature(
        observations,
        independent_accent_times=[1.5, 4.5, 7.5, 10.5],
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


def test_simple_meter_has_explicit_quarter_beat_semantics() -> None:
    for meter, beats_per_bar in (("2/4", 2), ("3/4", 3), ("4/4", 4)):
        semantics = resolve_beat_unit(meter)
        assert semantics["beat_unit"] == "quarter"
        assert semantics["beat_duration_quarters"] == pytest.approx(1.0)
        assert semantics["beats_per_bar"] == beats_per_bar
        assert semantics["bar_duration_quarters"] == pytest.approx(float(beats_per_bar))
        assert semantics["proven"] is True


@pytest.mark.parametrize(
    ("definition", "expected_unit", "expected_duration", "expected_beats"),
    [
        ("eighth_pulse", "eighth", 0.5, 6),
        ("dotted_quarter_pulse", "dotted_quarter", 1.5, 2),
    ],
)
def test_six_eighths_requires_explicit_pulse_definition(
    definition: str,
    expected_unit: str,
    expected_duration: float,
    expected_beats: int,
) -> None:
    semantics = resolve_beat_unit("6/8", definition)
    assert semantics["beat_unit"] == expected_unit
    assert semantics["beat_duration_quarters"] == pytest.approx(expected_duration)
    assert semantics["beats_per_bar"] == expected_beats
    assert semantics["bar_duration_quarters"] == pytest.approx(3.0)
    assert semantics["source"] == "explicit_candidate_definition"
    assert semantics["proven"] is True

    with pytest.raises(BeatGridError, match="explicitly declare"):
        resolve_beat_unit("6/8")


def test_explicit_eighth_pulse_maps_four_six_eight_bars_to_twelve_quarters() -> None:
    observations = _observations(24, step=0.5, beats_per_bar=6)
    grid = build_beat_grid(
        observations,
        duration_sec=12.0,
        meter_hint="6/8",
        beat_unit_definition="eighth_pulse",
    )
    assert grid["beat_duration_quarters"] == pytest.approx(0.5)
    assert grid["bar_duration_quarters"] == pytest.approx(3.0)
    assert len(grid["bars"]) == 4
    assert [bar["duration_quarters"] for bar in grid["bars"]] == pytest.approx([3.0] * 4)
    assert grid["bars"][-1]["end_quarter"] == pytest.approx(12.0)
    assert seconds_to_beat(
        12.0,
        grid["mapping"]["beat_times"],
        beat_duration_quarters=grid["beat_duration_quarters"],
    ) == pytest.approx(12.0)
    assert map_note_seconds(
        0.0,
        12.0,
        grid["mapping"]["beat_times"],
        beat_duration_quarters=grid["beat_duration_quarters"],
    ) == pytest.approx((0.0, 12.0))


def test_six_eighth_pulse_is_derived_from_dbn_six_position_state() -> None:
    grid = build_beat_grid(
        _observations(24, step=0.5, beats_per_bar=6),
        duration_sec=12.0,
        meter_hint="6/8",
    )
    assert grid["beat_unit_definition"] == "eighth"
    assert grid["beat_unit_source"] == "dbn_meter_state_definition"
    assert grid["dbn_position_count"] == 6
    assert grid["beat_duration_quarters"] == pytest.approx(0.5)
    assert grid["bars"][-1]["end_quarter"] == pytest.approx(12.0)


def test_six_eighth_dotted_quarter_pulse_is_derived_from_dbn_two_position_state() -> None:
    grid = build_beat_grid(
        _observations(8, step=1.5, beats_per_bar=2),
        duration_sec=12.0,
        meter_hint="6/8",
    )
    assert grid["beat_unit_definition"] == "dotted_quarter"
    assert grid["beat_unit_source"] == "dbn_meter_state_definition"
    assert grid["dbn_position_count"] == 2
    assert grid["beat_duration_quarters"] == pytest.approx(1.5)
    assert grid["bars"][-1]["end_quarter"] == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("numbers", "expected_definition"),
    [
        ([4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2], "eighth_pulse"),
        ([2, 1, 2, 1], "dotted_quarter_pulse"),
    ],
)
def test_dbn_meter_state_allows_only_valid_leading_and_trailing_partials(
    numbers: list[int], expected_definition: str
) -> None:
    result = infer_time_signature(
        normalize_beat_observations(_state_observations(numbers)),
        meter_hint="6/8",
    )
    assert result["beat_unit_definition"] == expected_definition


@pytest.mark.parametrize(
    "numbers",
    [
        [4, 5, 6, 1, 2, 3, 5, 6, 1, 2],  # internal six-state jump
        [2, 1, 1, 2, 1],  # internal two-state broken run
        [4, 5, 6, 1, 2],  # boundary fragments without a complete cycle
        [4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 1, 2],  # mixed six/two-state runs
    ],
)
def test_dbn_meter_state_rejects_broken_or_mixed_boundary_runs(numbers: list[int]) -> None:
    result = infer_time_signature(
        normalize_beat_observations(_state_observations(numbers)),
        meter_hint="6/8",
    )
    assert result["beat_unit_definition"] is None


def test_old_six_eighth_grid_is_explicitly_legacy_and_fails_new_semantic_gate() -> None:
    old_grid = {
        "schema_version": "1.0",
        "time_signature": {"selected": "6/8"},
        "beats": [{"time_sec": 0.0, "downbeat": True}, {"time_sec": 0.5}],
    }
    semantics = beat_unit_from_grid(old_grid)
    assert semantics["beat_duration_quarters"] == pytest.approx(1.0)
    assert semantics["source"] == "legacy_schema_default"
    assert semantics["proven"] is False
