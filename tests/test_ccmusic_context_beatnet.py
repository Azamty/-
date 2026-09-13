from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "ccmusic_context_beatnet.py"
    spec = importlib.util.spec_from_file_location("ccmusic_context_beatnet_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


context = _load()


def _full_grid() -> dict[str, Any]:
    beats = [
        {
            "index": index,
            "time_sec": float(index),
            "beat_number": index % 4 + 1,
            "downbeat": index % 4 == 0,
            "bar_index": index // 4,
            "quarter_position": float(index),
            "local_bpm": 60.0,
        }
        for index in range(10)
    ]
    return {
        "schema_version": "1.1",
        "engine": "beatnet",
        "mode": "offline",
        "inference": "DBN",
        "beat_unit_definition": "quarter",
        "beat_unit_source": "standard_meter_definition",
        "beat_unit_proven": True,
        "beat_duration_quarters": 1.0,
        "beats_per_bar": 4,
        "bar_duration_quarters": 4.0,
        "beats": beats,
        "downbeats": [item for item in beats if item["downbeat"]],
        "bars": [
            {
                "index": bar_index,
                "start_beat_index": bar_index * 4,
                "end_beat_index": min(10, (bar_index + 1) * 4),
                "start_sec": float(bar_index * 4),
                "end_sec": float(min(9, (bar_index + 1) * 4 - 1)),
                "start_quarter": float(bar_index * 4),
                "end_quarter": float(min(10, (bar_index + 1) * 4)),
                "beat_count": len(beats[bar_index * 4 : min(10, (bar_index + 1) * 4)]),
            }
            for bar_index in range(3)
        ],
        "time_signature": {"selected": "4/4", "confidence": 0.55},
        "tempo": {"selected_bpm": 60.0, "selected_factor": 1.0},
        "mapping": {
            "beat_times": [item["time_sec"] for item in beats],
            "score_origin": {
                "downbeat_index": 4,
                "downbeat_sec": 4.0,
                "pickup_candidate": True,
                "pickup_beats": 4.0,
                "origin_shift_beats": -4.0,
            },
        },
        "beatnet": {"version": "1.1.3"},
    }


def test_full_track_window_keeps_absolute_grid_and_one_boundary_each_side() -> None:
    grid = context.crop_full_track_beat_grid(
        _full_grid(),
        absolute_start_sec=2.25,
        duration_sec=3.0,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 10},
    )

    assert grid["context"]["mode"] == "full_track_absolute_window"
    assert grid["context"]["full_track_beat_count"] == 10
    assert grid["context"]["window"]["absolute_start_sec"] == 2.25
    assert grid["context"]["window"]["absolute_end_sec"] == 5.25
    assert grid["context"]["window"]["absolute_to_local_offset_sec"] == -2.25
    assert [item["full_track_index"] for item in grid["context"]["window"]["boundary_beats"]["before"]] == [2]
    assert [item["full_track_index"] for item in grid["context"]["window"]["boundary_beats"]["after"]] == [6]
    assert [item["full_track_index"] for item in grid["beats"]] == [3, 4, 5]
    assert [item["time_sec"] for item in grid["beats"]] == [0.75, 1.75, 2.75]
    assert [item["index"] for item in grid["beats"]] == [0, 1, 2]
    assert [item["bar_index"] for item in grid["beats"]] == [0, 1, 1]
    assert [item["beat_number"] for item in grid["beats"]] == [4, 1, 2]
    assert [item["quarter_position"] for item in grid["beats"]] == [0.0, 1.0, 2.0]
    assert [item["full_track_bar_index"] for item in grid["beats"]] == [0, 1, 1]
    assert [item["full_track_beat_number"] for item in grid["beats"]] == [4, 1, 2]
    assert [item["full_track_time_sec"] for item in grid["beats"]] == [3.0, 4.0, 5.0]
    assert [item["full_track_index"] for item in grid["downbeats"]] == [4]
    assert grid["mapping"]["beat_times"] == [0.75, 1.75, 2.75]
    assert grid["mapping"]["full_track_beat_times"] == list(range(10))
    assert grid["mapping"]["score_origin"]["downbeat_index"] == 1
    assert grid["mapping"]["score_origin"]["downbeat_sec"] == 1.75
    assert grid["mapping"]["score_origin"]["downbeat_status"] == "undetermined"
    assert grid["mapping"]["score_origin"]["pickup_candidate"] is False
    assert grid["mapping"]["score_origin"]["origin_shift_beats"] == 0.0
    assert grid["mapping"]["score_origin"]["full_track_downbeat_index"] == 4
    assert grid["mapping"]["full_track_score_origin"]["pickup_candidate"] is True
    assert grid["mapping"]["score_origin"]["warning"] in grid["warnings"]
    assert [bar["index"] for bar in grid["bars"]] == [0, 1]
    assert [(bar["start_beat_index"], bar["end_beat_index"]) for bar in grid["bars"]] == [(0, 1), (1, 3)]
    assert [(bar["start_sec"], bar["end_sec"]) for bar in grid["bars"]] == [(0.0, 0.75), (1.75, 3.0)]
    assert [bar["beat_count"] for bar in grid["bars"]] == [1, 2]
    assert all("start_quarter" not in bar for bar in grid["bars"])
    assert all("end_quarter" not in bar for bar in grid["bars"])
    assert all("duration_quarters" not in bar for bar in grid["bars"])
    assert grid["bars"][0]["full_track_beat_count"] == 4
    assert [bar["partial_window"] for bar in grid["bars"]] == [True, True]
    assert grid["bars"][0]["full_track_start_beat_index"] == 0
    assert grid["bars"][0]["full_track_end_beat_index"] == 4
    assert grid["context"]["evaluation"]["boundary_beats_excluded_from_metrics"] is True
    assert grid["context"]["window"]["boundary_beats"]["before"][0]["index"] == -1
    assert grid["context"]["window"]["boundary_beats"]["after"][0]["index"] == 3
    assert grid["context"]["window"]["boundary_beats"]["before"][0]["include_in_evaluation"] is False
    assert all(item["include_in_evaluation"] for item in grid["beats"])


def test_beat_on_window_end_does_not_create_zero_duration_bar() -> None:
    grid = context.crop_full_track_beat_grid(
        _full_grid(),
        absolute_start_sec=1.0,
        duration_sec=3.0,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 10},
    )

    assert [item["full_track_index"] for item in grid["beats"]] == [1, 2, 3, 4]
    assert len(grid["bars"]) == 1
    assert grid["bars"][0]["start_sec"] == 0.0
    assert grid["bars"][0]["end_sec"] == 2.0


def test_full_bar_keeps_exact_quarter_duration_between_partial_neighbors() -> None:
    grid = context.crop_full_track_beat_grid(
        _full_grid(),
        absolute_start_sec=0.5,
        duration_sec=8.0,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 10},
    )

    full_bar = next(bar for bar in grid["bars"] if not bar["partial_window"])
    assert full_bar["full_track_index"] == 1
    assert full_bar["beat_count"] == 4
    assert full_bar["duration_quarters"] == 4.0
    assert full_bar["end_quarter"] - full_bar["start_quarter"] == 4.0


def _grid_with_timing(
    times: list[float],
    *,
    downbeat_indices: set[int],
    meter: str = "4/4",
) -> dict[str, Any]:
    numerator = int(meter.split("/", 1)[0])
    beats = [
        {
            "index": index,
            "time_sec": time_sec,
            "beat_number": index % numerator + 1,
            "downbeat": index in downbeat_indices,
            "bar_index": index // numerator,
            "quarter_position": float(index),
            "local_bpm": 50.0 + index,
        }
        for index, time_sec in enumerate(times)
    ]
    bars = [
        {
            "index": bar_index,
            "start_beat_index": start,
            "end_beat_index": min(len(beats), start + numerator),
            "start_sec": times[start],
            "end_sec": times[min(len(beats) - 1, start + numerator - 1)],
            "start_quarter": float(start),
            "end_quarter": float(min(len(beats), start + numerator)),
            "beat_count": len(beats[start : min(len(beats), start + numerator)]),
        }
        for bar_index, start in enumerate(range(0, len(beats), numerator))
    ]
    return {
        **_full_grid(),
        "beats": beats,
        "downbeats": [item for item in beats if item["downbeat"]],
        "bars": bars,
        "time_signature": {"selected": meter, "confidence": 0.8},
        "mapping": {
            "beat_times": list(times),
            "score_origin": {"downbeat_index": min(downbeat_indices), "downbeat_sec": times[min(downbeat_indices)]},
        },
    }


def test_first_full_track_downbeat_can_be_local_window_index_zero() -> None:
    grid = context.crop_full_track_beat_grid(
        _grid_with_timing([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], downbeat_indices={1, 5}),
        absolute_start_sec=0.0,
        duration_sec=3.0,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 6},
    )

    assert [item["index"] for item in grid["beats"]] == [0, 1, 2, 3]
    assert [item["full_track_index"] for item in grid["beats"]] == [1, 2, 3, 4]
    assert grid["beats"][0]["downbeat"] is True
    assert grid["mapping"]["score_origin"]["downbeat_index"] == 0
    assert grid["mapping"]["score_origin"]["downbeat_sec"] == 0.0
    assert grid["mapping"]["score_origin"]["downbeat_status"] == "aligned"
    assert grid["mapping"]["score_origin"]["origin_shift_beats"] == 0.0


def test_long_opening_rest_is_local_undetermined_and_not_pickup() -> None:
    grid = context.crop_full_track_beat_grid(
        _grid_with_timing([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], downbeat_indices={5}),
        absolute_start_sec=0.0,
        duration_sec=4.5,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 7},
    )

    origin = grid["mapping"]["score_origin"]
    assert origin["downbeat_index"] == 4
    assert origin["downbeat_sec"] == 4.0
    assert origin["downbeat_status"] == "undetermined"
    assert origin["pickup_candidate"] is False
    assert origin["origin_shift_beats"] == 0.0
    assert grid["beats"][0]["bar_index"] == 0
    assert grid["beats"][4]["bar_index"] == 1


def test_window_keeps_local_coordinates_when_full_track_tempo_changes_at_edges() -> None:
    full = _grid_with_timing(
        [-1.0, 0.0, 0.4, 1.4, 3.0, 3.5, 4.5, 6.0],
        downbeat_indices={1, 5},
    )
    grid = context.crop_full_track_beat_grid(
        full,
        absolute_start_sec=0.4,
        duration_sec=3.1,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 8},
    )

    assert [item["time_sec"] for item in grid["beats"]] == [0.0, 1.0, 2.6, 3.1]
    assert [item["full_track_time_sec"] for item in grid["beats"]] == [0.4, 1.4, 3.0, 3.5]
    assert [item["local_bpm"] for item in grid["beats"]] == [52.0, 53.0, 54.0, 55.0]
    assert grid["mapping"]["beat_times"] == [0.0, 1.0, 2.6, 3.1]
    assert grid["mapping"]["full_track_beat_times"] == [-1.0, 0.0, 0.4, 1.4, 3.0, 3.5, 4.5, 6.0]


def test_bars_without_source_quarter_bounds_do_not_invent_local_endpoints() -> None:
    full = _full_grid()
    full["bars"] = [
        {
            "index": 0,
            "start_beat_index": 0,
            "end_beat_index": 4,
            "start_sec": 0.0,
            "end_sec": 3.0,
            "beat_count": 4,
        }
    ]
    grid = context.crop_full_track_beat_grid(
        full,
        absolute_start_sec=0.25,
        duration_sec=3.0,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 10},
    )

    bar = grid["bars"][0]
    assert (bar["start_sec"], bar["end_sec"]) == (0.0, 2.75)
    assert "start_quarter" not in bar
    assert "end_quarter" not in bar
    assert "duration_quarters" not in bar
    assert bar["full_track_start_sec"] == 0.0
    assert bar["full_track_end_sec"] == 3.0


def test_context_raw_reuses_game_notes_and_records_full_track_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"source": "old"}), encoding="utf-8")
    raw = {
        "schema_version": "1.0",
        "model_output": True,
        "notes": [{"midi": 60, "start_sec": 0.25, "end_sec": 0.75, "source": "game-cleaned"}],
        "analysis": {"bpm": 90.0, "time_signature": "2/4", "metadata": {}},
        "beat_grid": {},
        "provenance": {"recognizer_mode": "production", "recognizer_fingerprint": "fp"},
    }
    grid = context.crop_full_track_beat_grid(
        _full_grid(),
        absolute_start_sec=2.25,
        duration_sec=3.0,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 10},
    )
    payload = context._context_raw(
        raw,
        case={"case_id": "case-01", "duration_sec": 3.0},
        beat_grid=grid,
        full_audio={"path": "mix.wav", "sha256": "audio-hash"},
        full_grid_record={"path": "full-grid.json", "sha256": "grid-hash", "beat_count": 10},
        source_raw_path=source,
    )

    assert payload["notes"] == raw["notes"]
    assert payload["analysis"]["bpm"] == 60.0
    assert payload["analysis"]["time_signature"] == "4/4"
    assert payload["provenance"]["beat_source"] == "original_mix_full_track_context"
    assert payload["provenance"]["beat_context"]["evaluation"]["reference_grid_not_used"] is True
    assert payload["provenance"]["note_source_raw"]["note_count"] == 1
