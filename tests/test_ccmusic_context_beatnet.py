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
            "local_bpm": 60.0,
        }
        for index in range(10)
    ]
    return {
        "schema_version": "1.0",
        "engine": "beatnet",
        "mode": "offline",
        "inference": "DBN",
        "beats": beats,
        "downbeats": [item for item in beats if item["downbeat"]],
        "time_signature": {"selected": "4/4", "confidence": 0.55},
        "tempo": {"selected_bpm": 60.0, "selected_factor": 1.0},
        "mapping": {"beat_times": [item["time_sec"] for item in beats]},
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
    assert [item["full_track_index"] for item in grid["downbeats"]] == [4]
    assert grid["mapping"]["beat_times"] == [0.75, 1.75, 2.75]
    assert all(item["include_in_evaluation"] for item in grid["beats"])


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
