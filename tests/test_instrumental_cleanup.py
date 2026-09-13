from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

from backend.jianpu_score.instrumental_cleanup import clean_instrumental_model_notes
from backend.jianpu_score.musicxml_standardize import _append_instrumental_cleanup_alignment
from backend.jianpu_score.performance_midi import build_performance_midi


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "scripts" / "run_high_accuracy_batch.py"
    spec = importlib.util.spec_from_file_location("instrumental_cleanup_batch_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _note(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "start_sec": 1.0,
        "end_sec": 1.001,
        "midi": 85,
        "instrument_group": "voice",
        "program": 52,
        "velocity": None,
        "source": "muscriptor",
        "voice_id": "voice",
        "is_drum": False,
    }
    value.update(updates)
    return value


def test_instrumental_cleanup_merges_exact_duplicates_and_accounts_every_source() -> None:
    raw = [
        _note(),
        # Only binary timestamp spelling differs; the normalized interval is
        # identical and all semantic model fields match.
        _note(start_sec=1.0000000000001, end_sec=1.0010000000001),
        _note(start_sec=1.0, end_sec=1.001000001),
        _note(velocity=81),
    ]
    original = copy.deepcopy(raw)

    result = clean_instrumental_model_notes(raw)

    assert raw == original
    assert len(result.events) == 3
    assert result.report["source_note_count"] == 4
    assert result.report["cleaned_note_count"] == 3
    assert result.report["merged_count"] == 1
    assert result.report["accounted_source_count"] == 4
    merged = result.report["merged"][0]
    assert merged["source_index"] == 1
    assert merged["primary_source_index"] == 0
    assert merged["reason"] == "exact_model_duplicate"
    assert merged["normalized_start_sec"] == 1.0
    assert merged["normalized_end_sec"] == 1.001
    assert merged["midi"] == 85
    assert merged["semantic_fingerprint"] == result.report["duplicate_groups"][0]["semantic_fingerprint"]
    assert len(merged["semantic_fingerprint"]) == 64
    assert result.events[0]["_instrumental_cleanup"] == {
        "primary_source_index": 0,
        "source_indices": [0, 1],
        "merged_source_indices": [1],
    }


def test_instrumental_cleanup_does_not_merge_near_or_semantically_different_notes() -> None:
    result = clean_instrumental_model_notes(
        [
            _note(),
            _note(end_sec=1.001000001),
            _note(velocity=81),
            _note(program=53),
            _note(voice_id="other"),
        ]
    )

    assert len(result.events) == 5
    assert result.report["merged_count"] == 0
    assert result.report["duplicate_group_count"] == 0


def test_cleaned_instrumental_events_preserve_raw_accounting_in_performance_metadata() -> None:
    runner = _load_runner()
    raw = {
        "source_kind": "instrumental",
        "model_output": True,
        "notes": [_note(), _note(start_sec=1.0000000000001, end_sec=1.0010000000001)],
        "beat_grid": {"beats": [{"time_sec": 0.0}, {"time_sec": 0.5}]},
        "analysis": {
            "sample_rate": 22050,
            "duration_sec": 2.0,
            "bpm": 120.0,
            "time_signature": "4/4",
            "key": "C",
            "metadata": {"beat_engine": "beatnet", "beatnet_version": "1.1.3"},
        },
    }

    analysis, events = runner._analysis_and_events_from_raw(raw, {"id": "cleanup", "source_kind": "instrumental"})
    assert len(events) == 1
    assert events[0].metadata["instrumental_cleanup"]["source_indices"] == [0, 1]
    cleanup = analysis.metadata["instrumental_cleanup"]
    assert cleanup["source_note_count"] == 2
    assert cleanup["merged_count"] == 1

    midi_bytes, metadata = build_performance_midi(
        events,
        analysis,
        instrument_group="voice",
        program=52,
    )
    assert midi_bytes
    assert metadata["note_count"] == 1
    assert metadata["source_note_count"] == 2
    assert metadata["source_merged_count"] == 1
    assert metadata["notes"][0]["source_index"] == 0
    assert metadata["notes"][0]["source_indices"] == [0, 1]


def test_musicxml_alignment_appends_exact_duplicate_lineage_as_merged() -> None:
    primary = {
        "source_index": 0,
        "source_midi": 85,
        "source_start_tick_480": 480,
        "source_end_tick_480": 481,
        "source_start_tick": 48,
        "source_end_tick": 48,
        "musicxml_unit_id": 7,
        "musicxml_event_id": "p0:e1",
        "musicxml_event_ids": ["p0:e1"],
        "musicxml_start_tick": 48,
        "musicxml_end_tick": 48,
        "musicxml_chain_end_tick": 48,
        "score_start_tick": 48,
        "score_end_tick": 48,
        "source_to_score_movement_start_ticks": 0,
        "source_to_score_movement_end_ticks": 0,
        "musicxml_to_score_movement_start_ticks": 0,
        "musicxml_to_score_movement_end_ticks": 0,
        "reason": "matched_musicxml_event",
        "matching_evidence": "adaptive_quantization_window",
        "accounting_category": "matched",
    }
    alignment, primary_count = _append_instrumental_cleanup_alignment(
        [primary],
        {
            "instrumental_cleanup": {
                "source_note_count": 2,
                "cleaned_note_count": 1,
                "merged_count": 1,
                "merged": [
                    {
                        "source_index": 1,
                        "primary_source_index": 0,
                        "reason": "exact_model_duplicate",
                        "normalized_start_sec": 1.0,
                        "normalized_end_sec": 1.001,
                    }
                ],
            }
        },
    )

    assert primary_count == 1
    assert len(alignment) == 2
    assert alignment[1]["source_index"] == 1
    assert alignment[1]["accounting_category"] == "merged"
    assert alignment[1]["reason"] == "exact_model_duplicate"
    assert alignment[1]["merged_into_source_index"] == 0
