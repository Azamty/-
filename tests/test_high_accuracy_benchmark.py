from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("high_accuracy_benchmark", ROOT / "scripts" / "high_accuracy_benchmark.py")
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _write_midi(path: Path, *, pitch: int = 60) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="fixture"))
    track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
    track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(path)


def test_registry_records_pjs_and_marks_luv_letter_manual_only() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    assert {item["id"] for item in registry["cases"]} == {"pjs001", "pjs002", "pjs003", "pjs004", "pjs005", "luv-letter"}
    luv = next(item for item in registry["cases"] if item["id"] == "luv-letter")
    assert luv["evaluation_policy"] == "integrity_and_manual_listening_only"
    assert luv["reference_midi_reliable"] is False

    report = benchmark.build_report(registry)
    assert report["registered_count"] == 6
    assert report["evaluated_count"] == 0
    assert report["accuracy_claim_ready"] is False
    assert all(item["metrics"]["pitch_f1"] is None for item in report["cases"])


def test_evaluator_computes_real_midi_metrics_and_does_not_invent_missing_beats(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mid"
    _write_midi(reference)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fixture")
    result_root = tmp_path / "results" / "case-1"
    result_root.mkdir(parents=True)
    result_midi = result_root / "case-1.score.mid"
    _write_midi(result_midi)
    (result_root / "manifest.json").write_text(
        json.dumps({"status": "success", "artifacts": [{"kind": "instrument_score_midi", "relative_path": result_midi.name}]}),
        encoding="utf-8",
    )
    case = {
        "id": "case-1",
        "title": "fixture",
        "input": str(input_audio),
        "reference_midi": str(reference),
        "reference_midi_reliable": True,
        "beat_annotation": None,
        "evaluation_policy": "reference_metrics",
    }
    evaluated = benchmark.evaluate_case(case, result_root=tmp_path / "results")
    assert evaluated["status"] == "evaluated"
    assert evaluated["crash"] is False
    assert evaluated["metrics"]["pitch_f1"]["f1"] == 1.0
    assert evaluated["metrics"]["chord_retention"]["retention"] is None
    assert evaluated["metrics"]["beat_f1"] is None


def test_manual_only_reference_never_becomes_accuracy_result(tmp_path: Path) -> None:
    case = {
        "id": "manual",
        "title": "manual",
        "input": str(tmp_path / "input.wav"),
        "reference_midi": str(tmp_path / "reference.mid"),
        "reference_midi_reliable": False,
        "evaluation_policy": "integrity_and_manual_listening_only",
    }
    result_root = tmp_path / "results" / "manual"
    result_root.mkdir(parents=True)
    (result_root / "manifest.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    evaluated = benchmark.evaluate_case(case, result_root=tmp_path / "results")
    assert evaluated["status"] == "integrity_only"
    assert evaluated["metrics"]["pitch_f1"] is None
