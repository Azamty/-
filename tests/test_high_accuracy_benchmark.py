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


def _write_midi(path: Path, *, pitch: int = 60, ppq: int = 480, pitches: tuple[int, ...] | None = None) -> None:
    midi = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="fixture"))
    for index, note in enumerate(pitches or (pitch,)):
        track.append(mido.Message("note_on", note=note, velocity=80, time=0 if index else 0))
    for index, note in enumerate(pitches or (pitch,)):
        track.append(mido.Message("note_off", note=note, velocity=0, time=ppq if index == 0 else 0))
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


def test_evaluator_normalizes_different_ppq_and_does_not_invent_missing_beats(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mid"
    _write_midi(reference, ppq=480, pitches=(60, 64, 67))
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fixture")
    result_root = tmp_path / "results" / "case-1"
    result_root.mkdir(parents=True)
    result_midi = result_root / "case-1.score.mid"
    _write_midi(result_midi, ppq=96, pitches=(60, 64, 67))
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
    assert evaluated["metrics"]["chord_retention"]["retention"] == 1.0
    assert evaluated["metrics"]["rhythm_error"]["mean_onset_error_quarter"] == 0.0
    assert evaluated["metrics"]["rhythm_error"]["mean_duration_error_quarter"] == 0.0
    assert evaluated["metrics"]["rhythm_error"]["mean_rhythm_error_quarter"] == 0.0
    assert evaluated["metrics"]["beat_f1"] is None


def test_beat_grid_time_sec_and_downbeat_metrics_are_read_correctly(tmp_path: Path) -> None:
    beat_grid = tmp_path / "beat_grid.json"
    beat_grid.write_text(
        json.dumps({"beat_grid": {"beats": [
            {"time_sec": 0.0, "downbeat": True},
            {"time_sec": 0.5, "downbeat": False},
            {"time_sec": 1.0, "downbeat": True},
        ]}}),
        encoding="utf-8",
    )
    assert benchmark._read_time_points(beat_grid) == [0.0, 0.5, 1.0]
    assert benchmark._read_time_points(beat_grid, downbeats=True) == [0.0, 1.0]
    assert benchmark.beat_f1([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])["f1"] == 1.0
    assert benchmark.beat_f1([0.0, 1.0], [0.0, 1.0])["f1"] == 1.0


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


def test_accuracy_gate_requires_real_baseline_and_accepts_synthetic_passing_fixture() -> None:
    def case(case_id: int, rhythm: float, pitch: float = 0.9, chord: float = 0.9) -> dict[str, object]:
        return {
            "id": f"case-{case_id}",
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "metrics": {
                "pitch_f1": {"f1": pitch},
                "chord_retention": {"retention": chord},
                "rhythm_error": {"mean_rhythm_error_quarter": rhythm},
                "beat_f1": {"f1": 0.9},
                "downbeat_f1": {"f1": 0.8},
            },
        }

    new = [case(index, 0.1) for index in range(30)]
    baseline = [case(index, 0.2, pitch=0.91, chord=0.89) for index in range(30)]
    gate = benchmark.assess_accuracy_claim(new, baseline)
    assert gate["ready"] is True
    assert abs(float(gate["new_mean_rhythm_error_quarter"]) - 0.1) < 1e-9

    insufficient = benchmark.assess_accuracy_claim(new[:29], baseline)
    assert insufficient["ready"] is False
    assert "30" in insufficient["reason"]
