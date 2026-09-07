from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import mido

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load("high_accuracy_benchmark_dataset_test", ROOT / "scripts" / "high_accuracy_benchmark.py")
generator = _load("high_accuracy_fixture_generator_test", ROOT / "scripts" / "generate_high_accuracy_benchmarks.py")
runner_module = _load("high_accuracy_batch_runner_test", ROOT / "scripts" / "run_high_accuracy_batch.py")
maestro = _load("prepare_maestro_benchmark_test", ROOT / "scripts" / "prepare_maestro_benchmark.py")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_has_distinct_30_case_reliable_set() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    reliable = [case for case in registry["cases"] if case.get("reference_midi_reliable") is True]
    assert len(reliable) == 30
    assert len({case["id"] for case in reliable}) == 30
    assert {case["category"] for case in reliable} == {"synthetic_rendered", "official_piano_rendered", "vocal", "specialized_fixture"}
    assert all(case.get("source_id") in registry["sources"] for case in reliable)


def test_deterministic_smoke_cases_have_midi_audio_and_beat_annotation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    case_ids = ("synthetic-piano-01", "special-triplet", "special-tempo-change")
    for destination in (first, second):
        for case_id in case_ids:
            generator.generate_case(case_id, destination=destination)
    for case_id in case_ids:
        first_root = first / case_id
        second_root = second / case_id
        assert (first_root / f"{case_id}.mid").is_file()
        assert (first_root / f"{case_id}.wav").is_file()
        beat = json.loads((first_root / f"{case_id}.beat_grid.json").read_text(encoding="utf-8"))
        assert beat["source"] == "generated_from_reference_midi"
        assert beat["beat_grid"]["downbeats"]
        assert _hash(first_root / f"{case_id}.mid") == _hash(second_root / f"{case_id}.mid")
        assert _hash(first_root / f"{case_id}.wav") == _hash(second_root / f"{case_id}.wav")
        assert _hash(first_root / f"{case_id}.beat_grid.json") == _hash(second_root / f"{case_id}.beat_grid.json")


def test_generated_downbeat_accents_preserve_pitch_and_timing(tmp_path: Path) -> None:
    first = generator.generate_case("synthetic-guitar-01", destination=tmp_path / "generated")
    assert first["velocity_policy"] == {
        "kind": "deterministic_notated_downbeat_accents",
        "accent_delta": generator.DOWNBEAT_ACCENT_DELTA,
        "accent_delta_by_program_family": {"guitar": generator.GUITAR_ACCENT_DELTA},
        "max_velocity": 112,
        "pitch_and_timing_unchanged": True,
    }
    midi = mido.MidiFile(tmp_path / "generated" / "synthetic-guitar-01" / "synthetic-guitar-01.mid")
    starts: list[tuple[int, int, int]] = []
    absolute = 0
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for message in midi.tracks[1]:
        absolute += int(message.time)
        key = (int(message.channel), int(message.note)) if hasattr(message, "note") else None
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault(key, []).append((absolute, int(message.velocity)))
        elif message.type in {"note_off", "note_on"} and key in active and active[key]:
            start, velocity = active[key].pop(0)
            starts.append((start, int(message.note), velocity))
    assert len(starts) == 16
    bar_ticks = 4 * generator.PPQ
    downbeats = [velocity for start, _pitch, velocity in starts if start % bar_ticks == 0]
    other_beats = [velocity for start, _pitch, velocity in starts if start % bar_ticks != 0]
    assert downbeats and other_beats
    assert min(downbeats) > max(other_beats)

    source_tracks, _tempo, meter, _key = generator._spec_for("synthetic-guitar-01")
    source_notes = sorted(
        (int(note.start * generator.PPQ), int(note.end * generator.PPQ), note.pitch)
        for track in source_tracks
        for note in track.notes
    )
    assert sorted((start, start + 480, pitch) for start, pitch, _velocity in starts) == source_notes
    assert first["renderer_version"] == generator.RENDERER_VERSION
    assert meter == (4, 4)


def test_maestro_track_reader_preserves_source_note_velocity(tmp_path: Path) -> None:
    source = tmp_path / "velocity.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="source", time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=37, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=240))
    track.append(mido.Message("note_on", channel=0, note=64, velocity=101, time=120))
    track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=240))
    midi.tracks.append(track)
    midi.save(source)

    _loaded, tracks, _tempo = maestro._midi_tracks(source)

    assert len(tracks) == 1
    assert [(note.pitch, note.velocity) for note in tracks[0].notes] == [(60, 37), (64, 101)]


def test_batch_runner_recognizes_once_and_shares_immutable_raw_between_pipelines(tmp_path: Path) -> None:
    calls = {"recognizer": 0, "baseline": 0, "new": 0}
    seen: dict[str, int] = {}

    def recognizer(case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        calls["recognizer"] += 1
        return {"source": "test-recognizer", "model_output": True, "notes": [{"midi": 60}], "beat_grid": {"beats": [{"time_sec": 0.0, "downbeat": True}]}}

    def baseline(case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
        calls["baseline"] += 1
        seen["baseline"] = len(raw["notes"])
        # The adapter must not be able to mutate the copy passed to the new chain.
        raw["notes"].append({"midi": 61})  # type: ignore[attr-defined]
        destination.mkdir(parents=True, exist_ok=True)
        return {"artifact": "baseline"}

    def new_chain(case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
        calls["new"] += 1
        seen["new"] = len(raw["notes"])
        destination.mkdir(parents=True, exist_ok=True)
        return {"artifact": "new"}

    runner = runner_module.BenchmarkBatchRunner(recognizer=recognizer, baseline=baseline, new_chain=new_chain, timeout_sec=5)
    case = {"id": "smoke-case", "evaluation_scope": "test"}
    first = runner.run_case(case, result_root=tmp_path / "results")
    second = runner.run_case(case, result_root=tmp_path / "results")
    assert first["status"] == "success"
    assert second["status"] == "success"
    assert calls == {"recognizer": 1, "baseline": 1, "new": 1}
    assert seen == {"baseline": 1, "new": 1}
    raw = json.loads((tmp_path / "results" / "smoke-case" / "raw" / "recognition.json").read_text(encoding="utf-8"))
    assert raw["notes"] == [{"midi": 60}]
    assert raw["model_output"] is True


def _owned_recognizer(mode: str, *, model_output: bool):
    def recognize(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return {
            "source": mode,
            "model_output": model_output,
            "notes": [{"midi": 60, "start_sec": 0.0, "end_sec": 0.5}],
            "beat_grid": {"beats": [{"time_sec": 0.0, "downbeat": True}]},
        }

    recognize.recognizer_identity = {"mode": mode, "implementation": f"test-{mode}", "version": "1"}
    return recognize


def test_batch_runner_rejects_raw_resume_across_recognizer_modes(tmp_path: Path) -> None:
    case = {"id": "mode-switch", "evaluation_scope": "quantizer_isolation_fixture"}
    root = tmp_path / "shared"
    reference = runner_module.BenchmarkBatchRunner(
        recognizer=_owned_recognizer("reference-isolation", model_output=False),
        baseline=None,
        new_chain=None,
    )
    first = reference.run_case(case, result_root=root)
    assert first["raw"]["recognizer_mode"] == "reference-isolation"

    production = runner_module.BenchmarkBatchRunner(
        recognizer=_owned_recognizer("production", model_output=True),
        baseline=None,
        new_chain=None,
    )
    second = production.run_case(case, result_root=root)
    assert second["status"] == "failed"
    assert second["error"]["stage"] == "raw"
    assert "different recognizer" in second["error"]["message"]
    assert "mode-isolated result root" in second["error"]["message"]


def test_batch_runner_derives_effective_scope_from_raw_provenance(tmp_path: Path) -> None:
    case = {"id": "scope-switch", "evaluation_scope": "quantizer_isolation_fixture"}

    def adapter(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return {"artifact": "test"}

    reference_root = tmp_path / "reference"
    reference = runner_module.BenchmarkBatchRunner(
        recognizer=_owned_recognizer("reference-isolation", model_output=False),
        baseline=adapter,
        new_chain=adapter,
    )
    reference_outcome = reference.run_case(case, result_root=reference_root)
    assert reference_outcome["evaluation_scope"] == "quantizer_isolation"
    reference_manifest = json.loads((reference_root / "scope-switch" / "manifest.json").read_text(encoding="utf-8"))
    assert reference_manifest["evaluation_scope"] == "quantizer_isolation"
    assert reference_manifest["pipelines"]["new"]["effective_evaluation_scope"] == "quantizer_isolation"

    production_root = tmp_path / "production"
    production = runner_module.BenchmarkBatchRunner(
        recognizer=_owned_recognizer("production", model_output=True),
        baseline=adapter,
        new_chain=adapter,
    )
    production_outcome = production.run_case(case, result_root=production_root)
    assert production_outcome["evaluation_scope"] == "production_end_to_end"
    production_manifest = json.loads((production_root / "scope-switch" / "manifest.json").read_text(encoding="utf-8"))
    assert production_manifest["evaluation_scope"] == "production_end_to_end"
    assert production_manifest["pipelines"]["new"]["effective_evaluation_scope"] == "production_end_to_end"


def test_high_accuracy_adapter_preserves_raw_drums_but_excludes_them_from_notation(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_build(*, events, is_drum, output_dir, **_kwargs):
        captured["events"] = tuple(events)
        captured["is_drum"] = is_drum
        output_dir.mkdir(parents=True, exist_ok=True)
        final_midi = output_dir / "score.mid"
        final_midi.write_bytes(b"midi")
        manifest = output_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        artifact = SimpleNamespace(as_dict=lambda: {"kind": "score_midi", "relative_path": "score.mid"})
        return SimpleNamespace(
            artifacts=(artifact,),
            manifest_path=manifest,
            status="completed",
            jianpu_status="completed",
        )

    import backend.jianpu_score.high_accuracy_service as service

    monkeypatch.setattr(service, "build_high_accuracy_artifacts", fake_build)
    raw = {
        "notes": [
            {"midi": 60, "start_sec": 0.0, "end_sec": 0.5, "is_drum": False},
            {"midi": 36, "start_sec": 0.0, "end_sec": 0.25, "is_drum": True, "channel": 9},
        ],
        "beat_grid": {"beats": [{"time_sec": 0.0}, {"time_sec": 0.5}]},
        "analysis": {"bpm": 120.0, "time_signature": "4/4", "key": "C", "metadata": {"beat_engine": "beatnet", "beatnet_version": "1.1.3"}},
    }
    analysis, all_events = runner_module._analysis_and_events_from_raw(raw, {"id": "drum-filter"})
    assert len(all_events) == 2
    assert all_events[1].metadata["is_drum"] is True
    result = runner_module.high_accuracy_service_adapter(
        {"id": "drum-filter", "source_kind": "instrumental"},
        raw,
        tmp_path / "new",
    )
    assert result["final_midi"]
    assert captured["is_drum"] is False
    assert [event.midi for event in captured["events"]] == [60]


def test_batch_runner_records_unconfigured_pipeline_without_fabricating_success(tmp_path: Path) -> None:
    runner = runner_module.BenchmarkBatchRunner(
        recognizer=lambda _case, _raw, _destination: {"notes": [], "beat_grid": {}, "model_output": True},
        baseline=None,
        new_chain=None,
        timeout_sec=5,
    )
    outcome = runner.run_case({"id": "unconfigured", "evaluation_scope": "test"}, result_root=tmp_path / "results")
    assert outcome["status"] == "failed"
    assert outcome["pipelines"]["baseline"]["status"] == "failed"
    assert outcome["pipelines"]["new"]["status"] == "failed"
    manifest = json.loads((tmp_path / "results" / "unconfigured" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "not configured" in manifest["pipelines"]["new"]["error"]


def test_batch_runner_can_resume_failed_pipeline_without_rerunning_raw(tmp_path: Path) -> None:
    calls = {"recognizer": 0}

    def recognizer(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        calls["recognizer"] += 1
        return {"notes": [{"midi": 60}], "beat_grid": {"beats": []}, "model_output": True}

    root = tmp_path / "results"
    first = runner_module.BenchmarkBatchRunner(recognizer=recognizer, baseline=None, new_chain=None, timeout_sec=5)
    assert first.run_case({"id": "retry-case"}, result_root=root)["status"] == "failed"

    def adapter(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return {"ok": True}

    second = runner_module.BenchmarkBatchRunner(recognizer=recognizer, baseline=adapter, new_chain=adapter, timeout_sec=5)
    assert second.run_case({"id": "retry-case"}, result_root=root)["status"] == "success"
    assert calls["recognizer"] == 1


def test_batch_runner_writes_baseline_and_new_to_explicit_independent_roots(tmp_path: Path) -> None:
    def recognizer(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return {"notes": [{"midi": 60}], "beat_grid": {"beats": []}, "model_output": True}

    def adapter(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return {"ok": True}

    raw_root = tmp_path / "raw"
    baseline_root = tmp_path / "baseline"
    new_root = tmp_path / "new"
    outcome = runner_module.BenchmarkBatchRunner(recognizer=recognizer, baseline=adapter, new_chain=adapter).run_case(
        {"id": "separate", "evaluation_scope": "test"},
        result_root=raw_root,
        baseline_result_root=baseline_root,
        new_result_root=new_root,
    )
    assert outcome["status"] == "success"
    assert (raw_root / "separate" / "raw" / "recognition.json").is_file()
    assert (baseline_root / "separate" / "manifest.json").is_file()
    assert (new_root / "separate" / "manifest.json").is_file()
    assert not (raw_root / "separate" / "baseline").exists()
    assert not (raw_root / "separate" / "new").exists()


def test_maestro_selector_records_archive_and_member_hashes(tmp_path: Path, monkeypatch) -> None:
    import mido
    import zipfile

    source_midi = tmp_path / "source.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source_midi)
    archive = tmp_path / "maestro.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source_midi, "z/second.mid")
        bundle.write(source_midi, "a/first.mid")
    monkeypatch.setattr(maestro, "EXPECTED_SHA256", _hash(archive))
    result = maestro.prepare_archive(archive, output_root=tmp_path / "maestro", count=2)
    assert [item["archive_member"] for item in result["cases"]] == ["a/first.mid", "z/second.mid"]
    assert result["archive"]["sha256"] == _hash(archive)
    assert all(item["midi"]["sha256"] for item in result["cases"])
    assert (tmp_path / "maestro" / "selection_manifest.json").is_file()
