from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import mido
import numpy as np
import pytest

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
ccmusic = _load("prepare_ccmusic_benchmark_test", ROOT / "scripts" / "prepare_ccmusic_benchmark.py")
production_recognizer = _load("high_accuracy_production_recognizer_test", ROOT / "scripts" / "high_accuracy_production_recognizer.py")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_has_distinct_30_case_reliable_set() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    reliable = [case for case in registry["cases"] if case.get("reference_midi_reliable") is True]
    assert len(reliable) == 30
    assert len({case["id"] for case in reliable}) == 30
    assert {case["category"] for case in reliable} == {"synthetic_rendered", "official_piano_rendered", "vocal", "specialized_fixture"}
    assert all(case.get("source_id") in registry["sources"] for case in reliable)


def test_local_midi_cases_record_direct_renderer_and_input_hashes() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    policy = registry["source_policy"]["midi_render_policy"]
    assert policy["renderer"] == "fluidsynth_direct_ms_basic_v1"
    assert policy["fluidsynth_version"] == "2.6.0"
    assert policy["sample_rate"] == 44_100
    assert policy["channels"] == 2
    assert policy["sample_width_bytes"] == 2
    assert policy["effects"] == {"reverb": False, "chorus": False}
    assert policy["gain"] == 0.2
    assert policy["tail_sec"] == 0.25

    local_midi = [
        case
        for case in registry["cases"]
        if case.get("source_kind") in {"synthetic", "official-midi-rendered"}
    ]
    assert len(local_midi) == 25
    assert len({case["id"] for case in local_midi}) == 25
    for case in local_midi:
        assert case["renderer_version"] == policy["renderer"]
        assert case["source_event_complete"] is True
        assert case["render_manifest"].endswith(".render_manifest.json")
        for field in (
            "input_sha256",
            "reference_midi_sha256",
            "beat_annotation_sha256",
            "render_manifest_sha256",
        ):
            assert len(case[field]) == 64
            assert all(character in "0123456789abcdef" for character in case[field])

        input_path = ROOT / case["input"]
        if input_path.is_file():
            assert _hash(input_path) == case["input_sha256"]
        manifest_path = ROOT / case["render_manifest"]
        if manifest_path.is_file():
            render_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert render_manifest["renderer"]["renderer_version"] == policy["renderer"]
            assert render_manifest["verification"]["byte_deterministic"] is True
            assert render_manifest["verification"]["source_event_complete"] is True


def test_special_registry_records_fixed_music_context_rule() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    special = {
        item["id"]: item
        for item in registry["cases"]
        if item["category"] == "specialized_fixture"
    }
    assert set(special) == {
        "special-pickup-3-4",
        "special-6-8",
        "special-triplet",
        "special-tempo-change",
        "special-complex-chord",
    }
    assert all(
        item["music_context_policy"]["rule"] == generator.SPECIAL_CONTEXT_RULE
        and item["music_context_policy"]["minimum_complete_measures"] == 4
        and item["music_context_policy"]["preserve_tempo_events"] is True
        for item in special.values()
    )
    assert special["special-pickup-3-4"]["music_context_policy"]["pickup_quarters"] == "1"
    assert all(
        special[case_id]["music_context_policy"]["complete_measures_after_pickup"] == 4
        for case_id in special
    )


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
        render_manifest = json.loads((first_root / f"{case_id}.render_manifest.json").read_text(encoding="utf-8"))
        assert render_manifest["verification"]["byte_deterministic"] is True


def test_special_context_rule_keeps_four_complete_measures() -> None:
    expected = {
        "special-pickup-3-4": ("1", "13", 4),
        "special-6-8": ("0", "12", 4),
        "special-triplet": ("0", "16", 4),
        "special-tempo-change": ("0", "16", 4),
        "special-complex-chord": ("0", "12", 4),
    }
    for case_id, (pickup, required_end, complete_measures) in expected.items():
        tracks, _tempo, meter, _key = generator._spec_for(case_id)
        plan = generator._special_context_plan(case_id, tracks=tracks, meter=meter)
        assert plan["rule"] == generator.SPECIAL_CONTEXT_RULE
        assert plan["pickup_quarters"] == pickup
        assert plan["required_end_quarter"] == required_end
        assert plan["complete_measures_after_pickup"] == complete_measures
        assert plan["meets_requirement"] is True
        assert plan["preserve_tempo_events"] is True


def test_special_triplet_keeps_exact_triplet_ticks_and_pickup_grid(tmp_path: Path) -> None:
    generator.generate_case("special-triplet", destination=tmp_path / "generated")
    triplet = mido.MidiFile(tmp_path / "generated" / "special-triplet" / "special-triplet.mid")
    absolute = 0
    note_intervals: list[tuple[int, int, int]] = []
    active: dict[tuple[int, int], list[int]] = {}
    for message in triplet.tracks[1]:
        absolute += int(message.time)
        key = (int(message.channel), int(message.note)) if hasattr(message, "note") else None
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault(key, []).append(absolute)
        elif message.type in {"note_off", "note_on"} and key in active and active[key]:
            note_intervals.append((active[key].pop(0), absolute, int(message.note)))
    assert len(note_intervals) == 48
    assert {end - start for start, end, _pitch in note_intervals} == {160}
    assert note_intervals[-1][1] == 16 * generator.PPQ

    generator.generate_case("special-pickup-3-4", destination=tmp_path / "generated")
    pickup = json.loads(
        (tmp_path / "generated" / "special-pickup-3-4" / "special-pickup-3-4.beat_grid.json").read_text(
            encoding="utf-8"
        )
    )
    grid = pickup["beat_grid"]
    assert grid["pickup_is_explicit"] is True
    assert grid["pickup_quarters"] == "1"
    assert grid["beats"][0]["pickup"] is True
    assert grid["beats"][0]["downbeat"] is False
    assert [item["time_sec"] for item in grid["downbeats"][:4]] == pytest.approx([0.6, 2.4, 4.2, 6.0])


def test_legacy_renderer_manifest_requires_explicit_replacement(tmp_path: Path) -> None:
    case_root = tmp_path / "generated" / "synthetic-piano-01"
    case_root.mkdir(parents=True)
    (case_root / "case_manifest.json").write_text(
        json.dumps({"renderer_version": "deterministic_harmonic_oscillator_v1"}),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="stale or legacy renderer manifest"):
        generator.generate_case("synthetic-piano-01", destination=tmp_path / "generated")


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
    assert first["source_event_complete"] is True
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


def test_maestro_clip_window_is_source_aligned_and_preserves_intersected_notes() -> None:
    from fractions import Fraction

    notes = (
        generator.RenderNote(Fraction(3, 2), Fraction(5, 2), 60, 37),
        generator.RenderNote(Fraction(9), Fraction(10), 64, 101),
        generator.RenderNote(Fraction(35), Fraction(36), 67, 80),
    )
    track = generator.RenderTrack("source", 0, notes)
    start, end = maestro._clip_window((track,), (4, 4))
    assert (start, end) == (Fraction(0), Fraction(32))
    clipped = maestro._clip_tracks((track,), start, end)
    assert [(n.start, n.end, n.pitch, n.velocity) for n in clipped[0].notes] == [
        (Fraction(3, 2), Fraction(5, 2), 60, 37),
        (Fraction(9), Fraction(10), 64, 101),
    ]


def test_maestro_selector_records_clip_hashes_and_domain(tmp_path: Path, monkeypatch) -> None:
    source_midi = tmp_path / "source.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    track.append(mido.Message("note_on", note=60, velocity=37, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source_midi)
    archive = tmp_path / "maestro.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source_midi, "a/first.mid")
    monkeypatch.setattr(maestro, "EXPECTED_SHA256", _hash(archive))
    result = maestro.prepare_archive(archive, output_root=tmp_path / "maestro", count=1)
    record = result["cases"][0]
    assert result["render_domain"]["production_end_to_end"] is True
    assert result["render_domain"]["benchmark_role"] == "production_end_to_end_render_domain"
    assert record["midi"]["path"].startswith("rendered/clips/")
    assert record["audio"]["path"].startswith("rendered/clips/")
    assert record["beat_annotation"]["path"].startswith("rendered/clips/")
    assert record["render_manifest"]["path"].endswith(".render_manifest.json")
    assert record["source_event_complete"] is True
    assert record["clip"]["source_meter"] == "3/4"
    for key in ("midi", "audio", "beat_annotation"):
        path = tmp_path / "maestro" / record[key]["path"]
        assert path.is_file()
        assert record[key]["sha256"] == _hash(path)


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
    # Selected render-domain cases are registered for the production gate;
    # raw provenance still has authority to downgrade a reference run.
    case = {"id": "scope-switch", "evaluation_scope": "production_end_to_end"}

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


def test_batch_runner_raw_only_skips_score_pipelines(tmp_path: Path) -> None:
    calls = {"baseline": 0, "new": 0}

    def recognizer(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return {
            "notes": [{"midi": 60, "start_sec": 0.0, "end_sec": 0.5}],
            "beat_grid": {"beats": [{"time_sec": 0.0, "downbeat": True}]},
            "model_output": True,
        }

    def forbidden_pipeline(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        calls["baseline"] += 1
        calls["new"] += 1
        raise AssertionError("raw-only mode must not call score pipelines")

    outcome = runner_module.BenchmarkBatchRunner(
        recognizer=recognizer,
        baseline=forbidden_pipeline,
        new_chain=forbidden_pipeline,
        timeout_sec=5,
        raw_only=True,
    ).run_case({"id": "raw-only"}, result_root=tmp_path / "results")

    assert outcome["status"] == "success"
    assert outcome["raw_only"] is True
    assert outcome["pipelines"] == {}
    assert calls == {"baseline": 0, "new": 0}
    assert (tmp_path / "results" / "raw-only" / "raw" / "recognition.json").is_file()
    assert not (tmp_path / "results" / "raw-only" / "baseline").exists()
    assert not (tmp_path / "results" / "raw-only" / "new").exists()


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


def test_batch_runner_records_adapter_failure_and_retries_only_failed_pipeline(tmp_path: Path) -> None:
    calls = {"recognizer": 0, "baseline": 0, "new": 0}

    def recognizer(_case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        calls["recognizer"] += 1
        return {"notes": [{"midi": 60}], "beat_grid": {"beats": []}, "model_output": True}

    def baseline(_case: Mapping[str, Any], _raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
        calls["baseline"] += 1
        destination.mkdir(parents=True, exist_ok=True)
        return {"ok": "baseline"}

    def flaky_new(_case: Mapping[str, Any], _raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
        calls["new"] += 1
        destination.mkdir(parents=True, exist_ok=True)
        if calls["new"] == 1:
            (destination / "partial-service-log.txt").write_text("failed attempt", encoding="utf-8")
            raise RuntimeError("synthetic adapter failure")
        return {"ok": "new"}

    root = tmp_path / "results"
    runner = runner_module.BenchmarkBatchRunner(
        recognizer=recognizer,
        baseline=baseline,
        new_chain=flaky_new,
        timeout_sec=5,
    )
    first = runner.run_case({"id": "failed-pipeline", "evaluation_scope": "test"}, result_root=root)
    assert first["status"] == "partial"
    assert first["pipelines"]["baseline"]["status"] == "success"
    assert first["pipelines"]["new"]["status"] == "failed"
    failed_manifest = json.loads((root / "new" / "failed-pipeline" / "manifest.json").read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["stage"] == "new"
    assert failed_manifest["error"]["message"].endswith("synthetic adapter failure")
    assert failed_manifest["raw_recognition_sha256"] == first["raw"]["recognition_sha256"]

    second = runner.run_case({"id": "failed-pipeline", "evaluation_scope": "test"}, result_root=root)
    assert second["status"] == "success"
    assert calls == {"recognizer": 1, "baseline": 1, "new": 2}
    assert json.loads((root / "new" / "failed-pipeline" / "manifest.json").read_text(encoding="utf-8"))["status"] == "success"
    history_files = list((root / "new" / "failed-pipeline" / "retry_history").rglob("partial-service-log.txt"))
    assert len(history_files) == 1


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


def test_production_recognizer_keeps_note_and_audio_onset_sources_separate(monkeypatch, tmp_path: Path) -> None:
    import backend.jianpu_score.analysis as analysis
    import backend.jianpu_score.vocal_cleanup as vocal_cleanup

    captured: dict[str, Any] = {}
    source = tmp_path / "input.wav"
    source.write_bytes(b"audio")
    notes = [
        {
            "start_sec": 0.0,
            "end_sec": 0.4,
            "midi": 60,
            "confidence": 0.8,
            "velocity": 80,
            "voice_id": "voice-0",
            "source": "game",
            "raw_pitch": 60,
        },
    ]
    drums = tmp_path / "drums.wav"
    bass = tmp_path / "bass.wav"
    drums.write_bytes(b"drums")
    bass.write_bytes(b"bass")

    def fake_run(_audio: Path, _destination: Path, *, demucs_model: str | None = None):
        return notes, {
            "engine": "game",
            "route_input": "demucs_vocals",
            "onset_evidence_stems": {"drums": str(drums), "bass": str(bass)},
        }

    def fake_load(_path: Path, sample_rate: int = 22050):
        return np.zeros(sample_rate, dtype="float32"), sample_rate

    def fake_onsets(_samples, _sample_rate):
        return [0.25]

    def fake_analyze(_path: Path, *, source_onsets):
        captured["source_onsets"] = source_onsets
        return np.zeros(22050, dtype="float32"), SimpleNamespace(
            sample_rate=22050,
            duration_sec=1.0,
            bpm=120.0,
            time_signature="4/4",
            key="C",
            beat_times=[0.0, 0.5, 1.0],
            warnings=[],
            metadata={"beat_grid": {"tempo": {"evidence_sources": sorted(source_onsets)}}},
        )

    monkeypatch.setattr(production_recognizer, "_run_vocal", fake_run)
    monkeypatch.setattr(
        vocal_cleanup,
        "clean_vocal_events",
        lambda events, **_kwargs: SimpleNamespace(report={}, events=list(events)),
    )
    monkeypatch.setattr(analysis, "load_audio", fake_load)
    monkeypatch.setattr(analysis, "extract_onset_times", fake_onsets)
    monkeypatch.setattr(analysis, "analyze_audio", fake_analyze)

    payload = production_recognizer.recognize(source, source_kind="vocal", output=tmp_path / "out")

    assert sorted(captured["source_onsets"]) == ["all", "bass", "drums", "full_track"]
    assert payload["provenance"]["beat_onset_evidence"]["sources"] == ["all", "bass", "drums", "full_track"]
    assert payload["provenance"]["beat_onset_evidence"]["meter_inference_uses_independent_accents"] is False


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
    second = maestro.prepare_archive(archive, output_root=tmp_path / "maestro-second", count=2)
    assert [item["archive_member"] for item in result["cases"]] == ["a/first.mid", "z/second.mid"]
    assert result["archive"]["sha256"] == _hash(archive)
    assert all(item["midi"]["sha256"] for item in result["cases"])
    assert (tmp_path / "maestro" / "selection_manifest.json").is_file()
    assert [item["midi"]["sha256"] for item in result["cases"]] == [item["midi"]["sha256"] for item in second["cases"]]
    assert [item["audio"]["sha256"] for item in result["cases"]] == [item["audio"]["sha256"] for item in second["cases"]]
    assert [item["beat_annotation"]["sha256"] for item in result["cases"]] == [item["beat_annotation"]["sha256"] for item in second["cases"]]


def test_ccmusic_registry_cases_are_five_independent_beat_eligible_segments() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    cases = [item for item in registry["cases"] if item["id"].startswith("ccmusic-yueding-")]
    assert [item["id"] for item in cases] == [f"ccmusic-yueding-{index:02d}" for index in range(1, 6)]
    assert all(item["beat_annotation_independent"] is True for item in cases)
    assert all(item["evaluation_scope"] == "production_end_to_end" for item in cases)
    assert all(item["source_id"] == "ccmusic-demo" for item in cases)
    pjs = [item for item in registry["cases"] if item["id"].startswith("pjs")]
    assert len(pjs) == 5
    assert all(item["reference_midi_reliable"] is False for item in pjs)
    assert all(item["beat_annotation_independent"] is False for item in pjs)


def test_ccmusic_archive_verification_rejects_wrong_hash_and_traversal(tmp_path: Path, monkeypatch) -> None:
    import zipfile

    archive = tmp_path / "ccmusic.zip"
    names = [f"{ccmusic.SOURCE_PREFIX}{name}" for name in sorted(ccmusic.EXPECTED_MEMBERS)]
    with zipfile.ZipFile(archive, "w") as bundle:
        for index, name in enumerate(names):
            bundle.writestr(name, f"fixture-{index}".encode())
    monkeypatch.setattr(ccmusic, "EXPECTED_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(ccmusic, "EXPECTED_MD5", ccmusic._hash_file(archive, "md5").upper())
    monkeypatch.setattr(ccmusic, "EXPECTED_SHA256", ccmusic._sha256(archive))
    monkeypatch.setattr(ccmusic, "EXPECTED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ccmusic._verify_archive(archive)
    monkeypatch.setattr(ccmusic, "EXPECTED_SHA256", ccmusic._sha256(archive))
    _sha, selected = ccmusic._verify_archive(archive)
    assert sorted(Path(name).name for name in selected) == sorted(ccmusic.EXPECTED_MEMBERS)

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as bundle:
        bundle.writestr("../outside.txt", b"bad")
    monkeypatch.setattr(ccmusic, "EXPECTED_ARCHIVE_BYTES", traversal.stat().st_size)
    monkeypatch.setattr(ccmusic, "EXPECTED_MD5", ccmusic._hash_file(traversal, "md5").upper())
    monkeypatch.setattr(ccmusic, "EXPECTED_SHA256", ccmusic._sha256(traversal))
    with pytest.raises(ValueError, match="unsafe archive member"):
        ccmusic._verify_archive(traversal)


def test_ccmusic_beat_grid_has_exact_bar_downbeats(tmp_path: Path) -> None:
    output = tmp_path / "segment.beat_grid.json"
    ccmusic._write_beat_grid(output, 72)
    payload = json.loads(output.read_text(encoding="utf-8"))
    beats = payload["beat_grid"]["beats"]
    assert len(beats) == 17
    assert [item["score_quarter"] for item in beats[:3]] == [72, 73, 74]
    assert [item["index"] for item in payload["beat_grid"]["downbeats"]] == [0, 4, 8, 12, 16]
    assert beats[-1]["time_sec"] == 12.0


def test_ccmusic_alignment_provenance_composes_audio_maps_and_validates_placement(monkeypatch) -> None:
    score_to_guide = {
        "slope_sec_per_quarter": 0.75,
        "offset_sec": -29.5,
        "guide_zero_score_quarter": 39.3333333333,
        "anchors": [{"score_quarter": 40.0}, {"score_quarter": 64.0}],
    }
    score_to_vocal = {
        "slope_sec_per_quarter": 0.75,
        "offset_sec": -29.0,
        "anchors": [{"score_quarter": 40.0}, {"score_quarter": 64.0}],
    }
    guide_to_accompaniment = {
        "slope_accompaniment_sec_per_guide_sec": 1.0,
        "offset_sec": 29.0,
    }
    monkeypatch.setattr(ccmusic, "_fit_score_audio_affine", lambda _payload, _path, *, label: score_to_guide if label == "guide" else score_to_vocal)
    monkeypatch.setattr(ccmusic, "_fit_guide_accompaniment_alignment", lambda _guide, _accompaniment: guide_to_accompaniment)
    monkeypatch.setattr(
        ccmusic,
        "_estimate_direct_vocal_accompaniment_offset",
        lambda _vocal, _accompaniment, center: {"offset_sec": center + 0.04, "score": 0.6},
    )

    alignment = ccmusic._estimate_ccmusic_alignment(
        SimpleNamespace(),
        guide_path=Path("guide.wav"),
        vocal_path=Path("tuned-vocal.wav"),
        accompaniment_path=Path("accompaniment.wav"),
    )

    assert alignment["guide_zero"]["score_quarter"] == pytest.approx(39.3333333333)
    assert alignment["score_to_accompaniment"]["offset_sec"] == pytest.approx(-0.5)
    assert alignment["placement"]["vocal_mix_offset_sec"] == pytest.approx(28.5)
    assert alignment["placement"]["confidence"]["level"] == "high"
    assert alignment["placement"]["used_vocal_to_guide_latency"] is False
    assert alignment["placement"]["latency_application"].endswith("not_applied_to_mix_placement")
