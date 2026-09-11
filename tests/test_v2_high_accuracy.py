from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.high_accuracy import (
    resolve_musescore,
    resolve_notation_python,
)
from backend.jianpu_score.high_accuracy_service import (
    SERVICE_SCHEMA_VERSION,
    HighAccuracyBuildResult,
    HighAccuracyServiceError,
    ServiceArtifact,
)
from backend.jianpu_score.models.adapter import EngineResult
from backend.jianpu_score.quantize import _build_beat_mapper
from backend.jianpu_score.render import JIANPU, LILYPOND
from backend.job_manager import JobManager
from backend.muscriptor_v2 import stable_track_id
from backend.v2_job_manager import V2JobService

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"
REAL_READY = (
    resolve_musescore() is not None
    and resolve_notation_python().is_file()
    and PROFILE.is_file()
    and JIANPU.is_file()
    and LILYPOND.is_file()
)


def test_current_v2_web_service_does_not_call_legacy_uniform_quantizer() -> None:
    """The webpage's /api/v2 path must stay on the high-accuracy service.

    The legacy /api/jobs pipeline remains available for one rollback cycle, so
    this assertion is deliberately scoped to V2JobService rather than the
    whole repository.
    """

    source = inspect.getsource(V2JobService)
    assert "quantize_events" not in source
    assert "run_pipeline" not in source


class FakeHighAccuracyService:
    calls: ClassVar[list[dict[str, Any]]] = []
    failures: ClassVar[set[str]] = set()

    def build(self, **kwargs: Any) -> HighAccuracyBuildResult:
        instrument_id = str(kwargs["instrument_id"])
        output_dir = Path(kwargs["output_dir"]).resolve()
        variant = str(kwargs["variant"])
        self.calls.append({"instrument_id": instrument_id, "variant": variant, "is_drum": bool(kwargs["is_drum"])})
        if instrument_id in self.failures:
            raise RuntimeError(f"fixture failure for {instrument_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe = instrument_id.replace("-", "_")
        names = [
            f"{safe}.{variant}.note-events.json",
            f"{safe}.{variant}.performance.mid",
            f"{safe}.{variant}.performance.metadata.json",
            f"{safe}.{variant}.notated.musicxml",
            f"{safe}.score.json",
            f"{safe}.alignment_report.json",
            f"{safe}.score.jly",
            f"{safe}.score.ly",
            f"{safe}.score-1.svg",
            f"{safe}.score.long.svg",
            f"{safe}.score.mid",
        ]
        paths: list[Path] = []
        for name in names:
            path = output_dir / name
            path.write_text("{}" if path.suffix == ".json" else "fixture", encoding="utf-8")
            paths.append(path)
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": SERVICE_SCHEMA_VERSION,
                    "instrument_id": instrument_id,
                    "variant": variant,
                    "status": "completed",
                }
            ),
            encoding="utf-8",
        )
        artifacts = tuple(
            ServiceArtifact(
                artifact_id=path.name,
                kind="fixture",
                path=path,
                relative_path=path.relative_to(output_dir).as_posix(),
                sha256="fixture",
                bytes=path.stat().st_size,
            )
            for path in paths
        )
        return HighAccuracyBuildResult(
            instrument_id=instrument_id,
            title=str(kwargs["title"]),
            variant=variant,
            program=int(kwargs["program"]),
            is_drum=bool(kwargs["is_drum"]),
            status="completed",
            jianpu_status="completed",
            output_dir=output_dir,
            manifest_path=manifest,
            artifacts=artifacts,
            performance_metadata={"note_count": len(tuple(kwargs["events"]))},
        )


def _analysis() -> MusicAnalysis:
    beat_times = [0.0, 0.5, 1.0, 1.5]
    return MusicAnalysis(
        sample_rate=16_000,
        duration_sec=2.0,
        bpm=120,
        key="C",
        time_signature="4/4",
        beat_times=beat_times,
        metadata={
            "beat_engine": "beatnet",
            "beatnet_version": "1.1.3",
            "beat_source": "beatnet",
            "beat_grid": {"beats": [{"index": i, "time_sec": value, "downbeat": i == 0} for i, value in enumerate(beat_times)]},
        },
    )


def _instrumental_fixture(manager: JobManager, tmp_path: Path, *, two_tracks: bool = False) -> tuple[str, str, list[str]]:
    job_id, input_path = manager.create_v2_job(original_name="fixture.wav", source_kind="instrumental", title="fixture")
    input_path.write_bytes(b"fixture")
    track_specs = [("acoustic_guitar", 24), ("violin", 40)] if two_tracks else [("acoustic_guitar", 24)]
    tracks = [
        {
            "track_id": stable_track_id(group, program, False),
            "instrument_group": group,
            "program": program,
            "is_drum": False,
            "label_zh": group,
        }
        for group, program in track_specs
    ]
    notes = [
        {"instrument_group": group, "program": program, "is_drum": False, "pitch": 60 + index, "start_sec": 0.1, "end_sec": 0.4, "velocity": None}
        for index, (group, program) in enumerate(track_specs)
    ]
    analysis = _analysis()
    manager._update(
        job_id,
        status="selection_ready",
        phase="selection_ready",
        progress=1.0,
        v2={
            "stage": "selection_ready",
            "source_kind": "instrumental",
            "tracks": tracks,
            "notes": notes,
            "analysis": {"bpm": 120, "key": "C", "time_signature": "4/4"},
            "selection_revision": 0,
            "selection": None,
            "selection_history": [],
            "score_refusal": None,
        },
    )
    analysis_path = tmp_path / "jobs" / job_id / "output" / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    state = manager._read(job_id)
    state["v2"]["analysis_relative"] = "output/analysis.json"
    manager._write(state)
    ids = [str(item["track_id"]) for item in tracks]
    return job_id, str(input_path), ids


def test_recognition_retry_uses_new_attempt_and_preserves_raw_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="fixture.wav", source_kind="instrumental", title="fixture")
    input_path.write_bytes(b"fixture")
    manager.enqueue(job_id)
    analysis_calls = 0

    def fake_analysis(*_args: Any, **_kwargs: Any) -> tuple[None, MusicAnalysis]:
        nonlocal analysis_calls
        analysis_calls += 1
        if analysis_calls == 2:
            raise RuntimeError("BeatNet fixture failure")
        return None, _analysis()

    def fake_child(command: list[str], _job_id: str, progress_path: Path | None = None) -> int:
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        attempt = output.name
        (output / "original.mid").write_bytes(attempt.encode("ascii"))
        (output / "recognition.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "engine": "muscriptor",
                    "model": "fixture",
                    "notes": [
                        {
                            "instrument_group": "acoustic_guitar",
                            "program": 24,
                            "is_drum": False,
                            "pitch": 60,
                            "start_sec": 0.0,
                            "end_sec": 0.5,
                        }
                    ],
                    "tracks": [
                        {
                            "track_id": stable_track_id("acoustic_guitar", 24, False),
                            "instrument_group": "acoustic_guitar",
                            "program": 24,
                            "is_drum": False,
                            "label_zh": "原声吉他",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (output / "muscriptor-worker.log").write_text(attempt, encoding="utf-8")
        if progress_path is not None:
            progress_path.write_text(json.dumps({"completed": 1, "total": 1, "status": "selection_ready"}), encoding="utf-8")
        return 0

    monkeypatch.setattr("backend.v2_job_manager.analyze_audio", fake_analysis)
    monkeypatch.setattr(manager, "_run_isolated_child", fake_child)
    manager._run_job(job_id)
    first_root = tmp_path / "jobs" / job_id / "output" / "v2-recognition" / "attempt-0001"
    first_files = [first_root / name for name in ("recognition.json", "original.mid", "muscriptor-worker.log")]
    first_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first_files}
    assert manager._read(job_id)["status"] == "selection_ready"

    state = manager._read(job_id)
    state["status"] = "failed"
    state["phase"] = "failed"
    state["error"] = {"code": "fixture_retry", "message": "retry recognition"}
    state["v2"]["stage"] = "recognize"
    manager._write(state)

    manager.retry(job_id)
    manager._run_job(job_id)
    second_root = tmp_path / "jobs" / job_id / "output" / "v2-recognition" / "attempt-0002"
    assert all(path.is_file() for path in first_files)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == first_hashes[path.name] for path in first_files)
    assert (second_root / "recognition.json").is_file()
    state = manager._read(job_id)
    assert state["status"] == "failed"
    historical = next(item for item in state["artifacts"] if item["artifact_id"] == "v2-recognition-json-attempt-0001")
    historical_path, _ = manager.artifact_path(job_id, historical["artifact_id"])
    assert historical_path == first_root / "recognition.json"

    second_files = [second_root / name for name in ("recognition.json", "original.mid", "muscriptor-worker.log")]
    second_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in second_files}
    manager.retry(job_id)
    manager._run_job(job_id)
    third_root = tmp_path / "jobs" / job_id / "output" / "v2-recognition" / "attempt-0003"
    assert all(path.is_file() for path in second_files)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == second_hashes[path.name] for path in second_files)
    assert (third_root / "recognition.json").is_file()
    state = manager._read(job_id)
    assert state["status"] == "selection_ready"
    recognition = next(item for item in state["artifacts"] if item["artifact_id"] == "v2-recognition-json")
    original_midi = next(item for item in state["artifacts"] if item["artifact_id"] == "v2-original-midi")
    assert "attempt-0003" in recognition["relative_path"]
    assert "attempt-0003" in original_midi["relative_path"]
    assert any(item["artifact_id"] == "v2-recognition-json-attempt-0001" for item in state["artifacts"])
    assert analysis_calls == 3


def test_instrumental_export_registers_service_outputs_and_preserves_override_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path)
    manager.select_v2(job_id, ids)
    state = manager._read(job_id)
    state.update({"status": "running", "phase": "rendering"})
    manager._write(state)

    manager.v2._run_instrumental_export(job_id)

    result = manager._read(job_id)
    assert result["status"] == "completed"
    assert FakeHighAccuracyService.calls == [{"instrument_id": ids[0], "variant": "instrument-part", "is_drum": False}]
    kinds = {item["kind"] for item in result["artifacts"]}
    assert {"instrument_score_midi", "instrument_performance_midi", "instrument_musicxml", "instrument_score_json", "high_accuracy_manifest"} <= kinds
    assert result["summary"]["overrides"]["sources"]["bpm"] == "beatnet"
    assert result["summary"]["overrides"]["bpm_manual"] is False
    assert any(item["kind"] == "instrument_preview_midi" for item in result["artifacts"])
    assert not any(item["kind"] == "instrument_midi" for item in result["artifacts"])


def test_instrumental_export_passes_full_timeline_when_persisted_analysis_has_no_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[tuple[str, MusicAnalysis, tuple[NoteEvent, ...]]] = []

    class CapturingFakeHighAccuracyService(FakeHighAccuracyService):
        def build(self, **kwargs: Any) -> HighAccuracyBuildResult:
            captured.append(
                (
                    str(kwargs["instrument_id"]),
                    kwargs["analysis"],
                    tuple(kwargs["events"]),
                )
            )
            return super().build(**kwargs)

    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", CapturingFakeHighAccuracyService)
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path, two_tracks=True)
    state = manager._read(job_id)
    state["v2"]["notes"][0].update({"start_sec": 0.0, "end_sec": 0.2})
    state["v2"]["notes"][1].update({"start_sec": 1.25, "end_sec": 1.5})
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    full_analysis = _analysis().model_copy(
        update={
            "duration_sec": 2.5,
            "beat_times": beat_times,
            "metadata": {
                "beat_engine": "beatnet",
                "beatnet_version": "1.1.3",
                "beat_source": "beatnet",
                "beat_grid": {
                    "beats": [
                        {"index": index, "time_sec": time, "downbeat": index == 1}
                        for index, time in enumerate(beat_times)
                    ],
                    "mapping": {"score_origin": {"downbeat_index": 1, "downbeat_sec": 0.5}},
                },
            },
        }
    )
    assert full_analysis.note_events == []
    analysis_path = tmp_path / "jobs" / job_id / "output" / "analysis.json"
    analysis_path.write_text(full_analysis.model_dump_json(), encoding="utf-8")
    state["v2"]["analysis_relative"] = "output/analysis.json"
    manager._write(state)
    manager.select_v2(job_id, ids)
    running = manager._read(job_id)
    running.update({"status": "running", "phase": "rendering"})
    manager._write(running)

    manager.v2._run_instrumental_export(job_id)

    assert [item[0] for item in captured] == ids
    origins = []
    for _track_id, analysis, events in captured:
        assert len(analysis.metadata["shared_timeline_event_bounds"]) == 2
        assert analysis.metadata["shared_timeline_scope"] == "persisted_full_analysis"
        origins.append(_build_beat_mapper(analysis, list(events)).score_origin)
    assert {origin["timeline_scope"] for origin in origins} == {"persisted_full_analysis"}
    assert all(origin["origin_shift_beats"] == pytest.approx(0.0) for origin in origins)
    assert all(origin["timeline_offset_beats"] == pytest.approx(0.0) for origin in origins)


def test_analysis_for_events_ignores_reserved_shared_timeline_metadata_injection() -> None:
    base = _analysis()
    event = NoteEvent(start_sec=0.25, end_sec=0.5, midi=60)
    metadata_extra = {
        "shared_timeline_event_bounds": [{"start_sec": -9.0, "end_sec": -8.0}],
        "shared_timeline_scope": "spoofed",
    }

    derived = V2JobService._analysis_for_events(
        base,
        [event],
        bpm=120,
        key="C",
        time_signature="4/4",
        bpm_manual=False,
        key_manual=False,
        time_signature_manual=False,
        metadata_extra=metadata_extra,
    )
    assert "shared_timeline_event_bounds" not in derived.metadata
    assert "shared_timeline_scope" not in derived.metadata

    base_with_notes = base.model_copy(update={"note_events": [NoteEvent(start_sec=1.0, end_sec=1.5, midi=64)]})
    derived_from_base = V2JobService._analysis_for_events(
        base_with_notes,
        [event],
        bpm=120,
        key="C",
        time_signature="4/4",
        bpm_manual=False,
        key_manual=False,
        time_signature_manual=False,
        metadata_extra=metadata_extra,
    )
    assert derived_from_base.metadata["shared_timeline_event_bounds"] == [
        {"start_sec": 1.0, "end_sec": 1.5}
    ]
    assert derived_from_base.metadata["shared_timeline_scope"] == "persisted_full_analysis"


def test_instrumental_partial_failure_keeps_success_and_records_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path, two_tracks=True)
    FakeHighAccuracyService.failures = {ids[0]}
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)
    manager.select_v2(job_id, ids)
    manager._run_job(job_id)

    result = manager._read(job_id)
    failures = result["v2"]["track_failures"]
    assert result["status"] == "completed"
    assert failures[0]["track_id"] == ids[0]
    assert failures[0]["stage"] == "service"
    assert ids[1] in result["summary"]["successful_pitched_track_ids"]
    assert any(item["artifact_id"].startswith(f"v2-selection-r1-{ids[1]}-") for item in result["artifacts"])


def test_main_melody_is_attempted_from_selected_tracks_when_one_part_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path, two_tracks=True)
    FakeHighAccuracyService.failures = {ids[0]}
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)

    manager.select_v2(job_id, ids, merge_main_melody=True)
    manager._run_job(job_id)

    result = manager._read(job_id)
    assert result["status"] == "completed"
    assert result["v2"]["score_refusal"] is None
    assert any(item["track_id"] == ids[0] for item in result["v2"]["track_failures"])
    assert result["summary"]["main_melody_score_artifact_ids"]
    assert any(item["kind"] == "main_melody_selection" for item in result["artifacts"])
    assert any(call["instrument_id"] == "main-melody" for call in FakeHighAccuracyService.calls)


def test_all_parts_and_main_melody_failure_remains_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path)
    FakeHighAccuracyService.failures = {ids[0], "main-melody"}
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)

    manager.select_v2(job_id, ids, merge_main_melody=True)
    manager._run_job(job_id)

    result = manager._read(job_id)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "high_accuracy_all_tracks_failed"
    assert result["v2"]["score_refusal"]["code"] == "all_pitched_tracks_failed"
    assert {item["track_id"] for item in result["v2"]["track_failures"]} >= {ids[0], "main-melody"}
    assert any(item["kind"] == "main_melody_selection" for item in result["artifacts"])


def test_instrumental_all_pitched_failure_is_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path)
    FakeHighAccuracyService.failures = {ids[0]}
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)
    manager.select_v2(job_id, ids)
    manager._run_job(job_id)
    result = manager._read(job_id)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "high_accuracy_all_tracks_failed"
    assert result["v2"]["score_refusal"]["code"] == "all_pitched_tracks_failed"


def test_retrying_failed_revision_preserves_previous_revision_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path)

    manager.select_v2(job_id, ids)
    manager._run_job(job_id)
    first = manager._read(job_id)
    first_score = next(item for item in first["artifacts"] if item["artifact_id"] == f"v2-selection-r1-{ids[0]}-score-json")
    first_path, _ = manager.artifact_path(job_id, first_score["artifact_id"])
    first_hash = hashlib.sha256(first_path.read_bytes()).hexdigest()

    FakeHighAccuracyService.failures = {ids[0]}
    manager.select_v2(job_id, ids)
    manager._run_job(job_id)
    failed = manager._read(job_id)
    assert failed["status"] == "failed"
    assert any(item["artifact_id"] == first_score["artifact_id"] for item in failed["artifacts"])

    manager.retry(job_id)
    manager._run_job(job_id)
    retried = manager._read(job_id)
    retained = next(item for item in retried["artifacts"] if item["artifact_id"] == first_score["artifact_id"])
    retained_path, _ = manager.artifact_path(job_id, retained["artifact_id"])
    assert retained_path == first_path
    assert hashlib.sha256(retained_path.read_bytes()).hexdigest() == first_hash
    assert any(item["artifact_id"].startswith("v2-selection-r2-") for item in retried["artifacts"])


def test_instrumental_high_accuracy_error_registers_failure_manifest_and_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class PartialFailureService(FakeHighAccuracyService):
        failing_id: ClassVar[str] = ""

        def build(self, **kwargs: Any) -> HighAccuracyBuildResult:
            instrument_id = str(kwargs["instrument_id"])
            if instrument_id != self.failing_id:
                return super().build(**kwargs)
            output_dir = Path(kwargs["output_dir"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = output_dir / "manifest.json"
            log = output_dir / "musescore_import.log"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": SERVICE_SCHEMA_VERSION,
                        "instrument_id": instrument_id,
                        "variant": str(kwargs["variant"]),
                        "status": "failed",
                    }
                ),
                encoding="utf-8",
            )
            log.write_text("fixture cause", encoding="utf-8")
            raise HighAccuracyServiceError(
                "fixture high accuracy failure",
                instrument_id=instrument_id,
                stage="musescore_import",
                cause="fixture cause",
                manifest_path=manifest,
                log_path=log,
            )

    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path, two_tracks=True)
    PartialFailureService.failing_id = ids[0]
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", PartialFailureService)
    manager.select_v2(job_id, ids)
    manager._run_job(job_id)

    result = manager._read(job_id)
    failure = next(item for item in result["v2"]["track_failures"] if item["track_id"] == ids[0])
    assert result["status"] == "completed"
    assert failure["stage"] == "musescore_import"
    assert failure["error"] == "fixture cause"
    assert any(item["artifact_id"] == f"v2-selection-r1-{ids[0]}-manifest" for item in result["artifacts"])
    assert any(item["artifact_id"] == f"v2-selection-r1-{ids[0]}-failure-log" for item in result["artifacts"])
    json.dumps(result, ensure_ascii=False)


def test_drum_only_export_stays_midi_only_without_notation_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeHighAccuracyService.calls = []
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="drums.wav", source_kind="instrumental", title="drums")
    input_path.write_bytes(b"fixture")
    track_id = stable_track_id("drums", 128, True)
    manager._update(
        job_id,
        status="selection_ready",
        phase="selection_ready",
        progress=1.0,
        v2={
            "stage": "selection_ready",
            "source_kind": "instrumental",
            "tracks": [{"track_id": track_id, "instrument_group": "drums", "program": 128, "is_drum": True, "label_zh": "鼓组"}],
            "notes": [{"instrument_group": "drums", "program": 128, "is_drum": True, "pitch": 36, "start_sec": 0.1, "end_sec": 0.2}],
            "analysis": {"bpm": 120, "key": "C", "time_signature": "4/4"},
            "selection_revision": 0,
            "selection": None,
            "selection_history": [],
            "score_refusal": None,
        },
    )
    manager.select_v2(job_id, [track_id])
    state = manager._read(job_id)
    state.update({"status": "running", "phase": "rendering"})
    manager._write(state)

    manager.v2._run_instrumental_export(job_id)

    result = manager._read(job_id)
    assert result["status"] == "completed"
    assert result["v2"]["score_refusal"]["code"] == "no_pitched_tracks"
    assert FakeHighAccuracyService.calls == []
    assert any(item["artifact_id"] == "v2-selection-r1-midi" for item in result["artifacts"])


def test_v2_high_accuracy_registers_natural_page_numbers(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, _input = manager.create_v2_job(original_name="fixture.wav", source_kind="instrumental", title="fixture")
    output_dir = tmp_path / "jobs" / job_id / "output" / "track"
    output_dir.mkdir(parents=True)
    paths = []
    for page in range(12, 0, -1):
        path = output_dir / f"piano.score-{page}.svg"
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><text>page {page}</text></svg>',
            encoding="utf-8",
        )
        paths.append(path)
    manifest = output_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = HighAccuracyBuildResult(
        instrument_id="piano",
        title="钢琴",
        variant="instrument-part",
        program=0,
        is_drum=False,
        status="completed",
        jianpu_status="completed",
        output_dir=output_dir,
        manifest_path=manifest,
        artifacts=tuple(
            ServiceArtifact(
                artifact_id=path.name,
                kind="svg",
                path=path,
                relative_path=path.relative_to(output_dir).as_posix(),
                sha256="fixture",
                bytes=path.stat().st_size,
            )
            for path in paths
        ),
        performance_metadata={},
    )

    registered, score_ids = manager.v2._register_high_accuracy_result(
        job_id,
        result,
        prefix="v2-selection-r1-piano",
        label="钢琴",
        stem_id="piano",
        family="instrument",
    )
    pages = [item for item in registered if item["kind"] == "instrument_score_svg"]
    assert [item["page"] for item in pages] == list(range(1, 13))
    assert [item["filename"] for item in pages] == [f"piano.score-{page}.svg" for page in range(1, 13)]
    assert score_ids == [f"v2-selection-r1-piano-score-svg-{page}" for page in range(1, 13)]


def test_vocal_generation_persists_raw_cleanup_and_service_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeHighAccuracyService.calls = []
    FakeHighAccuracyService.failures = set()
    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", FakeHighAccuracyService)
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="voice.wav", source_kind="vocal", title="voice")
    input_path.write_bytes(b"fixture")
    job_dir = tmp_path / "jobs" / job_id
    vocal_path = job_dir / "output" / "vocal-prep" / "vocals.wav"
    vocal_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_path.write_bytes(b"stem")
    analysis = _analysis()
    analysis_path = job_dir / "output" / "vocal-prep" / "original-analysis.json"
    analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    state = manager._read(job_id)
    state.update(
        {
            "status": "running",
            "phase": "recognizing",
            "v2": {
                **state["v2"],
                "stage": "vocal_generate",
                "separation": {
                    "prepared_vocals_relative": "output/vocal-prep/vocals.wav",
                    "analysis_relative": "output/vocal-prep/original-analysis.json",
                },
            },
        }
    )
    manager._write(state)
    raw = [
        NoteEvent(start_sec=0.0, end_sec=0.20, midi=60, raw_pitch=60.02, source="game"),
        NoteEvent(start_sec=0.205, end_sec=0.50, midi=60, raw_pitch=59.98, source="game"),
    ]
    monkeypatch.setattr(
        "backend.v2_job_manager.run_engine",
        lambda *_args, **_kwargs: EngineResult(events=raw, engine="game", model="GAME"),
    )

    manager.v2._run_vocal_generation(job_id)

    result = manager._read(job_id)
    assert result["status"] == "completed"
    assert FakeHighAccuracyService.calls == [{"instrument_id": "vocals", "variant": "game-cleaned", "is_drum": False}]
    artifact_ids = {item["artifact_id"] for item in result["artifacts"]}
    assert {"v2-vocal-game-raw-notes", "v2-vocal-game-cleaned-notes", "v2-vocal-game-cleanup-report", "v2-vocal-analysis-cleaned"} <= artifact_ids
    assert any(item["kind"] == "vocal_score_midi" for item in result["artifacts"])
    assert result["v2"]["generation"]["analysis_reused_from_original"] is True
    assert result["summary"]["raw_note_count"] == 2
    assert result["summary"]["note_count"] == 1


def test_vocal_retry_keeps_previous_attempt_diagnostics_downloadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="voice.wav", source_kind="vocal", title="voice")
    input_path.write_bytes(b"fixture")
    job_dir = tmp_path / "jobs" / job_id
    vocal_path = job_dir / "output" / "vocal-prep" / "vocals.wav"
    vocal_path.parent.mkdir(parents=True, exist_ok=True)
    vocal_path.write_bytes(b"stem")
    analysis_path = job_dir / "output" / "vocal-prep" / "original-analysis.json"
    analysis_path.write_text(_analysis().model_dump_json(), encoding="utf-8")
    state = manager._read(job_id)
    state.update(
        {
            "status": "queued",
            "phase": "queued",
            "v2": {
                **state["v2"],
                "stage": "vocal_generate",
                "separation": {
                    "prepared_vocals_relative": "output/vocal-prep/vocals.wav",
                    "analysis_relative": "output/vocal-prep/original-analysis.json",
                },
            },
        }
    )
    manager._write(state)
    raw = [NoteEvent(start_sec=0.0, end_sec=0.20, midi=60, raw_pitch=60.02, source="game")]
    monkeypatch.setattr(
        "backend.v2_job_manager.run_engine",
        lambda *_args, **_kwargs: EngineResult(events=raw, engine="game", model="GAME"),
    )

    class RetryVocalService:
        calls = 0

        def build(self, **kwargs: Any) -> HighAccuracyBuildResult:
            type(self).calls += 1
            if type(self).calls == 1:
                output_dir = Path(kwargs["output_dir"]).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest = output_dir / "manifest.json"
                log = output_dir / "musescore_import.log"
                manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": SERVICE_SCHEMA_VERSION,
                            "instrument_id": "vocals",
                            "variant": "game-cleaned",
                            "status": "failed",
                        }
                    ),
                    encoding="utf-8",
                )
                log.write_text("first attempt cause", encoding="utf-8")
                raise HighAccuracyServiceError(
                    "first attempt failed",
                    instrument_id="vocals",
                    stage="musescore_import",
                    cause="first attempt cause",
                    manifest_path=manifest,
                    log_path=log,
                )
            return FakeHighAccuracyService().build(**kwargs)

    monkeypatch.setattr("backend.v2_job_manager.HighAccuracyArtifactService", RetryVocalService)
    manager._run_job(job_id)
    failed = manager._read(job_id)
    assert failed["status"] == "failed"
    assert failed["error"]["stage"] == "musescore_import"
    assert failed["error"]["instrument_id"] == "vocals"

    manager.retry(job_id)
    manager._run_job(job_id)
    result = manager._read(job_id)
    artifact_ids = [str(item["artifact_id"]) for item in result["artifacts"]]
    assert result["status"] == "completed"
    assert len(artifact_ids) == len(set(artifact_ids))
    for old_id in (
        "v2-vocal-game-raw-notes-attempt-0001",
        "v2-vocal-high-accuracy-manifest-attempt-0001",
        "v2-vocal-high-accuracy-failure-log-attempt-0001",
    ):
        old_path, _ = manager.artifact_path(job_id, old_id)
        assert old_path.is_file()
    latest_path, _ = manager.artifact_path(job_id, "v2-vocal-game-raw-notes")
    assert latest_path.is_file()
    assert any("attempt-0002" in str(item["relative_path"]) for item in result["artifacts"] if item["artifact_id"] == "v2-vocal-game-raw-notes")


@pytest.mark.skipif(not REAL_READY, reason="pinned high-accuracy toolchain is unavailable")
def test_v2_instrumental_real_musescore_stage56_class_bundle(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, _input, ids = _instrumental_fixture(manager, tmp_path)
    state = manager._read(job_id)
    state["v2"]["notes"][0].update({"start_sec": 0.0, "end_sec": 0.5})
    manager._write(state)
    manager.select_v2(job_id, ids)
    state = manager._read(job_id)
    state.update({"status": "running", "phase": "rendering"})
    manager._write(state)

    manager.v2._run_instrumental_export(job_id)

    result = manager._read(job_id)
    assert result["status"] == "completed"
    assert any(item["kind"] == "instrument_score_midi" for item in result["artifacts"])
    assert any(item["kind"] == "instrument_musicxml" for item in result["artifacts"])
    assert any(item["kind"] == "instrument_score_svg_long" for item in result["artifacts"])
