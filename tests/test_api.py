from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import UploadSizeLimitMiddleware, _effective_options, create_app
from backend.job_manager import JobManager, utc_now


def test_api_rejects_invalid_upload_and_unsafe_options(tmp_path: Path) -> None:
    app = create_app(jobs_root=tmp_path / "jobs")
    with TestClient(app) as client:
        invalid = client.post(
            "/api/jobs",
            files={"file": ("song.txt", b"not audio", "text/plain")},
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"]["code"] == "invalid_request"

        mixed_specialist = client.post(
            "/api/jobs",
            data={"engine": "specialist", "source_kind": "mixed", "voice_mode": "monophonic"},
            files={"file": ("song.wav", b"not audio", "audio/wav")},
        )
        assert mixed_specialist.status_code == 422
        assert "混合来源" in mixed_specialist.json()["detail"]["message"]

        nonfinite_bpm = client.post(
            "/api/jobs",
            data={"bpm": "NaN"},
            files={"file": ("song.wav", b"not audio", "audio/wav")},
        )
        assert nonfinite_bpm.status_code == 422
        assert "有限数字" in nonfinite_bpm.json()["detail"]["message"]


def test_effective_options_route_source_choices_to_stems() -> None:
    common = {
        "engine": "basic-pitch",
        "language": "mixed",
        "separate": False,
        "bpm": None,
        "key": None,
        "time_signature": None,
        "title": "route",
    }
    assert _effective_options(**common, voice_mode="monophonic", source_kind="vocal")["separate"] is True
    assert _effective_options(**common, voice_mode="monophonic", source_kind="instrumental")["separate"] is True
    assert _effective_options(**common, voice_mode="polyphonic", source_kind="instrumental")["separate"] is True
    assert _effective_options(**common, voice_mode="polyphonic", source_kind="mixed")["separate"] is True
    assert _effective_options(**common, voice_mode="monophonic", source_kind="mixed")["separate"] is False


def test_v2_capabilities_report_route_specific_demucs(tmp_path: Path) -> None:
    app = create_app(jobs_root=tmp_path / "jobs")
    with TestClient(app) as client:
        value = client.get("/api/capabilities").json()
    routes = value["api"]["v2"]["routes"]
    assert "use_demucs" not in value["api"]["v2"]
    assert routes["instrumental"]["use_demucs"] is False
    assert routes["vocal"]["use_demucs"] is True
    assert routes["vocal"]["separation_model"] == "htdemucs"
    assert routes["vocal"]["separation_models"]["default"] == "htdemucs"
    assert [item["id"] for item in routes["vocal"]["separation_models"]["options"]] == ["htdemucs", "htdemucs_ft"]
    assert "约慢 4 倍" in routes["vocal"]["separation_models"]["options"][1]["speed_note"]


def test_v2_rejects_unknown_separation_model_before_upload_probe(tmp_path: Path) -> None:
    app = create_app(jobs_root=tmp_path / "jobs")
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/jobs",
            data={"source_kind": "vocal", "separation_model": "unknown-model"},
            files={"file": ("voice.wav", b"not audio", "audio/wav")},
        )
    assert response.status_code == 422
    assert "htdemucs_ft" in response.json()["detail"]["message"]
    assert not list((tmp_path / "jobs").iterdir())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("key", "H", "unsupported key"), ("time_signature", "5/4", "unsupported time signature")],
)
def test_invalid_key_and_meter_are_structured_422(tmp_path: Path, field: str, value: str, message: str) -> None:
    app = create_app(jobs_root=tmp_path / "jobs")
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={field: value},
            files={"file": ("song.wav", b"not audio", "audio/wav")},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    assert message in response.json()["detail"]["message"]


def test_upload_body_limit_rejects_before_multipart_parser(tmp_path: Path) -> None:
    app = create_app(jobs_root=tmp_path / "jobs", max_upload_bytes=16, multipart_overhead_bytes=32)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"file": ("song.wav", b"x" * 128, "audio/wav")},
        )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "upload_too_large"
    assert not list((tmp_path / "jobs").iterdir())


def test_upload_body_limit_also_counts_chunked_requests_without_content_length() -> None:
    async def exercise() -> list[dict[str, object]]:
        received_by_app = False

        async def downstream(_scope: dict[str, object], receive: object, send: object) -> None:
            nonlocal received_by_app
            await receive()  # type: ignore[misc]
            received_by_app = True

        messages = [{"type": "http.request", "body": b"x" * 40, "more_body": False}]

        async def receive() -> dict[str, object]:
            return messages.pop(0)

        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        middleware = UploadSizeLimitMiddleware(downstream, max_upload_bytes=16, overhead_bytes=8)
        await middleware(
            {"type": "http", "method": "POST", "path": "/api/jobs", "headers": []},
            receive,
            send,
        )
        assert received_by_app is False
        return sent

    sent = asyncio.run(exercise())
    assert sent[0]["status"] == 413
    assert sent[1]["type"] == "http.response.body"


def test_uploading_state_is_not_retried_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    first = JobManager(root)
    job_id, input_path = first.create_job(original_name="partial.wav", options={})
    assert first.get(job_id)["status"] == "uploading"
    input_path.write_bytes(b"partial")

    recovered = JobManager(root)
    state = recovered.get(job_id)
    assert state["status"] == "interrupted"
    assert state["error"]["code"] == "upload_interrupted"
    assert state["retryable"] is False
    with pytest.raises(ValueError, match="上传未完成"):
        recovered.retry(job_id)


def test_idle_cleanup_failure_does_not_kill_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import backend.job_manager as job_manager_module
    from threading import Event

    monkeypatch.setattr(job_manager_module, "CLEANUP_INTERVAL_SECONDS", 0.01)
    manager = JobManager(tmp_path / "jobs")
    cleanup_seen = Event()
    job_seen = Event()
    cleanup_calls = 0

    def flaky_cleanup(**_kwargs: object) -> list[str]:
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_seen.set()
        if cleanup_calls == 1:
            raise OSError("simulated locked old job")
        return []

    def fake_run(_job_id: str) -> None:
        job_seen.set()

    monkeypatch.setattr(manager, "cleanup_expired", flaky_cleanup)
    monkeypatch.setattr(manager, "_run_job", fake_run)
    manager.start()
    try:
        assert cleanup_seen.wait(2.0)
        job_id, _ = manager.create_job(original_name="later.wav", options={})
        manager.enqueue(job_id)
        assert job_seen.wait(2.0)
    finally:
        manager.stop()
    assert cleanup_calls >= 1


def test_artifact_endpoint_only_serves_registered_files(tmp_path: Path) -> None:
    app = create_app(jobs_root=tmp_path / "jobs")
    manager: JobManager = app.state.jobs
    job_id, _input_path = manager.create_job(original_name="score.wav", options={})
    output = (tmp_path / "jobs" / job_id / "output").resolve()
    output.mkdir(parents=True)
    registered = output / "registered.txt"
    registered.write_text("safe", encoding="utf-8")
    artifact = manager._register(
        (tmp_path / "jobs" / job_id).resolve(),
        registered,
        artifact_id="registered",
        kind="text",
        label="registered",
        media_type="text/plain",
    )
    manager._update(job_id, status="completed", phase="completed", artifacts=[artifact])

    with TestClient(app) as client:
        response = client.get(f"/api/jobs/{job_id}/artifacts/registered")
        assert response.status_code == 200
        assert response.text == "safe"
        assert client.get(f"/api/jobs/{job_id}/artifacts/missing").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/artifacts/%2e%2e%2fjob.json").status_code == 404


def test_restart_marks_running_interrupted_and_cleanup_keeps_active_jobs(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    first = JobManager(root)
    running_id, _ = first.create_job(original_name="running.wav", options={})
    first._update(running_id, status="running", phase="recognizing", started_at=utc_now())
    queued_id, _ = first.create_job(original_name="queued.wav", options={})

    recovered = JobManager(root)
    recovered_state = recovered.get(running_id)
    assert recovered_state["status"] == "interrupted"
    assert recovered_state["error"]["code"] == "interrupted"
    old_id, _ = recovered.create_job(original_name="old.wav", options={})
    old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    recovered._update(old_id, status="failed", phase="failed", finished_at=old_time)
    removed = recovered.cleanup_expired()
    assert old_id in removed
    assert not (root / old_id).exists()
    assert (root / queued_id).is_dir()
    assert (root / running_id).is_dir()

    # The persisted state remains valid JSON after recovery and cleanup.
    json.loads((root / running_id / "job.json").read_text(encoding="utf-8"))


def test_create_app_defers_recovery_until_lifespan_starts(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    seed = JobManager(root)
    job_id, _ = seed.create_job(original_name="still-running.wav", options={})
    seed._update(job_id, status="running", phase="separating", started_at=utc_now())
    metadata = root / job_id / "job.json"
    before_import = metadata.read_bytes()

    app = create_app(jobs_root=root)
    assert metadata.read_bytes() == before_import
    assert app.state.jobs.get(job_id)["status"] == "running"

    with TestClient(app):
        recovered = app.state.jobs.get(job_id)
        assert recovered["status"] == "interrupted"
        assert recovered["error"]["code"] == "interrupted"
