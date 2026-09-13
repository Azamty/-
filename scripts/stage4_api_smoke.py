"""Run a real local API job and record stage4 acceptance evidence.

This script intentionally uses the prepared reference WAV and the real Basic
Pitch subprocess.  It does not replace transcription with a fixture or mock;
the small lifecycle checks are separate from the real model job.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient

from backend.app import MULTIPART_OVERHEAD_BYTES, create_app
from backend.jianpu_score.analysis import MAX_AUDIO_BYTES
from backend.job_manager import JobManager, utc_now


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "review" / "scale_reference.wav"
EVIDENCE_ROOT = ROOT / "artifacts" / "review" / "stage4-api-smoke"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    run_root = EVIDENCE_ROOT / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)
    jobs_root = run_root / "jobs"
    app = create_app(jobs_root=jobs_root)
    snapshots: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "reference": str(REFERENCE),
        "reference_bytes": REFERENCE.stat().st_size,
        "engine": "basic-pitch",
        "real_model": True,
        "started_at": utc_now(),
    }

    with TestClient(app) as client:
        capabilities = client.get("/api/capabilities")
        capabilities.raise_for_status()
        _json(run_root / "capabilities.json", capabilities.json())

        with REFERENCE.open("rb") as handle:
            response = client.post(
                "/api/jobs",
                data={
                    "engine": "basic-pitch",
                    "voice_mode": "monophonic",
                    "source_kind": "instrumental",
                    "language": "mixed",
                    "bpm": "80",
                    "key": "C",
                    "time_signature": "4/4",
                    "title": "Stage 4 real model smoke",
                },
                files={"file": ("scale_reference.wav", handle, "audio/wav")},
            )
        response.raise_for_status()
        job = response.json()
        job_id = str(job["id"])
        evidence["job_id"] = job_id
        evidence["created_status"] = job["status"]

        deadline = time.monotonic() + 180
        last_phase = None
        while time.monotonic() < deadline:
            status_response = client.get(f"/api/jobs/{job_id}")
            status_response.raise_for_status()
            status = status_response.json()
            if status["phase"] != last_phase or status["status"] in {"completed", "failed", "interrupted"}:
                snapshots.append(
                    {
                        "status": status["status"],
                        "phase": status["phase"],
                        "phase_label": status["phase_label"],
                        "attempt": status["attempt"],
                    }
                )
                last_phase = status["phase"]
            if status["status"] in {"completed", "failed", "interrupted"}:
                job = status
                break
            time.sleep(1.0)
        else:
            raise TimeoutError(f"real API job did not finish: {job_id}")

        evidence["snapshots"] = snapshots
        evidence["final_status"] = job["status"]
        evidence["final_phase"] = job["phase"]
        evidence["warnings"] = job.get("warnings", [])
        if job["status"] != "completed":
            _json(run_root / "summary.json", evidence)
            raise RuntimeError(f"real API job failed: {job.get('error')}")

        artifacts_response = client.get(f"/api/jobs/{job_id}/artifacts")
        artifacts_response.raise_for_status()
        artifacts = artifacts_response.json()["artifacts"]
        evidence["artifact_ids"] = [item["artifact_id"] for item in artifacts]
        evidence["artifact_kinds"] = {item["artifact_id"]: item["kind"] for item in artifacts}
        required = {"score-svg-1", "score-midi", "score-json", "score-svg-zip", "job-log"}
        missing = sorted(required - set(evidence["artifact_ids"]))
        if missing:
            raise AssertionError(f"missing registered artifacts: {missing}")

        for artifact_id in ("score-svg-1", "score-midi", "score-svg-zip", "job-log"):
            download = client.get(f"/api/jobs/{job_id}/artifacts/{artifact_id}")
            download.raise_for_status()
            (run_root / f"{artifact_id}.download").write_bytes(download.content)
            if not download.content:
                raise AssertionError(f"empty artifact: {artifact_id}")

        score_response = client.get(f"/api/jobs/{job_id}/score")
        score_response.raise_for_status()
        score = score_response.json()
        _json(run_root / "score.json", score)
        evidence["score_note_count"] = sum(len(voice.get("events", [])) for voice in score.get("voices", []))
        evidence["score_voice_count"] = len(score.get("voices", []))
        evidence["finished_at"] = utc_now()

        # The API route rejects traversal and only exposes persisted IDs.
        evidence["traversal_status"] = client.get(
            f"/api/jobs/{job_id}/artifacts/%2e%2e%2fjob.json"
        ).status_code
        if evidence["traversal_status"] != 404:
            raise AssertionError("artifact traversal was not rejected")

        # Exercise the request-body size gate without sending another model
        # job. The multipart envelope is allowed a small fixed overhead, so
        # this body is rejected before Starlette parses the upload.
        oversized = b"0" * (MAX_AUDIO_BYTES + MULTIPART_OVERHEAD_BYTES + 1)
        too_large = client.post(
            "/api/jobs",
            files={"file": ("oversized.wav", oversized, "audio/wav")},
        )
        evidence["oversized_status"] = too_large.status_code
        evidence["oversized_code"] = too_large.json().get("detail", {}).get("code")
        if too_large.status_code != 413 or evidence["oversized_code"] != "upload_too_large":
            raise AssertionError(f"oversized upload was not rejected: {too_large.status_code} {too_large.text}")

    # Verify the restart marker and terminal cleanup in a small isolated root;
    # these checks do not pretend to be model inference.
    lifecycle_root = run_root / "lifecycle-jobs"
    manager = JobManager(lifecycle_root)
    running_id, _ = manager.create_job(original_name="running.wav", options={})
    manager._update(running_id, status="running", phase="recognizing", started_at=utc_now())
    recovered = JobManager(lifecycle_root)
    evidence["restart_status"] = recovered.get(running_id)["status"]
    old_id, _ = recovered.create_job(original_name="expired.wav", options={})
    old_time = (datetime.now(timezone.utc).timestamp() - 25 * 3600)
    old_iso = datetime.fromtimestamp(old_time, timezone.utc).isoformat()
    recovered._update(old_id, status="failed", phase="failed", finished_at=old_iso)
    evidence["cleanup_removed"] = recovered.cleanup_expired()
    _json(run_root / "summary.json", evidence)
    print(json.dumps({"run_root": str(run_root), **evidence}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
