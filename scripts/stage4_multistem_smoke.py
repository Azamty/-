"""Run one real Basic Pitch multi-stem API job and record its routing evidence.

The reference is intentionally short. This smoke test exercises the normal
HTTP upload and durable worker path with the real Demucs and Basic Pitch
subprocesses; it is not a transcription quality benchmark.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.job_manager import utc_now


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "review" / "scale_reference.wav"
EVIDENCE_ROOT = ROOT / "artifacts" / "review" / "stage4-multistem-smoke"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    run_root = EVIDENCE_ROOT / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)
    app = create_app(jobs_root=run_root / "jobs")
    evidence: dict[str, Any] = {
        "reference": str(REFERENCE),
        "reference_bytes": REFERENCE.stat().st_size,
        "engine": "basic-pitch",
        "voice_mode": "polyphonic",
        "source_kind": "instrumental",
        "real_model": True,
        "started_at": utc_now(),
    }
    snapshots: list[dict[str, Any]] = []

    with TestClient(app) as client:
        with REFERENCE.open("rb") as handle:
            response = client.post(
                "/api/jobs",
                data={
                    "engine": "basic-pitch",
                    "voice_mode": "polyphonic",
                    "source_kind": "instrumental",
                    "language": "mixed",
                    "bpm": "80",
                    "key": "C",
                    "time_signature": "4/4",
                    "title": "Stage 4 multi-stem smoke",
                    # Deliberately omit the internal separate flag. The API
                    # must infer it from the source and voice choices.
                },
                files={"file": ("scale_reference.wav", handle, "audio/wav")},
            )
        response.raise_for_status()
        job = response.json()
        job_id = str(job["id"])
        evidence["job_id"] = job_id
        evidence["created_status"] = job["status"]
        evidence["effective_options"] = job["options"]

        deadline = time.monotonic() + 240
        last_phase: str | None = None
        while time.monotonic() < deadline:
            status_response = client.get(f"/api/jobs/{job_id}")
            status_response.raise_for_status()
            job = status_response.json()
            if job["phase"] != last_phase or job["status"] in {"completed", "failed", "interrupted"}:
                snapshots.append(
                    {
                        "status": job["status"],
                        "phase": job["phase"],
                        "phase_label": job["phase_label"],
                    }
                )
                last_phase = job["phase"]
            if job["status"] in {"completed", "failed", "interrupted"}:
                break
            time.sleep(1.0)
        else:
            raise TimeoutError(f"multi-stem API job did not finish: {job_id}")

        evidence["snapshots"] = snapshots
        evidence["final_status"] = job["status"]
        evidence["warnings"] = job.get("warnings", [])
        if job["status"] != "completed":
            _write_json(run_root / "summary.json", evidence)
            raise RuntimeError(f"multi-stem API job failed: {job.get('error')}")

        artifacts_response = client.get(f"/api/jobs/{job_id}/artifacts")
        artifacts_response.raise_for_status()
        artifacts = artifacts_response.json()["artifacts"]
        evidence["artifact_ids"] = [item["artifact_id"] for item in artifacts]
        evidence["stem_artifacts"] = [
            {"artifact_id": item["artifact_id"], "stem_id": item.get("stem_id"), "label": item["label"]}
            for item in artifacts
            if item["kind"] in {"stem_svg", "stem_midi"}
        ]

        score_response = client.get(f"/api/jobs/{job_id}/score")
        score_response.raise_for_status()
        score = score_response.json()
        _write_json(run_root / "score.json", score)
        metadata = score.get("metadata", {})
        evidence["source_stems"] = metadata.get("source_stems", [])
        evidence["engine_by_stem"] = metadata.get("engine_by_stem", {})
        evidence["routed_stems"] = sorted(evidence["engine_by_stem"])
        evidence["voice_stems"] = sorted({voice.get("stem_id") for voice in score.get("voices", []) if voice.get("stem_id")})
        evidence["note_count"] = sum(len(voice.get("events", [])) for voice in score.get("voices", []))

        expected_sources = {"bass", "other"}
        if set(evidence["routed_stems"]) != expected_sources:
            raise AssertionError(f"instrumental polyphonic route did not preserve bass/other sources: {evidence['routed_stems']}")
        if not evidence["voice_stems"] or "mixed" in evidence["voice_stems"]:
            raise AssertionError(f"score voices were incorrectly routed through mixed: {evidence['voice_stems']}")
        if any(item.get("stem_id") == "mixed" for item in evidence["stem_artifacts"]):
            raise AssertionError("multi-stem artifacts contain a mixed stem")
        if not any(item.get("kind") == "stem_svg" for item in artifacts):
            raise AssertionError("multi-stem job did not register a split SVG")

        evidence["finished_at"] = utc_now()

    _write_json(run_root / "summary.json", evidence)
    print(json.dumps({"run_root": str(run_root), **evidence}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
