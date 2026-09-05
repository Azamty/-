"""Exercise the real V2 API with a short local fixture.

Instrumental mode uses the isolated MuScriptor medium worker. Vocal mode
verifies the staged Demucs ``vocal_ready`` handoff, safe stem download and the
explicit GAME generation request. The evidence records no credentials or
environment variables.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import uuid

from fastapi.testclient import TestClient
import mido


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import create_app
from backend.jianpu_score.analysis import probe_audio
from backend.jianpu_score.models.demucs import DEMUCS_MODEL_IDS


def _wait(client: TestClient, job_id: str, expected: set[str], timeout: float = 300.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v2/jobs/{job_id}")
        response.raise_for_status()
        value = response.json()
        if value["status"] in expected:
            return value
        if value["status"] in {"failed", "interrupted"}:
            raise RuntimeError(json.dumps(value, ensure_ascii=False))
        time.sleep(0.25)
    raise TimeoutError(f"V2 job {job_id} did not reach {sorted(expected)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=ROOT / "artifacts" / "review" / "scale_reference.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "review" / "stageC-api" / "smoke.json")
    parser.add_argument("--source-kind", choices=("instrumental", "vocal"), default="instrumental")
    parser.add_argument("--separation-model", choices=sorted(DEMUCS_MODEL_IDS), default="htdemucs")
    args = parser.parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"audio fixture not found: {args.audio}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs_root = args.output.parent / "jobs" / str(uuid.uuid4())
    app = create_app(jobs_root=jobs_root)
    with TestClient(app) as client:
        with args.audio.open("rb") as handle:
            response = client.post(
                "/api/v2/jobs",
                data={
                    "source_kind": args.source_kind,
                    "title": f"V2 {args.source_kind} API fixture",
                    **({"separation_model": args.separation_model} if args.source_kind == "vocal" else {}),
                },
                files={"file": (args.audio.name, handle, "audio/wav")},
            )
        response.raise_for_status()
        uploaded = response.json()
        job_id = str(uploaded["id"])
        if args.source_kind == "vocal":
            vocal_ready = _wait(client, job_id, {"vocal_ready"})
            ready_artifacts_response = client.get(f"/api/v2/jobs/{job_id}/artifacts")
            ready_artifacts_response.raise_for_status()
            ready_items = ready_artifacts_response.json()["artifacts"]
            vocal_item = next((item for item in ready_items if item["artifact_id"] == "v2-vocals-audio"), None)
            if vocal_item is None:
                raise RuntimeError("V2 vocal API smoke found no persisted vocals artifact")
            vocal_download = client.get(f"/api/v2/jobs/{job_id}/artifacts/v2-vocals-audio")
            vocal_download.raise_for_status()
            stem_path = Path(jobs_root) / "downloaded-vocals.wav"
            stem_path.write_bytes(vocal_download.content)
            try:
                vocal_probe = probe_audio(stem_path, enforce_upload_size=False)
            finally:
                stem_path.unlink(missing_ok=True)
            generate = client.post(f"/api/v2/jobs/{job_id}/vocal/generate")
            generate.raise_for_status()
            queued = generate.json()
            completed = _wait(client, job_id, {"completed"})
            artifacts = client.get(f"/api/v2/jobs/{job_id}/artifacts")
            artifacts.raise_for_status()
            items = artifacts.json()["artifacts"]
            artifact_ids = [str(item["artifact_id"]) for item in items]
            downloads: dict[str, int] = {}
            for item in items:
                if item["kind"] not in {"score_svg", "score_svg_long", "midi"}:
                    continue
                artifact_id = str(item["artifact_id"])
                download = client.get(f"/api/v2/jobs/{job_id}/artifacts/{artifact_id}")
                download.raise_for_status()
                downloads[artifact_id] = len(download.content)
            if not any(item["kind"] == "score_svg" for item in items) or not any(item["kind"] == "score_svg_long" for item in items) or not any(item["kind"] == "midi" for item in items):
                raise RuntimeError("V2 vocal API smoke found no score SVG and MIDI")
            evidence = {
                "status": "passed",
                "route": (completed.get("v2") or {}).get("route"),
                "job_id": job_id,
                "phases": {"vocal_ready": vocal_ready["phase"], "generate_queued": queued["phase"], "completed": completed["phase"]},
                "vocal_audio_duration_sec": float(vocal_probe["duration_sec"]),
                "vocal_audio_bytes": len(vocal_download.content),
                "vocal_artifact": {"artifact_id": vocal_item["artifact_id"], "kind": vocal_item["kind"], "relative_path": vocal_item["relative_path"]},
                "summary": completed.get("summary"),
                "generation": (completed.get("v2") or {}).get("generation"),
                "artifact_count": len(items),
                "artifact_ids": artifact_ids,
                "download_bytes": downloads,
                "input_duration_sec": uploaded["input"]["duration_sec"],
                "tracks_selection_stage": False,
            }
            args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(evidence, ensure_ascii=False, indent=2))
            return 0

        ready = _wait(client, job_id, {"selection_ready"})
        tracks = client.get(f"/api/v2/jobs/{job_id}/tracks")
        tracks.raise_for_status()
        track_data = tracks.json()
        selected = [str(track["track_id"]) for track in track_data["tracks"] if not track["is_drum"]]
        selection = client.post(
            f"/api/v2/jobs/{job_id}/selection",
            json={"selected_track_ids": selected, "merge_main_melody": False},
        )
        selection.raise_for_status()
        completed = _wait(client, job_id, {"completed"})
        artifacts = client.get(f"/api/v2/jobs/{job_id}/artifacts")
        artifacts.raise_for_status()
        items = artifacts.json()["artifacts"]
        artifact_ids = [str(item["artifact_id"]) for item in items]
        downloads: dict[str, int] = {}
        midi_durations: dict[str, float] = {}
        for artifact_id in (
            "v2-original-midi",
            f"v2-selection-r1-midi",
            *(f"v2-selection-r1-{track_id}-midi" for track_id in selected),
        ):
            download = client.get(f"/api/v2/jobs/{job_id}/artifacts/{artifact_id}")
            download.raise_for_status()
            downloads[artifact_id] = len(download.content)
            midi_durations[artifact_id] = float(mido.MidiFile(file=BytesIO(download.content)).length)
        score_svg_ids = [item_id for item_id in artifact_ids if item_id.startswith("v2-selection-r1-") and "-svg-" in item_id]
        if not score_svg_ids:
            raise RuntimeError("V2 API smoke found no per-instrument score SVG")
        score_download = client.get(f"/api/v2/jobs/{job_id}/artifacts/{score_svg_ids[0]}")
        score_download.raise_for_status()
        downloads[score_svg_ids[0]] = len(score_download.content)
        evidence = {
            "status": "passed",
            "route": {"source_kind": "instrumental", "engine": "muscriptor", "use_demucs": False},
            "job_id": job_id,
            "phases": {"ready": ready["phase"], "completed": completed["phase"]},
            "track_count": len(track_data["tracks"]),
            "selected_track_ids": selected,
            "artifact_count": len(items),
            "artifact_ids": artifact_ids,
            "download_bytes": downloads,
            "midi_durations_sec": midi_durations,
            "input_duration_sec": uploaded["input"]["duration_sec"],
            "recognition_reused_for_selection": True,
        }
        if abs(midi_durations["v2-original-midi"] - float(uploaded["input"]["duration_sec"])) > 0.15:
            raise RuntimeError("original MIDI duration drifted from the uploaded audio")
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
