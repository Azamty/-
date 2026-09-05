"""Run the final full length local API acceptance job.

The input is a 180 second synthetic instrumental fixture. The test measures
pipeline scale and artifact consistency with real Demucs and Basic Pitch; it
does not claim Chinese, Japanese or mixed-language song accuracy.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
from typing import Any
import zipfile

import mido
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.job_manager import utc_now


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "artifacts" / "review" / "stage5-final" / "long_fixture.wav"
FIXTURE_META = ROOT / "artifacts" / "review" / "stage5-final" / "long_fixture.json"
EVIDENCE_ROOT = ROOT / "artifacts" / "review" / "stage5-final"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _midi_stats(content: bytes) -> dict[str, Any]:
    midi = mido.MidiFile(file=io.BytesIO(content))
    note_on_count = 0
    max_tick = 0
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += int(message.time)
            max_tick = max(max_tick, absolute)
            if message.type == "note_on" and getattr(message, "velocity", 0) > 0:
                note_on_count += 1
    return {
        "ticks_per_beat": midi.ticks_per_beat,
        "track_count": len(midi.tracks),
        "note_on_count": note_on_count,
        "max_tick": max_tick,
        "duration_sec": midi.length,
    }


def main() -> int:
    if not FIXTURE.is_file() or not FIXTURE_META.is_file():
        raise FileNotFoundError("run scripts.generate_stage5_fixture first")
    fixture_meta = json.loads(FIXTURE_META.read_text(encoding="utf-8"))
    run_root = EVIDENCE_ROOT / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=False)
    app = create_app(jobs_root=run_root / "jobs")
    evidence: dict[str, Any] = {
        "fixture": str(FIXTURE),
        "fixture_metadata": str(FIXTURE_META),
        "fixture_kind": fixture_meta.get("kind"),
        "accuracy_claim": fixture_meta.get("accuracy_claim"),
        "fixture_duration_sec": fixture_meta.get("duration_sec"),
        "fixture_bytes": FIXTURE.stat().st_size,
        "engine": "basic-pitch",
        "source_kind": "instrumental",
        "voice_mode": "monophonic",
        "real_model": True,
        "started_at": utc_now(),
    }
    snapshots: list[dict[str, Any]] = []
    wall_start = time.monotonic()

    with TestClient(app) as client:
        capabilities = client.get("/api/capabilities")
        capabilities.raise_for_status()
        _write_json(run_root / "capabilities.json", capabilities.json())

        with FIXTURE.open("rb") as handle:
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
                    "title": "Stage 5 180 second scale",
                    # Deliberately omit internal ``separate``; source routing
                    # must select Demucs other automatically.
                },
                files={"file": ("long_fixture.wav", handle, "audio/wav")},
            )
        response.raise_for_status()
        job = response.json()
        job_id = str(job["id"])
        evidence["job_id"] = job_id
        evidence["created_status"] = job["status"]
        evidence["effective_options"] = job["options"]

        deadline = time.monotonic() + 1800
        last_phase: str | None = None
        while time.monotonic() < deadline:
            status_response = client.get(f"/api/jobs/{job_id}")
            status_response.raise_for_status()
            job = status_response.json()
            if job["phase"] != last_phase or job["status"] in {"completed", "failed", "interrupted"}:
                snapshots.append(
                    {
                        "elapsed_sec": round(time.monotonic() - wall_start, 3),
                        "status": job["status"],
                        "phase": job["phase"],
                        "phase_label": job["phase_label"],
                        "attempt": job["attempt"],
                    }
                )
                last_phase = job["phase"]
            if job["status"] in {"completed", "failed", "interrupted"}:
                break
            time.sleep(3.0)
        else:
            raise TimeoutError(f"full-length API job did not finish: {job_id}")

        evidence["snapshots"] = snapshots
        evidence["final_status"] = job["status"]
        evidence["elapsed_sec"] = round(time.monotonic() - wall_start, 3)
        evidence["warnings"] = job.get("warnings", [])
        if job["status"] != "completed":
            _write_json(run_root / "summary.json", evidence)
            raise RuntimeError(f"full-length API job failed: {job.get('error')}")

        artifacts_response = client.get(f"/api/jobs/{job_id}/artifacts")
        artifacts_response.raise_for_status()
        artifacts = artifacts_response.json()["artifacts"]
        evidence["artifact_ids"] = [item["artifact_id"] for item in artifacts]
        evidence["score_svg_pages"] = sum(item["kind"] == "score_svg" for item in artifacts)
        evidence["stem_svg_pages"] = {}
        for item in artifacts:
            if item["kind"] == "stem_svg":
                evidence["stem_svg_pages"].setdefault(item.get("stem_id"), 0)
                evidence["stem_svg_pages"][item.get("stem_id")] += 1
        if evidence["score_svg_pages"] < 2:
            raise AssertionError(f"full-length score did not paginate: {evidence['score_svg_pages']} page(s)")

        downloads: dict[str, bytes] = {}
        for artifact_id in [
            "analysis-json",
            "score-json",
            "score-midi",
            "score-svg-zip",
            *[item["artifact_id"] for item in artifacts if item["kind"] in {"stem_midi", "stem_svg"}],
        ]:
            download = client.get(f"/api/jobs/{job_id}/artifacts/{artifact_id}")
            download.raise_for_status()
            downloads[artifact_id] = download.content
            (run_root / f"{artifact_id}.download").write_bytes(download.content)

        analysis = json.loads(downloads["analysis-json"].decode("utf-8"))
        score = json.loads(downloads["score-json"].decode("utf-8"))
        _write_json(run_root / "analysis.json", analysis)
        _write_json(run_root / "score.json", score)
        input_events = [item for item in analysis.get("note_events", []) if item.get("midi") is not None]
        score_notes = [
            event
            for voice in score.get("voices", [])
            for event in voice.get("events", [])
            if event.get("midi") is not None
        ]
        input_first = min((float(item["start_sec"]) for item in input_events), default=None)
        input_last = max((float(item["end_sec"]) for item in input_events), default=None)
        quarter_ticks = int(score["quarter_ticks"])
        score_bpm = float(score["bpm"])
        score_first_tick = min((int(item["start_tick"]) for item in score_notes), default=None)
        score_last_tick = max((int(item["start_tick"]) + int(item["duration_tick"]) for item in score_notes), default=None)
        evidence["analysis_duration_sec"] = analysis.get("duration_sec")
        evidence["analysis_event_count"] = len(input_events)
        evidence["input_first_event_sec"] = input_first
        evidence["input_last_event_sec"] = input_last
        evidence["score_total_ticks"] = score.get("total_ticks")
        evidence["score_note_count"] = len(score_notes)
        evidence["score_first_event_tick"] = score_first_tick
        evidence["score_last_event_tick"] = score_last_tick
        evidence["score_last_event_sec"] = (score_last_tick * 60.0 / (score_bpm * quarter_ticks)) if score_last_tick is not None else None
        evidence["engine_by_stem"] = score.get("metadata", {}).get("engine_by_stem", {})
        evidence["prepared_audio_stems"] = sorted(score.get("metadata", {}).get("prepared_audio", {}))

        fixture_duration = float(fixture_meta["duration_sec"])
        if abs(float(analysis.get("duration_sec", 0)) - fixture_duration) > 0.1:
            raise AssertionError(f"analysis duration was truncated: {analysis.get('duration_sec')}")
        if input_first is None or input_first > 1.5:
            raise AssertionError(f"first recognized event is missing or too late: {input_first}")
        if input_last is None or input_last < 177.0:
            raise AssertionError(f"tail marker was truncated: {input_last}")
        if score_last_tick is None or evidence["score_last_event_sec"] < 177.0:
            raise AssertionError(f"score tail was truncated: {evidence['score_last_event_sec']}")
        if evidence["prepared_audio_stems"] != ["other"]:
            raise AssertionError(f"unexpected default instrumental route: {evidence['prepared_audio_stems']}")

        midi_stats: dict[str, Any] = {"score-midi": _midi_stats(downloads["score-midi"])}
        for item in artifacts:
            if item["kind"] == "stem_midi":
                midi_stats[item["artifact_id"]] = _midi_stats(downloads[item["artifact_id"]])
        evidence["midi"] = midi_stats
        total_duration = float(midi_stats["score-midi"]["duration_sec"])
        evidence["midi_duration_delta_vs_analysis_sec"] = total_duration - fixture_duration
        if abs(total_duration - fixture_duration) > 1.0:
            raise AssertionError(f"total MIDI duration mismatch: {total_duration} vs {fixture_duration}")
        for artifact_id, stats in midi_stats.items():
            if artifact_id == "score-midi":
                continue
            if abs(float(stats["duration_sec"]) - total_duration) > 0.25:
                raise AssertionError(f"split MIDI duration mismatch for {artifact_id}: {stats['duration_sec']} vs {total_duration}")

        svg_files: list[str] = []
        for item in artifacts:
            if item["kind"] in {"score_svg", "stem_svg"}:
                download = client.get(f"/api/jobs/{job_id}/artifacts/{item['artifact_id']}")
                download.raise_for_status()
                target = run_root / f"{item['artifact_id']}.svg"
                target.write_bytes(download.content)
                svg_files.append(str(target))
                if not download.content.lstrip().startswith(b"<?xml") and b"<svg" not in download.content[:1000]:
                    raise AssertionError(f"artifact is not SVG: {item['artifact_id']}")
        evidence["svg_files"] = svg_files

        zip_members = sorted(zipfile.ZipFile(io.BytesIO(downloads["score-svg-zip"])).namelist())
        evidence["zip_members"] = zip_members
        if len(zip_members) < evidence["score_svg_pages"]:
            raise AssertionError("SVG ZIP omitted score pages")

    _write_json(run_root / "summary.json", evidence)
    print(json.dumps({"run_root": str(run_root), **evidence}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
