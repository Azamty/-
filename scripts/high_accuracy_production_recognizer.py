"""Run one production high accuracy recognition route in an isolated child.

The batch parent deliberately does not import any model runtime.  This worker
coordinates the already pinned model adapters and writes one normalized payload
for both benchmark score chains:

* instrumental: MuScriptor on the original mix, then BeatNet on the original
  mix once;
* vocal: Demucs on the original mix, GAME on the vocals stem, GAME cleanup,
  then BeatNet on the original mix once.

The parent owns this process tree and can terminate it on timeout.  The worker
therefore keeps model-specific subprocesses in the same tree and never starts
background work.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MUSCRIPTOR_PYTHON = ROOT / ".venv-model-muscriptor" / "Scripts" / "python.exe"
MUSCRIPTOR_SCRIPT = ROOT / "scripts" / "run_muscriptor_job.py"
MUSCRIPTOR_BENCHMARK_SEED = 20260907
PRODUCTION_RECOGNIZER_VERSION = "1.1"
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cached_muscriptor_model() -> Path | None:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--MuScriptor--muscriptor-medium" / "snapshots"
    candidates = [candidate / "model.safetensors" for candidate in cache.glob("*")]
    files = [candidate for candidate in candidates if candidate.is_file()]
    return max(files, key=lambda candidate: candidate.stat().st_mtime) if files else None


def _run_muscriptor(audio: Path, destination: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode the complete original mix with the pinned MuScriptor worker."""

    if not MUSCRIPTOR_PYTHON.is_file():
        raise RuntimeError(f"MuScriptor environment is missing: {MUSCRIPTOR_PYTHON}")
    if not MUSCRIPTOR_SCRIPT.is_file():
        raise RuntimeError(f"MuScriptor worker is missing: {MUSCRIPTOR_SCRIPT}")
    model_root = destination / "muscriptor"
    model_root.mkdir(parents=True, exist_ok=True)
    progress = model_root / "progress.json"
    command = [
        os.fspath(MUSCRIPTOR_PYTHON),
        os.fspath(MUSCRIPTOR_SCRIPT),
        "--audio",
        os.fspath(audio),
        "--output",
        os.fspath(model_root),
        "--progress",
        os.fspath(progress),
        "--seed",
        str(MUSCRIPTOR_BENCHMARK_SEED),
    ]
    model = _cached_muscriptor_model()
    if model is not None:
        command.extend(("--model", os.fspath(model)))
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    (model_root / "worker.log").write_text(log, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"MuScriptor worker failed ({result.returncode}): {log[-4000:]}")
    recognition_path = model_root / "recognition.json"
    if not recognition_path.is_file():
        raise RuntimeError("MuScriptor worker completed without recognition.json")
    recognition = json.loads(recognition_path.read_text(encoding="utf-8"))
    if not isinstance(recognition, Mapping) or not isinstance(recognition.get("notes"), list):
        raise RuntimeError("MuScriptor recognition payload is malformed")
    notes: list[dict[str, Any]] = []
    for index, item in enumerate(recognition["notes"]):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"MuScriptor note {index} is not an object")
        start = item.get("start_sec")
        end = item.get("end_sec")
        pitch = item.get("pitch", item.get("midi"))
        if start is None or end is None or pitch is None:
            raise RuntimeError(f"MuScriptor note {index} lacks start/end/pitch")
        notes.append(
            {
                "start_sec": float(start),
                "end_sec": float(end),
                "midi": int(pitch),
                "confidence": item.get("confidence"),
                "velocity": item.get("velocity"),
                "raw_pitch": item.get("raw_pitch"),
                "voice_id": str(item.get("instrument_group", "voice-0")),
                "source": "muscriptor",
                "instrument_group": item.get("instrument_group"),
                "is_drum": bool(item.get("is_drum", False)),
                "program": item.get("program"),
            }
        )
    if not notes:
        raise RuntimeError("MuScriptor returned no note events")
    reproducibility = None
    metadata = recognition.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("reproducibility")
        if isinstance(value, Mapping):
            reproducibility = dict(value)
    return notes, {
        "engine": "muscriptor",
        "model": recognition.get("model", "medium"),
        "worker": os.fspath(MUSCRIPTOR_SCRIPT),
        "recognition_relative": recognition_path.relative_to(destination).as_posix(),
        "route_input": "original_mix",
        "reproducibility": reproducibility,
    }


def _event_records(events: Iterable[Any], *, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        payload["source"] = source
        payload["midi"] = int(payload["midi"])
        records.append(payload)
    records.sort(key=lambda item: (float(item["start_sec"]), int(item["midi"]), float(item["end_sec"])))
    return records


def _run_vocal(audio: Path, destination: Path, *, demucs_model: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Separate vocals, run GAME, and retain the raw events for cleanup."""

    from backend.jianpu_score.models.demucs import separate_htdemucs
    from backend.jianpu_score.models.game import extract_game

    separation_root = destination / "demucs"
    stems = separate_htdemucs(audio, separation_root, model=demucs_model)
    vocal_path = stems["vocals"]
    game_result = extract_game(vocal_path, language="mixed", stem_id="vocals", enforce_upload_size=False)
    if not game_result.events:
        raise RuntimeError("GAME returned no note events")
    return (
        _event_records(game_result.events, source="game"),
        {
            "engine": "game",
            "model": game_result.model,
            "demucs_model": demucs_model or "htdemucs",
            "route_input": "demucs_vocals",
            "vocal_stem": os.fspath(vocal_path),
            "game_metadata": game_result.metadata,
        },
    )


def _analysis_payload(analysis: Any) -> dict[str, Any]:
    metadata = dict(analysis.metadata)
    return {
        "sample_rate": int(analysis.sample_rate),
        "duration_sec": float(analysis.duration_sec),
        "bpm": float(analysis.bpm),
        "time_signature": str(analysis.time_signature),
        "key": str(analysis.key),
        "warnings": list(analysis.warnings),
        "metadata": metadata,
    }


def recognize(
    audio: str | Path,
    *,
    source_kind: str,
    output: str | Path,
    demucs_model: str | None = None,
) -> dict[str, Any]:
    """Run one route and return the normalized production raw payload."""

    source = Path(audio).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"production recognizer input is unavailable: {source}")
    if source_kind not in {"instrumental", "vocal"}:
        raise ValueError("source_kind must be instrumental or vocal")
    destination.mkdir(parents=True, exist_ok=True)

    if source_kind == "instrumental":
        notes, route = _run_muscriptor(source, destination)
        cleanup_report: Mapping[str, Any] | None = None
        raw_game_events: list[dict[str, Any]] | None = None
        onset_evidence: Sequence[float] = [float(item["start_sec"]) for item in notes]
    else:
        raw_notes, route = _run_vocal(source, destination, demucs_model=demucs_model)
        notes = raw_notes
        raw_game_events = raw_notes
        onset_evidence = [float(item["start_sec"]) for item in raw_notes]
        cleanup_report = None

    # This is the single BeatNet call for the case.  It always receives the
    # original mix, even when note extraction used a stem.
    from backend.jianpu_score.analysis import analyze_audio

    _samples, analysis = analyze_audio(source, source_onsets=onset_evidence)

    if source_kind == "vocal":
        from backend.jianpu_score.domain import NoteEvent
        from backend.jianpu_score.vocal_cleanup import clean_vocal_events

        events = [NoteEvent.model_validate(item) for item in notes]
        cleanup = clean_vocal_events(
            events,
            bpm=analysis.bpm,
            beat_context={
                "bpm": analysis.bpm,
                "beat_times": list(analysis.beat_times),
                "beat_grid": dict(analysis.metadata.get("beat_grid") or {}),
            },
        )
        cleanup_report = cleanup.report
        notes = _event_records(cleanup.events, source="game-cleaned")
        route = {**route, "cleanup": "game-vocal-cleanup", "raw_event_count": len(raw_game_events or []), "cleaned_event_count": len(notes)}

    beat_grid = analysis.metadata.get("beat_grid")
    if not isinstance(beat_grid, Mapping):
        raise RuntimeError("BeatNet analysis did not produce a beat_grid object")
    provenance = {
        "evaluation_scope": "production_end_to_end",
        "model_output": True,
        "source_audio": os.fspath(source),
        "beat_source": "original_mix",
        "beat_engine": "beatnet",
        "beat_independent_of_reference": True,
        "route": route,
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "recognizer_version": PRODUCTION_RECOGNIZER_VERSION,
        "source": "production_audio_model",
        "model_output": True,
        "source_kind": source_kind,
        "notes": notes,
        "analysis": _analysis_payload(analysis),
        "beat_grid": dict(beat_grid),
        "provenance": provenance,
    }
    if cleanup_report is not None:
        payload["vocal_cleanup"] = cleanup_report
    _json_write(destination / "production_raw.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("instrumental", "vocal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demucs-model")
    args = parser.parse_args(argv)
    recognize(args.audio, source_kind=args.source_kind, output=args.output, demucs_model=args.demucs_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
