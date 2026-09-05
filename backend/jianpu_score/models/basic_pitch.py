"""Adapter that keeps Basic Pitch in its dedicated subprocess environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..analysis import probe_audio, resolve_ffmpeg, resolve_ffprobe
from ..domain import NoteEvent
from .adapter import EngineExecutionError, EngineUnavailableError


ROOT = Path(__file__).resolve().parents[3]


def basic_pitch_python() -> Path:
    configured = os.environ.get("JIANPU_BASIC_PITCH_PYTHON")
    candidate = Path(configured) if configured else ROOT / ".venv-model-basic-pitch" / "Scripts" / "python.exe"
    if not candidate.exists():
        raise EngineUnavailableError(f"Basic Pitch environment is missing: {candidate}")
    return candidate


def extract_basic_pitch(
    audio_path: str | Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.7,
    enforce_upload_size: bool = True,
) -> list[NoteEvent]:
    """Call Basic Pitch in a child process and validate its JSON result."""

    audio = Path(audio_path).resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    probe_audio(audio, enforce_upload_size=enforce_upload_size)
    cache_dir = ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="basic-pitch-", dir=os.fspath(cache_dir)) as temp:
        output = Path(temp) / "events.json"
        command = [
            os.fspath(basic_pitch_python()),
            os.fspath(Path(__file__).with_name("basic_pitch_worker.py")),
            "--audio",
            os.fspath(audio),
            "--output",
            os.fspath(output),
            "--onset-threshold",
            str(onset_threshold),
            "--frame-threshold",
            str(frame_threshold),
            "--minimum-note-length",
            str(minimum_note_length),
        ]
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        ffmpeg = resolve_ffmpeg()
        ffprobe = resolve_ffprobe()
        if ffmpeg is not None:
            environment["PATH"] = os.fspath(ffmpeg.parent) + os.pathsep + environment.get("PATH", "")
            environment["JIANPU_FFMPEG"] = os.fspath(ffmpeg)
        if ffprobe is not None:
            environment["JIANPU_FFPROBE"] = os.fspath(ffprobe)
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
        if result.returncode:
            raise EngineExecutionError(
                f"Basic Pitch subprocess failed ({result.returncode}): {result.stderr[-3000:]}"
            )
        if not output.is_file():
            raise EngineExecutionError(f"Basic Pitch produced no event file: {result.stdout}")
        try:
            raw = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EngineExecutionError(f"Basic Pitch event file is unreadable: {output}: {exc}") from exc
    return [NoteEvent.model_validate(item) for item in raw]
