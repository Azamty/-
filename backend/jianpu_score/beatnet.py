"""Strict adapter for the isolated BeatNet 1.1.3 offline/DBN runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .beat_grid import build_beat_grid
from .high_accuracy import BEATNET_VERSION, ROOT, resolve_beatnet_python


class BeatNetUnavailableError(RuntimeError):
    """Raised when the required BeatNet runtime cannot produce a grid."""


class BeatNetExecutionError(RuntimeError):
    """Raised when BeatNet returns malformed output or exits unsuccessfully."""


def run_beatnet(
    audio_path: str | Path,
    *,
    model: int = 1,
    device: str = "cpu",
    timeout_sec: int = 600,
) -> list[dict[str, Any]]:
    """Run the pinned isolated worker and return raw beat observations.

    The API process never imports BeatNet's Python 3.9 dependency set.  A
    missing or failed worker is an explicit error; this adapter never falls
    back to librosa's old uniform-grid detector.
    """

    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise BeatNetExecutionError(f"BeatNet audio input does not exist: {source}")
    python = resolve_beatnet_python()
    if not python.is_file():
        raise BeatNetUnavailableError(
            f"BeatNet {BEATNET_VERSION} environment is unavailable: {python}; "
            "run scripts/install_high_accuracy.ps1"
        )
    worker = ROOT / "scripts" / "run_beatnet_worker.py"
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [os.fspath(python), os.fspath(worker), "--audio", os.fspath(source), "--model", str(model), "--device", device],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BeatNetExecutionError(f"BeatNet timed out after {timeout_sec}s") from exc
    except OSError as exc:
        raise BeatNetExecutionError(f"BeatNet worker could not start: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise BeatNetExecutionError(f"BeatNet worker failed ({result.returncode}): {detail[-4000:]}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise BeatNetExecutionError("BeatNet worker returned no valid JSON") from exc
    if payload.get("engine") != "beatnet" or payload.get("version") != BEATNET_VERSION:
        raise BeatNetExecutionError(f"BeatNet worker contract mismatch: {payload}")
    beats = payload.get("beats")
    if not isinstance(beats, list) or len(beats) < 2:
        raise BeatNetExecutionError("BeatNet worker returned fewer than two beats")
    return beats


def analyze_with_beatnet(
    audio_path: str | Path,
    *,
    duration_sec: float | None = None,
    bpm_override: float | None = None,
    time_signature_override: str | None = None,
    source_onsets: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    model: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Decode the original audio and build the durable beat-grid payload."""

    observations = run_beatnet(audio_path, model=model, device=device)
    grid = build_beat_grid(
        observations,
        duration_sec=duration_sec,
        manual_bpm=bpm_override,
        manual_time_signature=time_signature_override,
        source_onsets=source_onsets,
    )
    grid["beatnet"] = {
        "version": BEATNET_VERSION,
        "model": model,
        "mode": "offline",
        "inference": "DBN",
        "confidence_is_derived": True,
    }
    return grid
