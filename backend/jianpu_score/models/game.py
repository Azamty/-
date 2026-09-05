"""GAME singing voice adapter running in its dedicated CPU environment."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from ..analysis import probe_audio, resolve_ffmpeg, resolve_ffprobe
from ..domain import NoteEvent
from .adapter import EngineExecutionError, EngineResult, EngineUnavailableError


ROOT = Path(__file__).resolve().parents[3]
GAME_ROOT = ROOT / "vendor" / "GAME-1.0.3"
DEFAULT_GAME_MODEL = ROOT / ".cache" / "models" / "game" / "GAME-1.0-small" / "model.pt"


def game_python() -> Path:
    configured = os.environ.get("JIANPU_GAME_PYTHON")
    candidate = Path(configured) if configured else ROOT / ".venv-model-game" / "Scripts" / "python.exe"
    if not candidate.exists():
        raise EngineUnavailableError(f"GAME environment is missing: {candidate}")
    return candidate.resolve()


def game_model_path() -> Path:
    configured = os.environ.get("JIANPU_GAME_MODEL")
    candidate = Path(configured) if configured else DEFAULT_GAME_MODEL
    if not candidate.is_file():
        raise EngineUnavailableError(f"GAME checkpoint is missing: {candidate}")
    return candidate.resolve()


def game_config_path(model_path: str | Path | None = None) -> Path:
    model = Path(model_path).resolve() if model_path is not None else game_model_path()
    candidate = model.with_name("config.yaml")
    if not candidate.is_file():
        raise EngineUnavailableError(f"GAME config is missing beside checkpoint: {candidate}")
    return candidate


def load_game_language_map(model_path: str | Path | None = None) -> dict[str, int]:
    """Read language IDs from the selected weight bundle, never from a constant."""

    model = Path(model_path).resolve() if model_path is not None else game_model_path()
    language_path = model.with_name("lang_map.json")
    if not language_path.is_file():
        raise EngineUnavailableError(f"GAME language map is missing beside checkpoint: {language_path}")
    try:
        raw = json.loads(language_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError(f"GAME language map is unreadable: {language_path}: {exc}") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) for key, value in raw.items()):
        raise EngineUnavailableError(f"GAME language map has invalid entries: {language_path}")
    return {key.lower(): int(value) for key, value in raw.items()}


def _language_request(language: str | None, lang_map: dict[str, int]) -> tuple[str | None, str, int | None]:
    normalized = (language or "mixed").strip().lower()
    if normalized in {"mixed", "multi", "multilingual", "auto"}:
        # GAME's omitted --language path is its documented language-agnostic
        # default.  We deliberately do not invent a language ID for a mix.
        return None, "language-agnostic GAME default (no language ID guessed)", None
    if normalized not in lang_map:
        supported = ", ".join(sorted(lang_map))
        raise EngineUnavailableError(
            f"GAME language {language!r} is absent from the selected weight map; "
            f"choose one of {supported} or mixed (mixed omits --language)"
        )
    return normalized, f"weight lang_map.json entry {normalized}={lang_map[normalized]}", lang_map[normalized]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    ffmpeg = resolve_ffmpeg()
    ffprobe = resolve_ffprobe()
    if ffmpeg is not None:
        environment["PATH"] = os.fspath(ffmpeg.parent) + os.pathsep + environment.get("PATH", "")
        environment["JIANPU_FFMPEG"] = os.fspath(ffmpeg)
    if ffprobe is not None:
        environment["JIANPU_FFPROBE"] = os.fspath(ffprobe)
    return environment


def extract_game(
    audio_path: str | Path,
    *,
    language: str = "mixed",
    stem_id: str | None = None,
    model_path: str | Path | None = None,
    enforce_upload_size: bool = True,
) -> EngineResult:
    """Run GAME and normalize its raw CSV onset/offset/floating pitch output."""

    audio = Path(audio_path).resolve()
    probe_audio(audio, enforce_upload_size=enforce_upload_size)
    if not GAME_ROOT.is_dir() or not (GAME_ROOT / "infer.py").is_file():
        raise EngineUnavailableError(f"GAME source tree is missing: {GAME_ROOT}")
    python = game_python()
    model = Path(model_path).resolve() if model_path is not None else game_model_path()
    if not model.is_file():
        raise EngineUnavailableError(f"GAME checkpoint is missing: {model}")
    config = game_config_path(model)
    lang_map = load_game_language_map(model)
    language_arg, language_strategy, language_id = _language_request(language, lang_map)

    cache_dir = ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="game-", dir=os.fspath(cache_dir)) as temp_dir:
        output_dir = Path(temp_dir)
        command = [
            os.fspath(python),
            os.fspath(GAME_ROOT / "infer.py"),
            "extract",
            os.fspath(audio),
            "-m",
            os.fspath(model),
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--output-formats",
            "csv",
            "--pitch-format",
            "number",
            "--output-dir",
            os.fspath(output_dir),
        ]
        if language_arg is not None:
            command.extend(["--language", language_arg])
        result = subprocess.run(
            command,
            cwd=GAME_ROOT,
            env=_environment(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise EngineExecutionError(
                f"GAME subprocess failed ({result.returncode}): {result.stderr[-4000:]}"
            )
        csv_files = sorted(output_dir.glob("*.csv"))
        if not csv_files:
            raise EngineExecutionError(
                f"GAME produced no CSV note output for {audio.name}: {result.stdout[-2000:]}"
            )
        csv_path = csv_files[0]
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise EngineExecutionError(f"GAME CSV cannot be read: {csv_path}: {exc}") from exc

    metadata_base = {
        "engine": "game",
        "environment": os.fspath(python),
        "model": os.fspath(model),
        "config": os.fspath(config),
        "language": language_arg or "mixed",
        "language_id": language_id,
        "language_strategy": language_strategy,
        "raw_format": "GAME CSV onset,offset,pitch",
    }
    events: list[NoteEvent] = []
    for row in rows:
        try:
            start = float(row["onset"])
            end = float(row["offset"])
            raw_pitch = float(row["pitch"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (start, end, raw_pitch)) or end <= start:
            continue
        midi = int(round(raw_pitch))
        if not 0 <= midi <= 127:
            continue
        event_metadata = {**metadata_base, "raw_pitch": raw_pitch}
        if stem_id is not None:
            event_metadata["stem_id"] = stem_id
        start = max(0.0, start)
        if end <= start:
            continue
        events.append(
            NoteEvent(
                start_sec=start,
                end_sec=end,
                midi=midi,
                confidence=None,
                raw_pitch=raw_pitch,
                stem_id=stem_id,
                source="game",
                metadata=event_metadata,
            )
        )
    events.sort(key=lambda event: (event.start_sec, event.end_sec, event.midi))
    warnings: list[str] = []
    if language_arg is None:
        warnings.append("GAME 使用语言无关的默认路径处理混合语言，未猜测语言 ID")
    return EngineResult(
        events=events,
        engine="game",
        model=os.fspath(model),
        metadata={**metadata_base, "event_count": len(events)},
        warnings=warnings,
    )
