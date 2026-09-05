"""Machine-readable engine availability and routing capabilities."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Any

from .analysis import resolve_ffmpeg, resolve_ffprobe
from .models.game import DEFAULT_GAME_MODEL, GAME_ROOT, game_config_path, game_model_path, game_python, load_game_language_map
from .models.tsumugi import STEM_MODEL_TYPES, TSUMUGI_CHECKPOINTS, TSUMUGI_ROOT, tsumugi_python


ROOT = Path(__file__).resolve().parents[2]


def _missing(paths: list[Path]) -> list[str]:
    return [os.fspath(path) for path in paths if not path.exists()]


def _environment_status(path: Path, *, label: str) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, f"{label} environment is missing: {path}"
    return True, None


def _module_probe(path: Path, modules: tuple[str, ...], *, label: str) -> tuple[bool, str | None]:
    """Check a dedicated runtime without importing its heavyweight packages here.

    ``find_spec`` runs in the target interpreter and catches half-installed
    environments (a python.exe alone is not an available engine).  The result
    is cached by :func:`get_capabilities`, so the HTTP capabilities endpoint
    does not launch model runtimes on every request.
    """

    ready, reason = _environment_status(path, label=label)
    if not ready:
        return ready, reason
    probe = (
        "import importlib.util, sys; "
        "missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; "
        "print('missing:' + ','.join(missing) if missing else 'ok'); "
        "raise SystemExit(3 if missing else 0)"
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [os.fspath(path), "-c", probe, *modules],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{label} runtime probe failed: {exc}"
    if result.returncode:
        detail = (result.stdout or result.stderr).strip().splitlines()[-1:]
        return False, f"{label} runtime is incomplete: {detail[0] if detail else 'module probe failed'}"
    return True, None


def _command_status(
    path: Path | None,
    *,
    label: str,
    arguments: tuple[str, ...] = ("--version",),
) -> tuple[bool, str | None]:
    """Check a renderer executable's version without rendering a score."""

    if path is None or not path.is_file():
        return False, f"{label} executable is missing"
    try:
        result = subprocess.run(
            [os.fspath(path), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{label} readiness probe failed: {exc}"
    if result.returncode:
        return False, f"{label} returned exit code {result.returncode}"
    return True, None


@lru_cache(maxsize=1)
def _cached_capabilities() -> dict[str, Any]:
    """Build capabilities once per server process.

    Capability checks are intentionally lightweight and cached.  A restart is
    the explicit refresh boundary after installing or removing a model env.
    """

    return _compute_capabilities()


def _game_status() -> tuple[bool, str | None, dict[str, Any]]:
    configured_model = os.environ.get("JIANPU_GAME_MODEL")
    details: dict[str, Any] = {
        "environment": None,
        "source": os.fspath(GAME_ROOT),
        "model": os.fspath(Path(configured_model).resolve()) if configured_model else os.fspath(DEFAULT_GAME_MODEL),
        "config": None,
    }
    try:
        python = game_python()
    except Exception as exc:
        return False, str(exc), details
    runtime_available, runtime_reason = _module_probe(
        python,
        ("torch", "lightning", "omegaconf", "librosa", "yaml", "click"),
        label="GAME",
    )
    if not runtime_available:
        return False, runtime_reason, details
    details["environment"] = os.fspath(python)
    try:
        model = game_model_path()
        details["model"] = os.fspath(model)
        details["config"] = os.fspath(game_config_path(model))
        language_map = load_game_language_map(model)
    except Exception as exc:
        return False, str(exc), details
    missing = _missing([GAME_ROOT / "infer.py"])
    if missing:
        return False, f"GAME files are missing: {', '.join(missing)}", details
    details["language_map"] = language_map
    return True, None, details


def _tsumugi_status() -> tuple[bool, str | None, dict[str, Any]]:
    details: dict[str, Any] = {
        "environment": None,
        "source": os.fspath(TSUMUGI_ROOT),
        "checkpoints": {key: os.fspath(path) for key, path in TSUMUGI_CHECKPOINTS.items()},
        "stem_models": dict(STEM_MODEL_TYPES),
    }
    try:
        python = tsumugi_python()
    except Exception as exc:
        return False, str(exc), details
    runtime_available, runtime_reason = _module_probe(
        python,
        ("torch", "torchaudio", "soundfile"),
        label="tsumugi",
    )
    if not runtime_available:
        return False, runtime_reason, details
    details["environment"] = os.fspath(python)
    missing = _missing([TSUMUGI_ROOT / "infer.py", *{path for path in TSUMUGI_CHECKPOINTS.values()}])
    if missing:
        return False, f"tsumugi files are missing: {', '.join(missing)}", details
    return True, None, details


def _muscriptor_status() -> tuple[bool, str | None, dict[str, Any]]:
    """Probe the isolated MuScriptor runtime without importing it in the API."""

    python = ROOT / ".venv-model-muscriptor" / "Scripts" / "python.exe"
    details: dict[str, Any] = {
        "environment": os.fspath(python),
        "model": "medium",
        "license": "CC BY-NC 4.0 (non-commercial use)",
        "cuda": {"available": False, "version": None, "device": None},
        "use_demucs": False,
    }
    runtime_available, runtime_reason = _module_probe(
        python,
        ("muscriptor", "torch", "safetensors"),
        label="MuScriptor",
    )
    if not runtime_available:
        return False, runtime_reason, details
    probe = (
        "import json, torch; "
        "print(json.dumps({'cuda_available':bool(torch.cuda.is_available()),"
        "'cuda_version':torch.version.cuda,"
        "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [os.fspath(python), "-c", probe],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        if result.returncode:
            return False, "MuScriptor CUDA probe failed", details
        import json

        value = json.loads(result.stdout.strip().splitlines()[-1])
        details["cuda"] = {
            "available": bool(value.get("cuda_available")),
            "version": value.get("cuda_version"),
            "device": value.get("device"),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return False, f"MuScriptor CUDA probe failed: {exc}", details
    return True, None, details


def _compute_capabilities() -> dict[str, Any]:
    """Return capabilities without importing heavyweight model packages."""

    basic_python = ROOT / ".venv-model-basic-pitch" / "Scripts" / "python.exe"
    demucs_python = ROOT / ".venv-model-demucs" / "Scripts" / "python.exe"
    base_python = ROOT / ".venv" / "Scripts" / "python.exe"
    game_available, game_reason, game_details = _game_status()
    tsumugi_available, tsumugi_reason, tsumugi_details = _tsumugi_status()
    muscriptor_available, muscriptor_reason, muscriptor_details = _muscriptor_status()
    basic_available, basic_reason = _module_probe(
        basic_python,
        ("basic_pitch", "onnxruntime"),
        label="Basic Pitch",
    )
    demucs_available, demucs_reason = _module_probe(
        demucs_python,
        ("demucs", "torch", "torchaudio", "soundfile"),
        label="Demucs",
    )
    base_available, base_reason = _module_probe(
        base_python,
        ("fastapi", "pydantic", "librosa", "soundfile", "numpy", "mido"),
        label="base",
    )
    jianpu = ROOT / "vendor" / "jianpu-ly" / "jianpu-ly.py"
    lilypond = ROOT / "tools" / "lilypond-2.24.4" / "bin" / "lilypond.exe"
    jianpu_available = jianpu.is_file()
    lilypond_available, lilypond_reason = _command_status(lilypond, label="LilyPond")
    ffmpeg = resolve_ffmpeg()
    ffprobe = resolve_ffprobe()
    ffmpeg_available, ffmpeg_reason = _command_status(ffmpeg, label="ffmpeg", arguments=("-version",))
    ffprobe_available, ffprobe_reason = _command_status(ffprobe, label="ffprobe", arguments=("-version",))

    engines: dict[str, dict[str, Any]] = {
        "basic-pitch": {
            "available": basic_available,
            "reason": basic_reason,
            "kind": "baseline",
            "environment": os.fspath(basic_python),
            "voice_modes": ["monophonic", "polyphonic"],
            "source_kinds": ["mixed", "vocal", "instrumental"],
        },
        "librosa": {
            "available": base_available,
            "reason": base_reason,
            "kind": "fallback",
            "environment": os.fspath(base_python),
            "voice_modes": ["monophonic", "polyphonic"],
            "source_kinds": ["mixed", "vocal", "instrumental"],
        },
        "game": {
            "available": game_available,
            "reason": game_reason,
            "kind": "specialist-lead-vocal",
            **game_details,
            "voice_modes": ["monophonic"],
            "source_kinds": ["mixed", "vocal"],
            "mixed_language_strategy": "omit --language and use GAME's language-agnostic default; no ID is guessed",
        },
        "tsumugi": {
            "available": tsumugi_available,
            "reason": tsumugi_reason,
            "kind": "specialist-stem-amt",
            **tsumugi_details,
            "voice_modes": ["monophonic", "polyphonic"],
            "source_kinds": ["vocal", "instrumental", "mixed"],
        },
        "muscriptor": {
            "available": muscriptor_available,
            "reason": muscriptor_reason,
            "kind": "instrumental-full-decode",
            **muscriptor_details,
            "voice_modes": ["polyphonic"],
            "source_kinds": ["instrumental"],
            "selection_stage": "after_full_decode",
        },
        "specialist": {
            "available": game_available and tsumugi_available,
            "reason": (
                None
                if game_available and tsumugi_available
                else "specialist requires both GAME and tsumugi capabilities; "
                f"GAME: {game_reason or 'available'}; tsumugi: {tsumugi_reason or 'available'}"
            ),
            "kind": "routed-specialist",
            "routes": {
                "vocal/monophonic": "GAME lead vocal",
                "instrumental/monophonic": "tsumugi other_v1_5",
                "instrumental/polyphonic": "tsumugi bass_v2 + other_v1_5",
                "vocal/polyphonic": "tsumugi vocal_harmony_v1_5 + bass_v2 + other_v1_5",
                "mixed/polyphonic": "tsumugi vocal_harmony_v1_5 + bass_v2 + other_v1_5",
            },
            "game": game_details,
            "tsumugi": tsumugi_details,
        },
    }
    return {
        "schema_version": "1.0",
        "default_engine": "basic-pitch",
        "engines": engines,
        "separation": {
            "demucs-htdemucs": {
                "available": demucs_available,
                "reason": demucs_reason,
                "environment": os.fspath(demucs_python),
                "stems": ["vocals", "drums", "bass", "other"],
            }
        },
        "rendering": {
            "jianpu-ly": {
                "available": jianpu_available,
                "reason": None if jianpu_available else f"jianpu-ly source is missing: {jianpu}",
                "path": os.fspath(jianpu),
            },
            "lilypond": {
                "available": lilypond_available,
                "reason": lilypond_reason,
                "path": os.fspath(lilypond),
            },
        },
        "audio": {
            "ffmpeg": {
                "available": ffmpeg_available,
                "reason": ffmpeg_reason,
                "path": os.fspath(ffmpeg) if ffmpeg else None,
            },
            "ffprobe": {
                "available": ffprobe_available,
                "reason": ffprobe_reason,
                "path": os.fspath(ffprobe) if ffprobe else None,
            },
        },
        "optional": {
            "chordscope": {
                "available": False,
                "reason": "optional Windows compatibility is not validated; no silent fallback is reported as chordscope",
            }
        },
        "v2": {
            "source_labels": {"instrumental": "伴奏/纯音乐", "vocal": "人声"},
            "instrumental": {
                "engine": "muscriptor",
                "model": "medium",
                "cuda": muscriptor_details["cuda"],
                "license": muscriptor_details["license"],
                "use_demucs": False,
            },
            "vocal": {
                "engine": "game",
                "use_demucs": True,
                "separation_engine": "demucs",
                "separation_model": "htdemucs",
            },
        },
    }


def get_capabilities() -> dict[str, Any]:
    """Return cached lightweight runtime and renderer readiness checks."""

    return _cached_capabilities()


def capabilities() -> dict[str, Any]:
    """Short alias for callers such as the future capabilities API."""

    return get_capabilities()
