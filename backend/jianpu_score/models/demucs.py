"""CPU Demucs adapter for the supported htdemucs model choices."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..analysis import probe_audio, resolve_ffmpeg


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DEMUCS_MODEL = "htdemucs"
DEMUCS_MODEL_CATALOG: dict[str, dict[str, object]] = {
    "htdemucs": {
        "id": "htdemucs",
        "label_zh": "快速",
        "speed_note": "基准速度",
        "quality_note": "Demucs 官方默认模型",
        "official_note": "官方默认的 Hybrid Transformer Demucs 模型",
    },
    "htdemucs_ft": {
        "id": "htdemucs_ft",
        "label_zh": "质量优先",
        "speed_note": "官方说明约慢 4 倍",
        "quality_note": "fine-tuned 版本，可能略好",
        "official_note": "官方说明：分离约慢 4 倍，但可能略好",
    },
}
DEMUCS_MODEL_IDS = frozenset(DEMUCS_MODEL_CATALOG)


def normalize_demucs_model(value: str | None) -> str:
    """Return a supported model id, defaulting old clients to htdemucs."""

    model = DEFAULT_DEMUCS_MODEL if value is None or not str(value).strip() else str(value).strip()
    if model not in DEMUCS_MODEL_IDS:
        allowed = "、".join(sorted(DEMUCS_MODEL_IDS))
        raise ValueError(f"Demucs 分离模型只允许：{allowed}")
    return model


def demucs_model_catalog() -> list[dict[str, object]]:
    """Return serializable model choices for capabilities and the API."""

    return [dict(DEMUCS_MODEL_CATALOG[model]) for model in ("htdemucs", "htdemucs_ft")]


def demucs_python() -> Path:
    configured = os.environ.get("JIANPU_DEMUCS_PYTHON")
    candidate = Path(configured) if configured else ROOT / ".venv-model-demucs" / "Scripts" / "python.exe"
    if not candidate.exists():
        raise RuntimeError(f"Demucs environment is missing: {candidate}")
    return candidate


def separate_htdemucs(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    model: str | None = None,
    process_holder: object | None = None,
) -> dict[str, Path]:
    """Run a supported Demucs model on CPU and return its four generated stems.

    ``process_holder`` is the single JobManager worker when called by V2.  It
    lets the service stop path terminate this child together with the other
    isolated model processes.
    """

    model = normalize_demucs_model(model)
    audio = Path(audio_path).resolve()
    probe_audio(audio)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "torch"
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["TORCH_HOME"] = os.fspath(cache)
    ffmpeg = resolve_ffmpeg()
    if ffmpeg is not None:
        environment["PATH"] = os.fspath(ffmpeg.parent) + os.pathsep + environment.get("PATH", "")
    command = [
        os.fspath(demucs_python()),
        "-m",
        "demucs.separate",
        "--name",
        model,
        "--out",
        os.fspath(destination),
        "--device",
        "cpu",
        "--float32",
        os.fspath(audio),
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_holder is not None:
        setattr(process_holder, "_child_process", process)
    try:
        stdout, stderr = process.communicate()
    finally:
        if process_holder is not None and getattr(process_holder, "_child_process", None) is process:
            setattr(process_holder, "_child_process", None)
    return_code = int(process.returncode or 0)
    if return_code:
        raise RuntimeError(f"Demucs subprocess failed ({return_code}): {stderr[-3000:]}")
    stem_dir = destination / model / audio.stem
    stems = {name: stem_dir / f"{name}.wav" for name in ("vocals", "drums", "bass", "other")}
    missing = [path for path in stems.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Demucs completed without stems: {missing}\n{stdout[-2000:]}")
    return stems
