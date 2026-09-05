"""Local cache and provenance for the MuScriptor browser soundfont.

The browser synthesizer consumes the same MIT MuseScore General SF3 asset used
by the upstream MuScriptor web application.  It is deliberately kept outside
git and copied into the project's ignored ``.cache`` directory after the
first request or an explicit prefetch.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_PYTHON = ROOT / ".venv-model-muscriptor" / "Scripts" / "python.exe"
SOUNDFONT_DIR = ROOT / ".cache" / "muscriptor"
SOUNDFONT_PATH = SOUNDFONT_DIR / "MuseScore_General.sf3"
SOUNDFONT_SOURCE = "hf://MuScriptor/assets/MuseScore_General.sf3"
SOUNDFONT_MIRROR = "https://huggingface.co/MuScriptor/assets"
SOUNDFONT_LICENSE = "MIT"
SOUNDFONT_SHA256 = "5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3"
_DOWNLOAD_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def soundfont_status() -> dict[str, Any]:
    present = SOUNDFONT_PATH.is_file()
    checksum = _sha256(SOUNDFONT_PATH) if present else None
    valid = checksum == SOUNDFONT_SHA256 if checksum else False
    return {
        "available": valid,
        "status": "ready" if valid else ("invalid" if present else "missing"),
        "filename": SOUNDFONT_PATH.name,
        "size_bytes": SOUNDFONT_PATH.stat().st_size if present else None,
        "sha256": checksum,
        "expected_sha256": SOUNDFONT_SHA256,
        "source": SOUNDFONT_SOURCE,
        "mirror": SOUNDFONT_MIRROR,
        "license": SOUNDFONT_LICENSE,
        "cache_path": os.fspath(SOUNDFONT_PATH),
        "message": (
            "MuScriptor 官方音色库已缓存，可由浏览器 SpessaSynth 加载。"
            if valid
            else "音色库尚未缓存；首次合成试听会从官方 MuScriptor assets 按需下载。"
        ),
    }


def ensure_soundfont() -> Path:
    """Download and verify the official SF3 asset without exposing credentials."""

    with _DOWNLOAD_LOCK:
        status = soundfont_status()
        if status["available"]:
            return SOUNDFONT_PATH
        if not MODEL_PYTHON.is_file():
            raise RuntimeError("MuScriptor 模型环境不存在，无法准备浏览器音色库。")
        SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)
        code = (
            "from muscriptor.soundfonts import SF3_URL; "
            "from muscriptor.utils.download import download_if_necessary; "
            "print(download_if_necessary(SF3_URL))"
        )
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        # stdout/stderr are captured and never returned: in particular, no
        # HuggingFace authentication material can become an API response.
        result = subprocess.run(
            [os.fspath(MODEL_PYTHON), "-c", code],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("官方 MuScriptor 音色库下载失败，请检查本机网络与模型环境。")
        candidates = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip().lower().endswith(".sf3")]
        source = next((path for path in reversed(candidates) if path.is_file()), None)
        if source is None:
            raise RuntimeError("官方 MuScriptor 音色库下载未返回有效文件。")
        temporary = SOUNDFONT_PATH.with_suffix(f".part-{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            if _sha256(temporary) != SOUNDFONT_SHA256:
                raise RuntimeError("官方 MuScriptor 音色库校验和不匹配。")
            os.replace(temporary, SOUNDFONT_PATH)
        finally:
            temporary.unlink(missing_ok=True)
        return SOUNDFONT_PATH
