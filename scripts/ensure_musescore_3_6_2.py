"""Download and unpack the pinned MuseScore 3.6.2 Windows x64 image.

The script uses the official GitHub release asset and keeps the installer in
the local package cache.  ``msiexec /a`` creates an administrative image under
the repository only; it does not register or install MuseScore system-wide.
Both the installer and extracted executable are verified before the metadata
manifest is written, so rerunning the script is offline and idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
VERSION = "3.6.2.548021803"
RELEASE_URL = "https://github.com/musescore/MuseScore/releases/tag/v3.6.2"
ASSET_URL = "https://github.com/musescore/MuseScore/releases/download/v3.6.2/MuseScore-3.6.2.548021803-x86_64.msi"
ASSET_NAME = "MuseScore-3.6.2.548021803-x86_64.msi"
EXPECTED_BYTES = 112_041_984
EXPECTED_SHA256 = "fa3ca0f8cc5b0e8b0c0bb8ef11e227b9b27b2a5c9da28dab58bafcbb0eb657d0"
CACHE_PATH = ROOT / ".cache" / "packages" / ASSET_NAME
INSTALL_ROOT = ROOT / "tools" / "musescore-3.6.2"
MANIFEST_PATH = INSTALL_ROOT / "install_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    """Use stable repository-relative paths where the path is inside ROOT."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return os.fspath(resolved)


def verify_asset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"MuseScore 3.6.2 asset is missing: {path}")
    size = path.stat().st_size
    digest = sha256(path)
    if size != EXPECTED_BYTES or digest.casefold() != EXPECTED_SHA256.casefold():
        raise RuntimeError(
            f"MuseScore 3.6.2 asset verification failed: bytes={size} sha256={digest}; "
            f"expected bytes={EXPECTED_BYTES} sha256={EXPECTED_SHA256}"
        )
    return {"path": os.fspath(path), "bytes": size, "sha256": digest}


def download_asset(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            return {"asset": verify_asset(path), "downloaded": False}
        except RuntimeError:
            # Preserve the old bytes for auditability.  A fresh temporary
            # download is atomically promoted only after verification.
            pass
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.unlink(missing_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        completed = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                "900",
                "--output",
                os.fspath(temporary),
                ASSET_URL,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=960,
            check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"MuseScore asset download failed ({completed.returncode}): {detail[-2000:]}")
    else:
        request = Request(ASSET_URL, headers={"User-Agent": "luna-high-accuracy-benchmark"})
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
    verified = verify_asset(temporary)
    temporary.replace(path)
    return {"asset": verified, "downloaded": True}


def find_executable(root: Path) -> Path | None:
    candidates = sorted(root.rglob("MuseScore3.exe"))
    return candidates[0] if candidates else None


def version_probe(executable: Path) -> str:
    for args in (("--version",), ("-v",)):
        try:
            completed = subprocess.run(
                [os.fspath(executable), *args],
                cwd=executable.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0 and output:
            return output
    raise RuntimeError(f"MuseScore 3.6.2 executable did not answer a version probe: {executable}")


def unpack_asset(asset: Path) -> tuple[Path, str]:
    INSTALL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    existing = find_executable(INSTALL_ROOT)
    if existing is not None and MANIFEST_PATH.is_file():
        probed = version_probe(existing)
        if "3.6.2" not in probed:
            raise RuntimeError(f"existing MuseScore executable has an unexpected version: {probed}")
        return existing, probed
    existing_entries = [item for item in INSTALL_ROOT.iterdir()] if INSTALL_ROOT.exists() else []
    existing_payload = [item for item in existing_entries if item.name != ".gitignore"]
    if existing_payload:
        raise RuntimeError(
            f"refusing to overwrite a non-empty incomplete install directory: {INSTALL_ROOT}; "
            "remove it explicitly after auditing it, then rerun"
        )
    staging = Path(tempfile.mkdtemp(prefix="musescore-3.6.2-", dir=INSTALL_ROOT.parent))
    log_path = staging / "msiexec.log"
    try:
        command = [
            "msiexec.exe",
            "/a",
            os.fspath(asset),
            "/qn",
            f"TARGETDIR={staging}",
            "/L*v",
            os.fspath(log_path),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"msiexec administrative extraction failed ({completed.returncode}): {detail[-2000:]}")
        executable = find_executable(staging)
        if executable is None:
            raise RuntimeError(f"msiexec succeeded but MuseScore3.exe was not found under {staging}")
        probed = version_probe(executable)
        # The staging directory lives beside the final directory.  A tracked
        # .gitignore may already be present in the destination, so move the
        # verified payload into that empty shell; otherwise keep the atomic
        # directory rename.
        if INSTALL_ROOT.exists():
            for item in staging.iterdir():
                item.replace(INSTALL_ROOT / item.name)
            staging.rmdir()
        else:
            staging.rename(INSTALL_ROOT)
        return INSTALL_ROOT / executable.relative_to(staging), probed
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    global INSTALL_ROOT, MANIFEST_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=CACHE_PATH)
    parser.add_argument("--install-root", type=Path, default=INSTALL_ROOT)
    args = parser.parse_args()
    # The default locations are part of the reproducibility contract.  Custom
    # paths remain useful for an offline audit without changing the release
    # identity recorded below.
    asset_path = args.asset.expanduser().resolve()
    install_root = args.install_root.expanduser().resolve()
    asset_result = download_asset(asset_path)
    INSTALL_ROOT = install_root
    MANIFEST_PATH = install_root / "install_manifest.json"
    executable, probed_version = unpack_asset(asset_path)
    manifest = {
        "schema_version": "1.0",
        "version": VERSION,
        "release_url": RELEASE_URL,
        "asset_url": ASSET_URL,
        "asset_name": ASSET_NAME,
        "asset": {
            **asset_result["asset"],
            "path": manifest_path(asset_path),
        },
        "downloaded": asset_result["downloaded"],
        "unpack_method": "msiexec /a administrative image under repository; no system installation",
        "install_root": manifest_path(install_root),
        "executable": manifest_path(executable),
        "version_probe": probed_version,
    }
    install_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
