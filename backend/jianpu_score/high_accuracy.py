"""High-accuracy notation toolchain discovery.

The model runtimes are intentionally isolated from the API environment.  This
module only probes their interpreters and command line tools; it never imports
BeatNet, music21, or MuseScore.  Keeping the probes cheap makes the result safe
to expose through ``/api/capabilities`` and makes missing optional tooling
explicit instead of silently selecting the legacy quantizer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Iterable
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
BEATNET_VERSION = "1.1.3"
MUSIC21_VERSION = "9.9.2"
MUSESCORE_VERSION = "4.7.4"
MUSESCORE_IMPORT_PROFILE_PATH = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"
MUSESCORE_RELEASE_URL = (
    "https://ftp.osuosl.org/pub/musescore-nightlies/windows/4x/stable/"
    "MuseScore-Studio-4.7.4.260706075-x86_64.msi"
)
MUSESCORE_RELEASE_SHA256 = "64FE70E5CB9FFE159D047D1E88DB567BD101F60D36B0DE28FEB674716929A378"
MUSESCORE_IMPORT_PROFILE = "adaptive-1/32-binary-triplet-tuplets-4-voices"
MUSESCORE_IMPORT_PROFILE_SHA256 = "86742B91922F921F725A1A5810572AB458EB7FB7AAC46FC683C92352B837C9FF"
MUSESCORE_IMPORT_PROFILE_EXPECTED: dict[str, str] = {
    "QuantValue": "3",  # shortest import unit: 1/32 note
    "VoiceCount": "3",  # MuseScore's value is zero-based: four voices
    "Duplets": "true",
    "Triplets": "true",
    "Quadruplets": "true",
    "Quintuplets": "false",
    "Septuplets": "false",
    "Nonuplets": "false",
    "HumanPerformance": "true",
}
# MuseScore Studio shares per-user crashpad state between headless invocations.
# The importer and the capability probe use this same process-wide lock so no
# MuseScore CLI entry point can overlap another one in this worker process.
MUSESCORE_CLI_LOCK = threading.Lock()


def _env_python(name: str) -> Path:
    return ROOT / name / "Scripts" / "python.exe"


def resolve_beatnet_python() -> Path:
    configured = os.environ.get("JIANPU_BEATNET_PYTHON")
    return Path(configured).expanduser().resolve() if configured else _env_python(".venv-model-beatnet")


def resolve_notation_python() -> Path:
    configured = os.environ.get("JIANPU_NOTATION_PYTHON")
    return Path(configured).expanduser().resolve() if configured else _env_python(".venv-notation")


def _candidate_musescore_paths() -> Iterable[Path]:
    configured = os.environ.get("JIANPU_MUSESCORE")
    if configured:
        # An explicit override is authoritative.  Falling through to a
        # machine-wide installation would make a misconfigured project appear
        # ready and would make the capability probe impossible to test.
        yield Path(configured).expanduser().resolve()
    yield from (
        ROOT / "tools" / "musescore-4.7.4" / "MuseScore4.exe",
        ROOT / "tools" / "musescore-4.7.4" / "bin" / "MuseScore4.exe",
        ROOT / "tools" / "musescore-4.7.4" / "MuseScore 4" / "bin" / "MuseScore4.exe",
        ROOT / "tools" / "musescore-4.7.4" / "bin" / "mscore.exe",
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 4\bin\mscore.exe"),
    )
    for command in ("MuseScore4.exe", "mscore.exe", "musescore.exe"):
        found = shutil.which(command)
        if found:
            yield Path(found).resolve()


def resolve_musescore() -> Path | None:
    if os.environ.get("JIANPU_MUSESCORE"):
        configured = Path(os.environ["JIANPU_MUSESCORE"]).expanduser().resolve()
        return configured if configured.is_file() else None
    seen: set[Path] = set()
    for path in _candidate_musescore_paths():
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def validate_musescore_import_profile(
    profile: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the pinned MIDI importer options before invoking MuseScore.

    The score normalizer uses 48 ticks per quarter, so only binary durations
    and 3:2 tuplets can be represented exactly.  A profile that enables
    5:4, 7:4, or 9:8 searches is therefore rejected at the boundary instead
    of allowing MuseScore to produce values that would later be rounded.
    """

    path = Path(profile).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"MuseScore MIDI import profile does not exist: {path}")
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"MuseScore MIDI import profile is invalid: {path}: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "MidiOptions":
        raise ValueError(f"MuseScore MIDI import profile has unexpected root: {root.tag!r}")
    values = {
        child.tag.rsplit("}", 1)[-1]: (child.text or "").strip().lower()
        for child in root
        if child.tag.rsplit("}", 1)[-1] in MUSESCORE_IMPORT_PROFILE_EXPECTED
    }
    mismatches = {
        name: {"expected": expected, "actual": values.get(name)}
        for name, expected in MUSESCORE_IMPORT_PROFILE_EXPECTED.items()
        if values.get(name) != expected
    }
    # Git may normalize this tracked XML between LF and CRLF on Windows;
    # pin the content hash independently of that transport detail.
    profile_bytes = path.read_bytes().replace(bytes((13, 10)), bytes((10,)))
    digest = hashlib.sha256(profile_bytes).hexdigest().upper()
    if expected_sha256 and digest != expected_sha256.upper():
        mismatches["sha256"] = {"expected": expected_sha256.upper(), "actual": digest}
    if mismatches:
        raise ValueError(f"MuseScore MIDI import profile does not match the pinned exact-grid policy: {mismatches}")
    return {
        "path": os.fspath(path),
        "sha256": digest,
        "options": values,
        "tuplets": {name: values[name] == "true" for name in ("Duplets", "Triplets", "Quadruplets", "Quintuplets", "Septuplets", "Nonuplets")},
    }


def _package_probe(python: Path, packages: tuple[str, ...], *, label: str) -> tuple[bool, str | None, dict[str, str]]:
    details = {"python": os.fspath(python), "packages": {}}
    if not python.is_file():
        return False, f"{label} environment is missing: {python}", details
    code = (
        "import importlib.metadata as m, json, sys; "
        "names=sys.argv[1:]; "
        "print(json.dumps({name: m.version(name) for name in names}))"
    )
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [os.fspath(python), "-c", code, *packages],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{label} package probe failed: {exc}", details
    if result.returncode:
        reason = (result.stderr or result.stdout).strip().splitlines()[-1:]
        return False, f"{label} package probe failed: {reason[0] if reason else 'unknown error'}", details
    try:
        versions = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return False, f"{label} package probe returned invalid JSON: {exc}", details
    details["packages"] = {str(name): str(version) for name, version in versions.items()}
    return True, None, details


def _command_probe(path: Path | None, *, label: str, expected_version: str | None = None) -> tuple[bool, str | None, dict[str, Any]]:
    details: dict[str, Any] = {"path": os.fspath(path) if path else None, "version": None}
    if path is None or not path.is_file():
        return False, f"{label} executable is missing", details
    with MUSESCORE_CLI_LOCK:
        try:
            result = subprocess.run(
                [os.fspath(path), "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"{label} version probe failed: {exc}", details
    output = "\n".join(filter(None, (result.stdout, result.stderr))).strip()
    details["version_output"] = output[:500]
    if result.returncode:
        return False, f"{label} returned exit code {result.returncode}", details
    # MuseScore prints a human-readable version.  Keep the exact output while
    # accepting wrappers that put the version on stderr.
    if expected_version and expected_version not in output:
        return False, f"{label} version mismatch: expected {expected_version}, got {output[:200]!r}", details
    details["version"] = expected_version or output
    return True, None, details


def get_high_accuracy_capabilities() -> dict[str, Any]:
    """Return honest readiness for the optional high-accuracy chain."""

    beatnet_python = resolve_beatnet_python()
    beatnet_ok, beatnet_reason, beatnet_details = _package_probe(
        beatnet_python,
        ("BeatNet", "numba", "madmom", "torch"),
        label="BeatNet",
    )
    beatnet_version = beatnet_details.get("packages", {}).get("BeatNet")
    if beatnet_ok and beatnet_version != BEATNET_VERSION:
        beatnet_ok = False
        beatnet_reason = f"BeatNet version mismatch: expected {BEATNET_VERSION}, got {beatnet_version}"
    beatnet_details.update({"required_version": BEATNET_VERSION, "python": os.fspath(beatnet_python)})

    notation_python = resolve_notation_python()
    notation_ok, notation_reason, notation_details = _package_probe(
        notation_python,
        ("music21",),
        label="music21",
    )
    notation_version = notation_details.get("packages", {}).get("music21")
    if notation_ok and notation_version != MUSIC21_VERSION:
        notation_ok = False
        notation_reason = f"music21 version mismatch: expected {MUSIC21_VERSION}, got {notation_version}"
    notation_details.update({"required_version": MUSIC21_VERSION, "python": os.fspath(notation_python)})

    muse_path = resolve_musescore()
    muse_ok, muse_reason, muse_details = _command_probe(
        muse_path,
        label="MuseScore",
        expected_version=MUSESCORE_VERSION,
    )
    muse_details.update(
        {
            "required_version": MUSESCORE_VERSION,
            "download_url": MUSESCORE_RELEASE_URL,
            "download_sha256": MUSESCORE_RELEASE_SHA256,
        }
    )
    try:
        profile_details = validate_musescore_import_profile(
            MUSESCORE_IMPORT_PROFILE_PATH,
            expected_sha256=MUSESCORE_IMPORT_PROFILE_SHA256,
        )
        profile_ok = True
        profile_reason = None
    except ValueError as exc:
        profile_details = {
            "path": os.fspath(MUSESCORE_IMPORT_PROFILE_PATH),
            "sha256": None,
            "options": {},
            "tuplets": {},
        }
        profile_ok = False
        profile_reason = str(exc)
    muse_details.update(
        {
            "import_profile": profile_details,
            "import_profile_available": profile_ok,
            "import_profile_reason": profile_reason,
        }
    )
    if muse_ok and not profile_ok:
        muse_ok = False
        muse_reason = f"MuseScore import profile is unavailable or invalid: {profile_reason}"

    available = beatnet_ok and notation_ok and muse_ok
    return {
        "schema_version": "1.0",
        "available": available,
        "reason": None if available else "; ".join(reason for reason in (beatnet_reason, notation_reason, muse_reason) if reason),
        "beatnet": {"available": beatnet_ok, "reason": beatnet_reason, **beatnet_details},
        "music21": {"available": notation_ok, "reason": notation_reason, **notation_details},
        "musescore": {"available": muse_ok, "reason": muse_reason, **muse_details},
        "runtime": {
            "notation_engine": "musescore-midi-import",
            "beat_engine": "beatnet",
            "score_ticks_per_quarter": 48,
            "musescore_import_profile": MUSESCORE_IMPORT_PROFILE,
            "musescore_import_profile_sha256": MUSESCORE_IMPORT_PROFILE_SHA256,
        },
    }
