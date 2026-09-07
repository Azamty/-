"""Strict MuseScore MIDI -> partwise MusicXML adapter for high accuracy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Mapping
import xml.etree.ElementTree as ET

from .high_accuracy import MUSESCORE_CLI_LOCK, MUSESCORE_VERSION, ROOT, resolve_musescore


DEFAULT_PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"


class MuseScoreImportError(RuntimeError):
    """Raised when the pinned MuseScore importer cannot produce MusicXML."""


@dataclass(frozen=True)
class MusicXMLArtifact:
    midi_path: Path
    musicxml_path: Path
    instrument_id: str
    command: tuple[str, ...]


def convert_performance_midi(
    midi_path: str | Path,
    musicxml_path: str | Path,
    *,
    instrument_id: str = "instrument",
    musescore_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    timeout_sec: int = 180,
    overwrite: bool = False,
) -> MusicXMLArtifact:
    """Convert exactly one performance MIDI with the fixed import profile.

    The adapter never invokes another importer when MuseScore fails.  A
    non-zero exit, timeout, missing output, or an empty MusicXML file is an
    explicit error for the caller to surface in the job result.
    """

    source = Path(midi_path).expanduser().resolve()
    destination = Path(musicxml_path).expanduser().resolve()
    profile = Path(profile_path).expanduser().resolve() if profile_path else DEFAULT_PROFILE.resolve()
    muse = Path(musescore_path).expanduser().resolve() if musescore_path else resolve_musescore()
    if not source.is_file():
        raise MuseScoreImportError(f"performance MIDI does not exist: {source}")
    if muse is None or not muse.is_file():
        raise MuseScoreImportError(f"MuseScore {MUSESCORE_VERSION} executable is unavailable")
    if not profile.is_file():
        raise MuseScoreImportError(f"MuseScore MIDI import profile does not exist: {profile}")
    if source == destination:
        raise MuseScoreImportError("MusicXML output must not overwrite the performance MIDI")
    if destination.exists() and not overwrite:
        raise MuseScoreImportError(f"MusicXML output already exists: {destination}")
    if timeout_sec <= 0:
        raise MuseScoreImportError("MuseScore timeout must be greater than zero")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = (
        os.fspath(muse),
        "--factory-settings",
        "--test-mode",
        "-M",
        os.fspath(profile),
        "-o",
        os.fspath(destination),
        os.fspath(source),
    )
    with MUSESCORE_CLI_LOCK:
        try:
            completed = subprocess.run(
                list(command),
                cwd=destination.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MuseScoreImportError(
                f"MuseScore {MUSESCORE_VERSION} timed out after {timeout_sec}s for {source.name}"
            ) from exc
        except OSError as exc:
            raise MuseScoreImportError(f"MuseScore could not start: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise MuseScoreImportError(
            f"MuseScore MIDI import failed ({completed.returncode}) for {instrument_id}: {detail[-4000:]}"
        )
    if not destination.is_file() or destination.stat().st_size < 200:
        raise MuseScoreImportError(
            f"MuseScore exited successfully but produced no usable MusicXML for {instrument_id}: {destination}"
        )
    try:
        root_tag = ET.parse(destination).getroot().tag.rsplit("}", 1)[-1]
    except (ET.ParseError, OSError) as exc:
        raise MuseScoreImportError(f"MuseScore produced invalid MusicXML for {instrument_id}: {exc}") from exc
    if root_tag != "score-partwise":
        raise MuseScoreImportError(
            f"MuseScore produced {root_tag!r} for {instrument_id}; expected partwise MusicXML"
        )
    return MusicXMLArtifact(source, destination, str(instrument_id), command)


def convert_selected_performance_tracks(
    tracks: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    musescore_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    timeout_sec: int = 180,
    overwrite: bool = False,
) -> list[MusicXMLArtifact]:
    """Convert each selected instrument independently with stable filenames."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[MusicXMLArtifact] = []
    used_names: set[str] = set()
    for instrument_id, midi_path in tracks.items():
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(instrument_id)).strip("_") or "instrument"
        if safe in used_names:
            safe = f"{safe}-{hashlib.sha1(str(instrument_id).encode('utf-8')).hexdigest()[:8]}"
        used_names.add(safe)
        artifacts.append(
            convert_performance_midi(
                midi_path,
                destination / f"{safe}.musicxml",
                instrument_id=str(instrument_id),
                musescore_path=musescore_path,
                profile_path=profile_path,
                timeout_sec=timeout_sec,
                overwrite=overwrite,
            )
        )
    return artifacts
