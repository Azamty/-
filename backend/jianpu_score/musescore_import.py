"""Strict MuseScore MIDI -> partwise MusicXML adapter for high accuracy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Mapping
import xml.etree.ElementTree as ET

from .high_accuracy import (
    MUSESCORE_CLI_LOCK,
    MUSESCORE_IMPORT_PROFILE_SHA256,
    MUSESCORE_IMPORT_PROFILE_PATH,
    MUSESCORE_VERSION,
    resolve_musescore,
    validate_musescore_import_profile,
)


DEFAULT_PROFILE = MUSESCORE_IMPORT_PROFILE_PATH


class MuseScoreImportError(RuntimeError):
    """Raised when the pinned MuseScore importer cannot produce MusicXML."""


@dataclass(frozen=True)
class MusicXMLArtifact:
    midi_path: Path
    musicxml_path: Path
    instrument_id: str
    command: tuple[str, ...]
    attempts: tuple[tuple[tuple[str, ...], int], ...] = ()


MUSESCORE_TRANSIENT_CRASH_CODES = frozenset({3221225477})
MUSESCORE_TRANSIENT_RETRY_DELAY_SEC = 0.2


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
    try:
        validate_musescore_import_profile(
            profile,
            expected_sha256=MUSESCORE_IMPORT_PROFILE_SHA256 if profile == DEFAULT_PROFILE.resolve() else None,
        )
    except ValueError as exc:
        raise MuseScoreImportError(str(exc)) from exc
    if source == destination:
        raise MuseScoreImportError("MusicXML output must not overwrite the performance MIDI")
    if destination.exists() and not overwrite:
        raise MuseScoreImportError(f"MusicXML output already exists: {destination}")
    if timeout_sec <= 0:
        raise MuseScoreImportError("MuseScore timeout must be greater than zero")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command_prefix = (
        os.fspath(muse),
        "--factory-settings",
        "--test-mode",
        "-M",
        os.fspath(profile),
    )
    attempts: list[tuple[tuple[str, ...], int]] = []
    attempt_details: list[str] = []
    with MUSESCORE_CLI_LOCK:
        for attempt_number in range(2):
            temporary_handle, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.musescore-{attempt_number + 1}-",
                suffix=destination.suffix or ".musicxml",
                dir=destination.parent,
            )
            os.close(temporary_handle)
            temporary_output = Path(temporary_name)
            try:
                # MuseScore expects to create the output itself.  Start from a
                # fresh path so a Crashpad abort cannot poison the retry.
                temporary_output.unlink(missing_ok=True)
                command = command_prefix + ("-o", os.fspath(temporary_output), os.fspath(source))
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
                return_code = int(completed.returncode)
                attempts.append((command, return_code))
                detail = (completed.stderr or completed.stdout or "").strip()
                attempt_details.append(
                    f"attempt={attempt_number + 1}; returncode={return_code}; command={' '.join(command)}; "
                    f"detail={detail[-1000:]}"
                )
                if return_code:
                    if return_code in MUSESCORE_TRANSIENT_CRASH_CODES and attempt_number == 0:
                        time.sleep(MUSESCORE_TRANSIENT_RETRY_DELAY_SEC)
                        continue
                    retry_note = "; ".join(attempt_details)
                    raise MuseScoreImportError(
                        f"MuseScore MIDI import failed ({return_code}) for {instrument_id}; "
                        f"attempts: {retry_note}"
                    )
                if not temporary_output.is_file() or temporary_output.stat().st_size < 200:
                    raise MuseScoreImportError(
                        f"MuseScore exited successfully but produced no usable MusicXML for {instrument_id}: "
                        f"{temporary_output}; attempts: {'; '.join(attempt_details)}"
                    )
                try:
                    root_tag = ET.parse(temporary_output).getroot().tag.rsplit("}", 1)[-1]
                except (ET.ParseError, OSError) as exc:
                    raise MuseScoreImportError(
                        f"MuseScore produced invalid MusicXML for {instrument_id}: {exc}; "
                        f"attempts: {'; '.join(attempt_details)}"
                    ) from exc
                if root_tag != "score-partwise":
                    raise MuseScoreImportError(
                        f"MuseScore produced {root_tag!r} for {instrument_id}; expected partwise MusicXML; "
                        f"attempts: {'; '.join(attempt_details)}"
                    )
                if destination.exists() and not overwrite:
                    raise MuseScoreImportError(f"MusicXML output appeared during conversion: {destination}")
                os.replace(temporary_output, destination)
                final_command = command_prefix + ("-o", os.fspath(destination), os.fspath(source))
                return MusicXMLArtifact(
                    source,
                    destination,
                    str(instrument_id),
                    final_command,
                    tuple(attempts),
                )
            finally:
                temporary_output.unlink(missing_ok=True)
    raise MuseScoreImportError(
        f"MuseScore MIDI import exhausted retry attempts for {instrument_id}; "
        f"attempts: {'; '.join(attempt_details)}"
    )


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
