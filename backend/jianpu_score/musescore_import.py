"""Strict MuseScore MIDI -> partwise MusicXML adapter for high accuracy."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import mido

from .high_accuracy import (
    MUSESCORE_CLI_LOCK,
    MUSESCORE_IMPORT_PROFILE_SHA256,
    MUSESCORE_IMPORT_PROFILE_PATH,
    MUSESCORE_IMPORT_PROFILE_EXPECTED,
    MUSESCORE_VOCAL_IMPORT_PROFILE_EXPECTED,
    MUSESCORE_VOCAL_IMPORT_PROFILE_PATH,
    MUSESCORE_VOCAL_IMPORT_PROFILE_SHA256,
    MUSESCORE_VERSION,
    resolve_musescore,
    validate_musescore_import_profile,
)


DEFAULT_PROFILE = MUSESCORE_IMPORT_PROFILE_PATH
MUSESCORE_ORIGIN_SENTINEL_NAME = "__JIANPU_SOURCE_ORIGIN_SENTINEL_v1__"
MUSESCORE_ORIGIN_SENTINEL_PITCH = 0
MUSESCORE_ORIGIN_SENTINEL_CHANNEL = 15
MUSESCORE_ORIGIN_SENTINEL_DURATION_TICKS = 1


class MuseScoreImportError(RuntimeError):
    """Raised when the pinned MuseScore importer cannot produce MusicXML."""


@dataclass(frozen=True)
class MusicXMLArtifact:
    midi_path: Path
    musicxml_path: Path
    instrument_id: str
    command: tuple[str, ...]
    attempts: tuple[tuple[tuple[str, ...], int], ...] = ()
    origin_sentinel: dict[str, Any] = field(default_factory=dict)


MUSESCORE_TRANSIENT_CRASH_CODES = frozenset({3221225477})
MUSESCORE_TRANSIENT_RETRY_DELAY_SEC = 0.2


def _prepare_origin_sentinel(source: Path, directory: Path) -> tuple[Path, dict[str, Any]]:
    """Create an importer-only MIDI copy with a traceable origin marker.

    MuseScore's adaptive human-performance importer can discard a leading
    performance offset when the first musical event starts after tick zero.
    The independent marker track makes tick zero observable to the importer;
    it is removed from MusicXML immediately after conversion.  The original
    performance MIDI is never modified.
    """

    try:
        midi = mido.MidiFile(os.fspath(source))
    except (OSError, ValueError, EOFError) as exc:
        raise MuseScoreImportError(f"could not read performance MIDI for origin sentinel: {source}: {exc}") from exc
    original_type = int(midi.type)
    if midi.type == 0:
        # A type-0 file cannot legally gain a second track.  This is an
        # importer-only copy, so promote it to type 1 while retaining the
        # original event track byte-for-byte at the message level.
        midi.type = 1
    sentinel_track = mido.MidiTrack()
    sentinel_track.append(mido.MetaMessage("track_name", name=MUSESCORE_ORIGIN_SENTINEL_NAME, time=0))
    sentinel_track.append(mido.MetaMessage("marker", text=MUSESCORE_ORIGIN_SENTINEL_NAME, time=0))
    sentinel_track.append(
        mido.Message(
            "program_change",
            channel=MUSESCORE_ORIGIN_SENTINEL_CHANNEL,
            program=0,
            time=0,
        )
    )
    sentinel_track.append(
        mido.Message(
            "note_on",
            channel=MUSESCORE_ORIGIN_SENTINEL_CHANNEL,
            note=MUSESCORE_ORIGIN_SENTINEL_PITCH,
            velocity=1,
            time=0,
        )
    )
    sentinel_track.append(
        mido.Message(
            "note_off",
            channel=MUSESCORE_ORIGIN_SENTINEL_CHANNEL,
            note=MUSESCORE_ORIGIN_SENTINEL_PITCH,
            velocity=0,
            time=MUSESCORE_ORIGIN_SENTINEL_DURATION_TICKS,
        )
    )
    sentinel_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(sentinel_track)
    handle, name = tempfile.mkstemp(
        prefix=f".{source.stem}.musescore-origin-sentinel-",
        suffix=".mid",
        dir=directory,
    )
    os.close(handle)
    temporary = Path(name)
    try:
        midi.save(os.fspath(temporary))
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise MuseScoreImportError(f"could not write origin-sentinel MIDI copy: {temporary}: {exc}") from exc
    return temporary, {
        "schema_version": "1.0",
        "name": MUSESCORE_ORIGIN_SENTINEL_NAME,
        "pitch": MUSESCORE_ORIGIN_SENTINEL_PITCH,
        "channel": MUSESCORE_ORIGIN_SENTINEL_CHANNEL + 1,
        "duration_ticks": MUSESCORE_ORIGIN_SENTINEL_DURATION_TICKS,
        "source_midi_type": original_type,
        "import_midi_type": int(midi.type),
        "import_track_index": len(midi.tracks) - 1,
        "original_midi_unchanged": True,
    }


def _strip_origin_sentinel_parts(musicxml_path: Path) -> dict[str, Any]:
    """Remove every MusicXML part carrying the exact origin marker name."""

    try:
        tree = ET.parse(musicxml_path)
    except (ET.ParseError, OSError) as exc:
        raise MuseScoreImportError(f"cannot inspect MusicXML origin sentinel: {musicxml_path}: {exc}") from exc
    root = tree.getroot()
    part_list = next((item for item in root if item.tag.rsplit("}", 1)[-1] == "part-list"), None)
    if part_list is None:
        raise MuseScoreImportError(f"MuseScore MusicXML has no part-list for origin sentinel: {musicxml_path}")
    part_names: dict[str, str] = {}
    score_parts: dict[str, ET.Element] = {}
    for item in part_list:
        if item.tag.rsplit("}", 1)[-1] != "score-part":
            continue
        part_id = item.attrib.get("id", "")
        score_parts[part_id] = item
        part_names[part_id] = next(
            (child.text or "" for child in item if child.tag.rsplit("}", 1)[-1] == "part-name"),
            "",
        )
    sentinel_ids = [part_id for part_id, name in part_names.items() if MUSESCORE_ORIGIN_SENTINEL_NAME in name]
    if not sentinel_ids:
        raise MuseScoreImportError(
            "MuseScore MusicXML did not preserve the origin sentinel part name; "
            "refusing to continue without auditable cleanup"
        )
    removed_parts: list[str] = []
    for item in list(root):
        if item.tag.rsplit("}", 1)[-1] == "part" and item.attrib.get("id", "") in sentinel_ids:
            root.remove(item)
            removed_parts.append(item.attrib.get("id", ""))
    if set(removed_parts) != set(sentinel_ids):
        raise MuseScoreImportError(
            f"MusicXML origin sentinel score-parts {sentinel_ids} do not have matching part elements"
        )
    for part_id in sentinel_ids:
        part_list.remove(score_parts[part_id])
    tree.write(musicxml_path, encoding="utf-8", xml_declaration=True)
    return {
        "name": MUSESCORE_ORIGIN_SENTINEL_NAME,
        "removed_part_ids": removed_parts,
        "removed_part_names": [part_names[part_id] for part_id in removed_parts],
        "remaining_part_ids": [
            item.attrib.get("id", "")
            for item in root
            if item.tag.rsplit("}", 1)[-1] == "part"
        ],
    }


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
        is_default_profile = profile == DEFAULT_PROFILE.resolve()
        is_vocal_profile = profile == MUSESCORE_VOCAL_IMPORT_PROFILE_PATH.resolve()
        validate_musescore_import_profile(
            profile,
            expected_sha256=(
                MUSESCORE_IMPORT_PROFILE_SHA256
                if is_default_profile
                else MUSESCORE_VOCAL_IMPORT_PROFILE_SHA256
                if is_vocal_profile
                else None
            ),
            expected_options=(
                MUSESCORE_IMPORT_PROFILE_EXPECTED
                if not is_vocal_profile
                else MUSESCORE_VOCAL_IMPORT_PROFILE_EXPECTED
            ),
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
    origin_sentinel: dict[str, Any] = {}
    with MUSESCORE_CLI_LOCK:
        import_source, origin_sentinel = _prepare_origin_sentinel(source, destination.parent)
        try:
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
                    command = command_prefix + ("-o", os.fspath(temporary_output), os.fspath(import_source))
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
                    origin_sentinel["musicxml_cleanup"] = _strip_origin_sentinel_parts(temporary_output)
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
                        dict(origin_sentinel),
                    )
                finally:
                    temporary_output.unlink(missing_ok=True)
        finally:
            import_source.unlink(missing_ok=True)
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
