"""Stage 9A orchestration for one high-accuracy instrument artifact bundle.

The service is deliberately independent from the V2 job manager.  It owns one
instrument directory and makes every boundary explicit:

``NoteEvent -> performance MIDI -> MuseScore MusicXML -> music21 Score -> SVG``

The original events and the 480 PPQ performance MIDI are retained beside the
notation outputs.  A failure never falls back to the legacy quantizer; a
failure manifest records the completed prefix of the pipeline instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping

import mido

from .domain import MusicAnalysis, NoteEvent, Score
from .high_accuracy import BEATNET_VERSION, MUSESCORE_VERSION
from .musescore_import import MusicXMLArtifact, MuseScoreImportError, convert_performance_midi
from .musicxml_standardize import MusicXMLStandardizationError, standardize_musicxml
from .performance_midi import build_performance_midi
from .render import RenderArtifacts, render_score
from .svg_long import merge_svg_pages


SERVICE_SCHEMA_VERSION = "1.0"
NOTATION_ENGINE = "musescore-midi-import"
BEAT_ENGINE = "beatnet"
SCORE_TICKS_PER_QUARTER = 48


class HighAccuracyServiceError(RuntimeError):
    """A stage failure with enough context for a job layer to report it."""

    def __init__(
        self,
        message: str,
        *,
        instrument_id: str,
        stage: str,
        cause: BaseException | str,
        log_path: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.instrument_id = instrument_id
        self.stage = stage
        self.cause = str(cause)
        self.log_path = log_path
        self.manifest_path = manifest_path
        super().__init__(message)


@dataclass(frozen=True)
class ServiceArtifact:
    """One output file recorded in the manifest."""

    artifact_id: str
    kind: str
    path: Path
    relative_path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class HighAccuracyBuildResult:
    """Typed result returned after one instrument finishes or skips notation."""

    instrument_id: str
    title: str
    variant: str
    program: int
    is_drum: bool
    status: str
    jianpu_status: str
    output_dir: Path
    manifest_path: Path
    artifacts: tuple[ServiceArtifact, ...]
    performance_metadata: dict[str, Any]
    score: Score | None = None
    alignment_report: dict[str, Any] | None = None


def _safe_component(value: str, *, fallback: str) -> str:
    text = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
    text = text.strip("._-")
    return text or fallback


def _safe_child(root: Path, relative: str | Path) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise HighAccuracyServiceError(
            f"artifact path escapes output directory: {relative}",
            instrument_id="unknown",
            stage="paths",
            cause=exc,
        ) from exc
    return candidate


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _is_link_or_reparse(path: Path) -> bool:
    """Identify links and Windows reparse points without following them."""

    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _reject_reparse_components(path: Path) -> None:
    """Reject an output path that would resolve through an unexpected link."""

    current = Path(path.anchor) if path.anchor else Path()
    for component in path.parts[1:] if path.anchor else path.parts:
        current /= component
        # Do not gate this on ``exists``: a broken symlink is still a path
        # redirection and must not be resolved and recreated outside root.
        if _is_link_or_reparse(current):
            raise ValueError(f"output path cannot contain a symlink or reparse point: {current}")


def _owned_manifest(destination: Path, *, instrument_id: str, variant: str) -> Path:
    """Return a matching service manifest or reject an unowned directory."""

    manifest_path = destination / "manifest.json"
    if _is_link_or_reparse(manifest_path) or not manifest_path.is_file():
        raise ValueError(
            f"refusing overwrite of unowned output directory; expected a regular service manifest: {destination}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"refusing overwrite; manifest.json is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SERVICE_SCHEMA_VERSION:
        raise ValueError(
            f"refusing overwrite; manifest schema does not match {SERVICE_SCHEMA_VERSION!r}: {manifest_path}"
        )
    if manifest.get("instrument_id") != instrument_id or manifest.get("variant") != variant:
        raise ValueError(
            "refusing overwrite; manifest instrument_id/variant does not match the requested build: "
            f"expected {instrument_id!r}/{variant!r}, got "
            f"{manifest.get('instrument_id')!r}/{manifest.get('variant')!r}"
        )
    return manifest_path


def _prepare_output_dir(
    output_dir: str | Path,
    *,
    overwrite: bool,
    instrument_id: str,
    variant: str,
) -> Path:
    unresolved = Path(output_dir).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    _reject_reparse_components(unresolved)
    destination = unresolved.resolve()
    if destination == destination.parent:
        raise ValueError("output directory must not be a filesystem root")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"output path is not a directory: {destination}")
    if destination.exists():
        entries = list(destination.iterdir())
        if entries and not overwrite:
            raise FileExistsError(
                f"output directory is not empty; use a new directory or overwrite=True: {destination}"
            )
        if overwrite:
            if entries:
                _owned_manifest(destination, instrument_id=instrument_id, variant=variant)
            for entry in entries:
                if _is_link_or_reparse(entry) or entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    _reject_reparse_components(entry)
                    shutil.rmtree(entry)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    return destination


def _write_log(destination: Path, stage: str, text: str) -> Path:
    path = _safe_child(destination, Path("logs") / f"{_safe_component(stage, fallback='stage')}.log")
    _atomic_write_bytes(path, (text.rstrip() + "\n").encode("utf-8", errors="replace"))
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, destination: Path) -> str:
    return path.resolve().relative_to(destination.resolve()).as_posix()


def _artifact(destination: Path, path: Path, *, artifact_id: str, kind: str) -> ServiceArtifact:
    resolved = path.resolve()
    relative = _relative(resolved, destination)
    return ServiceArtifact(
        artifact_id=artifact_id,
        kind=kind,
        path=resolved,
        relative_path=relative,
        sha256=_sha256(resolved),
        bytes=resolved.stat().st_size,
    )


def _collect_artifacts(destination: Path, *, manifest_path: Path) -> tuple[ServiceArtifact, ...]:
    artifacts: list[ServiceArtifact] = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if path.resolve() == manifest_path.resolve() or path.name.endswith(".tmp"):
            continue
        relative = _relative(path, destination)
        stem = Path(relative).as_posix().replace("/", "_")
        suffix = path.suffix.lower()
        kind = {
            ".json": "metadata" if "manifest" not in path.name else "manifest-support",
            ".mid": "midi",
            ".musicxml": "musicxml",
            ".jly": "jianpu-ly",
            ".ly": "lilypond",
            ".svg": "svg",
            ".log": "log",
        }.get(suffix, "file")
        artifacts.append(_artifact(destination, path, artifact_id=stem, kind=kind))
    return tuple(artifacts)


def _event_payload(events: tuple[NoteEvent, ...], *, instrument_id: str, title: str, variant: str) -> dict[str, Any]:
    return {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "instrument_id": instrument_id,
        "title": title,
        "variant": variant,
        "event_count": len(events),
        "events": [event.model_dump(mode="json") for event in events],
    }


def _validate_beatnet(analysis: MusicAnalysis) -> None:
    metadata = analysis.metadata
    if metadata.get("beat_engine") != BEAT_ENGINE:
        raise ValueError(
            f"BeatNet metadata is missing or marked as {metadata.get('beat_engine')!r}; expected {BEAT_ENGINE!r}"
        )
    if metadata.get("beatnet_version") != BEATNET_VERSION:
        raise ValueError(
            f"BeatNet version mismatch: expected {BEATNET_VERSION}, got {metadata.get('beatnet_version')!r}"
        )
    grid = metadata.get("beat_grid")
    if not isinstance(grid, Mapping) or not grid.get("beats"):
        raise ValueError("BeatNet beat_grid is missing or contains no beats")
    if len(analysis.beat_times) < 2:
        raise ValueError("BeatNet analysis must contain at least two beat_times")
    if any(not isinstance(item, Mapping) for item in grid["beats"]):
        raise ValueError("BeatNet beat_grid contains an invalid beat record")


def _validate_performance_metadata(events: tuple[NoteEvent, ...], metadata: Mapping[str, Any]) -> None:
    """Ensure the audit sidecar still identifies every input note in order."""

    expected_count = len(events)
    if int(metadata.get("note_count", -1)) != expected_count:
        raise ValueError(
            f"performance metadata note_count {metadata.get('note_count')} does not match input event count "
            f"{expected_count}"
        )
    notes = metadata.get("notes")
    if not isinstance(notes, list) or len(notes) != expected_count:
        raise ValueError(
            f"performance metadata notes has {len(notes) if isinstance(notes, list) else 'invalid'} entries; "
            f"expected {expected_count}"
        )
    for index, (event, item) in enumerate(zip(events, notes, strict=True)):
        if not isinstance(item, Mapping):
            raise ValueError(f"performance metadata note {index} is not an object")
        if int(item.get("index", -1)) != index or int(item.get("midi", -1)) != event.midi:
            raise ValueError(
                f"performance metadata note {index} does not match source index/pitch "
                f"{index}/{event.midi}: {item}"
            )
        if abs(float(item.get("start_sec", -1.0)) - event.start_sec) > 1e-9 or abs(
            float(item.get("end_sec", -1.0)) - event.end_sec
        ) > 1e-9:
            raise ValueError(f"performance metadata note {index} does not match source seconds")


def _score_pitch_set(score: Score) -> set[int]:
    return {
        pitch
        for voice in score.voices
        for event in voice.events
        for pitch in (event.chord_pitches or ([event.midi] if event.midi is not None else []))
    }


def _score_note_intervals(score: Score) -> list[tuple[int, int, int]]:
    """Return the pitched intervals that the rendered MIDI must contain.

    A tied chain is one playback note even though the Score keeps its
    measure-level segments.  Tie types are carried per chord pitch, so a
    partial chord tie does not accidentally merge its other pitches.
    """

    intervals: list[tuple[int, int, int]] = []
    for voice in score.voices:
        active: dict[int, tuple[int, int]] = {}
        for event in voice.events:
            pitches = event.chord_pitches or ([event.midi] if event.midi is not None else [])
            for index, pitch in enumerate(pitches):
                tie = event.tie_types[index] if index < len(event.tie_types) else event.tie
                if tie in {"stop", "continue"} and pitch in active:
                    start_tick, _previous_end = active[pitch]
                    active[pitch] = (start_tick, event.end_tick)
                    if tie == "stop":
                        intervals.append((pitch, start_tick, event.end_tick))
                        del active[pitch]
                elif tie == "start":
                    if pitch in active:
                        previous_start, previous_end = active.pop(pitch)
                        intervals.append((pitch, previous_start, previous_end))
                    active[pitch] = (event.start_tick, event.end_tick)
                else:
                    intervals.append((pitch, event.start_tick, event.end_tick))
        intervals.extend((pitch, start_tick, end_tick) for pitch, (start_tick, end_tick) in active.items())
    return intervals


def _verify_score_midi(score: Score, midi_path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(midi_path)
    expected_pitches = _score_pitch_set(score)
    actual_pitches = {
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }
    if actual_pitches != expected_pitches:
        raise ValueError(
            f"final score MIDI pitch set differs from Score: expected {sorted(expected_pitches)}, "
            f"got {sorted(actual_pitches)}"
        )
    actual_intervals: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        absolute = 0
        active: dict[tuple[int, int], list[int]] = {}
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault((message.channel, message.note), []).append(absolute)
            elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
                starts = active.get((message.channel, message.note))
                if starts:
                    start = starts.pop(0)
                    actual_intervals.append(
                        (
                            message.note,
                            round(start * score.quarter_ticks / midi.ticks_per_beat),
                            round(absolute * score.quarter_ticks / midi.ticks_per_beat),
                        )
                    )
                else:
                    raise ValueError(f"final score MIDI has an unmatched note-off for pitch {message.note}")
        if any(starts for starts in active.values()):
            raise ValueError("final score MIDI contains an unterminated note")
    expected_intervals = _score_note_intervals(score)
    if Counter(actual_intervals) != Counter(expected_intervals):
        raise ValueError(
            "final score MIDI note intervals differ from Score timeline: "
            f"expected {sorted(expected_intervals)}, got {sorted(actual_intervals)}"
        )
    expected_end = round(score.total_ticks * midi.ticks_per_beat / score.quarter_ticks)
    track_totals: list[int] = []
    note_on_count = 0
    for track in midi.tracks:
        absolute = 0
        has_notes = False
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                has_notes = True
                note_on_count += 1
        if has_notes:
            if absolute != expected_end:
                raise ValueError(
                    f"final score MIDI track ends at {absolute}, expected Score timeline end {expected_end}"
                )
            track_totals.append(absolute)
    if not track_totals:
        raise ValueError("final score MIDI contains no note track")
    return {
        "expected_pitch_set": sorted(expected_pitches),
        "actual_pitch_set": sorted(actual_pitches),
        "note_on_count": note_on_count,
        "expected_note_intervals": [list(item) for item in sorted(expected_intervals)],
        "actual_note_intervals": [list(item) for item in sorted(actual_intervals)],
        "expected_track_end_ticks": expected_end,
        "note_track_end_ticks": track_totals,
        "score_voice_count": len(score.voices),
    }


def _manifest_base(
    *,
    instrument_id: str,
    title: str,
    variant: str,
    program: int,
    is_drum: bool,
    status: str,
    jianpu_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "status": status,
        "instrument_id": instrument_id,
        "title": title,
        "variant": variant,
        "program": int(program),
        "is_drum": bool(is_drum),
        "jianpu_status": jianpu_status,
        "notation_engine": NOTATION_ENGINE,
        "beat_engine": BEAT_ENGINE,
        "beatnet_version": BEATNET_VERSION,
        "musescore_version": MUSESCORE_VERSION,
        "score_ticks_per_quarter": SCORE_TICKS_PER_QUARTER,
        "stages": {},
        "artifacts": [],
    }


class HighAccuracyArtifactService:
    """Build one instrument's complete high-accuracy artifact bundle."""

    def __init__(
        self,
        *,
        musescore_path: str | Path | None = None,
        profile_path: str | Path | None = None,
        notation_python: str | Path | None = None,
        timeout_sec: int = 180,
    ) -> None:
        self.musescore_path = musescore_path
        self.profile_path = profile_path
        self.notation_python = notation_python
        self.timeout_sec = timeout_sec

    def build(
        self,
        *,
        instrument_id: str,
        title: str,
        program: int,
        is_drum: bool,
        events: Iterable[NoteEvent],
        analysis: MusicAnalysis,
        output_dir: str | Path,
        variant: str = "source",
        overwrite: bool = False,
    ) -> HighAccuracyBuildResult:
        instrument = str(instrument_id).strip()
        if not instrument:
            raise ValueError("instrument_id must not be empty")
        safe_instrument = _safe_component(instrument, fallback="instrument")
        safe_variant = _safe_component(variant, fallback="source")
        safe_title = str(title).strip() or instrument
        destination = _prepare_output_dir(
            output_dir,
            overwrite=overwrite,
            instrument_id=instrument,
            variant=variant,
        )
        manifest_path = _safe_child(destination, "manifest.json")
        manifest = _manifest_base(
            instrument_id=instrument,
            title=safe_title,
            variant=variant,
            program=program,
            is_drum=is_drum,
            status="running",
            jianpu_status="pending",
        )
        artifacts: tuple[ServiceArtifact, ...] = ()
        performance_metadata: dict[str, Any] = {}
        score: Score | None = None
        alignment_report: dict[str, Any] | None = None
        materialized = tuple(events)

        def fail(stage: str, cause: BaseException | str) -> HighAccuracyServiceError:
            log_path = _write_log(destination, stage, f"instrument_id={instrument}\nstage={stage}\n{cause}")
            error = HighAccuracyServiceError(
                f"high-accuracy stage {stage} failed for {instrument}: {cause}",
                instrument_id=instrument,
                stage=stage,
                cause=cause,
                log_path=log_path,
                manifest_path=manifest_path,
            )
            manifest["status"] = "failed"
            manifest["jianpu_status"] = "failed" if not is_drum else "midi_only"
            manifest["stages"][stage] = {
                "status": "failed",
                "cause": str(cause),
                "log_relative_path": _relative(log_path, destination),
            }
            manifest["failure"] = {
                "instrument_id": instrument,
                "stage": stage,
                "cause": str(cause),
                "log_relative_path": _relative(log_path, destination),
            }
            manifest["artifacts"] = [item.as_dict() for item in _collect_artifacts(destination, manifest_path=manifest_path)]
            _atomic_write_json(manifest_path, manifest)
            return error

        try:
            if not materialized:
                raise fail("validate", "at least one NoteEvent is required")
            raw_path = _safe_child(destination, f"{safe_instrument}.{safe_variant}.note-events.json")
            try:
                _atomic_write_json(
                    raw_path,
                    _event_payload(materialized, instrument_id=instrument, title=safe_title, variant=variant),
                )
            except (OSError, TypeError, ValueError) as exc:
                raise fail("raw_notes", exc) from exc
            manifest["stages"]["raw_notes"] = {"status": "completed"}
            try:
                _validate_beatnet(analysis)
            except (TypeError, ValueError) as exc:
                raise fail("validate", exc) from exc
            manifest["stages"]["validate"] = {"status": "completed"}
            try:
                performance_bytes, performance_metadata = build_performance_midi(
                    materialized,
                    analysis,
                    instrument_group=instrument,
                    program=program,
                    is_drum=is_drum,
                    title=safe_title,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                raise fail("performance_midi", exc) from exc
            performance_metadata = {
                **performance_metadata,
                "instrument_id": instrument,
                "source_variant": variant,
                "notation_engine": NOTATION_ENGINE,
                "beat_engine": BEAT_ENGINE,
                "beatnet_version": BEATNET_VERSION,
                "musescore_version": MUSESCORE_VERSION,
                "score_ticks_per_quarter": SCORE_TICKS_PER_QUARTER,
            }
            try:
                _validate_performance_metadata(materialized, performance_metadata)
            except (TypeError, ValueError) as exc:
                raise fail("performance_midi", exc) from exc
            performance_path = _safe_child(destination, f"{safe_instrument}.{safe_variant}.performance.mid")
            performance_metadata_path = _safe_child(destination, f"{safe_instrument}.{safe_variant}.performance.metadata.json")
            try:
                _atomic_write_bytes(performance_path, performance_bytes)
                _atomic_write_json(performance_metadata_path, performance_metadata)
            except (OSError, TypeError, ValueError) as exc:
                raise fail("performance_midi", exc) from exc
            manifest["stages"]["performance_midi"] = {"status": "completed"}

            if is_drum:
                selected_path = _safe_child(destination, f"{safe_instrument}.{safe_variant}.selected.mid")
                _atomic_write_bytes(selected_path, performance_bytes)
                manifest["stages"]["notation"] = {"status": "skipped", "reason": "drum_midi_only"}
                manifest["status"] = "completed"
                manifest["jianpu_status"] = "midi_only"
                artifacts = _collect_artifacts(destination, manifest_path=manifest_path)
                manifest["artifacts"] = [item.as_dict() for item in artifacts]
                _atomic_write_json(manifest_path, manifest)
                return HighAccuracyBuildResult(
                    instrument,
                    safe_title,
                    variant,
                    int(program),
                    True,
                    "completed",
                    "midi_only",
                    destination,
                    manifest_path,
                    artifacts,
                    performance_metadata,
                )

            musicxml_path = _safe_child(destination, f"{safe_instrument}.{safe_variant}.notated.musicxml")
            try:
                musicxml_artifact: MusicXMLArtifact = convert_performance_midi(
                    performance_path,
                    musicxml_path,
                    instrument_id=instrument,
                    musescore_path=self.musescore_path,
                    profile_path=self.profile_path,
                    timeout_sec=self.timeout_sec,
                    overwrite=overwrite,
                )
            except (MuseScoreImportError, OSError, ValueError) as exc:
                raise fail("musescore_import", exc) from exc
            _write_log(destination, "musescore_import", "command=" + " ".join(musicxml_artifact.command))
            manifest["stages"]["musescore_import"] = {"status": "completed"}

            try:
                score, alignment_report = standardize_musicxml(
                    musicxml_path,
                    performance_metadata=performance_metadata,
                    title=safe_title,
                    notation_python=self.notation_python,
                    timeout_sec=self.timeout_sec,
                )
            except (MusicXMLStandardizationError, OSError, ValueError) as exc:
                raise fail("musicxml_standardize", exc) from exc
            source_count = int(alignment_report.get("source_note_count", -1))
            if source_count != len(materialized) or source_count != int(performance_metadata.get("note_count", -1)):
                raise fail(
                    "musicxml_standardize",
                    f"alignment source count {source_count} does not match input/performance "
                    f"{len(materialized)}/{performance_metadata.get('note_count')}",
                )
            if score.quarter_ticks != SCORE_TICKS_PER_QUARTER:
                raise fail(
                    "musicxml_standardize",
                    f"standardized Score uses {score.quarter_ticks} ticks per quarter; "
                    f"expected {SCORE_TICKS_PER_QUARTER}",
                )
            score_path = _safe_child(destination, f"{safe_instrument}.score.json")
            alignment_path = _safe_child(destination, f"{safe_instrument}.alignment_report.json")
            _atomic_write_json(score_path, score.model_dump(mode="json"))
            _atomic_write_json(alignment_path, alignment_report)
            manifest["stages"]["musicxml_standardize"] = {
                "status": "completed",
                "alignment_source_count": source_count,
            }

            try:
                render_artifacts: RenderArtifacts = render_score(
                    score,
                    destination,
                    basename=f"{safe_instrument}.score",
                )
                merge_svg_pages(
                    render_artifacts.svg_paths,
                    _safe_child(destination, f"{safe_instrument}.score.long.svg"),
                )
                midi_verification = _verify_score_midi(score, Path(render_artifacts.midi_path))
            except (OSError, RuntimeError, ValueError) as exc:
                raise fail("render", exc) from exc
            manifest["stages"]["render"] = {"status": "completed", "midi_verification": midi_verification}
            manifest["status"] = "completed"
            manifest["jianpu_status"] = "completed"
            artifacts = _collect_artifacts(destination, manifest_path=manifest_path)
            manifest["artifacts"] = [item.as_dict() for item in artifacts]
            _atomic_write_json(manifest_path, manifest)
            return HighAccuracyBuildResult(
                instrument,
                safe_title,
                variant,
                int(program),
                False,
                "completed",
                "completed",
                destination,
                manifest_path,
                artifacts,
                performance_metadata,
                score,
                alignment_report,
            )
        except HighAccuracyServiceError:
            raise
        except Exception as exc:
            raise fail("service", exc) from exc


def build_high_accuracy_artifacts(
    *,
    instrument_id: str,
    title: str,
    program: int,
    is_drum: bool,
    events: Iterable[NoteEvent],
    analysis: MusicAnalysis,
    output_dir: str | Path,
    variant: str = "source",
    overwrite: bool = False,
    musescore_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    notation_python: str | Path | None = None,
    timeout_sec: int = 180,
) -> HighAccuracyBuildResult:
    """Convenience API for one-shot service use without a service object."""

    return HighAccuracyArtifactService(
        musescore_path=musescore_path,
        profile_path=profile_path,
        notation_python=notation_python,
        timeout_sec=timeout_sec,
    ).build(
        instrument_id=instrument_id,
        title=title,
        program=program,
        is_drum=is_drum,
        events=events,
        analysis=analysis,
        output_dir=output_dir,
        variant=variant,
        overwrite=overwrite,
    )


__all__ = [
    "BEAT_ENGINE",
    "BEATNET_VERSION",
    "HighAccuracyArtifactService",
    "HighAccuracyBuildResult",
    "HighAccuracyServiceError",
    "NOTATION_ENGINE",
    "SCORE_TICKS_PER_QUARTER",
    "SERVICE_SCHEMA_VERSION",
    "ServiceArtifact",
    "build_high_accuracy_artifacts",
]
