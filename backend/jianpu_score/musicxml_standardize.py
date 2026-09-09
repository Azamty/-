"""Strict MusicXML -> 48 TPQ Score normalization.

MuseScore performs the performance-MIDI notation decisions.  This module only
invokes the isolated music21 worker, validates its versioned JSON, preserves
notation metadata, and converts the result into the renderer-independent
Score contract.  It never imports music21 and never falls back to the legacy
uniform quantizer.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import subprocess
import tempfile
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain import (
    Score,
    ScoreNote,
    ScoreVoice,
    TempoEvent,
    normalize_key,
    normalize_time_signature,
    sanitize_title,
)
from .high_accuracy import (
    MUSESCORE_VERSION,
    MUSIC21_VERSION,
    ROOT,
    resolve_notation_python,
)

WORKER_SCHEMA_VERSION = "1.0"
SCORE_QUARTER_TICKS = 48
PERFORMANCE_QUARTER_TICKS = 480
MAX_FINE_GRID_MOVEMENT_TICKS = 0.5
# jianpu-ly's smallest exact atom at the shared 48 TPQ grid is a 64th-note
# (3 ticks).  MuseScore can nevertheless emit a final 1/32-quarter fragment
# that rounds to one or two ticks.  Keep the 48 TPQ contract and repair only
# those notation fragments, with the movement recorded in the alignment
# report so the renderer never silently changes timing.
MIN_JIANPU_ATOM_TICKS = 3
MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS = 2
# A production recognizer can describe the same ordered notes in a different
# beat coordinate system than MuseScore's notation import.  Only accept an
# explicit affine/offset reconciliation when the pitch sequence is complete,
# one-to-one, and the residuals remain bounded.
MAX_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS = 12
MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS = 64
# A complete, unique pitch-order affine model is stronger evidence than a
# nearest-neighbour match. Permit the same 24-tick onset window that the
# ordinary matcher already uses for that model; pitch, residual, and
# one-to-one checks remain mandatory.
MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS = 24
MIN_SOURCE_ALIGNMENT_MODEL_POINTS = 3
MAX_SOURCE_ALIGNMENT_MOVEMENT_TICKS = 384
# MuseScore can split one imported performance into several synthetic parts
# (for example P1-Staff1/P1-Staff2 plus a voice-only part).  When the complete
# pitch multiset and monotonic order prove a one-to-one mapping, a single
# cross-part affine model is safe even if one synthetic part has fewer than
# three anchors.  Keep this fallback bounded to roughly one quarter-note of
# residual and 512 score ticks of raw movement; it is never a nearest-neighbor
# search and never changes MusicXML timing.
MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS = 64
MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS = 64
MAX_CROSS_PART_ALIGNMENT_MOVEMENT_TICKS = 512
MAX_SOURCE_RETRIGGER_RESIDUAL_TICKS = 64
MAX_SOURCE_RETRIGGER_INTERNAL_GAP_TICKS = 2
WORKER = ROOT / "scripts" / "musicxml_score_worker.py"


class MusicXMLStandardizationError(RuntimeError):
    """Raised for an explicit worker or normalization failure."""


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    kind: Literal["note", "chord", "rest", "grace"]
    offset_quarter: float = Field(ge=0)
    duration_quarter: float = Field(ge=0)
    pitches: list[int] = Field(default_factory=list)
    tie: str | None = None
    tie_types: list[str | None] = Field(default_factory=list)
    tuplet_actual: int | None = Field(default=None, gt=0)
    tuplet_normal: int | None = Field(default=None, gt=0)
    tuplet_type: str | None = None
    dots: int = Field(default=0, ge=0)
    grace: bool = False
    voice: str = "1"
    staff: int = Field(default=1, ge=1)
    measure_number: int | None = None

    @field_validator("pitches")
    @classmethod
    def validate_pitches(cls, value: list[int]) -> list[int]:
        if any(pitch < 0 or pitch > 127 for pitch in value):
            raise ValueError("worker pitches must be MIDI values between 0 and 127")
        return value

    @field_validator("tuplet_type")
    @classmethod
    def validate_tuplet_type(cls, value: str | None) -> str | None:
        if value is not None and value not in {"start", "stop", "continue"}:
            raise ValueError("tuplet_type must be start, stop, continue, or null")
        return value


class WorkerMeasure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_index: int
    number: int
    start_quarter: float = Field(ge=0)
    duration_quarter: float = Field(ge=0)
    end_quarter: float = Field(ge=0)
    time_signature: str | None = None
    is_pickup: bool = False


class WorkerPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_id: str
    name: str
    instrument: str = ""
    highest_time_quarter: float = Field(ge=0)
    events: list[WorkerEvent] = Field(default_factory=list)
    measures: list[WorkerMeasure] = Field(default_factory=list)


class WorkerTempo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset_quarter: float = Field(ge=0)
    bpm: float = Field(gt=0)


class WorkerTimeSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset_quarter: float = Field(ge=0)
    ratio: str
    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)


class WorkerKeySignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset_quarter: float = Field(ge=0)
    key: str
    sharps: int


class WorkerPickup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_pickup: bool = False
    duration_quarter: float = Field(default=0, ge=0)
    measure_number: int | None = None


class WorkerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    worker: str
    music21_version: str
    source_path: str
    title: str
    highest_time_quarter: float = Field(ge=0)
    parts: list[WorkerPart] = Field(min_length=1)
    measures: list[WorkerMeasure] = Field(default_factory=list)
    tempo_events: list[WorkerTempo] = Field(default_factory=list)
    time_signature_events: list[WorkerTimeSignature] = Field(default_factory=list)
    key_signature_events: list[WorkerKeySignature] = Field(default_factory=list)
    pickup: WorkerPickup = Field(default_factory=WorkerPickup)


@dataclass(frozen=True)
class StandardizedScoreArtifact:
    """A validated Score and its two audit JSON outputs."""

    musicxml_path: Path
    score_json_path: Path
    alignment_report_path: Path
    score: Score
    alignment_report: dict[str, Any]


@dataclass
class _RawEvent:
    event_id: str
    part_group: str
    part_id: str
    staff: int
    voice: str
    start_tick: int
    end_tick: int
    pitches: list[int]
    kind: str
    tie: str | None
    tie_types: list[str | None]
    tuplet_actual: int | None
    tuplet_normal: int | None
    dots: int
    measure_number: int | None
    metadata: dict[str, Any]
    # Optional for hand-built alignment fixtures written before explicit
    # MusicXML tuplet boundaries were carried through the worker.
    tuplet_type: str | None = None


@dataclass(frozen=True)
class _LogicalPitchUnit:
    """One pitched MusicXML note after collapsing a tie chain.

    Matching performance metadata against individual MusicXML fragments made
    adaptive quantization look like missing notes.  A unit is the smallest
    auditable thing that can be matched: one pitch slot and its complete tie
    chain.  The worker events remain the source of truth for Score timing.
    """

    unit_id: int
    pitch: int
    start_tick: int
    end_tick: int
    chain: tuple[tuple[_RawEvent, int], ...]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MusicXMLStandardizationError(f"cannot read worker JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MusicXMLStandardizationError("music21 worker output must be a JSON object")
    return value


def run_musicxml_worker(
    musicxml_path: str | Path,
    *,
    notation_python: str | Path | None = None,
    timeout_sec: int = 180,
) -> WorkerPayload:
    """Run the only music21 process and validate its versioned output."""

    source = Path(musicxml_path).expanduser().resolve()
    python = Path(notation_python).expanduser().resolve() if notation_python else resolve_notation_python()
    if not source.is_file():
        raise MusicXMLStandardizationError(f"MusicXML input does not exist: {source}")
    if not python.is_file():
        raise MusicXMLStandardizationError(f"music21 notation environment is unavailable: {python}")
    if not WORKER.is_file():
        raise MusicXMLStandardizationError(f"music21 worker is missing: {WORKER}")
    if timeout_sec <= 0:
        raise MusicXMLStandardizationError("music21 worker timeout must be greater than zero")
    with tempfile.TemporaryDirectory(prefix="jianpu-music21-worker-") as temporary:
        output = Path(temporary) / "worker.json"
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        command = [os.fspath(python), os.fspath(WORKER), "--input", os.fspath(source), "--output", os.fspath(output)]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MusicXMLStandardizationError(f"music21 worker timed out after {timeout_sec}s") from exc
        except OSError as exc:
            raise MusicXMLStandardizationError(f"music21 worker could not start: {exc}") from exc
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise MusicXMLStandardizationError(f"music21 worker failed ({completed.returncode}): {detail[-4000:]}")
        if not output.is_file() or output.stat().st_size < 2:
            raise MusicXMLStandardizationError("music21 worker exited successfully without a JSON output")
        raw = _load_json(output)
    try:
        payload = WorkerPayload.model_validate(raw)
    except Exception as exc:
        raise MusicXMLStandardizationError(f"invalid music21 worker schema: {exc}") from exc
    if payload.schema_version != WORKER_SCHEMA_VERSION:
        raise MusicXMLStandardizationError(
            f"music21 worker schema mismatch: expected {WORKER_SCHEMA_VERSION}, got {payload.schema_version}"
        )
    if payload.music21_version != MUSIC21_VERSION:
        raise MusicXMLStandardizationError(
            f"music21 worker version mismatch: expected {MUSIC21_VERSION}, got {payload.music21_version}"
        )
    return payload


def _part_group(part_id: str) -> str:
    marker = "-Staff"
    return part_id.split(marker, 1)[0] if marker in part_id else part_id


def _quarter_to_tick(
    value: float,
    *,
    context: str = "MusicXML quarter value",
    allow_finer_binary: bool = False,
    quantization_repairs: list[dict[str, Any]] | None = None,
) -> int:
    if not math.isfinite(value):
        raise MusicXMLStandardizationError(f"non-finite MusicXML quarter position: {value!r}")
    scaled = value * SCORE_QUARTER_TICKS
    rounded = round(scaled)
    movement = float(rounded - scaled)
    if abs(movement) > 1e-7:
        fraction = Fraction(str(value)).limit_denominator(4096)
        denominator = fraction.denominator
        is_finer_binary = denominator >= 32 and denominator & (denominator - 1) == 0
        if not (allow_finer_binary and is_finer_binary and abs(movement) <= MAX_FINE_GRID_MOVEMENT_TICKS):
            raise MusicXMLStandardizationError(
                f"{context} {value!r} cannot be represented exactly at {SCORE_QUARTER_TICKS} TPQ; "
                "the notation contains a tuplet or duration outside the supported exact grid"
            )
        if quantization_repairs is not None:
            quantization_repairs.append(
                {
                    "reason": "finer_binary_musicxml_value_quantized_to_48_tpq",
                    "context": context,
                    "original_quarter": float(value),
                    "original_fraction": f"{fraction.numerator}/{fraction.denominator}",
                    "score_tick": int(rounded),
                    "movement_ticks": movement,
                    "bounded_by_ticks": MAX_FINE_GRID_MOVEMENT_TICKS,
                }
            )
    return int(rounded)


_WORKER_KEY_PATTERN = re.compile(
    r"^\s*([A-Ga-g])\s*(?:(#|b|-))?\s*(?:(major|minor|maj|min|m))?\s*$",
    re.IGNORECASE,
)


def _normalize_worker_key(value: str) -> str:
    """Normalize music21's strict key spellings into the domain whitelist.

    music21 may serialize a flat as ``D-`` (and, for example, ``b- minor``)
    instead of the ``Db`` spelling accepted by the application.  Parse only
    the complete key grammar here: replacing arbitrary hyphens would turn an
    invalid key into a plausible one and would accidentally widen the domain.
    """

    text = str(value).strip()
    match = _WORKER_KEY_PATTERN.fullmatch(text)
    if match is None:
        raise MusicXMLStandardizationError(f"unsupported MusicXML key signature: {value!r}")
    root, accidental, mode = match.groups()
    accidental = "b" if accidental == "-" else (accidental or "")
    suffix = "m" if mode and mode.casefold() in {"m", "min", "minor"} else ""
    text = f"{root.upper()}{accidental}{suffix}"
    try:
        return normalize_key(text)
    except ValueError as exc:
        raise MusicXMLStandardizationError(f"unsupported MusicXML key signature: {value!r}") from exc


def _normalize_worker_meter(value: str) -> str:
    try:
        return normalize_time_signature(str(value))
    except ValueError as exc:
        raise MusicXMLStandardizationError(
            f"unsupported MusicXML time signature {value!r}; supported meters are 2/4, 3/4, 4/4, 6/8"
        ) from exc


def _source_notes(performance_metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not performance_metadata:
        return []
    values = performance_metadata.get("notes", [])
    if not isinstance(values, list):
        return []
    lane_names = performance_metadata.get("instrument_lane_track_names", [])
    if not isinstance(lane_names, list):
        lane_names = []
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            continue
        try:
            start_480 = int(value["start_tick"])
            end_480 = int(value["end_tick"])
            midi = int(value["midi"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_480 <= start_480:
            continue
        source = {
            "source_index": int(value.get("source_index", value.get("index", index))),
            "midi": midi,
            "start_tick_480": start_480,
            "end_tick_480": end_480,
            "start_tick": round(start_480 * SCORE_QUARTER_TICKS / PERFORMANCE_QUARTER_TICKS),
            "end_tick": round(end_480 * SCORE_QUARTER_TICKS / PERFORMANCE_QUARTER_TICKS),
            "voice_id": value.get("voice_id"),
        }
        # Keep the MIDI identity that survives into the performance file.
        # MuseScore can turn one imported track into several synthetic parts;
        # the lane identity is therefore evidence for partitioning an
        # otherwise ambiguous same-pitch assignment, never a replacement for
        # pitch/time matching.
        for key in ("midi_lane", "midi_track_index", "midi_channel"):
            try:
                if value.get(key) is not None:
                    source[key] = int(value[key])
            except (TypeError, ValueError):
                pass
        if value.get("midi_lane") is not None:
            try:
                lane = int(value["midi_lane"])
            except (TypeError, ValueError):
                lane = -1
            if 0 <= lane < len(lane_names) and isinstance(lane_names[lane], str):
                source["midi_track_name"] = lane_names[lane]
        result.append(source)
    return result


def _worker_raw_events(payload: WorkerPayload) -> tuple[list[_RawEvent], list[dict[str, Any]]]:
    events: list[_RawEvent] = []
    diagnostics: list[dict[str, Any]] = []
    for part in payload.parts:
        group = _part_group(part.part_id)
        for item in part.events:
            if item.kind == "grace" or item.grace or item.duration_quarter <= 0:
                diagnostics.append(
                    {
                        "musicxml_event_id": item.event_id,
                        "reason": "grace_event_not_representable_at_48_tpq",
                        "action": "omitted_with_diagnostic",
                    }
                )
                continue
            start_tick = _quarter_to_tick(
                item.offset_quarter,
                context="MusicXML event offset",
                allow_finer_binary=True,
                quantization_repairs=diagnostics,
            )
            end_tick = max(
                start_tick + 1,
                _quarter_to_tick(
                    item.offset_quarter + item.duration_quarter,
                    context="MusicXML event end",
                    allow_finer_binary=True,
                    quantization_repairs=diagnostics,
                ),
            )
            events.append(
                _RawEvent(
                    event_id=item.event_id,
                    part_group=group,
                    part_id=part.part_id,
                    staff=item.staff,
                    voice=item.voice,
                    start_tick=start_tick,
                    end_tick=end_tick,
                    pitches=list(item.pitches),
                    kind=item.kind,
                    tie=item.tie,
                    tie_types=list(item.tie_types),
                    tuplet_actual=item.tuplet_actual,
                    tuplet_normal=item.tuplet_normal,
                    tuplet_type=item.tuplet_type,
                    dots=item.dots,
                    measure_number=item.measure_number,
                    metadata={"musicxml_event_id": item.event_id},
                )
            )
    diagnostics.extend(_annotate_source_tuplet_groups(events))
    diagnostics.extend(_normalize_tied_event_voices(events))
    diagnostics.extend(_reassemble_split_source_tuplets(events))
    diagnostics.extend(_normalize_orphan_tuplet_markers(events))
    # A cross-voice tuplet repair can move a fragment that starts a tie.  Run
    # the same conservative tie-chain pass once more so its unique successor
    # follows the repaired fragment; otherwise the serializer would report a
    # dangling tie even though the original import was unambiguous.
    diagnostics.extend(_normalize_tied_event_voices(events))
    for event in events:
        event.metadata.pop(_SOURCE_TUPLET_GROUP_KEY, None)
    return events, diagnostics


def _normalize_tied_event_voices(events: list[_RawEvent]) -> list[dict[str, Any]]:
    """Keep each MusicXML tie chain in one serializable ScoreVoice.

    music21 can expose a legal MusicXML tie with different ``Voice`` context
    on either side of a barline.  The source MusicXML still identifies the
    same pitch and exact adjacent times, so this is safe to normalize when
    that successor is unique.  Ambiguous same-pitch candidates are left
    untouched and remain an explicit serializer error rather than being
    guessed.  Mixed chords are split by target voice so untied/new pitches do
    not inherit another pitch's tie repair.
    """

    def slot_tie(event: _RawEvent, pitch_index: int) -> str | None:
        return event.tie_types[pitch_index] if pitch_index < len(event.tie_types) else event.tie

    slots = [
        (event.event_id, pitch_index)
        for event in events
        for pitch_index, _pitch in enumerate(event.pitches)
    ]
    by_same_staff: dict[tuple[str, int, int, int], list[tuple[str, int]]] = {}
    by_any_staff: dict[tuple[str, int, int], list[tuple[str, int]]] = {}
    slot_values: dict[tuple[str, int], tuple[_RawEvent, int, int]] = {}
    for event in events:
        for pitch_index, pitch in enumerate(event.pitches):
            key = (event.event_id, pitch_index)
            slot_values[key] = (event, pitch_index, pitch)
            by_same_staff.setdefault((event.part_group, event.staff, pitch, event.start_tick), []).append(key)
            by_any_staff.setdefault((event.part_group, pitch, event.start_tick), []).append(key)

    def candidate(
        source_key: tuple[str, int],
    ) -> tuple[tuple[str, int] | None, str | None]:
        event, pitch_index, pitch = slot_values[source_key]
        if slot_tie(event, pitch_index) not in {"start", "continue"}:
            return None, None

        def valid(keys: list[tuple[str, int]]) -> list[tuple[str, int]]:
            return [
                key
                for key in keys
                if key != source_key
                and slot_tie(slot_values[key][0], slot_values[key][1]) in {"stop", "continue"}
            ]

        same_staff = valid(by_same_staff.get((event.part_group, event.staff, pitch, event.end_tick), []))
        if len(same_staff) > 1:
            same_voice = [key for key in same_staff if slot_values[key][0].voice == event.voice]
            same_staff = same_voice if len(same_voice) == 1 else []
        if len(same_staff) == 1:
            return same_staff[0], "same_staff"
        if same_staff:
            return None, None

        cross_staff = valid(by_any_staff.get((event.part_group, pitch, event.end_tick), []))
        if len(cross_staff) > 1:
            same_voice = [key for key in cross_staff if slot_values[key][0].voice == event.voice]
            cross_staff = same_voice if len(same_voice) == 1 else []
        if len(cross_staff) == 1:
            return cross_staff[0], "cross_staff"
        return None, None

    successors: dict[tuple[str, int], tuple[tuple[str, int], str]] = {}
    for source_key in slots:
        target_key, scope = candidate(source_key)
        if target_key is not None and scope is not None:
            successors[source_key] = (target_key, scope)
    predecessors: dict[tuple[str, int], tuple[tuple[str, int], str]] = {}
    ambiguous: set[tuple[str, int]] = set()
    for source_key, (target_key, scope) in successors.items():
        previous = predecessors.get(target_key)
        if previous is not None and previous[0] != source_key:
            ambiguous.add(target_key)
        else:
            predecessors[target_key] = (source_key, scope)
    for target_key in ambiguous:
        predecessors.pop(target_key, None)

    root_cache: dict[tuple[str, int], tuple[str, int, str]] = {}

    def root_for(slot_key: tuple[str, int], seen: set[tuple[str, int]] | None = None) -> tuple[str, int, str]:
        cached = root_cache.get(slot_key)
        if cached is not None:
            return cached
        seen = set() if seen is None else seen
        if slot_key in seen:
            event, _pitch_index, _pitch = slot_values[slot_key]
            result = (event.voice, event.staff, "ambiguous_cycle")
            root_cache[slot_key] = result
            return result
        seen.add(slot_key)
        predecessor = predecessors.get(slot_key)
        if predecessor is None:
            event, _pitch_index, _pitch = slot_values[slot_key]
            result = (event.voice, event.staff, "root")
        else:
            result = root_for(predecessor[0], seen)
        root_cache[slot_key] = result
        return result

    repairs: list[dict[str, Any]] = []
    normalized: list[_RawEvent] = []
    for event in events:
        if not event.pitches:
            normalized.append(event)
            continue
        groups: dict[tuple[str, int], list[tuple[int, str | None]]] = {}
        for pitch_index, pitch in enumerate(event.pitches):
            tie = slot_tie(event, pitch_index)
            target = (event.voice, event.staff)
            if tie in {"stop", "continue"} and (event.event_id, pitch_index) in predecessors:
                root_voice, root_staff, _reason = root_for((event.event_id, pitch_index))
                target = (root_voice, root_staff)
            groups.setdefault(target, []).append((pitch, tie))
        original_target = (event.voice, event.staff)
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: (item[0] != original_target, item[0][1], item[0][0]),
        )
        for group_index, ((target_voice, target_staff), values) in enumerate(ordered_groups):
            pitches = [pitch for pitch, _tie in values]
            tie_types = [tie for _pitch, tie in values]
            if len(pitches) == 1:
                kind = "note"
            else:
                kind = "chord"
            if not event.tie_types and event.tie is None:
                tie_types = []
            present_ties = [value for value in tie_types if value is not None]
            tie = (
                present_ties[0]
                if present_ties and len(present_ties) == len(tie_types) and all(value == present_ties[0] for value in present_ties)
                else None
            )
            changed = (target_voice, target_staff) != original_target
            metadata = dict(event.metadata)
            if changed:
                repair = {
                    "reason": "tie_chain_voice_reassigned" if len(ordered_groups) == 1 else "tie_chain_event_split",
                    "musicxml_event_id": event.event_id,
                    "source_voice": event.voice,
                    "source_staff": event.staff,
                    "target_voice": target_voice,
                    "target_staff": target_staff,
                    "pitches": pitches,
                }
                repairs.append(repair)
                metadata["tie_voice_repair"] = repair
            event_id = event.event_id
            if group_index and len(ordered_groups) > 1:
                event_id = f"{event.event_id}:tie-voice-{target_staff}-{target_voice}"
            normalized.append(
                _RawEvent(
                    event_id=event_id,
                    part_group=event.part_group,
                    part_id=event.part_id,
                    staff=target_staff,
                    voice=target_voice,
                    start_tick=event.start_tick,
                    end_tick=event.end_tick,
                    pitches=pitches,
                    kind=kind,
                    tie=tie,
                    tie_types=tie_types,
                    tuplet_actual=event.tuplet_actual,
                    tuplet_normal=event.tuplet_normal,
                    tuplet_type=event.tuplet_type,
                    dots=event.dots,
                    measure_number=event.measure_number,
                    metadata=metadata,
                )
            )
    events[:] = normalized
    return repairs


_SOURCE_TUPLET_GROUP_KEY = "_source_tuplet_group"


def _source_event_id(event: _RawEvent) -> str:
    value = event.metadata.get("musicxml_event_id")
    return str(value) if value is not None else event.event_id.split(":tie-voice-", 1)[0]


def _annotate_source_tuplet_groups(events: list[_RawEvent]) -> list[dict[str, Any]]:
    """Record complete MusicXML tuplet groups before tie voice normalization.

    Tie repair can split one MusicXML chord into multiple ScoreVoice fragments.
    We retain a compact source group only when the original worker events prove
    a single ratio, contiguous boundaries, and a closed actual/nominal span.
    Later repair may then reassemble that exact source group; incomplete or
    ambiguous markers receive no annotation and remain serializer errors.
    """

    grouped: dict[tuple[str, int, str, tuple[int, int]], list[_RawEvent]] = {}
    for event in events:
        ratio = _raw_tuplet_ratio(event)
        if ratio is None or ratio != (3, 2):
            continue
        grouped.setdefault((event.part_group, event.staff, event.voice, ratio), []).append(event)
    diagnostics: list[dict[str, Any]] = []
    group_number = 0
    for (part_group, staff, voice, ratio), candidates in grouped.items():
        ordered = sorted(candidates, key=lambda event: (event.start_tick, event.end_tick, event.event_id))
        for start in (event for event in ordered if event.tuplet_type == "start"):
            stops = [
                event
                for event in ordered
                if event.tuplet_type == "stop" and event.start_tick >= start.end_tick
            ]
            if not stops:
                continue
            stop = stops[0]
            component = _raw_tuplet_span(events, start, stop, allow_cross_voice=False)
            if component is None:
                continue
            source_event_ids = [_source_event_id(event) for event in component]
            if len(source_event_ids) != len(set(source_event_ids)):
                continue
            actual_ticks = sum(event.end_tick - event.start_tick for event in component)
            nominal_durations = [
                Fraction((event.end_tick - event.start_tick) * ratio[0], ratio[1]) for event in component
            ]
            nominal_ticks = sum(nominal_durations, Fraction(0))
            if (
                nominal_ticks.denominator != 1
                or nominal_ticks <= 0
                or any(value.denominator != 1 or value <= 0 for value in nominal_durations)
            ):
                continue
            group_number += 1
            group_id = f"{part_group}:{staff}:{voice}:{start.start_tick}:{stop.end_tick}:{group_number}"
            members = [
                {
                    "event_id": _source_event_id(event),
                    "start_tick": event.start_tick,
                    "end_tick": event.end_tick,
                    "pitches": list(event.pitches),
                    "kind": event.kind,
                    "tie": event.tie,
                    "tie_types": list(event.tie_types),
                    "tuplet_actual": event.tuplet_actual,
                    "tuplet_normal": event.tuplet_normal,
                    "tuplet_type": event.tuplet_type,
                    "dots": event.dots,
                    "measure_number": event.measure_number,
                }
                for event in component
            ]
            group = {
                "group_id": group_id,
                "part_group": part_group,
                "staff": staff,
                "source_voice": voice,
                "tuplet_actual": ratio[0],
                "tuplet_normal": ratio[1],
                "start_tick": start.start_tick,
                "end_tick": stop.end_tick,
                "actual_ticks": actual_ticks,
                "nominal_ticks": int(nominal_ticks),
                "nominal_duration_ticks": [int(value) for value in nominal_durations],
                "source_event_ids": source_event_ids,
                "members": members,
            }
            for event in component:
                metadata = dict(event.metadata)
                metadata[_SOURCE_TUPLET_GROUP_KEY] = group
                event.metadata = metadata
            diagnostics.append(
                {
                    "reason": "complete_source_tuplet_group_recorded",
                    "action": "retained_source_tuplet_group_for_voice_reconciliation",
                    "group_id": group_id,
                    "part_group": part_group,
                    "staff": staff,
                    "voice": voice,
                    "start_tick": start.start_tick,
                    "end_tick": stop.end_tick,
                    "tuplet_actual": ratio[0],
                    "tuplet_normal": ratio[1],
                    "source_event_ids": source_event_ids,
                    "actual_ticks": actual_ticks,
                    "nominal_ticks": int(nominal_ticks),
                    "nominal_duration_ticks": [int(value) for value in nominal_durations],
                }
            )
    return diagnostics


def _reassemble_split_source_tuplets(events: list[_RawEvent]) -> list[dict[str, Any]]:
    """Reassemble only a proven source tuplet split by tie voice repair."""

    groups: dict[str, dict[str, Any]] = {}
    grouped_events: dict[str, list[_RawEvent]] = {}
    for event in events:
        group = event.metadata.get(_SOURCE_TUPLET_GROUP_KEY)
        if not isinstance(group, Mapping):
            continue
        group_id = str(group.get("group_id", ""))
        if not group_id:
            continue
        groups[group_id] = dict(group)
        grouped_events.setdefault(group_id, []).append(event)

    repairs: list[dict[str, Any]] = []
    remove_ids: set[int] = set()
    replacements: list[_RawEvent] = []
    for group_id, group_events in grouped_events.items():
        group = groups[group_id]
        source_event_ids = [str(value) for value in group.get("source_event_ids", [])]
        audit = {
            "group_id": group_id,
            "part_group": group.get("part_group"),
            "staff": group.get("staff"),
            "voice": group.get("source_voice"),
            "start_tick": group.get("start_tick"),
            "end_tick": group.get("end_tick"),
            "tuplet_actual": group.get("tuplet_actual"),
            "tuplet_normal": group.get("tuplet_normal"),
            "actual_ticks": group.get("actual_ticks"),
            "nominal_ticks": group.get("nominal_ticks"),
            "nominal_duration_ticks": group.get("nominal_duration_ticks"),
            "source_event_ids": source_event_ids,
        }
        members = group.get("members")
        if not source_event_ids or not isinstance(members, list) or len(members) != len(source_event_ids):
            continue
        source_members = {str(item.get("event_id")): item for item in members if isinstance(item, Mapping)}
        if set(source_members) != set(source_event_ids):
            continue
        by_source: dict[str, list[_RawEvent]] = {event_id: [] for event_id in source_event_ids}
        invalid = False
        for event in group_events:
            source_id = _source_event_id(event)
            member = source_members.get(source_id)
            if member is None:
                invalid = True
                break
            if (
                event.part_group != group.get("part_group")
                or event.staff != int(group.get("staff", event.staff))
                or event.start_tick != int(member["start_tick"])
                or event.end_tick != int(member["end_tick"])
                or _raw_tuplet_ratio(event) != (int(group["tuplet_actual"]), int(group["tuplet_normal"]))
            ):
                invalid = True
                break
            by_source[source_id].append(event)
        if invalid or any(not values for values in by_source.values()):
            continue
        voices = {event.voice for event in group_events}
        repaired_voices = {
            str(event.metadata.get("tie_voice_repair", {}).get("target_voice"))
            for event in group_events
            if isinstance(event.metadata.get("tie_voice_repair"), Mapping)
            and event.metadata["tie_voice_repair"].get("target_voice") is not None
        }
        repaired_voices.discard("None")
        if len(repaired_voices) != 1 or len(voices) <= 1:
            # A source group that did not split because of the known tie repair
            # has no basis for a cross-voice reconstruction.
            if len(voices) > 1:
                repairs.append(
                    {
                        **audit,
                        "reason": "cross_voice_tuplet_marker_conflict",
                        "action": "preserved_split_tuplet_due_to_ambiguous_voice_repair",
                        "conflict": "missing_unique_tie_repair_target",
                        "candidate_voices": sorted(voices),
                        "candidate_repair_voices": sorted(repaired_voices),
                        "original_marker": [
                            {
                                "musicxml_event_id": str(member["event_id"]),
                                "tuplet_actual": member.get("tuplet_actual"),
                                "tuplet_normal": member.get("tuplet_normal"),
                                "tuplet_type": member.get("tuplet_type"),
                            }
                            for member in members
                        ],
                    }
                )
            continue
        target_voice = next(iter(repaired_voices))
        target_staff = int(group["staff"])
        group_event_ids = {id(event) for event in group_events}
        for event in events:
            if id(event) in group_event_ids:
                continue
            if event.part_group != group.get("part_group") or event.staff != target_staff or event.voice != target_voice:
                continue
            if not event.pitches:
                # Empty MusicXML timeline fillers can occupy the same raw
                # voice; _allocate_lanes will place them in a separate lane
                # after the proven pitched tuplet is reconstructed.
                continue
            if any(
                event.start_tick < int(member["end_tick"]) and int(member["start_tick"]) < event.end_tick
                for member in members
            ):
                invalid = True
                break
        if invalid:
            repairs.append(
                {
                    **audit,
                    "reason": "cross_voice_tuplet_marker_conflict",
                    "action": "preserved_split_tuplet_due_to_target_voice_overlap",
                    "conflict": "target_voice_overlap",
                    "target_voice": target_voice,
                }
            )
            continue

        reconstructed: list[_RawEvent] = []
        for member in members:
            source_id = str(member["event_id"])
            fragments = by_source[source_id]
            expected_pitches = [int(value) for value in member.get("pitches", [])]
            fragment_pitches = [pitch for event in fragments for pitch in event.pitches]
            if sorted(fragment_pitches) != sorted(expected_pitches) or len(fragment_pitches) != len(set(fragment_pitches)):
                invalid = True
                break
            template = fragments[0]
            tie_by_pitch: dict[int, str | None] = {}
            for fragment in fragments:
                for pitch_index, pitch in enumerate(fragment.pitches):
                    if pitch in tie_by_pitch:
                        invalid = True
                        break
                    tie_by_pitch[pitch] = (
                        fragment.tie_types[pitch_index]
                        if pitch_index < len(fragment.tie_types)
                        else fragment.tie
                    )
                if invalid:
                    break
            if invalid:
                break
            tie_types = [tie_by_pitch.get(pitch) for pitch in expected_pitches]
            present_ties = [value for value in tie_types if value is not None]
            tie = (
                present_ties[0]
                if present_ties and len(present_ties) == len(tie_types) and all(value == present_ties[0] for value in present_ties)
                else None
            )
            repair = {
                "reason": "cross_voice_tuplet_marker_reassembled",
                "action": "reassembled_complete_source_tuplet_after_tie_voice_split",
                "group_id": group_id,
                "part_group": group["part_group"],
                "staff": target_staff,
                "voice": group["source_voice"],
                "target_voice": target_voice,
                "start_tick": int(group["start_tick"]),
                "end_tick": int(group["end_tick"]),
                "tuplet_actual": int(group["tuplet_actual"]),
                "tuplet_normal": int(group["tuplet_normal"]),
                "actual_ticks": int(group["actual_ticks"]),
                "nominal_ticks": int(group["nominal_ticks"]),
                "nominal_duration_ticks": list(group.get("nominal_duration_ticks", [])),
                "source_event_ids": source_event_ids,
                "source_event_id": source_id,
                "original_marker": {
                    "tuplet_actual": member.get("tuplet_actual"),
                    "tuplet_normal": member.get("tuplet_normal"),
                    "tuplet_type": member.get("tuplet_type"),
                },
                "timing_preserved": True,
                "pitch_multiset_preserved": True,
                "one_to_one_source_events": True,
            }
            metadata = dict(template.metadata)
            metadata.pop(_SOURCE_TUPLET_GROUP_KEY, None)
            metadata["tuplet_boundary_repair"] = repair
            reconstructed.append(
                _RawEvent(
                    event_id=source_id,
                    part_group=template.part_group,
                    part_id=template.part_id,
                    staff=target_staff,
                    voice=target_voice,
                    start_tick=int(member["start_tick"]),
                    end_tick=int(member["end_tick"]),
                    pitches=expected_pitches,
                    kind=str(member.get("kind", template.kind)),
                    tie=tie,
                    tie_types=tie_types if expected_pitches else [],
                    tuplet_actual=int(member["tuplet_actual"]) if member.get("tuplet_actual") is not None else None,
                    tuplet_normal=int(member["tuplet_normal"]) if member.get("tuplet_normal") is not None else None,
                    tuplet_type=member.get("tuplet_type"),
                    dots=int(member.get("dots", template.dots)),
                    measure_number=member.get("measure_number"),
                    metadata=metadata,
                )
            )
        if invalid:
            repairs.append(
                {
                    **audit,
                    "reason": "cross_voice_tuplet_marker_conflict",
                    "action": "preserved_split_tuplet_due_to_pitch_or_tie_mismatch",
                    "conflict": "source_pitch_or_tie_slot_mismatch",
                    "target_voice": target_voice,
                }
            )
            continue
        remove_ids.update(group_event_ids)
        replacements.extend(reconstructed)
        repairs.append(
            {
                "reason": "cross_voice_tuplet_marker_reassembled",
                "action": "reassembled_complete_source_tuplet_after_tie_voice_split",
                "group_id": group_id,
                "part_group": group["part_group"],
                "staff": target_staff,
                "voice": group["source_voice"],
                "target_voice": target_voice,
                "start_tick": int(group["start_tick"]),
                "end_tick": int(group["end_tick"]),
                "tuplet_actual": int(group["tuplet_actual"]),
                "tuplet_normal": int(group["tuplet_normal"]),
                "actual_ticks": int(group["actual_ticks"]),
                "nominal_ticks": int(group["nominal_ticks"]),
                "nominal_duration_ticks": list(group.get("nominal_duration_ticks", [])),
                "source_event_ids": source_event_ids,
                "original_marker": [
                    {
                        "musicxml_event_id": str(member["event_id"]),
                        "tuplet_actual": member.get("tuplet_actual"),
                        "tuplet_normal": member.get("tuplet_normal"),
                        "tuplet_type": member.get("tuplet_type"),
                    }
                    for member in members
                ],
                "timing_preserved": True,
                "pitch_multiset_preserved": True,
                "one_to_one_source_events": True,
            }
        )
    if remove_ids:
        events[:] = [event for event in events if id(event) not in remove_ids]
        events.extend(replacements)
        events.sort(key=lambda event: (event.part_group, event.staff, event.start_tick, event.end_tick, event.voice, event.event_id))
    return repairs


def _raw_tuplet_ratio(event: _RawEvent) -> tuple[int, int] | None:
    if event.tuplet_actual is None and event.tuplet_normal is None:
        return None
    if event.tuplet_actual is None or event.tuplet_normal is None:
        return None
    return event.tuplet_actual, event.tuplet_normal


def _raw_tuplet_context(event: _RawEvent) -> tuple[str, int, tuple[int, int]] | None:
    ratio = _raw_tuplet_ratio(event)
    if ratio is None:
        return None
    return event.part_group, event.staff, ratio


def _raw_duration_is_serializable(event: _RawEvent) -> bool:
    """Check a raw event after its tuplet marker would be removed."""

    duration = event.end_tick - event.start_tick
    if duration <= 0:
        return False
    view = ScoreNote(
        start_tick=event.start_tick,
        duration_tick=duration,
        midi=min(event.pitches) if event.pitches else None,
    )
    return _score_duration_is_serializable(view)


def _raw_tuplet_span(
    events: list[_RawEvent],
    start: _RawEvent,
    stop: _RawEvent,
    *,
    allow_cross_voice: bool,
) -> list[_RawEvent] | None:
    """Return one contiguous explicit tuplet span, or ``None``.

    The serializer cannot represent one explicit bracket across ScoreVoice
    lanes.  A direct start/end boundary with one contiguous ratio-bearing
    timeline is the only cross-voice shape accepted here; gaps, overlaps, and
    ratio changes remain errors for the serializer to report.
    """

    ratio = _raw_tuplet_ratio(start)
    if ratio != (3, 2) or _raw_tuplet_ratio(stop) != ratio:
        return None
    if start.tuplet_type != "start" or stop.tuplet_type != "stop":
        return None
    if start.end_tick > stop.start_tick or (not allow_cross_voice and start.voice != stop.voice):
        return None
    if start.voice != stop.voice and not allow_cross_voice:
        return None
    context = [
        event
        for event in events
        if event.part_group == start.part_group
        and event.staff == start.staff
        and _raw_tuplet_ratio(event) == ratio
        and event.start_tick >= start.start_tick
        and event.end_tick <= stop.end_tick
    ]
    ordered_all = sorted(context, key=lambda event: (event.start_tick, event.end_tick, event.event_id))
    # Tie voice normalization can split one MusicXML chord into multiple raw
    # fragments with the same source event id and interval.  They are one
    # tuplet slot for contiguity, while the full fragment list must still be
    # returned so pitch fragments are included in overlap checks and audit.
    ordered: list[_RawEvent] = []
    seen_slots: set[tuple[str, int, int]] = set()
    for event in ordered_all:
        slot = (_source_event_id(event), event.start_tick, event.end_tick)
        if slot in seen_slots:
            continue
        seen_slots.add(slot)
        ordered.append(event)
    if not ordered or ordered[0] is not start or ordered[-1] is not stop:
        return None
    if ordered[-1].end_tick != stop.end_tick:
        return None
    if any(previous.end_tick != current.start_tick for previous, current in zip(ordered, ordered[1:])):
        return None
    if any(
        event is not start
        and event is not stop
        and event.tuplet_type in {"start", "stop"}
        for event in ordered
    ):
        return None
    if len({event.voice for event in ordered}) > 1 and not allow_cross_voice:
        return None
    return ordered_all


def _raw_tuplet_voice_has_overlap(
    events: list[_RawEvent],
    component: list[_RawEvent],
    *,
    target_voice: str,
) -> bool:
    component_ids = {id(event) for event in component}
    for event in component:
        for other in events:
            if id(other) in component_ids or other.voice != target_voice:
                continue
            if other.part_group != event.part_group or other.staff != event.staff:
                continue
            # A MusicXML rest is a timeline filler, not an occupied pitch
            # lane.  It may overlap a repaired note group and will be split
            # into its own ScoreVoice lane by _allocate_lanes; treating it as
            # a musical overlap would leave a proven tuplet fragment
            # stranded in a voice with no valid bracket.
            if not other.pitches:
                continue
            if other.start_tick < event.end_tick and event.start_tick < other.end_tick:
                return True
    return False


def _record_tuplet_marker_repair(
    event: _RawEvent,
    *,
    reason: str,
    action: str,
    target_voice: str | None = None,
    group_start_tick: int | None = None,
    group_end_tick: int | None = None,
) -> dict[str, Any]:
    repair = {
        "reason": reason,
        "action": action,
        "musicxml_event_id": event.event_id,
        "part_group": event.part_group,
        "part_id": event.part_id,
        "staff": event.staff,
        "voice": event.voice,
        "start_tick": event.start_tick,
        "end_tick": event.end_tick,
        "original_marker": {
            "tuplet_actual": event.tuplet_actual,
            "tuplet_normal": event.tuplet_normal,
            "tuplet_type": event.tuplet_type,
        },
        "tuplet_actual": event.tuplet_actual,
        "tuplet_normal": event.tuplet_normal,
        "tuplet_type": event.tuplet_type,
    }
    if target_voice is not None:
        repair["target_voice"] = target_voice
    if group_start_tick is not None:
        repair["group_start_tick"] = group_start_tick
    if group_end_tick is not None:
        repair["group_end_tick"] = group_end_tick
    return repair


def _normalize_orphan_tuplet_markers(events: list[_RawEvent]) -> list[dict[str, Any]]:
    """Repair only explicit 3:2 markers proven to be import artifacts.

    A tie voice repair can move the first fragment of a MusicXML tuplet into a
    different ScoreVoice while its stop fragment remains in the source voice.
    A contiguous bracket is then restored by reassigning the marker fragments
    to the start voice.  Isolated/incomplete markers are cleared only when
    their plain 48 TPQ durations are serializable.  Same-voice gaps, nested
    boundaries, and unsupported ratios are left untouched so the serializer
    still rejects them.
    """

    repairs: list[dict[str, Any]] = []
    contexts: dict[tuple[str, int, tuple[int, int]], list[_RawEvent]] = {}
    for event in events:
        context = _raw_tuplet_context(event)
        if context is not None and context[2] == (3, 2):
            contexts.setdefault(context, []).append(event)

    # First restore a directly contiguous bracket that was split between
    # voices.  This preserves the 3:2 timing rather than dropping a marker.
    for context, group_events in contexts.items():
        starts = sorted(
            (event for event in group_events if event.tuplet_type == "start"),
            key=lambda event: (event.start_tick, event.end_tick, event.event_id),
        )
        for start in starts:
            same_voice_stops = sorted(
                (
                    event
                    for event in group_events
                    if event.voice == start.voice
                    and event.tuplet_type == "stop"
                    and event.start_tick >= start.end_tick
                ),
                key=lambda event: (event.start_tick, event.end_tick, event.event_id),
            )
            # A later stop may belong to a different tuplet group with the
            # same part/staff/voice.  Treat it as authoritative only when it
            # forms a complete same-voice span; otherwise a directly adjacent
            # cross-voice stop can still prove that the import split one legal
            # group during an unrelated tie voice repair.  A malformed
            # same-voice group with no such cross-voice evidence remains
            # untouched and will still fail serializer validation.
            same_voice_components = [
                stop
                for stop in same_voice_stops
                if _raw_tuplet_span(events, start, stop, allow_cross_voice=False) is not None
            ]
            if same_voice_components:
                continue
            cross_voice_stops = sorted(
                (
                    event
                    for event in group_events
                    if event.voice != start.voice
                    and event.tuplet_type == "stop"
                    and event.start_tick == start.end_tick
                ),
                key=lambda event: (event.start_tick, event.end_tick, event.event_id),
            )
            if len(cross_voice_stops) != 1:
                continue
            stop = cross_voice_stops[0]
            component = _raw_tuplet_span(events, start, stop, allow_cross_voice=True)
            if component is None or _raw_tuplet_voice_has_overlap(events, component, target_voice=start.voice):
                continue
            for event in component:
                if event.voice == start.voice:
                    continue
                repair = _record_tuplet_marker_repair(
                    event,
                    reason="cross_voice_tuplet_marker_reassigned",
                    action="reassigned_tuplet_fragment_to_start_voice",
                    target_voice=start.voice,
                    group_start_tick=start.start_tick,
                    group_end_tick=stop.end_tick,
                )
                event.voice = start.voice
                event.metadata = dict(event.metadata)
                event.metadata["tuplet_boundary_repair"] = repair
                repairs.append(repair)

    # Recompute contexts after voice restoration and identify complete groups.
    contexts = {}
    for event in events:
        context = _raw_tuplet_context(event)
        if context is not None and context[2] == (3, 2):
            contexts.setdefault(context, []).append(event)
    valid_ids: set[int] = set()
    for group_events in contexts.values():
        for start in (event for event in group_events if event.tuplet_type == "start"):
            stops = sorted(
                (
                    event
                    for event in group_events
                    if event.voice == start.voice
                    and event.tuplet_type == "stop"
                    and event.start_tick >= start.end_tick
                ),
                key=lambda event: (event.start_tick, event.end_tick, event.event_id),
            )
            if not stops:
                continue
            component = _raw_tuplet_span(events, start, stops[0], allow_cross_voice=False)
            if component is not None:
                valid_ids.update(id(event) for event in component)

    def same_voice_context(event: _RawEvent) -> list[_RawEvent]:
        return sorted(
            (
                candidate
                for candidate in events
                if candidate.part_group == event.part_group
                and candidate.staff == event.staff
                and candidate.voice == event.voice
            ),
            key=lambda candidate: (candidate.start_tick, candidate.end_tick, candidate.event_id),
        )

    cleared_ids: set[int] = set()
    for group_events in contexts.values():
        for event in sorted(group_events, key=lambda value: (value.start_tick, value.end_tick, value.event_id)):
            if event.tuplet_type not in {"start", "continue", "stop"} or id(event) in valid_ids or id(event) in cleared_ids:
                continue
            voice_events = same_voice_context(event)
            same_voice_group = [candidate for candidate in voice_events if _raw_tuplet_context(candidate) == _raw_tuplet_context(event)]

            starts = [candidate for candidate in same_voice_group if candidate.tuplet_type == "start"]
            stops = [candidate for candidate in same_voice_group if candidate.tuplet_type == "stop"]
            if starts and stops:
                # This is an incomplete or malformed same-voice bracket; the
                # true gap/ratio error remains visible to score_to_jianpu.
                continue
            if event.tuplet_type == "start" and len(starts) != 1:
                continue
            if event.tuplet_type == "stop" and len(stops) != 1:
                continue
            if event.tuplet_type == "start" and stops:
                continue
            if event.tuplet_type == "stop" and starts:
                continue
            if event.tuplet_type == "continue" and (starts or stops):
                continue

            # A boundary with a conflicting ratio in either voice is evidence
            # of a real ratio error, not an orphan marker.
            conflicting = any(
                candidate.part_group == event.part_group
                and candidate.staff == event.staff
                and candidate.tuplet_type in {"start", "stop"}
                and _raw_tuplet_ratio(candidate) != _raw_tuplet_ratio(event)
                and (
                    candidate.voice == event.voice
                    or candidate.start_tick == event.end_tick
                    or candidate.end_tick == event.start_tick
                )
                for candidate in events
            )
            if conflicting:
                continue

            component: list[_RawEvent] = [event]
            if event.tuplet_type == "start":
                index = same_voice_group.index(event)
                while index + 1 < len(same_voice_group) and same_voice_group[index].end_tick == same_voice_group[index + 1].start_tick:
                    index += 1
                    component.append(same_voice_group[index])
            elif event.tuplet_type == "stop":
                index = same_voice_group.index(event)
                while index > 0 and same_voice_group[index - 1].end_tick == same_voice_group[index].start_tick:
                    index -= 1
                    component.insert(0, same_voice_group[index])
            if not all(_raw_duration_is_serializable(candidate) for candidate in component):
                continue
            for candidate in component:
                if id(candidate) in cleared_ids:
                    continue
                repair = _record_tuplet_marker_repair(
                    candidate,
                    reason="orphan_tuplet_marker_cleared",
                    action="cleared_incomplete_tuplet_marker",
                )
                candidate.tuplet_actual = None
                candidate.tuplet_normal = None
                candidate.tuplet_type = None
                candidate.metadata = dict(candidate.metadata)
                candidate.metadata["tuplet_boundary_repair"] = repair
                repairs.append(repair)
                cleared_ids.add(id(candidate))
    return repairs


def _tie_at(event: _RawEvent, pitch_index: int) -> str | None:
    return event.tie_types[pitch_index] if pitch_index < len(event.tie_types) else event.tie


def _logical_pitch_units(events: list[_RawEvent]) -> list[_LogicalPitchUnit]:
    """Collapse each MusicXML pitch-slot tie chain into one matching unit."""

    slots = [(event, pitch_index, pitch) for event in events for pitch_index, pitch in enumerate(event.pitches)]
    starts: dict[tuple[str, int, str, int, int], list[tuple[_RawEvent, int]]] = {}
    for event, pitch_index, pitch in slots:
        starts.setdefault(
            (event.part_group, event.staff, event.voice, pitch, event.start_tick),
            [],
        ).append((event, pitch_index))
    successors: dict[tuple[str, int], tuple[_RawEvent, int]] = {}
    for event, pitch_index, pitch in slots:
        if _tie_at(event, pitch_index) not in {"start", "continue"}:
            continue
        choices = [
            candidate
            for candidate in starts.get(
                (event.part_group, event.staff, event.voice, pitch, event.end_tick),
                [],
            )
            if _tie_at(candidate[0], candidate[1]) in {"stop", "continue"}
        ]
        if choices:
            successors[(event.event_id, pitch_index)] = min(
                choices,
                key=lambda value: (value[0].end_tick, value[0].event_id, value[1]),
            )
    predecessor_keys = {
        (value[0].event_id, value[1])
        for value in successors.values()
    }
    consumed: set[tuple[str, int]] = set()
    units: list[_LogicalPitchUnit] = []

    def append_chain(chain: list[tuple[_RawEvent, int]], pitch: int) -> None:
        unit_id = len(units)
        units.append(
            _LogicalPitchUnit(
                unit_id=unit_id,
                pitch=pitch,
                start_tick=chain[0][0].start_tick,
                end_tick=chain[-1][0].end_tick,
                chain=tuple(chain),
            )
        )

    for event, pitch_index, pitch in sorted(
        slots,
        key=lambda value: (value[0].start_tick, value[0].end_tick, value[0].event_id, value[1]),
    ):
        key = (event.event_id, pitch_index)
        if key in consumed or key in predecessor_keys:
            continue
        chain: list[tuple[_RawEvent, int]] = []
        current: tuple[_RawEvent, int] | None = (event, pitch_index)
        while current is not None:
            current_key = (current[0].event_id, current[1])
            if current_key in consumed or current_key in {(item[0].event_id, item[1]) for item in chain}:
                break
            chain.append(current)
            consumed.add(current_key)
            current = successors.get(current_key)
        append_chain(chain, pitch)
    for event, pitch_index, pitch in slots:
        if (event.event_id, pitch_index) not in consumed:
            append_chain([(event, pitch_index)], pitch)
    return units


def _source_retrigger_split_groups(
    source_rows: list[dict[str, Any]],
    unit_rows: list[_LogicalPitchUnit],
    *,
    scale: float,
    offset: float,
) -> list[list[dict[str, Any]]] | None:
    """Assign extra source retriggers to target units without dropping events."""

    ordered_sources = sorted(
        source_rows,
        key=lambda value: (int(value["start_tick"]), int(value["end_tick"]), int(value["source_index"])),
    )
    ordered_units = sorted(unit_rows, key=lambda value: (value.start_tick, value.end_tick, value.unit_id))
    if len(ordered_sources) <= len(ordered_units):
        return None
    if any(int(source["end_tick"]) <= int(source["start_tick"]) for source in ordered_sources):
        return None
    if any(len(unit.chain) != 1 for unit in ordered_units):
        return None
    if any(
        unit.chain[0][0].tie is not None
        or any(value is not None for value in unit.chain[0][0].tie_types)
        or unit.chain[0][0].tuplet_actual is not None
        or unit.chain[0][0].tuplet_normal is not None
        or unit.chain[0][0].tuplet_type is not None
        or unit.chain[0][0].dots
        for unit in ordered_units
    ):
        return None

    def candidate(
        unit: _LogicalPitchUnit,
        rows: list[dict[str, Any]],
    ) -> bool:
        predicted = [
            (
                scale * int(source["start_tick"]) + offset,
                scale * int(source["end_tick"]) + offset,
            )
            for source in rows
        ]
        if any(
            right[0] < left[1]
            or right[1] < right[0]
            or right[0] - left[1] > MAX_SOURCE_RETRIGGER_INTERNAL_GAP_TICKS
            for left, right in zip(predicted, predicted[1:], strict=False)
        ):
            return False
        if (
            abs(unit.start_tick - predicted[0][0]) > MAX_SOURCE_RETRIGGER_RESIDUAL_TICKS
            or abs(unit.end_tick - predicted[-1][1]) > MAX_SOURCE_RETRIGGER_RESIDUAL_TICKS
        ):
            return False
        source_intervals: list[tuple[int, int]] = []
        for index, (_source, bounds) in enumerate(zip(rows, predicted, strict=True)):
            start = unit.start_tick if index == 0 else round(bounds[0])
            end = unit.end_tick if index == len(rows) - 1 else round(predicted[index + 1][0])
            start = max(unit.start_tick, min(unit.end_tick, start))
            end = max(unit.start_tick, min(unit.end_tick, end))
            if end <= start:
                return False
            source_intervals.append((start, end))
        if any(left[1] != right[0] for left, right in zip(source_intervals, source_intervals[1:], strict=False)):
            return False
        return True

    source_count = len(ordered_sources)
    unit_count = len(ordered_units)
    ways = [[0] * (unit_count + 1) for _ in range(source_count + 1)]
    choices: list[list[tuple[int, int] | None]] = [[None] * (unit_count + 1) for _ in range(source_count + 1)]
    ways[0][0] = 1
    for source_position in range(source_count + 1):
        for unit_position in range(unit_count):
            if not ways[source_position][unit_position]:
                continue
            remaining_sources = source_count - source_position
            remaining_units = unit_count - unit_position
            max_group_size = remaining_sources - (remaining_units - 1)
            for group_size in range(1, max_group_size + 1):
                rows = ordered_sources[source_position : source_position + group_size]
                result = candidate(ordered_units[unit_position], rows)
                if not result:
                    continue
                target_source_position = source_position + group_size
                target_unit_position = unit_position + 1
                ways[target_source_position][target_unit_position] = min(
                    2,
                    ways[target_source_position][target_unit_position]
                    + ways[source_position][unit_position],
                )
                if choices[target_source_position][target_unit_position] is None:
                    choices[target_source_position][target_unit_position] = (source_position, group_size)
    if ways[source_count][unit_count] != 1:
        return None
    groups: list[list[dict[str, Any]]] = []
    source_position, unit_position = source_count, unit_count
    while unit_position:
        choice = choices[source_position][unit_position]
        if choice is None:
            return None
        previous_source_position, group_size = choice
        groups.append(ordered_sources[previous_source_position:source_position])
        source_position = previous_source_position
        unit_position -= 1
    if source_position != 0:
        return None
    groups.reverse()
    return groups


def _split_source_retrigger_events(
    events: list[_RawEvent],
    source_notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split only a proven MusicXML note that hides source retriggers.

    MuseScore may turn adjacent same-pitch MIDI note-ons into one MusicXML
    slot.  A split is accepted only when a complete pitch-order anchor model
    exists, all source rows are positive and strictly ordered, and the
    resulting fragments stay inside the original event boundaries.  Other
    pitch/count mismatches remain unresolved and fail normally.
    """

    units = _logical_pitch_units(events)
    source_by_pitch: dict[int, list[dict[str, Any]]] = {}
    units_by_pitch: dict[int, list[_LogicalPitchUnit]] = {}
    for source in source_notes:
        source_by_pitch.setdefault(int(source["midi"]), []).append(source)
    for unit in units:
        units_by_pitch.setdefault(unit.pitch, []).append(unit)
    mismatched = [pitch for pitch in sorted(source_by_pitch) if len(source_by_pitch[pitch]) > len(units_by_pitch.get(pitch, []))]
    if not mismatched or any(
        pitch not in source_by_pitch or len(source_by_pitch[pitch]) < len(units_by_pitch[pitch])
        for pitch in units_by_pitch
    ):
        return []
    anchor_pairs: list[tuple[dict[str, Any], _LogicalPitchUnit]] = []
    for pitch in sorted(source_by_pitch):
        if len(source_by_pitch[pitch]) != len(units_by_pitch.get(pitch, [])):
            continue
        sources = sorted(
            source_by_pitch[pitch],
            key=lambda value: (int(value["start_tick"]), int(value["end_tick"]), int(value["source_index"])),
        )
        targets = sorted(
            units_by_pitch[pitch],
            key=lambda value: (value.start_tick, value.end_tick, value.unit_id),
        )
        if len({(int(value["start_tick"]), int(value["end_tick"])) for value in sources}) != len(sources):
            return []
        if len({(value.start_tick, value.end_tick) for value in targets}) != len(targets):
            return []
        anchor_pairs.extend(zip(sources, targets, strict=True))
    if len(anchor_pairs) < MIN_SOURCE_ALIGNMENT_MODEL_POINTS:
        return []
    scale, offset = _fit_source_alignment_line(anchor_pairs)
    if not 0.5 <= scale <= 1.8:
        return []
    anchor_residuals = [
        _source_alignment_residuals(pair, scale=scale, offset=offset)
        for pair in anchor_pairs
    ]
    if any(
        start > MAX_SOURCE_RETRIGGER_RESIDUAL_TICKS or end > MAX_SOURCE_RETRIGGER_RESIDUAL_TICKS
        for start, end in anchor_residuals
    ):
        return []

    unit_source_groups: dict[int, list[dict[str, Any]]] = {}
    for pitch in sorted(source_by_pitch):
        sources = sorted(
            source_by_pitch[pitch],
            key=lambda value: (int(value["start_tick"]), int(value["end_tick"]), int(value["source_index"])),
        )
        targets = sorted(
            units_by_pitch[pitch],
            key=lambda value: (value.start_tick, value.end_tick, value.unit_id),
        )
        if len(sources) == len(targets):
            for source, target in zip(sources, targets, strict=True):
                unit_source_groups[target.unit_id] = [source]
            continue
        groups = _source_retrigger_split_groups(sources, targets, scale=scale, offset=offset)
        if groups is None or len(groups) != len(targets):
            return []
        for target, group in zip(targets, groups, strict=True):
            unit_source_groups[target.unit_id] = group

    plans_by_event: dict[str, list[tuple[int, list[dict[str, Any]], _LogicalPitchUnit]]] = {}
    for unit in units:
        rows = unit_source_groups.get(unit.unit_id)
        if rows is None or len(rows) <= 1:
            continue
        event = unit.chain[0][0] if len(unit.chain) == 1 else None
        if event is None:
            return []
        plans_by_event.setdefault(event.event_id, []).append((unit.pitch, rows, unit))
    if not plans_by_event:
        return []
    replacements: list[_RawEvent] = []
    repairs: list[dict[str, Any]] = []
    for event in events:
        plans = plans_by_event.get(event.event_id)
        if not plans:
            replacements.append(event)
            continue
        if (
            event.tie is not None
            or any(value is not None for value in event.tie_types)
            or event.tuplet_actual is not None
            or event.tuplet_normal is not None
            or event.tuplet_type is not None
            or event.dots
        ):
            return []
        pitch_segments: dict[int, list[tuple[int, int, list[int]]]] = {}
        planned_pitches = {pitch for pitch, _rows, _unit in plans}

        def fragment(
            event_id: str,
            start: int,
            end: int,
            pitches: list[int],
            metadata: dict[str, Any],
        ) -> _RawEvent:
            return replace(
                event,
                event_id=event_id,
                start_tick=start,
                end_tick=end,
                pitches=sorted(pitches),
                kind="chord" if len(pitches) > 1 else "note",
                tie=None,
                tie_types=[None] * len(pitches),
                tuplet_actual=None,
                tuplet_normal=None,
                tuplet_type=None,
                dots=0,
                metadata=metadata,
            )

        for pitch, rows, _unit in plans:
            predicted = [
                (
                    scale * int(source["start_tick"]) + offset,
                    scale * int(source["end_tick"]) + offset,
                )
                for source in rows
            ]
            boundaries = [event.start_tick, event.end_tick]
            boundaries.extend(
                round(predicted[index][0])
                for index in range(1, len(predicted))
            )
            boundaries = sorted({max(event.start_tick, min(event.end_tick, value)) for value in boundaries})
            if len(boundaries) != len(rows) + 1:
                return []
            segments: list[tuple[int, int, list[int]]] = []
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
                segment_source_indices = [int(rows[index]["source_index"])]
                if end <= start:
                    return []
                segments.append((start, end, segment_source_indices))
            pitch_segments[pitch] = segments
        intervals: dict[tuple[int, int], list[int]] = {}
        interval_source_indices: dict[tuple[int, int], list[int]] = {}
        for pitch, segments in pitch_segments.items():
            for start, end, source_indices in segments:
                intervals.setdefault((start, end), []).append(pitch)
                interval_source_indices.setdefault((start, end), []).extend(source_indices)
        fragment_ids: list[str] = []
        for fragment_index, ((start, end), pitches) in enumerate(sorted(intervals.items())):
            metadata = dict(event.metadata)
            repair = {
                "reason": "musicxml_event_split_for_source_retriggers",
                "action": "split_imported_pitch_slot_at_proven_source_boundaries",
                "original_musicxml_event_id": event.event_id,
                "original_start_tick": event.start_tick,
                "original_end_tick": event.end_tick,
                "fragment_start_tick": start,
                "fragment_end_tick": end,
                "pitches": sorted(pitches),
                "source_indices": sorted(set(interval_source_indices.get((start, end), []))),
                "anchor_scale": scale,
                "anchor_offset_ticks": offset,
                "timing_preserved_within_original_event": True,
                "one_to_one_source_events": True,
            }
            metadata["source_retrigger_split"] = repair
            event_id = f"{event.event_id}:source-retrigger:{fragment_index}"
            fragment_ids.append(event_id)
            replacements.append(fragment(event_id, start, end, pitches, metadata))
            repairs.append(repair)
        residual_pitches = [pitch for pitch in event.pitches if pitch not in planned_pitches]
        if residual_pitches:
            residual_id = f"{event.event_id}:source-retrigger:residual"
            residual_repair = {
                "reason": "musicxml_event_split_for_source_retriggers",
                "action": "preserve_unplanned_pitch_on_original_interval",
                "original_musicxml_event_id": event.event_id,
                "original_start_tick": event.start_tick,
                "original_end_tick": event.end_tick,
                "pitches": sorted(residual_pitches),
                "source_indices": [],
                "anchor_scale": scale,
                "anchor_offset_ticks": offset,
                "timing_preserved_within_original_event": True,
                "one_to_one_source_events": True,
            }
            residual_metadata = dict(event.metadata)
            residual_metadata["source_retrigger_split"] = residual_repair
            fragment_ids.append(residual_id)
            replacements.append(
                fragment(residual_id, event.start_tick, event.end_tick, residual_pitches, residual_metadata)
            )
        repairs.append(
            {
                "reason": "musicxml_event_split_for_source_retriggers",
                "action": "split_imported_event",
                "original_musicxml_event_id": event.event_id,
                "replacement_musicxml_event_ids": fragment_ids,
                "original_pitches": list(event.pitches),
                "source_retrigger_pitch_count": len(plans),
                "anchor_scale": scale,
                "anchor_offset_ticks": offset,
                "timing_preserved_within_original_event": True,
                "one_to_one_source_events": True,
            }
        )
    events[:] = replacements
    return repairs


def _alignment_unit_group(unit: _LogicalPitchUnit) -> tuple[str, int]:
    event = unit.chain[0][0]
    return event.part_group, event.staff


def _fit_source_alignment_line(
    pairs: list[tuple[dict[str, Any], _LogicalPitchUnit]],
) -> tuple[float, float]:
    """Fit ``musicxml_tick = scale * source_tick + offset`` to pair starts."""

    if not pairs:
        raise ValueError("cannot fit an empty source alignment")
    if len(pairs) == 1:
        source_tick = int(pairs[0][0]["start_tick"])
        return 1.0, float(pairs[0][1].start_tick - source_tick)
    source_values = [int(source["start_tick"]) for source, _unit in pairs]
    xml_values = [int(unit.start_tick) for _source, unit in pairs]
    source_mean = sum(source_values) / len(source_values)
    xml_mean = sum(xml_values) / len(xml_values)
    denominator = sum((value - source_mean) ** 2 for value in source_values)
    scale = (
        sum(
            (source - source_mean) * (xml - xml_mean)
            for source, xml in zip(source_values, xml_values, strict=True)
        )
        / denominator
        if denominator
        else 1.0
    )
    return float(scale), float(xml_mean - scale * source_mean)


def _source_alignment_residuals(
    pair: tuple[dict[str, Any], _LogicalPitchUnit],
    *,
    scale: float,
    offset: float,
) -> tuple[float, float]:
    source, unit = pair
    predicted_start = scale * int(source["start_tick"]) + offset
    predicted_end = scale * int(source["end_tick"]) + offset
    return abs(unit.start_tick - predicted_start), abs(unit.end_tick - predicted_end)


def _source_track_identity_partitions(
    events: list[_RawEvent],
    source_notes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Partition source notes by preserved MIDI lane when imported parts prove it.

    A multi-track performance can be imported as one parent part plus a
    track-specific part.  Pitch-order matching across those parts is unsafe
    when the same pitch occurs in both tracks.  We use a lane only when the
    MusicXML part group contains the lane's original track name and every
    remaining group has exactly one remaining lane.  Any incomplete identity
    evidence returns an explicit failed audit instead of falling back to a
    potentially wrong global match.
    """

    lane_values = [source.get("midi_lane") for source in source_notes]
    if not lane_values or any(value is None for value in lane_values):
        return None
    try:
        lanes = sorted({int(value) for value in lane_values})
    except (TypeError, ValueError):
        return None
    if len(lanes) < 2:
        return None

    def fail(reason: str, **details: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return [], {"applied": False, "reason": reason, **details}

    units = _logical_pitch_units(events)
    groups = sorted({_alignment_unit_group(unit)[0] for unit in units})
    names_by_lane: dict[int, str] = {}
    for source in source_notes:
        lane = int(source["midi_lane"])
        name = source.get("midi_track_name")
        if isinstance(name, str) and name:
            previous = names_by_lane.get(lane)
            if previous is not None and previous != name:
                return fail(
                    "source_midi_lane_track_names_conflict",
                    lane=lane,
                    names=sorted({previous, name}),
                )
            names_by_lane[lane] = name
    group_to_lane: dict[str, int] = {}
    for group in groups:
        candidates = [
            (len(name), lane)
            for lane, name in names_by_lane.items()
            if name.casefold() in group.casefold()
        ]
        if not candidates:
            continue
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0] and candidates[0][1] != candidates[1][1]:
            return fail(
                "source_midi_lane_part_identity_ambiguous",
                part_group=group,
                candidate_lanes=[lane for _length, lane in candidates],
            )
        group_to_lane[group] = candidates[0][1]
    if not group_to_lane:
        return None
    unassigned_lanes = [lane for lane in lanes if lane not in set(group_to_lane.values())]
    unassigned_groups = [group for group in groups if group not in group_to_lane]
    if len(unassigned_lanes) != 1:
        return fail(
            "source_midi_lane_part_identity_incomplete",
            mapped_groups=dict(group_to_lane),
            unassigned_lanes=unassigned_lanes,
            unassigned_groups=unassigned_groups,
        )
    fallback_lane = unassigned_lanes[0]
    for group in unassigned_groups:
        group_to_lane[group] = fallback_lane

    sources_by_lane: dict[int, list[dict[str, Any]]] = {lane: [] for lane in lanes}
    for source in source_notes:
        sources_by_lane[int(source["midi_lane"])].append(source)
    units_by_lane: dict[int, list[_LogicalPitchUnit]] = {lane: [] for lane in lanes}
    for unit in units:
        units_by_lane[group_to_lane[_alignment_unit_group(unit)[0]]].append(unit)
    partitions: list[dict[str, Any]] = []
    used_units: set[int] = set()
    for lane in lanes:
        source_rows = sources_by_lane[lane]
        unit_rows = units_by_lane[lane]
        source_by_pitch: dict[int, list[dict[str, Any]]] = {}
        units_by_pitch: dict[int, list[_LogicalPitchUnit]] = {}
        for source in source_rows:
            source_by_pitch.setdefault(int(source["midi"]), []).append(source)
        for unit in unit_rows:
            units_by_pitch.setdefault(unit.pitch, []).append(unit)
        if {pitch: len(rows) for pitch, rows in source_by_pitch.items()} != {
            pitch: len(rows) for pitch, rows in units_by_pitch.items()
        }:
            return fail(
                "source_midi_lane_pitch_counts_differ",
                lane=lane,
                source_pitch_counts={str(pitch): len(rows) for pitch, rows in sorted(source_by_pitch.items())},
                musicxml_pitch_counts={str(pitch): len(rows) for pitch, rows in sorted(units_by_pitch.items())},
            )
        pairs: list[tuple[dict[str, Any], _LogicalPitchUnit]] = []
        for pitch in sorted(source_by_pitch):
            ordered_sources = sorted(
                source_by_pitch[pitch],
                key=lambda value: (int(value["start_tick"]), int(value["end_tick"]), int(value["source_index"])),
            )
            ordered_units = sorted(
                units_by_pitch[pitch],
                key=lambda value: (value.start_tick, value.end_tick, value.unit_id),
            )
            source_timing = [(int(value["start_tick"]), int(value["end_tick"])) for value in ordered_sources]
            unit_timing = [(value.start_tick, value.end_tick) for value in ordered_units]
            if len(source_timing) != len(set(source_timing)) or len(unit_timing) != len(set(unit_timing)):
                return fail(
                    "source_midi_lane_alignment_not_unique_same_pitch_timing",
                    lane=lane,
                    pitch=pitch,
                )
            pairs.extend(zip(ordered_sources, ordered_units, strict=True))
        if len(pairs) != len(source_rows) or len({unit.unit_id for _source, unit in pairs}) != len(unit_rows):
            return fail(
                "source_midi_lane_alignment_not_one_to_one",
                lane=lane,
                source_count=len(source_rows),
                musicxml_count=len(unit_rows),
            )
        used_units.update(unit.unit_id for _source, unit in pairs)
        partitions.append(
            {
                "lane": lane,
                "pairs": pairs,
            }
        )
    if len(used_units) != len(units):
        return fail(
            "source_midi_lane_alignment_does_not_cover_musicxml_units",
            matched_unit_count=len(used_units),
            musicxml_unit_count=len(units),
        )
    return partitions, {
        "mapped_groups": dict(group_to_lane),
    }


def _track_identity_alignment(
    events: list[_RawEvent],
    source_notes: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]] | None:
    partitioned = _source_track_identity_partitions(events, source_notes)
    if partitioned is None:
        return None
    partitions, partition_audit = partitioned
    if not partitions:
        return {}, {**partition_audit, "applied": False}

    def fail(reason: str, **details: Any) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        return {}, {**partition_audit, "applied": False, "reason": reason, **details}

    source_hints: dict[int, dict[str, Any]] = {}
    models: list[dict[str, Any]] = []
    max_start_residual = 0.0
    max_end_residual = 0.0
    for partition in partitions:
        pairs = list(partition["pairs"])
        if len(pairs) >= MIN_SOURCE_ALIGNMENT_MODEL_POINTS:
            scale, offset = _fit_source_alignment_line(pairs)
            method = "midi_lane_affine_alignment"
        elif len(pairs) == 1:
            source, unit = pairs[0]
            scale = 1.0
            offset = float(unit.start_tick - int(source["start_tick"]))
            method = "midi_lane_singleton_offset_alignment"
        else:
            return fail(
                "source_midi_lane_alignment_has_insufficient_anchors",
                lane=partition["lane"],
                pair_count=len(pairs),
                minimum_model_points=MIN_SOURCE_ALIGNMENT_MODEL_POINTS,
            )
        if not 0.5 <= scale <= 1.8:
            return fail(
                "source_midi_lane_alignment_scale_out_of_bounds",
                lane=partition["lane"],
                scale=scale,
            )
        residuals = [
            _source_alignment_residuals(pair, scale=scale, offset=offset)
            for pair in pairs
        ]
        model_start_residual = max((value[0] for value in residuals), default=0.0)
        model_end_residual = max((value[1] for value in residuals), default=0.0)
        if (
            model_start_residual > MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS
            or model_end_residual > MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS
        ):
            return fail(
                "source_midi_lane_alignment_residual_exceeds_bound",
                lane=partition["lane"],
                max_start_residual_ticks=model_start_residual,
                max_end_residual_ticks=model_end_residual,
            )
        predicted_movements = [
            abs(scale * int(source[key]) + offset - int(source[key]))
            for source, _unit in pairs
            for key in ("start_tick", "end_tick")
        ]
        movement_bound = max(
            MAX_CROSS_PART_ALIGNMENT_MOVEMENT_TICKS,
            math.ceil(max(predicted_movements, default=0.0) + max(
                MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
                MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
            )),
        )
        raw_start = max(
            (abs(unit.start_tick - int(source["start_tick"])) for source, unit in pairs),
            default=0,
        )
        raw_end = max(
            (abs(unit.end_tick - int(source["end_tick"])) for source, unit in pairs),
            default=0,
        )
        if max(raw_start, raw_end) > movement_bound:
            return fail(
                "source_midi_lane_alignment_movement_exceeds_bound",
                lane=partition["lane"],
                max_raw_start_difference_ticks=raw_start,
                max_raw_end_difference_ticks=raw_end,
                movement_bound_ticks=movement_bound,
            )
        model_index = len(models)
        model = {
            "lane": partition["lane"],
            "scale": scale,
            "offset": offset,
            "method": method,
            "pair_count": len(pairs),
            "source_indices": [int(source["source_index"]) for source, _unit in pairs],
            "musicxml_unit_ids": [unit.unit_id for _source, unit in pairs],
            "movement_bound_ticks": movement_bound,
        }
        models.append(model)
        for position, (source, unit) in enumerate(pairs):
            start_residual, end_residual = residuals[position]
            source_hints[int(source["source_index"])] = {
                "scale": scale,
                "offset": offset,
                "aligned_start_tick": round(scale * int(source["start_tick"]) + offset),
                "aligned_end_tick": round(scale * int(source["end_tick"]) + offset),
                "musicxml_unit_id": unit.unit_id,
                "group": {
                    "scope": "midi_lane_identity",
                    "lane": partition["lane"],
                },
                "model_index": model_index,
                "method": method,
                "start_residual_ticks": start_residual,
                "end_residual_ticks": end_residual,
                "start_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
                "end_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
            }
            max_start_residual = max(max_start_residual, start_residual)
            max_end_residual = max(max_end_residual, end_residual)
    if len(source_hints) != len(source_notes) or len({hint["musicxml_unit_id"] for hint in source_hints.values()}) != len(source_notes):
        return fail(
            "source_midi_lane_alignment_is_not_one_to_one",
            source_count=len(source_notes),
            hint_count=len(source_hints),
        )
    classification = (
        "midi_lane_global_offset"
        if all(abs(float(model["scale"]) - 1.0) <= 0.01 for model in models)
        else "midi_lane_affine_scale_and_offset"
    )
    return source_hints, {
        **partition_audit,
        "applied": True,
        "method": "midi_lane_identity_affine_models",
        "classification": classification,
        "pairing": "monotonic_per_pitch_order_with_preserved_midi_lane_identity",
        "global_order_preserved": True,
        "pitch_multiset_equal": True,
        "one_to_one": True,
        "provisional_pair_count": len(source_notes),
        "model_count": len(models),
        "models": models,
        "max_start_residual_ticks": max_start_residual,
        "max_end_residual_ticks": max_end_residual,
        "strict_start_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
        "strict_end_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
        "max_raw_start_difference_ticks": max(
            abs(unit.start_tick - int(source["start_tick"]))
            for partition in partitions
            for source, unit in partition["pairs"]
        ),
        "max_raw_end_difference_ticks": max(
            abs(unit.end_tick - int(source["end_tick"]))
            for partition in partitions
            for source, unit in partition["pairs"]
        ),
        "movement_bound_ticks": max((int(model["movement_bound_ticks"]) for model in models), default=0),
    }


def _cross_part_alignment(
    provisional: list[tuple[dict[str, Any], _LogicalPitchUnit]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]] | None:
    """Build a bounded affine model across MuseScore's synthetic parts.

    MuseScore may turn one imported MIDI instrument into multiple ``part_id``
    values.  A per-staff model is then underdetermined for a short synthetic
    part even though the complete source/pitch assignment is unambiguous.  A
    global model is accepted only after the caller has established exact
    pitch counts, unique same-pitch timing, and global monotonic order.
    """

    if len(provisional) < MIN_SOURCE_ALIGNMENT_MODEL_POINTS:
        return None
    scale, offset = _fit_source_alignment_line(provisional)
    if not 0.5 <= scale <= 1.8:
        return None
    residuals = [
        _source_alignment_residuals(pair, scale=scale, offset=offset)
        for pair in provisional
    ]
    max_start_residual = max(item[0] for item in residuals)
    max_end_residual = max(item[1] for item in residuals)
    if (
        max_start_residual > MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS
        or max_end_residual > MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS
    ):
        return None
    raw_start_movement = max(
        abs(unit.start_tick - int(source["start_tick"]))
        for source, unit in provisional
    )
    raw_end_movement = max(
        abs(unit.end_tick - int(source["end_tick"]))
        for source, unit in provisional
    )
    predicted_movements = [
        abs(scale * int(source[key]) + offset - int(source[key]))
        for source, _unit in provisional
        for key in ("start_tick", "end_tick")
    ]
    movement_bound = max(
        MAX_CROSS_PART_ALIGNMENT_MOVEMENT_TICKS,
        math.ceil(
            max(predicted_movements, default=0.0)
            + max(
                MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
                MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
            )
        ),
    )
    if max(raw_start_movement, raw_end_movement) > movement_bound:
        return None
    needs_reconciliation = any(
        abs(unit.start_tick - int(source["start_tick"])) > 24
        or abs(unit.end_tick - int(source["end_tick"])) > 6
        for source, unit in provisional
    )
    if not needs_reconciliation:
        return None

    source_hints: dict[int, dict[str, Any]] = {}
    model = {
        "group": {"scope": "all_imported_parts"},
        "scale": scale,
        "offset": offset,
        "pair_positions": list(range(len(provisional))),
        "source_indices": [int(source["source_index"]) for source, _unit in provisional],
        "musicxml_unit_ids": [unit.unit_id for _source, unit in provisional],
        "method": "cross_part_global_affine_alignment",
    }
    for position, (source, unit) in enumerate(provisional):
        start_residual, end_residual = residuals[position]
        source_hints[int(source["source_index"])] = {
            "scale": scale,
            "offset": offset,
            "aligned_start_tick": round(scale * int(source["start_tick"]) + offset),
            "aligned_end_tick": round(scale * int(source["end_tick"]) + offset),
            "musicxml_unit_id": unit.unit_id,
            "group": dict(model["group"]),
            "model_index": 0,
            "method": model["method"],
            "start_residual_ticks": start_residual,
            "end_residual_ticks": end_residual,
            "start_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
            "end_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
            "movement_bound_ticks": movement_bound,
        }
    source_hints_count = len(source_hints)
    unit_ids = {hint["musicxml_unit_id"] for hint in source_hints.values()}
    if source_hints_count != len(provisional) or len(unit_ids) != len(provisional):
        return None
    classification = "global_offset" if abs(scale - 1.0) <= 0.01 else "global_affine_scale_and_offset"
    return source_hints, {
        "applied": True,
        "method": "monotonic_pitch_assignment_cross_part_affine_model",
        "classification": classification,
        "pairing": "monotonic_per_pitch_start_end_order",
        "global_order_preserved": True,
        "pitch_multiset_equal": True,
        "one_to_one": True,
        "provisional_pair_count": len(provisional),
        "model_count": 1,
        "models": [
            {
                "group": dict(model["group"]),
                "scale": scale,
                "offset_ticks": offset,
                "method": model["method"],
                "source_indices": list(model["source_indices"]),
                "musicxml_unit_ids": list(model["musicxml_unit_ids"]),
            }
        ],
        "max_start_residual_ticks": max_start_residual,
        "max_end_residual_ticks": max_end_residual,
        "strict_start_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_START_RESIDUAL_TICKS,
        "strict_end_residual_bound_ticks": MAX_CROSS_PART_ALIGNMENT_END_RESIDUAL_TICKS,
        "max_raw_start_difference_ticks": raw_start_movement,
        "max_raw_end_difference_ticks": raw_end_movement,
        "movement_bound_ticks": movement_bound,
    }


def _estimate_source_alignment(
    events: list[_RawEvent],
    source_notes: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Estimate a bounded source-to-MusicXML coordinate reconciliation.

    This is deliberately a diagnostic alignment, never a replacement for
    MusicXML timing.  Pairing is provisional only when every pitch has exactly
    the same number of logical XML units and the monotonic per-pitch order is
    unique.  The resulting models are fitted independently per imported staff;
    residuals and the provisional unit id are retained so the matcher cannot
    turn this into broad nearest-neighbour matching.
    """

    units = _logical_pitch_units(events)
    source_by_pitch: dict[int, list[dict[str, Any]]] = {}
    units_by_pitch: dict[int, list[_LogicalPitchUnit]] = {}
    for source in source_notes:
        source_by_pitch.setdefault(int(source["midi"]), []).append(source)
    for unit in units:
        units_by_pitch.setdefault(int(unit.pitch), []).append(unit)
    track_identity = _track_identity_alignment(events, source_notes)
    if track_identity is not None:
        return track_identity
    if set(source_by_pitch) != set(units_by_pitch):
        return {}, {
            "applied": False,
            "reason": "source_and_musicxml_pitch_sets_differ",
            "source_pitch_counts": {str(pitch): len(rows) for pitch, rows in sorted(source_by_pitch.items())},
            "musicxml_pitch_counts": {str(pitch): len(rows) for pitch, rows in sorted(units_by_pitch.items())},
        }

    provisional: list[tuple[dict[str, Any], _LogicalPitchUnit]] = []
    for pitch in sorted(source_by_pitch):
        source_rows = sorted(
            source_by_pitch[pitch],
            key=lambda value: (int(value["start_tick"]), int(value["end_tick"]), int(value["source_index"])),
        )
        unit_rows = sorted(
            units_by_pitch[pitch],
            key=lambda value: (value.start_tick, value.end_tick, value.unit_id),
        )
        if len(source_rows) != len(unit_rows):
            return {}, {
                "applied": False,
                "reason": "source_and_musicxml_pitch_counts_differ",
                "pitch": pitch,
                "source_count": len(source_rows),
                "musicxml_count": len(unit_rows),
            }
        source_timing_counts: dict[tuple[int, int], int] = {}
        unit_timing_counts: dict[tuple[int, int], int] = {}
        for source in source_rows:
            timing = (int(source["start_tick"]), int(source["end_tick"]))
            source_timing_counts[timing] = source_timing_counts.get(timing, 0) + 1
        for unit in unit_rows:
            timing = (unit.start_tick, unit.end_tick)
            unit_timing_counts[timing] = unit_timing_counts.get(timing, 0) + 1
        if any(count > 1 for count in source_timing_counts.values()) or any(
            count > 1 for count in unit_timing_counts.values()
        ):
            return {}, {
                "applied": False,
                "reason": "provisional_alignment_not_unique_same_pitch_timing",
                "pitch": pitch,
                "source_timing_duplicates": {
                    f"{start}:{end}": count
                    for (start, end), count in sorted(source_timing_counts.items())
                    if count > 1
                },
                "musicxml_timing_duplicates": {
                    f"{start}:{end}": count
                    for (start, end), count in sorted(unit_timing_counts.items())
                    if count > 1
                },
            }
        provisional.extend(zip(source_rows, unit_rows, strict=True))

    source_ordered = sorted(
        provisional,
        key=lambda pair: (
            int(pair[0]["start_tick"]),
            int(pair[0]["end_tick"]),
            int(pair[0]["source_index"]),
        ),
    )
    for previous, current in zip(source_ordered, source_ordered[1:], strict=False):
        previous_source, previous_unit = previous
        current_source, current_unit = current
        if (
            int(current_source["start_tick"]) > int(previous_source["start_tick"])
            and current_unit.start_tick < previous_unit.start_tick
        ):
            return {}, {
                "applied": False,
                "reason": "provisional_alignment_global_order_reversed",
                "previous_source_index": int(previous_source["source_index"]),
                "previous_musicxml_unit_id": previous_unit.unit_id,
                "current_source_index": int(current_source["source_index"]),
                "current_musicxml_unit_id": current_unit.unit_id,
            }

    imported_groups = {_alignment_unit_group(unit) for _source, unit in provisional}
    if len(imported_groups) > 1:
        cross_part = _cross_part_alignment(provisional)
        if cross_part is not None:
            return cross_part

    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], _LogicalPitchUnit]]] = {}
    for pair in provisional:
        grouped.setdefault(_alignment_unit_group(pair[1]), []).append(pair)

    models: list[dict[str, Any]] = []
    source_hints: dict[int, dict[str, Any]] = {}
    for group, group_pairs in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(
            group_pairs,
            key=lambda value: (
                int(value[0]["start_tick"]),
                int(value[0]["midi"]),
                int(value[0]["source_index"]),
            ),
        )
        remaining = list(range(len(ordered)))
        group_models: list[dict[str, Any]] = []
        while len(remaining) >= MIN_SOURCE_ALIGNMENT_MODEL_POINTS:
            candidates: list[tuple[int, float, float, float, list[int]]] = []
            for left_pos, right_pos in itertools.combinations(remaining, 2):
                left_source, left_unit = ordered[left_pos]
                right_source, right_unit = ordered[right_pos]
                left_tick = int(left_source["start_tick"])
                right_tick = int(right_source["start_tick"])
                if left_tick == right_tick:
                    continue
                scale = (right_unit.start_tick - left_unit.start_tick) / (right_tick - left_tick)
                if not 0.5 <= scale <= 1.8:
                    continue
                offset = left_unit.start_tick - scale * left_tick
                inliers = [
                    position
                    for position in remaining
                    if _source_alignment_residuals(ordered[position], scale=scale, offset=offset)[0]
                    <= MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS
                ]
                if len(inliers) < MIN_SOURCE_ALIGNMENT_MODEL_POINTS:
                    continue
                fitted_scale, fitted_offset = _fit_source_alignment_line([ordered[position] for position in inliers])
                inliers = [
                    position
                    for position in remaining
                    if _source_alignment_residuals(
                        ordered[position],
                        scale=fitted_scale,
                        offset=fitted_offset,
                    )[0]
                        <= MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS
                ]
                if len(inliers) < MIN_SOURCE_ALIGNMENT_MODEL_POINTS:
                    continue
                residuals = [
                    _source_alignment_residuals(
                        ordered[position],
                        scale=fitted_scale,
                        offset=fitted_offset,
                    )
                    for position in inliers
                ]
                candidates.append(
                    (
                        len(inliers),
                        max(item[0] for item in residuals),
                        max(item[1] for item in residuals),
                        abs(fitted_scale - 1.0),
                        inliers,
                    )
                )
                # Store the fitted values on the candidate tuple after ranking
                # without recomputing a second provisional model below.
            if not candidates:
                break
            candidates.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))
            _count, _start_residual, _end_residual, _scale_distance, inliers = candidates[0]
            fitted_scale, fitted_offset = _fit_source_alignment_line([ordered[position] for position in inliers])
            model = {
                "group": {"part_group": group[0], "staff": group[1]},
                "scale": fitted_scale,
                "offset": fitted_offset,
                "pair_positions": list(inliers),
                "source_indices": [int(ordered[position][0]["source_index"]) for position in inliers],
                "musicxml_unit_ids": [ordered[position][1].unit_id for position in inliers],
                "method": "affine_staff_alignment",
            }
            group_models.append(model)
            remaining = [position for position in remaining if position not in inliers]

        if remaining:
            if not group_models:
                return {}, {
                    "applied": False,
                    "reason": "insufficient_alignment_model_points",
                    "group": {"part_group": group[0], "staff": group[1]},
                    "candidate_count": len(remaining),
                    "minimum_model_points": MIN_SOURCE_ALIGNMENT_MODEL_POINTS,
                }
            # A short remainder can only be accepted when it has a bounded
            # start/end residual to an already established staff model.  If no
            # model fits, a singleton offset is retained as an explicit,
            # auditable fallback for a unique imported fragment.
            for position in remaining:
                pair = ordered[position]
                choices: list[tuple[float, dict[str, Any]]] = []
                for model in group_models:
                    start_residual, end_residual = _source_alignment_residuals(
                        pair,
                        scale=float(model["scale"]),
                        offset=float(model["offset"]),
                    )
                    if (
                        start_residual <= MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS
                        and end_residual <= MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS
                    ):
                        choices.append((start_residual + end_residual / 10.0, model))
                if choices:
                    model = min(choices, key=lambda value: value[0])[1]
                    model["pair_positions"].append(position)
                    model["source_indices"].append(int(pair[0]["source_index"]))
                    model["musicxml_unit_ids"].append(pair[1].unit_id)
                else:
                    source, unit = pair
                    model = {
                        "group": {"part_group": group[0], "staff": group[1]},
                        "scale": 1.0,
                        "offset": float(unit.start_tick - int(source["start_tick"])),
                        "pair_positions": [position],
                        "source_indices": [int(source["source_index"])],
                        "musicxml_unit_ids": [unit.unit_id],
                        "method": "singleton_offset_alignment",
                    }
                    start_residual, end_residual = _source_alignment_residuals(
                        pair,
                        scale=float(model["scale"]),
                        offset=float(model["offset"]),
                    )
                    if end_residual > MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS:
                        return {}, {
                            "applied": False,
                            "reason": "bounded_alignment_model_not_found",
                            "group": {"part_group": group[0], "staff": group[1]},
                            "source_index": int(source["source_index"]),
                            "start_residual": start_residual,
                            "end_residual": end_residual,
                        }
                group_models.append(model)

        for model_index, model in enumerate(group_models):
            model["model_index"] = model_index
            models.append(model)
            for position in model["pair_positions"]:
                source, unit = ordered[position]
                start_residual, end_residual = _source_alignment_residuals(
                    (source, unit),
                    scale=float(model["scale"]),
                    offset=float(model["offset"]),
                )
                if (
                    start_residual > MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS
                    or end_residual > MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS
                ):
                    return {}, {
                        "applied": False,
                        "reason": "alignment_residual_exceeds_bound",
                        "source_index": int(source["source_index"]),
                        "musicxml_unit_id": unit.unit_id,
                        "start_residual": start_residual,
                        "end_residual": end_residual,
                    }
                source_start_movement = abs(unit.start_tick - int(source["start_tick"]))
                source_end_movement = abs(unit.end_tick - int(source["end_tick"]))
                if max(source_start_movement, source_end_movement) > MAX_SOURCE_ALIGNMENT_MOVEMENT_TICKS:
                    return {}, {
                        "applied": False,
                        "reason": "source_alignment_movement_exceeds_bound",
                        "source_index": int(source["source_index"]),
                        "musicxml_unit_id": unit.unit_id,
                        "start_movement_ticks": source_start_movement,
                        "end_movement_ticks": source_end_movement,
                        "movement_bound_ticks": MAX_SOURCE_ALIGNMENT_MOVEMENT_TICKS,
                    }
                source_hints[int(source["source_index"])] = {
                    "scale": float(model["scale"]),
                    "offset": float(model["offset"]),
                    "aligned_start_tick": round(float(model["scale"]) * int(source["start_tick"]) + float(model["offset"])),
                    "aligned_end_tick": round(float(model["scale"]) * int(source["end_tick"]) + float(model["offset"])),
                    "musicxml_unit_id": unit.unit_id,
                    "group": dict(model["group"]),
                    "model_index": model_index,
                    "method": model["method"],
                    "start_residual_ticks": start_residual,
                    "end_residual_ticks": end_residual,
                    "start_residual_bound_ticks": MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS,
                    "end_residual_bound_ticks": MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS,
                }

    if len(source_hints) != len(source_notes) or len({hint["musicxml_unit_id"] for hint in source_hints.values()}) != len(source_notes):
        return {}, {
            "applied": False,
            "reason": "provisional_alignment_is_not_one_to_one",
            "source_count": len(source_notes),
            "hint_count": len(source_hints),
        }
    raw_start_differences = [
        abs(int(source["start_tick"]) - int(unit.start_tick))
        for source, unit in provisional
    ]
    raw_end_differences = [
        abs(int(source["end_tick"]) - int(unit.end_tick))
        for source, unit in provisional
    ]
    needs_reconciliation = any(
        abs(int(source["start_tick"]) - int(unit.start_tick)) > 24
        or abs(int(source["end_tick"]) - int(unit.end_tick)) > 6
        for source, unit in provisional
    )
    if not needs_reconciliation:
        return {}, {
            "applied": False,
            "reason": "existing_source_alignment_within_strict_window",
            "provisional_pair_count": len(provisional),
        }
    if len(models) == 1:
        classification = (
            "global_offset"
            if abs(float(models[0]["scale"]) - 1.0) <= 0.01
            else "global_affine_scale_and_offset"
        )
    elif all(abs(float(model["scale"]) - 1.0) <= 0.01 for model in models):
        classification = "segmented_offset_or_bar_shift"
    elif all(model["method"] == "affine_staff_alignment" for model in models):
        classification = "segmented_affine_staff_alignment"
    else:
        classification = "segmented_affine_and_offset_alignment"
    sample_pairs = []
    for source, unit in sorted(provisional, key=lambda pair: int(pair[0]["source_index"]))[:10]:
        sample_pairs.append(
            {
                "source_index": int(source["source_index"]),
                "pitch": int(source["midi"]),
                "source_start_tick": int(source["start_tick"]),
                "source_end_tick": int(source["end_tick"]),
                "musicxml_unit_id": unit.unit_id,
                "musicxml_start_tick": unit.start_tick,
                "musicxml_end_tick": unit.end_tick,
                "raw_start_difference_ticks": unit.start_tick - int(source["start_tick"]),
                "raw_end_difference_ticks": unit.end_tick - int(source["end_tick"]),
            }
        )
    return source_hints, {
        "applied": True,
        "method": "monotonic_pitch_assignment_affine_staff_models",
        "classification": classification,
        "pairing": "monotonic_per_pitch_start_end_order",
        "global_order_preserved": True,
        "pitch_multiset_equal": True,
        "one_to_one": True,
        "provisional_pair_count": len(provisional),
        "model_count": len(models),
        "models": [
            {
                "group": model["group"],
                "scale": float(model["scale"]),
                "offset_ticks": float(model["offset"]),
                "method": model["method"],
                "pair_count": len(model["pair_positions"]),
                "source_indices": list(model["source_indices"]),
                "musicxml_unit_ids": list(model["musicxml_unit_ids"]),
            }
            for model in models
        ],
        "max_start_residual_ticks": max(hint["start_residual_ticks"] for hint in source_hints.values()),
        "max_end_residual_ticks": max(hint["end_residual_ticks"] for hint in source_hints.values()),
        "strict_start_residual_bound_ticks": MAX_AFFINE_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS,
        "strict_end_residual_bound_ticks": MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS,
        "source_pitch_counts": {str(pitch): len(rows) for pitch, rows in sorted(source_by_pitch.items())},
        "musicxml_pitch_counts": {str(pitch): len(rows) for pitch, rows in sorted(units_by_pitch.items())},
        "max_raw_start_difference_ticks": max(raw_start_differences),
        "max_raw_end_difference_ticks": max(raw_end_differences),
        "movement_bound_ticks": MAX_SOURCE_ALIGNMENT_MOVEMENT_TICKS,
        "sample_pairs": sample_pairs,
    }


def _alignment_source_item(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "source_index": int(source["source_index"]),
        "source_midi": int(source["midi"]),
        "source_start_tick_480": int(source["start_tick_480"]),
        "source_end_tick_480": int(source["end_tick_480"]),
        "source_start_tick": int(source["start_tick"]),
        "source_end_tick": int(source["end_tick"]),
    }
    if isinstance(source.get("_alignment_hint"), Mapping):
        hint = source["_alignment_hint"]
        result.update(
            {
                "source_alignment_start_tick": int(hint["aligned_start_tick"]),
                "source_alignment_end_tick": int(hint["aligned_end_tick"]),
                "source_alignment_scale": float(hint["scale"]),
                "source_alignment_offset_ticks": float(hint["offset"]),
                "source_alignment_model": str(hint["method"]),
                "source_alignment_musicxml_unit_id": int(hint["musicxml_unit_id"]),
                "source_alignment_start_residual_ticks": float(hint["start_residual_ticks"]),
                "source_alignment_end_residual_ticks": float(hint["end_residual_ticks"]),
            }
        )
    return result


def _alignment_musicxml_item(
    source: Mapping[str, Any],
    unit: _LogicalPitchUnit,
    *,
    reason: str,
    evidence: str,
    score_end_tick: int | None = None,
    category: str = "matched",
) -> dict[str, Any]:
    first_event = unit.chain[0][0]
    chain_end = unit.end_tick
    final_end = chain_end if score_end_tick is None else score_end_tick
    result = _alignment_source_item(source)
    result.update(
        {
            "musicxml_unit_id": unit.unit_id,
            "musicxml_event_id": first_event.event_id,
            "musicxml_event_ids": [item[0].event_id for item in unit.chain],
            "musicxml_start_tick": first_event.start_tick,
            "musicxml_end_tick": first_event.end_tick,
            "musicxml_chain_end_tick": chain_end,
            "score_start_tick": unit.start_tick,
            "score_end_tick": final_end,
            "source_to_score_movement_start_ticks": unit.start_tick - int(source["start_tick"]),
            "source_to_score_movement_end_ticks": final_end - int(source["end_tick"]),
            "musicxml_to_score_movement_start_ticks": unit.start_tick - first_event.start_tick,
            "musicxml_to_score_movement_end_ticks": final_end - chain_end,
            "reason": reason,
            "matching_evidence": evidence,
            "accounting_category": category,
        }
    )
    return result


def _append_instrumental_cleanup_alignment(
    alignment: list[dict[str, Any]],
    performance_metadata: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Account for exact model duplicates removed before MIDI construction."""

    cleanup = performance_metadata.get("instrumental_cleanup") if performance_metadata else None
    if not isinstance(cleanup, Mapping):
        return alignment, len(alignment)
    merged_items = cleanup.get("merged", [])
    if not isinstance(merged_items, list):
        raise MusicXMLStandardizationError("instrumental cleanup report merged must be a list")
    by_source_index = {int(item["source_index"]): item for item in alignment}
    result = list(alignment)
    for merged in merged_items:
        if not isinstance(merged, Mapping):
            raise MusicXMLStandardizationError("instrumental cleanup report contains a non-object merge record")
        try:
            source_index = int(merged["source_index"])
            primary_index = int(merged["primary_source_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MusicXMLStandardizationError("instrumental cleanup merge record lacks source indices") from exc
        if str(merged.get("reason")) != "exact_model_duplicate":
            raise MusicXMLStandardizationError(
                f"unsupported instrumental cleanup merge reason for source index {source_index}"
            )
        primary = by_source_index.get(primary_index)
        if primary is None:
            raise MusicXMLStandardizationError(
                f"instrumental cleanup duplicate {source_index} references missing primary {primary_index}"
            )
        if source_index in by_source_index:
            raise MusicXMLStandardizationError(
                f"instrumental cleanup duplicate source index {source_index} is already aligned"
            )
        item = dict(primary)
        item.update(
            {
                "source_index": source_index,
                "reason": "exact_model_duplicate",
                "matching_evidence": "instrumental_postprocess_exact_model_duplicate",
                "accounting_category": "merged",
                "merged_into_source_index": primary_index,
                "cleanup_normalized_start_sec": merged.get("normalized_start_sec"),
                "cleanup_normalized_end_sec": merged.get("normalized_end_sec"),
            }
        )
        result.append(item)
    result.sort(key=lambda value: int(value["source_index"]))
    return result, len(alignment)


def _source_onset_groups(source_notes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Cluster source onsets within one 48-TPQ tick for chord evidence."""

    by_start: dict[int, list[dict[str, Any]]] = {}
    for source in sorted(source_notes, key=lambda value: (int(value["start_tick"]), int(value["source_index"]))):
        start = int(source["start_tick"])
        group_start = next((candidate for candidate in reversed(sorted(by_start)) if start - candidate <= 1), start)
        by_start.setdefault(group_start, []).append(source)
    return by_start


def _nearest_gap(values: list[int], position: int) -> int:
    if len(values) < 2:
        return 48
    index = bisect_left(values, position)
    candidates: list[int] = []
    if index:
        candidates.append(abs(position - values[index - 1]))
    if index < len(values):
        candidates.append(abs(values[index] - position))
    candidates = [value for value in candidates if value > 0]
    return min(candidates, default=48)


def _match_source_pitch(
    source_rows: list[dict[str, Any]],
    unit_rows: list[_LogicalPitchUnit],
    *,
    source_group_by_index: dict[int, list[dict[str, Any]]],
    unit_starts_by_pitch: dict[int, list[int]],
    alignment_hints: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[dict[int, tuple[_LogicalPitchUnit, dict[str, Any]]], dict[int, int], list[dict[str, Any]] | None]:
    """Monotonic sequence alignment for one MIDI pitch.

    XML units may be skipped (ornaments, rests and notation fragments), but a
    source note may not be skipped unless it is an overlapping duplicate with
    explicit evidence that the same logical XML unit already accounts for it.
    This keeps a distant same-pitch note from being silently attached to a
    later occurrence.
    """

    # Same-onset unisons can come from different piano staves.  Keep the
    # longest span first on both sides so duration ordering stays stable while
    # the DP handles the genuinely sequential part of the stream.
    source_rows = sorted(source_rows, key=lambda value: (int(value["start_tick"]), -int(value["end_tick"]), int(value["source_index"])))
    unit_rows = sorted(unit_rows, key=lambda value: (value.start_tick, -value.end_tick, value.unit_id))
    unit_starts = unit_starts_by_pitch.get(source_rows[0]["midi"], [])
    source_starts = sorted({int(item["start_tick"]) for item in source_rows})
    m = len(unit_rows)

    def group_support(source: Mapping[str, Any], unit: _LogicalPitchUnit) -> tuple[int, int]:
        cohort = source_group_by_index[int(source["source_index"])]
        source_counts: dict[int, int] = {}
        for item in cohort:
            source_counts[int(item["midi"])] = source_counts.get(int(item["midi"]), 0) + 1
        support = 0
        for pitch, count in source_counts.items():
            starts = unit_starts_by_pitch.get(pitch, [])
            # Long-distance matching is permitted only when the other chord
            # pitches share this exact quantized onset cluster.  A broad
            # +/-48 window would let dense nearby passages provide accidental
            # support for a wrong repeated pitch.
            left = bisect_left(starts, unit.start_tick - 1)
            right = bisect_right(starts, unit.start_tick + 1)
            support += min(count, max(0, right - left))
        return support, len(cohort)

    def option(source: Mapping[str, Any], unit: _LogicalPitchUnit) -> dict[str, Any] | None:
        hint = (alignment_hints or {}).get(int(source["source_index"]))
        if hint is not None and int(hint["musicxml_unit_id"]) != unit.unit_id:
            return None
        source_start_tick = int(hint["aligned_start_tick"]) if hint is not None else int(source["start_tick"])
        source_end_tick = int(hint["aligned_end_tick"]) if hint is not None else int(source["end_tick"])
        distance = abs(unit.start_tick - source_start_tick)
        unit_window = min(24, max(6, _nearest_gap(unit_starts, unit.start_tick) // 2 + 6))
        source_window = min(24, max(6, _nearest_gap(source_starts, source_start_tick) // 2 + 6))
        ordinary_window = min(unit_window, source_window)
        support, cohort_count = group_support(source, unit)
        source_duration = max(1, source_end_tick - source_start_tick)
        unit_duration = max(1, unit.end_tick - unit.start_tick)
        overlap = max(
            0,
            min(source_end_tick, unit.end_tick) - max(source_start_tick, unit.start_tick),
        )
        overlap_ratio = overlap / min(source_duration, unit_duration)
        end_distance = abs(unit.end_tick - source_end_tick)
        duration_supported = distance <= 48 and overlap_ratio >= 0.9 and end_distance <= 6
        start_residual_bound = float(
            hint.get("start_residual_bound_ticks", MAX_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS)
        ) if hint is not None else MAX_SOURCE_ALIGNMENT_START_RESIDUAL_TICKS
        end_residual_bound = float(
            hint.get("end_residual_bound_ticks", MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS)
        ) if hint is not None else MAX_SOURCE_ALIGNMENT_END_RESIDUAL_TICKS
        transformed_supported = (
            hint is not None
            and distance <= start_residual_bound
            and end_distance <= end_residual_bound
        )
        if transformed_supported:
            evidence = "monotonic_pitch_affine_alignment"
        elif distance <= ordinary_window:
            evidence = "adaptive_quantization_window"
        elif distance <= 48 and cohort_count >= 2 and support >= min(cohort_count, 3):
            evidence = "chord_onset_group_quantization_window"
        elif duration_supported:
            evidence = "duration_overlap_quantization_window"
        else:
            return None
        cost = distance / 6.0 + abs(unit_duration - source_duration) / 24.0
        if transformed_supported:
            cost += end_distance / 64.0
        if overlap == 0:
            cost += min(2.0, distance / 48.0)
        if distance > 24:
            cost -= min(2.0, support / 10.0)
        return {
            "cost": cost,
            "distance": distance,
            "evidence": evidence,
            "support": support,
            "cohort_count": cohort_count,
        }

    # Exact duplicate/overlap merging is considered only if a full matching
    # pass cannot account for all source rows.  This protects real polyphony
    # whenever MuseScore emitted separate XML units.
    active_rows = list(source_rows)
    merged_into: dict[int, int] = {}
    duplicate_candidates: list[tuple[int, int, int]] = []
    for left_index, left in enumerate(source_rows):
        for right_index in range(left_index + 1, len(source_rows)):
            right = source_rows[right_index]
            if int(left["end_tick"]) <= int(right["start_tick"]) or int(right["end_tick"]) <= int(left["start_tick"]):
                continue
            shared = []
            for unit in unit_rows:
                if option(left, unit) is None or option(right, unit) is None:
                    continue
                left_duration = max(1, int(left["end_tick"]) - int(left["start_tick"]))
                right_duration = max(1, int(right["end_tick"]) - int(right["start_tick"]))
                left_overlap = max(
                    0,
                    min(int(left["end_tick"]), unit.end_tick) - max(int(left["start_tick"]), unit.start_tick),
                )
                right_overlap = max(
                    0,
                    min(int(right["end_tick"]), unit.end_tick) - max(int(right["start_tick"]), unit.start_tick),
                )
                same_onset = abs(int(left["start_tick"]) - int(right["start_tick"])) <= 1
                unit_covers_both = (
                    left_overlap / min(left_duration, max(1, unit.end_tick - unit.start_tick)) >= 0.9
                    and right_overlap / min(right_duration, max(1, unit.end_tick - unit.start_tick)) >= 0.9
                )
                if same_onset or unit_covers_both:
                    shared.append(unit)
            if shared:
                duplicate_candidates.append((int(right["source_index"]), int(left["source_index"]), len(shared)))

    def run_dp(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], _LogicalPitchUnit, dict[str, Any]]] | None:
        count = len(rows)
        inf = float("inf")
        costs = [[inf] * (m + 1) for _ in range(count + 1)]
        choices: list[list[tuple[str, int, int, dict[str, Any] | None] | None]] = [
            [None] * (m + 1) for _ in range(count + 1)
        ]
        for column in range(m + 1):
            costs[0][column] = 0.0
        for row in range(1, count + 1):
            for column in range(1, m + 1):
                if costs[row][column - 1] <= costs[row][column]:
                    costs[row][column] = costs[row][column - 1]
                    choices[row][column] = ("skip_unit", row, column - 1, None)
                match_option = option(rows[row - 1], unit_rows[column - 1])
                if match_option is None:
                    continue
                candidate_cost = costs[row - 1][column - 1] + float(match_option["cost"])
                if candidate_cost < costs[row][column]:
                    costs[row][column] = candidate_cost
                    choices[row][column] = ("match", row - 1, column - 1, match_option)
        if not math.isfinite(costs[count][m]):
            return None
        row, column = count, m
        result: list[tuple[dict[str, Any], _LogicalPitchUnit, dict[str, Any]]] = []
        while row:
            choice = choices[row][column]
            if choice is None:
                return None
            if choice[0] == "skip_unit":
                column = choice[2]
                continue
            result.append((rows[choice[1]], unit_rows[choice[2]], choice[3] or {}))
            row -= 1
            column -= 1
        result.reverse()
        return result

    matched_rows = run_dp(active_rows)
    while matched_rows is None and duplicate_candidates:
        duplicate_source_index, primary_source_index, _ = duplicate_candidates.pop(0)
        remove_at = next(
            (index for index, row in enumerate(active_rows) if int(row["source_index"]) == duplicate_source_index),
            None,
        )
        if remove_at is None:
            continue
        active_rows.pop(remove_at)
        merged_into[duplicate_source_index] = primary_source_index
        matched_rows = run_dp(active_rows)
    if matched_rows is None:
        return {}, merged_into, None
    matched = {int(source["source_index"]): (unit, option_data) for source, unit, option_data in matched_rows}
    return matched, merged_into, matched_rows


def _align_source_notes(
    events: list[_RawEvent],
    source_notes: list[dict[str, Any]],
    *,
    alignment_hints: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Align source notes to logical MusicXML pitch units.

    MuseScore is allowed to move starts and ends during adaptive quantization;
    this function records those movements and never replaces them with source
    performance timing.  Tied XML fragments are first collapsed into one unit,
    then each MIDI pitch is aligned by a monotonic sequence DP.  The only
    source-side accounting other than a match is an explicitly evidenced
    overlapping duplicate that maps to a unit already matched by its primary
    source note.  Otherwise the source note is unresolved and the stage fails.
    """

    if not source_notes:
        return []
    units = _logical_pitch_units(events)
    units_by_pitch: dict[int, list[_LogicalPitchUnit]] = {}
    for unit in units:
        units_by_pitch.setdefault(unit.pitch, []).append(unit)
    for rows in units_by_pitch.values():
        rows.sort(key=lambda value: (value.start_tick, value.end_tick, value.unit_id))
    source_by_pitch: dict[int, list[dict[str, Any]]] = {}
    for source in source_notes:
        source_by_pitch.setdefault(int(source["midi"]), []).append(source)
    source_groups = _source_onset_groups(source_notes)
    source_group_by_index = {
        int(source["source_index"]): group
        for group in source_groups.values()
        for source in group
    }
    unit_starts_by_pitch = {
        pitch: [unit.start_tick for unit in rows]
        for pitch, rows in units_by_pitch.items()
    }
    matched: dict[int, tuple[_LogicalPitchUnit, dict[str, Any]]] = {}
    merged: dict[int, int] = {}
    unresolved: list[dict[str, Any]] = []
    for pitch, rows in source_by_pitch.items():
        pitch_matched, pitch_merged, _ = _match_source_pitch(
            rows,
            units_by_pitch.get(pitch, []),
            source_group_by_index=source_group_by_index,
            unit_starts_by_pitch=unit_starts_by_pitch,
            alignment_hints=alignment_hints,
        )
        matched.update(pitch_matched)
        merged.update(pitch_merged)
        accounted = set(pitch_matched) | set(pitch_merged)
        unresolved.extend(source for source in rows if int(source["source_index"]) not in accounted)

    reports: list[dict[str, Any]] = []
    for source in sorted(source_notes, key=lambda value: int(value["source_index"])):
        source_index = int(source["source_index"])
        if source_index in matched:
            unit, option = matched[source_index]
            score_end = unit.end_tick
            reason = (
                "matched_musicxml_affine_source_alignment"
                if source_index in (alignment_hints or {})
                else "matched_musicxml_tie_chain"
                if len(unit.chain) > 1
                else "matched_musicxml_event"
            )
            overlap_starts = [
                int(other["start_tick"])
                for other in source_notes
                if int(other["source_index"]) != source_index
                and int(other["midi"]) == int(source["midi"])
                and int(source["start_tick"]) < int(other["start_tick"]) < int(source["end_tick"])
            ]
            # Keep the historical, narrowly evidenced same-pitch overlap
            # repair.  It is only allowed when the XML note ends exactly at
            # the next source onset; ordinary adaptive timing is untouched.
            if (
                overlap_starts
                and all(len(item[0].pitches) == 1 for item in unit.chain)
                and unit.end_tick < int(source["end_tick"])
                and any(abs(unit.end_tick - start) <= 1 for start in overlap_starts)
            ):
                score_end = max(unit.start_tick + 1, int(source["end_tick"]))
                unit.chain[-1][0].end_tick = score_end
                reason = "musescore_truncated_source_span_restored"
            reports.append(
                _alignment_musicxml_item(
                    source,
                    unit,
                    reason=reason,
                    evidence=str(option.get("evidence", "monotonic_pitch_alignment")),
                    score_end_tick=score_end,
                )
            )
        elif source_index in merged:
            primary_index = merged[source_index]
            primary = matched.get(primary_index)
            if primary is None:
                unresolved.append(source)
                continue
            item = _alignment_musicxml_item(
                source,
                primary[0],
                reason="merged_overlapping_duplicate_source_note",
                evidence="same_pitch_source_overlap_shared_musicxml_unit",
                category="merged",
            )
            item["merged_into_source_index"] = primary_index
            reports.append(item)
        else:
            item = _alignment_source_item(source)
            item.update(
                {
                    "musicxml_event_id": None,
                    "musicxml_event_ids": [],
                    "score_start_tick": None,
                    "score_end_tick": None,
                    "source_to_score_movement_start_ticks": None,
                    "source_to_score_movement_end_ticks": None,
                    "musicxml_to_score_movement_start_ticks": None,
                    "musicxml_to_score_movement_end_ticks": None,
                    "reason": "unresolved_source_note",
                    "matching_evidence": "no safe monotonic MusicXML candidate",
                    "accounting_category": "unresolved",
                }
            )
            reports.append(item)
    if unresolved:
        unique_unresolved = {int(item["source_index"]): item for item in unresolved}
        details = "; ".join(
            f"index={int(item['source_index'])},midi={int(item['midi'])}"
            for item in sorted(unique_unresolved.values(), key=lambda value: int(value["source_index"]))
        )
        raise MusicXMLStandardizationError(
            "performance metadata contains source notes unresolved after logical MusicXML matching: "
            f"count={len(unique_unresolved)}; {details}"
        )
    return reports


def _allocate_lanes(events: Iterable[_RawEvent]) -> list[list[_RawEvent]]:
    lanes: list[list[_RawEvent]] = []
    lane_ends: list[int] = []
    def lane_order(event: _RawEvent) -> tuple[int, int, int, str]:
        has_incoming_tie = any(
            (event.tie_types[index] if index < len(event.tie_types) else event.tie) in {"stop", "continue"}
            for index in range(len(event.pitches))
        )
        return event.start_tick, 0 if has_incoming_tie else 1, event.end_tick, event.event_id

    for event in sorted(events, key=lane_order):
        tied_pitches = {
            pitch
            for index, pitch in enumerate(event.pitches)
            if (event.tie_types[index] if index < len(event.tie_types) else event.tie) in {"stop", "continue"}
        }
        event_ratio = (
            (event.tuplet_actual, event.tuplet_normal)
            if event.tuplet_actual is not None and event.tuplet_normal is not None
            else None
        )
        tuplet_lane_candidates: list[int] = []
        if event_ratio is not None and event.tuplet_type in {"continue", "stop"}:
            for index, (lane, end) in enumerate(zip(lanes, lane_ends, strict=True)):
                if end != event.start_tick or not lane:
                    continue
                previous = lane[-1]
                previous_ratio = (
                    (previous.tuplet_actual, previous.tuplet_normal)
                    if previous.tuplet_actual is not None and previous.tuplet_normal is not None
                    else None
                )
                if previous_ratio == event_ratio and previous.tuplet_type != "stop":
                    tuplet_lane_candidates.append(index)
        tie_lane_candidates: list[int] = []
        if tied_pitches:
            for index, (lane, end) in enumerate(zip(lanes, lane_ends, strict=True)):
                if end != event.start_tick or not lane:
                    continue
                previous = lane[-1]
                previous_ties = {
                    pitch
                    for tie_index, pitch in enumerate(previous.pitches)
                    if (previous.tie_types[tie_index] if tie_index < len(previous.tie_types) else previous.tie)
                    in {"start", "continue"}
                }
                if tied_pitches.issubset(set(previous.pitches) & previous_ties):
                    tie_lane_candidates.append(index)
        lane_index = (
            tuplet_lane_candidates[0]
            if len(tuplet_lane_candidates) == 1
            else tie_lane_candidates[0]
            if len(tie_lane_candidates) == 1
            else next((index for index, end in enumerate(lane_ends) if end <= event.start_tick), None)
        )
        if lane_index is None:
            lane_index = len(lanes)
            lanes.append([])
            lane_ends.append(0)
        lanes[lane_index].append(event)
        lane_ends[lane_index] = max(lane_ends[lane_index], event.end_tick)
    return lanes


def _score_voice_events(
    lane: list[_RawEvent],
    *,
    total_ticks: int,
    staff: int | None,
    source_voice: str,
) -> list[ScoreNote]:
    result: list[ScoreNote] = []
    cursor = 0
    for event in sorted(lane, key=lambda item: (item.start_tick, item.end_tick, item.event_id)):
        start = max(0, min(total_ticks, event.start_tick))
        end = max(start + 1, min(total_ticks, event.end_tick))
        if start > cursor:
            result.append(
                ScoreNote(
                    start_tick=cursor,
                    duration_tick=start - cursor,
                    midi=None,
                    voice_id=source_voice,
                    source="rest",
                    staff=staff,
                    source_voice=source_voice,
                    metadata={"implicit": True, "reason": "timeline_gap_filled"},
                )
            )
        if start < cursor:
            # Lane allocation should prevent this.  Keep the event visible by
            # moving it to the current cursor and record the repair reason.
            event.metadata["overlap_repair"] = "event_start_moved_to_lane_cursor"
            start = cursor
        if end <= start:
            continue
        pitches = list(event.pitches)
        result.append(
            ScoreNote(
                start_tick=start,
                duration_tick=end - start,
                midi=min(pitches) if pitches else None,
                chord_pitches=pitches,
                voice_id=source_voice,
                source="rest" if event.kind == "rest" else "musicxml",
                staff=staff,
                source_voice=event.voice,
                tie=event.tie,
                # Keep one tie slot per chord pitch.  None means that pitch
                # has no tie and is deliberately not removed or re-indexed.
                tie_types=list(event.tie_types),
                tuplet_actual=event.tuplet_actual,
                tuplet_normal=event.tuplet_normal,
                tuplet_type=event.tuplet_type,
                dots=event.dots,
                measure_number=event.measure_number,
                metadata=dict(event.metadata),
            )
        )
        cursor = end
    if cursor < total_ticks:
        result.append(
            ScoreNote(
                start_tick=cursor,
                duration_tick=total_ticks - cursor,
                midi=None,
                voice_id=source_voice,
                source="rest",
                staff=staff,
                source_voice=source_voice,
                metadata={"implicit": True, "reason": "timeline_tail_filled"},
            )
        )
    if not result:
        result.append(
            ScoreNote(
                start_tick=0,
                duration_tick=max(1, total_ticks),
                midi=None,
                voice_id=source_voice,
                source="rest",
                staff=staff,
                source_voice=source_voice,
                metadata={"implicit": True, "reason": "empty_voice_filled"},
            )
        )
    return result


def _alignment_items_for_events(
    alignment: list[dict[str, Any]],
    event_ids: set[str],
) -> list[dict[str, Any]]:
    if not event_ids:
        return []
    return [
        item
        for item in alignment
        if event_ids.intersection(str(value) for value in item.get("musicxml_event_ids", []))
    ]


def _update_alignment_start(item: dict[str, Any], start_tick: int) -> None:
    item["score_start_tick"] = int(start_tick)
    source_start = item.get("source_start_tick")
    musicxml_start = item.get("musicxml_start_tick")
    if source_start is not None:
        item["source_to_score_movement_start_ticks"] = int(start_tick) - int(source_start)
    if musicxml_start is not None:
        item["musicxml_to_score_movement_start_ticks"] = int(start_tick) - int(musicxml_start)


def _update_alignment_end(item: dict[str, Any], end_tick: int) -> None:
    item["score_end_tick"] = int(end_tick)
    source_end = item.get("source_end_tick")
    musicxml_end = item.get("musicxml_chain_end_tick")
    if source_end is not None:
        item["source_to_score_movement_end_ticks"] = int(end_tick) - int(source_end)
    if musicxml_end is not None:
        item["musicxml_to_score_movement_end_ticks"] = int(end_tick) - int(musicxml_end)


def _event_pitches(event: ScoreNote | None) -> tuple[int, ...]:
    if event is None:
        return ()
    return tuple(event.chord_pitches) if event.chord_pitches else ((event.midi,) if event.midi is not None else ())


def _event_tie_values(event: ScoreNote | None) -> list[str | None]:
    pitches = _event_pitches(event)
    if not pitches or event is None:
        return []
    if event.tie_types and len(event.tie_types) == len(pitches):
        return list(event.tie_types)
    return [event.tie] * len(pitches)


def _event_musicxml_id(event: ScoreNote) -> str | None:
    value = event.metadata.get("musicxml_event_id")
    return str(value) if value is not None else None


def _event_notated_duration(event: ScoreNote) -> Fraction:
    """Return the duration passed to the serializer for one event.

    Explicit tuplets are encoded with their performed duration on the shared
    score grid, while the jianpu serializer formats the corresponding nominal
    duration inside the tuplet bracket.  Dot hints must be checked against
    that nominal duration rather than the performed duration.
    """

    duration = Fraction(event.duration_tick)
    actual = event.tuplet_actual
    normal = event.tuplet_normal
    if actual is not None and normal is not None:
        duration *= Fraction(actual, normal)
    return duration


def _matching_dot_counts(event: ScoreNote) -> list[int]:
    """Return explicit dot counts that exactly describe an event's duration."""

    duration = _event_notated_duration(event)
    matches: list[int] = []
    for dots in range(1, 4):
        for denominator in (1, 2, 4, 8, 16, 32, 64):
            base = Fraction(SCORE_QUARTER_TICKS * 4, denominator)
            dotted = base * Fraction(2 ** (dots + 1) - 1, 2**dots)
            if duration == dotted:
                matches.append(dots)
                break
    return matches


def _score_duration_is_serializable(event: ScoreNote) -> bool:
    """Whether an event can be emitted with no explicit dot hint.

    This mirrors the serializer's exact duration atoms locally so the fine
    fragment repair can still run after an invalid MusicXML dot is cleared.
    It intentionally does not alter serializer validation or accept a
    fractional tuplet nominal duration.
    """

    duration = _event_notated_duration(event)
    if duration.denominator != 1 or duration <= 0:
        return False
    atoms = {
        int(Fraction(SCORE_QUARTER_TICKS * 4, denominator) * Fraction(2 ** (dots + 1) - 1, 2**dots))
        for denominator in (1, 2, 4, 8, 16, 32, 64)
        for dots in range(4)
        if (Fraction(SCORE_QUARTER_TICKS * 4, denominator) * Fraction(2 ** (dots + 1) - 1, 2**dots)).denominator == 1
    }
    remaining = int(duration)
    while remaining:
        atom = max((value for value in atoms if value <= remaining), default=0)
        if atom <= 0:
            return False
        remaining -= atom
    return True


def _repair_explicit_dots(
    voices: list[ScoreVoice],
) -> tuple[list[ScoreVoice], list[dict[str, Any]]]:
    """Validate MusicXML dot hints after conversion to the 48 TPQ score grid.

    MusicXML duration values are rounded before they become ``ScoreNote``
    events.  A dot attached to a finer renderer fragment can therefore stop
    describing the final event, even though the source XML carried a legal
    dot.  Preserve an exact hint, recompute it when another dotted atom fits,
    and otherwise clear it while retaining the authoritative timing and ties.
    """

    repairs: list[dict[str, Any]] = []
    repaired_voices: list[ScoreVoice] = []
    for voice in voices:
        events: list[ScoreNote] = []
        for event in voice.events:
            if event.dots <= 0:
                events.append(event)
                continue
            matching = _matching_dot_counts(event)
            if event.dots in matching:
                events.append(event)
                continue
            repaired_dots = matching[0] if matching else 0
            action = "recomputed" if repaired_dots else "cleared"
            reason = f"explicit_dots_{action}_after_duration_validation"
            repair = {
                "reason": reason,
                "action": f"{action}_explicit_dots",
                "voice_id": voice.voice_id,
                "musicxml_event_id": _event_musicxml_id(event),
                "start_tick": event.start_tick,
                "duration_tick": event.duration_tick,
                "original_dots": event.dots,
                "repaired_dots": repaired_dots,
                "notated_duration_ticks": (
                    int(_event_notated_duration(event))
                    if _event_notated_duration(event).denominator == 1
                    else str(_event_notated_duration(event))
                ),
                "tuplet_actual": event.tuplet_actual,
                "tuplet_normal": event.tuplet_normal,
            }
            metadata = dict(event.metadata)
            metadata["notation_grid_repair"] = repair
            events.append(event.model_copy(update={"dots": repaired_dots, "metadata": metadata}))
            repairs.append(repair)
        repaired_voices.append(voice.model_copy(update={"events": events}))
    return repaired_voices, repairs


_FINE_GRID_TUPLET_RATIOS: tuple[tuple[int, int], ...] = ((3, 2), (3, 1))


def _fine_tuplet_candidate(
    events: list[ScoreNote],
    start: int,
    end: int,
    ratio: tuple[int, int],
) -> bool:
    """Check one contiguous, exact fine-grid tuplet span.

    The worker has already supplied the authoritative tick boundaries.  This
    helper only annotates a span when every member can be represented by an
    integer nominal duration under the selected ratio and at least one member
    is otherwise outside the ordinary 48-TPQ jianpu atom grid.  Tied or
    already-tupleted events are deliberately excluded: their semantics must
    remain driven by MusicXML markers rather than a renderer repair.
    """

    if start < 0 or end > len(events) or end - start < 2:
        return False
    group = events[start:end]
    if any(
        event.tuplet_actual is not None
        or event.tuplet_normal is not None
        or event.tuplet_type is not None
        or event.tie is not None
        or any(value is not None for value in event.tie_types)
        or event.dots
        or event.metadata.get("fine_grid_tuplet")
        or event.metadata.get("fine_grid_tuplet_group_id")
        for event in group
    ):
        return False
    if any(left.end_tick != right.start_tick for left, right in zip(group, group[1:])):
        return False
    # Do not absorb an otherwise serializable voice tail or bar rest merely
    # because a tiny fragment precedes it.  The repair is for the compact
    # MuseScore binary fragment runs (the production examples span at most a
    # few 48-TPQ ticks), while the existing bounded note/rest repair remains
    # responsible for an isolated one/two-tick event.
    if sum(event.duration_tick for event in group) > 24 or max(event.duration_tick for event in group) > 24:
        return False
    actual, normal = ratio
    nominal = [Fraction(event.duration_tick * actual, normal) for event in group]
    if any(value.denominator != 1 or value <= 0 for value in nominal):
        return False
    return any(
        event.duration_tick < MIN_JIANPU_ATOM_TICKS or not _score_duration_is_serializable(event)
        for event in group
    )


def _fine_tuplet_repair(
    events: list[ScoreNote],
    index: int,
    *,
    voice_id: str,
) -> tuple[list[ScoreNote], dict[str, Any]] | None:
    """Annotate an exact tuplet span for a failing fine-grid event.

    The candidate order prefers a complete 3-member 3:2 span.  A 2-member
    3:2 span is retained for a MuseScore fragment split at a rest boundary;
    if the odd 1-tick residue cannot be a 3:2 member, a bounded 3:1
    explicit ratio is used instead.  These are serialized as explicit
    ``actual:normal[`` tokens and never change event timing.
    """

    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for ratio in _FINE_GRID_TUPLET_RATIOS:
        # A three-member triplet is the strongest evidence and is tried first
        # for each ratio, regardless of whether the failing event is first or
        # middle in the imported run.
        for start in (index - 1, index, index - 2):
            end = start + 3
            if start <= index < end:
                candidates.append((start, end, ratio))
        for start in (index - 1, index):
            end = start + 2
            if start <= index < end:
                candidates.append((start, end, ratio))
    # Stable de-duplication keeps the preference above auditable.
    seen: set[tuple[int, int, tuple[int, int]]] = set()
    for start, end, ratio in candidates:
        key = (start, end, ratio)
        if key in seen:
            continue
        seen.add(key)
        if not _fine_tuplet_candidate(events, start, end, ratio):
            continue
        actual, normal = ratio
        event_ids: list[str] = []
        durations: list[int] = []
        for event in events[start:end]:
            event_ids.append(_event_musicxml_id(event) or f"score:{event.start_tick}:{event.end_tick}")
            durations.append(event.duration_tick)
        group_id = ":".join(
            [
                "fine-grid",
                voice_id,
                str(events[start].start_tick),
                str(events[end - 1].end_tick),
                str(actual),
                str(normal),
                *event_ids,
            ]
        )
        for offset, event in enumerate(events[start:end]):
            metadata = dict(event.metadata)
            metadata["fine_grid_tuplet"] = True
            metadata["fine_grid_tuplet_ratio"] = {"actual": actual, "normal": normal}
            metadata["fine_grid_tuplet_group_id"] = group_id
            metadata["fine_grid_tuplet_member_index"] = offset
            events[start + offset] = event.model_copy(
                update={
                    "tuplet_actual": actual,
                    "tuplet_normal": normal,
                    "tuplet_type": "start" if offset == 0 else "stop" if offset == end - start - 1 else None,
                    "metadata": metadata,
                }
            )
        repair = {
            "reason": "fine_grid_fragment_encoded_as_explicit_tuplet",
            "action": "annotate_exact_fine_grid_tuplet",
            "voice_id": voice_id,
            "musicxml_event_ids": event_ids,
            "group_id": group_id,
            "member_count": len(event_ids),
            "start_tick": events[start].start_tick,
            "end_tick": events[end - 1].end_tick,
            "duration_ticks": durations,
            "tuplet_actual": actual,
            "tuplet_normal": normal,
            "movement_ticks": 0,
            "timing_preserved": True,
            "ordinary_atom_failure": True,
        }
        for event_index in range(start, end):
            metadata = dict(events[event_index].metadata)
            metadata["fine_grid_tuplet_repair"] = repair
            events[event_index] = events[event_index].model_copy(update={"metadata": metadata})
        return events, repair
    return None


def _fine_singleton_tuplet_repair(
    events: list[ScoreNote],
    index: int,
    *,
    voice_id: str,
) -> tuple[list[ScoreNote], dict[str, Any]] | None:
    """Encode one unsupported fine-grid event without consuming neighbors.

    The pinned renderer has exact ordinary atoms at 3 ticks and above only
    when the duration is a sum of those atoms.  A single 1, 2, 4, or 5 tick
    event can instead use one explicit 3:1 bracket: its nominal body is 3,
    6, 12, or 15 ticks.  Its event boundary and tie fields remain exactly as
    imported; no adjacent event is inferred into this representation.
    """

    event = events[index]
    if event.duration_tick not in {1, 2, 4, 5}:
        return None
    if (
        event.tuplet_actual is not None
        or event.tuplet_normal is not None
        or event.tuplet_type is not None
        or event.metadata.get("fine_grid_tuplet")
        or event.metadata.get("fine_grid_tuplet_group_id")
        or event.dots
        or event.metadata.get("implicit")
    ):
        return None
    actual, normal = 3, 1
    nominal_duration = event.duration_tick * actual
    candidate = event.model_copy(
        update={"tuplet_actual": actual, "tuplet_normal": normal, "tuplet_type": "start"}
    )
    if not _score_duration_is_serializable(candidate):
        return None
    event_id = _event_musicxml_id(event) or f"score:{event.start_tick}:{event.end_tick}"
    group_id = ":".join(
        [
            "fine-grid-single",
            voice_id,
            str(event.start_tick),
            str(event.end_tick),
            str(actual),
            str(normal),
            event_id,
        ]
    )
    repair = {
        "reason": "fine_grid_singleton_encoded_as_explicit_tuplet",
        "action": "annotate_exact_fine_grid_singleton_tuplet",
        "voice_id": voice_id,
        "musicxml_event_ids": [event_id],
        "group_id": group_id,
        "member_count": 1,
        "singleton": True,
        "start_tick": event.start_tick,
        "end_tick": event.end_tick,
        "original_start_tick": event.start_tick,
        "original_end_tick": event.end_tick,
        "duration_ticks": [event.duration_tick],
        "nominal_duration_ticks": [nominal_duration],
        "ratio": {"actual": actual, "normal": normal},
        "tuplet_actual": actual,
        "tuplet_normal": normal,
        "movement_ticks": 0,
        "timing_preserved": True,
        "ordinary_atom_failure": True,
    }
    metadata = dict(event.metadata)
    metadata.update(
        {
            "fine_grid_tuplet": True,
            "fine_grid_tuplet_single": True,
            "fine_grid_tuplet_ratio": {"actual": actual, "normal": normal},
            "fine_grid_tuplet_group_id": group_id,
            "fine_grid_tuplet_member_index": 0,
            "fine_grid_tuplet_repair": repair,
        }
    )
    events[index] = candidate.model_copy(update={"metadata": metadata})
    return events, repair


def _is_fine_grid_tuplet_member(event: ScoreNote | None) -> bool:
    return bool(event is not None and event.metadata.get("fine_grid_tuplet"))


def _validate_fine_grid_tuplet_groups(events: list[ScoreNote], *, voice_id: str) -> None:
    """Validate inferred fine-grid groups after all bounded repairs.

    Inferred tuplets are a renderer representation of already-authoritative
    event boundaries.  Later atom repairs must therefore never consume a group
    member or move one of its boundaries.  Keep this check close to the repair
    pass so a future mutation fails explicitly instead of reaching the
    serializer as a misleading nested or unclosed group.
    """

    grouped: dict[str, list[ScoreNote]] = {}
    repairs: dict[str, Mapping[str, Any]] = {}
    for event in events:
        marked = _is_fine_grid_tuplet_member(event)
        group_id = event.metadata.get("fine_grid_tuplet_group_id")
        if not marked and group_id is None:
            continue
        if not marked or not isinstance(group_id, str) or not group_id:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet member in voice {voice_id} has incomplete group metadata "
                f"at tick {event.start_tick}"
            )
        repair = event.metadata.get("fine_grid_tuplet_repair")
        if not isinstance(repair, Mapping) or repair.get("group_id") != group_id:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} has inconsistent repair metadata"
            )
        grouped.setdefault(group_id, []).append(event)
        previous = repairs.get(group_id)
        if previous is not None and previous != repair:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} has conflicting member metadata"
            )
        repairs[group_id] = repair

    spans: list[tuple[int, int, str]] = []
    for group_id, members in grouped.items():
        repair = repairs[group_id]
        expected_count = int(repair.get("member_count", 0))
        expected_durations = [int(value) for value in repair.get("duration_ticks", [])]
        expected_ids = [str(value) for value in repair.get("musicxml_event_ids", [])]
        if len(members) != expected_count or len(expected_durations) != expected_count:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} lost a member"
            )
        actual_ids = [
            _event_musicxml_id(event) or f"score:{event.start_tick}:{event.end_tick}"
            for event in members
        ]
        if actual_ids != expected_ids or [event.duration_tick for event in members] != expected_durations:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} changed its event boundaries"
            )
        actual, normal = int(repair["tuplet_actual"]), int(repair["tuplet_normal"])
        if any(
            event.tuplet_actual != actual
            or event.tuplet_normal != normal
            or event.metadata.get("fine_grid_tuplet_member_index") != index
            for index, event in enumerate(members)
        ):
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} has inconsistent ratios or member order"
            )
        expected_boundaries = (
            ["start"]
            if repair.get("singleton") is True
            else ["start", *([None] * (expected_count - 2)), "stop"]
        )
        if [event.tuplet_type for event in members] != expected_boundaries:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} is not closed"
            )
        if any(left.end_tick != right.start_tick for left, right in zip(members, members[1:])):
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} has a timeline gap or overlap"
            )
        start_tick, end_tick = members[0].start_tick, members[-1].end_tick
        if (
            int(repair.get("start_tick", -1)) != start_tick
            or int(repair.get("end_tick", -1)) != end_tick
            or sum(event.duration_tick for event in members) != end_tick - start_tick
        ):
            raise MusicXMLStandardizationError(
                f"fine-grid tuplet group {group_id!r} in voice {voice_id} did not preserve total ticks"
            )
        spans.append((start_tick, end_tick, group_id))

    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise MusicXMLStandardizationError(
                f"fine-grid tuplets in voice {voice_id} overlap or nest "
                f"at tick {current[0]} ({previous[2]!r}, {current[2]!r})"
            )


def _set_tie_after_merge(event: ScoreNote, merged_pitches: set[int]) -> ScoreNote:
    """Close or clear each surviving tie after a terminal fragment is removed.

    A preceding ``start`` has no incoming tie after its ``stop`` successor is
    removed, while a preceding ``continue``/``stop`` still closes a tie that
    began in an earlier event.  Chord tie slots are updated independently.
    """

    pitches = _event_pitches(event)
    values = _event_tie_values(event)
    if not pitches:
        return event.model_copy(update={"tie": None, "tie_types": []})
    for index, pitch in enumerate(pitches):
        if pitch not in merged_pitches:
            continue
        values[index] = "stop" if values[index] in {"stop", "continue"} else None
    if len(pitches) > 1:
        tie_types = values
    else:
        tie_types = values if values and values[0] is not None else []
    present = [value for value in tie_types if value is not None]
    tie = present[0] if present and len(present) == len(tie_types) and all(value == present[0] for value in present) else None
    return event.model_copy(update={"tie": tie, "tie_types": tie_types})


def _repair_fine_score_events(
    voices: list[ScoreVoice],
    alignment: list[dict[str, Any]],
    *,
    total_ticks: int,
) -> tuple[list[ScoreVoice], list[dict[str, Any]]]:
    """Make rounded finer-grid fragments representable by jianpu-ly.

    The raw worker events and the 48 TPQ rounding remain authoritative.  This
    pass only handles a renderer boundary: a one/two-tick terminal tie piece
    is folded into its preceding same-pitch event, and a following tiny note
    may move back to the preceding lane boundary when that yields the minimum
    three-tick atom.  Every changed boundary is written into source alignment.
    """

    repairs: list[dict[str, Any]] = []
    repaired_voices: list[ScoreVoice] = []
    for voice in voices:
        events = list(voice.events)
        index = 0
        while index < len(events):
            current = events[index]
            previous = events[index - 1] if index else None

            # A preceding tied fragment may have been shortened by one tick
            # to reach the jianpu atom grid.  If the next unmarked tiny event
            # still begins at the old boundary, move only that event back to
            # the repaired lane boundary before considering inferred tuplets.
            # This keeps the voice contiguous without inventing a tuplet from
            # a gap, and records the same bounded one/two-tick movement as the
            # ordinary tiny-note repair below.
            current_pitches = _event_pitches(current)
            current_ties = _event_tie_values(current)
            if (
                previous is not None
                and previous.end_tick < current.start_tick
                and current_pitches
                and current.duration_tick < MIN_JIANPU_ATOM_TICKS
                and current.tuplet_actual is None
                and current.tuplet_normal is None
                and current.tuplet_type is None
                and not any(value is not None for value in current_ties)
            ):
                target_start = previous.end_tick
                movement = target_start - current.start_tick
                if (
                    target_start < current.start_tick
                    and current.end_tick - target_start >= MIN_JIANPU_ATOM_TICKS
                    and abs(movement) <= MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS
                ):
                    current_id = _event_musicxml_id(current)
                    updated = current.model_copy(
                        update={
                            "start_tick": target_start,
                            "duration_tick": current.end_tick - target_start,
                        }
                    )
                    metadata = dict(updated.metadata)
                    metadata["notation_grid_repair"] = {
                        "reason": "fine_grid_note_shifted_to_jianpu_atom",
                        "action": "move_tiny_event_across_repair_gap_to_previous_boundary",
                        "original_start_tick": current.start_tick,
                        "previous_end_tick": previous.end_tick,
                    }
                    events[index] = updated.model_copy(update={"metadata": metadata})
                    if current_id is not None:
                        for item in _alignment_items_for_events(alignment, {current_id}):
                            _update_alignment_start(item, target_start)
                    repairs.append(
                        {
                            "reason": "fine_grid_note_shifted_to_jianpu_atom",
                            "action": "move_tiny_event_across_repair_gap_to_previous_boundary",
                            "voice_id": voice.voice_id,
                            "musicxml_event_id": current_id,
                            "pitch": current_pitches[0] if len(current_pitches) == 1 else None,
                            "pitches": list(current_pitches),
                            "original_start_tick": current.start_tick,
                            "repaired_start_tick": target_start,
                            "previous_end_tick": previous.end_tick,
                            "end_tick": current.end_tick,
                            "movement_ticks": movement,
                            "bounded_by_ticks": MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS,
                        }
                    )
                    index += 1
                    continue

            # A rounded renderer fragment can be four ticks even though the
            # source duration was a dotted 3/32-quarter value.  Once its dot
            # hint is cleared, let the existing bounded tie repair fold it
            # into the preceding logical note.  Valid atom durations keep the
            # normal timing and tie path unchanged.
            dot_repair = current.metadata.get("notation_grid_repair")
            requires_fragment_repair = (
                isinstance(dot_repair, Mapping)
                and str(dot_repair.get("reason", "")).startswith("explicit_dots_")
                and not _score_duration_is_serializable(current)
            )
            if (
                current.duration_tick >= MIN_JIANPU_ATOM_TICKS
                and _score_duration_is_serializable(current)
                and not requires_fragment_repair
            ):
                index += 1
                continue

            # Preserve an exact 4/8-tick fragment as a tuplet before the older
            # bounded atom repairs get a chance to move or merge it.  This is
            # strictly an annotation: event boundaries, pitches, ties, and
            # measure totals remain unchanged.
            tuplet_repair = _fine_tuplet_repair(
                events,
                index,
                voice_id=voice.voice_id,
            )
            if tuplet_repair is not None:
                events, repair = tuplet_repair
                repairs.append(repair)
                index += 1
                continue
            singleton_repair = _fine_singleton_tuplet_repair(
                events,
                index,
                voice_id=voice.voice_id,
            )
            if singleton_repair is not None:
                events, repair = singleton_repair
                repairs.append(repair)
                index += 1
                continue

            previous = events[index - 1] if index else None
            current_pitches = _event_pitches(current)
            previous_pitches = _event_pitches(previous)
            current_ties = _event_tie_values(current)
            current_id = _event_musicxml_id(current)
            previous_id = _event_musicxml_id(previous) if previous is not None else None

            # A terminal tied fragment is part of the preceding logical note.
            # Snap the complete tied span to the nearest 3-tick boundary and
            # close the tie on the surviving event.
            if (
                current_pitches
                and previous is not None
                and not _is_fine_grid_tuplet_member(current)
                and not _is_fine_grid_tuplet_member(previous)
                and previous_pitches == current_pitches
                and previous.end_tick == current.start_tick
                and all(value in {"stop", "continue"} for value in current_ties)
            ):
                combined_ticks = current.end_tick - previous.start_tick
                snapped_duration = max(
                    MIN_JIANPU_ATOM_TICKS,
                    round(combined_ticks / MIN_JIANPU_ATOM_TICKS) * MIN_JIANPU_ATOM_TICKS,
                )
                snapped_end = min(total_ticks, previous.start_tick + snapped_duration)
                movement = snapped_end - current.end_tick
                if abs(movement) <= MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS and snapped_end > previous.start_tick:
                    event_ids = {value for value in (previous_id, current_id) if value is not None}
                    updated_previous = _set_tie_after_merge(
                        previous.model_copy(update={"duration_tick": snapped_end - previous.start_tick}),
                        set(current_pitches),
                    )
                    updated_metadata = dict(updated_previous.metadata)
                    updated_metadata["notation_grid_repair"] = {
                        "reason": "fine_grid_tie_fragment_merged_for_jianpu_atom",
                        "removed_musicxml_event_id": current_id,
                    }
                    updated_previous = updated_previous.model_copy(update={"metadata": updated_metadata})
                    events[index - 1] = updated_previous
                    del events[index]
                    for item in _alignment_items_for_events(alignment, event_ids):
                        _update_alignment_end(item, snapped_end)
                    repairs.append(
                        {
                            "reason": "fine_grid_tie_fragment_merged_for_jianpu_atom",
                            "action": "removed_terminal_tie_fragment",
                            "voice_id": voice.voice_id,
                            "musicxml_event_ids": sorted(event_ids),
                            "pitch": current_pitches[0] if len(current_pitches) == 1 else None,
                            "pitches": list(current_pitches),
                            "original_start_tick": current.start_tick,
                            "original_end_tick": current.end_tick,
                            "repaired_end_tick": snapped_end,
                            "movement_ticks": movement,
                            "bounded_by_ticks": MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS,
                        }
                    )
                    continue

            # A tiny note after a preceding event can use that event's end as
            # its onset when the resulting atom is exactly representable.  In
            # the MuseScore fragment that triggered this repair this is a
            # two-tick bass note shifted one tick earlier to become 3 ticks.
            if (
                previous is not None
                and not _is_fine_grid_tuplet_member(current)
                and not _is_fine_grid_tuplet_member(previous)
                and previous.end_tick <= current.end_tick
            ):
                target_start = max(previous.end_tick, current.end_tick - MIN_JIANPU_ATOM_TICKS)
                movement = target_start - current.start_tick
                if (
                    target_start <= current.start_tick
                    and current.end_tick - target_start >= MIN_JIANPU_ATOM_TICKS
                    and abs(movement) <= MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS
                ):
                    updated = current.model_copy(
                        update={"start_tick": target_start, "duration_tick": current.end_tick - target_start}
                    )
                    metadata = dict(updated.metadata)
                    metadata["notation_grid_repair"] = {
                        "reason": "fine_grid_note_shifted_to_jianpu_atom",
                        "original_start_tick": current.start_tick,
                    }
                    events[index] = updated.model_copy(update={"metadata": metadata})
                    if current_id is not None:
                        for item in _alignment_items_for_events(alignment, {current_id}):
                            _update_alignment_start(item, target_start)
                    repairs.append(
                        {
                            "reason": "fine_grid_note_shifted_to_jianpu_atom",
                            "action": "move_note_onset_to_previous_lane_boundary",
                            "voice_id": voice.voice_id,
                            "musicxml_event_id": current_id,
                            "pitch": current_pitches[0] if len(current_pitches) == 1 else None,
                            "pitches": list(current_pitches),
                            "original_start_tick": current.start_tick,
                            "repaired_start_tick": target_start,
                            "end_tick": current.end_tick,
                            "movement_ticks": movement,
                            "bounded_by_ticks": MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS,
                        }
                    )
                    index += 1
                    continue

            # A standalone finer-grid note can round to one or two ticks even
            # when it has no tie predecessor.  If a rest follows immediately,
            # consume only the rest ticks needed to reach the smallest
            # representable jianpu atom.  This preserves the pitch and onset,
            # keeps the voice timeline contiguous, and records the bounded
            # end movement in the source alignment.
            if (
                current_pitches
                and not _is_fine_grid_tuplet_member(current)
                and current.duration_tick < MIN_JIANPU_ATOM_TICKS
                and index + 1 < len(events)
                and events[index + 1].is_rest
                and not _is_fine_grid_tuplet_member(events[index + 1])
                and events[index + 1].start_tick == current.end_tick
            ):
                following = events[index + 1]
                needed = MIN_JIANPU_ATOM_TICKS - current.duration_tick
                movement = min(needed, following.duration_tick)
                if (
                    movement > 0
                    and movement <= MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS
                    and following.duration_tick - movement >= 0
                ):
                    repaired_end = current.end_tick + movement
                    updated_current = current.model_copy(
                        update={"duration_tick": repaired_end - current.start_tick}
                    )
                    current_metadata = dict(updated_current.metadata)
                    current_metadata["notation_grid_repair"] = {
                        "reason": "fine_grid_note_extended_to_jianpu_atom",
                        "original_end_tick": current.end_tick,
                        "repaired_end_tick": repaired_end,
                    }
                    events[index] = updated_current.model_copy(update={"metadata": current_metadata})
                    if current_id is not None:
                        for item in _alignment_items_for_events(alignment, {current_id}):
                            _update_alignment_end(item, repaired_end)
                    remaining = following.duration_tick - movement
                    if remaining:
                        events[index + 1] = following.model_copy(
                            update={"start_tick": repaired_end, "duration_tick": remaining}
                        )
                    else:
                        del events[index + 1]
                    repairs.append(
                        {
                            "reason": "fine_grid_note_extended_to_jianpu_atom",
                            "action": "extend_note_end_into_following_rest",
                            "voice_id": voice.voice_id,
                            "musicxml_event_id": current_id,
                            "pitch": current_pitches[0] if len(current_pitches) == 1 else None,
                            "pitches": list(current_pitches),
                            "original_end_tick": current.end_tick,
                            "repaired_end_tick": repaired_end,
                            "movement_ticks": movement,
                            "bounded_by_ticks": MAX_FINE_SCORE_REPAIR_MOVEMENT_TICKS,
                        }
                    )
                    index += 1
                    continue

            # A tiny rest directly following another rest can be coalesced
            # without changing any pitched boundary.  This keeps the same
            # explicit policy available for an equivalent MuseScore rest
            # fragment.
            if (
                current.is_rest
                and previous is not None
                and previous.is_rest
                and not _is_fine_grid_tuplet_member(current)
                and not _is_fine_grid_tuplet_member(previous)
                and previous.end_tick == current.start_tick
            ):
                events[index - 1] = previous.model_copy(
                    update={"duration_tick": current.end_tick - previous.start_tick}
                )
                del events[index]
                continue

            index += 1

        _validate_fine_grid_tuplet_groups(events, voice_id=voice.voice_id)
        repaired_voices.append(voice.model_copy(update={"events": events}))
    return repaired_voices, repairs


def _dedupe_events(values: Iterable[Mapping[str, Any]], key: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for value in values:
        identity = tuple(value.get(item) for item in key)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(dict(value))
    return result


def _meter_bar_ticks(ratio: str) -> int:
    numerator, denominator = (int(value) for value in _normalize_worker_meter(ratio).split("/", 1))
    ticks = round(numerator * SCORE_QUARTER_TICKS * 4 / denominator)
    if ticks <= 0:
        raise MusicXMLStandardizationError(f"time signature {ratio!r} produces an empty measure")
    return ticks


def _payload_measure_metadata(payload: WorkerPayload) -> list[dict[str, Any]]:
    return [
        {
            "part_index": measure.part_index,
            "number": measure.number,
            "start_tick": _quarter_to_tick(measure.start_quarter),
            "duration_tick": _quarter_to_tick(measure.duration_quarter),
            "end_tick": _quarter_to_tick(measure.end_quarter),
            "time_signature": (
                _normalize_worker_meter(str(measure.time_signature))
                if measure.time_signature
                else None
            ),
            "is_pickup": measure.is_pickup,
        }
        for measure in payload.measures
    ]


def _deduped_timeline_measure_metadata(measure_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-part/staff measure records into one auditable timeline."""

    return sorted(
        _dedupe_events(
            (
                {
                    "start_tick": item["start_tick"],
                    "duration_tick": item["duration_tick"],
                    "end_tick": item["end_tick"],
                    "time_signature": item["time_signature"],
                    "is_pickup": item["is_pickup"],
                    "number": item["number"],
                }
                for item in measure_metadata
            ),
            ("start_tick", "duration_tick", "end_tick", "is_pickup"),
        ),
        key=lambda item: (int(item["start_tick"]), int(item["end_tick"])),
    )


def _rebuild_meter_timeline(
    imported_timeline: list[dict[str, Any]],
    *,
    total_ticks: int,
    time_events: list[dict[str, Any]],
    production_authoritative: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Rebuild measure boundaries from the final conductor meter.

    MusicXML/MuseScore may assign a short performance import an inferred
    initial meter that disagrees with production BeatNet/manual metadata.  A
    Score cannot keep those two authorities at the same boundary: the
    serializer would reject the contradiction and users would see the wrong
    bar grid.  Rebuild the timeline from the final meter events, retaining
    explicit meter changes as boundaries (including a deliberately partial
    segment when a change occurs mid nominal bar).
    """

    _validate_timeline_measures(imported_timeline, total_ticks)
    if not imported_timeline:
        raise MusicXMLStandardizationError("MusicXML contained no measure timeline")

    events = sorted(
        _dedupe_events(time_events, ("offset_quarter", "ratio")),
        key=lambda item: float(item["offset_quarter"]),
    )
    final_events: list[dict[str, Any]] = []
    for item in events:
        offset = float(item["offset_quarter"])
        if not math.isfinite(offset) or offset < 0:
            raise MusicXMLStandardizationError(f"invalid meter event offset {offset!r}")
        tick = _quarter_to_tick(offset)
        if tick > total_ticks:
            continue
        ratio = _normalize_worker_meter(str(item["ratio"]))
        numerator, denominator = (int(value) for value in ratio.split("/", 1))
        final_events.append(
            {
                "start_tick": tick,
                "time_signature": ratio,
                "numerator": numerator,
                "denominator": denominator,
            }
        )
    if not final_events or final_events[0]["start_tick"] != 0:
        imported_initial = imported_timeline[0].get("time_signature") or "4/4"
        ratio = _normalize_worker_meter(str(imported_initial))
        numerator, denominator = (int(value) for value in ratio.split("/", 1))
        final_events.insert(
            0,
            {
                "start_tick": 0,
                "time_signature": ratio,
                "numerator": numerator,
                "denominator": denominator,
            },
        )
    final_events_by_tick: dict[int, dict[str, Any]] = {}
    for item in final_events:
        tick = int(item["start_tick"])
        previous = final_events_by_tick.get(tick)
        if previous is not None and previous["time_signature"] != item["time_signature"]:
            raise MusicXMLStandardizationError(
                f"conflicting final meter events at tick {tick}: "
                f"{previous['time_signature']} vs {item['time_signature']}"
            )
        final_events_by_tick[tick] = item
    final_events = sorted(final_events_by_tick.values(), key=lambda item: int(item["start_tick"]))

    def active_meter(tick: int) -> str:
        current = str(final_events[0]["time_signature"])
        for item in final_events:
            if int(item["start_tick"]) > tick:
                break
            current = str(item["time_signature"])
        return current

    def imported_active_meter(tick: int) -> str:
        current = str(imported_timeline[0].get("time_signature") or final_events[0]["time_signature"])
        for item in imported_timeline:
            if int(item["start_tick"]) > tick:
                break
            if item.get("time_signature"):
                current = str(item["time_signature"])
        return current

    first_imported = imported_timeline[0]
    pickup = bool(first_imported.get("is_pickup", False))
    target_total_ticks = total_ticks
    terminal_padding: dict[str, Any] | None = None

    def build_timeline(limit: int) -> list[dict[str, Any]]:
        cursor = 0
        rebuilt: list[dict[str, Any]] = []
        next_number = 1
        if pickup:
            initial_meter = active_meter(0)
            initial_bar_ticks = _meter_bar_ticks(initial_meter)
            pickup_end = int(first_imported["end_tick"])
            if pickup_end <= 0 or pickup_end >= initial_bar_ticks:
                raise MusicXMLStandardizationError(
                    f"cannot safely rebar pickup: imported duration {pickup_end} is not shorter than "
                    f"final {initial_meter} bar {initial_bar_ticks} ticks"
                )
            rebuilt.append(
                {
                    "start_tick": 0,
                    "duration_tick": pickup_end,
                    "end_tick": pickup_end,
                    "time_signature": initial_meter,
                    "is_pickup": True,
                    "number": first_imported.get("number", 0),
                    "rebar_reason": "preserved_imported_pickup",
                }
            )
            cursor = pickup_end
            next_number = 1

        while cursor < limit:
            meter = active_meter(cursor)
            bar_ticks = _meter_bar_ticks(meter)
            next_change = next(
                (
                    int(item["start_tick"])
                    for item in final_events
                    if int(item["start_tick"]) > cursor
                ),
                None,
            )
            end = min(limit, cursor + bar_ticks)
            partial_reason = None
            if next_change is not None and next_change < end:
                # A real meter event is an authoritative boundary even when it
                # lands in the middle of the previous nominal bar.  The
                # shorter preceding span is explicit and keeps the score
                # timeline exact.
                end = next_change
                partial_reason = "meter_change_inside_nominal_bar"
            if end <= cursor:
                raise MusicXMLStandardizationError(
                    f"cannot safely rebar: meter boundary at tick {cursor} does not advance the timeline"
                )
            record: dict[str, Any] = {
                "start_tick": cursor,
                "duration_tick": end - cursor,
                "end_tick": end,
                "time_signature": meter,
                "is_pickup": False,
                "number": next_number,
            }
            if partial_reason:
                record["rebar_reason"] = partial_reason
            rebuilt.append(record)
            cursor = end
            next_number += 1
        return rebuilt

    rebuilt = build_timeline(target_total_ticks)
    if not pickup and rebuilt:
        imported_last = imported_timeline[-1]
        imported_last_meter = imported_active_meter(int(imported_last["start_tick"]))
        imported_last_bar_ticks = _meter_bar_ticks(imported_last_meter)
        final_last = rebuilt[-1]
        final_last_meter = str(final_last["time_signature"])
        final_last_bar_ticks = _meter_bar_ticks(final_last_meter)
        if (
            production_authoritative
            and
            int(imported_last["duration_tick"]) == imported_last_bar_ticks
            and int(final_last["duration_tick"]) < final_last_bar_ticks
            and int(final_last["end_tick"]) == total_ticks
        ):
            target_total_ticks = int(final_last["start_tick"]) + final_last_bar_ticks
            if target_total_ticks > total_ticks:
                terminal_padding = {
                    "applied": True,
                    "start_tick": total_ticks,
                    "end_tick": target_total_ticks,
                    "duration_tick": target_total_ticks - total_ticks,
                    "reason": "completed_terminal_bar_after_production_meter_rebar",
                    "imported_meter": imported_last_meter,
                    "final_meter": final_last_meter,
                }
                rebuilt = build_timeline(target_total_ticks)

    _validate_timeline_measures(rebuilt, target_total_ticks)
    imported_view = [
        {
            "start_tick": int(item["start_tick"]),
            "duration_tick": int(item["duration_tick"]),
            "end_tick": int(item["end_tick"]),
            "time_signature": item.get("time_signature"),
            "is_pickup": bool(item.get("is_pickup", False)),
            "number": item.get("number"),
        }
        for item in imported_timeline
    ]
    final_view = [
        {
            "start_tick": int(item["start_tick"]),
            "duration_tick": int(item["duration_tick"]),
            "end_tick": int(item["end_tick"]),
            "time_signature": str(item["time_signature"]),
            "is_pickup": bool(item.get("is_pickup", False)),
            "number": item.get("number"),
            **(
                {"rebar_reason": item["rebar_reason"]}
                if item.get("rebar_reason") is not None
                else {}
            ),
        }
        for item in rebuilt
    ]
    comparable_imported = [
        {
            **item,
            "time_signature": item["time_signature"] or imported_active_meter(item["start_tick"]),
        }
        for item in imported_view
    ]
    comparable_final = [
        {key: value for key, value in item.items() if key != "rebar_reason"}
        for item in final_view
    ]
    changed = comparable_imported != comparable_final
    audit = {
        "applied": changed,
        "production_meter_authoritative": production_authoritative,
        "imported_timeline": imported_view,
        "final_timeline": final_view,
        "reason": (
            "production_meter_authoritative_rebuilt_timeline"
            if production_authoritative and changed
            else "conductor_meter_rebuilt_timeline"
            if changed
            else "timeline_already_matches_final_meter"
        ),
        "terminal_padding": terminal_padding,
    }
    return rebuilt, audit, target_total_ticks


def _rebar_score_voices(
    voices: list[ScoreVoice],
    timeline: list[dict[str, Any]],
) -> tuple[list[ScoreVoice], list[dict[str, Any]]]:
    """Split events crossing new bars while preserving pitch and tie semantics."""

    boundaries = sorted(
        {
            int(item["start_tick"])
            for item in timeline
            if int(item["start_tick"]) > 0
        }
    )
    repairs: list[dict[str, Any]] = []

    def measure_number(tick: int) -> int | None:
        for item in timeline:
            if int(item["start_tick"]) <= tick < int(item["end_tick"]):
                value = item.get("number")
                return int(value) if value is not None else None
        return None

    def tie_fields(event: ScoreNote, pitches: tuple[int, ...], *, first: bool, last: bool) -> tuple[str | None, list[str | None]]:
        original = _event_tie_values(event)
        if len(original) != len(pitches):
            original = [event.tie] * len(pitches)
        incoming = {
            pitch
            for pitch, tie in zip(pitches, original)
            if tie in {"stop", "continue"}
        }
        outgoing = {
            pitch
            for pitch, tie in zip(pitches, original)
            if tie in {"start", "continue"}
        }
        if not first:
            incoming = set(pitches)
        if not last:
            outgoing = set(pitches)
        values = [
            "continue"
            if pitch in incoming and pitch in outgoing
            else "stop"
            if pitch in incoming
            else "start"
            if pitch in outgoing
            else None
            for pitch in pitches
        ]
        tie = (
            values[0]
            if len(values) == 1
            else values[0]
            if values and all(value is not None and value == values[0] for value in values)
            else None
        )
        tie_types = values if event.tie_types or any(value is not None for value in values) else []
        return tie, tie_types

    rebuilt_voices: list[ScoreVoice] = []
    for voice in voices:
        split_events: list[ScoreNote] = []
        for event in voice.events:
            start = int(event.start_tick)
            end = int(event.end_tick)
            cuts = [start, *(boundary for boundary in boundaries if start < boundary < end), end]
            if len(cuts) == 2:
                split_events.append(event)
                continue
            if event.tuplet_actual is not None or event.tuplet_normal is not None or event.tuplet_type is not None:
                raise MusicXMLStandardizationError(
                    f"cannot safely rebar explicit tuplet event {event.start_tick}:{event.end_tick} "
                    "across a meter boundary"
                )
            pitches = _event_pitches(event)
            segments: list[dict[str, Any]] = []
            for index, (segment_start, segment_end) in enumerate(zip(cuts, cuts[1:])):
                first = index == 0
                last = index == len(cuts) - 2
                tie, tie_types = tie_fields(event, pitches, first=first, last=last)
                metadata = dict(event.metadata)
                repair = {
                    "reason": "meter_rebar_event_split",
                    "voice_id": voice.voice_id,
                    "musicxml_event_id": _event_musicxml_id(event),
                    "original_start_tick": start,
                    "original_end_tick": end,
                    "segment_start_tick": segment_start,
                    "segment_end_tick": segment_end,
                    "segment_index": index,
                    "segment_count": len(cuts) - 1,
                    "pitches": list(pitches),
                }
                metadata["meter_rebar_split"] = repair
                segments.append(
                    {
                        "start_tick": segment_start,
                        "duration_tick": segment_end - segment_start,
                        "tie": tie,
                        "tie_types": tie_types,
                        "dots": event.dots if len(cuts) == 2 else 0,
                        "metadata": metadata,
                        "measure_number": measure_number(segment_start),
                        "tuplet_type": (
                            event.tuplet_type
                            if (event.tuplet_type == "continue" or (event.tuplet_type == "start" and first) or (event.tuplet_type == "stop" and last))
                            else None
                        ),
                    }
                )
                repairs.append(repair)
            split_events.extend(event.model_copy(update=segment) for segment in segments)
        rebuilt_voices.append(voice.model_copy(update={"events": split_events}))
    return rebuilt_voices, repairs


def _validate_timeline_measures(measures: list[dict[str, Any]], total_ticks: int) -> None:
    if not measures:
        raise MusicXMLStandardizationError("MusicXML contained no measure timeline")
    ordered = sorted(measures, key=lambda value: (value["start_tick"], value["end_tick"]))
    previous_end = 0
    for index, measure in enumerate(ordered):
        start = int(measure["start_tick"])
        duration = int(measure["duration_tick"])
        end = int(measure["end_tick"])
        if duration <= 0 or end <= start or end != start + duration:
            raise MusicXMLStandardizationError(
                f"invalid timeline measure {index}: start={start}, duration={duration}, end={end}"
            )
        if index == 0 and start != 0:
            raise MusicXMLStandardizationError(f"measure timeline starts at {start}, expected score tick 0")
        if index and start != previous_end:
            relation = "overlap" if start < previous_end else "gap"
            raise MusicXMLStandardizationError(
                f"measure timeline has an illegal {relation} between ticks {previous_end} and {start}"
            )
        previous_end = end
    if previous_end != total_ticks:
        raise MusicXMLStandardizationError(
            f"measure timeline ends at {previous_end}, expected Score total_ticks {total_ticks}"
        )


def _source_tempo_records(performance_metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not performance_metadata:
        return []
    points = performance_metadata.get("tempo_points", [])
    result: list[dict[str, Any]] = []
    if isinstance(points, list):
        for point in points:
            if not isinstance(point, Mapping):
                continue
            try:
                tick = int(point["tick"])
                bpm = float(point["bpm"])
            except (KeyError, TypeError, ValueError):
                continue
            if tick < 0 or not math.isfinite(bpm) or bpm <= 0:
                continue
            result.append({"offset_quarter": tick / PERFORMANCE_QUARTER_TICKS, "bpm": bpm})
    if not result:
        try:
            bpm = float(performance_metadata["bpm"])
        except (KeyError, TypeError, ValueError):
            bpm = 0.0
        if math.isfinite(bpm) and bpm > 0:
            result.append({"offset_quarter": 0.0, "bpm": bpm})
    return _dedupe_events(sorted(result, key=lambda value: value["offset_quarter"]), ("offset_quarter",))


def _source_key_signature(performance_metadata: Mapping[str, Any] | None) -> str | None:
    if not performance_metadata or not isinstance(performance_metadata.get("key"), str):
        return None
    return _normalize_worker_key(str(performance_metadata["key"]))


_KEY_SIGNATURE_SHARPS = {
    # Major key signatures.  The application whitelist includes both the
    # sharp and flat spellings below, so keep the mapping explicit rather
    # than deriving it from a root with the accidental removed.
    "C": 0,
    "C#": 7,
    "Db": -5,
    "D": 2,
    "Eb": -3,
    "E": 4,
    "F": -1,
    "F#": 6,
    "Gb": -6,
    "G": 1,
    "Ab": -4,
    "A": 3,
    "Bb": -2,
    "B": 5,
    # Relative minor key signatures.  Dbm/Gbm are accepted by the app for
    # display, but MIDI emits their enharmonic C#m/F#m spellings; the same
    # values are used when no emitted spelling is supplied.
    "Cm": -3,
    "C#m": 4,
    "Dbm": 4,
    "Dm": -1,
    "Ebm": -6,
    "Em": 1,
    "Fm": -4,
    "F#m": 3,
    "Gbm": 3,
    "Gm": -2,
    "Abm": -7,
    "Am": 0,
    "Bbm": -5,
    "Bm": 2,
}


def _key_sharps(value: str, *, emitted_key: str | None = None) -> int:
    """Return the MIDI-compatible signature while preserving display spelling.

    ``value`` is the user/analysis key that should remain visible in Score.
    Performance MIDI stores an enharmonic ``emitted_key`` for the two
    application keys that mido cannot encode (Dbm and Gbm); use it only for
    the numeric signature and never replace the display key with it.
    """

    display_key = _normalize_worker_key(value)
    signature_key = _normalize_worker_key(emitted_key) if emitted_key else display_key
    try:
        return _KEY_SIGNATURE_SHARPS[signature_key]
    except KeyError as exc:
        raise MusicXMLStandardizationError(
            f"unsupported key signature for {display_key!r} (emitted key {signature_key!r})"
        ) from exc


def _source_time_signature(performance_metadata: Mapping[str, Any] | None) -> tuple[str, int, int] | None:
    if not performance_metadata or not isinstance(performance_metadata.get("time_signature"), str):
        return None
    ratio = _normalize_worker_meter(str(performance_metadata["time_signature"]))
    numerator, denominator = (int(value) for value in ratio.split("/", 1))
    return ratio, numerator, denominator


def _source_time_signature_records(performance_metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return production meter events, with the selected meter authoritative at zero.

    Production metadata normally carries one selected meter.  Accepting an
    optional event list keeps the standardizer correct for callers that carry
    a real meter map as well, while still treating the explicit selected
    meter/manual override as the highest-priority initial value.
    """

    if not performance_metadata:
        return []
    records: list[dict[str, Any]] = []
    values = performance_metadata.get("time_signature_events", [])
    if isinstance(values, list):
        for value in values:
            if not isinstance(value, Mapping):
                continue
            raw_offset = value.get("offset_quarter")
            if raw_offset is None:
                raw_tick = value.get("tick", value.get("start_tick"))
                if raw_tick is None:
                    continue
                try:
                    raw_offset = float(raw_tick) / PERFORMANCE_QUARTER_TICKS
                except (TypeError, ValueError):
                    continue
            try:
                offset = float(raw_offset)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(offset) or offset < 0:
                continue
            raw_ratio = value.get("ratio", value.get("time_signature"))
            if not isinstance(raw_ratio, str):
                continue
            try:
                ratio = _normalize_worker_meter(raw_ratio)
            except MusicXMLStandardizationError:
                continue
            numerator, denominator = (int(item) for item in ratio.split("/", 1))
            records.append(
                {
                    "offset_quarter": offset,
                    "ratio": ratio,
                    "numerator": numerator,
                    "denominator": denominator,
                }
            )
    selected = _source_time_signature(performance_metadata)
    if selected is not None:
        ratio, numerator, denominator = selected
        records = [item for item in records if abs(float(item["offset_quarter"])) > 1e-9]
        records.append(
            {
                "offset_quarter": 0.0,
                "ratio": ratio,
                "numerator": numerator,
                "denominator": denominator,
            }
        )
    return _dedupe_events(sorted(records, key=lambda item: float(item["offset_quarter"])), ("offset_quarter",))


def _reconcile_conductor_metadata(
    payload: WorkerPayload,
    performance_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge production conductor metadata without replacing later XML changes."""

    xml_tempo = _dedupe_events(
        (
            {"offset_quarter": item.offset_quarter, "bpm": item.bpm}
            for item in payload.tempo_events
        ),
        ("offset_quarter",),
    )
    xml_tempo = sorted(xml_tempo, key=lambda value: float(value["offset_quarter"]))
    source_tempo = _source_tempo_records(performance_metadata)
    tempo_values = list(xml_tempo)
    reconciliation: list[dict[str, Any]] = []
    for source in source_tempo:
        offset = float(source["offset_quarter"])
        matching = next(
            (item for item in tempo_values if abs(float(item["offset_quarter"]) - offset) <= 1e-9),
            None,
        )
        if matching is None:
            tempo_values.append(dict(source))
            reconciliation.append(
                {
                    "field": "tempo",
                    "offset_quarter": offset,
                    "musicxml_value": None,
                    "production_value": source["bpm"],
                    "final_value": source["bpm"],
                    "reason": "production_metadata_backfilled_missing_tempo",
                }
            )
        elif offset == 0.0 and abs(float(matching["bpm"]) - float(source["bpm"])) > 0.01:
            tempo_values = [item for item in tempo_values if abs(float(item["offset_quarter"])) > 1e-9]
            tempo_values.append(dict(source))
            reconciliation.append(
                {
                    "field": "tempo",
                    "offset_quarter": offset,
                    "musicxml_value": matching["bpm"],
                    "production_value": source["bpm"],
                    "final_value": source["bpm"],
                    "reason": "production_metadata_replaced_changed_initial_tempo",
                }
            )
    if not tempo_values:
        tempo_values = [{"offset_quarter": 0.0, "bpm": 120.0}]
    tempo_values = sorted(tempo_values, key=lambda value: float(value["offset_quarter"]))

    xml_time = _dedupe_events(
        (
            {
                "offset_quarter": item.offset_quarter,
                "ratio": item.ratio,
                "numerator": item.numerator,
                "denominator": item.denominator,
            }
            for item in payload.time_signature_events
        ),
        ("offset_quarter", "ratio"),
    )
    source_time_records = _source_time_signature_records(performance_metadata)
    time_events = list(xml_time)
    for source in source_time_records:
        offset = float(source["offset_quarter"])
        matching = next(
            (item for item in time_events if abs(float(item["offset_quarter"]) - offset) <= 1e-9),
            None,
        )
        if matching is not None and matching["ratio"] == source["ratio"]:
            continue
        if matching is not None:
            time_events = [
                item
                for item in time_events
                if abs(float(item["offset_quarter"]) - offset) > 1e-9
            ]
        time_events.append(dict(source))
        reconciliation.append(
            {
                "field": "time_signature",
                "offset_quarter": offset,
                "musicxml_value": matching["ratio"] if matching else None,
                "production_value": source["ratio"],
                "final_value": source["ratio"],
                "reason": (
                    "production_metadata_replaced_changed_initial_time_signature"
                    if abs(offset) <= 1e-9
                    else "production_metadata_replaced_changed_time_signature"
                ),
            }
        )
    time_events = sorted(time_events, key=lambda value: float(value["offset_quarter"]))
    if not time_events:
        time_events = [{"offset_quarter": 0.0, "ratio": "4/4", "numerator": 4, "denominator": 4}]

    xml_key = _dedupe_events(
        (
            {"offset_quarter": item.offset_quarter, "key": item.key, "sharps": item.sharps}
            for item in payload.key_signature_events
        ),
        ("offset_quarter", "key"),
    )
    source_key = _source_key_signature(performance_metadata)
    key_events = list(xml_key)
    if source_key is not None:
        xml_initial = next((item for item in key_events if abs(float(item["offset_quarter"])) <= 1e-9), None)
        if xml_initial is None or _normalize_worker_key(str(xml_initial["key"])) != source_key:
            if xml_initial is not None:
                key_events = [item for item in key_events if abs(float(item["offset_quarter"])) > 1e-9]
            emitted_key = None
            if performance_metadata and isinstance(performance_metadata.get("emitted_key"), str):
                emitted_key = _normalize_worker_key(str(performance_metadata["emitted_key"]))
            key_events.append(
                {
                    "offset_quarter": 0.0,
                    "key": source_key,
                    "sharps": _key_sharps(source_key, emitted_key=emitted_key),
                }
            )
            reconciliation.append(
                {
                    "field": "key",
                    "offset_quarter": 0.0,
                    "musicxml_value": xml_initial["key"] if xml_initial else None,
                    "production_value": source_key,
                    "final_value": source_key,
                    "emitted_value": emitted_key,
                    "reason": "production_metadata_backfilled_changed_initial_key",
                }
            )
    key_events = sorted(key_events, key=lambda value: float(value["offset_quarter"]))
    if not key_events:
        key_events = [{"offset_quarter": 0.0, "key": "C", "sharps": 0}]

    return {
        "tempo_values": tempo_values,
        "time_events": time_events,
        "key_events": key_events,
        "key": _normalize_worker_key(str(key_events[0]["key"])),
        "time_signature": _normalize_worker_meter(str(time_events[0]["ratio"])),
        "reconciliation": reconciliation,
    }


def standardize_musicxml_payload(
    payload: WorkerPayload,
    *,
    performance_metadata: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> tuple[Score, dict[str, Any]]:
    """Convert validated worker data to a 48 TPQ Score and alignment report."""

    conductor = _reconcile_conductor_metadata(payload, performance_metadata)
    key = conductor["key"]
    time_signature = conductor["time_signature"]

    total_quarter = max(
        payload.highest_time_quarter,
        *(part.highest_time_quarter for part in payload.parts),
        *(measure.end_quarter for measure in payload.measures),
    )
    total_ticks = max(1, _quarter_to_tick(total_quarter))
    measure_metadata = _payload_measure_metadata(payload)
    imported_timeline = _deduped_timeline_measure_metadata(measure_metadata)
    timeline_measures, meter_rebar, total_ticks = _rebuild_meter_timeline(
        imported_timeline,
        total_ticks=total_ticks,
        time_events=conductor["time_events"],
        production_authoritative=bool(_source_time_signature_records(performance_metadata)),
    )
    raw_events, diagnostics = _worker_raw_events(payload)
    source_notes = _source_notes(performance_metadata)
    if source_notes:
        diagnostics.extend(_split_source_retrigger_events(raw_events, source_notes))
    source_alignment_hints: dict[int, dict[str, Any]] = {}
    source_alignment_report: dict[str, Any] = {
        "applied": False,
        "reason": "no_performance_source_metadata",
    }
    if source_notes:
        source_alignment_hints, source_alignment_report = _estimate_source_alignment(raw_events, source_notes)
        for source in source_notes:
            hint = source_alignment_hints.get(int(source["source_index"]))
            if hint is not None:
                source["_alignment_hint"] = hint
    tempo_values = conductor["tempo_values"]
    tempo_events = [
        TempoEvent(start_tick=max(0, _quarter_to_tick(float(item["offset_quarter"]))), bpm=float(item["bpm"]))
        for item in tempo_values
    ]
    if tempo_events[0].start_tick > 0:
        tempo_events.insert(0, TempoEvent(start_tick=0, bpm=tempo_events[0].bpm))
    alignment = (
        _align_source_notes(raw_events, source_notes, alignment_hints=source_alignment_hints)
        if source_notes
        else []
    )
    alignment, primary_alignment_count = _append_instrumental_cleanup_alignment(
        alignment,
        performance_metadata,
    )
    logical_units = _logical_pitch_units(raw_events)
    matched_musicxml_unit_ids = {
        int(item["musicxml_unit_id"])
        for item in alignment
        if item.get("musicxml_unit_id") is not None
    }
    musicxml_extras = [
        {
            "unit_id": unit.unit_id,
            "pitch": unit.pitch,
            "start_tick": unit.start_tick,
            "end_tick": unit.end_tick,
            "musicxml_event_ids": [item[0].event_id for item in unit.chain],
            "reason": "musicxml_logical_unit_not_referenced_by_source_metadata",
        }
        for unit in logical_units
        if unit.unit_id not in matched_musicxml_unit_ids
    ]

    grouped: dict[tuple[str, int, str], list[_RawEvent]] = {}
    for event in raw_events:
        grouped.setdefault((event.part_group, event.staff, event.voice), []).append(event)
    voices: list[ScoreVoice] = []
    lane_reasons: list[dict[str, Any]] = []
    for (part_group, staff, source_voice), group_events in sorted(grouped.items()):
        lanes = _allocate_lanes(group_events)
        for lane_index, lane in enumerate(lanes):
            voice_id = f"{part_group}:staff-{staff}:voice-{source_voice}:lane-{lane_index + 1}"
            voice_events = _score_voice_events(
                lane,
                total_ticks=total_ticks,
                staff=staff,
                source_voice=source_voice,
            )
            if lane_index:
                lane_reasons.append(
                    {
                        "voice_id": voice_id,
                        "reason": "overlapping_events_allocated_to_additional_score_voice",
                        "source_part": part_group,
                        "staff": staff,
                        "source_voice": source_voice,
                    }
                )
            voices.append(
                ScoreVoice(
                    voice_id=voice_id,
                    events=voice_events,
                    label=f"{part_group} staff {staff} voice {source_voice} lane {lane_index + 1}",
                    stem_id=part_group,
                    staff=staff,
                    source_voice=source_voice,
                )
            )
    if not voices:
        raise MusicXMLStandardizationError("MusicXML contained no printable notes, chords, or rests")
    meter_rebar_splits: list[dict[str, Any]] = []
    if meter_rebar["applied"]:
        voices, meter_rebar_splits = _rebar_score_voices(voices, timeline_measures)
    voices, dot_repairs = _repair_explicit_dots(voices)
    voices, fine_grid_repairs = _repair_fine_score_events(
        voices,
        alignment,
        total_ticks=total_ticks,
    )
    notation_grid_repairs = dot_repairs + fine_grid_repairs
    tuplet_marker_repairs = [
        item
        for item in diagnostics
        if item.get("reason")
        in {
            "cross_voice_tuplet_marker_reassigned",
            "cross_voice_tuplet_marker_reassembled",
            "cross_voice_tuplet_marker_conflict",
            "orphan_tuplet_marker_cleared",
        }
    ]

    time_signature_events = [
        {
            "start_tick": _quarter_to_tick(item["offset_quarter"]),
            "time_signature": _normalize_worker_meter(item["ratio"]),
            "numerator": item["numerator"],
            "denominator": item["denominator"],
        }
        for item in conductor["time_events"]
    ]
    key_signature_events = [
        {"start_tick": _quarter_to_tick(item["offset_quarter"]), "key": _normalize_worker_key(item["key"]), "sharps": item["sharps"]}
        for item in conductor["key_events"]
    ]
    matched_count = sum(item.get("accounting_category") == "matched" for item in alignment)
    merged_count = sum(item.get("accounting_category") == "merged" for item in alignment)
    dropped_count = sum(item.get("accounting_category") == "dropped" for item in alignment)
    unresolved_count = sum(item.get("accounting_category") == "unresolved" for item in alignment)
    movement_starts = [
        int(item["source_to_score_movement_start_ticks"])
        for item in alignment
        if item.get("source_to_score_movement_start_ticks") is not None
    ]
    movement_ends = [
        int(item["source_to_score_movement_end_ticks"])
        for item in alignment
        if item.get("source_to_score_movement_end_ticks") is not None
    ]
    movement_summary = {
        "start_ticks": {
            "count": len(movement_starts),
            "min": min(movement_starts) if movement_starts else None,
            "max": max(movement_starts) if movement_starts else None,
            "mean": (sum(movement_starts) / len(movement_starts)) if movement_starts else None,
            "absolute_max": max((abs(value) for value in movement_starts), default=0),
        },
        "end_ticks": {
            "count": len(movement_ends),
            "min": min(movement_ends) if movement_ends else None,
            "max": max(movement_ends) if movement_ends else None,
            "mean": (sum(movement_ends) / len(movement_ends)) if movement_ends else None,
            "absolute_max": max((abs(value) for value in movement_ends), default=0),
        },
    }
    instrumental_cleanup = (
        performance_metadata.get("instrumental_cleanup")
        if performance_metadata and isinstance(performance_metadata.get("instrumental_cleanup"), Mapping)
        else None
    )
    source_note_count = len(source_notes)
    if instrumental_cleanup is not None:
        try:
            source_note_count = int(instrumental_cleanup.get("source_note_count", source_note_count))
            expected_merged_count = int(instrumental_cleanup.get("merged_count", 0))
        except (TypeError, ValueError) as exc:
            raise MusicXMLStandardizationError("instrumental cleanup report has invalid source accounting") from exc
        if source_note_count != len(source_notes) + expected_merged_count:
            raise MusicXMLStandardizationError(
                "instrumental cleanup source accounting does not equal primary plus merged notes"
            )
        if merged_count < expected_merged_count:
            raise MusicXMLStandardizationError(
                "instrumental cleanup merged notes were not fully represented in alignment"
            )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_musicxml": payload.source_path,
        "music21_version": payload.music21_version,
        "source_note_count": source_note_count,
        "accounted_source_count": matched_count + merged_count + dropped_count,
        "matched_count": matched_count,
        "merged_count": merged_count,
        "dropped_count": dropped_count,
        "unresolved_count": unresolved_count,
        "alignment_summary": {
            "matched": matched_count,
            "merged": merged_count,
            "dropped": dropped_count,
            "unresolved": unresolved_count,
            "accounted_source_count": matched_count + merged_count + dropped_count,
            "movement": movement_summary,
            "musicxml_extra_count": len(musicxml_extras),
            "source_coordinate_reconciliation": source_alignment_report,
        },
        "musicxml_event_count": len(raw_events),
        "musicxml_logical_unit_count": len(logical_units),
        "musicxml_matched_logical_unit_count": len(logical_units) - len(musicxml_extras),
        "musicxml_extra_count": len(musicxml_extras),
        "musicxml_extras": musicxml_extras,
        "fine_grid_quantization": [
            item for item in diagnostics if item.get("reason") == "finer_binary_musicxml_value_quantized_to_48_tpq"
        ],
        "notation_grid_repairs": notation_grid_repairs,
        "tuplet_marker_repairs": tuplet_marker_repairs,
        "meter_rebar": {
            **meter_rebar,
            "event_splits": meter_rebar_splits,
            "event_split_count": len(meter_rebar_splits),
        },
        "score_voice_count": len(voices),
        "source_to_score": alignment,
        "instrumental_cleanup": instrumental_cleanup,
        "primary_alignment_count": primary_alignment_count,
        "source_coordinate_reconciliation": source_alignment_report,
        "repairs": diagnostics + notation_grid_repairs + meter_rebar_splits + lane_reasons,
        "tie_voice_repairs": [
            item
            for item in diagnostics
            if item.get("reason") in {"tie_chain_voice_reassigned", "tie_chain_event_split"}
        ],
        "source_note_policy": "performance metadata is used only for auditable source-to-MusicXML alignment; XML pitch/timing remains authoritative",
        "alignment_tick_semantics": "source_to_score_movement_* = final Score tick - source performance tick; musicxml_to_score_movement_* = final Score tick - MusicXML tick",
        "score_grid_precision_policy": "48 TPQ preserves exact 1/32-note, dotted, and supported 3:2 triplet values; compact finer binary MuseScore fragments are encoded as explicit 3:2 or bounded 3:1 fine-grid tuplets when every nominal atom is integral, including single 1/2/4/5-tick events without consuming adjacent events; isolated one/two-tick renderer fragments may still move within a 2-tick jianpu atom bound with every movement recorded; other fractional values are rejected explicitly",
        "conductor_reconciliation": conductor["reconciliation"],
    }
    warnings: list[str] = []
    if any(item.get("reason") == "grace_event_not_representable_at_48_tpq" for item in diagnostics):
        warnings.append("MusicXML contained grace events that cannot be represented at positive 48 TPQ duration")
    if any(item.get("reason") == "finer_binary_musicxml_value_quantized_to_48_tpq" for item in diagnostics):
        warnings.append("Finer binary MusicXML fragments were quantized to the nearest 48 TPQ tick within a 0.5 tick bound; inspect alignment_report.json")
    if any(str(item.get("reason", "")).startswith("explicit_dots_") for item in notation_grid_repairs):
        warnings.append("MusicXML explicit dot hints did not match the final 48 TPQ duration or tuplet context; hints were cleared or recomputed; inspect alignment_report.json")
    if notation_grid_repairs:
        warnings.append("Finer MusicXML fragments required bounded jianpu atom repairs; inspect alignment_report.json")
    if any(item.get("reason") == "fine_grid_fragment_encoded_as_explicit_tuplet" for item in notation_grid_repairs):
        warnings.append("Compact finer MusicXML fragments were preserved as explicit exact fine-grid tuplets; inspect alignment_report.json")
    if any(item.get("reason") == "fine_grid_singleton_encoded_as_explicit_tuplet" for item in notation_grid_repairs):
        warnings.append("Independent 1/2/4/5-tick MusicXML events were preserved as explicit exact 3:1 singleton tuplets; inspect alignment_report.json")
    if tuplet_marker_repairs:
        warnings.append("MusicXML explicit tuplet markers were repaired only for an auditable orphan or cross-voice import artifact; inspect alignment_report.json")
    if meter_rebar["applied"]:
        warnings.append("Production meter authority rebuilt MusicXML measure boundaries; inspect alignment_report.json for imported/final spans and event splits")
    if any(item.get("reason") in {"tie_chain_voice_reassigned", "tie_chain_event_split"} for item in diagnostics):
        warnings.append("MusicXML tie fragments were normalized into serializable ScoreVoice lanes")
    if any(item.get("reason") == "musicxml_event_split_for_source_retriggers" for item in diagnostics):
        warnings.append("An imported MusicXML pitch slot was split only at proven source retrigger boundaries; inspect alignment_report.json")
    if lane_reasons:
        warnings.append("Overlapping MusicXML events were preserved in additional ScoreVoice lanes")
    if any(item.get("reason") not in {"matched_musicxml_event", "matched_musicxml_tie_chain"} for item in alignment):
        warnings.append("Source performance alignment contains quantization movement or explicit accounting; inspect alignment_report.json")
    metadata: dict[str, Any] = {
        "notation_engine": "musescore-midi-import",
        "score_normalizer": "music21",
        "musescore_version": MUSESCORE_VERSION,
        "music21_version": payload.music21_version,
        "musicxml_worker_schema_version": payload.schema_version,
        "score_ticks_per_quarter": SCORE_QUARTER_TICKS,
        "score_grid_precision_policy": "exact 48 TPQ for supported notation; compact finer binary MusicXML fragments use explicit exact 3:2/3:1 fine-grid tuplets when possible, including independent 1/2/4/5-tick singleton events without adjacent-event merging, while any remaining isolated one/two-tick jianpu atom repair is bounded to 2 ticks and recorded in alignment_report.json; other unsupported fractions are rejected",
        "source_musicxml": payload.source_path,
        "parts": [
            {
                "part_id": part.part_id,
                "part_group": _part_group(part.part_id),
                "name": part.name,
                "instrument": part.instrument,
                "highest_time_quarter": part.highest_time_quarter,
            }
            for part in payload.parts
        ],
        "measures": measure_metadata,
        "timeline_measures": timeline_measures,
        "measure_total_ticks": max((item["end_tick"] for item in timeline_measures), default=total_ticks),
        "measure_duration_total_ticks": sum(item["duration_tick"] for item in timeline_measures),
        "time_signature_events": time_signature_events,
        "key_signature_events": key_signature_events,
        "conductor_reconciliation": conductor["reconciliation"],
        "meter_rebar": {
            **meter_rebar,
            "event_splits": meter_rebar_splits,
            "event_split_count": len(meter_rebar_splits),
        },
        "pickup": {
            "is_pickup": payload.pickup.is_pickup,
            "duration_tick": (
                _quarter_to_tick(payload.pickup.duration_quarter)
                if payload.pickup.is_pickup
                else 0
            ),
            "measure_number": payload.pickup.measure_number,
        },
        "alignment_report": report,
        "source_coordinate_reconciliation": source_alignment_report,
        "instrumental_cleanup": instrumental_cleanup,
        "chord_policy": "ScoreNote.chord_pitches retains every MusicXML chord pitch; midi is the lowest pitch for backward compatibility",
        "staff_policy": "Piano staff parts are grouped by the MusicXML parent id and retain staff on ScoreVoice/ScoreNote",
        "voice_policy": "Overlapping events receive additional lanes and are never deleted, including lanes beyond four",
        "source_performance_metadata": bool(source_notes),
        "tie_voice_repairs": [
            item
            for item in diagnostics
            if item.get("reason") in {"tie_chain_voice_reassigned", "tie_chain_event_split"}
        ],
    }
    score = Score(
        title=sanitize_title(title or payload.title),
        bpm=tempo_events[0].bpm,
        key=key,
        time_signature=time_signature,
        quarter_ticks=SCORE_QUARTER_TICKS,
        total_ticks=total_ticks,
        voices=voices,
        tempo_events=tempo_events,
        source="musicxml-music21",
        warnings=warnings,
        metadata=metadata,
    )
    return score, report


def standardize_musicxml(
    musicxml_path: str | Path,
    *,
    performance_metadata: Mapping[str, Any] | str | Path | None = None,
    title: str | None = None,
    notation_python: str | Path | None = None,
    timeout_sec: int = 180,
) -> tuple[Score, dict[str, Any]]:
    """Run the isolated worker and normalize its versioned payload."""

    metadata_value: Mapping[str, Any] | None
    if performance_metadata is None:
        metadata_value = None
    elif isinstance(performance_metadata, (str, Path)):
        loaded = _load_json(Path(performance_metadata).expanduser().resolve())
        metadata_value = loaded
    else:
        metadata_value = performance_metadata
    if metadata_value and (
        metadata_value.get("is_drum") is True
        or metadata_value.get("drum_jianpu_policy") == "midi_only"
    ):
        raise MusicXMLStandardizationError("drum performance is MIDI-only and does not produce a digital Score")
    payload = run_musicxml_worker(musicxml_path, notation_python=notation_python, timeout_sec=timeout_sec)
    return standardize_musicxml_payload(payload, performance_metadata=metadata_value, title=title)


def write_standardized_score(
    musicxml_path: str | Path,
    score_json_path: str | Path,
    *,
    alignment_report_path: str | Path | None = None,
    performance_metadata: Mapping[str, Any] | str | Path | None = None,
    title: str | None = None,
    notation_python: str | Path | None = None,
    timeout_sec: int = 180,
) -> StandardizedScoreArtifact:
    """Write Score JSON and alignment_report.json beside a MusicXML artifact."""

    musicxml = Path(musicxml_path).expanduser().resolve()
    score_destination = Path(score_json_path).expanduser().resolve()
    alignment_destination = (
        Path(alignment_report_path).expanduser().resolve()
        if alignment_report_path is not None
        else score_destination.with_name("alignment_report.json")
    )
    if musicxml == score_destination or musicxml == alignment_destination:
        raise MusicXMLStandardizationError("standardized outputs must not overwrite MusicXML")
    score, report = standardize_musicxml(
        musicxml,
        performance_metadata=performance_metadata,
        title=title,
        notation_python=notation_python,
        timeout_sec=timeout_sec,
    )
    score_destination.parent.mkdir(parents=True, exist_ok=True)
    alignment_destination.parent.mkdir(parents=True, exist_ok=True)
    score_destination.write_text(json.dumps(score.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    alignment_destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return StandardizedScoreArtifact(musicxml, score_destination, alignment_destination, score, report)
