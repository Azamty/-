"""Strict MusicXML -> 48 TPQ Score normalization.

MuseScore performs the performance-MIDI notation decisions.  This module only
invokes the isolated music21 worker, validates its versioned JSON, preserves
notation metadata, and converts the result into the renderer-independent
Score contract.  It never imports music21 and never falls back to the legacy
uniform quantizer.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
        result.append(
            {
                "source_index": int(value.get("index", index)),
                "midi": midi,
                "start_tick_480": start_480,
                "end_tick_480": end_480,
                "start_tick": round(start_480 * SCORE_QUARTER_TICKS / PERFORMANCE_QUARTER_TICKS),
                "end_tick": round(end_480 * SCORE_QUARTER_TICKS / PERFORMANCE_QUARTER_TICKS),
                "voice_id": value.get("voice_id"),
            }
        )
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
    diagnostics.extend(_normalize_tied_event_voices(events))
    diagnostics.extend(_normalize_orphan_tuplet_markers(events))
    # A cross-voice tuplet repair can move a fragment that starts a tie.  Run
    # the same conservative tie-chain pass once more so its unique successor
    # follows the repaired fragment; otherwise the serializer would report a
    # dangling tie even though the original import was unambiguous.
    diagnostics.extend(_normalize_tied_event_voices(events))
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
    ordered = sorted(context, key=lambda event: (event.start_tick, event.end_tick, event.event_id))
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
    return ordered


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


def _alignment_source_item(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_index": int(source["source_index"]),
        "source_midi": int(source["midi"]),
        "source_start_tick_480": int(source["start_tick_480"]),
        "source_end_tick_480": int(source["end_tick_480"]),
        "source_start_tick": int(source["start_tick"]),
        "source_end_tick": int(source["end_tick"]),
    }


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
        distance = abs(unit.start_tick - int(source["start_tick"]))
        unit_window = min(24, max(6, _nearest_gap(unit_starts, unit.start_tick) // 2 + 6))
        source_window = min(24, max(6, _nearest_gap(source_starts, int(source["start_tick"])) // 2 + 6))
        ordinary_window = min(unit_window, source_window)
        support, cohort_count = group_support(source, unit)
        source_duration = max(1, int(source["end_tick"]) - int(source["start_tick"]))
        unit_duration = max(1, unit.end_tick - unit.start_tick)
        overlap = max(
            0,
            min(int(source["end_tick"]), unit.end_tick) - max(int(source["start_tick"]), unit.start_tick),
        )
        overlap_ratio = overlap / min(source_duration, unit_duration)
        duration_supported = distance <= 48 and overlap_ratio >= 0.9 and abs(unit.end_tick - int(source["end_tick"])) <= 6
        if distance <= ordinary_window:
            evidence = "adaptive_quantization_window"
        elif distance <= 48 and cohort_count >= 2 and support >= min(cohort_count, 3):
            evidence = "chord_onset_group_quantization_window"
        elif duration_supported:
            evidence = "duration_overlap_quantization_window"
        else:
            return None
        cost = distance / 6.0 + abs(unit_duration - source_duration) / 24.0
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
            reason = "matched_musicxml_tie_chain" if len(unit.chain) > 1 else "matched_musicxml_event"
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
    for event in sorted(events, key=lambda item: (item.start_tick, item.end_tick, item.event_id)):
        lane_index = next((index for index, end in enumerate(lane_ends) if end <= event.start_tick), None)
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
            if current.duration_tick >= MIN_JIANPU_ATOM_TICKS and not requires_fragment_repair:
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
            if previous is not None and previous.end_tick <= current.end_tick:
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
                and current.duration_tick < MIN_JIANPU_ATOM_TICKS
                and index + 1 < len(events)
                and events[index + 1].is_rest
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
            if current.is_rest and previous is not None and previous.is_rest and previous.end_tick == current.start_tick:
                events[index - 1] = previous.model_copy(
                    update={"duration_tick": current.end_tick - previous.start_tick}
                )
                del events[index]
                continue

            index += 1

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
    tempo_values = conductor["tempo_values"]
    tempo_events = [
        TempoEvent(start_tick=max(0, _quarter_to_tick(float(item["offset_quarter"]))), bpm=float(item["bpm"]))
        for item in tempo_values
    ]
    if tempo_events[0].start_tick > 0:
        tempo_events.insert(0, TempoEvent(start_tick=0, bpm=tempo_events[0].bpm))

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
    alignment = _align_source_notes(raw_events, source_notes) if source_notes else []
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
        if item.get("reason") in {"cross_voice_tuplet_marker_reassigned", "orphan_tuplet_marker_cleared"}
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
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_musicxml": payload.source_path,
        "music21_version": payload.music21_version,
        "source_note_count": len(source_notes),
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
        "repairs": diagnostics + notation_grid_repairs + meter_rebar_splits + lane_reasons,
        "tie_voice_repairs": [
            item
            for item in diagnostics
            if item.get("reason") in {"tie_chain_voice_reassigned", "tie_chain_event_split"}
        ],
        "source_note_policy": "performance metadata is used only for auditable source-to-MusicXML alignment; XML pitch/timing remains authoritative",
        "alignment_tick_semantics": "source_to_score_movement_* = final Score tick - source performance tick; musicxml_to_score_movement_* = final Score tick - MusicXML tick",
        "score_grid_precision_policy": "48 TPQ preserves exact 1/32-note, dotted, and supported triplet values; finer binary MuseScore fragments are rounded within 0.5 score tick, and one/two-tick renderer fragments may move within a 2-tick jianpu atom bound with every movement recorded; other fractional values are rejected explicitly",
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
    if tuplet_marker_repairs:
        warnings.append("MusicXML explicit tuplet markers were repaired only for an auditable orphan or cross-voice import artifact; inspect alignment_report.json")
    if meter_rebar["applied"]:
        warnings.append("Production meter authority rebuilt MusicXML measure boundaries; inspect alignment_report.json for imported/final spans and event splits")
    if any(item.get("reason") in {"tie_chain_voice_reassigned", "tie_chain_event_split"} for item in diagnostics):
        warnings.append("MusicXML tie fragments were normalized into serializable ScoreVoice lanes")
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
        "score_grid_precision_policy": "exact 48 TPQ for supported notation; finer binary MusicXML fragments are quantized within 0.5 tick and any one/two-tick jianpu atom repair is bounded to 2 ticks and recorded in alignment_report.json; other unsupported fractions are rejected",
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
