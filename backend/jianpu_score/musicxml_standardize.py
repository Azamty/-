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


def _quarter_to_tick(value: float, *, context: str = "MusicXML quarter value") -> int:
    if not math.isfinite(value):
        raise MusicXMLStandardizationError(f"non-finite MusicXML quarter position: {value!r}")
    scaled = value * SCORE_QUARTER_TICKS
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-7:
        raise MusicXMLStandardizationError(
            f"{context} {value!r} cannot be represented exactly at {SCORE_QUARTER_TICKS} TPQ; "
            "the notation contains a tuplet or duration outside the supported exact grid"
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
            start_tick = _quarter_to_tick(item.offset_quarter)
            end_tick = max(start_tick + 1, _quarter_to_tick(item.offset_quarter + item.duration_quarter))
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
                    dots=item.dots,
                    measure_number=item.measure_number,
                    metadata={"musicxml_event_id": item.event_id},
                )
            )
    return events, diagnostics


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
            left = bisect_left(starts, unit.start_tick - 48)
            right = bisect_right(starts, unit.start_tick + 48)
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
    source_time = _source_time_signature(performance_metadata)
    time_events = list(xml_time)
    if source_time is not None:
        ratio, numerator, denominator = source_time
        xml_initial = next((item for item in time_events if abs(float(item["offset_quarter"])) <= 1e-9), None)
        if xml_initial is None or xml_initial["ratio"] != ratio:
            if xml_initial is not None:
                time_events = [item for item in time_events if abs(float(item["offset_quarter"])) > 1e-9]
            time_events.append(
                {
                    "offset_quarter": 0.0,
                    "ratio": ratio,
                    "numerator": numerator,
                    "denominator": denominator,
                }
            )
            reconciliation.append(
                {
                    "field": "time_signature",
                    "offset_quarter": 0.0,
                    "musicxml_value": xml_initial["ratio"] if xml_initial else None,
                    "production_value": ratio,
                    "final_value": ratio,
                    "reason": "production_metadata_backfilled_changed_initial_time_signature",
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
    raw_events, diagnostics = _worker_raw_events(payload)
    source_notes = _source_notes(performance_metadata)
    alignment = _align_source_notes(raw_events, source_notes) if source_notes else []
    logical_units = _logical_pitch_units(raw_events)
    matched_musicxml_event_ids = {
        event_id
        for item in alignment
        for event_id in item.get("musicxml_event_ids", [])
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
        if not any(event.event_id in matched_musicxml_event_ids for event, _ in unit.chain)
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

    measure_metadata: list[dict[str, Any]] = []
    for measure in payload.measures:
        measure_metadata.append(
            {
                "part_index": measure.part_index,
                "number": measure.number,
                "start_tick": _quarter_to_tick(measure.start_quarter),
                "duration_tick": _quarter_to_tick(measure.duration_quarter),
                "end_tick": _quarter_to_tick(measure.end_quarter),
                "time_signature": measure.time_signature,
                "is_pickup": measure.is_pickup,
            }
        )
    # A piano part is commonly split into one music21 part per staff.  Keep
    # each source record above for auditability, but derive one timeline for
    # duration checks so the same bar is not counted once per staff/part.
    timeline_measures = _dedupe_events(
        (
            {
                "start_tick": item["start_tick"],
                "duration_tick": item["duration_tick"],
                "end_tick": item["end_tick"],
                "time_signature": item["time_signature"],
                "is_pickup": item["is_pickup"],
            }
            for item in measure_metadata
        ),
        ("start_tick", "duration_tick", "end_tick", "is_pickup"),
    )
    _validate_timeline_measures(timeline_measures, total_ticks)
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
        "score_voice_count": len(voices),
        "source_to_score": alignment,
        "repairs": diagnostics + lane_reasons,
        "source_note_policy": "performance metadata is used only for auditable source-to-MusicXML alignment; XML pitch/timing remains authoritative",
        "alignment_tick_semantics": "source_to_score_movement_* = final Score tick - source performance tick; musicxml_to_score_movement_* = final Score tick - MusicXML tick",
        "score_grid_precision_policy": "48 TPQ accepts exact 1/32, dotted, and supported triplet values; other fractional values are rejected explicitly",
        "conductor_reconciliation": conductor["reconciliation"],
    }
    warnings: list[str] = []
    if diagnostics:
        warnings.append("MusicXML contained grace events that cannot be represented at positive 48 TPQ duration")
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
        "score_grid_precision_policy": "exact 48 TPQ; unsupported fractional MusicXML durations are rejected",
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
        "pickup": {
            "is_pickup": payload.pickup.is_pickup,
            "duration_tick": _quarter_to_tick(payload.pickup.duration_quarter),
            "measure_number": payload.pickup.measure_number,
        },
        "alignment_report": report,
        "chord_policy": "ScoreNote.chord_pitches retains every MusicXML chord pitch; midi is the lowest pitch for backward compatibility",
        "staff_policy": "Piano staff parts are grouped by the MusicXML parent id and retain staff on ScoreVoice/ScoreNote",
        "voice_policy": "Overlapping events receive additional lanes and are never deleted, including lanes beyond four",
        "source_performance_metadata": bool(source_notes),
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
