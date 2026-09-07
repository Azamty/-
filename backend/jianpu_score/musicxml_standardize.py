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
import subprocess
import tempfile
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


def _normalize_worker_key(value: str) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if lowered.endswith(" minor"):
        text = text[:-6].strip() + "m"
    elif lowered.endswith(" major"):
        text = text[:-6].strip()
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


def _align_source_notes(
    events: list[_RawEvent],
    source_notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align source notes without undoing MuseScore's notation decisions.

    Normal matches retain the MusicXML start/end selected by MuseScore.  The
    only span restoration is the observed same-pitch-overlap case where an
    XML note ends at the next overlapping source onset, or where such a source
    note is absent altogether.  A non-overlapping source note that cannot be
    matched is a hard error; silently producing a score with a missing pitch
    would violate the performance-to-score roundtrip contract.  Every report
    movement is ``final Score tick - source performance tick``; the separate
    MusicXML movement fields make the importer change auditable as well.
    """

    report: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    overlap_indices: set[int] = set()
    unmatched: list[dict[str, int]] = []
    for left_index, left in enumerate(source_notes):
        for right_index, right in enumerate(source_notes):
            if left_index == right_index or left["midi"] != right["midi"]:
                continue
            if left["start_tick"] < right["start_tick"] < left["end_tick"]:
                overlap_indices.add(int(left["source_index"]))
                overlap_indices.add(int(right["source_index"]))

    def tie_at(event: _RawEvent, pitch_index: int) -> str | None:
        if pitch_index < len(event.tie_types):
            return event.tie_types[pitch_index]
        return event.tie

    def extend_tie_chain(event: _RawEvent, pitch_index: int) -> tuple[list[tuple[_RawEvent, int]], int]:
        chain: list[tuple[_RawEvent, int]] = [(event, pitch_index)]
        current = event
        current_pitch_index = pitch_index
        while tie_at(current, current_pitch_index) in {"start", "continue"}:
            candidates: list[tuple[_RawEvent, int]] = []
            for candidate in events:
                if candidate.part_group != current.part_group or candidate.staff != current.staff or candidate.voice != current.voice:
                    continue
                if candidate.start_tick != current.end_tick:
                    continue
                for candidate_pitch_index, pitch in enumerate(candidate.pitches):
                    if pitch != event.pitches[pitch_index] or (candidate.event_id, candidate_pitch_index) in used:
                        continue
                    if tie_at(candidate, candidate_pitch_index) in {"stop", "continue"}:
                        candidates.append((candidate, candidate_pitch_index))
            if not candidates:
                break
            next_event, next_pitch_index = min(candidates, key=lambda value: (value[0].end_tick, value[0].event_id, value[1]))
            used.add((next_event.event_id, next_pitch_index))
            chain.append((next_event, next_pitch_index))
            current = next_event
            current_pitch_index = next_pitch_index
        return chain, chain[-1][0].end_tick

    for source in source_notes:
        source_index = int(source["source_index"])
        source_overlaps = source_index in overlap_indices
        overlap_starts = [
            other["start_tick"]
            for other in source_notes
            if other["midi"] == source["midi"]
            and other["start_tick"] > source["start_tick"]
            and other["start_tick"] < source["end_tick"]
        ]
        candidates: list[tuple[float, _RawEvent, int]] = []
        for event in events:
            for pitch_index, pitch in enumerate(event.pitches):
                if pitch != source["midi"] or (event.event_id, pitch_index) in used:
                    continue
                distance = abs(event.start_tick - source["start_tick"])
                if distance <= 4:
                    candidates.append((float(distance), event, pitch_index))
        if candidates:
            _distance, event, pitch_index = min(candidates, key=lambda value: (value[0], value[1].event_id, value[2]))
            used.add((event.event_id, pitch_index))
            original_start = event.start_tick
            original_end = event.end_tick
            tie_chain, alignment_end = extend_tie_chain(event, pitch_index)
            original_chain_end = tie_chain[-1][0].end_tick
            reason = "matched_musicxml_event"
            # A regular adaptive-quantized match is deliberately untouched.
            # Restore only a single-pitch event cut off at the next overlapping
            # source onset; this is the exact failure observed in the phase 4
            # same-pitch probe.
            if (
                source_overlaps
                and all(len(item[0].pitches) == 1 for item in tie_chain)
                and alignment_end < source["end_tick"]
                and any(abs(alignment_end - start) <= 1 for start in overlap_starts)
            ):
                tie_chain[-1][0].end_tick = max(tie_chain[-1][0].start_tick + 1, source["end_tick"])
                alignment_end = tie_chain[-1][0].end_tick
                reason = "musescore_truncated_source_span_restored"
            elif len(tie_chain) > 1:
                reason = "matched_musicxml_tie_chain"
            report.append(
                {
                    "source_index": source_index,
                    "source_midi": source["midi"],
                    "source_start_tick_480": source["start_tick_480"],
                    "source_end_tick_480": source["end_tick_480"],
                    "source_start_tick": source["start_tick"],
                    "source_end_tick": source["end_tick"],
                    "musicxml_event_id": event.event_id,
                    "musicxml_event_ids": [item[0].event_id for item in tie_chain],
                    "musicxml_start_tick": original_start,
                    "musicxml_end_tick": original_end,
                    "musicxml_chain_end_tick": original_chain_end,
                    "score_start_tick": event.start_tick,
                    "score_end_tick": alignment_end,
                    "source_to_score_movement_start_ticks": event.start_tick - source["start_tick"],
                    "source_to_score_movement_end_ticks": alignment_end - source["end_tick"],
                    "musicxml_to_score_movement_start_ticks": event.start_tick - original_start,
                    "musicxml_to_score_movement_end_ticks": alignment_end - original_chain_end,
                    "reason": reason,
                }
            )
        else:
            if not source_overlaps:
                report.append(
                    {
                        "source_index": source_index,
                        "source_midi": source["midi"],
                        "source_start_tick_480": source["start_tick_480"],
                        "source_end_tick_480": source["end_tick_480"],
                        "source_start_tick": source["start_tick"],
                        "source_end_tick": source["end_tick"],
                        "musicxml_event_id": None,
                        "score_start_tick": None,
                        "score_end_tick": None,
                        "source_to_score_movement_start_ticks": None,
                        "source_to_score_movement_end_ticks": None,
                        "musicxml_to_score_movement_start_ticks": None,
                        "musicxml_to_score_movement_end_ticks": None,
                        "reason": "missing_from_musicxml_unmatched",
                    }
                )
                unmatched.append(
                    {
                        "source_index": source_index,
                        "midi": int(source["midi"]),
                    }
                )
                continue
            restored = _RawEvent(
                event_id=f"restored-source-{source_index}",
                part_group="restored-source",
                part_id="restored-source",
                staff=1,
                voice="restored",
                start_tick=source["start_tick"],
                end_tick=max(source["start_tick"] + 1, source["end_tick"]),
                pitches=[source["midi"]],
                kind="note",
                tie=None,
                tie_types=[],
                tuplet_actual=None,
                tuplet_normal=None,
                dots=0,
                measure_number=None,
                metadata={"alignment_reason": "missing_from_musicxml_restored_from_performance_metadata"},
            )
            events.append(restored)
            report.append(
                {
                    "source_index": source_index,
                    "source_midi": source["midi"],
                    "source_start_tick_480": source["start_tick_480"],
                    "source_end_tick_480": source["end_tick_480"],
                    "source_start_tick": source["start_tick"],
                    "source_end_tick": source["end_tick"],
                    "musicxml_event_id": None,
                    "score_start_tick": restored.start_tick,
                    "score_end_tick": restored.end_tick,
                    "source_to_score_movement_start_ticks": 0,
                    "source_to_score_movement_end_ticks": 0,
                    "musicxml_to_score_movement_start_ticks": None,
                    "musicxml_to_score_movement_end_ticks": None,
                    "reason": "missing_from_musicxml_restored_from_performance_metadata",
                }
            )
    if unmatched:
        details = "; ".join(
            f"index={item['source_index']},midi={item['midi']}"
            for item in unmatched
        )
        raise MusicXMLStandardizationError(
            "performance metadata contains source notes missing from MusicXML "
            f"and not eligible for overlap restoration: count={len(unmatched)}; {details}"
        )
    return report


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
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "source_musicxml": payload.source_path,
        "music21_version": payload.music21_version,
        "source_note_count": len(source_notes),
        "musicxml_event_count": len(raw_events),
        "score_voice_count": len(voices),
        "source_to_score": alignment,
        "repairs": diagnostics + lane_reasons,
        "source_note_policy": "performance metadata restores only proven same-pitch overlap truncation or omission when supplied",
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
        warnings.append("Source performance alignment changed or restored note spans; inspect alignment_report.json")
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
