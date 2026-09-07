"""Conservative cleanup for single-voice GAME note events.

GAME is intentionally kept as the only producer of vocal notes in the
production pipeline.  This module only repairs the small timing and pitch
fragments that are common at model boundaries; it does not infer harmony or
create a second voice.  Raw note fields are copied into a lineage record so
the caller can write both the model output and the cleaned artifact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .domain import NoteEvent

CLEANUP_SCHEMA_VERSION = "1.0"


class VocalCleanupError(ValueError):
    """Raised when GAME output cannot be made into one legal voice."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report or {})


@dataclass(frozen=True)
class VocalCleanupConfig:
    """Versioned, conservative thresholds expressed in musical beats."""

    version: str = CLEANUP_SCHEMA_VERSION
    merge_gap_beats: float = 1 / 32
    overlap_tolerance_beats: float = 1 / 64
    vibrato_fragment_max_beats: float = 1 / 16
    vibrato_cluster_max_beats: float = 1 / 4
    vibrato_cluster_gap_beats: float = 1 / 32
    raw_pitch_boundary_tolerance: float = 0.12
    min_duration_sec: float = 1e-7

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("cleanup config version must not be empty")
        for name in (
            "merge_gap_beats",
            "overlap_tolerance_beats",
            "vibrato_fragment_max_beats",
            "vibrato_cluster_max_beats",
            "vibrato_cluster_gap_beats",
            "raw_pitch_boundary_tolerance",
            "min_duration_sec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.raw_pitch_boundary_tolerance >= 0.5:
            raise ValueError("raw_pitch_boundary_tolerance must be less than a semitone")

    def thresholds(self, seconds_per_beat: float) -> dict[str, float]:
        return {
            "merge_gap_beats": float(self.merge_gap_beats),
            "merge_gap_sec": float(self.merge_gap_beats * seconds_per_beat),
            "overlap_tolerance_beats": float(self.overlap_tolerance_beats),
            "overlap_tolerance_sec": float(self.overlap_tolerance_beats * seconds_per_beat),
            "vibrato_fragment_max_beats": float(self.vibrato_fragment_max_beats),
            "vibrato_fragment_max_sec": float(self.vibrato_fragment_max_beats * seconds_per_beat),
            "vibrato_cluster_max_beats": float(self.vibrato_cluster_max_beats),
            "vibrato_cluster_max_sec": float(self.vibrato_cluster_max_beats * seconds_per_beat),
            "vibrato_cluster_gap_beats": float(self.vibrato_cluster_gap_beats),
            "vibrato_cluster_gap_sec": float(self.vibrato_cluster_gap_beats * seconds_per_beat),
            "raw_pitch_boundary_tolerance_semitones": float(self.raw_pitch_boundary_tolerance),
            "min_duration_sec": float(self.min_duration_sec),
        }


@dataclass(frozen=True)
class VocalCleanupResult:
    """Immutable event collection plus the JSON-ready audit report."""

    events: tuple[NoteEvent, ...]
    report: dict[str, Any]

    def __iter__(self):
        # Permit the natural ``events, report = clean_vocal_events(...)`` form
        # while retaining named attributes for callers that prefer them.
        yield self.events
        yield self.report


@dataclass
class _WorkEvent:
    source_indices: tuple[int, ...]
    lineage: list[dict[str, Any]]
    start_sec: float
    end_sec: float
    midi: int
    confidence: float | None
    velocity: int | None
    raw_pitch: float | None
    voice_id: str
    source: str
    stem_id: str | None
    metadata: dict[str, Any]
    actions: list[dict[str, Any]]

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _effective_bpm(bpm: float | None, beat_context: Mapping[str, Any] | None) -> float:
    value: Any = bpm
    if value is None and beat_context is not None:
        for key in ("bpm", "tempo_bpm", "analysis_bpm"):
            if beat_context.get(key) is not None:
                value = beat_context[key]
                break
    if value is None:
        raise ValueError("GAME vocal cleanup requires bpm or beat_context.bpm")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("bpm must be finite and greater than zero")
    return result


def _seconds_per_beat(bpm: float, beat_context: Mapping[str, Any] | None) -> float:
    if beat_context is not None and beat_context.get("seconds_per_beat") is not None:
        value = float(beat_context["seconds_per_beat"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("beat_context.seconds_per_beat must be finite and greater than zero")
        return value
    return 60.0 / bpm


def _copy_json_value(value: Any) -> Any:
    """Copy metadata while keeping the report JSON-serializable by contract."""

    return deepcopy(value)


def _clean_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _copy_json_value(value)
        for key, value in metadata.items()
        if key != "vocal_cleanup"
    }


def _lineage_record(index: int, event: NoteEvent) -> dict[str, Any]:
    return {
        "index": int(index),
        "source_index": int(index),
        "midi": int(event.midi),
        "start_sec": float(event.start_sec),
        "end_sec": float(event.end_sec),
        "raw_pitch": None if event.raw_pitch is None else float(event.raw_pitch),
        "metadata": _clean_metadata(event.metadata),
    }


def _existing_lineage(index: int, event: NoteEvent) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current = event.metadata.get("vocal_cleanup")
    if not isinstance(current, Mapping):
        return [_lineage_record(index, event)], []
    raw_lineage = current.get("lineage")
    if not isinstance(raw_lineage, list) or not raw_lineage:
        return [_lineage_record(index, event)], []
    lineage: list[dict[str, Any]] = []
    for item in raw_lineage:
        if not isinstance(item, Mapping):
            continue
        try:
            raw_index = item.get("source_index")
            if raw_index is None:
                raw_index = item["index"]
            original_index = int(raw_index)
            midi = int(item["midi"])
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_pitch = item.get("raw_pitch")
        lineage.append(
            {
                "index": original_index,
                "source_index": original_index,
                "midi": midi,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "raw_pitch": None if raw_pitch is None else float(raw_pitch),
                "metadata": _copy_json_value(item.get("metadata", {})),
            }
        )
    if not lineage:
        lineage = [_lineage_record(index, event)]
    lineage.sort(key=lambda item: item["index"])
    actions = current.get("actions", [])
    existing_actions = [
        _copy_json_value(item) for item in actions if isinstance(item, Mapping)
    ] if isinstance(actions, list) else []
    return lineage, existing_actions


def _work_event(index: int, event: NoteEvent) -> _WorkEvent:
    lineage, actions = _existing_lineage(index, event)
    source_indices = tuple(sorted({int(item["index"]) for item in lineage}))
    return _WorkEvent(
        source_indices=source_indices,
        lineage=lineage,
        start_sec=float(event.start_sec),
        end_sec=float(event.end_sec),
        midi=int(event.midi),
        confidence=None if event.confidence is None else float(event.confidence),
        velocity=None if event.velocity is None else int(event.velocity),
        raw_pitch=None if event.raw_pitch is None else float(event.raw_pitch),
        voice_id=event.voice_id,
        source=event.source,
        stem_id=event.stem_id,
        metadata=_clean_metadata(event.metadata),
        actions=actions,
    )


def _weighted(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _value, weight in values)
    if total_weight <= 0:
        return values[0][0]
    return sum(value * weight for value, weight in values) / total_weight


def _dedupe_actions(actions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        copied = _copy_json_value(dict(action))
        identity = repr(sorted(copied.items()))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(copied)
    return result


def _combine_events(left: _WorkEvent, right: _WorkEvent, action: Mapping[str, Any]) -> _WorkEvent:
    left_weight = max(left.duration_sec, 0.0)
    right_weight = max(right.duration_sec, 0.0)
    confidence = _weighted(
        [
            (value, weight)
            for value, weight in ((left.confidence, left_weight), (right.confidence, right_weight))
            if value is not None
        ]
    )
    velocity_value = _weighted(
        [
            (float(value), weight)
            for value, weight in ((left.velocity, left_weight), (right.velocity, right_weight))
            if value is not None
        ]
    )
    raw_pitch = _weighted(
        [
            (value, weight)
            for value, weight in ((left.raw_pitch, left_weight), (right.raw_pitch, right_weight))
            if value is not None
        ]
    )
    actions = _dedupe_actions([*left.actions, *right.actions, action])
    lineage = sorted(
        [*deepcopy(left.lineage), *deepcopy(right.lineage)],
        key=lambda item: item["index"],
    )
    return _WorkEvent(
        source_indices=tuple(sorted({int(item["index"]) for item in lineage})),
        lineage=lineage,
        start_sec=left.start_sec,
        end_sec=right.end_sec,
        midi=left.midi,
        confidence=None if confidence is None else float(confidence),
        velocity=None if velocity_value is None else max(1, min(127, round(velocity_value))),
        raw_pitch=None if raw_pitch is None else float(raw_pitch),
        voice_id=left.voice_id,
        source=left.source,
        stem_id=left.stem_id,
        metadata=deepcopy(left.metadata),
        actions=actions,
    )


def _is_rearticulation(event: _WorkEvent) -> bool:
    metadata = event.metadata
    if metadata.get("rearticulation") or metadata.get("separate_onset") or metadata.get("preserve_boundary"):
        return True
    return str(metadata.get("articulation", "")).lower() in {"staccato", "rearticulated", "accented"}


def _append_action(event: _WorkEvent, action: Mapping[str, Any]) -> None:
    event.actions = _dedupe_actions([*event.actions, action])


def _prepare_events(
    materialized: list[NoteEvent],
    *,
    config: VocalCleanupConfig,
    seconds_per_beat: float,
    report: dict[str, Any],
) -> tuple[list[_WorkEvent], int, list[dict[str, Any]]]:
    voice_ids = {event.voice_id for event in materialized}
    if len(voice_ids) > 1:
        report["failure"] = {"reason": "multiple_voice_ids", "voice_ids": sorted(voice_ids)}
        raise VocalCleanupError(
            "GAME vocal cleanup accepts one voice only; multiple voice_ids were supplied",
            report=report,
        )
    ordered_indices = sorted(
        range(len(materialized)),
        key=lambda index: (materialized[index].start_sec, materialized[index].end_sec, index),
    )
    reordered_count = sum(index != ordered_indices[index] for index in range(len(ordered_indices)))
    work = [_work_event(index, materialized[index]) for index in ordered_indices]
    overlap_tolerance = config.overlap_tolerance_beats * seconds_per_beat
    overlap_adjustments = 0
    actions: list[dict[str, Any]] = []
    for previous, current in pairwise(work):
        overlap = previous.end_sec - current.start_sec
        if overlap <= 0:
            continue
        if overlap > overlap_tolerance:
            report["failure"] = {
                "reason": "large_overlap",
                "source_indices": [*previous.source_indices, *current.source_indices],
                "overlap_sec": float(overlap),
                "overlap_tolerance_sec": float(overlap_tolerance),
            }
            raise VocalCleanupError(
                f"GAME vocal notes overlap by {overlap:.6f}s, exceeding the "
                f"{overlap_tolerance:.6f}s cleanup tolerance",
                report=report,
            )
        boundary = (previous.end_sec + current.start_sec) / 2.0
        if boundary <= previous.start_sec + config.min_duration_sec or boundary >= current.end_sec - config.min_duration_sec:
            report["failure"] = {
                "reason": "overlap_cannot_be_adjusted",
                "source_indices": [*previous.source_indices, *current.source_indices],
                "overlap_sec": float(overlap),
            }
            raise VocalCleanupError("GAME overlap leaves no positive duration after adjustment", report=report)
        old_end = previous.end_sec
        old_start = current.start_sec
        previous.end_sec = boundary
        current.start_sec = boundary
        action = {
            "action": "overlap_adjustment",
            "reason": "small_boundary_overlap",
            "source_indices": [*previous.source_indices, *current.source_indices],
            "previous_end_before": float(old_end),
            "current_start_before": float(old_start),
            "boundary_after": float(boundary),
        }
        _append_action(previous, action)
        _append_action(current, action)
        actions.append(action)
        overlap_adjustments += 1
    report["reordered_count"] = reordered_count
    return work, overlap_adjustments, actions


def _boundary_evidence(event: _WorkEvent, target_midi: int, tolerance: float) -> bool:
    if event.raw_pitch is None or abs(event.midi - target_midi) != 1:
        return False
    boundary = min(event.midi, target_midi) + 0.5
    return abs(event.raw_pitch - boundary) <= tolerance


def _suppress_vibrato(
    work: list[_WorkEvent],
    *,
    config: VocalCleanupConfig,
    seconds_per_beat: float,
) -> tuple[int, list[dict[str, Any]]]:
    max_fragment = config.vibrato_fragment_max_beats * seconds_per_beat
    max_cluster = config.vibrato_cluster_max_beats * seconds_per_beat
    max_gap = config.vibrato_cluster_gap_beats * seconds_per_beat
    suppressed = 0
    actions: list[dict[str, Any]] = []
    index = 0
    while index < len(work):
        event = work[index]
        if event.duration_sec > max_fragment or event.raw_pitch is None:
            index += 1
            continue
        end = index
        while end + 1 < len(work):
            candidate = work[end + 1]
            gap = candidate.start_sec - work[end].end_sec
            if (
                candidate.duration_sec > max_fragment
                or candidate.raw_pitch is None
                or gap > max_gap
                or candidate.end_sec - event.start_sec > max_cluster
                or abs(candidate.midi - work[end].midi) > 1
            ):
                break
            end += 1
        run = work[index : end + 1]
        left = work[index - 1] if index else None
        right = work[end + 1] if end + 1 < len(work) else None
        target: int | None = None
        if (
            left is not None
            and right is not None
            and left.midi == right.midi
            and all(abs(item.midi - left.midi) <= 1 for item in run)
            and all(
                item.midi == left.midi or _boundary_evidence(item, left.midi, config.raw_pitch_boundary_tolerance)
                for item in run
            )
        ):
            target = left.midi
        if target is None and len(run) >= 3:
            candidates = [*run]
            if left is not None:
                candidates.append(left)
            if right is not None:
                candidates.append(right)
            counts: dict[int, int] = {}
            durations: dict[int, float] = {}
            for item in candidates:
                counts[item.midi] = counts.get(item.midi, 0) + 1
                durations[item.midi] = durations.get(item.midi, 0.0) + item.duration_sec
            ranked = sorted(counts, key=lambda midi: (-counts[midi], -durations[midi], midi))
            for candidate_midi in ranked:
                if counts[candidate_midi] < 2:
                    continue
                if all(
                    item.midi == candidate_midi
                    or _boundary_evidence(item, candidate_midi, config.raw_pitch_boundary_tolerance)
                    for item in run
                ):
                    target = candidate_midi
                    break
        if (
            target is None
            and len(run) == 1
            and left is not None
            and right is not None
            and left.midi == right.midi
            and _boundary_evidence(run[0], left.midi, config.raw_pitch_boundary_tolerance)
        ):
            target = left.midi
        if target is not None:
            for item in run:
                if item.midi == target:
                    continue
                if not _boundary_evidence(item, target, config.raw_pitch_boundary_tolerance):
                    continue
                old_midi = item.midi
                action = {
                    "action": "vibrato_suppressed",
                    "reason": "raw_pitch_half_semitone_boundary",
                    "source_indices": list(item.source_indices),
                    "from_midi": old_midi,
                    "to_midi": target,
                    "raw_pitch": item.raw_pitch,
                }
                item.midi = target
                _append_action(item, action)
                actions.append(action)
                suppressed += 1
        index = end + 1
    return suppressed, actions


def _merge_same_pitch(
    work: list[_WorkEvent],
    *,
    config: VocalCleanupConfig,
    seconds_per_beat: float,
) -> tuple[list[_WorkEvent], int, list[dict[str, Any]]]:
    threshold = config.merge_gap_beats * seconds_per_beat
    result: list[_WorkEvent] = []
    merge_count = 0
    actions: list[dict[str, Any]] = []
    for event in work:
        if result:
            previous = result[-1]
            gap = event.start_sec - previous.end_sec
            if previous.midi == event.midi and gap < threshold and not _is_rearticulation(previous) and not _is_rearticulation(event):
                action = {
                    "action": "same_pitch_merge",
                    "reason": "same_midi_within_conservative_gap",
                    "source_indices": [*previous.source_indices, *event.source_indices],
                    "gap_sec": float(gap),
                    "midi": previous.midi,
                }
                result[-1] = _combine_events(previous, event, action)
                actions.append(action)
                merge_count += 1
                continue
        result.append(event)
    return result, merge_count, actions


def _output_event(event: _WorkEvent) -> NoteEvent:
    metadata = deepcopy(event.metadata)
    metadata["vocal_cleanup"] = {
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "source_indices": list(event.source_indices),
        "lineage": deepcopy(event.lineage),
        "actions": deepcopy(event.actions),
    }
    return NoteEvent(
        start_sec=event.start_sec,
        end_sec=event.end_sec,
        midi=event.midi,
        confidence=event.confidence,
        voice_id=event.voice_id,
        source=event.source,
        velocity=event.velocity,
        raw_pitch=event.raw_pitch,
        stem_id=event.stem_id,
        metadata=metadata,
    )


def _base_report(
    *,
    raw_count: int,
    bpm: float,
    seconds_per_beat: float,
    config: VocalCleanupConfig,
    beat_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "cleanup_config_version": config.version,
        "engine": "game",
        "semantics": "single_voice",
        "raw_count": raw_count,
        "cleaned_count": 0,
        "merge_count": 0,
        "vibrato_suppressed_count": 0,
        "overlap_adjustment_count": 0,
        "retained_count": 0,
        "reordered_count": 0,
        "bpm": float(bpm),
        "seconds_per_beat": float(seconds_per_beat),
        "thresholds": config.thresholds(seconds_per_beat),
        "beat_context": _copy_json_value(dict(beat_context)) if beat_context is not None else {},
        "actions": [],
        "events": [],
    }


def clean_vocal_events(
    events: Iterable[NoteEvent],
    bpm: float | None = None,
    *,
    beat_context: Mapping[str, Any] | None = None,
    config: VocalCleanupConfig | None = None,
) -> VocalCleanupResult:
    """Clean GAME's single-voice output without mutating its input objects.

    The returned events are a tuple.  ``VocalCleanupResult`` also supports
    tuple unpacking as ``cleaned, report = clean_vocal_events(...)``.
    """

    if beat_context is not None and not isinstance(beat_context, Mapping):
        raise TypeError("beat_context must be a mapping")
    selected_config = config or VocalCleanupConfig()
    effective_bpm = _effective_bpm(bpm, beat_context)
    seconds_per_beat = _seconds_per_beat(effective_bpm, beat_context)
    materialized = list(events)
    report = _base_report(
        raw_count=len(materialized),
        bpm=effective_bpm,
        seconds_per_beat=seconds_per_beat,
        config=selected_config,
        beat_context=beat_context,
    )
    if not materialized:
        return VocalCleanupResult((), report)
    work, overlap_adjustments, overlap_actions = _prepare_events(
        materialized,
        config=selected_config,
        seconds_per_beat=seconds_per_beat,
        report=report,
    )
    suppressed, vibrato_actions = _suppress_vibrato(
        work,
        config=selected_config,
        seconds_per_beat=seconds_per_beat,
    )
    work, merge_count_first, merge_actions_first = _merge_same_pitch(
        work,
        config=selected_config,
        seconds_per_beat=seconds_per_beat,
    )
    # Suppression can make two newly adjacent fragments equal.  A second pass
    # keeps the operation deterministic and ensures the output is idempotent.
    work, merge_count_second, merge_actions_second = _merge_same_pitch(
        work,
        config=selected_config,
        seconds_per_beat=seconds_per_beat,
    )
    if any(event.end_sec <= event.start_sec for event in work) or any(
        right.start_sec < left.end_sec - selected_config.min_duration_sec
        for left, right in pairwise(work)
    ):
        report["failure"] = {"reason": "cleanup_output_not_single_voice"}
        raise VocalCleanupError("GAME cleanup could not produce a strictly non-overlapping voice", report=report)

    output = tuple(_output_event(event) for event in work)
    all_actions = [
        *overlap_actions,
        *vibrato_actions,
        *merge_actions_first,
        *merge_actions_second,
    ]
    report["cleaned_count"] = len(output)
    report["merge_count"] = merge_count_first + merge_count_second
    report["vibrato_suppressed_count"] = suppressed
    report["overlap_adjustment_count"] = overlap_adjustments
    report["retained_count"] = sum(not event.actions for event in work)
    report["actions"] = _dedupe_actions(all_actions)
    report["events"] = [
        {
            "output_index": index,
            "source_indices": list(event.source_indices),
            "lineage": deepcopy(event.lineage),
            "start_sec": float(event.start_sec),
            "end_sec": float(event.end_sec),
            "midi": int(event.midi),
            "raw_pitch": None if event.raw_pitch is None else float(event.raw_pitch),
            "actions": deepcopy(event.actions),
        }
        for index, event in enumerate(work)
    ]
    return VocalCleanupResult(output, report)


def cleanup_vocal_events(
    events: Iterable[NoteEvent],
    bpm: float | None = None,
    *,
    beat_context: Mapping[str, Any] | None = None,
    config: VocalCleanupConfig | None = None,
) -> VocalCleanupResult:
    """Compatibility spelling for callers that use the verb-first name."""

    return clean_vocal_events(events, bpm, beat_context=beat_context, config=config)


__all__ = [
    "CLEANUP_SCHEMA_VERSION",
    "VocalCleanupConfig",
    "VocalCleanupError",
    "VocalCleanupResult",
    "clean_vocal_events",
    "cleanup_vocal_events",
]
