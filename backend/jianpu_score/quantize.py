"""Voice selection, shared beat quantization and jianpu text generation."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from typing import Any

from .beat_grid import BeatGridError, beat_unit_from_grid, resolve_beat_unit
from .domain import (
    MusicAnalysis,
    NoteEvent,
    Score,
    ScoreNote,
    ScoreVoice,
    TempoEvent,
    confidence_value,
    normalize_key,
    normalize_time_signature,
    relative_major_key,
    sanitize_title,
)


# Keep onset jitter around the first downbeat on the shared phase rather than
# switching between a zero shift and a measure-relative alignment.
DOWNBEAT_PHASE_DEADBAND_QUARTERS = 0.5


def _time_signature_values(value: str) -> tuple[int, int]:
    normalized = normalize_time_signature(value)
    numerator, denominator = normalized.split("/", 1)
    return int(numerator), int(denominator)


class NoNotesError(ValueError):
    """Raised when an engine produces no usable note events."""


class JianpuSerializationError(ValueError):
    """Raised when a Score cannot be represented without changing its timing."""


def _shared_timeline_event_starts(
    analysis: MusicAnalysis,
    events: Iterable[NoteEvent],
) -> tuple[tuple[float, float], str]:
    """Return the full-song event bounds used to decide score phase.

    V2 renders each selected instrument separately, while its persisted
    ``MusicAnalysis.note_events`` still describes the complete recognition.
    The latter is therefore the shared timeline authority.  Direct callers
    that do not provide analysis events fall back to the materialized events;
    that scope is persisted in ``score_origin`` so it cannot be mistaken for
    a proven full-song origin.
    """

    source: Iterable[Any]
    scope: str
    shared_bounds = analysis.metadata.get("shared_timeline_event_bounds")
    if isinstance(shared_bounds, list) and shared_bounds:
        source = shared_bounds
        scope = str(
            analysis.metadata.get("shared_timeline_scope")
            or "shared_timeline_event_bounds"
        )
    elif analysis.note_events:
        source = analysis.note_events
        scope = "analysis_note_events"
    else:
        source = events
        scope = "materialized_events"
    bounds: list[tuple[float, float]] = []
    for item in source:
        if isinstance(item, NoteEvent):
            bounds.append((float(item.start_sec), float(item.end_sec)))
            continue
        if isinstance(item, Mapping):
            try:
                bounds.append((float(item["start_sec"]), float(item["end_sec"])))
            except (KeyError, TypeError, ValueError):
                continue
    return tuple(bounds), scope


def _voice_name(event: NoteEvent, layer_index: int) -> str:
    stem = event.stem_id.strip() if event.stem_id else ""
    return f"{stem}:voice-{layer_index}" if stem else f"voice-{layer_index}"


def _melody_event_score(event: NoteEvent) -> float:
    """Score one candidate for the dynamic-programming melody path."""

    confidence = confidence_value(event.confidence)
    duration_bonus = min(event.duration_sec / 0.5, 2.0) * 0.35
    register = max(0.0, min(1.0, (event.midi - 36) / 48.0))
    return 1.45 * confidence + duration_bonus + 0.25 * register


def _melody_transition_score(previous: NoteEvent, current: NoteEvent) -> float:
    pitch_jump = abs(current.midi - previous.midi)
    duration_ratio = abs(math.log((current.duration_sec + 1e-3) / (previous.duration_sec + 1e-3)))
    gap = max(0.0, current.start_sec - previous.end_sec)
    return 0.55 - 0.055 * min(pitch_jump, 24) - 0.08 * min(duration_ratio, 4.0) - 0.04 * min(gap, 2.0)


def _select_melody_path(ordered: list[NoteEvent]) -> list[NoteEvent]:
    """Choose a coherent non-overlapping candidate path with interval DP."""

    if not ordered:
        return []
    count = len(ordered)
    best_scores = [_melody_event_score(event) for event in ordered]
    predecessors: list[int | None] = [None] * count
    for current_index, current in enumerate(ordered):
        for previous_index in range(current_index):
            previous = ordered[previous_index]
            if previous.end_sec > current.start_sec + 1e-4:
                continue
            candidate = best_scores[previous_index] + _melody_transition_score(previous, current) + _melody_event_score(current)
            if candidate > best_scores[current_index]:
                best_scores[current_index] = candidate
                predecessors[current_index] = previous_index
    last_index = max(range(count), key=best_scores.__getitem__)
    path: list[NoteEvent] = []
    while last_index is not None:
        path.append(ordered[last_index])
        last_index = predecessors[last_index]
    return list(reversed(path))


def select_voice_events(events: Iterable[NoteEvent], mode: str = "polyphonic") -> list[NoteEvent]:
    """Select a coherent melody or partition every event into layers."""

    ordered = sorted(events, key=lambda event: (event.start_sec, -confidence_value(event.confidence), event.midi))
    if mode not in {"monophonic", "polyphonic"}:
        raise ValueError("mode must be monophonic or polyphonic")
    if mode == "monophonic":
        selected = _select_melody_path(ordered)
        return [event.model_copy(update={"voice_id": _voice_name(event, 0)}) for event in selected]

    # Layers are independent within each stem.  A long bass note must not
    # consume a layer number that later causes two other-stem notes to split
    # across unrelated score voices.
    layers_by_stem: dict[str, list[list[NoteEvent]]] = {}
    ends_by_stem: dict[str, list[float]] = {}
    for event in ordered:
        stem_key = event.stem_id or ""
        layers = layers_by_stem.setdefault(stem_key, [])
        layer_ends = ends_by_stem.setdefault(stem_key, [])
        placed = False
        for layer_index, end in enumerate(layer_ends):
            if event.start_sec >= end - 1e-4:
                layers[layer_index].append(event.model_copy(update={"voice_id": _voice_name(event, layer_index)}))
                layer_ends[layer_index] = max(layer_ends[layer_index], event.end_sec)
                placed = True
                break
        if not placed:
            layer_index = len(layers)
            layers.append([event.model_copy(update={"voice_id": _voice_name(event, layer_index)})])
            layer_ends.append(event.end_sec)
    return [event for layers in layers_by_stem.values() for layer in layers for event in layer]


@dataclass(frozen=True)
class _BeatMapper:
    bpm: float
    beat_times: tuple[float, ...]
    fixed: bool
    shift_beats: float = 0.0
    timeline_offset_beats: float = 0.0
    beat_scale: float = 1.0
    beat_duration_quarters: float = 1.0
    beat_unit: str = "quarter"
    beat_unit_source: str = "standard_meter_definition"
    beat_unit_proven: bool = True
    beats_per_bar: int = 4
    bar_duration_quarters: float = 4.0
    score_origin: dict[str, Any] = field(default_factory=dict)

    def seconds_to_beat(self, seconds: float) -> float:
        if self.fixed or len(self.beat_times) < 2:
            return seconds * self.bpm / 60.0 + self.shift_beats
        if seconds <= self.beat_times[0]:
            interval = self.beat_times[1] - self.beat_times[0]
            position = (seconds - self.beat_times[0]) / interval
        elif seconds >= self.beat_times[-1]:
            interval = self.beat_times[-1] - self.beat_times[-2]
            position = len(self.beat_times) - 1 + (seconds - self.beat_times[-1]) / interval
        else:
            right = bisect_right(self.beat_times, seconds)
            left = right - 1
            fraction = (seconds - self.beat_times[left]) / (self.beat_times[right] - self.beat_times[left])
            position = left + fraction
        return (
            position * self.beat_scale * self.beat_duration_quarters
            + self.shift_beats
            + self.timeline_offset_beats
        )


def _build_beat_mapper(analysis: MusicAnalysis, events: list[NoteEvent]) -> _BeatMapper:
    beat_times = tuple(analysis.beat_times)
    beat_grid = analysis.metadata.get("beat_grid")
    mapping = beat_grid.get("mapping", {}) if isinstance(beat_grid, dict) else {}
    beat_scale = float(mapping.get("manual_bpm_scale", 1.0) or 1.0)
    try:
        beat_unit = (
            beat_unit_from_grid(beat_grid, time_signature=analysis.time_signature, legacy_compat=True)
            if isinstance(beat_grid, dict) and beat_grid
            else resolve_beat_unit(analysis.time_signature, legacy_compat=True)
        )
    except (BeatGridError, ValueError) as exc:
        raise ValueError(f"invalid beat-grid unit semantics: {exc}") from exc
    beat_duration_quarters = float(beat_unit["beat_duration_quarters"])
    beats_per_bar = int(beat_unit["beats_per_bar"])
    bar_duration_quarters = float(beat_unit["bar_duration_quarters"])
    # A legacy hand-built MusicAnalysis marked manual_bpm has no BeatNet phase
    # map and must retain its historical fixed-grid behavior.  New BeatNet
    # analyses carry a beat_grid and keep the detected phase/local timing even
    # when the user overrides BPM.
    fixed = (analysis.metadata.get("beat_source") == "manual_bpm" and not beat_grid) or len(beat_times) < 2
    if fixed:
        return _BeatMapper(
            analysis.bpm,
            beat_times,
            True,
            beat_duration_quarters=beat_duration_quarters,
            beat_unit=str(beat_unit["beat_unit"]),
            beat_unit_source=str(beat_unit["source"]),
            beat_unit_proven=bool(beat_unit["proven"]),
            beats_per_bar=beats_per_bar,
            bar_duration_quarters=bar_duration_quarters,
            score_origin={
                "strategy": "legacy_fixed_bpm",
                "downbeat_status": "undetermined",
                "pickup_candidate": False,
                "pickup_beats": 0.0,
            },
        )
    unshifted = _BeatMapper(
        analysis.bpm,
        beat_times,
        False,
        beat_scale=beat_scale,
        beat_duration_quarters=beat_duration_quarters,
        beat_unit=str(beat_unit["beat_unit"]),
        beat_unit_source=str(beat_unit["source"]),
        beat_unit_proven=bool(beat_unit["proven"]),
        beats_per_bar=beats_per_bar,
        bar_duration_quarters=bar_duration_quarters,
    )

    if not isinstance(beat_grid, dict) or not beat_grid:
        # Keep the legacy non-BeatNet contract for hand-built analyses.  The
        # production BeatNet path always persists beat_grid.json and therefore
        # uses the explicit downbeat origin below.
        shift_beats = -unshifted.seconds_to_beat(0.0)
        return _BeatMapper(
            analysis.bpm,
            beat_times,
            False,
            shift_beats=shift_beats,
            beat_scale=beat_scale,
            beat_duration_quarters=beat_duration_quarters,
            beat_unit=str(beat_unit["beat_unit"]),
            beat_unit_source=str(beat_unit["source"]),
            beat_unit_proven=bool(beat_unit["proven"]),
            beats_per_bar=beats_per_bar,
            bar_duration_quarters=bar_duration_quarters,
            score_origin={
                "strategy": "legacy_audio_zero",
                "downbeat_status": "undetermined",
                "pickup_candidate": False,
                "pickup_beats": 0.0,
                "origin_shift_beats": shift_beats,
            },
        )

    # A BeatNet grid owns the phase of the score.  The old implementation
    # translated audio zero to Score zero, which silently moved a detected
    # downbeat into the middle of a measure whenever the first returned beat
    # was not the downbeat.  Keep that phase in one full-song coordinate system
    # and only align it when the first event is clearly outside the phase
    # ambiguity region.
    origin = mapping.get("score_origin") if isinstance(mapping, dict) else None
    grid_beats = beat_grid.get("beats", []) if isinstance(beat_grid, dict) else []
    first_downbeat_index: int | None = None
    first_downbeat_sec: float | None = None
    if isinstance(origin, dict) and origin.get("downbeat_index") is not None:
        try:
            candidate_index = int(origin["downbeat_index"])
        except (TypeError, ValueError):
            candidate_index = -1
        if 0 <= candidate_index < len(beat_times):
            first_downbeat_index = candidate_index
            first_downbeat_sec = float(beat_times[candidate_index])
    if first_downbeat_index is None and isinstance(grid_beats, list):
        for index, beat in enumerate(grid_beats):
            if (
                isinstance(beat, Mapping)
                and bool(beat.get("downbeat"))
                and index < len(beat_times)
            ):
                first_downbeat_index = index
                first_downbeat_sec = float(beat_times[index])
                break

    if first_downbeat_index is None:
        timeline_bounds, timeline_scope = _shared_timeline_event_starts(analysis, events)
        earliest_event_sec = min((start for start, _end in timeline_bounds), default=0.0)
        timeline_offset_beats = max(0.0, -unshifted.seconds_to_beat(earliest_event_sec))
        return _BeatMapper(
            analysis.bpm,
            beat_times,
            False,
            shift_beats=0.0,
            timeline_offset_beats=timeline_offset_beats,
            beat_scale=beat_scale,
            beat_duration_quarters=beat_duration_quarters,
            beat_unit=str(beat_unit["beat_unit"]),
            beat_unit_source=str(beat_unit["source"]),
            beat_unit_proven=bool(beat_unit["proven"]),
            beats_per_bar=beats_per_bar,
            bar_duration_quarters=bar_duration_quarters,
            score_origin={
                "strategy": "first_beat_fallback",
                "downbeat_status": "undetermined",
                "downbeat_index": None,
                "downbeat_sec": None,
                "downbeat_score_beat": None,
                "pickup_candidate": False,
                "pickup_beats": 0.0,
                "pickup_span_quarters": 0.0,
                "pickup_evidence": {
                    "observed_pre_downbeat": False,
                    "reason": "downbeat_index_unavailable",
                    "pickup_span_quarters": 0.0,
                },
                "timeline_offset_beats": timeline_offset_beats,
                "timeline_scope": timeline_scope,
                "timeline_event_count": len(timeline_bounds),
            },
        )

    downbeat_raw_beat = first_downbeat_index * beat_scale * beat_duration_quarters
    timeline_bounds, timeline_scope = _shared_timeline_event_starts(analysis, events)
    earliest_event_sec = min((start for start, _end in timeline_bounds), default=0.0)
    earliest_raw_beat = unshifted.seconds_to_beat(earliest_event_sec)
    # ScoreNote and MIDI ticks are non-negative.  Preserve a negative raw
    # extrapolation with one explicit full-song timeline offset rather than
    # silently clipping each note independently.
    timeline_offset_beats = max(0.0, -earliest_raw_beat)
    phase_delta_quarters = earliest_raw_beat - downbeat_raw_beat
    has_pre_downbeat_note = first_downbeat_sec is not None and earliest_event_sec < first_downbeat_sec
    phase_near_downbeat = abs(phase_delta_quarters) <= DOWNBEAT_PHASE_DEADBAND_QUARTERS
    phase_undetermined = has_pre_downbeat_note or phase_near_downbeat
    bar_beats = bar_duration_quarters
    raw_pre_downbeat_beat = earliest_raw_beat if has_pre_downbeat_note else None
    pickup_span = (
        max(0.0, downbeat_raw_beat - raw_pre_downbeat_beat)
        if raw_pre_downbeat_beat is not None
        else 0.0
    )
    # No production BeatNet payload currently carries a trustworthy explicit
    # pickup declaration.  Any pre-downbeat event is therefore phase
    # ambiguous, including a very small beat-relative onset deviation.  Keep
    # the measured span as evidence, but never let it authorize a relocation.
    # The same dead-band applies just after the downbeat so jitter cannot jump
    # between two origin conventions.
    if phase_undetermined:
        target_downbeat_beat = downbeat_raw_beat
        strategy = "downbeat_phase_undetermined"
        downbeat_status = "undetermined"
        pickup_reason = (
            "pickup_span_not_shorter_than_bar"
            if has_pre_downbeat_note and pickup_span >= bar_beats
            else (
                "pre_downbeat_phase_unconfirmed"
                if has_pre_downbeat_note
                else "downbeat_phase_within_deadband"
            )
        )
        warning = (
            "无法确认弱起（"
            f"{pickup_reason}；pickup_span_quarters={pickup_span:.6f}，"
            f"phase_delta_quarters={phase_delta_quarters:.6f}，"
            f"deadband_quarters={DOWNBEAT_PHASE_DEADBAND_QUARTERS:.2f}）；"
            "保留共享 BeatNet 原点"
        )
    else:
        target_downbeat_beat = 0.0
        strategy = "first_downbeat"
        downbeat_status = "aligned"
        pickup_reason = "no_pre_downbeat_note"
        warning = None
    # For the undetermined state this is intentionally an exact zero.  Do not
    # use a per-track audio origin or a millisecond tolerance to move a full
    # measure.  The ordinary branch retains the established downbeat alignment.
    shift_beats = target_downbeat_beat - downbeat_raw_beat
    if phase_undetermined:
        shift_beats = 0.0
    score_origin = {
        "strategy": strategy,
        "downbeat_status": downbeat_status,
        "downbeat_index": first_downbeat_index,
        "downbeat_sec": first_downbeat_sec,
        "downbeat_score_beat": target_downbeat_beat + timeline_offset_beats,
        "downbeat_bar_beats": bar_beats,
        "pickup_candidate": has_pre_downbeat_note,
        "pickup_beats": 0.0,
        "pickup_span_quarters": pickup_span,
        "pickup_evidence": {
            "observed_pre_downbeat": has_pre_downbeat_note,
            "reason": pickup_reason,
            "pickup_span_quarters": pickup_span,
            "phase_delta_quarters": phase_delta_quarters,
            "deadband_quarters": DOWNBEAT_PHASE_DEADBAND_QUARTERS,
        },
        "pre_downbeat_note_start_beat": raw_pre_downbeat_beat,
        "origin_shift_beats": shift_beats,
        "timeline_offset_beats": timeline_offset_beats,
        "timeline_scope": timeline_scope,
        "timeline_event_count": len(timeline_bounds),
        "warning": warning,
    }
    return _BeatMapper(
        analysis.bpm,
        beat_times,
        False,
        shift_beats=shift_beats,
        timeline_offset_beats=timeline_offset_beats,
        beat_scale=beat_scale,
        beat_duration_quarters=beat_duration_quarters,
        beat_unit=str(beat_unit["beat_unit"]),
        beat_unit_source=str(beat_unit["source"]),
        beat_unit_proven=bool(beat_unit["proven"]),
        beats_per_bar=beats_per_bar,
        bar_duration_quarters=bar_duration_quarters,
        score_origin=score_origin,
    )


def _straight_tick(value: float, quarter_ticks: int) -> int:
    sixteenth = max(1, quarter_ticks // 4)
    return round(value / sixteenth) * sixteenth


def _triplet_group_targets(raw: list[tuple[float, float, NoteEvent]], start_index: int, quarter_ticks: int) -> tuple[int, int, int, int] | None:
    if start_index + 2 >= len(raw):
        return None
    first, second, third = raw[start_index : start_index + 3]
    target_step = quarter_ticks / 3.0
    starts = [first[0], second[0], third[0]]
    ends = [first[1], second[1], third[1]]
    if any(abs((starts[index + 1] - starts[index]) - target_step) > 0.35 for index in range(2)):
        return None
    if any(abs((ends[index] - starts[index]) - target_step) > 0.45 for index in range(3)):
        return None
    if abs(first[1] - second[0]) > 0.45 or abs(second[1] - third[0]) > 0.45:
        return None
    anchor = round(first[0] / quarter_ticks) * quarter_ticks
    targets = (anchor, round(anchor + target_step), round(anchor + 2 * target_step), round(anchor + quarter_ticks))
    observed = [first[0], second[0], third[0], third[1]]
    straight_error = sum(abs(value - _straight_tick(value, quarter_ticks)) for value in observed)
    triplet_error = sum(abs(value - target) for value, target in zip(observed, targets))
    complexity_penalty = 0.15 * len(observed)
    if triplet_error + complexity_penalty >= straight_error:
        return None
    return targets


def _quantize_voice_boundaries(
    raw: list[tuple[float, float, NoteEvent]], quarter_ticks: int
) -> tuple[list[tuple[int, int, NoteEvent]], int]:
    snapped = [[_straight_tick(start, quarter_ticks), _straight_tick(end, quarter_ticks)] for start, end, _ in raw]
    triplet_groups = 0
    index = 0
    while index + 2 < len(raw):
        targets = _triplet_group_targets(raw, index, quarter_ticks)
        if targets is None:
            index += 1
            continue
        snapped[index] = [targets[0], targets[1]]
        snapped[index + 1] = [targets[1], targets[2]]
        snapped[index + 2] = [targets[2], targets[3]]
        triplet_groups += 1
        index += 3
    return [(start, end, event) for (start, end), (_, _, event) in zip(snapped, raw)], triplet_groups


def _tempo_points_for_mapper(mapper: _BeatMapper, quarter_ticks: int) -> list[tuple[int, float]]:
    """Return the shared score-tempo map used by Score and performance MIDI."""

    if mapper.fixed or len(mapper.beat_times) < 2:
        return [(0, float(mapper.bpm))]
    scale = float(mapper.beat_scale)
    beat_duration_quarters = float(mapper.beat_duration_quarters or 1.0)
    intervals = [right - left for left, right in zip(mapper.beat_times, mapper.beat_times[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("beat times must be strictly increasing")
    raw: list[tuple[int, float]] = []
    for index, interval in enumerate(intervals):
        bpm = 60.0 * scale * beat_duration_quarters / interval
        score_beat = (
            index * scale * beat_duration_quarters
            + mapper.shift_beats
            + mapper.timeline_offset_beats
        )
        raw.append((round(score_beat * quarter_ticks), bpm))
    raw_origin_position = (
        -(mapper.shift_beats + mapper.timeline_offset_beats)
        / (scale * beat_duration_quarters)
        if scale
        else 0.0
    )
    base_index = max(0, min(len(raw) - 1, math.floor(raw_origin_position)))
    points: dict[int, float] = {0: raw[base_index][1]}
    for tick, bpm in raw:
        if tick > 0:
            points[tick] = bpm
    compact: list[tuple[int, float]] = []
    for tick, bpm in sorted(points.items()):
        if compact and round(60_000_000.0 / bpm) == round(60_000_000.0 / compact[-1][1]):
            continue
        compact.append((tick, bpm))
    return compact


def _tempo_events(analysis: MusicAnalysis, mapper: _BeatMapper, quarter_ticks: int) -> list[TempoEvent]:
    return [
        TempoEvent(start_tick=tick, bpm=bpm)
        for tick, bpm in _tempo_points_for_mapper(mapper, quarter_ticks)
    ]


def quantize_events(
    events: Iterable[NoteEvent],
    analysis: MusicAnalysis,
    *,
    mode: str = "polyphonic",
    title: str = "Untitled",
    quarter_ticks: int = 12,
) -> Score:
    """Quantize events on a shared beat map and fill every gap with rests."""

    raw_events = list(events)
    if not raw_events:
        raise NoNotesError("NoNotes: the selected engine produced no usable note events")
    numerator, denominator = _time_signature_values(analysis.time_signature)
    bar_ticks = round(numerator * quarter_ticks * 4 / denominator)
    mapper = _build_beat_mapper(analysis, raw_events)
    selected = select_voice_events(raw_events, mode=mode)
    if not selected:
        raise NoNotesError("NoNotes: no candidate survived melody selection")
    grouped: dict[str, list[NoteEvent]] = {}
    for event in selected:
        grouped.setdefault(event.voice_id, []).append(event)

    total_beats = max(mapper.seconds_to_beat(analysis.duration_sec), 1 / quarter_ticks)
    last_event_beat = max((mapper.seconds_to_beat(event.end_sec) for event in selected), default=0.0)
    raw_total_ticks = max(total_beats, last_event_beat) * quarter_ticks
    quantization_tolerance = max(1.0, quarter_ticks / 12.0)
    bars = max(1, math.ceil((raw_total_ticks - quantization_tolerance) / bar_ticks))
    total_ticks = max(bar_ticks, bars * bar_ticks)
    score_voices: list[ScoreVoice] = []
    triplet_group_count = 0
    for voice_id, voice_events in sorted(grouped.items()):
        ordered_events = sorted(voice_events, key=lambda item: (item.start_sec, item.end_sec, item.midi))
        raw_boundaries = [
            (mapper.seconds_to_beat(event.start_sec) * quarter_ticks, mapper.seconds_to_beat(event.end_sec) * quarter_ticks, event)
            for event in ordered_events
        ]
        if any(start < -1e-6 or end < -1e-6 for start, end, _event in raw_boundaries):
            raise ValueError(
                "beat mapper produced a negative score coordinate; "
                "refusing to clip it silently"
            )
        snapped_events, group_count = _quantize_voice_boundaries(raw_boundaries, quarter_ticks)
        triplet_group_count += group_count
        cursor = 0
        score_events: list[ScoreNote] = []
        for start, end, event in snapped_events:
            start = max(cursor, min(total_ticks, start))
            if start > cursor:
                score_events.append(
                    ScoreNote(
                        start_tick=cursor,
                        duration_tick=start - cursor,
                        midi=None,
                        voice_id=voice_id,
                        source="silence",
                    )
                )
            minimum_duration = max(1, quarter_ticks // 4)
            end = min(total_ticks, max(end, start + minimum_duration))
            if end <= start:
                continue
            score_events.append(
                ScoreNote(
                    start_tick=start,
                    duration_tick=end - start,
                    midi=event.midi,
                    voice_id=voice_id,
                    confidence=event.confidence,
                    source=event.source,
                    velocity=event.velocity,
                    raw_pitch=event.raw_pitch,
                    stem_id=event.stem_id,
                    metadata=dict(event.metadata),
                )
            )
            cursor = end
        if cursor < total_ticks:
            score_events.append(
                ScoreNote(
                    start_tick=cursor,
                    duration_tick=total_ticks - cursor,
                    midi=None,
                    voice_id=voice_id,
                    source="silence",
                )
            )
        if not score_events:
            raise NoNotesError(f"NoNotes: voice {voice_id} has no printable events")
        stem_ids = {event.stem_id for event in ordered_events if event.stem_id}
        score_voices.append(
            ScoreVoice(
                voice_id=voice_id,
                events=score_events,
                label=voice_id,
                stem_id=next(iter(stem_ids)) if len(stem_ids) == 1 else None,
            )
        )

    tempo_events = _tempo_events(analysis, mapper, quarter_ticks)
    metadata = {
        "bar_ticks": bar_ticks,
        "beat_seconds": 60.0 / analysis.bpm,
        "analysis_bpm": analysis.bpm,
        "input_event_count": len(raw_events),
        "beat_source": analysis.metadata.get("beat_source", "beat_times" if len(analysis.beat_times) >= 2 else "fixed_bpm"),
        "beat_times": list(analysis.beat_times),
        "beat_shift_beats": mapper.shift_beats,
        "beat_scale": mapper.beat_scale,
        "beat_unit": mapper.beat_unit,
        "beat_unit_source": mapper.beat_unit_source,
        "beat_unit_proven": mapper.beat_unit_proven,
        "beat_duration_quarters": mapper.beat_duration_quarters,
        "beats_per_bar": mapper.beats_per_bar,
        "bar_duration_quarters": mapper.bar_duration_quarters,
        "beat_grid": analysis.metadata.get("beat_grid"),
        "beat_offset_sec": analysis.beat_times[0] if analysis.beat_times else 0.0,
        "score_origin": mapper.score_origin,
        "score_timeline_offset_beats": mapper.timeline_offset_beats,
        # ``pickup_beats`` is reserved for an explicitly confirmed pickup.
        # An undetermined pre-downbeat span is recorded separately so it can
        # never be mistaken for a printable pickup header.
        "pickup_beats": float(mapper.score_origin.get("pickup_beats", 0.0)),
        "pickup_span_quarters": float(mapper.score_origin.get("pickup_span_quarters", 0.0)),
        "downbeat_status": mapper.score_origin.get("downbeat_status", "undetermined"),
        "downbeat_warning": mapper.score_origin.get("warning"),
        "downbeat_warnings": (
            [str(mapper.score_origin["warning"])]
            if mapper.score_origin.get("warning")
            else []
        ),
        "triplet_group_count": triplet_group_count,
        "source_stems": sorted({event.stem_id for event in selected if event.stem_id}),
    }
    return Score(
        title=title,
        bpm=tempo_events[0].bpm,
        key=analysis.key,
        time_signature=analysis.time_signature,
        quarter_ticks=quarter_ticks,
        total_ticks=total_ticks,
        voices=score_voices,
        tempo_events=tempo_events,
        source=selected[0].source,
        warnings=list(
            dict.fromkeys(
                [
                    *analysis.warnings,
                    *(
                        [str(mapper.score_origin["warning"])]
                        if mapper.score_origin.get("warning")
                        else []
                    ),
                ]
            )
        ),
        metadata=metadata,
    )


NATURAL_SCALE = (0, 2, 4, 5, 7, 9, 11)
KEY_TO_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "Eb": 3, "D#": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "Ab": 8, "G#": 8, "A": 9, "Bb": 10, "A#": 10, "B": 11}


def _key_pitch(key: str) -> tuple[int, bool]:
    text = normalize_key(key)
    minor = text.endswith("m")
    root = relative_major_key(text) if minor else text
    return KEY_TO_PC[root], minor


def _jianpu_tonic_midi(key: str) -> int:
    """Match jianpu-ly's octave choice for a ``1=<major key>`` header."""

    root_pc, _minor = _key_pitch(key)
    tonic = 60 + root_pc
    return tonic - 12 if tonic > 66 else tonic


def midi_to_jianpu(midi: int, key: str) -> str:
    _root_pc, _minor = _key_pitch(key)
    # Number minor keys from their relative major so the tonic is degree 6
    # (for example Am -> C, F#m -> A, Cm -> Eb), with octave selection left
    # to the same nearest-pitch search used for major keys.
    root_midi = _jianpu_tonic_midi(key)
    candidates: list[tuple[int, int, int, int]] = []
    degree_intervals = {degree: offset for degree, offset in enumerate(NATURAL_SCALE, start=1)}
    for degree, offset in degree_intervals.items():
        # Include an adjacent octave when the tonic is near MIDI 0 or 127;
        # every chromatic pitch can then use the nearest diatonic degree with
        # at most one accidental while retaining the original MIDI number.
        for octave in range(-12, 13):
            nominal = root_midi + offset + 12 * octave
            candidates.append((abs(midi - nominal), midi - nominal, degree, octave))
    _, difference, degree, octave = min(
        candidates,
        key=lambda item: (item[0], abs(item[1]), abs(item[3]), item[2], item[3]),
    )
    accidental = "#" if difference > 0 else "b" if difference < 0 else ""
    octave_mark = "'" * octave if octave > 0 else "," * (-octave) if octave < 0 else ""
    return f"{accidental}{degree}{octave_mark}"


@dataclass(frozen=True)
class _Slice:
    start_tick: int
    duration_tick: int
    midi: int | None
    source_event_id: str | None = None
    chord_pitches: tuple[int, ...] = ()
    tie_before: frozenset[int] = frozenset()
    tie_after: frozenset[int] = frozenset()
    tuplet_actual: int | None = None
    tuplet_normal: int | None = None
    tuplet_type: str | None = None
    # Standard MusicXML tuplets use the pinned 3:2 policy or an explicitly
    # bounded 4:3 group.  The importer may additionally mark a bounded
    # fine-grid fragment for renderer-specific exact tuplets (for example
    # 3:1).  Keeping this bit on the slice, rather than widening all Score
    # JSON tuplets, prevents an arbitrary ratio from entering the production
    # serializer silently.
    fine_grid_tuplet: bool = False
    # A bounded 1/2/4/5-tick event can be represented as one explicit 3:1
    # bracket whose body is one or more ordinary atoms.  This marker closes
    # that bracket around the single Score event without treating a nearby
    # event as an inferred tuplet member.
    fine_grid_tuplet_single: bool = False
    dots: int = 0

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_tick

    @property
    def pitches(self) -> tuple[int, ...]:
        if self.chord_pitches:
            return self.chord_pitches
        return () if self.midi is None else (self.midi,)


@dataclass(frozen=True)
class _MeasureSpan:
    start_tick: int
    end_tick: int
    number: int | None = None
    time_signature: str | None = None
    is_pickup: bool = False

    @property
    def duration_tick(self) -> int:
        return self.end_tick - self.start_tick


@dataclass(frozen=True)
class _MeasureContext:
    """Notation state effective at the beginning of one measure."""

    time_signature: str
    key: str
    tempo_bpm: int


def _boundary_events(
    score: Score,
    spans: list[_MeasureSpan],
    metadata_key: str,
    value_key: str,
    normalize: Any,
) -> dict[int, Any]:
    """Validate and normalize metadata events that must start on a barline."""

    values = score.metadata.get(metadata_key, [])
    if values is None:
        return {}
    if not isinstance(values, list):
        raise JianpuSerializationError(f"metadata.{metadata_key} must be a list")
    boundaries = {span.start_tick for span in spans}
    result: dict[int, Any] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise JianpuSerializationError(f"metadata.{metadata_key}[{index}] must be an object")
        try:
            start_tick = int(value["start_tick"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JianpuSerializationError(
                f"metadata.{metadata_key}[{index}] must contain an integer start_tick"
            ) from exc
        if start_tick not in boundaries:
            raise JianpuSerializationError(
                f"{metadata_key} event at tick {start_tick} is not a measure boundary"
            )
        if value_key not in value:
            raise JianpuSerializationError(f"metadata.{metadata_key}[{index}] is missing {value_key}")
        try:
            normalized = normalize(value[value_key])
        except (TypeError, ValueError) as exc:
            raise JianpuSerializationError(
                f"metadata.{metadata_key}[{index}] has an invalid {value_key}"
            ) from exc
        previous = result.get(start_tick)
        if previous is not None and previous != normalized:
            raise JianpuSerializationError(
                f"conflicting {metadata_key} events at measure boundary {start_tick}"
            )
        result[start_tick] = normalized
    return result


def _rounded_tempo(bpm: float) -> int:
    """Use conventional half-up rounding for the integer jianpu command."""

    return max(1, math.floor(float(bpm) + 0.5))


def _pickup_duration_token(duration_tick: int, quarter_ticks: int) -> str:
    """Return the jianpu-ly comma suffix for an exact pickup duration."""

    whole_note_ticks = 4 * quarter_ticks
    target = Fraction(duration_tick, whole_note_ticks)
    for denominator in (1, 2, 4, 8, 16, 32, 64):
        base = Fraction(1, denominator)
        for dots in range(2):
            value = base * Fraction(2 ** (dots + 1) - 1, 2**dots)
            if target == value:
                return f"{denominator}{'.' * dots}"
    raise JianpuSerializationError(
        f"pickup duration {duration_tick} ticks cannot be represented by jianpu-ly "
        "as a power-of-two or dotted power-of-two value"
    )


def _measure_contexts(score: Score, spans: list[_MeasureSpan]) -> tuple[list[_MeasureContext], str]:
    """Build per-bar meter/key/tempo state and the initial meter header."""

    if not spans:
        raise JianpuSerializationError("score contains no measures")

    timeline_meter = spans[0].time_signature
    active_meter = normalize_time_signature(timeline_meter or score.time_signature)
    time_events = _boundary_events(
        score,
        spans,
        "time_signature_events",
        "time_signature",
        lambda value: normalize_time_signature(str(value)),
    )
    active_key = normalize_key(score.key)
    key_events = _boundary_events(
        score,
        spans,
        "key_signature_events",
        "key",
        lambda value: normalize_key(str(value)),
    )

    pickup_value = score.metadata.get("pickup")
    if pickup_value is not None and not isinstance(pickup_value, Mapping):
        raise JianpuSerializationError("metadata.pickup must be an object")
    pickup_flag = bool(spans[0].is_pickup)
    pickup_duration = spans[0].duration_tick
    if isinstance(pickup_value, Mapping):
        pickup_flag = pickup_flag or bool(pickup_value.get("is_pickup", False))
        declared_duration = pickup_value.get("duration_tick")
        if declared_duration not in (None, 0):
            try:
                declared_duration = int(declared_duration)
            except (TypeError, ValueError) as exc:
                raise JianpuSerializationError("metadata.pickup.duration_tick must be an integer") from exc
            if declared_duration != pickup_duration:
                raise JianpuSerializationError(
                    f"pickup duration metadata {declared_duration} does not match first measure {pickup_duration}"
                )
    initial_meter = time_events.get(0, active_meter)
    initial_numerator, initial_denominator = _time_signature_values(initial_meter)
    initial_bar_ticks = round(initial_numerator * score.quarter_ticks * 4 / initial_denominator)
    pickup_suffix: str | None = None
    if pickup_flag:
        if "timeline_measures" not in score.metadata:
            raise JianpuSerializationError("pickup metadata requires timeline_measures")
        if pickup_duration >= initial_bar_ticks:
            raise JianpuSerializationError(
                f"pickup first measure duration {pickup_duration} must be shorter than {initial_bar_ticks} ticks"
            )
        pickup_suffix = _pickup_duration_token(pickup_duration, score.quarter_ticks)
        # jianpu-ly requires the final bar to make up the first anacrusis.  A
        # changed meter before the final bar has its own full-bar semantics;
        # in that case the vendor remains the final authority.  For the
        # common unchanged-meter case, reject a malformed complement early.
        if len(spans) > 1:
            final_meter = spans[-1].time_signature or initial_meter
            if final_meter == initial_meter:
                expected_final = initial_bar_ticks - pickup_duration
                if spans[-1].duration_tick != expected_final:
                    raise JianpuSerializationError(
                        f"final pickup bar has {spans[-1].duration_tick} ticks; expected {expected_final}"
                    )

    contexts: list[_MeasureContext] = []
    tempo_index = 0
    # Direct scores display the overall tempo. BeatNet's local timing still
    # goes into the playback MIDI; printing its jitter each bar obscures notes.
    tempo_events = [] if score.metadata.get("notation_engine") == "direct-jianpu" else score.tempo_events
    current_tempo = float(score.bpm)
    previous_meter: str | None = None
    for index, span in enumerate(spans):
        if span.start_tick in time_events:
            event_meter = time_events[span.start_tick]
            if span.time_signature and span.time_signature != event_meter:
                raise JianpuSerializationError(
                    f"timeline measure {index} meter {span.time_signature} conflicts with "
                    f"time_signature_events at tick {span.start_tick}"
                )
            active_meter = event_meter
        elif span.time_signature:
            # A timeline from MusicXML carries the meter on each measure even
            # when the worker did not emit a separate event for an unchanged
            # bar.  A changed value is therefore an implicit boundary event.
            active_meter = span.time_signature
        if index == 0 and span.time_signature:
            active_meter = span.time_signature
            if 0 in time_events and time_events[0] != active_meter:
                raise JianpuSerializationError("initial timeline meter conflicts with time_signature_events")
        if previous_meter is not None and span.time_signature and span.time_signature != previous_meter:
            # The change is safe because the timeline itself places it at this
            # span's start.  Mid-measure events were rejected above.
            active_meter = span.time_signature
        previous_meter = active_meter
        while tempo_index < len(tempo_events) and tempo_events[tempo_index].start_tick <= span.start_tick:
            current_tempo = tempo_events[tempo_index].bpm
            tempo_index += 1
        contexts.append(
            _MeasureContext(
                time_signature=active_meter,
                key=key_events.get(span.start_tick, active_key),
                tempo_bpm=_rounded_tempo(current_tempo),
            )
        )
        active_key = contexts[-1].key

    meter_header = contexts[0].time_signature if pickup_suffix is None else f"{contexts[0].time_signature},{pickup_suffix}"
    return contexts, meter_header


def _measure_prefix(
    index: int,
    contexts: list[_MeasureContext],
    *,
    include_initial: bool = False,
) -> list[str]:
    """Return visible commands at a measure boundary."""

    context = contexts[index]
    if index == 0:
        if not include_initial:
            return []
        return [_key_command(context.key), f"4={context.tempo_bpm}", context.time_signature]
    previous = contexts[index - 1]
    output: list[str] = []
    if context.time_signature != previous.time_signature:
        output.append(context.time_signature)
    if context.key != previous.key:
        output.append(_key_command(context.key))
    if context.tempo_bpm != previous.tempo_bpm:
        output.append(f"4={context.tempo_bpm}")
    return output


def _duration_options(quarter_ticks: int, note: str) -> list[tuple[int, list[str]]]:
    """Return exact jianpu-ly duration atoms, longest first.

    ``q/s/d/h`` are eighth through sixty-fourth notes, dots augment those
    values, and dashes extend minims and longer values.  All calculations use
    exact fractions before converting to the Score tick grid.
    """

    options: list[tuple[int, list[str]]] = []
    prefixes = {4: "", 8: "q", 16: "s", 32: "d", 64: "h"}
    for denominator in (1, 2, 4, 8, 16, 32, 64):
        base = Fraction(quarter_ticks * 4, denominator)
        for dots in range(4):
            duration = base * Fraction(2 ** (dots + 1) - 1, 2**dots)
            if duration.denominator != 1:
                continue
            ticks = int(duration)
            if denominator <= 2:
                # Jianpu-ly documents dashes for minims and semibreves.  A
                # dotted minim is ``1 - -`` and a whole note is ``1 - - -``.
                dash_count = ticks // quarter_ticks - 1
                if dash_count < 0 or ticks % quarter_ticks:
                    continue
                tokens = [note, *(["-"] * dash_count)]
            else:
                tokens = [f"{prefixes[denominator]}{note}{'.' * dots}"]
            options.append((ticks, tokens))
    unique: dict[int, list[str]] = {}
    for ticks, tokens in sorted(options, key=lambda value: (value[0], len(value[1])), reverse=True):
        unique.setdefault(ticks, tokens)
    return sorted(unique.items(), key=lambda value: value[0], reverse=True)


def _duration_chunks(duration_tick: int, quarter_ticks: int) -> list[int]:
    if duration_tick <= 0:
        return []
    remaining = duration_tick
    chunks: list[int] = []
    allowed = [ticks for ticks, _tokens in _duration_options(quarter_ticks, "1")]
    while remaining:
        choice = next((value for value in allowed if value <= remaining), None)
        if choice is None:
            raise JianpuSerializationError(
                f"duration {duration_tick} ticks cannot be represented by jianpu-ly "
                f"with quarter_ticks={quarter_ticks}"
            )
        chunks.append(choice)
        remaining -= choice
    return chunks


def _format_duration(
    note: str,
    duration_tick: int,
    quarter_ticks: int,
    *,
    dots: int = 0,
) -> list[str]:
    """Format one note/rest duration, inserting ties between split atoms."""

    if duration_tick <= 0:
        return []
    if dots:
        if dots > 3:
            raise JianpuSerializationError(f"jianpu-ly supports at most three dots, got {dots}")
        prefixes = {4: "", 8: "q", 16: "s", 32: "d", 64: "h"}
        for denominator in (1, 2, 4, 8, 16, 32, 64):
            base = Fraction(quarter_ticks * 4, denominator)
            duration = base * Fraction(2 ** (dots + 1) - 1, 2**dots)
            if duration.denominator != 1 or int(duration) != duration_tick:
                continue
            if denominator <= 2:
                if duration_tick % quarter_ticks:
                    continue
                dash_count = duration_tick // quarter_ticks - 1
                return [note, *("-" for _ in range(dash_count))]
            return [f"{prefixes[denominator]}{note}{'.' * dots}"]
        raise JianpuSerializationError(
            f"explicit dots={dots} do not match duration {duration_tick} ticks "
            f"with quarter_ticks={quarter_ticks}"
        )
    options = _duration_options(quarter_ticks, note)
    groups: list[list[str]] = []
    remaining = duration_tick
    while remaining:
        choice = next(((ticks, tokens) for ticks, tokens in options if ticks <= remaining), None)
        if choice is None:
            raise JianpuSerializationError(
                f"duration {duration_tick} ticks cannot be represented by jianpu-ly "
                f"with quarter_ticks={quarter_ticks}"
            )
        ticks, tokens = choice
        if note == "0" and len(tokens) > 1:
            # A dash after a rest is stateful in jianpu-ly.  Repeating rests is
            # unambiguous and retains the same duration without a fake tie.
            groups.append(["0"] * len(tokens))
        else:
            groups.append(list(tokens))
        remaining -= ticks
    result: list[str] = []
    for index, group in enumerate(groups):
        if index and note != "0":
            result.append("~")
        result.extend(group)
    return result


def _event_pitches(event: ScoreNote) -> tuple[int, ...]:
    if event.midi is None:
        if event.chord_pitches:
            raise JianpuSerializationError("a rest cannot carry chord_pitches")
        return ()
    pitches = tuple(event.chord_pitches) if event.chord_pitches else (event.midi,)
    if event.midi not in pitches:
        raise JianpuSerializationError(
            f"ScoreNote at {event.start_tick} has midi={event.midi} outside chord_pitches={list(pitches)}"
        )
    return pitches


def _event_tie_sets(event: ScoreNote) -> tuple[frozenset[int], frozenset[int]]:
    pitches = _event_pitches(event)
    if not pitches:
        if event.tie or event.tie_types:
            raise JianpuSerializationError(f"rest at {event.start_tick} cannot carry a tie")
        return frozenset(), frozenset()
    if event.tie_types:
        if len(event.tie_types) != len(pitches):
            raise JianpuSerializationError(
                f"tie_types length {len(event.tie_types)} does not match {len(pitches)} pitches at {event.start_tick}"
            )
        before = frozenset(pitch for pitch, tie in zip(pitches, event.tie_types) if tie in {"stop", "continue"})
        after = frozenset(pitch for pitch, tie in zip(pitches, event.tie_types) if tie in {"start", "continue"})
        return before, after
    if event.tie == "stop":
        return frozenset(pitches), frozenset()
    if event.tie == "start":
        return frozenset(), frozenset(pitches)
    if event.tie == "continue":
        return frozenset(pitches), frozenset(pitches)
    return frozenset(), frozenset()


def _event_tie_map(event: ScoreNote) -> dict[int, str | None]:
    """Return tie metadata keyed by MIDI pitch without losing None slots."""

    pitches = _event_pitches(event)
    if not pitches:
        return {}
    if event.tie_types:
        if len(event.tie_types) != len(pitches):
            raise JianpuSerializationError(
                f"tie_types length {len(event.tie_types)} does not match {len(pitches)} pitches at {event.start_tick}"
            )
        return dict(zip(pitches, event.tie_types))
    return {pitch: event.tie for pitch in pitches}


def _partial_tie_pitches(voice: ScoreVoice) -> set[int]:
    """Find chord pitches whose tie cannot be represented by one chord tie."""

    special: set[int] = set()
    for event in voice.events:
        special.update(_partial_tie_pitches_for_event(event))
    return special


def _copy_event_for_pitches(event: ScoreNote, selected: list[int]) -> ScoreNote:
    """Make a lane event while retaining the original timeline and notation."""

    if not selected:
        return event.model_copy(
            update={
                "midi": None,
                "chord_pitches": [],
                "tie": None,
                "tie_types": [],
            }
        )
    tie_map = _event_tie_map(event)
    tie_types = [tie_map.get(pitch) for pitch in selected]
    present = [value for value in tie_types if value is not None]
    tie = present[0] if len(present) == len(tie_types) and present and all(value == present[0] for value in present) else None
    return event.model_copy(
        update={
            "midi": min(selected),
            "chord_pitches": list(selected),
            "tie": tie,
            "tie_types": tie_types,
        }
    )


def _chord_split_records(
    voices: list[ScoreVoice],
    keys: Iterable[str],
) -> list[dict[str, Any]]:
    """Find simple-chord tokens that jianpu-ly cannot spell losslessly.

    The vendor's simple-chord grammar has one accidental state for a token.
    An accidental on a later figure can therefore be applied to an earlier
    figure (the B-flat-minor ``6,,#1,3,`` case is a concrete example).  Such a
    chord is rendered with one lane per pitch by ``_serialization_voices``.
    Those lanes are local to the tie-connected component and are reused by
    later unsafe chords; other events remain in the base voice or become rests
    in a supplemental lane.
    """

    normalized_keys = tuple(dict.fromkeys(normalize_key(key) for key in keys))
    records: list[dict[str, Any]] = []
    for voice in voices:
        for event_index, event in enumerate(voice.events):
            pitches = _event_pitches(event)
            if len(pitches) <= 1:
                continue
            for key in normalized_keys:
                parts = [midi_to_jianpu(pitch, key) for pitch in sorted(pitches)]
                accidental_positions = [
                    index
                    for index, part in enumerate(parts)
                    if "#" in part or "b" in part
                ]
                if not any(index > 0 for index in accidental_positions):
                    continue
                records.append(
                    {
                        "voice_id": voice.voice_id,
                        "event_index": event_index,
                        "start_tick": event.start_tick,
                        "duration_tick": event.duration_tick,
                        "pitches": sorted(pitches),
                        "key": key,
                        "token": "".join(parts),
                        "reason": "non_leading_accidental_in_simple_chord",
                    }
                )
                break
    return records


def _partial_tie_indices(voice: ScoreVoice) -> set[int]:
    """Return events whose chord tie needs independent pitch lanes."""

    return {
        index
        for index, event in enumerate(voice.events)
        if _partial_tie_pitches_for_event(event)
    }


def _partial_tie_pitches_for_event(event: ScoreNote) -> set[int]:
    pitches = _event_pitches(event)
    if len(pitches) <= 1:
        return set()
    before, after = _event_tie_sets(event)
    all_pitches = frozenset(pitches)
    special: set[int] = set()
    if before and before != all_pitches:
        special.update(before)
    if after and after != all_pitches:
        special.update(after)
    return special


def _chord_lane_plan(
    voice: ScoreVoice,
    keys: Iterable[str],
) -> tuple[dict[int, dict[int, int]], set[int], int, list[dict[str, Any]]]:
    """Build minimal pitch-to-lane assignments for one source voice.

    A plan is made per tie-connected component containing an unsafe chord (or
    a partial chord tie).  Lane numbers are local slots: slot zero is the
    original voice, and the remaining slots are reusable supplemental voices.
    This keeps ordinary melodic events in the base voice and prevents one
    isolated chord from turning every pitch in a long part into its own lane.
    """

    unsafe_records = _chord_split_records([voice], keys)
    unsafe_indices = {int(record["event_index"]) for record in unsafe_records}
    partial_tie_indices = _partial_tie_indices(voice)
    if not unsafe_indices and not partial_tie_indices:
        return {}, set(), 0, unsafe_records
    seed_indices = unsafe_indices | partial_tie_indices

    events = voice.events
    tie_sets = [_event_tie_sets(event) for event in events]
    edges: dict[int, set[int]] = {index: set() for index in range(len(events))}
    for index in range(len(events) - 1):
        if tie_sets[index][1] & tie_sets[index + 1][0]:
            edges[index].add(index + 1)
            edges[index + 1].add(index)

    components: list[set[int]] = []
    unvisited = set(seed_indices)
    while unvisited:
        root = min(unvisited)
        unvisited.remove(root)
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in edges[current]:
                if neighbor in component:
                    continue
                component.add(neighbor)
                unvisited.discard(neighbor)
                stack.append(neighbor)
        components.append(component)

    assignments: dict[int, dict[int, int]] = {}
    split_indices: set[int] = set()
    component_lane_count: dict[int, int] = {}
    for component in components:
        max_size = max(len(_event_pitches(events[index])) for index in component)
        split_indices.update(component)
        active: dict[int, int] = {}
        for index in sorted(component):
            pitches = sorted(_event_pitches(events[index]))
            before, after = tie_sets[index]
            assignment: dict[int, int] = {}
            used: set[int] = set()
            for pitch in pitches:
                if pitch in before:
                    if pitch not in active:
                        raise JianpuSerializationError(
                            f"cannot assign tied chord pitch {pitch} at tick {events[index].start_tick}"
                        )
                    assignment[pitch] = active[pitch]
                    used.add(active[pitch])

            # Pitches with the same tie state can share a chord token.  A
            # partial-tie-only component therefore needs one lane for the held
            # pitch and one for the untied chord pitches, instead of one lane
            # for every pitch.  Unsafe accidental chords remain one pitch per
            # lane because jianpu-ly has one accidental state per token.
            remaining = [pitch for pitch in pitches if pitch not in assignment]
            if index in unsafe_indices:
                groups = [[pitch] for pitch in remaining]
            else:
                by_tie: dict[str | None, list[int]] = {}
                tie_map = _event_tie_map(events[index])
                for pitch in remaining:
                    by_tie.setdefault(tie_map.get(pitch), []).append(pitch)
                groups = [by_tie[key] for key in sorted(by_tie, key=lambda value: (value is not None, str(value)))]
            for group in groups:
                slot = next((candidate for candidate in range(max_size) if candidate not in used), None)
                if slot is None:
                    raise JianpuSerializationError(
                        f"chord at tick {events[index].start_tick} needs more than {max_size} lanes"
                    )
                for pitch in group:
                    assignment[pitch] = slot
                used.add(slot)
            assignments[index] = assignment
            active = {pitch: assignment[pitch] for pitch in after if pitch in assignment}
        lane_count = max(
            (max(assignment.values(), default=-1) + 1 for index, assignment in assignments.items() if index in component),
            default=1,
        )
        component_lane_count.update({index: lane_count for index in component})

    for record in unsafe_records:
        event_index = int(record["event_index"])
        assignment = assignments.get(event_index, {})
        record["lane_assignments"] = [
            {"pitch": pitch, "lane": assignment[pitch]}
            for pitch in sorted(assignment)
        ]
        record["component_lane_count"] = component_lane_count.get(event_index, 1)
    maximum_lane_count = max(component_lane_count.values(), default=1)
    return assignments, split_indices, maximum_lane_count, unsafe_records


def _serialization_voices(
    voices: list[ScoreVoice],
    *,
    keys: Iterable[str] | None = None,
) -> list[ScoreVoice]:
    """Split unsafe chord ties/accidentals into safe jianpu-ly parts.

    jianpu-ly's ``~`` applies to every note in a chord.  A MusicXML chord
    whose tie applies to only some pitches therefore needs independent parts
    for those pitches; emitting one chord-level tie would incorrectly tie the
    untied notes as well.  Its simple-chord accidental state has a similar
    limitation for a non-leading accidental, so those chords use one lane per
    pitch as well.
    """

    result: list[ScoreVoice] = []
    normalized_keys = tuple(dict.fromkeys(normalize_key(key) for key in (keys or ())))
    for voice in voices:
        assignments, split_indices, lane_count, _unsafe_records = _chord_lane_plan(voice, normalized_keys)
        if split_indices:
            label = voice.label or voice.voice_id
            lane_events: list[list[ScoreNote]] = [[] for _ in range(lane_count)]
            used_lanes: set[int] = set()
            for index, event in enumerate(voice.events):
                assignment = assignments.get(index)
                if assignment is None:
                    lane_events[0].append(event)
                    for lane in range(1, lane_count):
                        lane_events[lane].append(_copy_event_for_pitches(event, []))
                    continue
                by_lane: dict[int, list[int]] = {}
                for pitch, lane in assignment.items():
                    by_lane.setdefault(lane, []).append(pitch)
                    used_lanes.add(lane)
                for lane in range(lane_count):
                    lane_events[lane].append(
                        _copy_event_for_pitches(event, sorted(by_lane.get(lane, [])))
                    )
            result.append(voice.model_copy(update={"events": lane_events[0]}))
            for lane in range(1, lane_count):
                if lane not in used_lanes:
                    continue
                result.append(
                    voice.model_copy(
                        update={
                            "voice_id": f"{voice.voice_id}:chord-lane-{lane}",
                            "label": f"{label} chord lane {lane}",
                            "events": lane_events[lane],
                        }
                    )
                )
            continue
        special = sorted(_partial_tie_pitches(voice))
        if not special:
            result.append(voice)
            continue
        base_events: list[ScoreNote] = []
        for event in voice.events:
            pitches = list(_event_pitches(event))
            base_events.append(_copy_event_for_pitches(event, [pitch for pitch in pitches if pitch not in special]))
        base_has_notes = any(event.midi is not None for event in base_events)
        label = voice.label or voice.voice_id
        if base_has_notes:
            result.append(
                voice.model_copy(
                    update={
                        "voice_id": f"{voice.voice_id}:untied",
                        "label": f"{label} untied pitches",
                        "events": base_events,
                    }
                )
            )
        for pitch in special:
            lane_events = []
            for event in voice.events:
                pitches = list(_event_pitches(event))
                lane_events.append(_copy_event_for_pitches(event, [pitch] if pitch in pitches else []))
            result.append(
                voice.model_copy(
                    update={
                        "voice_id": f"{voice.voice_id}:tie-{pitch}",
                        "label": f"{label} tie pitch {pitch}",
                        "events": lane_events,
                    }
                )
            )
    return result


def _validate_explicit_ties(voice: ScoreVoice) -> None:
    events = voice.events
    tie_sets = [_event_tie_sets(event) for event in events]
    for index, (event, (before, after)) in enumerate(zip(events, tie_sets)):
        if not before and not after:
            continue
        previous = events[index - 1] if index else None
        following = events[index + 1] if index + 1 < len(events) else None
        if before:
            previous_after = tie_sets[index - 1][1] if previous is not None else frozenset()
            if previous is None or previous.end_tick != event.start_tick or not before <= previous_after:
                raise JianpuSerializationError(
                    f"dangling tie into voice {voice.voice_id} at tick {event.start_tick}: pitches={sorted(before)}"
                )
        if after:
            following_before = tie_sets[index + 1][0] if following is not None else frozenset()
            if following is None or following.start_tick != event.end_tick or not after <= following_before:
                raise JianpuSerializationError(
                    f"dangling tie out of voice {voice.voice_id} at tick {event.end_tick}: pitches={sorted(after)}"
                )


def _fixed_measure_spans(score: Score) -> list[_MeasureSpan]:
    """Return the score's authoritative measure timeline.

    MusicXML normalization stores the real measure boundaries in metadata.  A
    fixed grid is still used for old Score JSON, but once a timeline is
    present it must be consumed verbatim: silently rebuilding it from the
    initial meter would lose pickup bars and meter changes.
    """

    if "timeline_measures" in score.metadata:
        values = score.metadata.get("timeline_measures")
        if not isinstance(values, list) or not values:
            raise JianpuSerializationError("metadata.timeline_measures must be a non-empty list")
        spans: list[_MeasureSpan] = []
        previous_end = 0
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise JianpuSerializationError(f"timeline measure {index} must be an object")
            try:
                start = int(value["start_tick"])
                duration = int(value["duration_tick"])
                end = int(value["end_tick"])
            except (KeyError, TypeError, ValueError) as exc:
                raise JianpuSerializationError(
                    f"timeline measure {index} must contain integer start_tick, duration_tick, and end_tick"
                ) from exc
            if start < 0 or duration <= 0 or end != start + duration:
                raise JianpuSerializationError(
                    f"invalid timeline measure {index}: start={start}, duration={duration}, end={end}"
                )
            if index == 0 and start != 0:
                raise JianpuSerializationError(f"measure timeline starts at {start}, expected score tick 0")
            if index and start != previous_end:
                relation = "overlap" if start < previous_end else "gap"
                raise JianpuSerializationError(
                    f"measure timeline has an illegal {relation} between ticks {previous_end} and {start}"
                )
            time_signature: str | None = None
            if value.get("time_signature") is not None:
                try:
                    time_signature = normalize_time_signature(str(value["time_signature"]))
                except ValueError as exc:
                    raise JianpuSerializationError(
                        f"timeline measure {index} has an unsupported time signature"
                    ) from exc
            number = value.get("number")
            if number is not None:
                try:
                    number = int(number)
                except (TypeError, ValueError) as exc:
                    raise JianpuSerializationError(f"timeline measure {index} has an invalid number") from exc
            spans.append(
                _MeasureSpan(
                    start,
                    end,
                    number=number,
                    time_signature=time_signature,
                    is_pickup=bool(value.get("is_pickup", False)),
                )
            )
            previous_end = end
        if previous_end != score.total_ticks:
            raise JianpuSerializationError(
                f"measure timeline ends at {previous_end}, expected Score total_ticks {score.total_ticks}"
            )
        return spans

    numerator, denominator = _time_signature_values(score.time_signature)
    bar_ticks = round(numerator * score.quarter_ticks * 4 / denominator)
    if bar_ticks <= 0:
        raise JianpuSerializationError("time signature produces an empty measure")
    spans: list[_MeasureSpan] = []
    cursor = 0
    while cursor < score.total_ticks:
        spans.append(_MeasureSpan(cursor, min(score.total_ticks, cursor + bar_ticks), time_signature=score.time_signature))
        cursor += bar_ticks
    return spans


def _slice_voice_events(voice: ScoreVoice, spans: list[_MeasureSpan]) -> list[list[_Slice]]:
    """Split one ScoreVoice at fixed Score.time_signature bar boundaries."""

    _validate_explicit_ties(voice)
    result: list[list[_Slice]] = [[] for _span in spans]
    for event in voice.events:
        pitches = _event_pitches(event)
        tie_before, tie_after = _event_tie_sets(event)
        segment_count = 0
        last_end = event.start_tick
        for span_index, span in enumerate(spans):
            if event.end_tick <= span.start_tick or event.start_tick >= span.end_tick:
                continue
            start = max(event.start_tick, span.start_tick)
            end = min(event.end_tick, span.end_tick)
            if start >= end:
                continue
            segment_count += 1
            is_first = start == event.start_tick
            is_last = end == event.end_tick
            result[span_index].append(
                _Slice(
                    start_tick=start,
                    duration_tick=end - start,
                    midi=event.midi,
                    source_event_id=(
                        str(event.metadata.get("musicxml_event_id"))
                        if event.metadata.get("musicxml_event_id") is not None
                        else None
                    ),
                    chord_pitches=pitches if event.midi is not None else (),
                    tie_before=frozenset(pitches if not is_first and pitches else tie_before),
                    tie_after=frozenset(pitches if not is_last and pitches else tie_after),
                    tuplet_actual=event.tuplet_actual,
                    tuplet_normal=event.tuplet_normal,
                    # A boundary belongs to the first/last slice when a
                    # MusicXML event itself crosses a barline.  Keeping it
                    # on the wrong fragment would make a legal cross-bar
                    # group appear unclosed to the serializer.
                    tuplet_type=(
                        event.tuplet_type
                        if (
                            (event.tuplet_type == "start" and is_first)
                            or (event.tuplet_type == "stop" and is_last)
                            or event.tuplet_type == "continue"
                        )
                        else None
                    ),
                    fine_grid_tuplet=bool(event.metadata.get("fine_grid_tuplet", False)),
                    fine_grid_tuplet_single=bool(event.metadata.get("fine_grid_tuplet_single", False)),
                    dots=event.dots if is_first and is_last else 0,
                )
            )
            last_end = end
        if segment_count == 0 or last_end != event.end_tick:
            raise JianpuSerializationError(
                f"ScoreVoice {voice.voice_id} event {event.start_tick}:{event.end_tick} lies outside score"
            )
    for span_index, (span, bar) in enumerate(zip(spans, result)):
        bar.sort(key=lambda item: (item.start_tick, item.end_tick))
        cursor = span.start_tick
        for item in bar:
            if item.start_tick != cursor:
                raise JianpuSerializationError(
                    f"voice {voice.voice_id} has a measure gap or overlap at tick {cursor}"
                )
            cursor = item.end_tick
        if cursor != span.end_tick:
            raise JianpuSerializationError(
                f"voice {voice.voice_id} measure {span_index + 1} ends at {cursor}, expected {span.end_tick}"
            )
    return result


def _pitch_token(pitches: tuple[int, ...], key: str) -> str:
    if not pitches:
        return "0"
    return "".join(midi_to_jianpu(pitch, key) for pitch in sorted(pitches))


def _slice_tokens(
    item: _Slice,
    key: str,
    quarter_ticks: int,
    *,
    duration_tick: int | None = None,
    dots: int = 0,
    include_tie: bool = True,
) -> list[str]:
    note = _pitch_token(item.pitches, key)
    tokens = _format_duration(note, duration_tick or item.duration_tick, quarter_ticks, dots=dots)
    if include_tie and item.tie_after and item.pitches:
        tokens.append("~")
    return tokens


def _slice_context(
    slices: list[_Slice],
    index: int,
    span: _MeasureSpan,
    *,
    voice_id: str | None,
) -> str:
    """Describe one failing slice without changing its notation semantics.

    A one or two tick atom can be an exact imported event, but jianpu-ly has
    no ordinary 48-TPQ token for it.  The failure must retain enough local
    provenance to distinguish that case from a malformed tuplet or tie; in
    particular, adjacent events are evidence only, never an implicit tuplet.
    """

    def tuplet_text(item: _Slice) -> str:
        if item.tuplet_actual is None and item.tuplet_normal is None:
            return "none"
        return f"{item.tuplet_actual}:{item.tuplet_normal}/{item.tuplet_type or 'continue'}"

    def describe(item: _Slice | None) -> str:
        if item is None:
            return "none"
        if item.midi is None:
            kind = "rest"
        elif len(item.pitches) > 1:
            kind = "chord"
        else:
            kind = "note"
        return (
            f"{kind}@{item.start_tick}:{item.end_tick}"
            f"/pitches={list(item.pitches)}"
            f"/source_event_id={item.source_event_id!r}"
            f"/tie_before={sorted(item.tie_before)}"
            f"/tie_after={sorted(item.tie_after)}"
            f"/tuplet={tuplet_text(item)}"
        )

    item = slices[index]
    return (
        f"voice={voice_id!r} event={item.start_tick}:{item.end_tick}"
        f" kind={'rest' if item.midi is None else 'chord' if len(item.pitches) > 1 else 'note'}"
        f" pitches={list(item.pitches)} source_event_id={item.source_event_id!r}"
        f" tie_before={sorted(item.tie_before)} tie_after={sorted(item.tie_after)}"
        f" tuplet={tuplet_text(item)} dots={item.dots}"
        f" measure={span.start_tick}:{span.end_tick}"
        f" remaining_after={span.end_tick - item.end_tick}"
        f" previous={describe(slices[index - 1] if index else None)}"
        f" following={describe(slices[index + 1] if index + 1 < len(slices) else None)}"
    )


def _serialize_slice_tokens(
    item: _Slice,
    key: str,
    quarter_ticks: int,
    *,
    context: str,
    duration_tick: int | None = None,
    dots: int = 0,
    include_tie: bool = True,
) -> list[str]:
    """Serialize a slice and retain local context when exact output fails."""

    try:
        return _slice_tokens(
            item,
            key,
            quarter_ticks,
            duration_tick=duration_tick,
            dots=dots,
            include_tie=include_tie,
        )
    except JianpuSerializationError as exc:
        raise JianpuSerializationError(f"{exc}; {context}") from exc


def _validate_tuplet_ratio(item: _Slice) -> tuple[int, int] | None:
    supplied = item.tuplet_actual is not None or item.tuplet_normal is not None
    if not supplied:
        return None
    if item.tuplet_actual is None or item.tuplet_normal is None:
        raise JianpuSerializationError("tuplet_actual and tuplet_normal must be supplied together")
    ratio = (item.tuplet_actual, item.tuplet_normal)
    if ratio not in {(3, 2), (4, 3)} and not (item.fine_grid_tuplet and ratio == (3, 1)):
        raise JianpuSerializationError(
            "jianpu-ly serializer only supports explicit 3:2 or 4:3 tuplets; "
            f"bounded fine-grid tuplets may be 3:1, got {ratio[0]}:{ratio[1]}"
        )
    return ratio


def _explicit_tuplet_groups(slices: list[_Slice]) -> dict[int, tuple[tuple[int, int], int]]:
    """Return ``start -> (ratio, exclusive end)`` for explicit tuplets.

    MusicXML can split one 3:2 or 4:3 group into any number of note, chord,
    rest, or tie fragments.  In particular a start/stop pair may surround
    four fragments.  The old implementation assumed exactly three events and
    therefore rejected a legal group at the final fragment.  Explicit
    MusicXML boundaries are the authority; without them we retain the legacy,
    conservative three-slice inference only for 3:2 because a longer
    unbounded 4:3 run is ambiguous.
    """

    groups: dict[int, tuple[tuple[int, int], int]] = {}
    marked = [(_validate_tuplet_ratio(item), item.tuplet_type) for item in slices]
    open_start: int | None = None
    open_ratio: tuple[int, int] | None = None

    for index, (ratio, boundary) in enumerate(marked):
        if slices[index].fine_grid_tuplet_single and boundary != "start":
            raise JianpuSerializationError(
                f"fine-grid singleton at tick {slices[index].start_tick} is missing its start boundary"
            )
        if boundary == "start" and slices[index].fine_grid_tuplet_single:
            if ratio == (4, 3):
                raise JianpuSerializationError(
                    f"4:3 tuplet at tick {slices[index].start_tick} requires explicit start and stop boundaries"
                )
            if open_start is not None:
                raise JianpuSerializationError(
                    f"nested/overlapping explicit tuplets at tick {slices[index].start_tick}"
                )
            if ratio is None:
                raise JianpuSerializationError(
                    f"fine-grid singleton at tick {slices[index].start_tick} has no ratio"
                )
            groups[index] = (ratio, index + 1)
            continue
        if boundary == "start":
            if open_start is not None:
                raise JianpuSerializationError(
                    f"nested/overlapping explicit tuplets at tick {slices[index].start_tick}"
                )
            if ratio is None:
                raise JianpuSerializationError(
                    f"tuplet start at tick {slices[index].start_tick} has no supported ratio"
                )
            open_start, open_ratio = index, ratio
        elif boundary == "stop" and open_start is None:
            raise JianpuSerializationError(
                f"tuplet stop at tick {slices[index].start_tick} has no matching start"
            )

        if open_start is not None:
            if ratio is None or ratio != open_ratio:
                raise JianpuSerializationError(
                    f"explicit tuplet at tick {slices[index].start_tick} has a gap or inconsistent ratio"
                )
            if index and slices[index - 1].end_tick != slices[index].start_tick:
                raise JianpuSerializationError(
                    f"explicit tuplet has a timeline gap before tick {slices[index].start_tick}"
                )
            if boundary == "stop":
                assert open_ratio is not None
                groups[open_start] = (open_ratio, index + 1)
                open_start = None
                open_ratio = None
        elif ratio is not None:
            # Boundary-free records are handled below as the old Score JSON
            # compatibility path.  A ``continue`` without an open group is
            # not enough evidence to infer where that group starts.
            if boundary == "continue":
                raise JianpuSerializationError(
                    f"unanchored explicit tuplet at tick {slices[index].start_tick}"
                )

    if open_start is not None:
        raise JianpuSerializationError(
            f"explicit tuplet at tick {slices[open_start].start_tick} has no stop boundary"
        )

    boundary_indices = {
        index
        for index, (_ratio, boundary) in enumerate(marked)
        if boundary in {"start", "stop", "continue"}
    }
    covered = {index for start, (_ratio, end) in groups.items() for index in range(start, end)}
    if boundary_indices & covered:
        # Every boundary event must be inside the group that owns it.  This
        # also catches a malformed stop after a previous group was closed.
        if boundary_indices - covered:
            index = min(boundary_indices - covered)
            raise JianpuSerializationError(
                f"explicit tuplet boundary at tick {slices[index].start_tick} is not part of a complete group"
            )

    # 4:3 is accepted only when the source supplied both group boundaries.
    # In particular, do not let the legacy boundary-free three-slice
    # compatibility path reinterpret a 4:3 run as a complete group.
    for index, (ratio, _boundary) in enumerate(marked):
        if ratio == (4, 3) and index not in covered:
            raise JianpuSerializationError(
                f"4:3 tuplet at tick {slices[index].start_tick} requires explicit start and stop boundaries"
            )

    # Legacy Score JSON has ratios but no MusicXML boundary metadata.  Only a
    # complete, contiguous three-slice run is safe to infer.  Longer runs are
    # deliberately rejected rather than arbitrarily grouping the first three.
    consumed = set(covered)
    index = 0
    while index < len(slices):
        if index in consumed or marked[index][0] is None or marked[index][1] is not None:
            index += 1
            continue
        run_start = index
        group = slices[run_start : run_start + 3]
        if len(group) != 3 or any(left.end_tick != right.start_tick for left, right in pairwise(group)):
            bad = slices[run_start]
            raise JianpuSerializationError(
                f"explicit tuplet at tick {bad.start_tick} does not form a complete three-note group"
            )
        explicit = [marked[run_start + offset][0] for offset in range(3) if marked[run_start + offset][0] is not None]
        ratio = explicit[0] if explicit else None
        # Older Score JSON (including the checked-in stage56 fixture) may
        # carry the ratio only on two of three fragments.  The contiguous
        # three-slice shape plus two matching ratio slots is the narrow
        # compatibility inference; a longer boundary-free run is rejected.
        if len(explicit) < 2 or any(value != ratio for value in explicit):
            bad = slices[run_start]
            raise JianpuSerializationError(
                f"explicit tuplet at tick {bad.start_tick} does not form a complete three-note group"
            )
        assert ratio is not None
        groups[run_start] = (ratio, run_start + 3)
        consumed.update(range(run_start, run_start + 3))
        index = run_start + 3
    return groups


def _legacy_triplet(slices: list[_Slice], index: int, quarter_ticks: int, bar_end: int) -> bool:
    if index + 2 >= len(slices):
        return False
    group = slices[index : index + 3]
    if any(item.tuplet_actual is not None or item.tuplet_normal is not None for item in group):
        return False
    triplet_duration = Fraction(quarter_ticks, 3)
    if any(item.duration_tick != triplet_duration for item in group):
        return False
    if any(left.end_tick != right.start_tick for left, right in pairwise(group)):
        return False
    return group[-1].end_tick <= bar_end and group[0].start_tick + quarter_ticks <= bar_end


def _serialize_measure(
    slices: list[_Slice],
    span: _MeasureSpan,
    key: str,
    quarter_ticks: int,
    *,
    voice_id: str | None = None,
    explicit_groups: Mapping[int, tuple[tuple[int, int], int]] | None = None,
    global_offset: int = 0,
) -> list[str]:
    if explicit_groups is None:
        local_groups = _explicit_tuplet_groups(slices)
        explicit_groups = {
            start: (ratio, end) for start, (ratio, end) in local_groups.items()
        }
    output: list[str] = []
    index = 0
    cursor = span.start_tick
    while index < len(slices):
        item = slices[index]
        if item.start_tick != cursor:
            raise JianpuSerializationError(f"measure serializer gap at tick {cursor}")
        global_index = global_offset + index
        group_info = next(
            (
                (start, ratio, end)
                for start, (ratio, end) in explicit_groups.items()
                if start <= global_index < end
            ),
            None,
        )
        if group_info is not None:
            start, ratio, end = group_info
            if global_index == start:
                # The vendor syntax ``3[`` is the historical shorthand for
                # a 3:2 group.  Fine-grid ratios use an explicit ``a:b[``
                # form so the LilyPond ``\\times b/a`` factor is unambiguous.
                output.append(
                    f"{ratio[0]}[" if ratio == (3, 2) else f"{ratio[0]}:{ratio[1]}["
                )
            nominal = Fraction(item.duration_tick * ratio[0], ratio[1])
            if nominal.denominator != 1:
                raise JianpuSerializationError(
                    f"tuplet note at tick {item.start_tick} is not exact at {quarter_ticks} TPQ"
                )
            is_last = global_index + 1 == end
            output.extend(
                _serialize_slice_tokens(
                    item,
                    key,
                    quarter_ticks,
                    context=_slice_context(slices, index, span, voice_id=voice_id),
                    duration_tick=int(nominal),
                    dots=item.dots,
                    include_tie=not is_last,
                )
            )
            if is_last:
                output.append("]")
                if item.tie_after and item.pitches:
                    output.append("~")
            cursor = item.end_tick
            index += 1
            continue
        if _legacy_triplet(slices, index, quarter_ticks, span.end_tick):
            group = slices[index : index + 3]
            output.append("3[")
            for group_index, member in enumerate(group):
                nominal = Fraction(member.duration_tick * 3, 2)
                if nominal.denominator != 1:
                    raise JianpuSerializationError("legacy triplet duration is not exact")
                output.extend(
                    _serialize_slice_tokens(
                        member,
                        key,
                        quarter_ticks,
                        context=_slice_context(slices, index + group_index, span, voice_id=voice_id),
                        duration_tick=int(nominal),
                        dots=0,
                        include_tie=group_index < 2,
                    )
                )
            output.append("]")
            if group[-1].tie_after and group[-1].pitches:
                output.append("~")
            cursor = group[-1].end_tick
            index += 3
            continue
        output.extend(
            _serialize_slice_tokens(
                item,
                key,
                quarter_ticks,
                context=_slice_context(slices, index, span, voice_id=voice_id),
                dots=item.dots,
            )
        )
        cursor = item.end_tick
        index += 1
    if cursor != span.end_tick:
        raise JianpuSerializationError(f"measure serializer ended at {cursor}, expected {span.end_tick}")
    output.append("|")
    return output


def _key_command(key: str) -> str:
    # Preserve the historical serializer contract: minor scores are written
    # from their relative major (F#m becomes 1=A).
    return f"1={relative_major_key(normalize_key(key))}"


def jianpu_serialization_diagnostics(score: Score) -> dict[str, Any]:
    """Describe renderer workarounds required by this Score.

    The report is deliberately renderer-facing: the Score remains the source
    of truth, while the report explains why a chord may become several
    ``NextPart`` lanes in the jianpu-ly input.  Consumers that persist Score
    metadata (the high-accuracy service does this before rendering) can keep
    this audit trail with the other notation diagnostics.
    """

    spans = _fixed_measure_spans(score)
    contexts, _meter_header = _measure_contexts(score, spans)
    keys = [context.key for context in contexts]
    records: list[dict[str, Any]] = []
    per_voice: list[dict[str, Any]] = []
    actual_added_lane_count = 0
    for voice in score.voices:
        assignments, _split_indices, _lane_count, voice_records = _chord_lane_plan(voice, keys)
        used_lanes = {
            lane
            for assignment in assignments.values()
            for lane in assignment.values()
            if lane > 0
        }
        actual_added_lane_count += len(used_lanes)
        records.extend(voice_records)
        per_voice.append(
            {
                "voice_id": voice.voice_id,
                "occurrence_count": len(voice_records),
                "actual_added_lane_count": len(used_lanes),
                "occurrences": voice_records,
            }
        )
    return {
        "schema_version": "1.0",
        "occurrence_count": len(records),
        "actual_added_lane_count": actual_added_lane_count,
        "chord_voice_split_count": len(records),
        "chord_voice_split_lane_count": actual_added_lane_count,
        "chord_voice_splits": records,
        "voices": per_voice,
        "strategy": "minimal_reusable_pitch_lanes_for_non_leading_accidental_chords",
    }


def score_to_jianpu(score: Score) -> str:
    """Serialize a 12/48 TPQ Score into notation-preserving jianpu-ly input."""

    spans = _fixed_measure_spans(score)
    contexts, meter_header = _measure_contexts(score, spans)
    title = sanitize_title(score.title)
    key = contexts[0].key
    lines = [
        f"title={title}",
        _key_command(key),
        f"4={contexts[0].tempo_bpm}",
        meter_header,
        # A chord token may contain octave marks between its figures (for
        # example ``1,,11''``).  jianpu-ly needs the explicit policy to know
        # that a mark after a figure belongs to that preceding figure.  The
        # directive is repeated for every NextPart because the vendor resets
        # NoteheadMarkup state at each part boundary.
        "OctavesAfter",
        "",
    ]
    serialization_voices = _serialization_voices(
        score.voices,
        keys=[context.key for context in contexts],
    )
    voice_bars = [_slice_voice_events(voice, spans) for voice in serialization_voices]
    for voice_index, (voice, bars) in enumerate(zip(serialization_voices, voice_bars)):
        if len(serialization_voices) > 1:
            label = sanitize_title(voice.label or voice.voice_id).replace("=", " ")
            lines.append(f"instrument={label}")
            if voice_index:
                # jianpu-ly parses each ``NextPart`` independently.  Repeat
                # the score context for later parts so compound meters (and
                # their key/tempo context) are not reset to the vendor
                # defaults when a new voice starts.
                lines.extend(
                    [_key_command(key), f"4={contexts[0].tempo_bpm}", meter_header, "OctavesAfter"]
                )
        output: list[str] = []
        flattened = [item for bar in bars for item in bar]
        global_tuplet_groups = _explicit_tuplet_groups(flattened)
        slice_offset = 0
        for index, (span, slices, context) in enumerate(zip(spans, bars, contexts)):
            output.extend(_measure_prefix(index, contexts))
            output.extend(
                _serialize_measure(
                    slices,
                    span,
                    context.key,
                    score.quarter_ticks,
                    voice_id=voice.voice_id,
                    explicit_groups=global_tuplet_groups,
                    global_offset=slice_offset,
                )
            )
            slice_offset += len(slices)
            if score.metadata.get("notation_engine") == "direct-jianpu" and (index+1) % 4 == 0 and index+1 < len(spans):
                output.append(r"\break")
        lines.append(" ".join(output))
        if voice_index + 1 < len(serialization_voices):
            lines.append("NextPart")
    return "\n".join(lines) + "\n"
