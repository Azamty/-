"""Voice selection, shared beat quantization and jianpu text generation."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

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


def _time_signature_values(value: str) -> tuple[int, int]:
    normalized = normalize_time_signature(value)
    numerator, denominator = normalized.split("/", 1)
    return int(numerator), int(denominator)


class NoNotesError(ValueError):
    """Raised when an engine produces no usable note events."""


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
        return position + self.shift_beats


def _build_beat_mapper(analysis: MusicAnalysis, events: list[NoteEvent]) -> _BeatMapper:
    beat_times = tuple(analysis.beat_times)
    fixed = analysis.metadata.get("beat_source") == "manual_bpm" or len(beat_times) < 2
    if fixed:
        return _BeatMapper(analysis.bpm, beat_times, True)
    unshifted = _BeatMapper(analysis.bpm, beat_times, False)
    # Preserve the original audio zero on the non-negative Score timeline.
    # Beat detectors commonly return their first beat after the audio starts;
    # translating that map keeps an onset at t=0 instead of clamping it away.
    shift_beats = -unshifted.seconds_to_beat(0.0)
    return _BeatMapper(analysis.bpm, beat_times, False, shift_beats=shift_beats)


def _straight_tick(value: float, quarter_ticks: int) -> int:
    sixteenth = max(1, quarter_ticks // 4)
    return max(0, int(round(value / sixteenth)) * sixteenth)


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


def _tempo_events(analysis: MusicAnalysis, mapper: _BeatMapper, quarter_ticks: int) -> list[TempoEvent]:
    if mapper.fixed or len(mapper.beat_times) < 2:
        return [TempoEvent(start_tick=0, bpm=analysis.bpm)]
    first_interval = mapper.beat_times[1] - mapper.beat_times[0]
    values: list[TempoEvent] = [TempoEvent(start_tick=0, bpm=60.0 / first_interval)]
    for index, (left, right) in enumerate(zip(mapper.beat_times, mapper.beat_times[1:])):
        interval = right - left
        if interval <= 0:
            continue
        bpm = 60.0 / interval
        tick = max(0, int(round((index + mapper.shift_beats) * quarter_ticks)))
        if tick == values[-1].start_tick:
            values[-1] = TempoEvent(start_tick=tick, bpm=bpm)
        elif abs(bpm - values[-1].bpm) > 0.01:
            values.append(TempoEvent(start_tick=tick, bpm=bpm))
    return values


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
    bar_ticks = int(round(numerator * quarter_ticks * 4 / denominator))
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
    bars = max(1, int(math.ceil((raw_total_ticks - quantization_tolerance) / bar_ticks)))
    total_ticks = max(bar_ticks, bars * bar_ticks)
    score_voices: list[ScoreVoice] = []
    triplet_group_count = 0
    for voice_id, voice_events in sorted(grouped.items()):
        ordered_events = sorted(voice_events, key=lambda item: (item.start_sec, item.end_sec, item.midi))
        raw_boundaries = [
            (mapper.seconds_to_beat(event.start_sec) * quarter_ticks, mapper.seconds_to_beat(event.end_sec) * quarter_ticks, event)
            for event in ordered_events
        ]
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
        "beat_offset_sec": analysis.beat_times[0] if analysis.beat_times else 0.0,
        "pickup_beats": max(0.0, -mapper.seconds_to_beat(0.0)),
        "downbeat_status": "undetermined",
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
        warnings=list(analysis.warnings),
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
    continues: bool


def _duration_chunks(duration_tick: int, quarter_ticks: int) -> list[int]:
    if duration_tick <= 0:
        return []
    remaining = duration_tick
    chunks: list[int] = []
    allowed = [quarter_ticks * 3 // 2, quarter_ticks, quarter_ticks // 2, quarter_ticks // 4]
    allowed = sorted({value for value in allowed if value > 0}, reverse=True)
    while remaining:
        choice = next((value for value in allowed if value <= remaining), None)
        if choice is None:
            raise ValueError(
                f"duration {duration_tick} ticks cannot be represented by jianpu-ly "
                f"with quarter_ticks={quarter_ticks}"
            )
        chunks.append(choice)
        remaining -= choice
    return chunks


def _format_duration(note: str, duration_tick: int, quarter_ticks: int) -> list[str]:
    if duration_tick <= 0:
        return []
    if duration_tick == quarter_ticks * 3 // 2:
        return [f"{note}."]
    if duration_tick == quarter_ticks:
        return [note]
    if duration_tick == quarter_ticks // 2:
        return [f"q{note}"]
    if duration_tick == quarter_ticks // 4:
        return [f"s{note}"]
    return [token for chunk in _duration_chunks(duration_tick, quarter_ticks) for token in _format_duration(note, chunk, quarter_ticks)]


def score_to_jianpu(score: Score) -> str:
    """Serialize a Score into jianpu-ly input with explicit bars and tuplets."""

    numerator, denominator = _time_signature_values(score.time_signature)
    bar_ticks = int(round(numerator * score.quarter_ticks * 4 / denominator))
    title = sanitize_title(score.title)
    key = normalize_key(score.key)
    key_command = f"1={relative_major_key(key)}"
    lines = [f"title={title}", key_command, f"4={round(score.bpm)}", normalize_time_signature(score.time_signature), ""]
    for voice_index, voice in enumerate(score.voices):
        slices: list[_Slice] = []
        for event in voice.events:
            end = event.end_tick
            cursor = event.start_tick
            while cursor < end:
                boundary = ((cursor // bar_ticks) + 1) * bar_ticks
                segment_end = min(end, boundary)
                slices.append(_Slice(cursor, segment_end - cursor, event.midi, segment_end < end))
                cursor = segment_end
        output: list[str] = []
        cursor = 0
        index = 0
        while cursor < score.total_ticks:
            bar_end = min(score.total_ticks, ((cursor // bar_ticks) + 1) * bar_ticks)
            bar_tokens: list[str] = []
            while cursor < bar_end and index < len(slices):
                current = slices[index]
                if current.start_tick != cursor:
                    raise ValueError(f"voice {voice.voice_id} serializer gap at {cursor}")
                if (
                    index + 2 < len(slices)
                    and current.duration_tick == score.quarter_ticks // 3
                    and slices[index + 1].duration_tick == score.quarter_ticks // 3
                    and slices[index + 2].duration_tick == score.quarter_ticks // 3
                    and current.start_tick + score.quarter_ticks // 3 == slices[index + 1].start_tick
                    and current.start_tick + 2 * score.quarter_ticks // 3 == slices[index + 2].start_tick
                    and current.start_tick + score.quarter_ticks <= bar_end
                ):
                    notes = []
                    for triplet in slices[index : index + 3]:
                        number = "0" if triplet.midi is None else midi_to_jianpu(triplet.midi, score.key)
                        notes.append(f"q{number}")
                    bar_tokens.extend(["3[", *notes, "]"])
                    cursor += score.quarter_ticks
                    index += 3
                    continue
                number = "0" if current.midi is None else midi_to_jianpu(current.midi, score.key)
                chunks = _format_duration(number, current.duration_tick, score.quarter_ticks)
                for chunk_index, token in enumerate(chunks):
                    bar_tokens.append(token)
                    if chunk_index < len(chunks) - 1 or current.continues:
                        if current.midi is not None:
                            bar_tokens.append("~")
                cursor += current.duration_tick
                index += 1
            if cursor != bar_end:
                raise ValueError(f"voice {voice.voice_id} serializer bar mismatch at {cursor}, expected {bar_end}")
            bar_tokens.append("|")
            output.extend(bar_tokens)
        lines.append(" ".join(output))
        if voice_index + 1 < len(score.voices):
            lines.append("NextPart")
    return "\n".join(lines) + "\n"
