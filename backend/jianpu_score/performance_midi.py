"""Write unquantized 480 PPQ performance MIDI from the high-accuracy map.

This module is deliberately separate from :mod:`quantize`.  It preserves the
source note timing and polyphony for the MuseScore MIDI importer; it does not
call ``quantize_events`` or snap notes to a notation grid.  The only rounding
is the unavoidable conversion from fractional score beats to integer MIDI
ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mido

from .domain import MusicAnalysis, NoteEvent, normalize_key, normalize_time_signature
from .quantize import _build_beat_mapper


PERFORMANCE_TICKS_PER_QUARTER = 480
PERFORMANCE_SCHEMA_VERSION = "1.0"
DEFAULT_VELOCITY = 80
DRUM_CHANNEL = 9  # MIDI channel 10 in one-based terminology.


@dataclass(frozen=True)
class PerformanceMidiArtifact:
    """Files and audit metadata for one independent instrument performance."""

    midi_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PerformanceTrack:
    """One selected instrument to render as its own performance MIDI."""

    instrument_group: str
    notes: tuple[NoteEvent, ...]
    program: int = 0
    is_drum: bool = False
    title: str | None = None


def _finite_positive(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a finite number greater than zero")
    return result


def _bounded_program(value: int) -> int:
    program = int(value)
    if not 0 <= program <= 127:
        raise ValueError("MIDI program must be between 0 and 127")
    return program


def _bounded_velocity(value: int | None) -> int:
    if value is None:
        return DEFAULT_VELOCITY
    return max(1, min(127, int(value)))


def _midi_key(value: str) -> str:
    """Validate a jianpu key and return the mido-compatible spelling."""

    return normalize_key(value)


def _midi_denominator(denominator: int) -> int:
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("MIDI time-signature denominator must be a power of two")
    # mido's MetaMessage API accepts the musical denominator (8 for 6/8) and
    # encodes the power-of-two exponent while serializing the MIDI bytes.
    return denominator


def _score_origin_audio_seconds(mapper: Any) -> float:
    """Return the source second represented by score beat zero.

    This value is metadata for playback/audit.  A pickup may intentionally
    place score zero before the first detected BeatNet beat, so the value can
    be outside the observed beat interval.
    """

    if mapper.fixed or len(mapper.beat_times) < 2:
        return 0.0
    raw_position = -float(mapper.shift_beats) / float(mapper.beat_scale or 1.0)
    times = mapper.beat_times
    if raw_position <= 0:
        interval = times[1] - times[0]
        return float(times[0] + raw_position * interval)
    if raw_position >= len(times) - 1:
        interval = times[-1] - times[-2]
        return float(times[-1] + (raw_position - (len(times) - 1)) * interval)
    left = int(math.floor(raw_position))
    fraction = raw_position - left
    return float(times[left] + fraction * (times[left + 1] - times[left]))


def _tempo_points(mapper: Any) -> list[tuple[int, int, float]]:
    """Return ``(absolute_tick, microseconds_per_beat, bpm)`` points.

    Each BeatNet interval gets its own tempo.  A point at score tick zero is
    always present, even when the first observed beat is a pickup after score
    zero, so MIDI playback has a defined tempo before the first change.
    """

    if mapper.fixed or len(mapper.beat_times) < 2:
        bpm = _finite_positive(mapper.bpm, label="BPM")
        return [(0, int(round(mido.bpm2tempo(bpm))), bpm)]
    times = mapper.beat_times
    scale = float(mapper.beat_scale)
    shift = float(mapper.shift_beats)
    intervals = [right - left for left, right in zip(times, times[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("beat times must be strictly increasing")

    raw: list[tuple[int, int, float]] = []
    for index, interval in enumerate(intervals):
        bpm = _finite_positive(60.0 * scale / interval, label="tempo map BPM")
        score_beat = index * scale + shift
        tick = int(round(score_beat * PERFORMANCE_TICKS_PER_QUARTER))
        raw.append((tick, int(round(mido.bpm2tempo(bpm))), bpm))

    # MIDI cannot represent negative delta time.  Select the tempo interval
    # that contains score zero after applying the explicit score origin, then
    # retain later points in the non-negative score timeline.  This matters
    # when the first detected downbeat is a later BeatNet beat: the interval
    # before that downbeat must not become the playback tempo at tick zero.
    raw_origin_position = -shift / scale if scale else 0.0
    base_index = max(0, min(len(raw) - 1, int(math.floor(raw_origin_position))))
    points: dict[int, tuple[int, float]] = {0: (raw[base_index][1], raw[base_index][2])}
    for tick, tempo, bpm in raw:
        if tick <= 0:
            continue
        points[tick] = (tempo, bpm)
    return [(tick, value[0], value[1]) for tick, value in sorted(points.items())]


def _tick_to_seconds(tick: int, tempo_points: Sequence[tuple[int, int, float]]) -> float:
    """Convert a non-negative score tick using the emitted piecewise tempo."""

    if tick <= 0:
        return 0.0
    elapsed = 0.0
    previous_tick = 0
    previous_tempo = tempo_points[0][1]
    for point_tick, tempo, _bpm in tempo_points[1:]:
        if point_tick >= tick:
            break
        elapsed += mido.tick2second(point_tick - previous_tick, PERFORMANCE_TICKS_PER_QUARTER, previous_tempo)
        previous_tick = point_tick
        previous_tempo = tempo
    elapsed += mido.tick2second(tick - previous_tick, PERFORMANCE_TICKS_PER_QUARTER, previous_tempo)
    return float(elapsed)


def _absolute_note_ticks(
    notes: Sequence[NoteEvent],
    mapper: Any,
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for index, note in enumerate(notes):
        start_beat = float(mapper.seconds_to_beat(note.start_sec))
        end_beat = float(mapper.seconds_to_beat(note.end_sec))
        start_tick = max(0, int(round(start_beat * PERFORMANCE_TICKS_PER_QUARTER)))
        end_tick = max(start_tick + 1, int(round(end_beat * PERFORMANCE_TICKS_PER_QUARTER)))
        mapped.append(
            {
                "index": index,
                "midi": int(note.midi),
                "start_sec": float(note.start_sec),
                "end_sec": float(note.end_sec),
                "start_beat": start_beat,
                "end_beat": end_beat,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "duration_ticks": end_tick - start_tick,
                "velocity": _bounded_velocity(note.velocity),
                "source": note.source,
                "voice_id": note.voice_id,
                "stem_id": note.stem_id,
            }
        )
    return mapped


def _write_track_messages(
    notes: Sequence[Mapping[str, Any]],
    *,
    title: str,
    channel: int,
    program: int,
    is_drum: bool,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=title, time=0))
    if not is_drum:
        track.append(mido.Message("program_change", channel=channel, program=program, time=0))
    events: list[tuple[int, int, int, mido.Message]] = []
    for sequence, note in enumerate(notes):
        pitch = max(0, min(127, int(note["midi"])))
        velocity = _bounded_velocity(note.get("velocity"))
        start_tick = int(note["start_tick"])
        end_tick = int(note["end_tick"])
        # Note-offs sort before note-ons at the same tick.  This keeps the
        # generated stream legal and deterministic for adjacent same-pitch
        # events while retaining simultaneous chords and independent voices.
        events.append(
            (
                start_tick,
                1,
                sequence,
                mido.Message("note_on", channel=channel, note=pitch, velocity=velocity, time=0),
            )
        )
        events.append(
            (
                end_tick,
                0,
                sequence,
                mido.Message("note_off", channel=channel, note=pitch, velocity=0, time=0),
            )
        )
    previous = 0
    for tick, _priority, _sequence, message in sorted(events, key=lambda value: (value[0], value[1], value[2])):
        message.time = max(0, tick - previous)
        track.append(message)
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _write_conductor_track(
    *,
    title: str,
    analysis: MusicAnalysis,
    tempo_points: Sequence[tuple[int, int, float]],
) -> mido.MidiTrack:
    numerator, denominator = (int(value) for value in normalize_time_signature(analysis.time_signature).split("/"))
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=title, time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=numerator,
            denominator=_midi_denominator(denominator),
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    track.append(mido.MetaMessage("key_signature", key=_midi_key(analysis.key), time=0))
    previous = 0
    for tick, tempo, _bpm in tempo_points:
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=max(0, tick - previous)))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def build_performance_midi(
    events: Iterable[NoteEvent],
    analysis: MusicAnalysis,
    *,
    instrument_group: str,
    program: int = 0,
    is_drum: bool = False,
    title: str = "Performance",
) -> tuple[bytes, dict[str, Any]]:
    """Build one independent 480 PPQ performance MIDI and audit metadata."""

    materialized = tuple(events)
    if not materialized:
        raise ValueError("performance MIDI requires at least one note")
    group = str(instrument_group).strip() or "unknown"
    bounded_program = _bounded_program(program)
    mapper = _build_beat_mapper(analysis, list(materialized))
    mapped = _absolute_note_ticks(materialized, mapper)
    tempo_points = _tempo_points(mapper)
    channel = DRUM_CHANNEL if is_drum else 0
    track_title = str(title).strip() or group
    midi = mido.MidiFile(type=1, ticks_per_beat=PERFORMANCE_TICKS_PER_QUARTER)
    midi.tracks.append(
        _write_conductor_track(title=track_title, analysis=analysis, tempo_points=tempo_points)
    )
    midi.tracks.append(
        _write_track_messages(
            mapped,
            title=group,
            channel=channel,
            program=bounded_program,
            is_drum=is_drum,
        )
    )
    metadata: dict[str, Any] = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "artifact_kind": "performance_midi",
        "ticks_per_quarter": PERFORMANCE_TICKS_PER_QUARTER,
        "title": track_title,
        "instrument_group": group,
        "program": bounded_program,
        "is_drum": bool(is_drum),
        "channel": channel + 1,
        "source": "unquantized_note_events",
        "note_count": len(mapped),
        "time_signature": analysis.time_signature,
        "key": analysis.key,
        "tempo_points": [
            {"tick": tick, "bpm": round(bpm, 9), "microseconds_per_beat": tempo}
            for tick, tempo, bpm in tempo_points
        ],
        "tempo_map_source": "beatnet_local_beat_intervals" if not mapper.fixed else "analysis_bpm",
        "score_origin": dict(mapper.score_origin),
        "score_origin_audio_sec": _score_origin_audio_seconds(mapper),
        "beat_scale": mapper.beat_scale,
        "beat_shift_beats": mapper.shift_beats,
        "manual_bpm_override": analysis.metadata.get("manual_bpm_override", False),
        "manual_time_signature_override": analysis.metadata.get("manual_time_signature_override", False),
        "beat_engine": analysis.metadata.get("beat_engine", "beatnet"),
        "beatnet_version": analysis.metadata.get("beatnet_version", "1.1.3"),
        "source_artifact_policy": "write_adjacent_performance_artifact_without_overwriting_input",
        "notes": [
            {
                **{key: value for key, value in item.items() if key not in {"start_beat", "end_beat"}},
                "playback_start_sec": _tick_to_seconds(int(item["start_tick"]), tempo_points),
                "playback_end_sec": _tick_to_seconds(int(item["end_tick"]), tempo_points),
            }
            for item in mapped
        ],
        "drum_jianpu_policy": "midi_only" if is_drum else "eligible_for_jianpu",
    }
    return _midi_bytes(midi), metadata


def _midi_bytes(midi: mido.MidiFile) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def write_performance_midi(
    events: Iterable[NoteEvent],
    analysis: MusicAnalysis,
    destination: str | Path,
    *,
    instrument_group: str,
    program: int = 0,
    is_drum: bool = False,
    title: str = "Performance",
    metadata_destination: str | Path | None = None,
    overwrite: bool = False,
) -> PerformanceMidiArtifact:
    """Write ``*.performance.mid`` and its adjacent audit JSON.

    The default refuses an existing file so original MuScriptor/GAME MIDI
    artifacts cannot be overwritten accidentally.  Callers can opt into an
    explicit replacement for a generated performance artifact only.
    """

    path = Path(destination).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"performance MIDI already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_bytes, metadata = build_performance_midi(
        events,
        analysis,
        instrument_group=instrument_group,
        program=program,
        is_drum=is_drum,
        title=title,
    )
    path.write_bytes(midi_bytes)
    metadata_path = (
        Path(metadata_destination).expanduser().resolve()
        if metadata_destination is not None
        else path.with_suffix(".metadata.json")
    )
    if metadata_path.exists() and not overwrite:
        path.unlink(missing_ok=True)
        raise FileExistsError(f"performance MIDI metadata already exists: {metadata_path}")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PerformanceMidiArtifact(path, metadata_path, metadata)


def write_performance_midi_bundle(
    tracks: Iterable[PerformanceTrack],
    analysis: MusicAnalysis,
    output_dir: str | Path,
    *,
    title: str = "Performance",
    overwrite: bool = False,
) -> list[PerformanceMidiArtifact]:
    """Write one stable ``<instrument>.performance.mid`` per selected track."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result: list[PerformanceMidiArtifact] = []
    for track in tracks:
        safe_group = "".join(char if char.isalnum() or char in "-_" else "_" for char in track.instrument_group).strip("_") or "unknown"
        result.append(
            write_performance_midi(
                track.notes,
                analysis,
                destination / f"{safe_group}.performance.mid",
                instrument_group=track.instrument_group,
                program=track.program,
                is_drum=track.is_drum,
                title=track.title or title,
                overwrite=overwrite,
            )
        )
    return result
