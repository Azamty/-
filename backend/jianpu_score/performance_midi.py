"""Write unquantized 480 PPQ performance MIDI from the high-accuracy map.

This module is deliberately separate from :mod:`quantize`.  It preserves the
source note timing and polyphony for the MuseScore MIDI importer; it does not
call ``quantize_events`` or snap notes to a notation grid.  The only rounding
is the unavoidable conversion from fractional score beats to integer MIDI
ticks.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

from .domain import MusicAnalysis, NoteEvent, normalize_key, normalize_time_signature
from .quantize import _build_beat_mapper

PERFORMANCE_TICKS_PER_QUARTER = 480
PERFORMANCE_SCHEMA_VERSION = "1.0"
DEFAULT_VELOCITY = 80
DRUM_CHANNEL = 9  # MIDI channel 10 in one-based terminology.
# MuseScore keeps at most four voices on one staff, but a type-1 MIDI file can
# carry additional independent tracks.  The notation importer turns those
# tracks into additional parts/ScoreVoice lanes.  Four remains the preferred
# per-staff voice count; it is not a lossless ceiling for performance MIDI.
PREFERRED_MIDI_VOICE_LANES_PER_STAFF = 4
MIDI_MELODIC_CHANNELS = tuple(channel for channel in range(16) if channel != DRUM_CHANNEL)


def _midi_text(value: str, *, fallback: str = "track") -> str:
    """Return a deterministic ASCII SMF track name.

    The full Unicode title is kept in the adjacent metadata/manifest.  SMF
    text fields are limited by the mido codec, so use a readable transliterated
    prefix plus a hash whenever the original title cannot be represented.
    """

    original = str(value).strip() or fallback
    readable = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode("ascii")
    readable = "".join(char if char.isalnum() or char in " ._-" else "_" for char in readable)
    readable = " ".join(readable.split()).strip(" ._-") or fallback
    if readable != original:
        digest = hashlib.sha1(original.encode()).hexdigest()[:10]
        readable = f"{readable[:51].rstrip(' ._-')}-{digest}"
    return readable[:63]


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
    track_id: str | None = None


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
    """Validate a jianpu key and return the mido-compatible spelling.

    mido follows the compact MIDI key-signature vocabulary and does not accept
    the enharmonic minor spellings ``Dbm`` and ``Gbm`` even though they are
    valid application keys.  Keep the analysis spelling in JSON and emit an
    equivalent key signature for MIDI consumers.
    """

    normalized = normalize_key(value)
    return {"Dbm": "C#m", "Gbm": "F#m"}.get(normalized, normalized)


def _default_track_id(instrument_group: str, program: int, is_drum: bool) -> str:
    identity = f"{instrument_group}|{program}|{int(is_drum)}".encode()
    return "track-" + hashlib.sha1(identity).hexdigest()[:12]


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
    left = math.floor(raw_position)
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
        return [(0, round(mido.bpm2tempo(bpm)), bpm)]
    times = mapper.beat_times
    scale = float(mapper.beat_scale)
    shift = float(mapper.shift_beats)
    intervals = [right - left for left, right in itertools.pairwise(times)]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("beat times must be strictly increasing")

    raw: list[tuple[int, int, float]] = []
    for index, interval in enumerate(intervals):
        bpm = _finite_positive(60.0 * scale / interval, label="tempo map BPM")
        score_beat = index * scale + shift
        tick = round(score_beat * PERFORMANCE_TICKS_PER_QUARTER)
        raw.append((tick, round(mido.bpm2tempo(bpm)), bpm))

    # MIDI cannot represent negative delta time.  Select the tempo interval
    # that contains score zero after applying the explicit score origin, then
    # retain later points in the non-negative score timeline.  This matters
    # when the first detected downbeat is a later BeatNet beat: the interval
    # before that downbeat must not become the playback tempo at tick zero.
    raw_origin_position = -shift / scale if scale else 0.0
    base_index = max(0, min(len(raw) - 1, math.floor(raw_origin_position)))
    points: dict[int, tuple[int, float]] = {0: (raw[base_index][1], raw[base_index][2])}
    for tick, tempo, bpm in raw:
        if tick <= 0:
            continue
        points[tick] = (tempo, bpm)
    # A local beat grid emits one candidate tempo per interval.  Consecutive
    # intervals that quantize to the same MIDI tempo are semantically one
    # tempo segment; retaining every duplicate needlessly asks MusicXML
    # source alignment to prove a mapping for redundant events.  Collapse
    # only adjacent equal encoded tempos, preserving every actual change and
    # the original tick of each change.
    compact: list[tuple[int, int, float]] = []
    for tick, value in sorted(points.items()):
        point = (tick, value[0], value[1])
        if compact and point[1] == compact[-1][1]:
            continue
        compact.append(point)
    return compact


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
        start_tick = max(0, round(start_beat * PERFORMANCE_TICKS_PER_QUARTER))
        end_tick = max(start_tick + 1, round(end_beat * PERFORMANCE_TICKS_PER_QUARTER))
        cleanup = note.metadata.get("instrumental_cleanup")
        source_index = index
        source_indices = [index]
        if isinstance(cleanup, Mapping):
            try:
                source_index = int(cleanup.get("primary_source_index", index))
                source_indices = [int(value) for value in cleanup.get("source_indices", [source_index])]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"note {index} has invalid instrumental cleanup lineage") from exc
        mapped.append(
            {
                "index": index,
                "source_index": source_index,
                "source_indices": source_indices,
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


def _assign_midi_voice_lanes(
    notes: Sequence[Mapping[str, Any]],
    *,
    max_lanes: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Partition notes so a channel/pitch pair keeps every onset identifiable.

    MIDI permits overlapping notes with the same pitch on one channel, but
    importers are free to pair the note-off with the wrong note-on.  Keep the
    normal single-track representation for ordinary chords.  A same-pitch
    retrigger whose previous note ends at the next note's start gets another
    lane as well: MuseScore can otherwise interpret an exact boundary as one
    sustained note and lose an onset.  A genuine same-pitch interval overlap
    also gets another lane.  The greedy coloring is minimal for these
    constraints.  When ``max_lanes`` is ``None`` (the production path), each
    additional lane is a separate MIDI track so note-on/off identity remains
    lossless even when a source has more than four same-pitch intervals.  A
    finite limit is retained for callers that want an explicit resource guard;
    it always fails before dropping or merging source notes.
    """

    if max_lanes is not None and max_lanes <= 0:
        raise ValueError("max_lanes must be greater than zero")
    lanes: list[list[dict[str, Any]]] = []
    pitch_ends: list[dict[int, int]] = []
    for note in sorted(notes, key=lambda value: (int(value["start_tick"]), int(value["end_tick"]), int(value["index"]))):
        pitch = int(note["midi"])
        start_tick = int(note["start_tick"])
        existing_ends = [int(ends[pitch]) for ends in pitch_ends if pitch in ends]
        lane_index = next(
            (
                index
                for index, ends in enumerate(pitch_ends)
                if pitch not in ends or int(ends[pitch]) < start_tick
            ),
            None,
        )
        if lane_index is None:
            lane_index = len(lanes)
            if max_lanes is not None and lane_index >= max_lanes:
                raise ValueError(
                    "performance MIDI requires more than "
                    f"{max_lanes} same-pitch voice lanes at tick {start_tick}; "
                    "preserve the notes with additional ScoreVoice lanes or an "
                    "explicit lossless MIDI representation"
                )
            lanes.append([])
            pitch_ends.append({})
        copied = dict(note)
        copied["midi_lane"] = lane_index
        if not existing_ends:
            copied["midi_lane_reason"] = "primary_lane"
        elif not any(end < start_tick for end in existing_ends):
            if any(end == start_tick for end in existing_ends):
                copied["midi_lane_reason"] = "adjacent_retrigger"
            else:
                copied["midi_lane_reason"] = "overlap"
        else:
            copied["midi_lane_reason"] = "reused_lane"
        lanes[lane_index].append(copied)
        pitch_ends[lane_index][pitch] = max(
            int(pitch_ends[lane_index].get(pitch, 0)),
            int(note["end_tick"]),
        )
    return lanes


def _midi_lane_channel(lane_index: int, *, is_drum: bool) -> int:
    """Return a stable channel for a performance voice lane."""

    # Keep the first drum lane on General MIDI channel 10.  Additional drum
    # lanes use ordinary channels so same-pitch overlaps remain unambiguous;
    # drum notation is never derived from this playback-only artifact.
    if lane_index < 0:
        raise ValueError("lane_index must be non-negative")
    if is_drum and lane_index == 0:
        return DRUM_CHANNEL
    melodic_index = lane_index if not is_drum else lane_index - 1
    # A track boundary, rather than a unique channel, disambiguates repeated
    # channels after the 15 melodic channels are exhausted.  Every individual
    # track still has non-overlapping same-pitch spans by construction.
    return MIDI_MELODIC_CHANNELS[melodic_index % len(MIDI_MELODIC_CHANNELS)]


def _write_track_messages(
    notes: Sequence[Mapping[str, Any]],
    *,
    title: str,
    channel: int,
    program: int,
    is_drum: bool,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=_midi_text(title, fallback="Instrument"), time=0))
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
    track.append(mido.MetaMessage("track_name", name=_midi_text(title, fallback="Performance"), time=0))
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
    track_id: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Build one independent 480 PPQ performance MIDI and audit metadata."""

    materialized = tuple(events)
    if not materialized:
        raise ValueError("performance MIDI requires at least one note")
    group = str(instrument_group).strip() or "unknown"
    bounded_program = _bounded_program(program)
    resolved_track_id = str(track_id).strip() if track_id is not None and str(track_id).strip() else _default_track_id(group, bounded_program, is_drum)
    mapper = _build_beat_mapper(analysis, list(materialized))
    mapped = _absolute_note_ticks(materialized, mapper)
    lane_notes = _assign_midi_voice_lanes(mapped)
    tempo_points = _tempo_points(mapper)
    track_title = str(title).strip() or group
    conductor_track_name = _midi_text(track_title, fallback="Performance")
    instrument_track_names = [
        _midi_text(group if index == 0 else f"{group} voice {index + 1}", fallback="Instrument")
        for index in range(len(lane_notes))
    ]
    midi = mido.MidiFile(type=1, ticks_per_beat=PERFORMANCE_TICKS_PER_QUARTER)
    midi.tracks.append(
        _write_conductor_track(title=track_title, analysis=analysis, tempo_points=tempo_points)
    )
    for lane_index, notes in enumerate(lane_notes):
        midi.tracks.append(
            _write_track_messages(
                notes,
                title=instrument_track_names[lane_index],
                channel=_midi_lane_channel(lane_index, is_drum=is_drum),
                program=bounded_program,
                is_drum=is_drum,
            )
        )
    channels = [_midi_lane_channel(index, is_drum=is_drum) for index in range(len(lane_notes))]
    adjacent_retrigger_indices = [
        int(item["source_index"])
        for lane in lane_notes
        for item in lane
        if item.get("midi_lane_reason") == "adjacent_retrigger"
    ]
    metadata: dict[str, Any] = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "artifact_kind": "performance_midi",
        "ticks_per_quarter": PERFORMANCE_TICKS_PER_QUARTER,
        "title": track_title,
        "midi_track_names": {
            "conductor": conductor_track_name,
            "instrument": instrument_track_names[0],
        },
        "instrument_lane_track_names": instrument_track_names,
        "instrument_group": group,
        "track_id": resolved_track_id,
        "program": bounded_program,
        "is_drum": bool(is_drum),
        "channel": channels[0] + 1,
        "channels": [channel_value + 1 for channel_value in channels],
        "voice_lane_count": len(lane_notes),
        "voice_lane_policy": "same_pitch_interval_coloring_lossless_midi_tracks",
        "lane_assignment": {
            "schema_version": "1.0",
            "reuse_condition": "same_pitch_previous_end_tick_strictly_less_than_next_start_tick",
            "adjacent_retrigger_split_count": len(adjacent_retrigger_indices),
            "adjacent_retrigger_source_indices": adjacent_retrigger_indices,
        },
        "preferred_voice_lanes_per_staff": PREFERRED_MIDI_VOICE_LANES_PER_STAFF,
        "track_channel_reuse_policy": "channels may repeat after 15 melodic lanes because each lane has an independent MIDI track",
        "source": "unquantized_note_events",
        "note_count": len(mapped),
        "source_note_count": int(
            (analysis.metadata.get("instrumental_cleanup") or {}).get("source_note_count", len(mapped))
            if isinstance(analysis.metadata.get("instrumental_cleanup"), Mapping)
            else len(mapped)
        ),
        "source_merged_count": int(
            (analysis.metadata.get("instrumental_cleanup") or {}).get("merged_count", 0)
            if isinstance(analysis.metadata.get("instrumental_cleanup"), Mapping)
            else 0
        ),
        "instrumental_cleanup": analysis.metadata.get("instrumental_cleanup"),
        "time_signature": analysis.time_signature,
        "key": analysis.key,
        "emitted_key": _midi_key(analysis.key),
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
                "midi_channel": channels[int(item["midi_lane"]) ] + 1,
                "midi_track_index": int(item["midi_lane"]) + 1,
                "playback_start_sec": _tick_to_seconds(int(item["start_tick"]), tempo_points),
                "playback_end_sec": _tick_to_seconds(int(item["end_tick"]), tempo_points),
            }
            for item in sorted(
                (item for lane in lane_notes for item in lane),
                key=lambda value: int(value["index"]),
            )
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
    track_id: str | None = None,
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
        track_id=track_id,
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
    used_stems: dict[str, int] = {}
    for track in tracks:
        safe_group = "".join(char if char.isalnum() or char in "-_" else "_" for char in track.instrument_group).strip("_") or "unknown"
        bounded_program = _bounded_program(track.program)
        resolved_track_id = str(track.track_id).strip() if track.track_id is not None and str(track.track_id).strip() else _default_track_id(track.instrument_group, bounded_program, track.is_drum)
        safe_track_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in resolved_track_id).strip("_") or "track"
        base_stem = f"{safe_group}.p{bounded_program}.d{int(track.is_drum)}.{safe_track_id}.performance"
        occurrence = used_stems.get(base_stem, 0) + 1
        used_stems[base_stem] = occurrence
        stem = base_stem if occurrence == 1 else f"{base_stem}.{occurrence}"
        result.append(
            write_performance_midi(
                track.notes,
                analysis,
                destination / f"{stem}.mid",
                instrument_group=track.instrument_group,
                program=bounded_program,
                is_drum=track.is_drum,
                title=track.title or title,
                overwrite=overwrite,
                track_id=resolved_track_id,
            )
        )
    return result
