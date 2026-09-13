"""Direct beat-grid notation with separate vocal and polyphonic policies.

Recognition stays immutable. Every merge/drop is accounted for in the report;
no song names, note indices, keys, or modulation times are built into the policy.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain import MusicAnalysis, NoteEvent, Score, ScoreNote, ScoreVoice, TempoEvent, normalize_key
from .quantize import _build_beat_mapper, _tempo_points_for_mapper, midi_to_jianpu

DIRECT_ENGINE = "direct-jianpu"
NotationEngine = Literal["direct-jianpu", "musescore-midi-import"]
TPQ = 48


class KeyChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bar: int = Field(ge=1)
    key: str

    @field_validator("key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        return normalize_key(value)


class DirectNotationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    beat_divisor: Literal[1, 2] = 1
    subdivisions: Literal[2, 4, 8] = 4
    vocal_min_duration_sec: float = Field(default=0.09, ge=0, le=0.3)
    vocal_gap_sec: float = Field(default=0.16, ge=0, le=0.3)
    bass_split_midi: int = Field(default=50, ge=0, le=127)
    melody_min_midi: int = Field(default=60, ge=0, le=127)
    key_changes: list[KeyChange] = Field(default_factory=list, max_length=128)


def build_direct_score(
    events: list[NoteEvent], analysis: MusicAnalysis, *, title: str,
    mode: Literal["vocal", "polyphonic"] = "polyphonic",
    options: DirectNotationOptions | None = None,
) -> tuple[Score, dict[str, Any]]:
    """Produce the common Score without MIDI-import notation inference.

    Use the project's shared origin and tempo map so original-audio playback,
    pickups, manual BPM overrides and multiple selected tracks stay aligned.
    """
    opts = options or DirectNotationOptions()
    if mode not in {"vocal", "polyphonic"}:
        raise ValueError("direct notation mode must be vocal or polyphonic")
    if not events:
        raise ValueError("direct notation requires pitched notes")
    mapper = _build_beat_mapper(analysis, events)
    grid = TPQ // opts.subdivisions
    divisor = opts.beat_divisor
    bar_ticks = round(mapper.bar_duration_quarters * TPQ)
    actions: list[dict[str, Any]] = []

    def snap(seconds: float) -> int:
        return max(0, round(mapper.seconds_to_beat(seconds) * TPQ / divisor / grid) * grid)

    material: list[dict[str, Any]] = []
    for i, note in enumerate(events):
        if mode == "vocal" and note.end_sec - note.start_sec < opts.vocal_min_duration_sec:
            actions.append({"action": "short_vocal_event", "source_indices": [i]})
            continue
        a = snap(note.start_sec)
        material.append({"a": a, "z": max(a + grid, snap(note.end_sec)), "p": note.midi,
                         "ids": [i], "start": note.start_sec, "end": note.end_sec,
                         "velocity": note.velocity, "stem": note.stem_id})
    material.sort(key=lambda n: (n["a"], n["start"], n["p"]))
    kept: list[dict[str, Any]] = []
    if mode == "vocal":
        for note in material:
            if kept and note["a"] == kept[-1]["a"]:
                previous = kept[-1]
                winner, loser = (note, previous) if note["end"] - note["start"] > previous["end"] - previous["start"] else (previous, note)
                actions.append({"action": "vocal_grid_collision", "source_indices": loser["ids"], "kept_source_indices": winner["ids"]})
                kept[-1] = winner
                continue
            if kept:
                previous = kept[-1]
                if previous["z"] > note["a"]:
                    actions.append({"action": "vocal_overlap", "source_indices": previous["ids"], "old_end_tick": previous["z"], "new_end_tick": note["a"]})
                    previous["z"] = note["a"]
                elif 0 < note["a"] - previous["z"] <= grid and note["start"] - previous["end"] < opts.vocal_gap_sec:
                    actions.append({"action": "vocal_gap", "source_indices": previous["ids"], "old_end_tick": previous["z"], "new_end_tick": note["a"]})
                    previous["z"] = note["a"]
            kept.append(note)
    else:
        unique: dict[tuple[str | None, int, int], dict[str, Any]] = {}
        for note in material:
            identity = (note["stem"], note["a"], note["p"])
            if identity in unique:
                previous = unique[identity]
                previous["z"] = max(previous["z"], note["z"])
                previous["ids"].extend(note["ids"])
                actions.append({"action": "same_key_grid_duplicate", "source_indices": note["ids"]})
            else:
                unique[identity] = note
        kept = list(unique.values())
        last_key: dict[tuple[str | None, int], dict[str, Any]] = {}
        for note in kept:
            identity = (note["stem"], note["p"])
            previous = last_key.get(identity)
            if previous and previous["z"] > note["a"]:
                actions.append({"action": "same_key_rearticulation", "source_indices": previous["ids"], "old_end_tick": previous["z"], "new_end_tick": note["a"]})
                previous["z"] = note["a"]
            last_key[identity] = note
    if not kept:
        raise ValueError("direct notation has no notes after vocal cleanup")

    shared_ends = [n.end_sec for n in analysis.note_events]
    shared_ends.extend(float(n["end_sec"]) for n in analysis.metadata.get("shared_timeline_event_bounds", []))
    total = max(max(n["z"] for n in kept), snap(max(shared_ends, default=0)))
    total = math.ceil(total / bar_ticks) * bar_ticks
    changes: dict[int, str] = {}
    for change in opts.key_changes:
        t = (change.bar - 1) * bar_ticks
        if t >= total or t in changes:
            raise ValueError("key changes must use distinct bars within the score")
        changes[t] = change.key

    attacks: dict[tuple[str | None, int], list[dict[str, Any]]] = defaultdict(list)
    for note in kept:
        attacks[(note["stem"], note["a"])].append(note)
    grouped: dict[tuple[str | None, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (stem, _a), chord in attacks.items():
        top = max(n["p"] for n in chord)
        for n in chord:
            role = "人声" if mode == "vocal" else "高音" if n["p"] == top and n["p"] >= opts.melody_min_midi else "低音" if n["p"] < opts.bass_split_midi else "和声"
            key = analysis.key
            for t, value in sorted(changes.items()):
                if t <= n["a"]:
                    key = value
            if role == "和声" and midi_to_jianpu(n["p"], key).startswith(("#", "b")):
                role = "和声变音"
            grouped[(stem, role, n["a"], n["z"])].append(n)

    voices: list[ScoreVoice] = []
    stems = sorted({n["stem"] or "" for n in kept})
    for stem in stems:
        for role in ("人声", "高音", "和声", "和声变音", "低音"):
            lanes: list[list[ScoreNote]] = []
            for (s, r, a, z), chord in sorted(grouped.items(), key=lambda item: (item[0][2], -max(n["p"] for n in item[1]))):
                if (s or "") != stem or r != role:
                    continue
                lane = next((i for i, events_ in enumerate(lanes) if events_[-1].end_tick <= a), len(lanes))
                if lane == len(lanes):
                    lanes.append([])
                voice_id = f"{stem or 'direct'}:{role}:{lane}"
                pitches = sorted(n["p"] for n in chord)
                lanes[lane].append(ScoreNote(start_tick=a, duration_tick=z-a, midi=pitches[0],
                    chord_pitches=pitches if len(pitches)>1 else [], voice_id=voice_id, source=DIRECT_ENGINE,
                    velocity=chord[0]["velocity"], stem_id=stem or None,
                    metadata={"source_indices": sorted(i for n in chord for i in n["ids"])}))
            for i, lane in enumerate(lanes):
                filled: list[ScoreNote] = []
                cursor = 0
                for n in lane:
                    if cursor < n.start_tick:
                        filled.append(ScoreNote(start_tick=cursor, duration_tick=n.start_tick-cursor, voice_id=n.voice_id))
                    filled.append(n)
                    cursor = n.end_tick
                if cursor < total:
                    filled.append(ScoreNote(start_tick=cursor, duration_tick=total-cursor, voice_id=lane[0].voice_id))
                label = role + (str(i+1) if len(lanes)>1 else "")
                voices.append(ScoreVoice(voice_id=lane[0].voice_id, label=label, stem_id=stem or None, events=filled))
    tempos: dict[int, float] = {}
    for t, bpm in _tempo_points_for_mapper(mapper, TPQ):
        t = round(t/divisor)
        if t < total:
            tempos[t] = bpm/divisor
    report = {"engine": DIRECT_ENGINE, "mode": mode, "options": opts.model_dump(mode="json"),
              "source_note_count": len(events), "accounted_source_count": len(events), "unresolved_count": 0,
              "output_note_count": len(kept), "actions": actions,
              "quantized_intervals": [[n["p"], n["a"], n["z"]] for n in kept]}
    score = Score(title=title, bpm=analysis.bpm/divisor, key=changes.get(0, analysis.key),
        time_signature=analysis.time_signature, quarter_ticks=TPQ, total_ticks=total, voices=voices,
        tempo_events=[TempoEvent(start_tick=t, bpm=b) for t,b in sorted(tempos.items())], source=DIRECT_ENGINE,
        warnings=[*analysis.warnings, "直接记谱按规则整理；三连音、滑音、踏板及声部分配仍需复核。"],
        metadata={"notation_engine": DIRECT_ENGINE, "direct_notation": report,
                  "score_origin": mapper.score_origin,
                  "key_signature_events": [{"start_tick": t, "key": key} for t,key in sorted(changes.items())]})
    return score, report


def combine_direct_scores(parts: list[tuple[str, str, Score]], *, title: str) -> tuple[Score, dict[str, Any]]:
    """Concatenate the chosen instrument voices without reselecting a melody."""
    if not parts:
        raise ValueError("no direct scores to combine")
    reference = parts[0][2]
    voices = []
    for track_id, label, score in parts:
        if (score.quarter_ticks, score.total_ticks, score.time_signature, score.key, score.tempo_events,
            score.metadata.get("key_signature_events")) != (
            reference.quarter_ticks, reference.total_ticks, reference.time_signature, reference.key,
            reference.tempo_events, reference.metadata.get("key_signature_events")):
            raise ValueError("selected direct scores must share a timeline, tempo and key map")
        for voice in score.voices:
            voice_id = f"{track_id}:{voice.voice_id}"
            voices.append(voice.model_copy(update={"voice_id": voice_id,
                "label": f"{label} {voice.label}" if len(parts)>1 else voice.label,
                "events": [n.model_copy(update={"voice_id": voice_id}) for n in voice.events]}))
    report = {"engine": DIRECT_ENGINE, "status": "completed", "policy": "preserve_direct_voices",
              "selected_track_ids": [track_id for track_id, _label, _score in parts]}
    return reference.model_copy(update={"title": title, "voices": voices,
        "metadata": {**reference.metadata, "direct_composition": report}}), report
