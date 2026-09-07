"""Create deterministic, local benchmark fixtures for the high accuracy chain.

The repository deliberately keeps generated audio and MIDI below ``.cache``.
This script only creates those rebuildable files; it never downloads a corpus
and never treats a reference MIDI file as a model recognition result.  The
``--case-id`` option is useful for a small smoke run before the full benchmark
is authorized.  Synthetic fixtures use deterministic notated downbeat
velocity accents and a modest additive harmonic envelope; pitch and timing are
unchanged from the fixture recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import wave
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_ROOT = ROOT / ".cache" / "high-accuracy-benchmarks" / "generated"
GENERATOR_VERSION = "1.1"
DEFAULT_SEED = 20260907
SAMPLE_RATE = 16_000
PPQ = 480
DOWNBEAT_ACCENT_DELTA = 24
RENDERER_VERSION = "deterministic_harmonic_oscillator_v1"


@dataclass(frozen=True)
class RenderNote:
    start: Fraction
    end: Fraction
    pitch: int
    velocity: int = 80
    channel: int = 0


@dataclass(frozen=True)
class RenderTrack:
    name: str
    program: int
    notes: tuple[RenderNote, ...]
    channel: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _q(value: int | float | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _frequency(pitch: int) -> float:
    return 440.0 * (2.0 ** ((int(pitch) - 69) / 12.0))


def _tempo_seconds(points: Sequence[tuple[Fraction, float]], beat: Fraction) -> float:
    """Convert a quarter-note position to seconds using a tempo map."""

    if not points:
        points = ((Fraction(0), 120.0),)
    ordered = sorted(points)
    elapsed = 0.0
    previous_q = ordered[0][0]
    previous_bpm = float(ordered[0][1])
    if beat <= previous_q:
        return float(beat - previous_q) * 60.0 / previous_bpm
    for change_q, bpm in ordered[1:]:
        if change_q >= beat:
            break
        if change_q > previous_q:
            elapsed += float(change_q - previous_q) * 60.0 / previous_bpm
        previous_q = change_q
        previous_bpm = float(bpm)
    elapsed += float(beat - previous_q) * 60.0 / previous_bpm
    return elapsed


def _midi_events(tracks: Sequence[RenderTrack], tempo: Sequence[tuple[Fraction, float]], meter: tuple[int, int], key: str) -> mido.MidiFile:
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="benchmark conductor", time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1], clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    first_tempo = sorted(tempo)[0] if tempo else (Fraction(0), 120.0)
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(first_tempo[1]), time=0))
    last_tick = 0
    for change_q, bpm in sorted(tempo)[1:]:
        tick = round(float(change_q) * PPQ)
        conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=max(0, tick - last_tick)))
        last_tick = tick
    conductor.append(mido.MetaMessage("key_signature", key=key, time=0))
    midi.tracks.append(conductor)
    for track in tracks:
        output = mido.MidiTrack()
        output.append(mido.MetaMessage("track_name", name=track.name, time=0))
        output.append(mido.Message("program_change", channel=track.channel, program=track.program, time=0))
        events: list[tuple[int, int, mido.Message]] = []
        for note in track.notes:
            start = max(0, round(float(note.start) * PPQ))
            end = max(start + 1, round(float(note.end) * PPQ))
            on = mido.Message("note_on", channel=track.channel, note=note.pitch, velocity=max(1, min(127, note.velocity)), time=0)
            off = mido.Message("note_off", channel=track.channel, note=note.pitch, velocity=0, time=0)
            events.extend(((start, 1, on), (end, 0, off)))
        events.sort(key=lambda item: (item[0], item[1], item[2].note if hasattr(item[2], "note") else -1))
        last_tick = 0
        for tick, _priority, message in events:
            message.time = max(0, tick - last_tick)
            output.append(message)
            last_tick = tick
        midi.tracks.append(output)
    return midi


def _harmonic_weights(program: int) -> tuple[tuple[int, float], ...]:
    """Return a small deterministic additive timbre for one MIDI program."""

    if 24 <= program <= 31:  # nylon/acoustic/electric guitar family
        return ((1, 1.0), (2, 0.34), (3, 0.16), (4, 0.06))
    if 32 <= program <= 39:  # bass family
        return ((1, 1.0), (2, 0.24), (3, 0.08))
    if 0 <= program <= 7:  # piano family
        return ((1, 1.0), (2, 0.22), (3, 0.10), (4, 0.04))
    return ((1, 1.0), (2, 0.28), (3, 0.12), (4, 0.04))


def _accent_downbeats(tracks: Sequence[RenderTrack], meter: tuple[int, int]) -> tuple[RenderTrack, ...]:
    """Raise only notated bar starts while retaining every pitch and boundary."""

    bar_quarters = Fraction(meter[0] * 4, meter[1])
    accented: list[RenderTrack] = []
    for track in tracks:
        notes = tuple(
            replace(
                note,
                velocity=min(112, max(1, int(note.velocity)) + DOWNBEAT_ACCENT_DELTA),
            )
            if note.start % bar_quarters == 0
            else note
            for note in track.notes
        )
        accented.append(replace(track, notes=notes))
    return tuple(accented)


def _render_audio(path: Path, tracks: Sequence[RenderTrack], tempo: Sequence[tuple[Fraction, float]], *, seed: int) -> None:
    notes = [note for track in tracks for note in track.notes]
    if not notes:
        raise ValueError("cannot render an empty benchmark fixture")
    end_q = max(note.end for note in notes) + Fraction(1, 2)
    duration = max(0.25, _tempo_seconds(tempo, end_q))
    frame_count = int(math.ceil(duration * SAMPLE_RATE))
    samples = [0.0] * frame_count
    rng = random.Random(seed)
    # A tiny deterministic instrument-dependent detune makes the rendered
    # samples audibly distinct while preserving the exact MIDI reference.
    detune = (rng.random() - 0.5) * 0.002
    for track_index, track in enumerate(tracks):
        weights = _harmonic_weights(track.program)
        weight_total = sum(weight for _harmonic, weight in weights)
        for note in track.notes:
            start_sec = max(0.0, _tempo_seconds(tempo, note.start))
            end_sec = max(start_sec + 1.0 / SAMPLE_RATE, _tempo_seconds(tempo, note.end))
            start_frame = max(0, int(start_sec * SAMPLE_RATE))
            end_frame = min(frame_count, int(math.ceil(end_sec * SAMPLE_RATE)))
            frequency = _frequency(note.pitch) * (1.0 + detune * (track_index + 1))
            amplitude = 0.12 * (max(1, min(127, note.velocity)) / 127.0) / max(1.0, math.sqrt(len(tracks)))
            attack = max(1, int(0.006 * SAMPLE_RATE))
            release = max(1, int(0.030 * SAMPLE_RATE))
            for frame in range(start_frame, end_frame):
                local = frame - start_frame
                remaining = end_frame - frame
                envelope = min(1.0, local / attack, remaining / release)
                phase = 2.0 * math.pi * frequency * (frame / SAMPLE_RATE)
                tone = sum(weight * math.sin(harmonic * phase) for harmonic, weight in weights) / weight_total
                samples[frame] += amplitude * envelope * tone
    peak = max(1.0, max(abs(value) for value in samples) * 1.02)
    pcm = b"".join(struct.pack("<h", max(-32767, min(32767, round(value / peak * 32767)))) for value in samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def _beat_annotation(path: Path, *, tempo: Sequence[tuple[Fraction, float]], meter: tuple[int, int], end_q: Fraction, source: str) -> None:
    # For compound 6/8 keep the notated eighth-note beat positions.  The
    # selected downbeat list remains explicit, so readers never infer every
    # beat to be a downbeat.
    beat_step = Fraction(1, 2) if meter == (6, 8) else Fraction(1)
    beats: list[dict[str, Any]] = []
    index = 0
    current = Fraction(0)
    beats_per_bar = 6 if meter == (6, 8) else meter[0]
    while current <= end_q:
        bar_index, beat_index = divmod(index, beats_per_bar)
        downbeat = beat_index == 0
        beats.append({"index": index, "bar_index": bar_index, "beat_index": beat_index, "time_sec": _tempo_seconds(tempo, current), "downbeat": downbeat})
        index += 1
        current += beat_step
    _atomic_json(path, {"schema_version": "1.0", "source": source, "beat_grid": {"beats": beats, "downbeats": [item for item in beats if item["downbeat"]], "time_signature": f"{meter[0]}/{meter[1]}", "annotation_policy": "derived_from_reference_midi"}})


def _melody(root: int, length: int = 16, *, step: Fraction = Fraction(1)) -> tuple[RenderNote, ...]:
    scale = (0, 2, 4, 5, 7, 9, 11, 12)
    return tuple(RenderNote(index * step, (index + 1) * step, root + scale[index % len(scale)], 82) for index in range(length))


def _chords(root: int, bars: int = 4) -> tuple[RenderNote, ...]:
    triads = ((0, 4, 7), (5, 9, 12), (7, 11, 14), (0, 4, 7))
    notes: list[RenderNote] = []
    for bar in range(bars):
        start = Fraction(bar * 4)
        for pitch in triads[bar % len(triads)]:
            notes.append(RenderNote(start, start + 4, root + pitch, 72))
    return tuple(notes)


def _spec_for(case_id: str) -> tuple[tuple[RenderTrack, ...], tuple[tuple[Fraction, float], ...], tuple[int, int], str]:
    if case_id.startswith("synthetic-piano"):
        variant = int(case_id.rsplit("-", 1)[-1])
        melody = _melody(60 + variant - 1, 16 if variant != 3 else 24)
        tracks = (RenderTrack("piano", 0, tuple((*melody, *_chords(48 + variant)))),)
        return tracks, ((Fraction(0), 108.0 + variant * 6),), (4, 4), "C"
    if case_id.startswith("synthetic-guitar"):
        variant = int(case_id.rsplit("-", 1)[-1])
        notes = tuple(RenderNote(Fraction(index), Fraction(index + 1), 55 + ((index * (2 + variant)) % 12), 76) for index in range(16))
        return (RenderTrack("guitar", 24, notes),), ((Fraction(0), 96.0 + variant * 12),), (4, 4), "G"
    if case_id.startswith("synthetic-bass"):
        variant = int(case_id.rsplit("-", 1)[-1])
        notes = tuple(RenderNote(Fraction(index * 2), Fraction(index * 2 + 1), 36 + ((index + variant) % 5) * 2, 88) for index in range(8))
        return (RenderTrack("bass", 32, notes),), ((Fraction(0), 84.0 + variant * 8),), (4, 4), "Am"
    if case_id.startswith("synthetic-multitrack"):
        variant = int(case_id.rsplit("-", 1)[-1])
        piano = RenderTrack("piano", 0, _chords(48 + variant))
        lead = RenderTrack("lead", 40, _melody(67, 16))
        bass = RenderTrack("bass", 32, tuple(RenderNote(Fraction(i * 2), Fraction(i * 2 + 2), 36 + (i % 4) * 2, 84) for i in range(8)), channel=1)
        tracks = (piano, lead, bass) if variant >= 2 else (piano, lead)
        return tracks, ((Fraction(0), 120.0),), (4, 4), "D"
    if case_id == "special-pickup-3-4":
        notes = (RenderNote(Fraction(1, 2), Fraction(1), 67),) + _melody(60, 9)
        return (RenderTrack("pickup", 0, notes),), ((Fraction(0), 100.0),), (3, 4), "C"
    if case_id == "special-6-8":
        notes = tuple(RenderNote(Fraction(i, 2), Fraction(i + 1, 2), 60 + (i % 6), 80) for i in range(24))
        return (RenderTrack("compound", 0, notes),), ((Fraction(0), 90.0),), (6, 8), "F"
    if case_id == "special-triplet":
        notes = tuple(RenderNote(Fraction(i, 3), Fraction(i + 1, 3), 72 + (i % 3), 82) for i in range(12))
        return (RenderTrack("triplet", 40, notes),), ((Fraction(0), 110.0),), (4, 4), "G"
    if case_id == "special-tempo-change":
        notes = _melody(60, 16)
        return (RenderTrack("tempo change", 0, notes),), ((Fraction(0), 72.0), (Fraction(8), 132.0)), (4, 4), "Am"
    if case_id == "special-complex-chord":
        first = tuple(RenderNote(Fraction(0), Fraction(2), pitch, 76) for pitch in (48, 52, 55, 59))
        second = tuple(RenderNote(Fraction(2), Fraction(4), pitch, 76) for pitch in (50, 53, 57, 60))
        return (RenderTrack("complex chords", 0, (*first, *second)),), ((Fraction(0), 104.0),), (3, 4), "Eb"
    raise ValueError(f"no deterministic fixture recipe for {case_id!r}")


def _case_ids_from_registry(registry_path: Path) -> list[str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return [str(item["id"]) for item in payload.get("cases", []) if item.get("source_kind") == "synthetic"]


def generate_case(case_id: str, *, destination: Path = DEFAULT_ROOT, seed: int = DEFAULT_SEED, overwrite: bool = False) -> dict[str, Any]:
    tracks, tempo, meter, key = _spec_for(case_id)
    tracks = _accent_downbeats(tracks, meter)
    case_root = (destination / case_id).resolve()
    if case_root.exists() and not overwrite:
        existing = case_root / "case_manifest.json"
        if existing.is_file():
            return json.loads(existing.read_text(encoding="utf-8"))
        raise FileExistsError(f"fixture directory exists without a manifest: {case_root}")
    case_root.mkdir(parents=True, exist_ok=True)
    midi_path = case_root / f"{case_id}.mid"
    audio_path = case_root / f"{case_id}.wav"
    beat_path = case_root / f"{case_id}.beat_grid.json"
    midi = _midi_events(tracks, tempo, meter, key)
    midi.save(midi_path)
    _render_audio(audio_path, tracks, tempo, seed=seed + sum(ord(char) for char in case_id))
    end_q = max(note.end for track in tracks for note in track.notes)
    _beat_annotation(beat_path, tempo=tempo, meter=meter, end_q=end_q, source="generated_from_reference_midi")
    files = {name: {"path": file.name, "bytes": file.stat().st_size, "sha256": _sha256(file)} for name, file in (("reference_midi", midi_path), ("input_audio", audio_path), ("beat_annotation", beat_path))}
    manifest = {
        "schema_version": "1.0",
        "generator_version": GENERATOR_VERSION,
        "renderer_version": RENDERER_VERSION,
        "case_id": case_id,
        "seed": seed,
        "source_kind": "synthetic",
        "evaluation_scope": "quantizer_isolation_fixture",
        "tracks": [{"name": track.name, "program": track.program, "channel": track.channel, "note_count": len(track.notes)} for track in tracks],
        "velocity_policy": {
            "kind": "deterministic_notated_downbeat_accents",
            "accent_delta": DOWNBEAT_ACCENT_DELTA,
            "max_velocity": 112,
            "pitch_and_timing_unchanged": True,
        },
        "tempo_map": [{"quarter": str(position), "bpm": bpm} for position, bpm in tempo],
        "time_signature": f"{meter[0]}/{meter[1]}",
        "key": key,
        "files": files,
        "license": "project-generated-deterministic-fixture",
        "source": {"url": None, "license": "project-generated-deterministic-fixture"},
    }
    _atomic_json(case_root / "case_manifest.json", manifest)
    return manifest


def generate_reference_derived_beat(case: Mapping[str, Any], *, destination: Path = DEFAULT_ROOT) -> Path | None:
    """Create a transparent beat annotation from an existing reference MIDI.

    This is useful for the PJS smoke set, but it is marked reference-derived
    and must not be presented as an independently annotated BeatNet benchmark.
    """

    reference = Path(str(case.get("reference_midi", "")))
    if not reference.is_absolute():
        reference = ROOT / reference
    if not reference.is_file():
        return None
    midi = mido.MidiFile(reference)
    tempo = 120.0
    meter = (4, 4)
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = float(mido.tempo2bpm(message.tempo))
            elif message.type == "time_signature":
                meter = (int(message.numerator), int(message.denominator))
    end_tick = max((sum(int(message.time) for message in track) for track in midi.tracks), default=0)
    end_q = Fraction(end_tick, midi.ticks_per_beat)
    output = destination / "pjs" / f"{case['id']}.beat_grid.json"
    _beat_annotation(output, tempo=((Fraction(0), tempo),), meter=meter, end_q=end_q, source="reference_midi_derived_for_smoke_only")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-id", action="append", dest="case_ids", help="只生成指定 case；可重复")
    parser.add_argument("--limit", type=int, help="只生成清单中前 N 个 synthetic case")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    registry_payload = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    requested = list(args.case_ids or [])
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be non-negative")
        requested.extend(_case_ids_from_registry(args.manifest.resolve())[: args.limit])
    if not requested:
        requested = _case_ids_from_registry(args.manifest.resolve())
    known = {str(item["id"]): item for item in registry_payload.get("cases", [])}
    for case_id in requested:
        if case_id in known and known[case_id].get("source_kind") == "vocal":
            generate_reference_derived_beat(known[case_id], destination=args.output_root.resolve())
        elif case_id in known and known[case_id].get("source_kind") == "synthetic":
            generate_case(case_id, destination=args.output_root.resolve(), seed=args.seed, overwrite=args.overwrite)
        else:
            raise SystemExit(f"case {case_id!r} is not a local synthetic or reference-derived case")
    print(json.dumps({"generated": requested, "output_root": str(args.output_root.resolve()), "seed": args.seed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
