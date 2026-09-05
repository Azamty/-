"""V2 MuScriptor routing and post-transcription selection policy.

The model always receives the complete instrumental mix.  Instrument choices
are applied after full decoding so the UI can offer the model's complete
instrument inventory without running one inference per selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import mido


SourceKind = Literal["instrumental", "vocal"]
EngineName = Literal["muscriptor", "game"]
DRUMS = "drums"

# These labels are deliberately kept in the backend contract so a future UI
# does not need to invent translations for model vocabulary.  Unknown model
# groups remain readable through the underscore-to-space fallback.
INSTRUMENT_LABELS_ZH: dict[str, str] = {
    "acoustic_piano": "原声钢琴",
    "electric_piano": "电钢琴",
    "chromatic_percussion": "固定音高打击乐",
    "organ": "风琴",
    "acoustic_guitar": "原声吉他",
    "clean_electric_guitar": "清音电吉他",
    "distorted_electric_guitar": "失真电吉他",
    "acoustic_bass": "原声贝斯",
    "electric_bass": "电贝斯",
    "violin": "小提琴",
    "viola": "中提琴",
    "cello": "大提琴",
    "contrabass": "低音提琴",
    "orchestral_harp": "竖琴",
    "timpani": "定音鼓",
    "string_ensemble": "弦乐组",
    "synth_strings": "合成弦乐",
    "voice": "人声",
    "trumpet": "小号",
    "trombone": "长号",
    "tuba": "大号",
    "french_horn": "圆号",
    "soprano_and_alto_sax": "高音/中音萨克斯",
    "tenor_sax": "次中音萨克斯",
    "baritone_sax": "上低音萨克斯",
    "oboe": "双簧管",
    "english_horn": "英国管",
    "bassoon": "巴松管",
    "clarinet": "单簧管",
    "flutes": "长笛",
    "synth_lead": "合成主音",
    "synth_pad": "合成铺底",
    DRUMS: "鼓组",
}


def instrument_label_zh(instrument_group: str) -> str:
    """Return the stable Chinese label exposed by the V2 API."""

    return INSTRUMENT_LABELS_ZH.get(
        instrument_group,
        instrument_group.replace("_", " ").strip() or "未知乐器",
    )


def stable_track_id(instrument_group: str, program: int, is_drum: bool) -> str:
    """Derive a deterministic track id from model identity, never selection."""

    identity = f"{instrument_group}|{int(program)}|{int(bool(is_drum))}".encode("utf-8")
    return "track-" + hashlib.sha1(identity).hexdigest()[:12]


def write_unquantized_midi(
    notes: Iterable[Mapping[str, object]],
    destination: str | Path,
    *,
    title: str,
    ticks_per_beat: int = 960,
    bpm: float = 120.0,
) -> Path:
    """Write a selected-track MIDI from source seconds.

    Jianpu quantization never enters this path.  A fixed playback velocity is
    used only because MIDI playback requires one; source NoteEvents keep their
    velocity as ``None`` and the caller records this policy in metadata.
    """

    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_title = str(title).encode("ascii", "replace").decode("ascii")
    tempo = int(round(60_000_000 / float(bpm)))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name=midi_title))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    midi.tracks.append(tempo_track)
    by_track: dict[tuple[str, int, bool], list[Mapping[str, object]]] = {}
    for note in notes:
        group = str(note.get("instrument_group", "unknown"))
        program = int(note.get("program", 0))
        is_drum = bool(note.get("is_drum", False))
        by_track.setdefault((group, program, is_drum), []).append(note)
    for (group, program, is_drum), track_notes in sorted(by_track.items()):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=group.replace("_", " "), time=0))
        channel = 9 if is_drum else min(15, max(0, len(midi.tracks) - 1))
        if not is_drum:
            track.append(mido.Message("program_change", channel=channel, program=max(0, min(127, program)), time=0))
        events: list[tuple[int, int, mido.Message]] = []
        for note in track_notes:
            start = max(0.0, float(note["start_sec"]))
            end = max(start + 0.001, float(note["end_sec"]))
            pitch = max(0, min(127, int(note["pitch"])))
            start_tick = int(round(mido.second2tick(start, ticks_per_beat, tempo)))
            end_tick = max(start_tick + 1, int(round(mido.second2tick(end, ticks_per_beat, tempo))))
            events.append((start_tick, 1, mido.Message("note_on", channel=channel, note=pitch, velocity=80, time=0)))
            events.append((end_tick, 0, mido.Message("note_off", channel=channel, note=pitch, velocity=0, time=0)))
        previous = 0
        for tick, _priority, message in sorted(events, key=lambda item: (item[0], item[1])):
            message.time = max(0, tick - previous)
            track.append(message)
            previous = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(track)
    midi.save(path)
    return path


@dataclass(frozen=True)
class MuscriptorNote:
    """A normalized MuScriptor event suitable for policy and artifact code."""

    instrument: str
    pitch: int
    start_sec: float
    end_sec: float

    @property
    def is_drum(self) -> bool:
        return self.instrument == DRUMS


@dataclass(frozen=True)
class TranscriptionPlan:
    """The V2 choices that follow a complete model decode."""

    source_kind: SourceKind
    engine: EngineName
    use_demucs: bool
    detected_instruments: tuple[str, ...]
    selected_pitched_instruments: tuple[str, ...]
    drum_preview_enabled: bool
    merge_main_melody: bool

    @property
    def score_instruments(self) -> tuple[str, ...]:
        """Pitched instruments eligible for jianpu score generation."""

        return self.selected_pitched_instruments


def route_for_source(source_kind: str) -> tuple[EngineName, bool]:
    """Return the V2 engine and separation choice for an input source.

    V2 intentionally does not call Demucs.  MuScriptor receives an
    instrumental mix directly, while GAME handles a vocal source directly.
    """

    if source_kind == "instrumental":
        return "muscriptor", False
    if source_kind == "vocal":
        return "game", False
    raise ValueError("source_kind must be 'instrumental' or 'vocal'")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return tuple(result)


def make_transcription_plan(
    source_kind: str,
    detected_instruments: Iterable[str],
    selected_instruments: Sequence[str] | None = None,
    *,
    include_drums: bool = True,
    merge_main_melody: bool = False,
) -> TranscriptionPlan:
    """Build post-decode instrument and score selection.

    ``selected_instruments=None`` means every detected pitched instrument is
    selected.  Drums remain available for playback and MIDI, but are never in
    ``score_instruments``.
    """

    engine, use_demucs = route_for_source(source_kind)
    detected = _ordered_unique(detected_instruments)
    detected_pitched = tuple(name for name in detected if name != DRUMS)
    detected_has_drums = DRUMS in detected

    if selected_instruments is None:
        selected_pitched = detected_pitched
    else:
        requested = _ordered_unique(selected_instruments)
        unknown = tuple(name for name in requested if name not in detected)
        if unknown:
            raise ValueError(
                "selected instruments were not detected: " + ", ".join(unknown)
            )
        selected_pitched = tuple(name for name in requested if name != DRUMS)

    return TranscriptionPlan(
        source_kind=source_kind,  # type: ignore[arg-type]
        engine=engine,
        use_demucs=use_demucs,
        detected_instruments=detected,
        selected_pitched_instruments=selected_pitched,
        drum_preview_enabled=bool(include_drums and detected_has_drums),
        merge_main_melody=merge_main_melody,
    )


def partition_notes(
    notes: Iterable[MuscriptorNote], plan: TranscriptionPlan
) -> Mapping[str, tuple[MuscriptorNote, ...]]:
    """Partition full decode events into score stems and optional drum MIDI."""

    partitions: dict[str, list[MuscriptorNote]] = {
        name: [] for name in plan.selected_pitched_instruments
    }
    if plan.drum_preview_enabled:
        partitions[DRUMS] = []
    for note in notes:
        if note.instrument == DRUMS:
            if plan.drum_preview_enabled:
                partitions[DRUMS].append(note)
        elif note.instrument in partitions:
            partitions[note.instrument].append(note)
    return {name: tuple(values) for name, values in partitions.items()}
