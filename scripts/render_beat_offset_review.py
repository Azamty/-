"""Render a late first beat and verify the audio-zero onset in MIDI."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.quantize import quantize_events
from backend.jianpu_score.render import render_score


def main() -> int:
    output = ROOT / "artifacts" / "review" / "stage2-beat-offset"
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        beat_times=[0.25, 0.75, 1.5, 1.75],
        metadata={"beat_source": "librosa"},
    )
    source_events = [
        NoteEvent(start_sec=0.0, end_sec=0.2, midi=60, source="review"),
        NoteEvent(start_sec=0.25, end_sec=0.75, midi=62, source="review"),
    ]
    score = quantize_events(source_events, analysis, mode="monophonic", title="beat offset")
    artifacts = render_score(score, output, basename="offset")
    midi = mido.MidiFile(artifacts.midi_path)
    starts: list[tuple[int, int]] = []
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                starts.append((absolute, message.note))
    starts.sort()
    expected = [(0, 60), (192, 62)]
    if starts != expected:
        raise RuntimeError(f"beat-zero MIDI onset mismatch: {starts} != {expected}")
    absolute = 0
    tempo_map: dict[int, float] = {}
    for message in midi.tracks[0]:
        absolute += message.time
        if message.type == "set_tempo":
            tempo_map[absolute] = 60_000_000 / message.tempo
    expected_tempo = {0: 120.0, 576: 80.0, 960: 240.0}
    if any(abs(tempo_map.get(tick, -1) - bpm) > 0.01 for tick, bpm in expected_tempo.items()):
        raise RuntimeError(f"shifted MIDI tempo mismatch: {tempo_map} != {expected_tempo}")
    manifest = {
        "source_events": [event.model_dump(mode="json") for event in source_events],
        "quantized_notes": [event.model_dump(mode="json") for voice in score.voices for event in voice.events if event.midi is not None],
        "beat_shift_beats": score.metadata["beat_shift_beats"],
        "downbeat_status": score.metadata["downbeat_status"],
        "tempo_events": [event.model_dump(mode="json") for event in score.tempo_events],
        "midi_tempo_map": tempo_map,
        "midi_note_starts": starts,
        "svg_paths": artifacts.svg_paths,
        "midi_path": artifacts.midi_path,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
