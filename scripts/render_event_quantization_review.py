"""Exercise real NoteEvents through beat mapping, tuplet quantization and render."""

from __future__ import annotations

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
    output = ROOT / "artifacts" / "review" / "stage2-triplet"
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=1.0,
        bpm=120,
        key="C",
        time_signature="4/4",
        beat_times=[0.0, 0.5, 1.0],
        metadata={"beat_source": "librosa"},
    )
    source_events = [
        NoteEvent(start_sec=0.0, end_sec=1 / 6, midi=60, source="review"),
        NoteEvent(start_sec=1 / 6, end_sec=2 / 6, midi=62, source="review"),
        NoteEvent(start_sec=2 / 6, end_sec=3 / 6, midi=64, source="review"),
    ]
    score = quantize_events(source_events, analysis, mode="polyphonic", title="real event triplet")
    artifacts = render_score(score, output, basename="triplet")
    jly = Path(artifacts.jly_path).read_text(encoding="utf-8")
    midi = mido.MidiFile(artifacts.midi_path)
    starts: list[tuple[int, int]] = []
    absolute = 0
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                starts.append((absolute, message.note))
    starts.sort()
    expected = [(0, 60), (128, 62), (256, 64)]
    if "3[" not in jly or starts != expected:
        raise RuntimeError(f"triplet render mismatch: jly={jly!r}, midi_starts={starts}")
    manifest = {
        "source_events": [event.model_dump(mode="json") for event in source_events],
        "quantized_events": [event.model_dump(mode="json") for event in score.voices[0].events if event.midi is not None],
        "triplet_group_count": score.metadata["triplet_group_count"],
        "jly_path": artifacts.jly_path,
        "svg_paths": artifacts.svg_paths,
        "midi_path": artifacts.midi_path,
        "midi_note_starts": starts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
