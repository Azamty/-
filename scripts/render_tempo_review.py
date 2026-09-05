"""Verify that Score tempo events survive the LilyPond MIDI render."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice, TempoEvent
from backend.jianpu_score.render import render_score


def main() -> int:
    output = ROOT / "artifacts" / "review" / "stage2-tempo"
    score = Score(
        title="tempo map",
        bpm=120,
        key="C",
        time_signature="4/4",
        total_ticks=48,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[ScoreNote(start_tick=index * 12, duration_tick=12, midi=60 + index) for index in range(4)],
            )
        ],
        tempo_events=[TempoEvent(start_tick=0, bpm=120), TempoEvent(start_tick=12, bpm=80), TempoEvent(start_tick=24, bpm=120)],
    )
    artifacts = render_score(score, output, basename="tempo")
    midi = mido.MidiFile(artifacts.midi_path)
    absolute = 0
    tempo_map: dict[int, float] = {}
    for message in midi.tracks[0]:
        absolute += message.time
        if message.type == "set_tempo":
            tempo_map[absolute] = 60_000_000 / message.tempo
    expected = {0: 120.0, 384: 80.0, 768: 120.0}
    if any(abs(tempo_map.get(tick, -1) - bpm) > 0.01 for tick, bpm in expected.items()):
        raise RuntimeError(f"MIDI tempo map mismatch: {tempo_map} != {expected}")
    manifest = {"score_tempo_events": [event.model_dump(mode="json") for event in score.tempo_events], "midi_tempo_map": tempo_map, "svg_paths": artifacts.svg_paths, "midi_path": artifacts.midi_path}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
