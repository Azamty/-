"""Verify same-bar accidental scope through jianpu-ly/LilyPond MIDI output."""

from __future__ import annotations

import json
from pathlib import Path

import mido

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.render import render_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "review" / "stage5-final" / "accidental-review"


def main() -> int:
    expected = [61, 60, 61, 60]
    score = Score(
        title="accidental scope",
        bpm=80,
        key="C",
        time_signature="4/4",
        quarter_ticks=12,
        total_ticks=48,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[
                    ScoreNote(start_tick=index * 12, duration_tick=12, midi=pitch)
                    for index, pitch in enumerate(expected)
                ],
            )
        ],
    )
    artifacts = render_score(score, OUTPUT, basename="score")
    jly = Path(artifacts.jly_path).read_text(encoding="utf-8")
    midi = mido.MidiFile(artifacts.midi_path)
    actual = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]
    if actual != expected:
        raise AssertionError(f"same-bar accidental changed MIDI pitches: {actual} != {expected}")
    manifest = {
        "jly_path": artifacts.jly_path,
        "midi_path": artifacts.midi_path,
        "svg_paths": artifacts.svg_paths,
        "jianpu": jly,
        "expected_midi_pitches": expected,
        "actual_midi_pitches": actual,
        "passed": True,
    }
    output = OUTPUT / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
