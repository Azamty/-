"""Render C major and minor key examples and verify MIDI pitches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.quantize import midi_to_jianpu
from backend.jianpu_score.render import render_score


CASES = {
    "C": ([60, 62, 64, 65, 67, 69, 71, 72], "1=C"),
    "Am": ([69, 71, 72, 74, 76, 77, 79, 81], "1=C"),
    "F#m": ([66, 68, 69, 71, 73, 74, 76, 78], "1=A"),
    "Cm": ([60, 62, 63, 65, 67, 68, 70, 72], "1=Eb"),
}


def render_case(key: str, pitches: list[int], expected_header: str) -> dict[str, object]:
    destination = ROOT / "artifacts" / "review" / "stage2-keys" / key.replace("#", "s")
    score = Score(
        title=f"Key {key}",
        bpm=80,
        key=key,
        time_signature="4/4",
        total_ticks=96,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[ScoreNote(start_tick=index * 12, duration_tick=12, midi=pitch) for index, pitch in enumerate(pitches)],
            )
        ],
    )
    artifacts = render_score(score, destination, basename="score")
    jly = Path(artifacts.jly_path).read_text(encoding="utf-8")
    midi = mido.MidiFile(artifacts.midi_path)
    output_pitches = [message.note for track in midi.tracks for message in track if message.type == "note_on" and message.velocity > 0]
    if output_pitches != pitches:
        raise RuntimeError(f"MIDI pitch mismatch for {key}: {output_pitches} != {pitches}")
    if expected_header not in jly:
        raise RuntimeError(f"missing safe key header for {key}: {jly}")
    expected_degrees = [midi_to_jianpu(pitch, key) for pitch in pitches]
    if not all(f"{degree} " in jly or jly.rstrip().endswith(degree) for degree in expected_degrees):
        raise RuntimeError(f"missing relative-major degrees for {key}: {expected_degrees}; jly={jly}")
    return {
        "key": key,
        "expected_header": expected_header,
        "expected_degrees": expected_degrees,
        "midi_pitches": output_pitches,
        "svg_paths": artifacts.svg_paths,
        "midi_path": artifacts.midi_path,
        "jly_path": artifacts.jly_path,
    }


def main() -> int:
    manifest = {key: render_case(key, pitches, expected_header) for key, (pitches, expected_header) in CASES.items()}
    destination = ROOT / "artifacts" / "review" / "stage2-keys"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
