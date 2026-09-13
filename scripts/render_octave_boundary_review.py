"""Render low/high MIDI boundaries and record the exact MIDI round trip."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mido

ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "artifacts" / "review" / "luv-letter-octave-fix" / "boundary"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.render import render_score


def main() -> int:
    expected = [0, 22, 60, 127]
    score = Score(
        title="octave boundary review",
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
    artifacts = render_score(score, REVIEW_ROOT, basename="boundary")
    midi = mido.MidiFile(artifacts.midi_path)
    actual = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]
    if actual != expected:
        raise RuntimeError(f"octave MIDI round trip mismatch: {actual!r} != {expected!r}")
    jly = Path(artifacts.jly_path).read_text(encoding="utf-8")
    required_tokens = ["1,,,,,", "6,,,,", "1", "5'''''"]
    missing = [token for token in required_tokens if token not in jly]
    if missing:
        raise RuntimeError(f"octave JLY tokens missing: {missing!r}")
    manifest = {
        "status": "passed",
        "expected_midi_pitches": expected,
        "actual_midi_pitches": actual,
        "jianpu_tokens": required_tokens,
        "jly": str(Path(artifacts.jly_path).resolve()),
        "lilypond": str(Path(artifacts.lilypond_path).resolve()),
        "svg": [str(Path(path).resolve()) for path in artifacts.svg_paths],
        "midi": str(Path(artifacts.midi_path).resolve()),
    }
    (REVIEW_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
