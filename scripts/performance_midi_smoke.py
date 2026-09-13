"""Build and parse one high-accuracy performance MIDI without the API path."""

from __future__ import annotations

from io import BytesIO
import json
import sys

import mido

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.performance_midi import PERFORMANCE_TICKS_PER_QUARTER, build_performance_midi


def main() -> int:
    beat_times = [0.0, 0.5, 1.25, 1.75, 2.25]
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.5,
        bpm=120.0,
        key="D",
        time_signature="4/4",
        beat_times=beat_times,
        metadata={
            "beat_source": "beatnet",
            "beat_engine": "beatnet",
            "beatnet_version": "1.1.3",
            "beat_grid": {
                "mapping": {
                    "manual_bpm_scale": 1.0,
                    "score_origin": {"downbeat_index": 0, "downbeat_sec": 0.0},
                },
                "beats": [{"index": index, "time_sec": value, "downbeat": index == 0} for index, value in enumerate(beat_times)],
            },
        },
    )
    midi_bytes, metadata = build_performance_midi(
        [
            NoteEvent(start_sec=0.13, end_sec=0.61, midi=60),
            NoteEvent(start_sec=0.13, end_sec=0.61, midi=64),
            NoteEvent(start_sec=0.75, end_sec=1.63, midi=67),
        ],
        analysis,
        instrument_group="acoustic_piano",
        program=0,
        title="Performance smoke",
    )
    midi = mido.MidiFile(file=BytesIO(midi_bytes))
    if midi.ticks_per_beat != PERFORMANCE_TICKS_PER_QUARTER or len(midi.tracks) != 2:
        raise RuntimeError("performance MIDI header/track contract failed")
    messages = list(mido.merge_tracks(midi.tracks))
    note_on_count = sum(message.type == "note_on" and message.velocity > 0 for message in messages)
    if note_on_count != 3:
        raise RuntimeError(f"performance MIDI note count mismatch: {note_on_count}")
    if not any(message.type == "set_tempo" for message in messages):
        raise RuntimeError("performance MIDI has no tempo map")
    print(
        json.dumps(
            {
                "status": "ok",
                "ticks_per_quarter": midi.ticks_per_beat,
                "track_count": len(midi.tracks),
                "note_on_count": note_on_count,
                "tempo_points": metadata["tempo_points"],
                "score_origin": metadata["score_origin"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
