"""Create the long synthetic scale used by the final local smoke test.

It has four musical sections, explicit rests, a low bass line and a marked
tail. The fixture is a pipeline scale test only; it says nothing about the
accuracy of Chinese, Japanese or mixed-language song transcription.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "review" / "stage5-final" / "long_fixture.wav"
METADATA = ROOT / "artifacts" / "review" / "stage5-final" / "long_fixture.json"
SAMPLE_RATE = 44_100
BPM = 80.0
BEAT_SECONDS = 60.0 / BPM
DURATION_SECONDS = 180.0


def _add_note(buffer: np.ndarray, start: float, duration: float, midi: int, amplitude: float) -> None:
    first = max(0, int(round(start * SAMPLE_RATE)))
    last = min(len(buffer), int(round((start + duration) * SAMPLE_RATE)))
    if last <= first:
        return
    time = np.arange(last - first, dtype=np.float32) / SAMPLE_RATE
    frequency = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    waveform = (
        np.sin(2 * np.pi * frequency * time)
        + 0.22 * np.sin(4 * np.pi * frequency * time)
        + 0.08 * np.sin(6 * np.pi * frequency * time)
    )
    attack = min(0.025, duration / 4.0)
    release = min(0.08, duration / 4.0)
    envelope = np.ones_like(time)
    if attack > 0:
        envelope *= np.minimum(1.0, time / attack)
    if release > 0:
        envelope *= np.minimum(1.0, np.maximum(0.0, (duration - time) / release))
    buffer[first:last] += amplitude * waveform * envelope


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(round(DURATION_SECONDS * SAMPLE_RATE))
    mono = np.zeros(sample_count, dtype=np.float32)

    # Bars 16-19, 36-39 and 52-55 are deliberate rests. The final four bars
    # carry a higher tail marker so a recognizer can be checked near 180 sec.
    active_bars = [*range(0, 16), *range(20, 36), *range(40, 52), *range(56, 60)]
    sections = [
        {"name": "A", "bars": [0, 16], "lead": [60, 62, 64, 65], "bass": 36},
        {"name": "B", "bars": [20, 36], "lead": [67, 65, 64, 62], "bass": 43},
        {"name": "C", "bars": [40, 52], "lead": [64, 67, 69, 72], "bass": 41},
        {"name": "tail", "bars": [56, 60], "lead": [72, 74, 76, 79], "bass": 36},
    ]

    for bar in active_bars:
        section = next(item for item in sections if item["bars"][0] <= bar < item["bars"][1])
        pattern = list(section["lead"])
        if section["name"] == "tail":
            pattern = [note + 12 for note in pattern]
        bar_start = bar * 4 * BEAT_SECONDS
        # A short gap between adjacent notes makes rests and note boundaries
        # explicit without turning this into a click track.
        for beat, midi in enumerate(pattern):
            _add_note(mono, bar_start + beat * BEAT_SECONDS, 0.62, int(midi), 0.25)
        _add_note(mono, bar_start, 2.55, int(section["bass"]), 0.18)
        # A quiet fifth and third provide an independent accompaniment layer.
        _add_note(mono, bar_start, 2.55, int(section["bass"]) + 7, 0.06)

    # An isolated high tail marker sits in the final second of the 180 second
    # file and is included in the fixture metadata for coverage checks.
    _add_note(mono, 179.0, 0.72, 84, 0.30)
    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono *= 0.85 / peak
    stereo = np.column_stack((mono, mono)).astype(np.float32)
    sf.write(OUTPUT, stereo, SAMPLE_RATE, subtype="PCM_16", format="WAV")

    metadata = {
        "kind": "synthetic_scale_pipeline_fixture",
        "accuracy_claim": "not a Chinese/Japanese/mixed-language song benchmark",
        "path": str(OUTPUT),
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "duration_sec": DURATION_SECONDS,
        "bpm": BPM,
        "active_bars": active_bars,
        "rest_bars": [16, 17, 18, 19, 36, 37, 38, 39, 52, 53, 54, 55],
        "sections": sections,
        "tail_marker": {"start_sec": 179.0, "end_sec": 179.72, "midi": 84},
    }
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"audio": str(OUTPUT), "metadata": str(METADATA), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
