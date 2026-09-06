"""Run a real BeatNet 1.1.3 offline/DBN decode on a short synthetic click track."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from backend.jianpu_score.beatnet import analyze_with_beatnet


def _click_track(sample_rate: int = 22050, duration_sec: float = 8.0) -> np.ndarray:
    samples = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    for index, start_sec in enumerate(np.arange(0.15, duration_sec, 0.5)):
        start = int(round(float(start_sec) * sample_rate))
        length = min(int(0.08 * sample_rate), len(samples) - start)
        if length <= 0:
            continue
        time = np.arange(length, dtype=np.float32) / sample_rate
        frequency = 110.0 if index % 4 == 0 else 880.0
        envelope = np.exp(-45.0 * time)
        samples[start : start + length] += (0.8 if index % 4 == 0 else 0.45) * envelope * np.sin(2 * np.pi * frequency * time)
    return np.clip(samples, -1.0, 1.0)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="jianpu-beatnet-smoke-") as temporary:
        audio = Path(temporary) / "clicks.wav"
        sf.write(audio, _click_track(), 22050)
        grid = analyze_with_beatnet(audio, duration_sec=8.0)
    beats = grid.get("beats", [])
    if len(beats) < 4:
        raise RuntimeError(f"BeatNet smoke returned too few beats: {len(beats)}")
    if grid.get("mode") != "offline" or grid.get("inference") != "DBN":
        raise RuntimeError(f"BeatNet smoke used an unexpected decoder: {grid.get('mode')}/{grid.get('inference')}")
    if grid.get("beatnet", {}).get("version") != "1.1.3":
        raise RuntimeError(f"BeatNet version contract mismatch: {grid.get('beatnet')}")
    print(
        json.dumps(
            {
                "status": "ok",
                "beat_count": len(beats),
                "first_beats": beats[:4],
                "last_beat_sec": beats[-1]["time_sec"],
                "time_signature": grid["time_signature"],
                "tempo": {
                    key: value
                    for key, value in grid["tempo"].items()
                    if key not in {"candidates"}
                },
                "bar_count": len(grid.get("bars", [])),
                "warnings": grid.get("warnings", []),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
