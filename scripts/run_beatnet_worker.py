"""Run the pinned BeatNet 1.1.3 offline/DBN decoder in its isolated env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"audio file not found: {args.audio}")

    import numpy as np
    from BeatNet.BeatNet import BeatNet

    estimator = BeatNet(
        args.model,
        mode="offline",
        inference_model="DBN",
        plot=[],
        thread=False,
        device=args.device,
    )
    output = np.asarray(estimator.process(str(args.audio)), dtype=float)
    if output.ndim != 2 or output.shape[1] < 2 or output.shape[0] < 2:
        raise RuntimeError(f"BeatNet returned an invalid result shape: {output.shape}")
    beats = []
    for row in output:
        time_sec = float(row[0])
        beat_number = int(round(float(row[1])))
        if not np.isfinite(time_sec) or beat_number <= 0:
            continue
        beats.append(
            {
                "time_sec": time_sec,
                "beat_number": beat_number,
                "downbeat": beat_number == 1,
            }
        )
    if len(beats) < 2:
        raise RuntimeError("BeatNet returned fewer than two finite beats")
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "engine": "beatnet",
                "version": "1.1.3",
                "mode": "offline",
                "inference": "DBN",
                "confidence_source": "derived_downbeat_and_interval_stability",
                "beats": beats,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
