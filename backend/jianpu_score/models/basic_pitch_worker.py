"""Run Basic Pitch inside its dedicated environment and emit JSON events."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    parser.add_argument("--minimum-note-length", type=float, default=127.7)
    args = parser.parse_args()

    # This worker filename lives beside the adapter named ``basic_pitch.py``;
    # remove the script directory before importing the installed distribution.
    script_dir = Path(__file__).resolve().parent
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != script_dir]
    from basic_pitch.inference import predict

    # The library prints progress to stdout; reserve stdout for a small status
    # line and make the JSON file the subprocess contract.
    with contextlib.redirect_stdout(sys.stderr):
        _model_output, _midi, note_events = predict(
            str(args.audio),
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            minimum_note_length=args.minimum_note_length,
            midi_tempo=120,
        )
    serialized = []
    for start, end, pitch, amplitude, _pitch_bends in note_events:
        start_value, end_value, raw_pitch = float(start), float(end), float(pitch)
        if not all(math.isfinite(value) for value in (start_value, end_value, raw_pitch)):
            continue
        midi = int(round(raw_pitch))
        if end_value <= start_value or not 0 <= midi <= 127:
            continue
        try:
            amplitude_value = float(amplitude)
        except (TypeError, ValueError):
            amplitude_value = float("nan")
        metadata = {"amplitude": amplitude_value} if math.isfinite(amplitude_value) else {}
        serialized.append(
            {
                "start_sec": start_value,
                "end_sec": end_value,
                "midi": midi,
                # Basic Pitch's fourth tuple item is an amplitude, not a
                # calibrated probability.  Keep it in metadata instead of
                # presenting it as confidence.
                "confidence": None,
                "raw_pitch": raw_pitch,
                "voice_id": "voice-0",
                "source": "basic-pitch",
                "metadata": metadata,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    print(f"basic_pitch_events={len(serialized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
