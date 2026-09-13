"""Render cached recognition and BeatNet observations through the direct service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.beat_grid import build_beat_grid
from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.high_accuracy_service import HighAccuracyArtifactService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--beats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--meter", default="4/4")
    parser.add_argument("--instrument", default="acoustic_piano")
    parser.add_argument("--title")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vocal", action="store_true")
    parser.add_argument("--options", default="{}", help="DirectNotationOptions JSON")
    args = parser.parse_args()
    payload = json.loads(args.notes.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        events = [NoteEvent.model_validate(n) for n in payload]
    else:
        events = [NoteEvent(midi=n["pitch"], start_sec=n["start_sec"], end_sec=n["end_sec"],
                            velocity=n.get("velocity"), stem_id=args.instrument, source="muscriptor")
                  for n in payload["notes"] if n["instrument_group"] == args.instrument]
    grid = build_beat_grid(json.loads(args.beats.read_text(encoding="utf-8")),
                          duration_sec=args.duration, manual_bpm=args.bpm,
                          manual_time_signature=args.meter)
    analysis = MusicAnalysis(sample_rate=22050, duration_sec=args.duration,
        bpm=grid["tempo"]["selected_bpm"], key=args.key, time_signature=args.meter,
        beat_times=grid["mapping"]["beat_times"], metadata={
            "beat_engine": "beatnet", "beatnet_version": "1.1.3", "beat_grid": grid,
            "notation_engine": "direct-jianpu", "direct_options": json.loads(args.options)})
    result = HighAccuracyArtifactService().build(instrument_id=args.instrument,
        title=args.title or args.notes.parent.name, program=0, is_drum=False, events=events,
        analysis=analysis, output_dir=args.output, variant="vocal" if args.vocal else "source", overwrite=args.overwrite)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    print(json.dumps({"input_notes": len(events),
        "output_notes": result.alignment_report["output_note_count"],
        "voices": len(result.score.voices), "bpm": result.score.bpm,
        "verified_note_intervals": len(manifest["stages"]["render"]["midi_verification"]["actual_note_intervals"]),
        "output": str(result.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
