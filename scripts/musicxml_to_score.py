"""Run one pinned MuseScore MIDI import and one isolated music21 pass.

This is a stage 5/6 command-line adapter.  It deliberately handles one
performance MIDI at a time so a failed instrument has an explicit result and
cannot overwrite another instrument's MusicXML or Score JSON.  It is not
connected to the production job pipeline until the complete chain is reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.musicxml_standardize import (  # noqa: E402
    MusicXMLStandardizationError,
    write_standardized_score,
)
from backend.jianpu_score.musescore_import import MuseScoreImportError, convert_performance_midi  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert one performance MIDI to MusicXML and a 48 TPQ Score JSON")
    parser.add_argument("--midi", required=True, type=Path)
    parser.add_argument("--musicxml", required=True, type=Path)
    parser.add_argument("--score-json", required=True, type=Path)
    parser.add_argument("--alignment-json", type=Path)
    parser.add_argument("--performance-metadata", type=Path)
    parser.add_argument("--instrument-id", default="instrument")
    parser.add_argument("--title")
    parser.add_argument("--musescore", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--notation-python", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        musicxml = convert_performance_midi(
            args.midi,
            args.musicxml,
            instrument_id=args.instrument_id,
            musescore_path=args.musescore,
            profile_path=args.profile,
            timeout_sec=args.timeout_sec,
            overwrite=args.overwrite,
        )
        artifact = write_standardized_score(
            musicxml.musicxml_path,
            args.score_json,
            alignment_report_path=args.alignment_json,
            performance_metadata=args.performance_metadata,
            title=args.title,
            notation_python=args.notation_python,
            timeout_sec=args.timeout_sec,
        )
    except (MuseScoreImportError, MusicXMLStandardizationError, OSError, ValueError) as exc:
        print(f"musicxml-stage56-failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "instrument_id": args.instrument_id,
                "musicxml": str(artifact.musicxml_path),
                "score_json": str(artifact.score_json_path),
                "alignment_report": str(artifact.alignment_report_path),
                "score_ticks_per_quarter": artifact.score.quarter_ticks,
                "voices": len(artifact.score.voices),
                "total_ticks": artifact.score.total_ticks,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
