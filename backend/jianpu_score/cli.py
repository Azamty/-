"""Command line entry point for a full audio -> Score -> SVG/MIDI run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .capabilities import get_capabilities
from .models.adapter import EngineError
from .pipeline import SUPPORTED_ENGINES, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert local audio to numbered notation SVG and MIDI")
    parser.add_argument("--input", required=False, type=Path, help="MP3, WAV, FLAC or M4A input")
    parser.add_argument("--output", required=False, type=Path, help="artifact directory")
    parser.add_argument("--engine", choices=SUPPORTED_ENGINES, default="basic-pitch")
    parser.add_argument("--voice-mode", choices=("monophonic", "polyphonic"), default="monophonic")
    parser.add_argument("--source", choices=("mixed", "vocal", "instrumental"), default="mixed", help="source stem; vocal/instrumental require --separate")
    parser.add_argument("--separate", action="store_true", help="run Demucs htdemucs before pitch extraction")
    parser.add_argument("--bpm", type=float, default=None, help="manual BPM override")
    parser.add_argument("--key", default=None, help="manual key override, e.g. C or G")
    parser.add_argument("--time-signature", default=None, help="manual time signature, e.g. 4/4")
    parser.add_argument("--language", choices=("zh", "ja", "mixed"), default="mixed", help="GAME language routing; mixed omits the language ID")
    parser.add_argument("--title", default=None)
    parser.add_argument("--print-capabilities", action="store_true", help="print JSON engine capabilities and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_capabilities:
        print(json.dumps(get_capabilities(), ensure_ascii=False, indent=2))
        return 0
    if args.input is None or args.output is None:
        build_parser().error("--input and --output are required unless --print-capabilities is used")
    try:
        analysis, score, artifacts = run_pipeline(
            args.input,
            args.output,
            engine=args.engine,
            voice_mode=args.voice_mode,
            source_kind=args.source,
            separate=args.separate,
            bpm_override=args.bpm,
            key_override=args.key,
            time_signature_override=args.time_signature,
            language=args.language,
            title=args.title,
        )
    except EngineError as exc:
        build_parser().error(str(exc))
    manifest = {
        "input": os.fspath(args.input.resolve()),
        "analysis": analysis.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
        "artifacts": artifacts.model_dump(mode="json"),
    }
    (Path(args.output).resolve() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"analysis": os.fspath(Path(args.output).resolve() / "analysis.json"), "score": os.fspath(Path(args.output).resolve() / "score.json"), "artifacts": artifacts.model_dump(mode="json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
