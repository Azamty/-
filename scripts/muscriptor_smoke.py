"""Run a real MuScriptor decode and record V2 routing evidence.

This script deliberately decodes the complete instrumental input first.  The
instrument selection in the manifest is a post-decode policy decision, so the
same model run can feed per-instrument score stems and the drum MIDI preview.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from backend.muscriptor_v2 import MuscriptorNote, make_transcription_plan, partition_notes


def _collect_notes(events: Iterable[Any]) -> tuple[list[MuscriptorNote], list[Any]]:
    starts: dict[int, Any] = {}
    notes: list[MuscriptorNote] = []
    materialized = list(events)
    for event in materialized:
        if event.__class__.__name__ == "NoteStartEvent":
            starts[event.index] = event
        elif event.__class__.__name__ == "NoteEndEvent":
            start = starts.pop(event.start_event_index)
            notes.append(
                MuscriptorNote(
                    instrument=str(start.instrument),
                    pitch=int(start.pitch),
                    start_sec=float(start.start_time),
                    end_sec=float(event.end_time),
                )
            )
    if starts:
        raise RuntimeError(f"MuScriptor returned {len(starts)} unclosed note starts")
    notes.sort(key=lambda note: (note.start_sec, note.instrument, note.pitch, note.end_sec))
    return notes, materialized


def _event_classes() -> tuple[type[Any], type[Any], type[Any]]:
    from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

    return NoteStartEvent, NoteEndEvent, ProgressEvent


def _progress(events: Iterable[Any]) -> list[dict[str, int]]:
    _start, _end, progress_type = _event_classes()
    return [
        {"completed": int(event.completed), "total": int(event.total)}
        for event in events
        if isinstance(event, progress_type)
    ]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real MuScriptor V2 CUDA smoke")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="medium", help="size keyword or local safetensors path")
    parser.add_argument("--device", default="cuda", help="torch device; V2 smoke defaults to CUDA")
    parser.add_argument("--dtype", default=None, help="optional torch dtype override")
    parser.add_argument("--source-kind", choices=("instrumental", "vocal"), default="instrumental")
    parser.add_argument("--select", nargs="*", default=None, help="post-decode instrument names")
    parser.add_argument("--no-drums", action="store_true", help="omit the drum MIDI preview")
    parser.add_argument("--merge-main-melody", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/review/stageB-muscriptor/muscriptor-smoke.json"),
    )
    parser.add_argument(
        "--midi-output",
        type=Path,
        default=Path("artifacts/review/stageB-muscriptor/muscriptor-full.mid"),
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        raise SystemExit(f"audio file not found: {args.audio}")

    import torch
    from muscriptor import TranscriptionModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    load_started = time.perf_counter()
    model = TranscriptionModel.load_model(args.model, device=args.device, dtype=args.dtype)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    decode_started = time.perf_counter()
    raw_events = list(model.transcribe(args.audio, instruments=None, prelude_forcing=True))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    notes, materialized_events = _collect_notes(raw_events)
    instruments = tuple(dict.fromkeys(note.instrument for note in notes))
    plan = make_transcription_plan(
        args.source_kind,
        instruments,
        selected_instruments=args.select,
        include_drums=not args.no_drums,
        merge_main_melody=args.merge_main_melody,
    )
    partitions = partition_notes(notes, plan)

    args.midi_output.parent.mkdir(parents=True, exist_ok=True)
    args.midi_output.write_bytes(model.events_to_midi_bytes(iter(materialized_events)))
    audio_duration = max((note.end_sec for note in notes), default=0.0)
    manifest: dict[str, Any] = {
        "status": "passed",
        "model": args.model,
        "audio": os.fspath(args.audio.resolve()),
        "audio_duration_sec": audio_duration,
        "device": str(getattr(model, "_device", args.device)),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        },
        "timings_sec": {
            "load_model": load_seconds,
            "decode": decode_seconds,
            "real_time_factor": decode_seconds / audio_duration if audio_duration else None,
        },
        "progress": _progress(raw_events),
        "note_count": len(notes),
        "instrument_counts": dict(Counter(note.instrument for note in notes)),
        "instrument_inventory": list(instruments),
        "routing": {
            "source_kind": plan.source_kind,
            "engine": plan.engine,
            "use_demucs": plan.use_demucs,
            "full_decode_before_selection": True,
        },
        "selection": {
            "detected_instruments": list(plan.detected_instruments),
            "selected_pitched_instruments": list(plan.selected_pitched_instruments),
            "drum_preview_enabled": plan.drum_preview_enabled,
            "merge_main_melody": plan.merge_main_melody,
            "partition_counts": {name: len(values) for name, values in partitions.items()},
        },
        "midi_output": os.fspath(args.midi_output.resolve()),
        "note_pitch_range": {
            "min": min((note.pitch for note in notes), default=None),
            "max": max((note.pitch for note in notes), default=None),
        },
    }
    _write_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
