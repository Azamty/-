"""Isolated MuScriptor worker used by the V2 persistent queue.

The API process intentionally does not import the model runtime.  This worker
is launched from ``.venv-model-muscriptor`` and writes a small, durable JSON
recognition record plus the model's complete MIDI output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))
DEFAULT_BENCHMARK_SEED = 20260907


def _configure_reproducibility(seed: int) -> dict[str, Any]:
    """Make the greedy model route repeatable across benchmark invocations."""

    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        numpy_available = False
    else:
        np.random.seed(seed)
        numpy_available = True
    import torch

    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "decoder": "greedy",
        "sampling": False,
        "numpy_seeded": numpy_available,
        "torch_deterministic_algorithms": True,
        "cuda_deterministic_flags": cuda_available,
        "autocast": "disabled_float32",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _local_model_path() -> str:
    explicit = os.environ.get("MUSCRIPTOR_MODEL_PATH")
    if explicit:
        return explicit
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--MuScriptor--muscriptor-medium" / "snapshots"
    candidates = sorted(
        (path / "model.safetensors" for path in cache.glob("*")),
        key=lambda path: path.stat().st_mtime if path.is_file() else 0,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_file():
            return os.fspath(candidate)
    # The package will use its normal local cache resolution if this machine
    # has a compatible cache in another location.
    return "medium"


def _collect_events(events: list[Any], model: Any) -> list[dict[str, Any]]:
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    starts: dict[int, Any] = {}
    notes: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, NoteStartEvent):
            starts[event.index] = event
            continue
        if not isinstance(event, NoteEndEvent):
            continue
        start = starts.pop(event.start_event_index)
        instrument = str(start.instrument)
        is_drum = instrument == "drums"
        program = 128 if is_drum else int(model._program_for_instrument(instrument))
        start_sec = max(0.0, float(start.start_time))
        end_sec = max(start_sec + 0.001, float(event.end_time))
        notes.append(
            {
                "instrument_group": instrument,
                "program": program,
                "is_drum": is_drum,
                "pitch": int(start.pitch),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "velocity": None,
                "metadata": {"playback_default": 80},
            }
        )
    if starts:
        raise RuntimeError(f"MuScriptor returned {len(starts)} unclosed note starts")
    notes.sort(key=lambda note: (note["start_sec"], note["instrument_group"], note["pitch"], note["end_sec"]))
    return notes


def _tracks(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from backend.muscriptor_v2 import instrument_label_zh, stable_track_id

    grouped: dict[tuple[str, int, bool], list[dict[str, Any]]] = {}
    for note in notes:
        key = (str(note["instrument_group"]), int(note["program"]), bool(note["is_drum"]))
        grouped.setdefault(key, []).append(note)
    result: list[dict[str, Any]] = []
    for (group, program, is_drum), group_notes in grouped.items():
        track_id = stable_track_id(group, program, is_drum)
        result.append(
            {
                "track_id": track_id,
                "instrument_group": group,
                "label_zh": instrument_label_zh(group),
                "program": program,
                "is_drum": is_drum,
                "note_count": len(group_notes),
                "duration_sec": max(float(item["end_sec"]) for item in group_notes),
                "preview_available": True,
            }
        )
    return sorted(result, key=lambda item: (bool(item["is_drum"]), str(item["instrument_group"]), int(item["program"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--deterministic", action="store_true", help="enable the reproducible benchmark decode contract")
    args = parser.parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"audio file not found: {args.audio}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from muscriptor import TranscriptionModel
    from muscriptor.events import ProgressEvent

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MuScriptor CUDA is unavailable")
    if args.deterministic:
        reproducibility = _configure_reproducibility(args.seed)
    else:
        reproducibility = {
            "seed": None,
            "decoder": "greedy",
            "sampling": False,
            "numpy_seeded": False,
            "torch_deterministic_algorithms": False,
            "cuda_deterministic_flags": False,
            "autocast": "package_default",
        }
    model = TranscriptionModel.load_model(args.model or _local_model_path(), device=args.device)
    if args.deterministic:
        # The package enables fp16 autocast on CUDA even when the checkpoint is
        # loaded as float32.  Disabling that private context avoids marginal
        # CUDA reduction differences changing a greedy token at a tied logit
        # boundary.
        model._model.autocast.enabled = False
    raw_events: list[Any] = []
    last_progress = {"completed": 0, "total": 0}
    for event in model.transcribe(args.audio, instruments=None, prelude_forcing=True):
        raw_events.append(event)
        if isinstance(event, ProgressEvent):
            last_progress = {"completed": int(event.completed), "total": int(event.total)}
            _write_json(args.progress, {**last_progress, "status": "recognizing"})
    notes = _collect_events(raw_events, model)
    tracks = _tracks(notes)
    midi_path = args.output / "original.mid"
    midi_path.write_bytes(model.events_to_midi_bytes(iter(raw_events)))
    recognition = {
        "schema_version": "2.0",
        "engine": "muscriptor",
        "model": "medium",
        "device": str(getattr(model, "_device", args.device)),
        "source_kind": "instrumental",
        "use_demucs": False,
        "progress": {**last_progress, "status": "selection_ready"},
        "notes": notes,
        "tracks": tracks,
        "midi_filename": midi_path.name,
        "duration_sec": max((float(note["end_sec"]) for note in notes), default=0.0),
        "note_count": len(notes),
        "instrument_counts": dict(Counter(str(note["instrument_group"]) for note in notes)),
        "metadata": {
            "full_decode_before_selection": True,
            "velocity_policy": "playback_default",
            "note_event_velocity": None,
            "time_basis": "source_seconds",
            "reproducibility": reproducibility,
        },
    }
    _write_json(args.output / "recognition.json", recognition)
    _write_json(args.progress, {**last_progress, "status": "selection_ready"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
