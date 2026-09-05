"""CPU tsumugi adapter with checkpoint-specific stem routing."""

from __future__ import annotations

from collections import defaultdict
import math
import os
import subprocess
import tempfile
from pathlib import Path

import mido

from ..analysis import probe_audio, resolve_ffmpeg, resolve_ffprobe
from ..domain import NoteEvent
from .adapter import EngineExecutionError, EngineResult, EngineUnavailableError


ROOT = Path(__file__).resolve().parents[3]
TSUMUGI_ROOT = ROOT / "vendor" / "tsumugi-57b79ac4e1fa30c6f3eb95f14c77271fab637eeb"
TSUMUGI_CHECKPOINTS = {
    "bass": ROOT / ".cache" / "models" / "tsumugi" / "best_model_bass_v2.pth",
    "bass_v2": ROOT / ".cache" / "models" / "tsumugi" / "best_model_bass_v2.pth",
    "other": ROOT / ".cache" / "models" / "tsumugi" / "best_model_other_v1_5.pth",
    "other_v1_5": ROOT / ".cache" / "models" / "tsumugi" / "best_model_other_v1_5.pth",
    "vocal": ROOT / ".cache" / "models" / "tsumugi" / "best_model_vocal_harmony_v1_5.pth",
    "vocal_harmony": ROOT / ".cache" / "models" / "tsumugi" / "best_model_vocal_harmony_v1_5.pth",
    "vocal_harmony_v1_5": ROOT / ".cache" / "models" / "tsumugi" / "best_model_vocal_harmony_v1_5.pth",
}
STEM_MODEL_TYPES = {
    "vocals": "vocal_harmony_v1_5",
    "bass": "bass_v2",
    "other": "other_v1_5",
}


def tsumugi_python() -> Path:
    configured = os.environ.get("JIANPU_TSUMUGI_PYTHON")
    candidate = Path(configured) if configured else ROOT / ".venv-model-tsumugi" / "Scripts" / "python.exe"
    if not candidate.exists():
        raise EngineUnavailableError(f"tsumugi environment is missing: {candidate}")
    return candidate.resolve()


def model_type_for_stem(stem_id: str) -> str:
    try:
        return STEM_MODEL_TYPES[stem_id]
    except KeyError:
        raise EngineUnavailableError(
            f"tsumugi has no configured specialist checkpoint for stem {stem_id!r}; "
            "expected vocals, bass, or other"
        ) from None


def resolve_tsumugi_checkpoint(model_type: str | Path) -> tuple[str, Path]:
    candidate_path = Path(model_type)
    if candidate_path.is_file():
        return candidate_path.stem, candidate_path.resolve()
    normalized = str(model_type).strip().lower()
    if normalized not in TSUMUGI_CHECKPOINTS:
        supported = ", ".join(sorted(TSUMUGI_CHECKPOINTS))
        raise EngineUnavailableError(f"unknown tsumugi model {model_type!r}; choose one of {supported}")
    checkpoint = TSUMUGI_CHECKPOINTS[normalized]
    if not checkpoint.is_file():
        raise EngineUnavailableError(f"tsumugi checkpoint is missing: {checkpoint}")
    return normalized, checkpoint.resolve()


def checkpoint_for_stem(stem_id: str) -> tuple[str, Path]:
    return resolve_tsumugi_checkpoint(model_type_for_stem(stem_id))


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    ffmpeg = resolve_ffmpeg()
    ffprobe = resolve_ffprobe()
    if ffmpeg is not None:
        environment["PATH"] = os.fspath(ffmpeg.parent) + os.pathsep + environment.get("PATH", "")
        environment["JIANPU_FFMPEG"] = os.fspath(ffmpeg)
    if ffprobe is not None:
        environment["JIANPU_FFPROBE"] = os.fspath(ffprobe)
    return environment


def _tempo_points(midi: mido.MidiFile) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = [(0, 500000)]
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += int(message.time)
            if message.type == "set_tempo":
                points.append((absolute, int(message.tempo)))
    points.sort(key=lambda item: item[0])
    deduplicated: list[tuple[int, int]] = []
    for tick, tempo in points:
        if deduplicated and deduplicated[-1][0] == tick:
            deduplicated[-1] = (tick, tempo)
        else:
            deduplicated.append((tick, tempo))
    return deduplicated


def _tick_to_seconds(tick: int, *, ticks_per_beat: int, tempo_points: list[tuple[int, int]]) -> float:
    seconds = 0.0
    previous_tick = 0
    tempo = 500000
    for change_tick, change_tempo in tempo_points:
        if change_tick <= previous_tick:
            tempo = change_tempo
            continue
        if tick <= change_tick:
            break
        seconds += (change_tick - previous_tick) * tempo / 1_000_000.0 / ticks_per_beat
        previous_tick = change_tick
        tempo = change_tempo
    if tick > previous_tick:
        seconds += (tick - previous_tick) * tempo / 1_000_000.0 / ticks_per_beat
    return seconds


def _midi_events(midi_path: Path, *, stem_id: str | None, model: str, checkpoint: Path) -> list[NoteEvent]:
    midi = mido.MidiFile(os.fspath(midi_path))
    tempo_points = _tempo_points(midi)
    events: list[NoteEvent] = []
    for track_index, track in enumerate(midi.tracks):
        track_name = next(
            (str(message.name) for message in track if message.type == "track_name"),
            f"track-{track_index}",
        )
        lowered_name = track_name.lower()
        if "drum" in lowered_name or "percussion" in lowered_name:
            continue
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        absolute = 0
        for message in track:
            absolute += int(message.time)
            # MIDI channel 10 (zero based channel 9) is the General MIDI
            # percussion channel.  Tsumugi is used for pitched stems; keep
            # an unexpected drum event from becoming a pitched NoteEvent.
            message_channel = getattr(message, "channel", None)
            if message_channel == 9 and message.type in {"note_on", "note_off"}:
                continue
            if message.type == "note_on" and int(message.velocity) > 0:
                active[(int(message.channel), int(message.note))].append((absolute, int(message.velocity)))
                continue
            is_note_off = message.type == "note_off" or (message.type == "note_on" and int(message.velocity) == 0)
            if not is_note_off:
                continue
            key = (int(message.channel), int(message.note))
            starts = active.get(key)
            if not starts:
                continue
            start_tick, velocity = starts.pop(0)
            if absolute <= start_tick:
                continue
            start_sec = _tick_to_seconds(start_tick, ticks_per_beat=midi.ticks_per_beat, tempo_points=tempo_points)
            end_sec = _tick_to_seconds(absolute, ticks_per_beat=midi.ticks_per_beat, tempo_points=tempo_points)
            metadata = {
                "engine": "tsumugi",
                "model": model,
                "checkpoint": os.fspath(checkpoint),
                "track_index": track_index,
                "track_name": track_name,
                "raw_format": "tsumugi MIDI",
            }
            if stem_id is not None:
                metadata["stem_id"] = stem_id
            events.append(
                NoteEvent(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    midi=int(message.note),
                    confidence=None,
                    velocity=max(1, min(127, velocity)),
                    raw_pitch=float(message.note),
                    stem_id=stem_id,
                    source="tsumugi",
                    metadata=metadata,
                )
            )
        # Close a malformed open note at the track end rather than silently
        # dropping its already valid onset.
        for (channel, note), starts in active.items():
            for start_tick, velocity in starts:
                if absolute <= start_tick:
                    continue
                start_sec = _tick_to_seconds(start_tick, ticks_per_beat=midi.ticks_per_beat, tempo_points=tempo_points)
                end_sec = _tick_to_seconds(absolute, ticks_per_beat=midi.ticks_per_beat, tempo_points=tempo_points)
                events.append(
                    NoteEvent(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        midi=int(note),
                        confidence=None,
                        velocity=max(1, min(127, velocity)),
                        raw_pitch=float(note),
                        stem_id=stem_id,
                        source="tsumugi",
                        metadata={
                            "engine": "tsumugi",
                            "model": model,
                            "checkpoint": os.fspath(checkpoint),
                            "track_index": track_index,
                            "track_name": track_name,
                            "raw_format": "tsumugi MIDI",
                            "closed_at_track_end": True,
                        },
                    )
                )
    events.sort(key=lambda event: (event.start_sec, event.end_sec, event.midi, event.stem_id or ""))
    return events


def extract_tsumugi(
    audio_path: str | Path,
    *,
    model_type: str | Path = "other_v1_5",
    stem_id: str | None = None,
    enforce_upload_size: bool = True,
) -> EngineResult:
    """Run the selected CPU checkpoint and parse its MIDI into NoteEvents."""

    audio = Path(audio_path).resolve()
    probe_audio(audio, enforce_upload_size=enforce_upload_size)
    if not TSUMUGI_ROOT.is_dir() or not (TSUMUGI_ROOT / "infer.py").is_file():
        raise EngineUnavailableError(f"tsumugi source tree is missing: {TSUMUGI_ROOT}")
    python = tsumugi_python()
    model_name, checkpoint = resolve_tsumugi_checkpoint(model_type)
    cache_dir = ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tsumugi-", dir=os.fspath(cache_dir)) as temp_dir:
        output_midi = Path(temp_dir) / "events.mid"
        command = [
            os.fspath(python),
            os.fspath(TSUMUGI_ROOT / "infer.py"),
            "--audio",
            os.fspath(audio),
            "--output-midi",
            os.fspath(output_midi),
            "--checkpoint",
            os.fspath(checkpoint),
            "--device",
            "cpu",
            "--semi-crf-backend",
            "torch",
            "--window-batch-size",
            "1",
            "--disable-tqdm",
        ]
        result = subprocess.run(
            command,
            cwd=TSUMUGI_ROOT,
            env=_environment(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise EngineExecutionError(
                f"tsumugi subprocess failed ({result.returncode}): {result.stderr[-4000:]}"
            )
        if not output_midi.is_file():
            raise EngineExecutionError(
                f"tsumugi produced no MIDI output for {audio.name}: {result.stdout[-2000:]}"
            )
        events = _midi_events(output_midi, stem_id=stem_id, model=model_name, checkpoint=checkpoint)

    warnings: list[str] = []
    if model_name == "vocal_harmony_v1_5":
        warnings.append("tsumugi vocal_harmony_v1_5 是和声模型，与 GAME 人声主旋律模型不同")
    return EngineResult(
        events=events,
        engine="tsumugi",
        model=model_name,
        metadata={
            "engine": "tsumugi",
            "environment": os.fspath(python),
            "model": model_name,
            "checkpoint": os.fspath(checkpoint),
            "device": "cpu",
            "semi_crf_backend": "torch",
            "stem_id": stem_id,
            "event_count": len(events),
        },
        warnings=warnings,
    )
