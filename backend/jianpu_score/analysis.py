"""Audio loading, beat/key analysis and an honest librosa pitch fallback."""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import librosa
import numpy as np
import soundfile as sf

from .beatnet import analyze_with_beatnet
from .domain import MusicAnalysis, NoteEvent, normalize_key, normalize_time_signature


KNOWN_FFMPEG = Path(r"E:\develop\y\ffmpeg-9.0.1-full_build\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
KNOWN_FFPROBE = Path(r"E:\develop\y\ffmpeg-9.0.1-full_build\ffmpeg-9.0.1-full_build\bin\ffprobe.exe")
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_DURATION_SEC = 15 * 60
# Demucs writes trusted decoded stems as WAV.  Their compressed upload size is
# irrelevant, but a duration and decoded-file guard still prevents an
# accidental unbounded internal file from entering a model process.
MAX_INTERNAL_AUDIO_BYTES = 2 * 1024 * 1024 * 1024
TIME_SIGNATURE_CANDIDATES = ("2/4", "3/4", "4/4", "6/8")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def resolve_ffmpeg() -> Path | None:
    configured = os.environ.get("JIANPU_FFMPEG")
    if configured and Path(configured).exists():
        return Path(configured)
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    if KNOWN_FFMPEG.exists():
        return KNOWN_FFMPEG
    return None


def resolve_ffprobe() -> Path | None:
    configured = os.environ.get("JIANPU_FFPROBE")
    if configured and Path(configured).exists():
        return Path(configured)
    found = shutil.which("ffprobe")
    if found:
        return Path(found)
    if KNOWN_FFPROBE.exists():
        return KNOWN_FFPROBE
    return None


def _load_with_ffmpeg(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    ffmpeg = resolve_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for compressed audio but was not found")
    result = subprocess.run(
        [
            os.fspath(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            os.fspath(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {error}")
    samples, actual_rate = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    return np.asarray(samples, dtype=np.float32), int(actual_rate)


def probe_audio(
    path: str | Path,
    *,
    enforce_upload_size: bool = True,
) -> dict[str, float | int | str]:
    """Validate duration and the appropriate external or trusted-stem limit.

    The 100 MB rule applies to the original upload.  Demucs stems are trusted
    internal decoded files, so they use a larger decoded-file guard while
    retaining the same 15 minute duration limit.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if enforce_upload_size and size > MAX_AUDIO_BYTES:
        raise ValueError(f"audio exceeds the 100 MB limit: {size} bytes")
    if not enforce_upload_size and size > MAX_INTERNAL_AUDIO_BYTES:
        raise ValueError(f"decoded internal audio exceeds the 2 GB limit: {size} bytes")
    ffprobe = resolve_ffprobe()
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate audio duration")
    result = subprocess.run(
        [
            os.fspath(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            os.fspath(source),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise ValueError(f"ffprobe could not read audio metadata: {result.stderr.strip()}")
    try:
        duration = float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        raise ValueError("ffprobe returned no finite audio duration") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"ffprobe returned invalid duration: {duration!r}")
    if duration > MAX_AUDIO_DURATION_SEC:
        raise ValueError(f"audio exceeds the 15 minute limit: {duration:.3f} seconds")
    return {"path": os.fspath(source), "bytes": size, "duration_sec": duration}


def load_audio(path: str | Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load supported audio without relying on a global ffmpeg PATH."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported audio extension: {source.suffix}")
    if source.suffix.lower() in {".mp3", ".m4a"}:
        return _load_with_ffmpeg(source, sample_rate)
    samples, actual_rate = librosa.load(os.fspath(source), sr=sample_rate, mono=True)
    return np.asarray(samples, dtype=np.float32), int(actual_rate)


def _estimate_key_candidates(samples: np.ndarray, sample_rate: int) -> list[str]:
    if samples.size < sample_rate // 4:
        return ["C"]
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sample_rate)
    profile = np.mean(chroma, axis=1)
    profile = profile / (np.linalg.norm(profile) + 1e-9)
    scores: list[tuple[float, str]] = []
    for root in range(12):
        rotated = np.roll(profile, -root)
        scores.append((float(np.dot(rotated, MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE))), KEY_NAMES[root]))
        scores.append((float(np.dot(rotated, MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE))), f"{KEY_NAMES[root]}m"))
    return [key for _score, key in sorted(scores, reverse=True)[:4]]


def _estimate_key(samples: np.ndarray, sample_rate: int) -> str:
    return _estimate_key_candidates(samples, sample_rate)[0]


def _regular_beat_grid(duration_sec: float, bpm: float) -> list[float]:
    interval = 60.0 / bpm
    count = max(1, int(math.ceil(duration_sec / interval)))
    return [float(index * interval) for index in range(count + 1)]


def _estimate_beats(samples: np.ndarray, sample_rate: int, bpm_override: float | None) -> tuple[float, list[float], list[str], str]:
    warnings: list[str] = []
    if bpm_override is not None:
        if not math.isfinite(bpm_override) or bpm_override <= 0:
            raise ValueError("bpm override must be finite and greater than zero")
        bpm = float(bpm_override)
        return bpm, _regular_beat_grid(len(samples) / sample_rate, bpm), warnings, "manual_bpm"
    tempo, beats = librosa.beat.beat_track(y=samples, sr=sample_rate, units="time")
    tempo_array = np.asarray(tempo).reshape(-1)
    estimated = float(tempo_array[0]) if tempo_array.size else 0.0
    if estimated > 1 and math.isfinite(estimated):
        bpm = estimated
        beat_source = "librosa"
    else:
        bpm = 120.0
        warnings.append("节拍分析未得到可靠速度，已使用默认 120 BPM")
        beat_source = "fallback_bpm"
    beat_times = [float(value) for value in np.asarray(beats).reshape(-1)]
    if len(beat_times) < 2 or any(not math.isfinite(value) for value in beat_times):
        beat_times = _regular_beat_grid(len(samples) / sample_rate, bpm)
        warnings.append("未检测到可靠拍点，已按 BPM 生成均匀拍点网格")
    return bpm, beat_times, warnings, beat_source


def _bpm_candidates(bpm: float, source: str) -> list[float]:
    """Return the measured tempo plus useful half/double-time alternatives."""

    values = [float(bpm)]
    if source not in {"manual_bpm", "fallback_bpm"}:
        values.extend((float(bpm) / 2.0, float(bpm) * 2.0))
    return list(dict.fromkeys(round(value, 2) for value in values if 30.0 <= value <= 300.0))


def analyze_audio(
    path: str | Path,
    *,
    bpm_override: float | None = None,
    key_override: str | None = None,
    time_signature_override: str | None = None,
    source_onsets: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    sample_rate: int = 22050,
) -> tuple[np.ndarray, MusicAnalysis]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    probe = probe_audio(path)
    samples, actual_rate = load_audio(path, sample_rate=sample_rate)
    # The high-accuracy chain always analyzes the original audio through the
    # isolated BeatNet offline/DBN runtime.  A worker error is intentionally
    # propagated to the task; falling back to the legacy uniform grid here
    # would make the resulting Score impossible to audit.
    beat_grid = analyze_with_beatnet(
        path,
        duration_sec=max(len(samples) / actual_rate, 1e-6),
        bpm_override=bpm_override,
        time_signature_override=time_signature_override,
        source_onsets=source_onsets,
    )
    bpm = float(beat_grid["tempo"]["selected_bpm"])
    beat_times = [float(value) for value in beat_grid["mapping"]["beat_times"]]
    warnings = list(beat_grid.get("warnings", []))
    beat_source = "beatnet"
    if key_override is not None:
        key = normalize_key(key_override)
        key_candidates = [key]
    else:
        key_candidates = _estimate_key_candidates(samples, actual_rate)
        key = normalize_key(key_candidates[0])
    time_signature = normalize_time_signature(str(beat_grid["time_signature"]["selected"]))
    time_signature_source = "manual" if time_signature_override is not None else "beatnet_derived"
    tempo_candidates = [float(item["bpm"]) for item in beat_grid["tempo"]["candidates"]]
    meter_candidates = [str(item["value"]) for item in beat_grid["time_signature"]["candidates"]]
    analysis = MusicAnalysis(
        sample_rate=actual_rate,
        duration_sec=max(len(samples) / actual_rate, 1e-6),
        bpm=bpm,
        key=key,
        time_signature=time_signature,
        beat_times=beat_times,
        warnings=warnings,
        metadata={
            "audio_path": os.fspath(Path(path).resolve()),
            "ffmpeg": os.fspath(resolve_ffmpeg()) if resolve_ffmpeg() else None,
            "ffprobe": os.fspath(resolve_ffprobe()) if resolve_ffprobe() else None,
            "probe": probe,
            "beat_source": beat_source,
            "beat_engine": "beatnet",
            "beatnet_version": beat_grid.get("beatnet", {}).get("version", "1.1.3"),
            "beatnet_mode": beat_grid.get("mode", "offline"),
            "beatnet_inference": beat_grid.get("inference", "DBN"),
            "beat_grid": beat_grid,
            "bpm_candidates": tempo_candidates,
            "key_candidates": key_candidates,
            "time_signature_source": time_signature_source,
            "time_signature_candidates": meter_candidates,
            "manual_bpm_override": bpm_override is not None,
            "manual_time_signature_override": time_signature_override is not None,
        },
    )
    return samples, analysis


def extract_librosa_events(samples: np.ndarray, sample_rate: int, *, source: str = "librosa") -> list[NoteEvent]:
    """Extract contiguous monophonic f0 regions as real NoteEvents."""

    if samples.size < sample_rate // 20:
        return []
    f0, voiced, probability = librosa.pyin(
        samples,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=2048,
        hop_length=256,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=256)
    events: list[NoteEvent] = []

    def probability_metadata(values: np.ndarray) -> tuple[float | None, dict[str, float]]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return None, {}
        value = float(np.mean(finite))
        return value, {"voiced_probability": value}

    begin: int | None = None
    for index, (frequency, is_voiced) in enumerate(zip(f0, voiced)):
        if bool(is_voiced) and np.isfinite(frequency):
            if begin is None:
                begin = index
        elif begin is not None:
            end = index
            if end - begin >= 2:
                segment = f0[begin:end]
                hz = float(np.nanmedian(segment))
                if not math.isfinite(hz):
                    begin = None
                    continue
                confidence, metadata = probability_metadata(probability[begin:end])
                events.append(
                    NoteEvent(
                        start_sec=float(times[begin]),
                        end_sec=float(times[min(end, len(times) - 1)]),
                        midi=int(np.clip(np.rint(librosa.hz_to_midi(hz)), 0, 127)),
                        confidence=confidence,
                        raw_pitch=float(librosa.hz_to_midi(hz)),
                        source=source,
                        metadata=metadata,
                    )
                )
            begin = None
    if begin is not None and len(f0) - begin >= 2:
        segment = f0[begin:]
        hz = float(np.nanmedian(segment))
        if not math.isfinite(hz):
            return events
        confidence, metadata = probability_metadata(probability[begin:])
        events.append(
            NoteEvent(
                start_sec=float(times[begin]),
                end_sec=float(times[-1] + 256 / sample_rate),
                midi=int(np.clip(np.rint(librosa.hz_to_midi(hz)), 0, 127)),
                confidence=confidence,
                raw_pitch=float(librosa.hz_to_midi(hz)),
                source=source,
                metadata=metadata,
            )
        )
    return events
