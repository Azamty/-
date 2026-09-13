"""Prepare five reproducible Chinese mixed-song benchmark segments.

The CCMusic demo archive is supplied locally and is never downloaded by this
script.  It verifies the official archive bytes and hashes, extracts only the
five named Yueding members through safe paths, parses the score in the pinned
``.venv-notation`` music21 worker, and writes rebuildable audio/MIDI/beat
artifacts below ``.cache``.  The generated files are intentionally excluded
from Git because they contain corpus audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import librosa
import mido
import numpy as np
import soundfile as sf
from scipy.optimize import differential_evolution
from scipy.signal import correlate, correlation_lags, resample_poly
from scipy.stats import median_abs_deviation, theilslopes

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DEFAULT_ARCHIVE = ROOT / ".cache" / "packages" / "ccmusic-database-demo.zip"
DEFAULT_SOURCE_ROOT = ROOT / ".cache" / "high-accuracy-benchmarks" / "ccmusic-demo"
DEFAULT_OUTPUT_ROOT = ROOT / ".cache" / "high-accuracy-benchmarks" / "ccmusic-yueding"
EXPECTED_ARCHIVE_BYTES = 302_024_881
EXPECTED_MD5 = "DBDC4A7E019C6B7A1424D99FDD8A7838"
EXPECTED_SHA256 = "477B5466936EEC40CEF7DFD43205900E3E4A651B8EC671FDCCAFF48910523053"
SOURCE_PREFIX = "ccmusic-database-demo/cpop/"
EXPECTED_MEMBERS = {
    "Yueding accompaniment.wav",
    "Yueding vocal tuning.wav",
    "Yueding vocal.wav",
    "Yueding xml-01.wav",
    "Yueding.musicxml",
}
TEMPO_BPM = 80.0
TIME_SIGNATURE = (4, 4)
SEGMENT_QUARTERS = 16
SEGMENT_STARTS = (40, 56, 72, 88, 104)
MIDI_TICKS_PER_QUARTER = 480
OUTPUT_SAMPLE_RATE = 48_000
ONSET_FRAME_SEC = 0.04
LATENCY_SEARCH_SEC = 2.0
VOCAL_GAIN = 0.75
ALIGNMENT_SAMPLE_RATE = 22_050
ALIGNMENT_FRAME_SEC = 0.05
ALIGNMENT_ANCHOR_COUNT = 6
ALIGNMENT_ANCHOR_SEARCH_SEC = 0.75
ALIGNMENT_LOCAL_WINDOW_SEC = 12.0
ALIGNMENT_GUIDE_SHIFT_MARGIN_SEC = 1.25
ALIGNMENT_DTW_MARGIN_SEC = 0.75
ALIGNMENT_SHIFT_STEP_SEC = 0.05
ALIGNMENT_SEED = 20260907
RENDERER_VERSION = "ccmusic_aligned_mix_v2"


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return _hash_file(path, "sha256")


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = Path(normalized)
    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def _member_record(bundle: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    digest = hashlib.sha256()
    with bundle.open(info) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"member": _safe_member_name(info.filename), "bytes": info.file_size, "sha256": digest.hexdigest()}


def _verify_archive(archive: Path) -> tuple[str, dict[str, zipfile.ZipInfo]]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError(
            f"CCMusic archive size mismatch: expected {EXPECTED_ARCHIVE_BYTES}, got {archive.stat().st_size}"
        )
    md5 = _hash_file(archive, "md5").upper()
    sha256 = _sha256(archive).lower()
    if md5 != EXPECTED_MD5:
        raise ValueError(f"CCMusic archive MD5 mismatch: expected {EXPECTED_MD5}, got {md5}")
    if sha256 != EXPECTED_SHA256.lower():
        raise ValueError(f"CCMusic archive SHA-256 mismatch: expected {EXPECTED_SHA256}, got {sha256}")
    with zipfile.ZipFile(archive) as bundle:
        infos: dict[str, zipfile.ZipInfo] = {}
        for info in bundle.infolist():
            normalized = _safe_member_name(info.filename.rstrip("/")) if info.is_dir() else _safe_member_name(info.filename)
            if not info.is_dir() and normalized in infos:
                raise ValueError(f"duplicate archive member after normalization: {normalized}")
            if not info.is_dir():
                infos[normalized] = info
        expected = {f"{SOURCE_PREFIX}{name}" for name in EXPECTED_MEMBERS}
        missing = sorted(expected - infos.keys())
        if missing:
            raise ValueError(f"CCMusic archive is missing expected members: {missing}")
        selected = {name: infos[name] for name in sorted(expected)}
        # Hash selected contents while the archive is still open.  The
        # returned hash is checked again when extraction writes each file.
        records = [_member_record(bundle, selected[name]) for name in sorted(selected)]
    return sha256, {record["member"]: selected[record["member"]] for record in records}


def _extract_members(archive: Path, source_root: Path, *, overwrite: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive_sha256, selected = _verify_archive(archive)
    source_root = source_root.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        for member, info in sorted(selected.items()):
            destination = (source_root / Path(member).name).resolve()
            if not destination.is_relative_to(source_root):
                raise ValueError(f"archive extraction escaped source root: {member}")
            member_hash = _member_record(bundle, info)["sha256"]
            if destination.exists() and not overwrite:
                if _sha256(destination) != member_hash:
                    raise FileExistsError(f"source member exists with a different hash; use --overwrite: {destination}")
            else:
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with bundle.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(destination)
            output_hash = _sha256(destination)
            if output_hash != member_hash:
                raise ValueError(f"extracted member hash mismatch: {destination}")
            records.append(
                {
                    "archive_member": member,
                    "member_bytes": info.file_size,
                    "member_sha256": member_hash,
                    "path": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": output_hash,
                }
            )
    return {"path": str(archive.resolve()), "bytes": archive.stat().st_size, "md5": EXPECTED_MD5, "sha256": archive_sha256}, records


def _audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": float(info.frames / info.samplerate),
        "subtype": info.subtype,
    }


def _onset_feature(path: Path, *, frame_sec: float = ONSET_FRAME_SEC) -> tuple[np.ndarray, float]:
    samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.asarray(samples, dtype=np.float32).mean(axis=1)
    frame_size = max(1, round(sample_rate * frame_sec))
    frame_count = len(mono) // frame_size
    if frame_count < 2:
        raise ValueError(f"audio is too short for onset analysis: {path}")
    framed = mono[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    onset = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    onset = (onset - float(onset.mean())) / (float(onset.std()) + 1e-9)
    return onset.astype(np.float64), float(sample_rate / frame_size)


def estimate_vocal_guide_latency(vocal_path: Path, guide_path: Path) -> dict[str, Any]:
    """Estimate one global shift that overlays tuned vocal on XML guide.

    ``correlation_lag_sec`` is the positive lag in ``vocal[t + lag]`` versus
    ``guide[t]``.  ``vocal_to_guide_shift_sec`` is the actual shift to add to
    vocal timestamps; a negative value means the tuned vocal is late relative
    to the guide.  This convention records the observed approximately -0.44s
    alignment without hiding the correlation sign.
    """

    vocal, vocal_rate = _onset_feature(vocal_path)
    guide, guide_rate = _onset_feature(guide_path)
    if abs(vocal_rate - guide_rate) > 1e-6:
        raise ValueError(f"onset feature rates differ: vocal={vocal_rate}, guide={guide_rate}")
    count = min(len(vocal), len(guide))
    vocal = vocal[:count]
    guide = guide[:count]
    correlations = correlate(vocal, guide, mode="full", method="fft")
    lags = correlation_lags(len(vocal), len(guide), mode="full")
    allowed = np.abs(lags / vocal_rate) <= LATENCY_SEARCH_SEC
    if not np.any(allowed):
        raise ValueError("onset correlation produced no allowed latency candidates")
    candidate_indices = np.flatnonzero(allowed)
    index = int(candidate_indices[np.argmax(correlations[allowed])])
    lag_sec = float(lags[index] / vocal_rate)
    score = float(correlations[index] / max(1, count))
    return {
        "method": "normalized_rms_onset_cross_correlation",
        "frame_sec": ONSET_FRAME_SEC,
        "search_window_sec": LATENCY_SEARCH_SEC,
        "correlation_lag_sec": lag_sec,
        "vocal_to_guide_shift_sec": -lag_sec,
        "interpretation": "relative tuned-vocal versus XML-guide diagnostic; negative shift means the tuned vocal is late relative to the guide",
        "application": "diagnostic_only_not_applied_to_mix_placement",
        "used_for_mix_placement": False,
        "correlation_score": score,
        "vocal_feature_frames": len(vocal),
        "guide_feature_frames": len(guide),
    }


def _first_non_grace_note(payload: Any) -> Any:
    events = [event for part in payload.parts for event in part.events if event.kind in {"note", "chord"} and event.pitches and not event.grace]
    if not events:
        raise ValueError("MusicXML has no non-grace pitched event")
    return min(events, key=lambda event: (event.offset_quarter, event.event_id))


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _alignment_features(path: Path) -> dict[str, Any]:
    """Load deterministic chroma/onset features for cross-recording alignment."""

    samples, sample_rate = librosa.load(os.fspath(path), sr=ALIGNMENT_SAMPLE_RATE, mono=True)
    hop_length = max(1, round(sample_rate * ALIGNMENT_FRAME_SEC))
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sample_rate, hop_length=hop_length, n_chroma=12)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    onset = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=hop_length)
    onset = (onset - float(np.mean(onset))) / (float(np.std(onset)) + 1e-9)
    onset = np.clip(onset, -3.0, 3.0)
    frame_count = min(chroma.shape[1], onset.size)
    if frame_count < 8:
        raise ValueError(f"audio is too short for chroma/onset alignment: {path}")
    return {
        "chroma": np.asarray(chroma[:, :frame_count], dtype=np.float64),
        "onset": np.asarray(onset[:frame_count], dtype=np.float64),
        "frame_sec": float(hop_length / sample_rate),
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
        "duration_sec": float(len(samples) / sample_rate),
    }


def _score_alignment_events(payload: Any) -> dict[str, np.ndarray]:
    starts: list[float] = []
    durations: list[float] = []
    vectors: list[np.ndarray] = []
    for part in payload.parts:
        for event in part.events:
            kind = _event_value(event, "kind")
            pitches = _event_value(event, "pitches") or []
            if kind not in {"note", "chord"} or not pitches or _event_value(event, "grace", False):
                continue
            starts.append(float(_event_value(event, "offset_quarter")))
            durations.append(float(_event_value(event, "duration_quarter")))
            vector = np.zeros(12, dtype=np.float64)
            for pitch in pitches:
                vector[int(pitch) % 12] = 1.0
            vectors.append(vector)
    if len(starts) < 8:
        raise ValueError("MusicXML has too few pitched events for audio alignment")
    order = np.argsort(np.asarray(starts, dtype=np.float64), kind="stable")
    return {
        "start_quarter": np.asarray(starts, dtype=np.float64)[order],
        "duration_quarter": np.asarray(durations, dtype=np.float64)[order],
        "pitch_class": np.asarray(vectors, dtype=np.float64)[order],
    }


def _score_chroma_affine(
    scale_offset: Sequence[float],
    events: Mapping[str, np.ndarray],
    features: Mapping[str, Any],
) -> float:
    scale, offset = (float(value) for value in scale_offset)
    centers = events["start_quarter"] + events["duration_quarter"] / 2.0
    times = scale * centers + offset
    frame_sec = float(features["frame_sec"])
    chroma = np.asarray(features["chroma"])
    valid = (times >= 0.0) & (times <= (chroma.shape[1] - 1) * frame_sec)
    if int(np.count_nonzero(valid)) < max(8, int(0.8 * len(centers))):
        return -10.0
    frame_times = np.arange(chroma.shape[1], dtype=np.float64) * frame_sec
    audio = np.stack([np.interp(times[valid], frame_times, chroma[index]) for index in range(12)], axis=1)
    audio = audio / (np.linalg.norm(audio, axis=1, keepdims=True) + 1e-9)
    reference = events["pitch_class"][valid]
    similarities = np.sum(audio * reference, axis=1) / (np.linalg.norm(reference, axis=1) + 1e-9)
    trim = max(1, len(similarities) // 10)
    trimmed = np.sort(similarities)[trim:-trim] if len(similarities) > 2 * trim else similarities
    return float(0.6 * np.median(similarities) + 0.4 * np.mean(trimmed))


def _fit_score_audio_affine(payload: Any, audio_path: Path, *, label: str) -> dict[str, Any]:
    """Fit score-quarter to audio seconds from multiple pitch-class anchors."""

    events = _score_alignment_events(payload)
    features = _alignment_features(audio_path)
    score_start = float(np.min(events["start_quarter"]))
    score_end = float(np.max(events["start_quarter"] + events["duration_quarter"]))
    initial_scale = float(features["duration_sec"] / max(score_end - score_start, 1e-6))
    scale_bounds = (max(0.2, initial_scale * 0.7), min(2.0, initial_scale * 1.3))
    offset_bounds = (-float(features["duration_sec"]), float(features["duration_sec"]))
    global_fit = differential_evolution(
        lambda values: -_score_chroma_affine(values, events, features),
        [scale_bounds, offset_bounds],
        seed=ALIGNMENT_SEED,
        maxiter=80,
        popsize=12,
        tol=1e-7,
        polish=True,
    )
    global_scale, global_offset = (float(value) for value in global_fit.x)
    edges = np.linspace(score_start, score_end, ALIGNMENT_ANCHOR_COUNT + 1)
    anchors: list[dict[str, Any]] = []
    scale_window = max(0.01, min(0.03, initial_scale * 0.05))
    for index in range(ALIGNMENT_ANCHOR_COUNT):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (events["start_quarter"] >= lower) & (
            events["start_quarter"] < upper if index + 1 < ALIGNMENT_ANCHOR_COUNT else events["start_quarter"] <= upper
        )
        local_events = {key: value[mask] for key, value in events.items()}
        if len(local_events["start_quarter"]) < 4:
            continue
        local_fit = differential_evolution(
            lambda values: -_score_chroma_affine(values, local_events, features),
            [
                (max(scale_bounds[0], global_scale - scale_window), min(scale_bounds[1], global_scale + scale_window)),
                (global_offset - ALIGNMENT_ANCHOR_SEARCH_SEC, global_offset + ALIGNMENT_ANCHOR_SEARCH_SEC),
            ],
            seed=ALIGNMENT_SEED + index + 1,
            maxiter=50,
            popsize=8,
            tol=1e-7,
            polish=True,
        )
        local_scale, local_offset = (float(value) for value in local_fit.x)
        centers = local_events["start_quarter"] + local_events["duration_quarter"] / 2.0
        anchor_quarter = float(np.median(centers))
        anchors.append(
            {
                "score_quarter": anchor_quarter,
                "audio_sec": local_scale * anchor_quarter + local_offset,
                "local_scale_sec_per_quarter": local_scale,
                "local_offset_sec": local_offset,
                "feature_score": float(-local_fit.fun),
            }
        )
    if len(anchors) < 3:
        raise ValueError(f"{label} score/audio alignment produced too few anchors")
    anchor_quarters = np.asarray([item["score_quarter"] for item in anchors], dtype=np.float64)
    anchor_times = np.asarray([item["audio_sec"] for item in anchors], dtype=np.float64)
    robust_fit = theilslopes(anchor_times, anchor_quarters)
    scale = float(robust_fit.slope)
    offset = float(robust_fit.intercept)
    residuals = anchor_times - (scale * anchor_quarters + offset)
    residual_mad = float(median_abs_deviation(residuals, scale=1.0))
    residual_limit = max(0.05, 3.0 * 1.4826 * residual_mad)
    for item, residual in zip(anchors, residuals):
        item["residual_sec"] = float(residual)
        item["inlier"] = bool(abs(float(residual)) <= residual_limit)
    inlier_count = sum(bool(item["inlier"]) for item in anchors)
    score = _score_chroma_affine((scale, offset), events, features)
    if inlier_count < max(3, len(anchors) // 2) or score < 0.35:
        raise ValueError(f"{label} score/audio alignment is weak: anchors={inlier_count}/{len(anchors)}, score={score:.3f}")
    return {
        "method": "musicxml_pitch_class_chroma_multi_anchor_theil_sen_affine",
        "label": label,
        "feature": {
            "kind": "audio_chroma_cqt",
            "sample_rate": features["sample_rate"],
            "hop_length": features["hop_length"],
            "frame_sec": features["frame_sec"],
            "audio_duration_sec": features["duration_sec"],
        },
        "slope_sec_per_quarter": scale,
        "offset_sec": offset,
        "score_quarter_range": [score_start, score_end],
        "guide_zero_score_quarter": float(-offset / scale) if label == "guide" else None,
        "feature_score": float(score),
        "anchor_count": len(anchors),
        "inlier_count": inlier_count,
        "residual_mad_sec": residual_mad,
        "residual_limit_sec": residual_limit,
        "confidence": {
            "level": "high" if inlier_count >= 5 and score >= 0.75 and residual_mad <= 0.05 else "medium",
            "basis": "multi-anchor pitch-class chroma score and robust affine residuals",
        },
        "anchors": anchors,
    }


def _joint_audio_window_score(
    guide: Mapping[str, Any],
    accompaniment: Mapping[str, Any],
    *,
    guide_start_sec: float,
    window_sec: float,
    shift_sec: float,
) -> tuple[float, float, float]:
    frame_sec = float(guide["frame_sec"])
    guide_start = round(guide_start_sec / frame_sec)
    accompaniment_start = round((guide_start_sec + shift_sec) / frame_sec)
    frame_count = round(window_sec / frame_sec)
    if guide_start < 0 or accompaniment_start < 0:
        return -10.0, 0.0, 0.0
    if guide_start + frame_count > guide["chroma"].shape[1] or accompaniment_start + frame_count > accompaniment["chroma"].shape[1]:
        return -10.0, 0.0, 0.0
    chroma_similarity = np.sum(
        guide["chroma"][:, guide_start : guide_start + frame_count]
        * accompaniment["chroma"][:, accompaniment_start : accompaniment_start + frame_count],
        axis=0,
    )
    trim = max(1, len(chroma_similarity) // 10)
    sorted_similarity = np.sort(chroma_similarity)
    chroma_score = float(np.mean(sorted_similarity[trim:-trim])) if len(sorted_similarity) > 2 * trim else float(np.mean(sorted_similarity))
    onset_score = float(
        np.corrcoef(
            guide["onset"][guide_start : guide_start + frame_count],
            accompaniment["onset"][accompaniment_start : accompaniment_start + frame_count],
        )[0, 1]
    )
    if not np.isfinite(onset_score):
        onset_score = 0.0
    return 0.75 * chroma_score + 0.25 * ((onset_score + 1.0) / 2.0), chroma_score, onset_score


def _dtw_window_anchor(
    guide: Mapping[str, Any],
    accompaniment: Mapping[str, Any],
    *,
    guide_start_sec: float,
    window_sec: float,
    coarse_shift_sec: float,
) -> dict[str, Any] | None:
    frame_sec = float(guide["frame_sec"])
    guide_start = round(guide_start_sec / frame_sec)
    guide_frames = round(window_sec / frame_sec)
    accompaniment_start_sec = guide_start_sec + coarse_shift_sec - ALIGNMENT_DTW_MARGIN_SEC
    accompaniment_start = round(accompaniment_start_sec / frame_sec)
    accompaniment_frames = round((window_sec + 2.0 * ALIGNMENT_DTW_MARGIN_SEC) / frame_sec)
    if guide_start < 0 or accompaniment_start < 0:
        return None
    if guide_start + guide_frames > guide["chroma"].shape[1] or accompaniment_start + accompaniment_frames > accompaniment["chroma"].shape[1]:
        return None
    guide_matrix = np.vstack(
        (guide["chroma"][:, guide_start : guide_start + guide_frames], 0.5 * guide["onset"][guide_start : guide_start + guide_frames])
    )
    accompaniment_matrix = np.vstack(
        (
            accompaniment["chroma"][:, accompaniment_start : accompaniment_start + accompaniment_frames],
            0.5 * accompaniment["onset"][accompaniment_start : accompaniment_start + accompaniment_frames],
        )
    )
    try:
        costs, path = librosa.sequence.dtw(X=guide_matrix, Y=accompaniment_matrix, metric="cosine", subseq=True, backtrack=True)
    except (ValueError, IndexError):
        return None
    path = path[::-1]
    guide_times = guide_start_sec + path[:, 0] * frame_sec
    accompaniment_times = accompaniment_start_sec + path[:, 1] * frame_sec
    center = guide_start_sec + window_sec / 2.0
    center_index = int(np.argmin(np.abs(guide_times - center)))
    offset = float(accompaniment_times[center_index] - guide_times[center_index])
    return {
        "guide_sec": float(guide_times[center_index]),
        "accompaniment_sec": float(accompaniment_times[center_index]),
        "dtw_shift_sec": offset,
        "dtw_cost": float(costs[-1, path[-1, 1]] / max(1, len(path))),
        "path_offset_start_sec": float(accompaniment_times[0] - guide_times[0]),
        "path_offset_end_sec": float(accompaniment_times[-1] - guide_times[-1]),
    }


def _fit_guide_accompaniment_alignment(guide_path: Path, accompaniment_path: Path) -> dict[str, Any]:
    guide = _alignment_features(guide_path)
    accompaniment = _alignment_features(accompaniment_path)
    if abs(float(guide["frame_sec"]) - float(accompaniment["frame_sec"])) > 1e-9:
        raise ValueError("guide/accompaniment feature frame rates differ")
    maximum_shift = max(0.0, float(accompaniment["duration_sec"] - guide["duration_sec"]))
    coarse_values = np.arange(0.0, maximum_shift + 0.5, 0.1)
    coarse = [
        (float(_joint_audio_window_score(guide, accompaniment, guide_start_sec=0.0, window_sec=guide["duration_sec"], shift_sec=float(shift))[0]), float(shift))
        for shift in coarse_values
    ]
    coarse_score, coarse_shift = max(coarse)
    window_sec = min(ALIGNMENT_LOCAL_WINDOW_SEC, max(4.0, float(guide["duration_sec"]) / 4.0))
    start_limit = max(0.0, float(guide["duration_sec"] - window_sec))
    starts = np.linspace(0.0, start_limit, min(11, max(3, round(start_limit / 8.0) + 1)))
    anchors: list[dict[str, Any]] = []
    for start in starts:
        candidates = []
        for shift in np.arange(coarse_shift - ALIGNMENT_GUIDE_SHIFT_MARGIN_SEC, coarse_shift + ALIGNMENT_GUIDE_SHIFT_MARGIN_SEC + ALIGNMENT_SHIFT_STEP_SEC / 2.0, ALIGNMENT_SHIFT_STEP_SEC):
            score, chroma_score, onset_score = _joint_audio_window_score(
                guide,
                accompaniment,
                guide_start_sec=float(start),
                window_sec=window_sec,
                shift_sec=float(shift),
            )
            candidates.append((score, float(shift), chroma_score, onset_score))
        direct_score, direct_shift, chroma_score, onset_score = max(candidates)
        dtw_anchor = _dtw_window_anchor(
            guide,
            accompaniment,
            guide_start_sec=float(start),
            window_sec=window_sec,
            coarse_shift_sec=direct_shift,
        )
        if dtw_anchor is None:
            continue
        dtw_anchor.update(
            {
                "guide_window_start_sec": float(start),
                "guide_window_duration_sec": float(window_sec),
                "direct_shift_sec": direct_shift,
                "direct_score": direct_score,
                "chroma_score": chroma_score,
                "onset_score": onset_score,
                "accepted": bool(abs(float(dtw_anchor["dtw_shift_sec"]) - direct_shift) <= ALIGNMENT_DTW_MARGIN_SEC),
            }
        )
        if dtw_anchor["accepted"]:
            anchors.append(dtw_anchor)
    if len(anchors) < 3:
        raise ValueError(f"guide/accompaniment alignment produced too few DTW anchors: {len(anchors)}")
    guide_times = np.asarray([item["guide_sec"] for item in anchors], dtype=np.float64)
    accompaniment_times = np.asarray([item["accompaniment_sec"] for item in anchors], dtype=np.float64)
    robust_fit = theilslopes(accompaniment_times, guide_times)
    scale = float(robust_fit.slope)
    offset = float(robust_fit.intercept)
    residuals = accompaniment_times - (scale * guide_times + offset)
    residual_mad = float(median_abs_deviation(residuals, scale=1.0))
    residual_limit = max(0.1, 3.0 * 1.4826 * residual_mad)
    for item, residual in zip(anchors, residuals):
        item["affine_residual_sec"] = float(residual)
        item["affine_inlier"] = bool(abs(float(residual)) <= residual_limit)
    inlier_count = sum(bool(item["affine_inlier"]) for item in anchors)
    if inlier_count < max(3, len(anchors) // 2):
        raise ValueError(f"guide/accompaniment affine alignment is weak: {inlier_count}/{len(anchors)} inliers")
    median_direct_score = float(np.median([item["direct_score"] for item in anchors]))
    median_dtw_cost = float(np.median([item["dtw_cost"] for item in anchors]))
    return {
        "method": "chroma_onset_local_dtw_multi_anchor_theil_sen_affine",
        "feature": {
            "kind": "chroma_cqt_plus_onset_strength",
            "sample_rate": guide["sample_rate"],
            "hop_length": guide["hop_length"],
            "frame_sec": guide["frame_sec"],
            "guide_duration_sec": guide["duration_sec"],
            "accompaniment_duration_sec": accompaniment["duration_sec"],
        },
        "coarse_shift_sec": coarse_shift,
        "coarse_score": coarse_score,
        "slope_accompaniment_sec_per_guide_sec": scale,
        "offset_sec": offset,
        "anchor_count": len(anchors),
        "inlier_count": inlier_count,
        "residual_mad_sec": residual_mad,
        "residual_limit_sec": residual_limit,
        "confidence": {
            "level": "high" if inlier_count >= 5 and median_direct_score >= 0.5 and residual_mad <= 0.3 else "medium",
            "basis": "six or more local chroma/onset DTW anchors and robust affine residuals",
            "median_direct_score": median_direct_score,
            "median_dtw_cost": median_dtw_cost,
        },
        "anchors": anchors,
    }


def _compose_affine(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slope_sec_per_quarter": float(second["slope_accompaniment_sec_per_guide_sec"] * first["slope_sec_per_quarter"]),
        "offset_sec": float(second["slope_accompaniment_sec_per_guide_sec"] * first["offset_sec"] + second["offset_sec"]),
    }


def _estimate_direct_vocal_accompaniment_offset(vocal_path: Path, accompaniment_path: Path, center_sec: float) -> dict[str, Any]:
    vocal = _alignment_features(vocal_path)
    accompaniment = _alignment_features(accompaniment_path)
    candidates = []
    for offset in np.arange(center_sec - 2.0, center_sec + 2.0 + ALIGNMENT_SHIFT_STEP_SEC / 2.0, ALIGNMENT_SHIFT_STEP_SEC):
        score, chroma_score, onset_score = _joint_audio_window_score(
            vocal,
            accompaniment,
            guide_start_sec=0.0,
            window_sec=vocal["duration_sec"],
            shift_sec=float(offset),
        )
        candidates.append((score, float(offset), chroma_score, onset_score))
    score, offset, chroma_score, onset_score = max(candidates)
    return {
        "method": "full_recording_chroma_onset_offset_validation",
        "offset_sec": offset,
        "score": score,
        "chroma_score": chroma_score,
        "onset_score": onset_score,
        "search_center_sec": center_sec,
        "search_window_sec": 2.0,
    }


def _estimate_ccmusic_alignment(
    payload: Any,
    *,
    guide_path: Path,
    vocal_path: Path,
    accompaniment_path: Path,
) -> dict[str, Any]:
    score_to_guide = _fit_score_audio_affine(payload, guide_path, label="guide")
    score_to_vocal = _fit_score_audio_affine(payload, vocal_path, label="tuned_vocal")
    guide_to_accompaniment = _fit_guide_accompaniment_alignment(guide_path, accompaniment_path)
    score_to_accompaniment = _compose_affine(score_to_guide, guide_to_accompaniment)
    score_quarters = np.asarray([item["score_quarter"] for item in score_to_vocal["anchors"]], dtype=np.float64)
    placement_offsets = [
        (score_to_accompaniment["slope_sec_per_quarter"] * float(quarter) + score_to_accompaniment["offset_sec"])
        - (score_to_vocal["slope_sec_per_quarter"] * float(quarter) + score_to_vocal["offset_sec"])
        for quarter in score_quarters
    ]
    vocal_mix_offset = float(np.median(placement_offsets))
    placement_mad = float(median_abs_deviation(placement_offsets, scale=1.0))
    direct_validation = _estimate_direct_vocal_accompaniment_offset(vocal_path, accompaniment_path, vocal_mix_offset)
    placement_difference = abs(float(direct_validation["offset_sec"]) - vocal_mix_offset)
    if placement_difference > 0.75:
        raise ValueError(
            "score-derived vocal/accompaniment placement disagrees with direct audio validation: "
            f"{vocal_mix_offset:.3f}s vs {direct_validation['offset_sec']:.3f}s"
        )
    return {
        "score_to_guide": score_to_guide,
        "score_to_vocal": score_to_vocal,
        "guide_to_accompaniment": guide_to_accompaniment,
        "score_to_accompaniment": score_to_accompaniment,
        "guide_zero": {
            "score_quarter": score_to_guide["guide_zero_score_quarter"],
            "interpretation": "guide audio t=0 mapped by MusicXML pitch/chroma affine fit; it is not score quarter zero",
        },
        "placement": {
            "method": "score_pitch_chroma_affine_difference_plus_guide_accompaniment_dtw",
            "vocal_mix_offset_sec": round(vocal_mix_offset, 6),
            "score_anchor_quarters": [float(value) for value in score_quarters],
            "anchor_offsets_sec": [float(value) for value in placement_offsets],
            "residual_mad_sec": placement_mad,
            "confidence": {
                "level": "high" if placement_difference <= 0.1 and placement_mad <= 0.1 else "medium",
                "basis": "score-anchor offset spread and independent full-recording chroma validation",
                "direct_difference_sec": placement_difference,
            },
            "direct_audio_validation": direct_validation,
            "latency_application": "vocal_to_guide_latency_is_recorded_separately_and_not_applied_to_mix_placement",
            "used_vocal_to_guide_latency": False,
        },
    }


def _load_and_mix(accompaniment_path: Path, vocal_path: Path, offset_sec: float, output: Path) -> dict[str, Any]:
    accompaniment, accompaniment_rate = sf.read(accompaniment_path, always_2d=True, dtype="float32")
    vocal, vocal_rate = sf.read(vocal_path, always_2d=True, dtype="float32")
    if accompaniment_rate != OUTPUT_SAMPLE_RATE:
        raise ValueError(f"CCMusic accompaniment sample rate must be {OUTPUT_SAMPLE_RATE}, got {accompaniment_rate}")
    if vocal.shape[1] == 1 and accompaniment.shape[1] == 2:
        vocal = np.repeat(vocal, 2, axis=1)
    if vocal.shape[1] != accompaniment.shape[1]:
        raise ValueError(f"accompaniment/vocal channel mismatch: {accompaniment.shape[1]} vs {vocal.shape[1]}")
    if vocal_rate != accompaniment_rate:
        vocal = resample_poly(vocal, accompaniment_rate, vocal_rate, axis=0)
    offset_frames = round(offset_sec * accompaniment_rate)
    if offset_frames < 0 or offset_frames >= len(accompaniment):
        raise ValueError(f"vocal placement is outside accompaniment: {offset_sec:.3f}s")
    mix = accompaniment.copy()
    end = min(len(mix), offset_frames + len(vocal))
    mix[offset_frames:end] += VOCAL_GAIN * vocal[: end - offset_frames]
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 1.0:
        raise ValueError(f"aligned mix would clip at peak {peak:.6f}; lower the fixed vocal gain")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, mix, accompaniment_rate, subtype="PCM_16", format="WAV")
    return {
        "path": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "sample_rate": int(accompaniment_rate),
        "channels": int(mix.shape[1]),
        "frames": int(mix.shape[0]),
        "duration_sec": float(mix.shape[0] / accompaniment_rate),
        "vocal_gain": VOCAL_GAIN,
        "vocal_offset_sec": offset_sec,
        "vocal_offset_frames": offset_frames,
        "resampler": "scipy.signal.resample_poly",
    }


def _segment_notes(payload: Any, start_quarter: Fraction, end_quarter: Fraction) -> list[tuple[Fraction, Fraction, int]]:
    notes: list[tuple[Fraction, Fraction, int]] = []
    for part in payload.parts:
        for event in part.events:
            if event.kind not in {"note", "chord"} or not event.pitches or event.grace:
                continue
            event_start = Fraction(str(event.offset_quarter))
            event_end = event_start + Fraction(str(event.duration_quarter))
            overlap_start = max(start_quarter, event_start)
            overlap_end = min(end_quarter, event_end)
            if overlap_end <= overlap_start:
                continue
            for pitch in event.pitches:
                notes.append((overlap_start - start_quarter, overlap_end - start_quarter, int(pitch)))
    return sorted(notes, key=lambda item: (item[0], item[2], item[1]))


def _musicxml_tempo(payload: Any) -> float:
    events = getattr(payload, "tempo_events", [])
    if events:
        value = float(_event_value(events[0], "bpm", TEMPO_BPM))
        if np.isfinite(value) and value > 0:
            return value
    return TEMPO_BPM


def _musicxml_time_signature(payload: Any) -> tuple[int, int]:
    events = getattr(payload, "time_signature_events", [])
    if events:
        event = events[0]
        ratio = str(_event_value(event, "ratio", "4/4"))
        try:
            numerator, denominator = (int(value) for value in ratio.split("/", 1))
            if numerator > 0 and denominator > 0:
                return numerator, denominator
        except (TypeError, ValueError):
            pass
    return TIME_SIGNATURE


def _write_reference_midi(
    path: Path,
    notes: Sequence[tuple[Fraction, Fraction, int]],
    *,
    tempo_bpm: float = TEMPO_BPM,
    time_signature: tuple[int, int] = TIME_SIGNATURE,
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=MIDI_TICKS_PER_QUARTER)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="CCMusic Yueding score ground truth", time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=time_signature[0], denominator=time_signature[1], clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    midi.tracks.append(conductor)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="vocal reference", time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for start, end, pitch in notes:
        start_tick = round(float(start) * MIDI_TICKS_PER_QUARTER)
        end_tick = max(start_tick + 1, round(float(end) * MIDI_TICKS_PER_QUARTER))
        events.append((start_tick, 1, mido.Message("note_on", channel=0, note=pitch, velocity=80, time=0)))
        events.append((end_tick, 0, mido.Message("note_off", channel=0, note=pitch, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1], int(getattr(item[2], "note", -1))))
    previous = 0
    for tick, _priority, message in events:
        message.time = max(0, tick - previous)
        track.append(message)
        previous = tick
    midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def _write_beat_grid(
    path: Path,
    start_quarter: int,
    *,
    tempo_bpm: float = TEMPO_BPM,
    time_signature: tuple[int, int] = TIME_SIGNATURE,
    audio_start_sec: float | None = None,
    score_to_audio: Mapping[str, Any] | None = None,
) -> None:
    beats_per_bar = int(time_signature[0])
    beats: list[dict[str, Any]] = []
    for index in range(SEGMENT_QUARTERS + 1):
        beat = {
            "index": index,
            "bar_index": index // beats_per_bar,
            "beat_index": index % beats_per_bar,
            "score_quarter": start_quarter + index,
            "time_sec": index * 60.0 / tempo_bpm,
            "downbeat": index % beats_per_bar == 0,
        }
        beats.append(beat)
    payload = {
        "schema_version": "1.0",
        "source": "ccmusic_musicxml_score_ground_truth",
        "beat_grid": {
            "beats": beats,
            "downbeats": [item for item in beats if item["downbeat"]],
            "time_signature": f"{time_signature[0]}/{time_signature[1]}",
            "tempo_bpm": tempo_bpm,
            "score_start_quarter": start_quarter,
            "score_end_quarter": start_quarter + SEGMENT_QUARTERS,
            "annotation_policy": "MusicXML score timing independent of model output; not a reference-derived recognizer beat",
        },
    }
    if audio_start_sec is not None:
        payload["beat_grid"]["audio_start_sec"] = float(audio_start_sec)
    if score_to_audio is not None:
        payload["beat_grid"]["score_to_audio_alignment"] = dict(score_to_audio)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def prepare_archive(
    archive: Path = DEFAULT_ARCHIVE,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    archive = archive.resolve()
    archive_record, extracted = _extract_members(archive, source_root, overwrite=overwrite)
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = {record["path"]: source_root / record["path"] for record in extracted}
    xml_path = source_paths["Yueding.musicxml"]
    from backend.jianpu_score.high_accuracy import resolve_notation_python
    from backend.jianpu_score.musicxml_standardize import run_musicxml_worker

    worker_payload = run_musicxml_worker(xml_path, notation_python=resolve_notation_python())
    worker_json = worker_payload.model_dump(mode="json")
    worker_json["source_path"] = xml_path.name
    worker_path = output_root / "Yueding.musicxml.worker.json"
    worker_path.write_text(json.dumps(worker_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    first_note = _first_non_grace_note(worker_payload)
    pitched_events = [event for part in worker_payload.parts for event in part.events if event.kind in {"note", "chord"} and event.pitches and not event.grace]
    last_note = max(pitched_events, key=lambda event: (event.offset_quarter + event.duration_quarter, event.event_id))
    tempo_bpm = _musicxml_tempo(worker_payload)
    time_signature = _musicxml_time_signature(worker_payload)
    latency = estimate_vocal_guide_latency(source_paths["Yueding vocal tuning.wav"], source_paths["Yueding xml-01.wav"])
    alignment = _estimate_ccmusic_alignment(
        worker_payload,
        guide_path=source_paths["Yueding xml-01.wav"],
        vocal_path=source_paths["Yueding vocal tuning.wav"],
        accompaniment_path=source_paths["Yueding accompaniment.wav"],
    )
    placement = alignment["placement"]
    full_mix_path = output_root / "Yueding aligned accompaniment vocal.wav"
    full_mix = _load_and_mix(source_paths["Yueding accompaniment.wav"], source_paths["Yueding vocal tuning.wav"], placement["vocal_mix_offset_sec"], full_mix_path)
    cases: list[dict[str, Any]] = []
    for index, start in enumerate(SEGMENT_STARTS, start=1):
        end = start + SEGMENT_QUARTERS
        case_id = f"ccmusic-yueding-{index:02d}"
        case_root = output_root / case_id
        if case_root.exists() and not overwrite:
            existing_manifest = case_root / "case_manifest.json"
            if existing_manifest.is_file():
                cases.append(json.loads(existing_manifest.read_text(encoding="utf-8")))
                continue
            raise FileExistsError(f"segment directory exists without a manifest; use --overwrite: {case_root}")
        case_root.mkdir(parents=True, exist_ok=True)
        start_sec = float(alignment["score_to_accompaniment"]["slope_sec_per_quarter"] * start + alignment["score_to_accompaniment"]["offset_sec"])
        duration_sec = SEGMENT_QUARTERS * 60.0 / tempo_bpm
        nominal_score_start_sec = start * 60.0 / tempo_bpm
        audio, sample_rate = sf.read(full_mix_path, always_2d=True, start=round(start_sec * OUTPUT_SAMPLE_RATE), frames=round(duration_sec * OUTPUT_SAMPLE_RATE), dtype="float32")
        if sample_rate != OUTPUT_SAMPLE_RATE or len(audio) != round(duration_sec * OUTPUT_SAMPLE_RATE):
            raise ValueError(f"aligned mix is too short for {case_id}: start={start_sec}, frames={len(audio)}")
        input_path = case_root / f"{case_id}.wav"
        sf.write(input_path, audio, sample_rate, subtype="PCM_16", format="WAV")
        reference_notes = _segment_notes(worker_payload, Fraction(start), Fraction(end))
        midi_path = case_root / f"{case_id}.mid"
        _write_reference_midi(midi_path, reference_notes, tempo_bpm=tempo_bpm, time_signature=time_signature)
        beat_path = case_root / f"{case_id}.beat_grid.json"
        _write_beat_grid(
            beat_path,
            start,
            tempo_bpm=tempo_bpm,
            time_signature=time_signature,
            audio_start_sec=start_sec,
            score_to_audio=alignment["score_to_accompaniment"],
        )
        case_manifest = {
            "schema_version": "1.0",
            "case_id": case_id,
            "source": "ccmusic-yueding",
            "source_song": "Yueding",
            "score_start_quarter": start,
            "score_end_quarter": end,
            "start_sec": start_sec,
            "audio_start_sec": start_sec,
            "nominal_score_start_sec": nominal_score_start_sec,
            "duration_sec": duration_sec,
            "time_signature": f"{time_signature[0]}/{time_signature[1]}",
            "tempo_bpm": tempo_bpm,
            "score_to_accompaniment_alignment": alignment["score_to_accompaniment"],
            "reference_note_count": len(reference_notes),
            "files": {
                "input_audio": {**_file_record(input_path), "sample_rate": sample_rate, "channels": int(audio.shape[1]), "frames": int(len(audio))},
                "reference_midi": {**_file_record(midi_path), "ticks_per_beat": MIDI_TICKS_PER_QUARTER},
                "beat_annotation": _file_record(beat_path),
            },
        }
        case_manifest_path = case_root / "case_manifest.json"
        case_manifest_path.write_text(json.dumps(case_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cases.append(case_manifest)
    manifest = {
        "schema_version": "1.0",
        "generator_version": RENDERER_VERSION,
        "source": {
            "name": "CCMusic database demo / cpop Yueding",
            "archive": archive_record,
            "license_note": "Official Zenodo demo is free for computational musicology; the record exposes no SPDX license identifier.",
            "selected_members": extracted,
            "source_root": str(source_root),
        },
        "musicxml": {
            "file": xml_path.name,
            "worker": "music21 isolated worker",
            "worker_schema_version": worker_payload.schema_version,
            "music21_version": worker_payload.music21_version,
            "worker_payload": _file_record(worker_path),
            "tempo_bpm": tempo_bpm,
            "time_signature": f"{time_signature[0]}/{time_signature[1]}",
            "first_non_grace_note_quarter": float(first_note.offset_quarter),
            "last_pitched_note_end_quarter": float(last_note.offset_quarter + last_note.duration_quarter),
            "pitched_event_count": len(pitched_events),
        },
        "alignment": {
            "latency": latency,
            "score_to_guide": alignment["score_to_guide"],
            "score_to_vocal": alignment["score_to_vocal"],
            "guide_to_accompaniment": alignment["guide_to_accompaniment"],
            "score_to_accompaniment": alignment["score_to_accompaniment"],
            "guide_zero": alignment["guide_zero"],
            "placement": placement,
            "full_mix": full_mix,
            "vocal_source": _audio_info(source_paths["Yueding vocal tuning.wav"]),
            "guide_source": _audio_info(source_paths["Yueding xml-01.wav"]),
            "accompaniment_source": _audio_info(source_paths["Yueding accompaniment.wav"]),
        },
        "segment_policy": {
            "score_starts_quarter": list(SEGMENT_STARTS),
            "segment_duration_quarter": SEGMENT_QUARTERS,
            "segment_duration_sec": SEGMENT_QUARTERS * 60.0 / tempo_bpm,
            "beat_annotation": "MusicXML score timing with the tempo/time signature read from the score; independent of model output",
            "audio_crop_policy": "crop each score window at the fitted score-to-accompaniment affine mapping; preserve the independent score beat grid",
            "reference_policy": "cropped vocal-score MIDI; notes crossing a segment boundary are clipped at the boundary",
        },
        "cases": cases,
    }
    manifest_path = output_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = prepare_archive(args.archive, source_root=args.source_root, output_root=args.output_root, overwrite=args.overwrite)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "archive_sha256": manifest["source"]["archive"]["sha256"], "cases": [case["case_id"] for case in manifest["cases"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
