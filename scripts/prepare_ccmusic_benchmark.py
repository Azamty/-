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
import shutil
import sys
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import mido
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, find_peaks, resample_poly

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
VOCAL_ONSET_GUARD_SEC = 3.0
VOCAL_GAIN = 0.75
RENDERER_VERSION = "ccmusic_aligned_mix_v1"


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
        "correlation_score": score,
        "vocal_feature_frames": len(vocal),
        "guide_feature_frames": len(guide),
    }


def _first_non_grace_note(payload: Any) -> Any:
    events = [event for part in payload.parts for event in part.events if event.kind in {"note", "chord"} and event.pitches and not event.grace]
    if not events:
        raise ValueError("MusicXML has no non-grace pitched event")
    return min(events, key=lambda event: (event.offset_quarter, event.event_id))


def _estimate_vocal_mix_offset(payload: Any, vocal_path: Path) -> dict[str, Any]:
    first_note = _first_non_grace_note(payload)
    first_note_sec = float(first_note.offset_quarter) * 60.0 / TEMPO_BPM
    onset, rate = _onset_feature(vocal_path)
    start_index = max(0, round(VOCAL_ONSET_GUARD_SEC * rate))
    peaks, _ = find_peaks(onset, distance=max(1, round(0.12 * rate)), prominence=max(float(onset.std()) * 0.25, 1e-6))
    candidates = [int(index) for index in peaks if index >= start_index]
    if not candidates:
        raise ValueError("could not find a stable tuned-vocal onset after the pre-roll guard")
    onset_index = candidates[0]
    onset_sec = float(onset_index / rate)
    offset_sec = first_note_sec - onset_sec
    if offset_sec < 0 or offset_sec > 60:
        raise ValueError(f"implausible tuned-vocal placement offset: {offset_sec:.3f}s")
    return {
        "method": "first_non_grace_musicxml_note_minus_first_stable_vocal_onset",
        "pre_roll_guard_sec": VOCAL_ONSET_GUARD_SEC,
        "first_musicxml_note": {"quarter": float(first_note.offset_quarter), "midi": list(first_note.pitches)},
        "first_musicxml_note_sec": first_note_sec,
        "first_stable_vocal_onset_sec": onset_sec,
        "vocal_mix_offset_sec": offset_sec,
        "observed_alignment_reference_sec": 26.285,
        "observed_alignment_tolerance_sec": 0.2,
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


def _write_reference_midi(path: Path, notes: Sequence[tuple[Fraction, Fraction, int]]) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=MIDI_TICKS_PER_QUARTER)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="CCMusic Yueding score ground truth", time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(TEMPO_BPM), time=0))
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


def _write_beat_grid(path: Path, start_quarter: int) -> None:
    beats: list[dict[str, Any]] = []
    for index in range(SEGMENT_QUARTERS + 1):
        beat = {
            "index": index,
            "bar_index": index // TIME_SIGNATURE[0],
            "beat_index": index % TIME_SIGNATURE[0],
            "score_quarter": start_quarter + index,
            "time_sec": index * 60.0 / TEMPO_BPM,
            "downbeat": index % TIME_SIGNATURE[0] == 0,
        }
        beats.append(beat)
    payload = {
        "schema_version": "1.0",
        "source": "ccmusic_musicxml_score_ground_truth",
        "beat_grid": {
            "beats": beats,
            "downbeats": [item for item in beats if item["downbeat"]],
            "time_signature": "4/4",
            "tempo_bpm": TEMPO_BPM,
            "score_start_quarter": start_quarter,
            "score_end_quarter": start_quarter + SEGMENT_QUARTERS,
            "annotation_policy": "MusicXML score timing independent of model output; not a reference-derived recognizer beat",
        },
    }
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
    latency = estimate_vocal_guide_latency(source_paths["Yueding vocal tuning.wav"], source_paths["Yueding xml-01.wav"])
    placement = _estimate_vocal_mix_offset(worker_payload, source_paths["Yueding vocal tuning.wav"])
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
        start_sec = start * 60.0 / TEMPO_BPM
        duration_sec = SEGMENT_QUARTERS * 60.0 / TEMPO_BPM
        audio, sample_rate = sf.read(full_mix_path, always_2d=True, start=round(start_sec * OUTPUT_SAMPLE_RATE), frames=round(duration_sec * OUTPUT_SAMPLE_RATE), dtype="float32")
        if sample_rate != OUTPUT_SAMPLE_RATE or len(audio) != round(duration_sec * OUTPUT_SAMPLE_RATE):
            raise ValueError(f"aligned mix is too short for {case_id}: start={start_sec}, frames={len(audio)}")
        input_path = case_root / f"{case_id}.wav"
        sf.write(input_path, audio, sample_rate, subtype="PCM_16", format="WAV")
        reference_notes = _segment_notes(worker_payload, Fraction(start), Fraction(end))
        midi_path = case_root / f"{case_id}.mid"
        _write_reference_midi(midi_path, reference_notes)
        beat_path = case_root / f"{case_id}.beat_grid.json"
        _write_beat_grid(beat_path, start)
        case_manifest = {
            "schema_version": "1.0",
            "case_id": case_id,
            "source": "ccmusic-yueding",
            "source_song": "Yueding",
            "score_start_quarter": start,
            "score_end_quarter": end,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "time_signature": "4/4",
            "tempo_bpm": TEMPO_BPM,
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
            "tempo_bpm": TEMPO_BPM,
            "time_signature": "4/4",
            "first_non_grace_note_quarter": float(first_note.offset_quarter),
            "last_pitched_note_end_quarter": float(last_note.offset_quarter + last_note.duration_quarter),
            "pitched_event_count": len(pitched_events),
        },
        "alignment": {
            "latency": latency,
            "placement": placement,
            "full_mix": full_mix,
            "vocal_source": _audio_info(source_paths["Yueding vocal tuning.wav"]),
            "guide_source": _audio_info(source_paths["Yueding xml-01.wav"]),
            "accompaniment_source": _audio_info(source_paths["Yueding accompaniment.wav"]),
        },
        "segment_policy": {
            "score_starts_quarter": list(SEGMENT_STARTS),
            "segment_duration_quarter": SEGMENT_QUARTERS,
            "segment_duration_sec": SEGMENT_QUARTERS * 60.0 / TEMPO_BPM,
            "beat_annotation": "MusicXML score timing with exact 80 BPM 4/4 beat/downbeat grid; independent of model output",
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
