"""Pilot deterministic MuseScore Basic SoundFont rendering.

The pilot deliberately does not alter the benchmark registry.  It renders a
fixed set of source MIDI files twice with the pinned MuseScore installation,
exports the imported MIDI once for source-preservation checks, and records
audio and timing diagnostics.  A successful byte-for-byte export is not by
itself sufficient: MuseScore's MIDI importer must also preserve the source
note multiset and timing within the declared tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import mido
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUSESCORE = ROOT / "tools" / "musescore-4.7.4" / "MuseScore 4" / "bin" / "MuseScore4.exe"
DEFAULT_SOUNDFONT = ROOT / "tools" / "musescore-4.7.4" / "MuseScore 4" / "sound" / "MS Basic.sf3"
DEFAULT_PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "musescore-soundfont-pilot-v1"
DEFAULT_CASE_IDS = (
    "synthetic-bass-02",
    "special-triplet",
    "special-complex-chord",
    "maestro-midi-07",
    "synthetic-piano-01",
    "maestro-midi-01",
)
REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
TIMING_TOLERANCE_SEC = 0.010


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    midi: Path
    reference_audio: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(case_ids: Iterable[str]) -> tuple[PilotCase, ...]:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry = {str(item["id"]): item for item in payload["cases"]}
    cases: list[PilotCase] = []
    for case_id in case_ids:
        if case_id not in registry:
            raise ValueError(f"case is not present in registry: {case_id}")
        item = registry[case_id]
        midi = (ROOT / str(item["reference_midi"])).resolve()
        audio = (ROOT / str(item["input"])).resolve()
        if not midi.is_file():
            raise FileNotFoundError(f"reference MIDI not found for {case_id}: {midi}")
        if not audio.is_file():
            raise FileNotFoundError(f"reference audio not found for {case_id}: {audio}")
        cases.append(PilotCase(case_id, midi, audio))
    return tuple(cases)


def _tempo_segments(mid: mido.MidiFile) -> tuple[list[tuple[int, float, int]], list[int]]:
    changes: dict[int, int] = {0: 500000}
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                changes[tick] = int(message.tempo)
    ordered = sorted(changes.items())
    segments: list[tuple[int, float, int]] = []
    current_tick, current_sec, current_tempo = ordered[0][0], 0.0, ordered[0][1]
    segments.append((current_tick, current_sec, current_tempo))
    for tick, tempo in ordered[1:]:
        current_sec += (tick - current_tick) / mid.ticks_per_beat * current_tempo / 1_000_000
        current_tick, current_tempo = tick, tempo
        segments.append((current_tick, current_sec, current_tempo))
    return segments, [item[0] for item in segments]


def _tick_to_seconds(mid: mido.MidiFile, segments: list[tuple[int, float, int]], starts: list[int], tick: int) -> float:
    index = bisect_right(starts, int(tick)) - 1
    start_tick, start_sec, tempo = segments[max(0, index)]
    return start_sec + (int(tick) - start_tick) / mid.ticks_per_beat * tempo / 1_000_000


def midi_notes(path: Path) -> tuple[mido.MidiFile, list[dict[str, Any]]]:
    mid = mido.MidiFile(path)
    notes: list[dict[str, Any]] = []
    max_tick = 0
    for track_index, track in enumerate(mid.tracks):
        tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for message in track:
            tick += int(message.time)
            max_tick = max(max_tick, tick)
            channel = int(getattr(message, "channel", 0))
            if message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                active[(channel, int(message.note))].append((tick, int(message.velocity)))
            elif message.type == "note_off" or (
                message.type == "note_on" and int(getattr(message, "velocity", 0)) == 0
            ):
                key = (channel, int(message.note))
                if active[key]:
                    start, velocity = active[key].pop(0)
                    notes.append(
                        {
                            "track": track_index,
                            "channel": channel,
                            "pitch": int(message.note),
                            "start_tick": start,
                            "end_tick": max(start, tick),
                            "velocity": velocity,
                        }
                    )
    segments, starts = _tempo_segments(mid)
    for note in notes:
        note["start_sec"] = _tick_to_seconds(mid, segments, starts, note["start_tick"])
        note["end_sec"] = _tick_to_seconds(mid, segments, starts, note["end_tick"])
    notes.sort(key=lambda item: (item["pitch"], item["start_tick"], item["end_tick"], item["track"]))
    return mid, notes


def midi_summary(path: Path) -> dict[str, Any]:
    mid, notes = midi_notes(path)
    segments, starts = _tempo_segments(mid)
    max_tick = max((note["end_tick"] for note in notes), default=0)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "ticks_per_beat": int(mid.ticks_per_beat),
        "track_count": len(mid.tracks),
        "note_count": len(notes),
        "pitch_counts": dict(sorted(Counter(int(note["pitch"]) for note in notes).items())),
        "first_note_sec": min((float(note["start_sec"]) for note in notes), default=None),
        "last_note_end_sec": max((float(note["end_sec"]) for note in notes), default=0.0),
        "midi_length_sec": float(mid.length),
        "event_end_sec": float(_tick_to_seconds(mid, segments, starts, max_tick)),
        "tempo_events": [
            {
                "tick": int(tick),
                "bpm": float(60_000_000 / tempo),
                "time_sec": float(sec),
            }
            for tick, sec, tempo in segments
        ],
    }


def compare_imported_midi(source_path: Path, imported_path: Path) -> dict[str, Any]:
    source_mid, source_notes = midi_notes(source_path)
    imported_mid, imported_notes = midi_notes(imported_path)
    source_by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    imported_by_pitch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in source_notes:
        source_by_pitch[int(note["pitch"])].append(note)
    for note in imported_notes:
        imported_by_pitch[int(note["pitch"])].append(note)
    deltas: list[dict[str, Any]] = []
    unmatched_source = 0
    unmatched_imported = 0
    for pitch in sorted(set(source_by_pitch) | set(imported_by_pitch)):
        source_items = sorted(source_by_pitch[pitch], key=lambda item: (item["start_tick"], item["end_tick"]))
        imported_items = sorted(imported_by_pitch[pitch], key=lambda item: (item["start_tick"], item["end_tick"]))
        matched = min(len(source_items), len(imported_items))
        unmatched_source += len(source_items) - matched
        unmatched_imported += len(imported_items) - matched
        for source, imported in zip(source_items, imported_items):
            deltas.append(
                {
                    "pitch": pitch,
                    "source_start_tick": source["start_tick"],
                    "imported_start_tick": imported["start_tick"],
                    "source_end_tick": source["end_tick"],
                    "imported_end_tick": imported["end_tick"],
                    "start_delta_sec": float(imported["start_sec"] - source["start_sec"]),
                    "end_delta_sec": float(imported["end_sec"] - source["end_sec"]),
                }
            )
    start_deltas = [item["start_delta_sec"] for item in deltas]
    end_deltas = [item["end_delta_sec"] for item in deltas]
    max_start = max((abs(value) for value in start_deltas), default=0.0)
    max_end = max((abs(value) for value in end_deltas), default=0.0)
    source_counts = Counter(int(note["pitch"]) for note in source_notes)
    imported_counts = Counter(int(note["pitch"]) for note in imported_notes)
    return {
        "source_note_count": len(source_notes),
        "imported_note_count": len(imported_notes),
        "matched_pitch_order_count": len(deltas),
        "unmatched_source_count": unmatched_source,
        "unmatched_imported_count": unmatched_imported,
        "pitch_multiset_equal": source_counts == imported_counts,
        "ticks_per_beat_equal": source_mid.ticks_per_beat == imported_mid.ticks_per_beat,
        "mean_abs_start_delta_sec": float(np.mean(np.abs(start_deltas))) if start_deltas else 0.0,
        "max_abs_start_delta_sec": float(max_start),
        "mean_abs_end_delta_sec": float(np.mean(np.abs(end_deltas))) if end_deltas else 0.0,
        "max_abs_end_delta_sec": float(max_end),
        "timing_tolerance_sec": TIMING_TOLERANCE_SEC,
        "timing_ok": (
            source_counts == imported_counts
            and source_mid.ticks_per_beat == imported_mid.ticks_per_beat
            and unmatched_source == 0
            and unmatched_imported == 0
            and max_start <= TIMING_TOLERANCE_SEC
            and max_end <= TIMING_TOLERANCE_SEC
        ),
        "first_deltas": deltas[:10],
    }


def audio_summary(path: Path, source_notes: list[dict[str, Any]]) -> dict[str, Any]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1)
    onset_times = librosa.onset.onset_detect(
        y=mono,
        sr=sample_rate,
        hop_length=512,
        units="time",
        backtrack=False,
        wait=1,
        delta=0.07,
    )
    expected = np.asarray([float(note["start_sec"]) for note in source_notes], dtype=np.float64)
    detected = np.asarray(onset_times, dtype=np.float64)
    detected_error = [
        float(np.min(np.abs(expected - value))) for value in detected if expected.size
    ]
    expected_error = [
        float(np.min(np.abs(detected - value))) for value in expected if detected.size
    ]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": float(info.duration),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "onset_count": int(len(detected)),
        "first_onset_sec": float(detected[0]) if detected.size else None,
        "last_onset_sec": float(detected[-1]) if detected.size else None,
        "mean_nearest_detected_to_midi_onset_sec": float(np.mean(detected_error)) if detected_error else None,
        "max_nearest_detected_to_midi_onset_sec": float(np.max(detected_error)) if detected_error else None,
        "mean_nearest_midi_to_detected_onset_sec": float(np.mean(expected_error)) if expected_error else None,
        "max_nearest_midi_to_detected_onset_sec": float(np.max(expected_error)) if expected_error else None,
    }


def run_musescore(
    executable: Path,
    profile: Path,
    source: Path,
    destination: Path,
    *,
    sound_profile: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    command = [
        str(executable),
        "--factory-settings",
        "--test-mode",
        "-M",
        str(profile),
    ]
    if sound_profile:
        command.extend(["--sound-profile", sound_profile])
    command.extend(["-o", str(destination), str(source)])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "return_code": None,
            "timeout": True,
            "stdout": str(error.stdout or "")[-4000:],
            "stderr": str(error.stderr or "")[-4000:],
            "exists": destination.is_file(),
        }
    return {
        "command": command,
        "return_code": int(completed.returncode),
        "timeout": False,
        "stdout": (completed.stdout or "")[-4000:],
        "stderr": (completed.stderr or "")[-4000:],
        "exists": destination.is_file(),
    }


def run_pilot(
    *,
    cases: tuple[PilotCase, ...],
    executable: Path,
    soundfont: Path,
    profile: Path,
    output_root: Path,
    overwrite: bool,
    timeout_sec: float,
) -> dict[str, Any]:
    for required in (executable, soundfont, profile):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output root exists; pass --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [str(executable), "--long-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "purpose": "fixed MuseScore Basic SoundFont pilot",
        "recognition_rerun": False,
        "registry_changed": False,
        "musescore": {
            "path": str(executable.resolve()),
            "sha256": sha256(executable),
            "version_command_return_code": version.returncode,
            "version": (version.stdout or version.stderr).strip(),
        },
        "soundfont": {
            "path": str(soundfont.resolve()),
            "sha256": sha256(soundfont),
            "bytes": soundfont.stat().st_size,
            "sound_profile": "MuseScore Basic",
        },
        "midi_import_profile": {
            "path": str(profile.resolve()),
            "sha256": sha256(profile),
        },
        "timing_tolerance_sec": TIMING_TOLERANCE_SEC,
        "cases": [],
    }
    for case in cases:
        _source_mid, source_notes = midi_notes(case.midi)
        case_root = output_root / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        first_wav = case_root / "render-a.wav"
        second_wav = case_root / "render-b.wav"
        reexport_mid = case_root / "imported.mid"
        musicxml = case_root / "imported.musicxml"
        first = run_musescore(
            executable,
            profile,
            case.midi,
            first_wav,
            sound_profile="MuseScore Basic",
            timeout_sec=timeout_sec,
        )
        second = run_musescore(
            executable,
            profile,
            case.midi,
            second_wav,
            sound_profile="MuseScore Basic",
            timeout_sec=timeout_sec,
        )
        imported = run_musescore(
            executable,
            profile,
            case.midi,
            reexport_mid,
            sound_profile=None,
            timeout_sec=timeout_sec,
        )
        xml_export = run_musescore(
            executable,
            profile,
            case.midi,
            musicxml,
            sound_profile=None,
            timeout_sec=timeout_sec,
        )
        deterministic = (
            first["exists"]
            and second["exists"]
            and first_wav.is_file()
            and second_wav.is_file()
            and sha256(first_wav) == sha256(second_wav)
        )
        imported_midi = compare_imported_midi(case.midi, reexport_mid) if reexport_mid.is_file() else None
        audio_a = audio_summary(first_wav, source_notes) if first_wav.is_file() else None
        source_summary = midi_summary(case.midi)
        reference_audio = sf.info(case.reference_audio)
        if audio_a is not None:
            audio_a["source_midi_length_delta_sec"] = float(
                audio_a["duration_sec"] - source_summary["midi_length_sec"]
            )
            audio_a["source_last_note_end_delta_sec"] = float(
                audio_a["duration_sec"] - source_summary["last_note_end_sec"]
            )
        case_report = {
            "case_id": case.case_id,
            "source_midi": source_summary,
            "reference_audio": {
                "path": str(case.reference_audio),
                "sha256": sha256(case.reference_audio),
                "sample_rate": int(reference_audio.samplerate),
                "channels": int(reference_audio.channels),
                "duration_sec": float(reference_audio.duration),
            },
            "exports": {
                "render_a": first,
                "render_b": second,
                "imported_midi": imported,
                "musicxml": xml_export,
            },
            "render_deterministic": deterministic,
            "render_a": audio_a,
            "imported_midi_comparison": imported_midi,
            "pilot_timing_ok": bool(imported_midi and imported_midi["timing_ok"]),
            "pilot_case_ok": bool(deterministic and imported_midi and imported_midi["timing_ok"]),
        }
        report["cases"].append(case_report)
    report["pilot_ok"] = bool(report["cases"]) and all(item["pilot_case_ok"] for item in report["cases"])
    report["pilot_policy"] = (
        "Use one fixed renderer for all selected instrumental cases only if every case is "
        "byte-deterministic and MuseScore import preserves the source MIDI pitch multiset and "
        f"onset/end timing within {TIMING_TOLERANCE_SEC:.3f}s. No model output or reference "
        "note selection is used by this decision."
    )
    (output_root / "pilot-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--musescore", type=Path, default=DEFAULT_MUSESCORE)
    parser.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    args = parser.parse_args()
    case_ids = tuple(args.case_ids or DEFAULT_CASE_IDS)
    report = run_pilot(
        cases=load_cases(case_ids),
        executable=args.musescore.resolve(),
        soundfont=args.soundfont.resolve(),
        profile=args.profile.resolve(),
        output_root=args.output_root.resolve(),
        overwrite=args.overwrite,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "pilot_ok": report["pilot_ok"],
        "cases": [
            {
                "case_id": item["case_id"],
                "render_deterministic": item["render_deterministic"],
                "pilot_timing_ok": item["pilot_timing_ok"],
            }
            for item in report["cases"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0 if report["pilot_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
