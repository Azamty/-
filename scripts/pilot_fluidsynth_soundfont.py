"""Pilot deterministic FluidSynth rendering from the original MIDI.

This is deliberately a renderer pilot.  It does not change the benchmark
registry or substitute reference notes for model output.  Each selected MIDI
file is sent directly to the pinned FluidSynth executable twice with the
pinned MS Basic SoundFont.  The raw FluidSynth tail is then trimmed to a
fixed, auditable duration after the last source note.  The event dump is
checked independently so a successful WAV export cannot hide MIDI loss.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

from scripts.pilot_musescore_soundfont import (
    DEFAULT_CASE_IDS,
    DEFAULT_SOUNDFONT,
    audio_summary,
    load_cases,
    midi_notes,
    midi_summary,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLUIDSYNTH = (
    ROOT
    / "tools"
    / "fluidsynth-2.6.0"
    / "fluidsynth-v2.6.0-win10-x64-cpp11"
    / "bin"
    / "fluidsynth.exe"
)
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "fluidsynth-soundfont-pilot-v1"
SAMPLE_RATE = 44_100
TAIL_SEC = 0.25
FLUIDSYNTH_VERSION = "2.6.0"
FLUIDSYNTH_RELEASE_URL = "https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.0"
FLUIDSYNTH_ASSET_URL = (
    "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.0/"
    "fluidsynth-v2.6.0-win10-x64-cpp11.zip"
)
FLUIDSYNTH_ASSET_BYTES = 2_722_370
FLUIDSYNTH_ASSET_SHA256 = (
    "817262DEACAA748EDB3AF6731DFFE1766B00146790BECFCCC949A9F701E76681"
)
FLUIDSYNTH_EXECUTABLE_SHA256 = (
    "08C72384A47F67B0C5BE9EE8C88B1F0B6AFE39A8217ED2ADB83A88B41C051632"
)
MS_BASIC_SOUNDFONT_SHA256 = (
    "5EA2375E8BD7D8E71DEF1036978C1621E85B66934169B6A2744B27B9B3C2D99C"
)
EVENT_RE = re.compile(r"event_post_(noteon|noteoff)\s+\d+\s+(\d+)(?:\s+(\d+))?")


def parse_event_dump(text: str) -> dict[str, Any]:
    """Parse FluidSynth's deterministic event-posting diagnostics."""

    note_on: list[tuple[int, int]] = []
    note_off: list[tuple[int, int]] = []
    missing_velocity = 0
    for line in text.splitlines():
        match = EVENT_RE.search(line)
        if not match:
            continue
        event_type, pitch, velocity = match.groups()
        if velocity is None:
            # FluidSynth's debug writer can splice the next startup message
            # into a note-off line when stdout is captured.  MIDI note-off
            # velocity is semantically irrelevant, so retain the event and
            # record the diagnostic instead of falsely reporting a lost note.
            missing_velocity += 1
        event = (int(pitch), int(velocity or 0))
        (note_on if event_type == "noteon" else note_off).append(event)
    return {
        "note_on_count": len(note_on),
        "note_off_count": len(note_off),
        "note_on_pitches": dict(sorted(Counter(item[0] for item in note_on).items())),
        "note_off_pitches": dict(sorted(Counter(item[0] for item in note_off).items())),
        "missing_velocity_count": missing_velocity,
        "note_on": [{"pitch": pitch, "velocity": velocity} for pitch, velocity in note_on],
        "note_off": [{"pitch": pitch, "velocity": velocity} for pitch, velocity in note_off],
    }


def compare_event_dump(source_notes: list[dict[str, Any]], dump: dict[str, Any]) -> dict[str, Any]:
    expected = Counter(int(note["pitch"]) for note in source_notes)
    actual_on = Counter({int(key): int(value) for key, value in dump["note_on_pitches"].items()})
    actual_off = Counter({int(key): int(value) for key, value in dump["note_off_pitches"].items()})
    return {
        "source_note_count": len(source_notes),
        "note_on_count": int(dump["note_on_count"]),
        "note_off_count": int(dump["note_off_count"]),
        "pitch_multiset_equal_on": actual_on == expected,
        "pitch_multiset_equal_off": actual_off == expected,
        "event_complete": (
            int(dump["note_on_count"]) == len(source_notes)
            and int(dump["note_off_count"]) == len(source_notes)
            and actual_on == expected
            and actual_off == expected
        ),
    }


def trim_wave(source: Path, destination: Path, target_frames: int) -> dict[str, Any]:
    """Copy a FluidSynth WAV through a fixed frame boundary without resampling."""

    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        source_frames = int(reader.getnframes())
        frames = reader.readframes(min(source_frames, max(0, int(target_frames))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    with wave.open(str(destination), "rb") as reader:
        final_frames = int(reader.getnframes())
    return {
        "source_frames": source_frames,
        "target_frames": int(target_frames),
        "final_frames": final_frames,
        "source_duration_sec": source_frames / params.framerate,
        "final_duration_sec": final_frames / params.framerate,
        "sample_rate": int(params.framerate),
        "channels": int(params.nchannels),
        "sample_width_bytes": int(params.sampwidth),
    }


def _run_fluid_synth(
    executable: Path,
    soundfont: Path,
    source: Path,
    destination: Path,
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "-q",
        "-ni",
        "-d",
        "-R",
        "0",
        "-C",
        "0",
        "-F",
        str(destination),
        "-r",
        str(SAMPLE_RATE),
        str(soundfont),
        str(source),
    ]
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
            "stdout": str(error.stdout or "")[-6000:],
            "stderr": str(error.stderr or "")[-6000:],
            "exists": destination.is_file(),
            "event_dump": parse_event_dump(str(error.stdout or "")),
        }
    stdout = completed.stdout or ""
    return {
        "command": command,
        "return_code": int(completed.returncode),
        "timeout": False,
        "stdout": stdout[-6000:],
        "stderr": (completed.stderr or "")[-6000:],
        "exists": destination.is_file(),
        "event_dump": parse_event_dump(stdout),
    }


def _audio_format(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "subtype": str(info.subtype),
        "format": str(info.format),
        "duration_sec": float(info.duration),
    }


def run_pilot(
    *,
    cases: Iterable[Any],
    executable: Path,
    soundfont: Path,
    output_root: Path,
    overwrite: bool,
    timeout_sec: float,
) -> dict[str, Any]:
    for required in (executable, soundfont):
        if not required.is_file():
            raise FileNotFoundError(required)
    executable_hash = sha256(executable).upper()
    soundfont_hash = sha256(soundfont).upper()
    if executable_hash != FLUIDSYNTH_EXECUTABLE_SHA256:
        raise ValueError(
            "FluidSynth executable hash does not match the pinned 2.6.0 asset: "
            f"expected {FLUIDSYNTH_EXECUTABLE_SHA256}, got {executable_hash}"
        )
    if soundfont_hash != MS_BASIC_SOUNDFONT_SHA256:
        raise ValueError(
            "SoundFont hash does not match the pinned MS Basic asset: "
            f"expected {MS_BASIC_SOUNDFONT_SHA256}, got {soundfont_hash}"
        )
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output root exists; pass --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "purpose": "fixed FluidSynth direct MIDI SoundFont pilot",
        "recognition_rerun": False,
        "registry_changed": False,
        "renderer_policy": {
            "fluidsynth_version": FLUIDSYNTH_VERSION,
            "release_url": FLUIDSYNTH_RELEASE_URL,
            "asset_url": FLUIDSYNTH_ASSET_URL,
            "asset_bytes": FLUIDSYNTH_ASSET_BYTES,
            "asset_sha256": FLUIDSYNTH_ASSET_SHA256,
            "executable_sha256": executable_hash,
            "soundfont_sha256": soundfont_hash,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "subtype": "PCM_16",
            "effects": {"reverb": False, "chorus": False},
            "tail_sec": TAIL_SEC,
            "tail_rule": "ceil((last source note end + fixed tail) * sample_rate) frames",
            "version_output": (version.stdout or version.stderr).strip(),
        },
        "cases": [],
    }
    for case in cases:
        _source_mid, source_notes = midi_notes(case.midi)
        source_summary = midi_summary(case.midi)
        target_frames = math.ceil((float(source_summary["last_note_end_sec"]) + TAIL_SEC) * SAMPLE_RATE)
        case_root = output_root / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        raw_a = case_root / "raw-a.wav"
        raw_b = case_root / "raw-b.wav"
        final_a = case_root / "render-a.wav"
        final_b = case_root / "render-b.wav"
        run_a = _run_fluid_synth(executable, soundfont, case.midi, raw_a, timeout_sec=timeout_sec)
        run_b = _run_fluid_synth(executable, soundfont, case.midi, raw_b, timeout_sec=timeout_sec)
        trim_a = trim_wave(raw_a, final_a, target_frames) if raw_a.is_file() else None
        trim_b = trim_wave(raw_b, final_b, target_frames) if raw_b.is_file() else None
        event_a = compare_event_dump(source_notes, run_a["event_dump"])
        event_b = compare_event_dump(source_notes, run_b["event_dump"])
        deterministic = final_a.is_file() and final_b.is_file() and sha256(final_a) == sha256(final_b)
        audio_a = audio_summary(final_a, source_notes) if final_a.is_file() else None
        audio_format = _audio_format(final_a) if final_a.is_file() else None
        duration_ok = bool(
            trim_a
            and trim_b
            and trim_a["final_frames"] == target_frames
            and trim_b["final_frames"] == target_frames
        )
        format_ok = bool(
            audio_format
            and audio_format["sample_rate"] == SAMPLE_RATE
            and audio_format["channels"] == 2
            and audio_format["subtype"] == "PCM_16"
        )
        case_report = {
            "case_id": case.case_id,
            "source_midi": source_summary,
            "reference_audio": {
                "path": str(case.reference_audio),
                "sha256": sha256(case.reference_audio),
                "sample_rate": int(sf.info(case.reference_audio).samplerate),
                "channels": int(sf.info(case.reference_audio).channels),
                "duration_sec": float(sf.info(case.reference_audio).duration),
            },
            "target_frames": int(target_frames),
            "target_duration_sec": target_frames / SAMPLE_RATE,
            "render_a": {**run_a, "trim": trim_a},
            "render_b": {**run_b, "trim": trim_b},
            "render_a_audio": audio_a,
            "render_a_format": audio_format,
            "event_comparison_a": event_a,
            "event_comparison_b": event_b,
            "render_hash_a": sha256(final_a) if final_a.is_file() else None,
            "render_hash_b": sha256(final_b) if final_b.is_file() else None,
            "render_deterministic": deterministic,
            "duration_policy_ok": duration_ok,
            "audio_format_ok": format_ok,
            "pilot_case_ok": bool(deterministic and duration_ok and format_ok and event_a["event_complete"] and event_b["event_complete"]),
        }
        report["cases"].append(case_report)
    report["pilot_ok"] = bool(report["cases"]) and all(item["pilot_case_ok"] for item in report["cases"])
    report["pilot_policy"] = (
        "Use one fixed direct FluidSynth renderer for all selected instrumental cases only if every case "
        "is byte-deterministic, every source MIDI note-on/off is observed in the renderer event dump, "
        "and all outputs are fixed 44.1 kHz stereo PCM16 with the declared tail policy. The decision "
        "does not inspect model output or select reference notes."
    )
    (output_root / "pilot-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--fluidsynth", type=Path, default=DEFAULT_FLUIDSYNTH)
    parser.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    args = parser.parse_args(argv)
    report = run_pilot(
        cases=load_cases(tuple(args.case_ids or DEFAULT_CASE_IDS)),
        executable=args.fluidsynth.resolve(),
        soundfont=args.soundfont.resolve(),
        output_root=args.output_root.resolve(),
        overwrite=args.overwrite,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps({"pilot_ok": report["pilot_ok"], "cases": len(report["cases"])}, ensure_ascii=False))
    return 0 if report["pilot_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
