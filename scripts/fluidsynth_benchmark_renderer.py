"""Deterministic direct MIDI renderer for the local benchmark domain.

The benchmark inputs in the generated and MAESTRO local-render domains are
rendered directly from their source MIDI with the pinned FluidSynth binary
and MS Basic SoundFont.  MuseScore is intentionally not involved in this
audio preparation step: the source MIDI remains the authority for pitch,
onset, duration, velocity, tempo, and meter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import wave
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import mido
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = (
    ROOT
    / "tools"
    / "fluidsynth-2.6.0"
    / "fluidsynth-v2.6.0-win10-x64-cpp11"
    / "bin"
    / "fluidsynth.exe"
)
DEFAULT_SOUNDFONT = ROOT / "tools" / "musescore-4.7.4" / "MuseScore 4" / "sound" / "MS Basic.sf3"
RENDERER_VERSION = "fluidsynth_direct_ms_basic_v1"
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
SAMPLE_RATE = 44_100
CHANNELS = 2
SAMPLE_WIDTH_BYTES = 2
GAIN = 0.2
TAIL_SEC = 0.25
DEFAULT_RENDER_TIMEOUT_SEC = 180.0
VERSION_CHECK_TIMEOUT_SEC = 30.0
EVENT_RE = re.compile(r"event_post_(noteon|noteoff)\s+\d+\s+(\d+)(?:\s+(\d+))?")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tempo_segments(mid: mido.MidiFile) -> tuple[list[tuple[int, float, int]], list[int]]:
    changes: dict[int, int] = {0: 500_000}
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


def _tick_to_seconds(
    mid: mido.MidiFile,
    segments: list[tuple[int, float, int]],
    starts: list[int],
    tick: int,
) -> float:
    index = bisect_right(starts, int(tick)) - 1
    start_tick, start_sec, tempo = segments[max(0, index)]
    return start_sec + (int(tick) - start_tick) / mid.ticks_per_beat * tempo / 1_000_000


def _meta_summary(mid: mido.MidiFile) -> dict[str, list[dict[str, Any]]]:
    tempo: list[dict[str, Any]] = []
    meters: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for track_index, track in enumerate(mid.tracks):
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                tempo.append(
                    {
                        "tick": tick,
                        "time_sec": _tick_to_seconds(mid, *_tempo_segments(mid), tick),
                        "bpm": float(mido.tempo2bpm(message.tempo)),
                        "track": track_index,
                    }
                )
            elif message.type == "time_signature":
                meters.append(
                    {
                        "tick": tick,
                        "numerator": int(message.numerator),
                        "denominator": int(message.denominator),
                        "track": track_index,
                    }
                )
            elif message.type == "key_signature":
                keys.append({"tick": tick, "key": str(message.key), "track": track_index})
    def dedupe(items: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        result: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda value: (int(value["tick"]), int(value.get("track", 0)))):
            key = tuple(item.get(field) for field in fields)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
    return {
        "tempo": dedupe(tempo, ("tick", "bpm")),
        "meter": dedupe(meters, ("tick", "numerator", "denominator")),
        "key": dedupe(keys, ("tick", "key")),
    }


def midi_notes(path: Path) -> tuple[mido.MidiFile, list[dict[str, Any]]]:
    """Read source note events without importing them through MuseScore."""

    mid = mido.MidiFile(path)
    notes: list[dict[str, Any]] = []
    for track_index, track in enumerate(mid.tracks):
        tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for message in track:
            tick += int(message.time)
            channel = int(getattr(message, "channel", 0))
            if message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                active[(channel, int(message.note))].append((tick, int(message.velocity)))
            elif message.type == "note_off" or (
                message.type == "note_on" and int(getattr(message, "velocity", 0)) == 0
            ):
                key = (channel, int(message.note))
                if active[key]:
                    start, velocity = active[key].pop(0)
                    if tick > start:
                        notes.append(
                            {
                                "track": track_index,
                                "channel": channel,
                                "pitch": int(message.note),
                                "start_tick": start,
                                "end_tick": tick,
                                "velocity": velocity,
                            }
                        )
    segments, starts = _tempo_segments(mid)
    for note in notes:
        note["start_sec"] = _tick_to_seconds(mid, segments, starts, note["start_tick"])
        note["end_sec"] = _tick_to_seconds(mid, segments, starts, note["end_tick"])
    notes.sort(key=lambda item: (item["start_tick"], item["track"], item["channel"], item["pitch"], item["end_tick"]))
    return mid, notes


def source_midi_summary(path: Path) -> dict[str, Any]:
    mid, notes = midi_notes(path)
    metadata = _meta_summary(mid)
    source_events = [
        {
            key: note[key]
            for key in ("track", "channel", "pitch", "start_tick", "end_tick", "velocity")
        }
        for note in notes
    ]
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "ticks_per_beat": int(mid.ticks_per_beat),
        "track_count": len(mid.tracks),
        "note_count": len(notes),
        "pitch_counts": dict(sorted(Counter(int(note["pitch"]) for note in notes).items())),
        "velocity_counts": dict(sorted(Counter(int(note["velocity"]) for note in notes).items())),
        "first_note_sec": min((float(note["start_sec"]) for note in notes), default=None),
        "last_note_end_sec": max((float(note["end_sec"]) for note in notes), default=0.0),
        "tempo_events": metadata["tempo"],
        "meter_events": metadata["meter"],
        "key_events": metadata["key"],
        "note_events_sha256": _canonical_sha256(source_events),
        "note_event_fields": ["track", "channel", "pitch", "start_tick", "end_tick", "velocity"],
    }


def parse_event_dump(text: str) -> dict[str, Any]:
    """Parse FluidSynth's event-posting diagnostics."""

    note_on: list[tuple[int, int]] = []
    note_off: list[tuple[int, int]] = []
    missing_velocity = 0
    for line in text.splitlines():
        match = EVENT_RE.search(line)
        if not match:
            continue
        event_type, pitch, velocity = match.groups()
        if velocity is None:
            missing_velocity += 1
        event = (int(pitch), int(velocity or 0))
        (note_on if event_type == "noteon" else note_off).append(event)
    return {
        "note_on_count": len(note_on),
        "note_off_count": len(note_off),
        "note_on_pitches": dict(sorted(Counter(item[0] for item in note_on).items())),
        "note_off_pitches": dict(sorted(Counter(item[0] for item in note_off).items())),
        "note_on_velocities": dict(sorted(Counter(item[1] for item in note_on).items())),
        "missing_velocity_count": missing_velocity,
        "note_on": [{"pitch": pitch, "velocity": velocity} for pitch, velocity in note_on],
        "note_off": [{"pitch": pitch, "velocity": velocity} for pitch, velocity in note_off],
    }


def compare_event_dump(source_notes: list[Mapping[str, Any]], dump: Mapping[str, Any]) -> dict[str, Any]:
    expected_pitch = Counter(int(note["pitch"]) for note in source_notes)
    actual_on = Counter({int(key): int(value) for key, value in dump.get("note_on_pitches", {}).items()})
    actual_off = Counter({int(key): int(value) for key, value in dump.get("note_off_pitches", {}).items()})
    expected_velocity = Counter(
        int(note["velocity"])
        for note in source_notes
        if note.get("velocity") is not None
    )
    actual_velocity = Counter(
        int(key) for key, value in dump.get("note_on_velocities", {}).items() for _ in range(int(value))
    )
    velocity_checked = bool(expected_velocity)
    return {
        "source_note_count": len(source_notes),
        "note_on_count": int(dump.get("note_on_count", 0)),
        "note_off_count": int(dump.get("note_off_count", 0)),
        "pitch_multiset_equal_on": actual_on == expected_pitch,
        "pitch_multiset_equal_off": actual_off == expected_pitch,
        "velocity_multiset_equal_on": actual_velocity == expected_velocity if velocity_checked else None,
        "velocity_checked": velocity_checked,
        "event_complete": (
            int(dump.get("note_on_count", 0)) == len(source_notes)
            and int(dump.get("note_off_count", 0)) == len(source_notes)
            and actual_on == expected_pitch
            and actual_off == expected_pitch
            and (not velocity_checked or actual_velocity == expected_velocity)
        ),
    }


def trim_wave(source: Path, destination: Path, target_frames: int) -> dict[str, Any]:
    """Copy the renderer output to the fixed source-note-plus-tail boundary."""

    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        source_frames = int(reader.getnframes())
        target_frames = max(0, int(target_frames))
        frame_width = int(params.nchannels) * int(params.sampwidth)
        frames = reader.readframes(min(source_frames, target_frames))
    # FluidSynth normally emits a longer release tail than the benchmark
    # boundary, but padding makes the declared tail rule exact even when a
    # future SoundFont/configuration emits a shorter file.
    frames = frames[: target_frames * frame_width]
    frames += b"\x00" * max(0, target_frames * frame_width - len(frames))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    with wave.open(str(destination), "rb") as reader:
        final_frames = int(reader.getnframes())
    return {
        "source_frames": source_frames,
        "target_frames": target_frames,
        "final_frames": final_frames,
        "source_duration_sec": source_frames / params.framerate,
        "final_duration_sec": final_frames / params.framerate,
        "sample_rate": int(params.framerate),
        "channels": int(params.nchannels),
        "sample_width_bytes": int(params.sampwidth),
    }


def _text_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _terminate_process_tree(process: subprocess.Popen[str]) -> dict[str, Any]:
    """Terminate FluidSynth and descendants without a second pipe deadlock."""

    details: dict[str, Any] = {
        "pid": int(getattr(process, "pid", 0) or 0),
        "method": "already_exited",
        "taskkill_return_code": None,
        "taskkill_timeout": False,
        "waited": False,
        "wait_timeout": False,
    }
    if process.poll() is not None:
        return details
    if os.name == "nt":
        details["method"] = "taskkill_tree_force"
        try:
            result = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            details["taskkill_return_code"] = int(result.returncode)
        except subprocess.TimeoutExpired:
            details["taskkill_timeout"] = True
    else:
        details["method"] = "process_group_sigkill"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=5)
        details["waited"] = True
    except subprocess.TimeoutExpired:
        details["wait_timeout"] = True
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1)
            details["waited"] = True
        except subprocess.TimeoutExpired:
            pass
    details["return_code"] = process.poll()
    return details


def _partial_artifact(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not info["exists"]:
        return info
    try:
        info["bytes"] = int(path.stat().st_size)
        info["sha256"] = sha256(path)
    except OSError as error:
        info["error"] = str(error)
    return info


def _run_succeeded(result: Mapping[str, Any]) -> bool:
    return (
        result.get("return_code") == 0
        and result.get("timeout") is False
        and result.get("exists") is True
    )


def _run_fluid_synth(
    executable: Path,
    soundfont: Path,
    source: Path,
    destination: Path,
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    if not math.isfinite(float(timeout_sec)) or float(timeout_sec) <= 0:
        raise ValueError("FluidSynth timeout_sec must be a finite number greater than zero")
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
        "-g",
        str(GAIN),
        "-L",
        str(CHANNELS),
        "-T",
        "wav",
        "-O",
        "s16",
        "-F",
        str(destination),
        "-r",
        str(SAMPLE_RATE),
        str(soundfont),
        str(source),
    ]
    creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=float(timeout_sec))
    except subprocess.TimeoutExpired as error:
        termination = _terminate_process_tree(process)
        stdout = _text_output(getattr(error, "stdout", None))
        stderr = _text_output(getattr(error, "stderr", None))
        partial_artifact = _partial_artifact(destination)
        return {
            "command": command,
            "return_code": None,
            "timeout": True,
            "stdout": stdout[-6000:],
            "stderr": stderr[-6000:],
            "exists": partial_artifact["exists"],
            "partial_artifact": partial_artifact,
            "termination": termination,
            "event_dump": parse_event_dump(stdout),
        }
    finally:
        _close_process_pipes(process)
    stdout = stdout or ""
    return {
        "command": command,
        "return_code": int(process.returncode),
        "timeout": False,
        "stdout": stdout[-6000:],
        "stderr": (stderr or "")[-6000:],
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


def renderer_metadata(*, executable: Path = DEFAULT_EXECUTABLE, soundfont: Path = DEFAULT_SOUNDFONT) -> dict[str, Any]:
    """Return and validate the pinned renderer identity."""

    executable = executable.resolve()
    soundfont = soundfont.resolve()
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
    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=VERSION_CHECK_TIMEOUT_SEC,
        check=False,
    )
    version_text = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or "2.6.0" not in version_text:
        raise ValueError(f"unexpected FluidSynth version output: {version_text!r}")
    return {
        "renderer_version": RENDERER_VERSION,
        "fluidsynth_version": FLUIDSYNTH_VERSION,
        "release_url": FLUIDSYNTH_RELEASE_URL,
        "asset_url": FLUIDSYNTH_ASSET_URL,
        "asset_bytes": FLUIDSYNTH_ASSET_BYTES,
        "asset_sha256": FLUIDSYNTH_ASSET_SHA256,
        "executable": str(executable),
        "executable_sha256": executable_hash,
        "soundfont": str(soundfont),
        "soundfont_sha256": soundfont_hash,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_width_bytes": SAMPLE_WIDTH_BYTES,
        "subtype": "PCM_16",
        "effects": {"reverb": False, "chorus": False},
        "gain": GAIN,
        "tail_sec": TAIL_SEC,
        "tail_rule": "ceil((last source note end + fixed tail) * sample_rate) frames",
        "version_output": version_text,
    }


def render_midi(
    source_midi: Path,
    destination: Path,
    *,
    manifest_path: Path | None = None,
    executable: Path = DEFAULT_EXECUTABLE,
    soundfont: Path = DEFAULT_SOUNDFONT,
    timeout_sec: float = DEFAULT_RENDER_TIMEOUT_SEC,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one source MIDI twice and publish one verified PCM16 WAV."""

    source_midi = source_midi.resolve()
    destination = destination.resolve()
    if not math.isfinite(float(timeout_sec)) or float(timeout_sec) <= 0:
        raise ValueError("FluidSynth timeout_sec must be a finite number greater than zero")
    if not source_midi.is_file():
        raise FileNotFoundError(source_midi)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    manifest_path = (manifest_path or destination.with_suffix(".render_manifest.json")).resolve()
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(manifest_path)
    renderer = renderer_metadata(executable=executable, soundfont=soundfont)
    _mid, source_notes = midi_notes(source_midi)
    source_summary = source_midi_summary(source_midi)
    target_frames = math.ceil((float(source_summary["last_note_end_sec"]) + TAIL_SEC) * SAMPLE_RATE)
    work_root = destination.parent / f".{destination.stem}.fluidsynth-verification"
    if work_root.exists():
        for item in work_root.glob("*"):
            if item.is_file():
                item.unlink()
    work_root.mkdir(parents=True, exist_ok=True)
    raw_a = work_root / "raw-a.wav"
    raw_b = work_root / "raw-b.wav"
    trim_a = work_root / "render-a.wav"
    trim_b = work_root / "render-b.wav"
    run_a = _run_fluid_synth(executable.resolve(), soundfont.resolve(), source_midi, raw_a, timeout_sec=timeout_sec)
    if _run_succeeded(run_a):
        run_b = _run_fluid_synth(executable.resolve(), soundfont.resolve(), source_midi, raw_b, timeout_sec=timeout_sec)
    else:
        run_b = {
            "command": [],
            "return_code": None,
            "timeout": False,
            "skipped": True,
            "skip_reason": "run_a_failed",
            "stdout": "",
            "stderr": "",
            "exists": False,
            "event_dump": {},
        }
    if not _run_succeeded(run_a) or not _run_succeeded(run_b):
        failure_path = work_root / "render_failure.json"
        failure = {
            "schema_version": "1.0",
            "failure": {
                "stage": "fluidsynth_render",
                "timeout_sec": float(timeout_sec),
                "source_midi": str(source_midi),
                "destination": str(destination),
                "renderer": renderer,
                "runs": {"run_a": run_a, "run_b": run_b},
                "partial_artifacts": {
                    "run_a": _partial_artifact(raw_a),
                    "run_b": _partial_artifact(raw_b),
                },
            },
        }
        failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"FluidSynth render failed; diagnostics preserved at {failure_path}: {failure['failure']['runs']}")
    trim_info_a = trim_wave(raw_a, trim_a, target_frames)
    trim_info_b = trim_wave(raw_b, trim_b, target_frames)
    audio_a = _audio_format(trim_a)
    audio_b = _audio_format(trim_b)
    format_ok = all(
        audio.get("sample_rate") == SAMPLE_RATE
        and audio.get("channels") == CHANNELS
        and audio.get("subtype") == "PCM_16"
        for audio in (audio_a, audio_b)
    )
    if not format_ok:
        raise ValueError(f"FluidSynth output format is not fixed PCM16 stereo: {audio_a}, {audio_b}")
    event_a = compare_event_dump(source_notes, run_a["event_dump"])
    event_b = compare_event_dump(source_notes, run_b["event_dump"])
    hash_a = sha256(trim_a)
    hash_b = sha256(trim_b)
    deterministic = hash_a == hash_b
    if not deterministic:
        raise ValueError(f"FluidSynth output is not byte-deterministic: {hash_a} != {hash_b}")
    if not event_a["event_complete"] or not event_b["event_complete"]:
        raise ValueError(f"FluidSynth source event accounting failed: {event_a}, {event_b}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    trim_a.replace(destination)
    manifest = {
        "schema_version": "1.0",
        "purpose": "benchmark production input rendered directly from source MIDI",
        "source_midi": source_summary,
        "renderer": renderer,
        "target_frames": target_frames,
        "target_duration_sec": target_frames / SAMPLE_RATE,
        "verification": {
            "run_a": {**run_a, "trim": trim_info_a, "audio": audio_a, "sha256": hash_a},
            "run_b": {**run_b, "trim": trim_info_b, "audio": audio_b, "sha256": hash_b},
            "byte_deterministic": deterministic,
            "event_comparison_a": event_a,
            "event_comparison_b": event_b,
            "source_event_complete": bool(event_a["event_complete"] and event_b["event_complete"]),
        },
        "output": {
            "path": str(destination),
            "sha256": hash_a,
            "bytes": destination.stat().st_size,
            "audio": _audio_format(destination),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in work_root.glob("*"):
        if item.is_file():
            item.unlink()
    try:
        work_root.rmdir()
    except OSError:
        pass
    return manifest
