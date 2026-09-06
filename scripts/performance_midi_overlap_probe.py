"""Probe same-channel/same-pitch overlap through the MuseScore importer.

This is intentionally observational.  A successful MIDI parse proves that
the performance writer emitted both source notes; the MusicXML observation is
recorded for stage 5 voice allocation and timing validation and is not used to
claim full overlap preservation.
"""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mido

ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.high_accuracy import resolve_musescore
from backend.jianpu_score.performance_midi import build_performance_midi


FIXTURE = ROOT / "fixtures" / "high_accuracy" / "performance_same_pitch_overlap.json"
PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"


def _absolute_messages(track: mido.MidiTrack) -> list[tuple[int, mido.Message | mido.MetaMessage]]:
    tick = 0
    result: list[tuple[int, mido.Message | mido.MetaMessage]] = []
    for message in track:
        tick += message.time
        result.append((tick, message))
    return result


def _musicxml_pitch_midi(note: ET.Element) -> int | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    step = pitch.findtext("step")
    octave = pitch.findtext("octave")
    if step is None or octave is None:
        return None
    pitch_classes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    if step not in pitch_classes:
        return None
    alter = int(pitch.findtext("alter") or "0")
    return (int(octave) + 1) * 12 + pitch_classes[step] + alter


def _probe_musicxml(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    notes = [note for note in root.findall(".//note") if note.find("pitch") is not None]
    same_pitch = [note for note in notes if _musicxml_pitch_midi(note) == 60]
    voices = sorted({note.findtext("voice") or "" for note in same_pitch})
    durations = [int(note.findtext("duration") or "0") for note in same_pitch]
    return {
        "musicxml_note_count": len(notes),
        "same_pitch_note_count": len(same_pitch),
        "same_pitch_voices": voices,
        "same_pitch_durations": durations,
        "musicxml_tie_count": len(root.findall(".//tie")),
    }


def main() -> int:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.25,
        bpm=120.0,
        key=str(fixture["key"]),
        time_signature=str(fixture["time_signature"]),
        beat_times=[float(value) for value in fixture["beat_times"]],
        metadata={
            "beat_source": "beatnet",
            "beat_engine": "beatnet",
            "beatnet_version": "1.1.3",
            "beat_grid": {
                "beats": [
                    {"index": index, "time_sec": value, "downbeat": index == int(fixture["downbeat_index"])}
                    for index, value in enumerate(fixture["beat_times"])
                ],
                "mapping": {
                    "manual_bpm_scale": 1.0,
                    "score_origin": {
                        "downbeat_index": int(fixture["downbeat_index"]),
                        "downbeat_sec": float(fixture["beat_times"][int(fixture["downbeat_index"])]),
                    },
                },
            },
        },
    )
    events = [
        NoteEvent(
            midi=int(note["midi"]),
            start_sec=float(note["start_sec"]),
            end_sec=float(note["end_sec"]),
            voice_id=str(note["voice_id"]),
            source="overlap-fixture",
        )
        for note in fixture["notes"]
    ]
    midi_bytes, metadata = build_performance_midi(
        events,
        analysis,
        instrument_group=str(fixture["instrument_group"]),
        program=int(fixture["program"]),
        title="Same pitch overlap probe",
    )
    midi = mido.MidiFile(file=BytesIO(midi_bytes))
    parsed_notes = [
        (tick, message)
        for tick, message in _absolute_messages(midi.tracks[1])
        if message.type in {"note_on", "note_off"} and message.note == 60
    ]
    source_note_count = sum(message.type == "note_on" and message.velocity > 0 for _tick, message in parsed_notes)
    if source_note_count != int(fixture["expected_source_note_count"]):
        raise RuntimeError(f"performance MIDI source overlap count mismatch: {source_note_count}")

    result: dict[str, object] = {
        "status": "ok",
        "fixture": os.fspath(FIXTURE),
        "performance_midi": {
            "ticks_per_quarter": midi.ticks_per_beat,
            "same_pitch_source_note_count": source_note_count,
            "same_pitch_event_ticks": [tick for tick, _message in parsed_notes],
            "metadata_track_id": metadata["track_id"],
        },
        "musescore": {"available": False, "observation": "not_run"},
        "stage5_policy": fixture["stage5_policy"],
    }
    muse = resolve_musescore()
    if muse is None or not PROFILE.is_file():
        result["musescore"] = {
            "available": False,
            "observation": "MuseScore 4.7.4 or import profile unavailable",
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0

    with tempfile.TemporaryDirectory(prefix="jianpu-overlap-probe-") as temp:
        temp_dir = Path(temp)
        midi_path = temp_dir / "same-pitch-overlap.performance.mid"
        musicxml_path = temp_dir / "same-pitch-overlap.musicxml"
        midi_path.write_bytes(midi_bytes)
        command = [
            os.fspath(muse),
            "--factory-settings",
            "--test-mode",
            "-M",
            os.fspath(PROFILE),
            "-o",
            os.fspath(musicxml_path),
            os.fspath(midi_path),
        ]
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
        observation: dict[str, object] = {
            "available": True,
            "returncode": completed.returncode,
            "musicxml_exists": musicxml_path.is_file(),
        }
        if completed.returncode == 0 and musicxml_path.is_file():
            observation.update(_probe_musicxml(musicxml_path))
            observation["source_count_preserved"] = observation["same_pitch_note_count"] == source_note_count
            observation["timing_match_established"] = False
        else:
            observation["stderr_tail"] = (completed.stderr or completed.stdout)[-1000:]
        result["musescore"] = observation
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"performance-overlap-probe-failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
