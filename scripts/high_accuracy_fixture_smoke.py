"""Run the offline stage-A MIDI -> MuseScore -> MusicXML fixture smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mido


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

from backend.jianpu_score.high_accuracy import resolve_musescore, resolve_notation_python  # noqa: E402
FIXTURE = ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json"
PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"


def _write_fixture_midi(path: Path, payload: dict, *, force_single_channel: bool = False) -> None:
    source = payload["source"]
    midi = mido.MidiFile(type=1, ticks_per_beat=int(source["ticks_per_quarter"]))
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("track_name", name="Stage A tempo"))
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(source["tempo_bpm"])), time=0))
    numerator, denominator = (int(part) for part in source["time_signature"].split("/"))
    tempo_track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=numerator,
            denominator=denominator,
            clocks_per_click=36,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    tempo_track.append(mido.MetaMessage("key_signature", key="D", time=0))
    midi.tracks.append(tempo_track)

    note_track = mido.MidiTrack()
    note_track.append(mido.MetaMessage("track_name", name="Stage A polyphonic part", time=0))
    messages: list[tuple[int, int, mido.Message]] = []
    for note in payload["notes"]:
        start = int(note["start_tick"])
        end = start + int(note["duration_tick"])
        pitch = int(note["midi"])
        channel = 0 if force_single_channel else int(note["voice"]) - 1
        messages.append((start, 1, mido.Message("note_on", note=pitch, velocity=80, channel=channel)))
        messages.append((end, 0, mido.Message("note_off", note=pitch, velocity=0, channel=channel)))
    previous = 0
    for tick, _priority, message in sorted(messages, key=lambda item: (item[0], item[1], item[2].note)):
        message.time = tick - previous
        note_track.append(message)
        previous = tick
    note_track.append(mido.MetaMessage("end_of_track", time=max(0, int(source["duration_ticks"]) - previous)))
    midi.tracks.append(note_track)
    midi.save(path)


def _parse_musicxml(notation_python: Path, musicxml: Path) -> dict[str, int]:
    code = (
        "import json, sys; from music21 import converter; "
        "score=converter.parse(sys.argv[1]); "
        "flat=score.flatten(); elements=list(flat.notes); "
        "pitches=sorted(int(p.midi) for n in elements for p in (n.pitches if hasattr(n, 'pitches') else (n.pitch,))); "
        "tuplets=sum(1 for n in elements if getattr(n.duration, 'tuplets', ())); "
        "shortest=min((float(n.duration.quarterLength) for n in elements), default=0.0); "
        "meters=sorted(set(str(ts.ratioString) for ts in score.recurse().getElementsByClass('TimeSignature'))); "
        "voices=sum(len(part.getElementsByClass('Voice')) for part in score.parts); "
        "print(json.dumps({'parts':len(score.parts), 'notes':len(elements), "
        "'measures':len(score.parts[0].getElementsByClass('Measure')) if score.parts else 0, "
        "'pitches':pitches, 'tuplets':tuplets, 'shortest_quarter_length':shortest, "
        "'meters':meters, 'voices':voices}))"
    )
    result = subprocess.run(
        [os.fspath(notation_python), "-c", code, os.fspath(musicxml)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"music21 parse failed ({result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _musicxml_voice_keys(musicxml: Path) -> list[str]:
    root = ET.parse(musicxml).getroot()
    keys: set[str] = set()
    for part_index, part in enumerate(root.findall("part")):
        for note in part.findall(".//note"):
            voice = note.findtext("voice")
            if voice:
                keys.add(f"{part_index}:{note.findtext('staff') or '1'}:{voice}")
    return sorted(keys)


def _convert(muse: Path, profile: Path, midi: Path, musicxml: Path, cwd: Path) -> None:
    result = subprocess.run(
        [
            os.fspath(muse),
            "--factory-settings",
            "--test-mode",
            "-M",
            os.fspath(profile),
            "-o",
            os.fspath(musicxml),
            os.fspath(midi),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"MuseScore MIDI import failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    if not musicxml.is_file() or musicxml.stat().st_size < 200:
        raise RuntimeError("MuseScore exited successfully but did not produce MusicXML")


def main() -> int:
    muse = resolve_musescore()
    notation_python = resolve_notation_python()
    if muse is None:
        raise RuntimeError("MuseScore 4.7.4 is unavailable; install the fixed MSI before running this smoke test")
    if not notation_python.is_file():
        raise RuntimeError(f"notation environment is unavailable: {notation_python}")
    if not PROFILE.is_file():
        raise RuntimeError(f"MuseScore MIDI import profile is missing: {PROFILE}")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    profile_root = ET.parse(PROFILE).getroot()
    configured_voice_count = int(profile_root.findtext("VoiceCount") or "-1") + 1
    expected = payload["expected"]
    if configured_voice_count != int(expected["voice_count"]):
        raise RuntimeError(
            f"MuseScore VoiceCount semantics changed: profile value 3 should allow "
            f"{expected['voice_count']} voices, got {configured_voice_count}"
        )
    with tempfile.TemporaryDirectory(prefix="jianpu-stage-a-") as temp:
        temp_dir = Path(temp)
        midi = temp_dir / "fixture.mid"
        musicxml = temp_dir / "fixture.musicxml"
        _write_fixture_midi(midi, payload)
        _convert(muse, PROFILE, midi, musicxml, temp_dir)
        parsed = _parse_musicxml(notation_python, musicxml)
        required_pitches = sorted(int(value) for value in expected["triplet_pitches"])
        if parsed["meters"] != [expected["requires_meter"]]:
            raise RuntimeError(f"MuseScore -M meter mismatch: expected {expected['requires_meter']}, got {parsed['meters']}")
        if not set(required_pitches).issubset(parsed["pitches"]):
            raise RuntimeError(f"MuseScore -M lost fixture pitches: expected {required_pitches}, got {parsed['pitches']}")
        if parsed["tuplets"] < 1:
            raise RuntimeError("MuseScore -M did not emit an observable tuplet for the triplet fixture")
        if parsed["shortest_quarter_length"] > 1 / 3 + 1e-6:
            raise RuntimeError(
                "MuseScore -M did not retain the short triplet timing: "
                f"shortest quarterLength={parsed['shortest_quarter_length']}"
            )
        voice_keys = _musicxml_voice_keys(musicxml)
        if len(voice_keys) < 2:
            raise RuntimeError(f"MuseScore -M did not preserve multi-voice structure: voice_keys={voice_keys}")

        voice_payload = payload["voice_stress"]
        voice_midi = temp_dir / "voice-stress.mid"
        voice_musicxml = temp_dir / "voice-stress.musicxml"
        _write_fixture_midi(voice_midi, voice_payload, force_single_channel=True)
        _convert(muse, PROFILE, voice_midi, voice_musicxml, temp_dir)
        voice_keys = _musicxml_voice_keys(voice_musicxml)
        if len(voice_keys) != int(expected["voice_count"]):
            raise RuntimeError(
                f"MuseScore VoiceCount=3 did not yield the expected four voices: "
                f"voice_keys={voice_keys}"
            )
    print(
        json.dumps(
            {"status": "ok", "fixture": os.fspath(FIXTURE), "musicxml": parsed, "voice_keys": voice_keys},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"stage-a-smoke-failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
