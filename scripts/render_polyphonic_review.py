"""Render a small Score with independent voices for stage2 review evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mido

# Running a script by path places ``scripts`` first on sys.path; make the
# project package import explicit for the reproducible review command.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.render import render_score, write_score_json


OUT = ROOT / "artifacts" / "review" / "stage2-polyphonic"


def note(start: int, duration: int, midi: int | None, voice: str) -> ScoreNote:
    return ScoreNote(start_tick=start, duration_tick=duration, midi=midi, voice_id=voice, source="review")


def build_score() -> Score:
    return Score(
        title="Stage2 polyphonic review",
        bpm=80,
        key="C",
        time_signature="4/4",
        quarter_ticks=12,
        total_ticks=96,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[
                    note(0, 12, 60, "voice-0"),
                    note(12, 12, 62, "voice-0"),
                    note(24, 18, None, "voice-0"),
                    note(42, 12, 69, "voice-0"),  # crosses the bar at tick 48
                    note(54, 18, None, "voice-0"),
                    note(72, 12, 64, "voice-0"),
                    note(84, 12, None, "voice-0"),
                ],
                label="melody",
            ),
            ScoreVoice(
                voice_id="voice-1",
                events=[
                    note(0, 6, 64, "voice-1"),
                    note(6, 6, None, "voice-1"),
                    note(12, 4, 67, "voice-1"),
                    note(16, 4, 69, "voice-1"),
                    note(20, 4, 71, "voice-1"),
                    note(24, 12, None, "voice-1"),
                    note(36, 12, 67, "voice-1"),
                    note(48, 12, None, "voice-1"),
                    note(60, 12, 71, "voice-1"),
                    note(72, 18, None, "voice-1"),
                    note(90, 6, 72, "voice-1"),
                ],
                label="harmony 1",
            ),
            ScoreVoice(
                voice_id="voice-2",
                events=[
                    note(0, 12, 67, "voice-2"),
                    note(12, 6, None, "voice-2"),
                    note(18, 12, 72, "voice-2"),
                    note(30, 18, None, "voice-2"),
                    note(48, 12, 74, "voice-2"),
                    note(60, 18, None, "voice-2"),
                    note(78, 12, 76, "voice-2"),
                    note(90, 6, None, "voice-2"),
                ],
                label="harmony 2",
            ),
        ],
        source="review",
        metadata={"purpose": "stage2 independent async voices"},
    )


def inspect_midi(path: Path) -> dict[str, object]:
    midi = mido.MidiFile(path)
    tracks: list[dict[str, object]] = []
    for index, track in enumerate(midi.tracks):
        absolute = 0
        starts: list[tuple[int, int]] = []
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                starts.append((absolute, message.note))
        if starts:
            tracks.append({"index": index, "track_end": absolute, "starts": starts})

    starts_by_tick: dict[int, list[int]] = {}
    for track in tracks:
        for tick, pitch in track["starts"]:
            starts_by_tick.setdefault(tick, []).append(pitch)
    chord_ticks = {tick: sorted(pitches) for tick, pitches in starts_by_tick.items() if len(pitches) >= 3}
    all_start_ticks = sorted(starts_by_tick)
    evidence: dict[str, object] = {
        "ticks_per_beat": midi.ticks_per_beat,
        "track_count_with_notes": len(tracks),
        "tracks": [
            {"index": track["index"], "track_end": track["track_end"], "note_on_count": len(track["starts"])}
            for track in tracks
        ],
        "chord_ticks": {str(tick): pitches for tick, pitches in sorted(chord_ticks.items())},
        "all_start_ticks": all_start_ticks,
        "checks": {
            "three_independent_voice_tracks": len(tracks) == 3,
            "real_three_note_chord": bool(chord_ticks.get(0)),
            "asynchronous_voice_start": len(all_start_ticks) > 3,
            "two_bar_track_end": all(track["track_end"] >= 8 * midi.ticks_per_beat for track in tracks),
        },
    }
    if not all(evidence["checks"].values()):
        raise RuntimeError(f"polyphonic MIDI checks failed: {evidence}")
    return evidence


def main() -> int:
    score = build_score()
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = render_score(score, OUT, basename="polyphonic")
    write_score_json(score, OUT / "score.json")
    if not artifacts.midi_path:
        raise RuntimeError("polyphonic render did not produce MIDI")
    evidence = inspect_midi(Path(artifacts.midi_path))
    manifest = {"score": str(OUT / "score.json"), "artifacts": artifacts.model_dump(mode="json"), "midi_evidence": evidence}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
