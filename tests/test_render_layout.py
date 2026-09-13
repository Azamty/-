from __future__ import annotations

from pathlib import Path

import mido
import pytest

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.quantize import score_to_jianpu
from backend.jianpu_score.render import LILYPOND, JIANPU, _is_melody_harmony_score, render_score


pytestmark = pytest.mark.skipif(
    not JIANPU.is_file() or not LILYPOND.is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)


def _combined_fixture(*, combined: bool) -> Score:
    quarter_ticks = 48
    main_pitches = [60, 62, 64, 65] * 4
    accompaniment_pitches = [48, 50, 52, 53] * 2
    main_events = [
        *[
            ScoreNote(start_tick=index * quarter_ticks, duration_tick=quarter_ticks, midi=pitch)
            for index, pitch in enumerate(main_pitches)
        ],
    ]
    accompaniment_events = [
        *[
            ScoreNote(start_tick=index * quarter_ticks, duration_tick=quarter_ticks, midi=None)
            for index in range(8)
        ],
        *[
            ScoreNote(start_tick=index * quarter_ticks, duration_tick=quarter_ticks, midi=pitch)
            for index, pitch in zip(range(8, 16), accompaniment_pitches)
        ],
    ]
    metadata = {"melody_harmony": {"role_order": ["melody", "accompaniment"]}} if combined else {}
    return Score(
        title="small melody harmony layout fixture",
        bpm=96,
        key="C",
        time_signature="4/4",
        quarter_ticks=quarter_ticks,
        total_ticks=16 * quarter_ticks,
        voices=[
            ScoreVoice(
                voice_id="melody:voice-0",
                label="主旋律（候选） untied pitches",
                stem_id="melody-harmony",
                staff=1,
                source_voice="melody",
                events=main_events,
            ),
            ScoreVoice(
                voice_id="accompaniment:voice-0",
                label="伴奏和弦 4 chord lane 1",
                stem_id="melody-harmony",
                staff=2,
                source_voice="accompaniment",
                events=accompaniment_events,
            ),
        ],
        source="melody-harmony-composition" if combined else "instrument-part",
        metadata=metadata,
    )


def _sparse_combined_fixture() -> Score:
    """Use q0/s0 placeholders before a later system with real accompaniment."""

    quarter_ticks = 48
    bar_pattern = [24, 24, 24, 24] + [12] * 8
    positions: list[tuple[int, int]] = []
    cursor = 0
    for _ in range(8):
        for duration in bar_pattern:
            positions.append((cursor, duration))
            cursor += duration
    main_events = [
        ScoreNote(start_tick=start, duration_tick=duration, midi=60 + (index % 4) * 2)
        for index, (start, duration) in enumerate(positions)
    ]
    accompaniment_events = [
        ScoreNote(
            start_tick=start,
            duration_tick=duration,
            midi=None if index < len(positions) // 2 else 48 + (index % 4) * 2,
        )
        for index, (start, duration) in enumerate(positions)
    ]
    return Score(
        title="sparse q s melody harmony fixture",
        bpm=96,
        key="C",
        time_signature="4/4",
        quarter_ticks=quarter_ticks,
        total_ticks=cursor,
        voices=[
            ScoreVoice(
                voice_id="melody:voice-0",
                label="主旋律 helper",
                stem_id="melody-harmony",
                staff=1,
                source_voice="melody",
                events=main_events,
            ),
            ScoreVoice(
                voice_id="accompaniment:voice-0",
                label="伴奏和弦 helper",
                stem_id="melody-harmony",
                staff=2,
                source_voice="accompaniment",
                events=accompaniment_events,
            ),
        ],
        source="melody-harmony-composition",
        metadata={"melody_harmony": {"role_order": ["melody", "accompaniment"]}},
    )


def _midi_intervals(path: str, quarter_ticks: int) -> list[tuple[int, int, int]]:
    midi = mido.MidiFile(path)
    active: dict[tuple[int, int], list[int]] = {}
    intervals: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            key = (int(getattr(message, "channel", 0)), int(getattr(message, "note", -1)))
            if message.type == "note_on" and message.velocity:
                active.setdefault(key, []).append(absolute)
            elif message.type in {"note_off", "note_on"} and not message.velocity:
                starts = active.get(key, [])
                if starts:
                    intervals.append((key[1], starts.pop(0), absolute))
    scale = quarter_ticks / midi.ticks_per_beat
    return sorted((pitch, round(start * scale), round(end * scale)) for pitch, start, end in intervals)


def test_combined_render_hides_empty_lanes_and_uses_short_role_labels(tmp_path: Path) -> None:
    score = _combined_fixture(combined=True)
    assert _is_melody_harmony_score(score)

    artifacts = render_score(score, tmp_path, basename="combined-layout")
    lilypond = Path(artifacts.lilypond_path).read_text(encoding="utf-8")

    assert "\\RemoveAllEmptyStaves" in lilypond
    assert "indent = 26\\mm" in lilypond
    assert "short-indent = 18\\mm" in lilypond
    assert 'instrumentName = "主旋律"' in lilypond
    assert 'shortInstrumentName = "主旋律"' in lilypond
    assert 'instrumentName = "伴奏和弦"' in lilypond
    assert 'shortInstrumentName = "和弦"' in lilypond
    visible_labels = "\n".join(
        line for line in lilypond.splitlines() if "instrumentName" in line or "shortInstrumentName" in line
    )
    assert "untied pitches" not in visible_labels
    assert "tie pitch" not in visible_labels
    assert "chord lane" not in visible_labels
    assert artifacts.svg_paths

    expected = sorted(
        (event.midi, event.start_tick, event.end_tick)
        for voice in score.voices
        for event in voice.events
        if event.midi is not None
    )
    assert _midi_intervals(artifacts.midi_path or "", score.quarter_ticks) == expected


def test_combined_rest_hack_lane_hides_then_returns_without_midi_change(tmp_path: Path) -> None:
    score = _sparse_combined_fixture()
    jly = score_to_jianpu(score)
    assert "q0" in jly
    assert "s0" in jly

    artifacts = render_score(score, tmp_path, basename="sparse-combined-layout")
    lilypond = Path(artifacts.lilypond_path).read_text(encoding="utf-8")
    svg = "\n".join(Path(path).read_text(encoding="utf-8") for path in artifacts.svg_paths)

    # The rest hack's pitched c placeholders carry the marker but no longer
    # keep the otherwise empty first-system lane alive.  The same lane has
    # real notes later, so its short label still appears in the SVG.
    assert lilypond.count("composition-rest") >= 1
    assert "\\override VerticalAxisGroup.before-line-breaking = #composition-filter-rest-heads" in lilypond
    assert "伴奏和弦" not in svg
    assert "和弦" in svg

    expected = sorted(
        (event.midi, event.start_tick, event.end_tick)
        for voice in score.voices
        for event in voice.events
        if event.midi is not None
    )
    assert _midi_intervals(artifacts.midi_path or "", score.quarter_ticks) == expected


def test_non_combined_render_keeps_default_lilypond_layout(tmp_path: Path) -> None:
    score = _combined_fixture(combined=False)
    assert not _is_melody_harmony_score(score)

    artifacts = render_score(score, tmp_path, basename="ordinary-layout")
    lilypond = Path(artifacts.lilypond_path).read_text(encoding="utf-8")

    assert "\\RemoveAllEmptyStaves" not in lilypond
    assert "indent = 26\\mm" not in lilypond
    assert "composition-filter-rest-heads" not in lilypond
    assert "untied pitches" in lilypond
