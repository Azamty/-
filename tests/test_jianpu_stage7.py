from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from backend.jianpu_score import render as render_module
from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice, TempoEvent
from backend.jianpu_score.musicxml_standardize import _align_source_notes, _RawEvent
from backend.jianpu_score.quantize import (
    JianpuSerializationError,
    jianpu_serialization_diagnostics,
    score_to_jianpu,
)
from backend.jianpu_score.render import render_score
from scripts.musicxml_score_worker import _ordered_pitch_ties


def _comprehensive_score() -> Score:
    """A two-bar 48-TPQ score covering 7A notation boundaries."""

    durations = [
        ScoreNote(start_tick=0, duration_tick=6, midi=61),  # 32nd, accidental
        ScoreNote(start_tick=6, duration_tick=9, midi=74, dots=1),  # dotted 32nd
        ScoreNote(start_tick=15, duration_tick=12, midi=65),  # 16th
        ScoreNote(start_tick=27, duration_tick=18, midi=67, dots=1),  # dotted 16th
        ScoreNote(start_tick=45, duration_tick=36, midi=60, dots=1),  # dotted 8th
        ScoreNote(start_tick=81, duration_tick=48, midi=62),  # quarter
        ScoreNote(start_tick=129, duration_tick=24, midi=64),  # 8th
        ScoreNote(start_tick=153, duration_tick=18, midi=65, dots=1),
        ScoreNote(start_tick=171, duration_tick=12, midi=67),
        ScoreNote(start_tick=183, duration_tick=9, midi=69, dots=1),
        ScoreNote(start_tick=192, duration_tick=192, midi=60),  # whole note with dashes
    ]
    chord_crossing_bar = [
        ScoreNote(start_tick=0, duration_tick=144, midi=None),
        ScoreNote(start_tick=144, duration_tick=60, midi=61, chord_pitches=[61, 74]),
        ScoreNote(start_tick=204, duration_tick=144, midi=None),
        ScoreNote(start_tick=348, duration_tick=36, midi=None),
    ]
    explicit_triplet = [
        ScoreNote(start_tick=0, duration_tick=16, midi=60, tuplet_actual=3, tuplet_normal=2),
        ScoreNote(start_tick=16, duration_tick=8, midi=62, tuplet_actual=3, tuplet_normal=2),
        ScoreNote(start_tick=24, duration_tick=6, midi=64, tuplet_actual=3, tuplet_normal=2),
        ScoreNote(start_tick=30, duration_tick=144, midi=None),
        ScoreNote(start_tick=174, duration_tick=18, midi=None),
        ScoreNote(start_tick=192, duration_tick=72, midi=60, dots=1),  # dotted quarter
        ScoreNote(start_tick=264, duration_tick=72, midi=None),
        ScoreNote(start_tick=336, duration_tick=48, midi=None),
    ]
    return Score(
        title="stage 7A comprehensive",
        bpm=96,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=384,
        voices=[
            ScoreVoice(voice_id="durations", label="durations", events=durations),
            ScoreVoice(voice_id="chord-tie", label="chord tie", events=chord_crossing_bar),
            ScoreVoice(voice_id="tuplet", label="explicit triplet", events=explicit_triplet),
        ],
    )


def test_48_tpq_serializer_preserves_durations_chord_and_explicit_tuplet() -> None:
    jianpu = score_to_jianpu(_comprehensive_score())

    assert "d#1" in jianpu
    assert "d2'." in jianpu
    assert "s4" in jianpu and "s5." in jianpu
    assert "q1." in jianpu
    assert "1." in jianpu
    assert "1 - - -" in jianpu
    assert "#12' ~" in jianpu  # C#4 + D5 chord, followed by a cross-bar tie
    assert "3[ q1 s2 d3. ]" in jianpu
    assert jianpu.count("NextPart") == 2
    assert jianpu.count("|") == 6
    assert jianpu.count("OctavesAfter") == 3


def test_chord_token_preserves_independent_extreme_octaves() -> None:
    from backend.jianpu_score.quantize import _pitch_token

    # Vendor syntax is a single simple-chord token with octave marks after
    # each figure; the repeated OctavesAfter directive disambiguates the
    # marks.  No pitch is reduced to the chord's lowest tone.
    assert _pitch_token((0, 60, 127), "C") == "1,,,,,15'''''"


@pytest.mark.skipif(
    not render_module.JIANPU.is_file() or not render_module.LILYPOND.is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)
def test_extreme_multi_octave_chord_renders_without_pitch_loss(tmp_path: Path) -> None:
    score = Score(
        title="extreme octave chord",
        bpm=96,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[
            ScoreVoice(
                voice_id="extreme",
                events=[
                    ScoreNote(start_tick=0, duration_tick=48, midi=0, chord_pitches=[0, 60, 127]),
                    ScoreNote(start_tick=48, duration_tick=144, midi=None),
                ],
            )
        ],
    )

    artifacts = render_score(score, tmp_path, basename="extreme-octave")
    assert artifacts.svg_paths
    assert artifacts.midi_path is not None
    midi = mido.MidiFile(artifacts.midi_path)
    pitches = {message.note for track in midi.tracks for message in track if message.type == "note_on" and message.velocity}
    assert {0, 60, 127} <= pitches


@pytest.mark.skipif(
    not render_module.JIANPU.is_file() or not render_module.LILYPOND.is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)
def test_b_flat_minor_nonleading_accidental_chord_splits_without_pitch_or_timing_loss(tmp_path: Path) -> None:
    """A vendor simple-chord accidental bug must be isolated to pitch lanes.

    In B-flat minor, ``[46, 50, 53]`` would serialize as ``6,,#1,3,``;
    jianpu-ly applies that middle accidental to the wrong figure.  The
    serializer therefore emits three NextPart lanes, each with the original
    onset and duration.  This also proves that an unrelated voice is not
    copied into each lane.
    """

    score = Score(
        title="B-flat minor chord",
        bpm=68,
        key="Bbm",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[
            ScoreVoice(
                voice_id="chord",
                events=[
                    ScoreNote(start_tick=0, duration_tick=48, midi=46, chord_pitches=[46, 50, 53]),
                    ScoreNote(start_tick=48, duration_tick=144, midi=None),
                ],
            ),
            ScoreVoice(
                voice_id="melody",
                events=[
                    ScoreNote(start_tick=0, duration_tick=192, midi=60),
                ],
            ),
        ],
    )

    diagnostics = jianpu_serialization_diagnostics(score)
    assert diagnostics["chord_voice_split_count"] == 1
    assert diagnostics["chord_voice_split_lane_count"] == 2
    assert diagnostics["chord_voice_splits"][0]["reason"] == "non_leading_accidental_in_simple_chord"
    jianpu = score_to_jianpu(score)
    assert "6,,#1,3," not in jianpu
    assert jianpu.count("NextPart") == 3

    artifacts = render_score(score, tmp_path, basename="bbm-chord")
    assert artifacts.midi_path is not None
    midi = mido.MidiFile(artifacts.midi_path)
    intervals: dict[int, list[tuple[int, int]]] = {}
    for track in midi.tracks:
        absolute = 0
        active: dict[int, list[int]] = {}
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity:
                active.setdefault(message.note, []).append(absolute)
            elif message.type in {"note_off", "note_on"} and not message.velocity:
                starts = active.get(message.note, [])
                if starts:
                    intervals.setdefault(message.note, []).append((starts.pop(0), absolute))

    assert {46, 50, 53, 60} <= set(intervals)
    assert all(intervals[pitch] == [(0, 384)] for pitch in (46, 50, 53))
    assert intervals[60] == [(0, 1536)]


@pytest.mark.skipif(
    not render_module.JIANPU.is_file() or not render_module.LILYPOND.is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)
def test_unsafe_chord_lanes_reuse_while_base_keeps_melody(tmp_path: Path) -> None:
    score = Score(
        title="reusable chord lanes",
        bpm=68,
        key="Bbm",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=384,
        voices=[
            ScoreVoice(
                voice_id="melody",
                events=[
                    ScoreNote(start_tick=0, duration_tick=48, midi=60),
                    ScoreNote(start_tick=48, duration_tick=48, midi=46, chord_pitches=[46, 50, 53]),
                    ScoreNote(start_tick=96, duration_tick=48, midi=62),
                    ScoreNote(start_tick=144, duration_tick=48, midi=50, chord_pitches=[50, 54, 57]),
                    ScoreNote(start_tick=192, duration_tick=192, midi=None),
                ],
            )
        ],
    )

    diagnostics = jianpu_serialization_diagnostics(score)
    assert diagnostics["occurrence_count"] == 2
    assert diagnostics["actual_added_lane_count"] == 2
    assert diagnostics["voices"][0]["actual_added_lane_count"] == 2
    assert [item["lane_assignments"] for item in diagnostics["chord_voice_splits"]] == [
        [
            {"pitch": 46, "lane": 0},
            {"pitch": 50, "lane": 1},
            {"pitch": 53, "lane": 2},
        ],
        [
            {"pitch": 50, "lane": 0},
            {"pitch": 54, "lane": 1},
            {"pitch": 57, "lane": 2},
        ],
    ]
    assert score_to_jianpu(score).count("NextPart") == 2

    artifacts = render_score(score, tmp_path, basename="reusable-chord-lanes")
    midi = mido.MidiFile(artifacts.midi_path)
    intervals: dict[int, list[tuple[int, int]]] = {}
    for track in midi.tracks:
        absolute = 0
        active: dict[int, list[int]] = {}
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity:
                active.setdefault(message.note, []).append(absolute)
            elif message.type in {"note_off", "note_on"} and not message.velocity:
                starts = active.get(message.note, [])
                if starts:
                    intervals.setdefault(message.note, []).append((starts.pop(0), absolute))

    for values in intervals.values():
        values.sort()
    assert intervals[60] == [(0, 384)]
    assert intervals[62] == [(768, 1152)]
    assert intervals[46] == [(384, 768)]
    assert intervals[50] == [(384, 768), (1152, 1536)]
    assert intervals[53] == [(384, 768)]
    assert intervals[54] == [(1152, 1536)]
    assert intervals[57] == [(1152, 1536)]


@pytest.mark.skipif(
    not render_module.JIANPU.is_file() or not render_module.LILYPOND.is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)
def test_unsafe_chord_tie_chain_stays_on_the_same_reusable_lanes(tmp_path: Path) -> None:
    score = Score(
        title="tied unsafe chord",
        bpm=80,
        key="Bbm",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[
            ScoreVoice(
                voice_id="tied-chord",
                events=[
                    ScoreNote(start_tick=0, duration_tick=48, midi=60),
                    ScoreNote(
                        start_tick=48,
                        duration_tick=48,
                        midi=46,
                        chord_pitches=[46, 50, 53],
                        tie_types=["start", "start", "start"],
                    ),
                    ScoreNote(
                        start_tick=96,
                        duration_tick=48,
                        midi=46,
                        chord_pitches=[46, 50, 53],
                        tie_types=["stop", "stop", "stop"],
                    ),
                    ScoreNote(start_tick=144, duration_tick=48, midi=None),
                ],
            )
        ],
    )

    diagnostics = jianpu_serialization_diagnostics(score)
    assert diagnostics["actual_added_lane_count"] == 2
    assignments = [item["lane_assignments"] for item in diagnostics["chord_voice_splits"]]
    assert assignments[0] == assignments[1]
    jianpu = score_to_jianpu(score)
    assert jianpu.count("~") == 3

    artifacts = render_score(score, tmp_path, basename="tied-unsafe-chord")
    midi = mido.MidiFile(artifacts.midi_path)
    intervals: dict[int, list[tuple[int, int]]] = {}
    for track in midi.tracks:
        absolute = 0
        active: dict[int, list[int]] = {}
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity:
                active.setdefault(message.note, []).append(absolute)
            elif message.type in {"note_off", "note_on"} and not message.velocity:
                starts = active.get(message.note, [])
                if starts:
                    intervals.setdefault(message.note, []).append((starts.pop(0), absolute))

    for values in intervals.values():
        values.sort()
    assert all(intervals[pitch] == [(384, 1152)] for pitch in (46, 50, 53))


def test_explicit_tuplet_ratio_is_required_to_be_three_over_two() -> None:
    score = Score(
        title="unsupported tuplet",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=48,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[
                    ScoreNote(start_tick=0, duration_tick=16, midi=60, tuplet_actual=5, tuplet_normal=4),
                    ScoreNote(start_tick=16, duration_tick=16, midi=62, tuplet_actual=5, tuplet_normal=4),
                    ScoreNote(start_tick=32, duration_tick=16, midi=64, tuplet_actual=5, tuplet_normal=4),
                ],
            )
        ],
    )

    with pytest.raises(JianpuSerializationError, match="only supports explicit 3:2"):
        score_to_jianpu(score)


def test_explicit_tuplet_metadata_takes_precedence_over_legacy_shape() -> None:
    score = _comprehensive_score()
    # The first three events have unequal source durations, so only their
    # explicit 3:2 metadata can authorize the tuplet group.
    assert "3[" in score_to_jianpu(score)


def _tuple_fragment_score(
    fragments: list[tuple[int, int, int | None, str | None]],
    *,
    title: str,
) -> Score:
    """Build one bar with explicit MusicXML-shaped tuplet fragments."""

    events: list[ScoreNote] = []
    cursor = 0
    for duration, midi, actual, boundary in fragments:
        events.append(
            ScoreNote(
                start_tick=cursor,
                duration_tick=duration,
                midi=midi,
                tuplet_actual=actual,
                tuplet_normal=2 if actual is not None else None,
                tuplet_type=boundary,
            )
        )
        cursor += duration
    if cursor < 192:
        events.append(ScoreNote(start_tick=cursor, duration_tick=192 - cursor, midi=None))
    return Score(
        title=title,
        bpm=96,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[ScoreVoice(voice_id="tuplet", events=events)],
    )


def test_explicit_tuplet_boundary_accepts_four_note_rest_fragments() -> None:
    # MuseScore/music21 emits this shape for a legal 3:2 group: a 4-tick
    # rest, 8-tick note, 4-tick rest, and 8-tick note.  It is one group by
    # MusicXML start/stop boundaries even though it has four events.
    score = _tuple_fragment_score(
        [
            (4, None, 3, "start"),
            (8, 80, 3, None),
            (4, None, 3, None),
            (8, 77, 3, "stop"),
        ],
        title="explicit four-fragment triplet",
    )

    jianpu = score_to_jianpu(score)

    assert jianpu.count("3[") == 1
    assert jianpu.count("]") == 1
    assert "d0" in jianpu and "s#5'" in jianpu


def test_explicit_tuplet_boundary_accepts_chord_and_rest_members() -> None:
    score = _tuple_fragment_score(
        [
            (8, None, 3, "start"),
            (8, 60, 3, None),
            (8, 64, 3, "stop"),
        ],
        title="explicit rest chord triplet",
    )
    # Make the middle event a chord without changing its timeline.
    score.voices[0].events[1].chord_pitches = [60, 64, 67]

    jianpu = score_to_jianpu(score)

    assert "3[" in jianpu and "]" in jianpu
    assert "135" in jianpu


def test_explicit_tuplet_boundary_can_cross_a_measure() -> None:
    events = [
        ScoreNote(start_tick=0, duration_tick=144, midi=None),
        ScoreNote(start_tick=144, duration_tick=36, midi=None),
        ScoreNote(start_tick=180, duration_tick=6, midi=None),
        ScoreNote(start_tick=186, duration_tick=8, midi=60, tuplet_actual=3, tuplet_normal=2, tuplet_type="start"),
        ScoreNote(start_tick=194, duration_tick=8, midi=None, tuplet_actual=3, tuplet_normal=2),
        ScoreNote(start_tick=202, duration_tick=8, midi=62, tuplet_actual=3, tuplet_normal=2, tuplet_type="stop"),
        ScoreNote(start_tick=210, duration_tick=144, midi=None),
        ScoreNote(start_tick=354, duration_tick=24, midi=None),
        ScoreNote(start_tick=378, duration_tick=6, midi=None),
    ]
    score = Score(
        title="cross measure triplet",
        bpm=96,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=384,
        voices=[ScoreVoice(voice_id="cross", events=events)],
    )

    jianpu = score_to_jianpu(score)
    lines = [line for line in jianpu.splitlines() if "3[" in line or "]" in line]

    assert lines
    assert "3[" in lines[0].split("|")[0]
    assert "]" in lines[0].split("|")[1]


def test_explicit_tuplet_boundary_rejects_incomplete_or_ambiguous_groups() -> None:
    incomplete = _tuple_fragment_score(
        [(8, 60, 3, "start"), (8, 62, 3, None)],
        title="incomplete explicit triplet",
    )
    with pytest.raises(JianpuSerializationError, match="gap or inconsistent ratio|no stop boundary"):
        score_to_jianpu(incomplete)

    ambiguous = _tuple_fragment_score(
        [(8, 60, 3, None), (8, 62, 3, None), (8, 64, 3, None), (8, 65, 3, None)],
        title="ambiguous unbounded triplet",
    )
    with pytest.raises(JianpuSerializationError, match="complete three-note group"):
        score_to_jianpu(ambiguous)


def test_partial_chord_tie_is_split_into_safe_parts() -> None:
    score = Score(
        title="partial chord tie",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                label="partial chord",
                events=[
                    ScoreNote(
                        start_tick=0,
                        duration_tick=48,
                        midi=60,
                        chord_pitches=[60, 64, 67],
                        tie_types=["start", None, None],
                    ),
                    ScoreNote(
                        start_tick=48,
                        duration_tick=48,
                        midi=60,
                        chord_pitches=[60, 64, 67],
                        tie_types=["stop", None, None],
                    ),
                    ScoreNote(start_tick=96, duration_tick=96, midi=None),
                ],
            )
        ],
    )

    jianpu = score_to_jianpu(score)

    assert jianpu.count("NextPart") == 1
    assert "35 35" in jianpu
    assert "1 ~ 1" in jianpu
    assert "35 ~" not in jianpu


def test_worker_keeps_partial_chord_tie_slots_attached_to_sorted_pitches() -> None:
    class Pitch:
        def __init__(self, midi: int) -> None:
            self.midi = midi

    class Tie:
        def __init__(self, kind: str | None) -> None:
            self.type = kind

    class ChordNote:
        def __init__(self, midi: int, tie: str | None) -> None:
            self.pitch = Pitch(midi)
            self.tie = Tie(tie) if tie is not None else None

    pitches, tie_types = _ordered_pitch_ties(
        [ChordNote(67, None), ChordNote(60, "start"), ChordNote(64, None)]
    )

    assert pitches == [60, 64, 67]
    assert tie_types == ["start", None, None]


def test_source_alignment_follows_only_the_tied_chord_pitch() -> None:
    events = [
        _RawEvent(
            event_id="first",
            part_group="p1",
            part_id="p1",
            staff=1,
            voice="1",
            start_tick=0,
            end_tick=48,
            pitches=[60, 64, 67],
            kind="chord",
            tie=None,
            tie_types=["start", None, None],
            tuplet_actual=None,
            tuplet_normal=None,
            dots=0,
            measure_number=1,
            metadata={},
        ),
        _RawEvent(
            event_id="second",
            part_group="p1",
            part_id="p1",
            staff=1,
            voice="1",
            start_tick=48,
            end_tick=96,
            pitches=[60, 64, 67],
            kind="chord",
            tie=None,
            tie_types=["stop", None, None],
            tuplet_actual=None,
            tuplet_normal=None,
            dots=0,
            measure_number=1,
            metadata={},
        ),
    ]
    source_notes = [
        {
            "source_index": index,
            "midi": pitch,
            "start_tick_480": start * 10,
            "end_tick_480": end * 10,
            "start_tick": start,
            "end_tick": end,
        }
        for index, (pitch, start, end) in enumerate(
            [(60, 0, 96), (64, 0, 48), (67, 0, 48), (64, 48, 96), (67, 48, 96)]
        )
    ]

    report = _align_source_notes(events, source_notes)
    c_report = next(item for item in report if item["source_midi"] == 60)
    e_reports = [item for item in report if item["source_midi"] == 64]
    g_reports = [item for item in report if item["source_midi"] == 67]

    assert c_report["musicxml_event_ids"] == ["first", "second"]
    assert all(item["musicxml_event_ids"] == ["first"] for item in e_reports[:1])
    assert all(item["musicxml_event_ids"] == ["first"] for item in g_reports[:1])
    assert all(item["musicxml_event_ids"] == ["second"] for item in e_reports[1:])
    assert all(item["musicxml_event_ids"] == ["second"] for item in g_reports[1:])


def test_explicit_dots_must_match_the_encoded_duration() -> None:
    score = Score(
        title="invalid dots",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=48,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[ScoreNote(start_tick=0, duration_tick=48, midi=60, dots=1)],
            )
        ],
    )

    with pytest.raises(JianpuSerializationError, match="explicit dots=1"):
        score_to_jianpu(score)


def test_dotted_half_uses_jianpu_dash_extension() -> None:
    score = Score(
        title="dotted half",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[
                    ScoreNote(start_tick=0, duration_tick=144, midi=60, dots=1),
                    ScoreNote(start_tick=144, duration_tick=48, midi=None),
                ],
            )
        ],
    )

    assert "1 - - 0 |" in score_to_jianpu(score)


def test_comprehensive_48_tpq_score_renders_to_svg(tmp_path) -> None:
    artifacts = render_score(_comprehensive_score(), tmp_path, basename="stage7a")

    assert artifacts.svg_paths
    assert all(path.endswith(".svg") for path in artifacts.svg_paths)
    assert artifacts.midi_path is not None


@pytest.mark.skipif(
    not render_module.JIANPU.is_file() or not render_module.LILYPOND.is_file(),
    reason="pinned jianpu-ly or LilyPond is unavailable",
)
def test_real_stage56_score_roundtrips_to_svg_and_preserves_pitch_set(tmp_path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "high_accuracy" / "stage56" / "stage56.score.json"
    score = Score.model_validate(json.loads(fixture.read_text(encoding="utf-8")))
    artifacts = render_score(score, tmp_path, basename="stage56-review")

    expected_pitches = {
        pitch
        for voice in score.voices
        for event in voice.events
        for pitch in (event.chord_pitches or ([event.midi] if event.midi is not None else []))
    }
    expected_note_count = sum(
        len(event.chord_pitches or ([event.midi] if event.midi is not None else []))
        for voice in score.voices
        for event in voice.events
    )
    midi = mido.MidiFile(artifacts.midi_path)
    note_ons = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]

    assert artifacts.svg_paths
    assert set(note_ons) == expected_pitches
    assert len(expected_pitches) <= len(note_ons) <= expected_note_count


def _pickup_score(meter: str, pickup_ticks: int) -> Score:
    quarter_ticks = 48
    full_ticks = {
        "4/4": 192,
        "6/8": 144,
    }[meter]
    final_ticks = full_ticks - pickup_ticks
    total_ticks = pickup_ticks + full_ticks + final_ticks
    return Score(
        title=f"{meter} pickup",
        bpm=108,
        key="C",
        time_signature=meter,
        quarter_ticks=quarter_ticks,
        total_ticks=total_ticks,
        voices=[
            ScoreVoice(
                voice_id="pickup",
                events=[
                    ScoreNote(start_tick=0, duration_tick=pickup_ticks, midi=60),
                    ScoreNote(start_tick=pickup_ticks, duration_tick=full_ticks, midi=62),
                    ScoreNote(start_tick=pickup_ticks + full_ticks, duration_tick=final_ticks, midi=64),
                ],
            )
        ],
        metadata={
            "timeline_measures": [
                {
                    "start_tick": 0,
                    "duration_tick": pickup_ticks,
                    "end_tick": pickup_ticks,
                    "time_signature": meter,
                    "is_pickup": True,
                },
                {
                    "start_tick": pickup_ticks,
                    "duration_tick": full_ticks,
                    "end_tick": pickup_ticks + full_ticks,
                    "time_signature": meter,
                    "is_pickup": False,
                },
                {
                    "start_tick": pickup_ticks + full_ticks,
                    "duration_tick": final_ticks,
                    "end_tick": total_ticks,
                    "time_signature": meter,
                    "is_pickup": False,
                },
            ],
            "pickup": {"is_pickup": True, "duration_tick": pickup_ticks},
        },
    )


@pytest.mark.parametrize(
    ("meter", "pickup_ticks", "header"),
    [("4/4", 24, "4/4,8"), ("6/8", 48, "6/8,4"), ("6/8", 72, "6/8,4.")],
)
def test_timeline_pickups_use_exact_jianpu_header_and_render(
    tmp_path, meter: str, pickup_ticks: int, header: str
) -> None:
    score = _pickup_score(meter, pickup_ticks)
    jianpu = score_to_jianpu(score)

    assert header in jianpu.splitlines()
    artifacts = render_score(score, tmp_path / meter.replace("/", "-"), basename="pickup")
    assert artifacts.svg_paths
    midi = mido.MidiFile(artifacts.midi_path)
    expected_end = score.total_ticks * midi.ticks_per_beat // score.quarter_ticks
    note_tracks = [track for track in midi.tracks if any(message.type == "note_on" for message in track)]
    assert note_tracks
    assert all(sum(message.time for message in track) == expected_end for track in note_tracks)
    assert {
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    } == {60, 62, 64}


def _meter_key_change_score() -> Score:
    quarter_ticks = 48
    measure_lengths = [144, 192, 192]
    starts = [0, 144, 336]
    total_ticks = sum(measure_lengths)
    voice_one = [
        ScoreNote(start_tick=start, duration_tick=48, midi=midi)
        for start, midi in zip(
            [0, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480],
            [60, 62, 64, 66, 67, 69, 71, 69, 71, 72, 74],
        )
    ]
    voice_two = [
        ScoreNote(start_tick=0, duration_tick=144, midi=None),
        ScoreNote(start_tick=144, duration_tick=96, midi=60),
        ScoreNote(start_tick=240, duration_tick=96, midi=None),
        ScoreNote(start_tick=336, duration_tick=96, midi=69),
        ScoreNote(start_tick=432, duration_tick=96, midi=None),
    ]
    return Score(
        title="meter key tempo changes",
        bpm=120,
        key="C",
        time_signature="3/4",
        quarter_ticks=quarter_ticks,
        total_ticks=total_ticks,
        voices=[
            ScoreVoice(voice_id="melody", label="melody", events=voice_one),
            ScoreVoice(voice_id="harmony", label="harmony", events=voice_two),
        ],
        tempo_events=[
            TempoEvent(start_tick=0, bpm=120),
            TempoEvent(start_tick=48, bpm=120.4),
            TempoEvent(start_tick=144, bpm=121),
            TempoEvent(start_tick=336, bpm=122),
            TempoEvent(start_tick=384, bpm=122.4),
        ],
        metadata={
            "timeline_measures": [
                {
                    "start_tick": start,
                    "duration_tick": duration,
                    "end_tick": start + duration,
                    "time_signature": meter,
                    "is_pickup": False,
                }
                for start, duration, meter in zip(starts, measure_lengths, ["3/4", "4/4", "4/4"])
            ],
            "time_signature_events": [
                {"start_tick": 0, "time_signature": "3/4"},
                {"start_tick": 144, "time_signature": "4/4"},
            ],
            "key_signature_events": [
                {"start_tick": 0, "key": "C"},
                {"start_tick": 144, "key": "D"},
                {"start_tick": 336, "key": "Am"},
            ],
        },
    )


def test_timeline_meter_key_tempo_changes_are_per_measure_and_render(tmp_path) -> None:
    score = _meter_key_change_score()
    jianpu = score_to_jianpu(score)

    assert "3/4\n" in jianpu
    assert "4/4 1=D 4=121" in jianpu
    assert "| 1=C 4=122" in jianpu
    assert jianpu.count("4=120") == 2  # initial header, once for each NextPart
    assert jianpu.count("4=121") == 2
    assert jianpu.count("4=122") == 2
    artifacts = render_score(score, tmp_path, basename="meter-key")
    assert artifacts.svg_paths
    midi = mido.MidiFile(artifacts.midi_path)
    expected_end = score.total_ticks * midi.ticks_per_beat // score.quarter_ticks
    note_tracks = [track for track in midi.tracks if any(message.type == "note_on" for message in track)]
    assert len(note_tracks) == 2
    assert all(sum(message.time for message in track) == expected_end for track in note_tracks)
    assert {
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    } == {60, 62, 64, 66, 67, 69, 71, 72, 74}


def test_timeline_measure_validation_and_boundary_events_are_strict() -> None:
    score = Score(
        title="invalid timeline",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[ScoreVoice(voice_id="voice-0", events=[ScoreNote(start_tick=0, duration_tick=192, midi=60)])],
        metadata={
            "timeline_measures": [
                {"start_tick": 0, "duration_tick": 96, "end_tick": 96, "time_signature": "4/4"},
                {"start_tick": 100, "duration_tick": 92, "end_tick": 192, "time_signature": "4/4"},
            ],
            "time_signature_events": [{"start_tick": 48, "time_signature": "3/4"}],
        },
    )

    with pytest.raises(JianpuSerializationError, match="illegal gap"):
        score_to_jianpu(score)

    score.metadata["timeline_measures"][1]["start_tick"] = 96
    score.metadata["timeline_measures"][1]["duration_tick"] = 96
    with pytest.raises(JianpuSerializationError, match="not a measure boundary"):
        score_to_jianpu(score)


def test_pickup_duration_must_be_exactly_representable() -> None:
    score = _pickup_score("4/4", 16)
    with pytest.raises(JianpuSerializationError, match="pickup duration"):
        score_to_jianpu(score)


def test_unmarked_fine_atom_failure_keeps_local_import_context() -> None:
    score = Score(
        title="unmarked fine atom",
        bpm=100,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=192,
        voices=[
            ScoreVoice(
                voice_id="P1:staff-1:voice-2:lane-1",
                events=[
                    ScoreNote(start_tick=0, duration_tick=24, midi=None),
                    ScoreNote(
                        start_tick=24,
                        duration_tick=2,
                        midi=None,
                        metadata={"musicxml_event_id": "p0:e41"},
                    ),
                    ScoreNote(start_tick=26, duration_tick=166, midi=60),
                ],
            )
        ],
    )

    with pytest.raises(JianpuSerializationError) as raised:
        score_to_jianpu(score)

    message = str(raised.value)
    assert "duration 2 ticks cannot be represented" in message
    assert "voice='P1:staff-1:voice-2:lane-1'" in message
    assert "event=24:26" in message
    assert "kind=rest" in message
    assert "source_event_id='p0:e41'" in message
    assert "tie_before=[] tie_after=[] tuplet=none" in message
    assert "measure=0:192 remaining_after=166" in message
    assert "previous=rest@0:24" in message
    assert "following=note@26:192/pitches=[60]" in message


def test_timeline_key_change_uses_boundary_and_rejects_mid_measure() -> None:
    score = _meter_key_change_score()
    score.metadata["key_signature_events"] = [{"start_tick": 96, "key": "D"}]

    with pytest.raises(JianpuSerializationError, match="key_signature_events event.*boundary"):
        score_to_jianpu(score)
