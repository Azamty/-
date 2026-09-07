from __future__ import annotations

import pytest

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.quantize import JianpuSerializationError, score_to_jianpu
from backend.jianpu_score.render import render_score


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


def test_comprehensive_48_tpq_score_renders_to_svg(tmp_path) -> None:
    artifacts = render_score(_comprehensive_score(), tmp_path, basename="stage7a")

    assert artifacts.svg_paths
    assert all(path.endswith(".svg") for path in artifacts.svg_paths)
    assert artifacts.midi_path is not None
