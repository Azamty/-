from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from backend.jianpu_score.domain import Score, ScoreNote, ScoreVoice
from backend.jianpu_score.high_accuracy_service import HighAccuracyBuildResult, _score_note_intervals
from backend.jianpu_score.quantize import score_to_jianpu
from backend.v2_job_manager import V2JobService


def _note(
    start: int,
    end: int,
    pitch: int | None,
    event_id: str,
    *,
    tuplet_actual: int | None = None,
    tuplet_normal: int | None = None,
    tuplet_type: str | None = None,
    tie: str | None = None,
) -> ScoreNote:
    return ScoreNote(
        start_tick=start,
        duration_tick=end - start,
        midi=pitch,
        chord_pitches=[pitch] if pitch is not None else [],
        tie=tie,
        tie_types=[tie] if pitch is not None else [],
        tuplet_actual=tuplet_actual,
        tuplet_normal=tuplet_normal,
        tuplet_type=tuplet_type,
        voice_id="source",
        metadata={"musicxml_event_id": event_id},
    )


def _score(voices: list[ScoreVoice], *, total_ticks: int) -> Score:
    return Score(
        title="composition fixture",
        bpm=120,
        key="C",
        time_signature="4/4",
        quarter_ticks=48,
        total_ticks=total_ticks,
        voices=voices,
        source="fixture",
    )


def _result(score: Score, source_to_score: list[dict[str, object]]) -> HighAccuracyBuildResult:
    return HighAccuracyBuildResult(
        instrument_id="fixture-track",
        title="fixture",
        variant="instrument-part",
        program=0,
        is_drum=False,
        status="completed",
        jianpu_status="completed",
        output_dir=Path("."),
        manifest_path=Path("manifest.json"),
        artifacts=(),
        performance_metadata={},
        score=score,
        alignment_report={"source_to_score": source_to_score},
    )


def _alignment(index: int, pitch: int, start: int, end: int, *event_ids: str) -> dict[str, object]:
    return {
        "source_index": index,
        "source_midi": pitch,
        "score_start_tick": start,
        "score_end_tick": end,
        "musicxml_event_ids": list(event_ids),
    }


def _compose(
    score: Score,
    alignment: list[dict[str, object]],
    candidate_indices: set[int],
) -> tuple[Score, dict[str, object]]:
    result = _result(score, alignment)
    return V2JobService._compose_melody_harmony_score(
        (({"track_id": "fixture-track"}, (), result),),
        {"fixture-track": candidate_indices},
        title="fixture",
        selected_track_ids=["fixture-track"],
    )


def test_composition_uses_alignment_identity_for_duplicate_pitch_slots() -> None:
    score = _score(
        [
            ScoreVoice(
                voice_id="voice-a",
                events=[_note(0, 48, 60, "e0")],
            ),
            ScoreVoice(
                voice_id="voice-b",
                events=[_note(0, 48, 60, "e1")],
            ),
        ],
        total_ticks=48,
    )
    composed, report = _compose(
        score,
        [_alignment(0, 60, 0, 48, "e0"), _alignment(1, 60, 0, 48, "e1")],
        {0},
    )

    assert report["melody_pitch_slot_count"] == 1
    assert report["accompaniment_pitch_slot_count"] == 1
    assert Counter(_score_note_intervals(composed)) == Counter(_score_note_intervals(score))
    assert score_to_jianpu(composed)


def test_composition_keeps_tuplet_grid_and_tied_source_lane() -> None:
    source_voice = ScoreVoice(
        voice_id="voice-a",
        events=[
            _note(0, 16, 60, "e0", tuplet_actual=3, tuplet_normal=2, tuplet_type="start"),
            _note(16, 32, 62, "e1", tuplet_actual=3, tuplet_normal=2),
            _note(32, 48, 64, "e2", tuplet_actual=3, tuplet_normal=2, tuplet_type="stop"),
            _note(48, 96, 65, "e3", tie="start"),
            _note(96, 144, 65, "e4", tie="stop"),
        ],
    )
    # This independent event starts at the held note's stop boundary.  A
    # greedy global lane allocator would put it in the tie lane and break the
    # source chain; source-voice lane affinity must keep it separate.
    accompaniment_voice = ScoreVoice(
        voice_id="voice-b",
        events=[
            _note(0, 96, None, "r0"),
            _note(96, 120, 55, "e5"),
            _note(120, 144, None, "r1"),
        ],
    )
    score = _score([source_voice, accompaniment_voice], total_ticks=144)
    composed, report = _compose(
        score,
        [
            _alignment(0, 60, 0, 16, "e0"),
            _alignment(1, 62, 16, 32, "e1"),
            _alignment(2, 64, 32, 48, "e2"),
            # Only the first tie fragment is selected by identity; the
            # composer must propagate the role to its stop fragment.
            _alignment(3, 65, 48, 96, "e3"),
            _alignment(4, 55, 96, 120, "e5"),
        ],
        {0, 3},
    )

    assert report["melody_pitch_slot_count"] == 3
    assert report["accompaniment_pitch_slot_count"] == 3
    assert Counter(_score_note_intervals(composed)) == Counter(_score_note_intervals(score))
    # This exercises explicit start/stop tuplets, role-local tuplet rests, and
    # tied notes after the projection rather than merely checking note counts.
    assert score_to_jianpu(composed)


def test_composition_rejects_mismatched_source_timelines() -> None:
    score_a = _score([ScoreVoice(voice_id="a", events=[_note(0, 48, 60, "e0")])], total_ticks=48)
    score_b = _score([ScoreVoice(voice_id="b", events=[_note(0, 64, 64, "e1")])], total_ticks=64)
    result_a = _result(score_a, [_alignment(0, 60, 0, 48, "e0")])
    result_b = _result(score_b, [_alignment(0, 64, 0, 64, "e1")])

    with pytest.raises(ValueError, match="tempo/timeline"):
        V2JobService._compose_melody_harmony_score(
            (
                ({"track_id": "a"}, (), result_a),
                ({"track_id": "b"}, (), result_b),
            ),
            {"a": {0}, "b": set()},
            title="fixture",
            selected_track_ids=["a", "b"],
        )


def test_composition_merges_only_ordinary_same_instrument_accompaniment_chords() -> None:
    score = _score(
        [
            ScoreVoice(
                voice_id="voice-a",
                events=[_note(0, 48, 60, "e0"), _note(48, 96, 52, "e1")],
            ),
            ScoreVoice(
                voice_id="voice-b",
                events=[_note(0, 48, None, "r0"), _note(48, 96, 55, "e2")],
            ),
        ],
        total_ticks=96,
    )
    composed, report = _compose(
        score,
        [
            _alignment(0, 60, 0, 48, "e0"),
            _alignment(1, 52, 48, 96, "e1"),
            _alignment(2, 55, 48, 96, "e2"),
        ],
        {0},
    )

    assert report["accompaniment_merged_group_count"] == 1
    chord_events = [
        event
        for voice in composed.voices
        for event in voice.events
        if set(event.chord_pitches) == {52, 55}
    ]
    assert len(chord_events) == 1
    assert Counter(_score_note_intervals(composed)) == Counter(_score_note_intervals(score))
    assert score_to_jianpu(composed)


def test_melody_lanes_reuse_disjoint_source_voices_but_reserve_tie_blocks() -> None:
    score = _score(
        [
            ScoreVoice(
                voice_id="voice-a",
                events=[
                    _note(0, 48, 69, "a0"),
                    _note(48, 96, 69, "a1", tie="start"),
                    _note(96, 144, 69, "a2", tie="stop"),
                    _note(144, 192, None, "a-rest"),
                ],
            ),
            ScoreVoice(
                voice_id="voice-b",
                events=[
                    _note(0, 48, None, "b-rest"),
                    _note(48, 96, 67, "b0"),
                    _note(96, 144, None, "b-rest-2"),
                    _note(144, 192, 64, "b1"),
                ],
            ),
        ],
        total_ticks=192,
    )
    composed, report = _compose(
        score,
        [
            _alignment(0, 69, 0, 48, "a0"),
            _alignment(1, 69, 48, 96, "a1"),
            _alignment(2, 69, 96, 144, "a2"),
            _alignment(3, 67, 48, 96, "b0"),
            _alignment(4, 64, 144, 192, "b1"),
        ],
        {0, 1, 2, 3, 4},
    )

    # voice-b's first note overlaps the held voice-a tie, so it needs a
    # second lane; its later note can reuse the first lane after the tie.
    assert report["lane_counts"]["melody"] == 2
    assert Counter(_score_note_intervals(composed)) == Counter(_score_note_intervals(score))
    assert score_to_jianpu(composed)


def test_melody_lanes_share_non_overlapping_source_voice_fragments() -> None:
    score = _score(
        [
            ScoreVoice(
                voice_id="voice-a",
                events=[_note(0, 48, 69, "a0"), _note(48, 144, None, "a-rest")],
            ),
            ScoreVoice(
                voice_id="voice-b",
                events=[
                    _note(0, 48, None, "b-rest"),
                    _note(48, 96, 67, "b0"),
                    _note(96, 144, 64, "b1"),
                ],
            ),
        ],
        total_ticks=144,
    )
    composed, report = _compose(
        score,
        [
            _alignment(0, 69, 0, 48, "a0"),
            _alignment(1, 67, 48, 96, "b0"),
            _alignment(2, 64, 96, 144, "b1"),
        ],
        {0, 1, 2},
    )

    assert report["lane_counts"]["melody"] == 1
    assert Counter(_score_note_intervals(composed)) == Counter(_score_note_intervals(score))
    assert score_to_jianpu(composed)
