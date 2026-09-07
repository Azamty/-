from __future__ import annotations

import io

import mido
import pytest

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.performance_midi import build_performance_midi
from backend.jianpu_score.vocal_cleanup import (
    VocalCleanupError,
    clean_vocal_events,
)


def _event(
    start: float,
    end: float,
    midi: int,
    *,
    raw_pitch: float | None = None,
    confidence: float | None = 0.8,
    velocity: int | None = 80,
    metadata: dict | None = None,
    voice_id: str = "voice-0",
) -> NoteEvent:
    return NoteEvent(
        start_sec=start,
        end_sec=end,
        midi=midi,
        raw_pitch=raw_pitch,
        confidence=confidence,
        velocity=velocity,
        voice_id=voice_id,
        source="game-test",
        stem_id="vocals",
        metadata=metadata or {},
    )


def test_same_pitch_fragments_merge_with_weighted_fields_and_lineage() -> None:
    raw = [
        _event(0.0, 0.2, 60, raw_pitch=60.1, confidence=0.4, velocity=60, metadata={"tag": "first"}),
        _event(0.205, 0.4, 60, raw_pitch=60.2, confidence=1.0, velocity=100, metadata={"tag": "second"}),
    ]

    result = clean_vocal_events(raw, bpm=120)

    assert isinstance(result.events, tuple)
    assert len(result.events) == 1
    merged = result.events[0]
    assert merged.start_sec == 0.0 and merged.end_sec == pytest.approx(0.4)
    assert merged.confidence == pytest.approx((0.4 * 0.2 + 1.0 * 0.195) / 0.395)
    assert merged.velocity == 80
    assert merged.metadata["tag"] == "first"
    lineage = merged.metadata["vocal_cleanup"]["lineage"]
    assert [item["index"] for item in lineage] == [0, 1]
    assert result.report["merge_count"] == 1
    assert result.report["raw_count"] == 2
    assert result.report["cleaned_count"] == 1
    assert result.report["events"][0]["source_indices"] == [0, 1]


def test_clear_rest_and_rearticulation_are_not_merged() -> None:
    rest_gap = clean_vocal_events(
        [_event(0, 0.2, 60), _event(0.4, 0.6, 60)],
        bpm=120,
    )
    rearticulated = clean_vocal_events(
        [
            _event(0, 0.2, 60, metadata={"rearticulation": True}),
            _event(0.205, 0.4, 60),
        ],
        bpm=120,
    )

    assert len(rest_gap.events) == 2
    assert len(rearticulated.events) == 2
    assert rest_gap.report["merge_count"] == 0
    assert rearticulated.report["merge_count"] == 0


def test_boundary_raw_pitch_evidence_suppresses_short_half_semitone_fragment() -> None:
    raw = [
        _event(0.0, 0.1, 60, raw_pitch=60.48),
        _event(0.1, 0.13, 61, raw_pitch=60.52),
        _event(0.13, 0.23, 60, raw_pitch=60.48),
    ]

    result = clean_vocal_events(raw, bpm=120)

    assert len(result.events) == 1
    assert result.events[0].midi == 60
    assert result.events[0].start_sec == 0.0
    assert result.events[0].end_sec == pytest.approx(0.23)
    assert result.report["vibrato_suppressed_count"] == 1
    action = next(item for item in result.report["actions"] if item["action"] == "vibrato_suppressed")
    assert action["source_indices"] == [1]
    assert action["from_midi"] == 61 and action["to_midi"] == 60


def test_real_chromatic_motion_long_fragment_and_missing_raw_pitch_are_retained() -> None:
    chromatic = clean_vocal_events(
        [
            _event(0.0, 0.1, 60, raw_pitch=60.0),
            _event(0.1, 0.13, 61, raw_pitch=61.0),
            _event(0.13, 0.23, 60, raw_pitch=60.0),
        ],
        bpm=120,
    )
    long_fragment = clean_vocal_events(
        [
            _event(0.0, 0.1, 60, raw_pitch=60.48),
            _event(0.1, 0.2, 61, raw_pitch=60.52),
            _event(0.2, 0.3, 60, raw_pitch=60.48),
        ],
        bpm=120,
    )
    no_raw = clean_vocal_events(
        [
            _event(0.0, 0.1, 60),
            _event(0.1, 0.13, 61),
            _event(0.13, 0.23, 60),
        ],
        bpm=120,
    )

    assert [event.midi for event in chromatic.events] == [60, 61, 60]
    assert [event.midi for event in long_fragment.events] == [60, 61, 60]
    assert [event.midi for event in no_raw.events] == [60, 61, 60]
    assert chromatic.report["vibrato_suppressed_count"] == 0
    assert long_fragment.report["vibrato_suppressed_count"] == 0
    assert no_raw.report["vibrato_suppressed_count"] == 0


def test_small_overlap_is_adjusted_but_large_overlap_fails_with_diagnostic() -> None:
    adjusted = clean_vocal_events(
        [_event(0, 0.1, 60), _event(0.098, 0.2, 62)],
        bpm=120,
    )
    assert adjusted.report["overlap_adjustment_count"] == 1
    assert adjusted.events[0].end_sec == pytest.approx(adjusted.events[1].start_sec)
    assert adjusted.events[0].end_sec > adjusted.events[0].start_sec
    assert adjusted.events[1].end_sec > adjusted.events[1].start_sec

    with pytest.raises(VocalCleanupError, match="exceeding") as error:
        clean_vocal_events([_event(0, 0.4, 60), _event(0.1, 0.2, 62)], bpm=120)
    assert error.value.report["failure"]["reason"] == "large_overlap"


def test_out_of_order_input_is_sorted_and_multiple_voice_ids_are_rejected() -> None:
    result = clean_vocal_events([_event(0.2, 0.3, 62), _event(0.0, 0.1, 60)], bpm=120)
    assert [event.midi for event in result.events] == [60, 62]
    assert result.report["reordered_count"] == 2

    with pytest.raises(VocalCleanupError, match="one voice"):
        clean_vocal_events([_event(0, 0.1, 60), _event(0.1, 0.2, 62, voice_id="voice-1")], bpm=120)


def test_cleanup_is_idempotent_and_accepts_beat_context() -> None:
    raw = [
        _event(0.0, 0.1, 60, raw_pitch=60.48),
        _event(0.1, 0.13, 61, raw_pitch=60.52),
        _event(0.13, 0.23, 60, raw_pitch=60.48),
    ]
    first = clean_vocal_events(raw, beat_context={"bpm": 120, "beat_index": 4})
    second = clean_vocal_events(first.events, bpm=120)

    assert [event.model_dump(mode="json") for event in first.events] == [
        event.model_dump(mode="json") for event in second.events
    ]
    assert first.report["thresholds"]["merge_gap_sec"] == pytest.approx(1 / 32 * 0.5)
    assert second.report["vibrato_suppressed_count"] == 0


def test_empty_input_has_versioned_structured_report() -> None:
    result = clean_vocal_events([], bpm=100)

    assert result.events == ()
    assert result.report["schema_version"] == "1.0"
    assert result.report["raw_count"] == 0
    assert result.report["cleaned_count"] == 0
    assert "thresholds" in result.report


def test_cleaned_events_feed_performance_midi_without_changing_source() -> None:
    raw = [
        _event(0.0, 0.2, 60, raw_pitch=60.1, velocity=70),
        _event(0.205, 0.4, 60, raw_pitch=60.2, velocity=90),
    ]
    raw_snapshot = [event.model_dump(mode="json") for event in raw]
    result = clean_vocal_events(raw, bpm=120)
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=1.0,
        bpm=120,
        key="C",
        time_signature="4/4",
        beat_times=[0.0, 0.5, 1.0],
        metadata={"beat_source": "beatnet"},
    )

    midi_bytes, metadata = build_performance_midi(
        result.events,
        analysis,
        instrument_group="vocals",
        title="cleaned vocals",
    )
    midi = mido.MidiFile(file=io.BytesIO(midi_bytes))

    assert metadata["note_count"] == 1
    assert any(message.type == "note_on" and message.velocity > 0 for track in midi.tracks for message in track)
    assert [event.model_dump(mode="json") for event in raw] == raw_snapshot
