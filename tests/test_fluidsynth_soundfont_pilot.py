from __future__ import annotations

import wave
from pathlib import Path

from scripts.pilot_fluidsynth_soundfont import (
    SAMPLE_RATE,
    compare_event_dump,
    parse_event_dump,
    trim_wave,
)


def test_parse_event_dump_keeps_noteoff_when_startup_text_is_spliced() -> None:
    dump = parse_event_dump(
        "event_post_noteon 0 60 80\n"
        "event_post_noteoff 0 60FluidSynth runtime version 2.6.0\n"
    )

    assert dump["note_on_count"] == 1
    assert dump["note_off_count"] == 1
    assert dump["note_off_pitches"] == {60: 1}
    assert dump["missing_velocity_count"] == 1


def test_event_comparison_requires_pitch_multiset_and_both_edges() -> None:
    source = [{"pitch": 60}, {"pitch": 64}]
    complete = parse_event_dump(
        "event_post_noteon 0 60 80\n"
        "event_post_noteoff 0 60 0\n"
        "event_post_noteon 0 64 80\n"
        "event_post_noteoff 0 64 0\n"
    )
    missing = parse_event_dump(
        "event_post_noteon 0 60 80\n"
        "event_post_noteoff 0 60 0\n"
        "event_post_noteon 0 64 80\n"
    )

    assert compare_event_dump(source, complete)["event_complete"] is True
    assert compare_event_dump(source, missing)["event_complete"] is False


def test_trim_wave_uses_exact_fixed_frame_boundary(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    destination = tmp_path / "trimmed.wav"
    with wave.open(str(source), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(b"\x00\x00\x00\x00" * 100)

    result = trim_wave(source, destination, 37)

    assert result["source_frames"] == 100
    assert result["target_frames"] == 37
    assert result["final_frames"] == 37
    with wave.open(str(destination), "rb") as reader:
        assert reader.getframerate() == SAMPLE_RATE
        assert reader.getnchannels() == 2
        assert reader.getsampwidth() == 2
