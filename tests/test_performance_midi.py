from __future__ import annotations

from io import BytesIO
import json

import mido
import pytest

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.performance_midi import (
    PERFORMANCE_TICKS_PER_QUARTER,
    PerformanceTrack,
    build_performance_midi,
    write_performance_midi,
    write_performance_midi_bundle,
)


def _analysis(
    *,
    beat_times: list[float],
    key: str = "C",
    time_signature: str = "4/4",
    downbeat_index: int = 0,
    manual_scale: float = 1.0,
    manual_bpm: bool = False,
    manual_meter: bool = False,
) -> MusicAnalysis:
    beats = [
        {
            "index": index,
            "time_sec": value,
            "downbeat": index == downbeat_index,
        }
        for index, value in enumerate(beat_times)
    ]
    return MusicAnalysis(
        sample_rate=22050,
        duration_sec=max(beat_times[-1], 1.0) + 0.25,
        bpm=120.0 * manual_scale,
        key=key,
        time_signature=time_signature,
        beat_times=beat_times,
        metadata={
            "beat_source": "beatnet",
            "beat_engine": "beatnet",
            "beatnet_version": "1.1.3",
            "manual_bpm_override": manual_bpm,
            "manual_time_signature_override": manual_meter,
            "beat_grid": {
                "beats": beats,
                "mapping": {
                    "beat_times": beat_times,
                    "manual_bpm_scale": manual_scale,
                    "score_origin": {
                        "downbeat_index": downbeat_index,
                        "downbeat_sec": beat_times[downbeat_index],
                    },
                },
            },
        },
    )


def _messages(midi_bytes: bytes) -> tuple[mido.MidiFile, list[mido.Message | mido.MetaMessage]]:
    midi = mido.MidiFile(file=BytesIO(midi_bytes))
    return midi, list(mido.merge_tracks(midi.tracks))


def _absolute_messages(track: mido.MidiTrack) -> list[tuple[int, mido.Message | mido.MetaMessage]]:
    tick = 0
    result: list[tuple[int, mido.Message | mido.MetaMessage]] = []
    for message in track:
        tick += message.time
        result.append((tick, message))
    return result


def _parsed_midi_tick_to_seconds(midi: mido.MidiFile, tick: int) -> float:
    tempo_points = [
        (absolute_tick, message.tempo)
        for absolute_tick, message in _absolute_messages(midi.tracks[0])
        if message.type == "set_tempo"
    ]
    elapsed = 0.0
    previous_tick = 0
    previous_tempo = tempo_points[0][1]
    for point_tick, tempo in tempo_points[1:]:
        if point_tick >= tick:
            break
        elapsed += mido.tick2second(point_tick - previous_tick, midi.ticks_per_beat, previous_tempo)
        previous_tick = point_tick
        previous_tempo = tempo
    elapsed += mido.tick2second(tick - previous_tick, midi.ticks_per_beat, previous_tempo)
    return elapsed


def test_performance_midi_keeps_single_notes_chords_and_overlapping_voices() -> None:
    analysis = _analysis(beat_times=[0.0, 0.5, 1.0, 1.5, 2.0])
    events = [
        NoteEvent(start_sec=0.0, end_sec=0.5, midi=60, velocity=91, voice_id="voice-0"),
        NoteEvent(start_sec=0.0, end_sec=0.5, midi=64, voice_id="voice-1"),
        NoteEvent(start_sec=0.0, end_sec=0.5, midi=67, voice_id="voice-1"),
        NoteEvent(start_sec=0.25, end_sec=0.75, midi=72, voice_id="voice-2"),
    ]

    midi_bytes, metadata = build_performance_midi(
        events,
        analysis,
        instrument_group="acoustic_piano",
        program=0,
        title="Piano performance",
    )
    midi, messages = _messages(midi_bytes)
    assert midi.ticks_per_beat == PERFORMANCE_TICKS_PER_QUARTER
    notes = [message for message in messages if message.type in {"note_on", "note_off"}]
    assert [message.note for message in notes if message.type == "note_on"] == [60, 64, 67, 72]
    assert metadata["note_count"] == 4
    assert [item["start_tick"] for item in metadata["notes"]] == [0, 0, 0, 240]
    assert all(item["duration_ticks"] >= 1 for item in metadata["notes"])


def test_variable_tempo_map_round_trips_note_seconds_with_tick_rounding() -> None:
    analysis = _analysis(beat_times=[0.0, 0.5, 1.25, 1.75, 2.25])
    events = [
        NoteEvent(start_sec=0.13, end_sec=0.61, midi=60),
        NoteEvent(start_sec=0.75, end_sec=1.63, midi=67),
    ]

    midi_bytes, metadata = build_performance_midi(
        events,
        analysis,
        instrument_group="violin",
        program=40,
    )
    midi, messages = _messages(midi_bytes)
    tempos = metadata["tempo_points"]
    assert [item["bpm"] for item in tempos[:4]] == [120.0, 80.0, 120.0, 120.0]
    assert any(message.type == "set_tempo" and message.time > 0 for message in messages)
    origin = float(metadata["score_origin_audio_sec"])
    parsed_note_starts = [
        tick
        for tick, message in _absolute_messages(midi.tracks[1])
        if message.type == "note_on" and message.velocity > 0
    ]
    for source, mapped in zip(events, metadata["notes"]):
        assert origin + float(mapped["playback_start_sec"]) == pytest.approx(source.start_sec, abs=0.006)
        assert origin + float(mapped["playback_end_sec"]) == pytest.approx(source.end_sec, abs=0.006)
    assert [
        origin + _parsed_midi_tick_to_seconds(midi, tick)
        for tick in parsed_note_starts
    ] == pytest.approx([event.start_sec for event in events], abs=0.006)


def test_score_origin_preserves_downbeat_phase_and_records_pickup() -> None:
    analysis = _analysis(
        beat_times=[0.25, 0.75, 1.25, 1.75, 2.25],
        downbeat_index=1,
    )
    midi_bytes, metadata = build_performance_midi(
        [NoteEvent(start_sec=0.75, end_sec=1.0, midi=60)],
        analysis,
        instrument_group="piano",
    )
    assert len(midi_bytes) > 0
    assert metadata["score_origin"]["strategy"] == "first_downbeat"
    assert metadata["score_origin"]["downbeat_score_beat"] == pytest.approx(0.0)
    assert metadata["notes"][0]["start_tick"] == 0
    assert metadata["score_origin_audio_sec"] == pytest.approx(0.75)

    _pickup_bytes, pickup_metadata = build_performance_midi(
        [NoteEvent(start_sec=0.4, end_sec=0.6, midi=55), NoteEvent(start_sec=0.75, end_sec=1.0, midi=60)],
        analysis,
        instrument_group="piano",
    )
    pickup = pickup_metadata["score_origin"]
    assert pickup["strategy"] == "first_downbeat_with_pickup_candidate"
    assert pickup["pickup_candidate"] is True
    assert pickup["downbeat_score_beat"] == pytest.approx(4.0)
    assert pickup["downbeat_score_beat"] % pickup["downbeat_bar_beats"] == pytest.approx(0.0)
    assert pickup_metadata["notes"][1]["start_tick"] == 1920


def test_program_key_meter_and_drum_channel_are_written() -> None:
    analysis = _analysis(
        beat_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        key="F#m",
        time_signature="6/8",
    )
    midi_bytes, metadata = build_performance_midi(
        [NoteEvent(start_sec=0.0, end_sec=0.5, midi=36)],
        analysis,
        instrument_group="drums",
        program=118,
        is_drum=True,
        title="Drum performance",
    )
    midi, messages = _messages(midi_bytes)
    conductor = [message for message in messages if message.type in {"time_signature", "key_signature"}]
    assert next(message for message in conductor if message.type == "time_signature").numerator == 6
    assert next(message for message in conductor if message.type == "time_signature").denominator == 8
    assert next(message for message in conductor if message.type == "key_signature").key == "F#m"
    drum_track = midi.tracks[1]
    assert all(message.channel == 9 for message in drum_track if hasattr(message, "channel"))
    assert not any(message.type == "program_change" for message in drum_track)
    assert metadata["channel"] == 10
    assert metadata["drum_jianpu_policy"] == "midi_only"


def test_manual_bpm_scale_is_reflected_in_performance_tempo_map() -> None:
    analysis = _analysis(
        beat_times=[0.0, 0.5, 1.0, 1.5],
        manual_scale=0.5,
        manual_bpm=True,
        time_signature="3/4",
        manual_meter=True,
    )
    _midi_bytes, metadata = build_performance_midi(
        [NoteEvent(start_sec=0.5, end_sec=1.0, midi=60)],
        analysis,
        instrument_group="piano",
    )
    assert metadata["beat_scale"] == pytest.approx(0.5)
    assert metadata["tempo_points"][0]["bpm"] == pytest.approx(60.0)
    assert metadata["manual_bpm_override"] is True
    assert metadata["manual_time_signature_override"] is True


def test_write_performance_midi_uses_independent_names_and_refuses_accidental_overwrite(tmp_path) -> None:
    analysis = _analysis(beat_times=[0.0, 0.5, 1.0])
    destination = tmp_path / "acoustic_piano.performance.mid"
    artifact = write_performance_midi(
        [NoteEvent(start_sec=0.0, end_sec=0.5, midi=60)],
        analysis,
        destination,
        instrument_group="acoustic_piano",
    )
    assert artifact.midi_path == destination.resolve()
    assert artifact.metadata_path.name == "acoustic_piano.performance.metadata.json"
    saved = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert saved["artifact_kind"] == "performance_midi"
    with pytest.raises(FileExistsError):
        write_performance_midi(
            [NoteEvent(start_sec=0.0, end_sec=0.5, midi=60)],
            analysis,
            destination,
            instrument_group="acoustic_piano",
        )


def test_bundle_writes_one_performance_artifact_per_selected_instrument(tmp_path) -> None:
    analysis = _analysis(beat_times=[0.0, 0.5, 1.0])
    artifacts = write_performance_midi_bundle(
        [
            PerformanceTrack("acoustic_piano", (NoteEvent(start_sec=0.0, end_sec=0.5, midi=60),), program=0),
            PerformanceTrack("violin", (NoteEvent(start_sec=0.0, end_sec=0.5, midi=67),), program=40),
        ],
        analysis,
        tmp_path,
    )
    assert [artifact.midi_path.name for artifact in artifacts] == [
        "acoustic_piano.performance.mid",
        "violin.performance.mid",
    ]
    assert all(artifact.midi_path.is_file() and artifact.metadata_path.is_file() for artifact in artifacts)
