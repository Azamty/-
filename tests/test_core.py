from __future__ import annotations

from pathlib import Path

import mido
import pytest

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent, Score, ScoreNote, ScoreVoice, relative_major_key
from backend.jianpu_score.capabilities import get_capabilities
from backend.jianpu_score.models.adapter import EngineResult, EngineUnavailableError
from backend.jianpu_score.models.game import _language_request
from backend.jianpu_score.pipeline import _engine_plan
from backend.jianpu_score.pipeline import prepare_sources
from backend.jianpu_score.quantize import (
    NoNotesError,
    midi_to_jianpu,
    quantize_events,
    score_to_jianpu,
    select_voice_events,
)
from backend.jianpu_score.render import render_score


def _event(start: float, end: float, midi: int, confidence: float = 0.8) -> NoteEvent:
    return NoteEvent(start_sec=start, end_sec=end, midi=midi, confidence=confidence, source="test")


def test_monophonic_mode_keeps_strongest_overlapping_note() -> None:
    events = [_event(0, 0.75, 60, 0.5), _event(0, 0.75, 64, 0.9), _event(0.75, 1.5, 67, 0.7)]

    selected = select_voice_events(events, mode="monophonic")

    assert [(item.midi, item.voice_id) for item in selected] == [(64, "voice-0"), (67, "voice-0")]


def test_monophonic_dynamic_path_prefers_continuous_melody_over_long_bass() -> None:
    bass = _event(0, 4, 36, 0.95)
    melody = [_event(index * 0.8, index * 0.8 + 0.7, 60 + index, 0.8) for index in range(5)]

    selected = select_voice_events([bass, *melody], mode="monophonic")

    assert [event.midi for event in selected] == [60, 61, 62, 63, 64]


def test_empty_events_raise_explicit_no_notes_error() -> None:
    analysis = MusicAnalysis(sample_rate=22050, duration_sec=1.0, bpm=120)

    with pytest.raises(NoNotesError, match="NoNotes"):
        quantize_events([], analysis)


def test_unknown_key_and_nonfinite_confidence_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        MusicAnalysis(sample_rate=22050, duration_sec=1.0, bpm=120, key="H")
    with pytest.raises(ValueError):
        NoteEvent(start_sec=0, end_sec=1, midi=60, confidence=float("nan"))


def test_polyphonic_mode_preserves_simultaneous_chord_and_layers() -> None:
    events = [_event(0, 0.5, 60), _event(0, 0.5, 64), _event(0, 0.5, 67), _event(0.5, 1, 72)]

    selected = select_voice_events(events, mode="polyphonic")

    assert len(selected) == len(events)
    assert {item.midi for item in selected if item.start_sec == 0} == {60, 64, 67}
    assert {item.voice_id for item in selected} == {"voice-0", "voice-1", "voice-2"}

    score = quantize_events(
        events,
        MusicAnalysis(sample_rate=22050, duration_sec=1.0, bpm=120, key="C", time_signature="4/4"),
        mode="polyphonic",
        title="chord",
    )
    assert len(score.voices) == 3
    assert {voice.events[0].midi for voice in score.voices} == {60, 64, 67}
    assert all(voice.events[-1].end_tick == score.total_ticks for voice in score.voices)


def test_polyphonic_layers_are_numbered_per_stem() -> None:
    events = [
        _event(0, 2, 36),
        _event(0, 0.5, 60),
        _event(2, 2.5, 62),
    ]
    events[0] = events[0].model_copy(update={"stem_id": "bass"})
    events[1] = events[1].model_copy(update={"stem_id": "other"})
    events[2] = events[2].model_copy(update={"stem_id": "other"})

    selected = select_voice_events(events, mode="polyphonic")

    assert [(event.stem_id, event.voice_id) for event in selected] == [
        ("bass", "bass:voice-0"),
        ("other", "other:voice-0"),
        ("other", "other:voice-0"),
    ]


def test_stem_routing_keeps_independent_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_stems = {name: tmp_path / f"{name}.wav" for name in ("vocals", "bass", "drums", "other")}
    monkeypatch.setattr("backend.jianpu_score.pipeline.separate_htdemucs", lambda _audio, _output: fake_stems)

    mono_instrumental = prepare_sources("input.wav", source_kind="instrumental", separate=True, voice_mode="monophonic", output_dir=tmp_path)
    poly_vocal = prepare_sources("input.wav", source_kind="vocal", separate=True, voice_mode="polyphonic", output_dir=tmp_path)

    assert mono_instrumental == {"other": fake_stems["other"]}
    assert set(poly_vocal) == {"vocals", "bass", "other"}


def test_quantize_fills_gaps_and_uses_one_shared_timeline() -> None:
    analysis = MusicAnalysis(sample_rate=22050, duration_sec=2.0, bpm=120, key="C", time_signature="4/4")
    events = [_event(0.5, 1.0, 60), _event(1.5, 2.0, 64)]

    score = quantize_events(events, analysis, mode="monophonic", title="gaps")

    voice = score.voices[0]
    assert score.total_ticks == 48
    assert voice.events[0].is_rest and voice.events[0].duration_tick == 12
    assert [(event.start_tick, event.duration_tick, event.midi) for event in voice.events] == [
        (0, 12, None),
        (12, 12, 60),
        (24, 12, None),
        (36, 12, 64),
    ]
    assert voice.events[-1].end_tick == score.total_ticks


def test_beat_times_drive_offset_and_variable_tempo_mapping() -> None:
    event = _event(0.25, 0.75, 60)
    aligned_analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        beat_times=[0.0, 0.5, 1.0, 1.5],
        metadata={"beat_source": "librosa"},
    )
    aligned_score = quantize_events([event], aligned_analysis, mode="monophonic")
    offset_analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        beat_times=[0.25, 0.75, 1.25, 1.75],
        metadata={"beat_source": "librosa"},
    )
    offset_score = quantize_events([event], offset_analysis, mode="monophonic")
    assert aligned_score.voices[0].events[0].duration_tick == 6
    assert [(item.start_tick, item.duration_tick, item.midi) for item in offset_score.voices[0].events[:2]] == [(0, 6, None), (6, 12, 60)]
    assert offset_score.metadata["beat_shift_beats"] == pytest.approx(0.5)
    assert offset_score.metadata["pickup_beats"] == pytest.approx(0.0)
    assert offset_score.metadata["downbeat_status"] == "undetermined"

    variable_analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        beat_times=[0.0, 0.5, 1.25, 1.75],
        metadata={"beat_source": "librosa"},
    )
    variable_score = quantize_events([_event(0.5, 1.25, 60)], variable_analysis, mode="monophonic")
    assert [(tempo.start_tick, tempo.bpm) for tempo in variable_score.tempo_events] == [
        (0, pytest.approx(120.0)),
        (12, pytest.approx(80.0)),
        (24, pytest.approx(120.0)),
    ]

    manual_analysis = variable_analysis.model_copy(update={"metadata": {"beat_source": "manual_bpm"}})
    manual_score = quantize_events([event], manual_analysis, mode="monophonic")
    assert manual_score.voices[0].events[0].is_rest
    assert manual_score.voices[0].events[0].duration_tick == 6
    assert manual_score.tempo_events[0].bpm == pytest.approx(120.0)


def test_beat_map_preserves_audio_zero_before_a_late_first_beat() -> None:
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        beat_times=[0.25, 0.75, 1.25, 1.75],
        metadata={"beat_source": "librosa"},
    )
    events = [_event(0.0, 0.2, 60), _event(0.25, 0.75, 62)]

    score = quantize_events(events, analysis, mode="monophonic")
    notes = [event for event in score.voices[0].events if event.midi is not None]

    assert [(event.start_tick, event.duration_tick, event.midi) for event in notes] == [(0, 6, 60), (6, 12, 62)]
    assert score.metadata["beat_shift_beats"] == pytest.approx(0.5)
    assert score.metadata["downbeat_status"] == "undetermined"

    variable = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=120,
        beat_times=[0.25, 0.75, 1.5, 1.75],
        metadata={"beat_source": "librosa"},
    )
    variable_score = quantize_events([events[0]], variable, mode="monophonic")
    assert [(event.start_tick, event.bpm) for event in variable_score.tempo_events] == [
        (0, pytest.approx(120.0)),
        (18, pytest.approx(80.0)),
        (30, pytest.approx(240.0)),
    ]


def test_serializer_emits_triplet_group_and_cross_bar_tie() -> None:
    voice_events = [
        ScoreNote(start_tick=0, duration_tick=12, midi=60),
        ScoreNote(start_tick=12, duration_tick=4, midi=62),
        ScoreNote(start_tick=16, duration_tick=4, midi=64),
        ScoreNote(start_tick=20, duration_tick=4, midi=65),
        ScoreNote(start_tick=24, duration_tick=18, midi=None),
        ScoreNote(start_tick=42, duration_tick=12, midi=69),
        ScoreNote(start_tick=54, duration_tick=42, midi=None),
    ]
    score = Score(
        title="notation features",
        bpm=80,
        key="C",
        time_signature="4/4",
        quarter_ticks=12,
        total_ticks=96,
        voices=[ScoreVoice(voice_id="voice-0", events=voice_events)],
    )

    jianpu = score_to_jianpu(score)

    assert "3[" in jianpu
    assert "~" in jianpu
    assert "0" in jianpu
    assert jianpu.count("|") == 2


def test_real_note_events_quantize_to_triplet_and_keep_metadata() -> None:
    events = [
        NoteEvent(start_sec=0, end_sec=1 / 6, midi=60, confidence=None, raw_pitch=60.25, velocity=88, stem_id="vocals"),
        NoteEvent(start_sec=1 / 6, end_sec=2 / 6, midi=62, confidence=None, raw_pitch=62.1, velocity=89, stem_id="vocals"),
        NoteEvent(start_sec=2 / 6, end_sec=3 / 6, midi=64, confidence=None, raw_pitch=64.4, velocity=90, stem_id="vocals"),
    ]
    analysis = MusicAnalysis(sample_rate=22050, duration_sec=1.0, bpm=120, beat_times=[0.0, 0.5, 1.0], metadata={"beat_source": "librosa"})

    score = quantize_events(events, analysis, mode="polyphonic", title="triplet")
    notes = [event for event in score.voices[0].events if event.midi is not None]

    assert [(event.start_tick, event.duration_tick) for event in notes] == [(0, 4), (4, 4), (8, 4)]
    assert score.metadata["triplet_group_count"] == 1
    assert any("3[" in line for line in score_to_jianpu(score).splitlines())
    assert notes[0].confidence is None and notes[0].raw_pitch == pytest.approx(60.25)
    assert notes[0].velocity == 88 and notes[0].stem_id == "vocals"


def test_keys_titles_and_time_signatures_are_renderer_safe() -> None:
    score = Score(
        title="bad\nNextPart\\LPH{unsafe}",
        bpm=80,
        key="Am",
        time_signature="6/8",
        total_ticks=36,
        voices=[ScoreVoice(voice_id="voice-0", events=[ScoreNote(start_tick=0, duration_tick=36, midi=69)])],
    )

    jianpu = score_to_jianpu(score)

    assert "\nNextPart" not in jianpu
    assert jianpu.splitlines()[0] == "title=bad NextPart LPH unsafe"
    assert jianpu.splitlines()[1] == "1=C"


def test_minor_keys_use_relative_major_degrees() -> None:
    assert {key: relative_major_key(key) for key in ("Am", "F#m", "Cm")} == {
        "Am": "C",
        "F#m": "A",
        "Cm": "Eb",
    }
    assert score_to_jianpu(Score(
        title="minor",
        bpm=80,
        key="F#m",
        time_signature="4/4",
        total_ticks=12,
        voices=[ScoreVoice(voice_id="voice-0", events=[ScoreNote(start_tick=0, duration_tick=12, midi=66)])],
    )).splitlines()[1] == "1=A"


def test_same_bar_accidental_scope_preserves_real_midi_pitches(tmp_path: Path) -> None:
    expected = [61, 60, 61, 60]
    score = Score(
        title="accidental scope",
        bpm=80,
        key="C",
        time_signature="4/4",
        quarter_ticks=12,
        total_ticks=48,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[
                    ScoreNote(start_tick=index * 12, duration_tick=12, midi=pitch)
                    for index, pitch in enumerate(expected)
                ],
            )
        ],
    )
    artifacts = render_score(score, tmp_path, basename="accidentals")
    midi = mido.MidiFile(artifacts.midi_path)
    actual = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]
    assert actual == expected


def test_extreme_octaves_keep_real_midi_pitches(tmp_path: Path) -> None:
    expected = [0, 22, 60, 127]
    assert midi_to_jianpu(22, "Db") == "6,,,,"
    score = Score(
        title="extreme octaves",
        bpm=80,
        key="C",
        time_signature="4/4",
        quarter_ticks=12,
        total_ticks=48,
        voices=[
            ScoreVoice(
                voice_id="voice-0",
                events=[
                    ScoreNote(start_tick=index * 12, duration_tick=12, midi=pitch)
                    for index, pitch in enumerate(expected)
                ],
            )
        ],
    )
    artifacts = render_score(score, tmp_path, basename="extreme-octaves")
    jianpu = Path(artifacts.jly_path).read_text(encoding="utf-8")
    assert "1,,,,," in jianpu
    assert "6,,,," in jianpu
    assert "5'''''" in jianpu
    midi = mido.MidiFile(artifacts.midi_path)
    actual = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]
    assert actual == expected


def test_pipeline_persists_score_before_external_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=1.0,
        bpm=120,
        key="C",
        time_signature="4/4",
        beat_times=[0.0, 0.5, 1.0],
        metadata={"beat_source": "manual_bpm"},
    )
    event = _event(0.0, 0.5, 60)

    monkeypatch.setattr("backend.jianpu_score.pipeline._ensure_requested_engine", lambda _engine: None)
    monkeypatch.setattr(
        "backend.jianpu_score.pipeline.analyze_audio",
        lambda _path, **_kwargs: (None, analysis),
    )
    monkeypatch.setattr(
        "backend.jianpu_score.pipeline.prepare_sources",
        lambda _path, **_kwargs: {"mixed": input_path.resolve()},
    )
    monkeypatch.setattr(
        "backend.jianpu_score.pipeline.run_engine",
        lambda *_args, **_kwargs: EngineResult(events=[event], engine="test"),
    )

    def fail_renderer(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("renderer probe")

    monkeypatch.setattr("backend.jianpu_score.pipeline.render_score", fail_renderer)

    from backend.jianpu_score.pipeline import run_pipeline

    with pytest.raises(RuntimeError, match="renderer probe"):
        run_pipeline(input_path, tmp_path / "output", engine="basic-pitch", title="persist")

    assert (tmp_path / "output" / "analysis.json").is_file()
    assert (tmp_path / "output" / "score.json").is_file()
    assert '"input_event_count": 1' in (tmp_path / "output" / "score.json").read_text(encoding="utf-8")


def test_specialist_routes_each_stem_to_the_declared_checkpoint() -> None:
    assert _engine_plan(
        "specialist", source_kind="vocal", voice_mode="monophonic", stem_id="vocals"
    ) == ("game", None)
    assert _engine_plan(
        "specialist", source_kind="instrumental", voice_mode="monophonic", stem_id="other"
    ) == ("tsumugi", "other_v1_5")
    assert _engine_plan(
        "specialist", source_kind="instrumental", voice_mode="polyphonic", stem_id="bass"
    ) == ("tsumugi", "bass_v2")
    assert _engine_plan(
        "specialist", source_kind="vocal", voice_mode="polyphonic", stem_id="vocals"
    ) == ("tsumugi", "vocal_harmony_v1_5")

    with pytest.raises(EngineUnavailableError, match="mixed monophonic"):
        _engine_plan("specialist", source_kind="mixed", voice_mode="monophonic", stem_id="mixed")
    with pytest.raises(EngineUnavailableError, match="GAME"):
        _engine_plan("game", source_kind="instrumental", voice_mode="monophonic", stem_id="other")


def test_capabilities_expose_isolated_specialist_routes() -> None:
    capabilities = get_capabilities()
    assert set(("basic-pitch", "librosa", "game", "tsumugi", "specialist")) <= set(capabilities["engines"])
    assert capabilities["engines"]["specialist"]["routes"]["instrumental/polyphonic"] == "tsumugi bass_v2 + other_v1_5"
    assert capabilities["engines"]["game"]["mixed_language_strategy"].startswith("omit --language")
    assert capabilities["optional"]["chordscope"]["available"] is False


def test_game_language_request_reads_map_and_does_not_guess_mixed() -> None:
    lang_map = {"zh": 41, "ja": 42}
    assert _language_request("zh", lang_map) == ("zh", "weight lang_map.json entry zh=41", 41)
    assert _language_request("ja", lang_map)[2] == 42
    assert _language_request("mixed", lang_map)[0] is None
    with pytest.raises(EngineUnavailableError, match="absent from the selected weight map"):
        _language_request("en", lang_map)
