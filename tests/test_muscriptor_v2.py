from backend.muscriptor_v2 import (
    DRUMS,
    MuscriptorNote,
    make_transcription_plan,
    partition_notes,
    route_for_source,
)
from backend.jianpu_score.domain import MusicAnalysis
from backend.v2_job_manager import _analysis_suggestion


def test_v2_routes_demucs_only_for_vocal() -> None:
    assert route_for_source("instrumental") == ("muscriptor", False)
    assert route_for_source("vocal") == ("game", True)


def test_default_selection_keeps_pitched_stems_and_drum_preview() -> None:
    plan = make_transcription_plan(
        "instrumental", ["drums", "acoustic_piano", "violin", "acoustic_piano"]
    )

    assert plan.detected_instruments == ("drums", "acoustic_piano", "violin")
    assert plan.score_instruments == ("acoustic_piano", "violin")
    assert plan.drum_preview_enabled is True
    assert plan.merge_main_melody is False


def test_selection_is_post_decode_and_drums_are_not_score_stems() -> None:
    plan = make_transcription_plan(
        "instrumental",
        ["acoustic_piano", "drums", "violin"],
        selected_instruments=["violin", "drums"],
    )
    notes = [
        MuscriptorNote("acoustic_piano", 60, 0.0, 0.5),
        MuscriptorNote("violin", 67, 0.5, 1.0),
        MuscriptorNote(DRUMS, 36, 0.5, 0.51),
    ]

    partitions = partition_notes(notes, plan)
    assert tuple(partitions) == ("violin", DRUMS)
    assert [note.pitch for note in partitions["violin"]] == [67]
    assert [note.pitch for note in partitions[DRUMS]] == [36]


def test_music_analysis_suggestion_keeps_candidates_and_warnings() -> None:
    analysis = MusicAnalysis(
        sample_rate=22050,
        duration_sec=2.0,
        bpm=96,
        key="Am",
        time_signature="4/4",
        warnings=["自动拍号识别尚未启用，暂按 4/4；生成后请确认"],
        metadata={
            "beat_source": "librosa",
            "key_candidates": ["Am", "C"],
            "bpm_candidates": [96.0, 48.0, 192.0],
            "time_signature_source": "fallback",
            "time_signature_candidates": ["2/4", "3/4", "4/4", "6/8"],
        },
    )

    suggestion = _analysis_suggestion(analysis)
    assert suggestion["bpm"] == 96.0
    assert suggestion["candidates"]["key"] == ["Am", "C"]
    assert suggestion["candidates"]["time_signature"] == ["2/4", "3/4", "4/4", "6/8"]
    assert suggestion["warnings"]
    assert suggestion["sources"]["time_signature"] == "fallback"
