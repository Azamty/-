from backend.muscriptor_v2 import (
    DRUMS,
    MuscriptorNote,
    make_transcription_plan,
    partition_notes,
    route_for_source,
)


def test_v2_routes_without_demucs() -> None:
    assert route_for_source("instrumental") == ("muscriptor", False)
    assert route_for_source("vocal") == ("game", False)


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
