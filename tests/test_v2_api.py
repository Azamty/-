from __future__ import annotations

import json
from pathlib import Path

import mido

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent, Score, ScoreNote, ScoreVoice
from backend.job_manager import JobManager
from backend.muscriptor_v2 import stable_track_id


def _instrumental_state(
    manager: JobManager,
    job_id: str,
    notes: list[dict[str, object]],
    tracks: list[dict[str, object]],
    *,
    analysis: dict[str, object] | None = None,
) -> None:
    manager._update(
        job_id,
        status="selection_ready",
        phase="selection_ready",
        progress=1.0,
        v2={
            "stage": "selection_ready",
            "source_kind": "instrumental",
            "route": {"engine": "muscriptor", "use_demucs": False},
            "tracks": tracks,
            "notes": notes,
            "analysis": analysis,
            "selection_revision": 0,
            "selection": None,
            "selection_history": [],
            "score_refusal": None,
        },
    )


def test_v2_selection_is_a_persistent_revision_snapshot(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="fixture.wav", source_kind="instrumental", title="fixture")
    input_path.write_bytes(b"fixture")
    guitar_id = stable_track_id("acoustic_guitar", 24, False)
    drum_id = stable_track_id("drums", 128, True)
    tracks = [
        {
            "track_id": guitar_id,
            "instrument_group": "acoustic_guitar",
            "program": 24,
            "is_drum": False,
            "label_zh": "原声吉他",
            "preview_available": True,
        },
        {
            "track_id": drum_id,
            "instrument_group": "drums",
            "program": 128,
            "is_drum": True,
            "label_zh": "鼓组",
            "preview_available": True,
        },
    ]
    notes = [
        {"instrument_group": "acoustic_guitar", "program": 24, "is_drum": False, "pitch": 60, "start_sec": 0.13, "end_sec": 0.61, "velocity": None},
        {"instrument_group": "drums", "program": 128, "is_drum": True, "pitch": 36, "start_sec": 0.25, "end_sec": 0.35, "velocity": None},
    ]
    _instrumental_state(manager, job_id, notes, tracks)

    manager.select_v2(
        job_id,
        [guitar_id, drum_id],
        merge_main_melody=True,
        bpm_override=96,
        key_override="Am",
        time_signature_override="6/8",
    )

    persisted = json.loads((tmp_path / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "queued"
    assert persisted["v2"]["selection"]["revision"] == 1
    assert persisted["v2"]["selection"]["selected_track_ids"] == [guitar_id, drum_id]
    assert persisted["v2"]["selection"]["merge_main_melody"] is True
    assert persisted["v2"]["selection"]["bpm_override"] == 96
    assert persisted["v2"]["selection"]["key_override"] == "Am"
    assert persisted["v2"]["selection"]["time_signature_override"] == "6/8"


def test_v2_selection_uses_music_analysis_suggestion_when_unmodified(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="fixture.wav", source_kind="instrumental", title="fixture")
    input_path.write_bytes(b"fixture")
    guitar_id = stable_track_id("acoustic_guitar", 24, False)
    tracks = [{
        "track_id": guitar_id,
        "instrument_group": "acoustic_guitar",
        "program": 24,
        "is_drum": False,
        "label_zh": "原声吉他",
        "preview_available": True,
    }]
    notes = [{"instrument_group": "acoustic_guitar", "program": 24, "is_drum": False, "pitch": 60, "start_sec": 0.13, "end_sec": 0.61, "velocity": None}]
    _instrumental_state(
        manager,
        job_id,
        notes,
        tracks,
        analysis={"bpm": 96, "key": "Am", "time_signature": "6/8"},
    )

    manager.select_v2(job_id, [guitar_id])

    selection = manager._read(job_id)["v2"]["selection"]
    assert selection["bpm_override"] == 96
    assert selection["key_override"] == "Am"
    assert selection["time_signature_override"] == "6/8"


def test_v2_drum_only_export_keeps_midi_and_refuses_jianpu(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="fixture.wav", source_kind="instrumental", title="fixture")
    input_path.write_bytes(b"fixture")
    drum_id = stable_track_id("drums", 128, True)
    track = {
        "track_id": drum_id,
        "instrument_group": "drums",
        "program": 128,
        "is_drum": True,
        "label_zh": "鼓组",
        "preview_available": True,
        "note_count": 1,
    }
    notes = [{"instrument_group": "drums", "program": 128, "is_drum": True, "pitch": 36, "start_sec": 0.13, "end_sec": 0.23, "velocity": None}]
    _instrumental_state(manager, job_id, notes, [track])
    manager.select_v2(job_id, [drum_id])
    state = manager._read(job_id)
    state["status"] = "running"
    state["phase"] = "rendering"
    manager._write(state)

    manager.v2._run_instrumental_export(job_id)

    result = manager._read(job_id)
    assert result["status"] == "completed"
    assert result["v2"]["score_refusal"]["code"] == "no_pitched_tracks"
    artifact_ids = {item["artifact_id"] for item in result["artifacts"]}
    assert "v2-selection-r1-midi" in artifact_ids
    assert not any("instrument_score_svg" == item["kind"] for item in result["artifacts"])
    midi_path, _ = manager.artifact_path(job_id, "v2-selection-r1-midi")
    midi = mido.MidiFile(midi_path)
    assert any(message.type == "note_on" and message.channel == 9 for track in midi.tracks for message in track)

    manager.select_v2(job_id, [])
    second = manager._read(job_id)
    second["status"] = "running"
    second["phase"] = "rendering"
    manager._write(second)
    manager.v2._run_instrumental_export(job_id)
    revisions = {item["artifact_id"] for item in manager._read(job_id)["artifacts"]}
    assert "v2-selection-r1-midi" in revisions
    assert "v2-selection-r2-midi" in revisions


def test_v2_vocal_route_calls_game_without_demucs(monkeypatch, tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs")
    job_id, input_path = manager.create_v2_job(original_name="voice.wav", source_kind="vocal", title="voice")
    input_path.write_bytes(b"fixture")
    manager._update(job_id, status="running", phase="recognizing", v2={"stage": "recognize", "source_kind": "vocal"})
    calls: dict[str, object] = {}
    analysis = MusicAnalysis(
        sample_rate=16000,
        duration_sec=1.0,
        bpm=120,
        note_events=[NoteEvent(start_sec=0, end_sec=0.5, midi=60, source="game")],
    )
    score = Score(
        title="voice",
        bpm=120,
        total_ticks=12,
        voices=[ScoreVoice(voice_id="voice-0", events=[ScoreNote(start_tick=0, duration_tick=12, midi=60)])],
    )

    def fake_pipeline(*_args, **kwargs):
        calls.update(kwargs)
        return analysis, score, object()

    monkeypatch.setattr("backend.v2_job_manager.run_pipeline", fake_pipeline)
    monkeypatch.setattr(manager, "_package_artifacts", lambda *_args: [])
    manager.v2._run_vocal(job_id)

    result = manager._read(job_id)
    assert calls["engine"] == "game"
    assert calls["voice_mode"] == "monophonic"
    assert calls["source_kind"] == "mixed"
    assert calls["separate"] is False
    assert calls["language"] == "mixed"
    assert calls["title"] == "voice"
    assert result["summary"]["source_kind"] == "vocal"
    assert result["summary"]["route"] == {"engine": "game", "use_demucs": False}
