from __future__ import annotations

import json
import shutil
from pathlib import Path

import mido
import pytest

from backend.jianpu_score import high_accuracy_service as service_module
from backend.jianpu_score.domain import MusicAnalysis, NoteEvent, Score
from backend.jianpu_score.high_accuracy import (
    resolve_musescore,
    resolve_notation_python,
)
from backend.jianpu_score.musescore_import import MuseScoreImportError, MusicXMLArtifact
from backend.jianpu_score.musicxml_standardize import MusicXMLStandardizationError
from backend.jianpu_score.render import JIANPU, LILYPOND

ROOT = Path(__file__).resolve().parents[1]
STAGE56_INPUT = ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json"
STAGE56_SCORE = ROOT / "fixtures" / "high_accuracy" / "stage56" / "stage56.score.json"
STAGE56_MUSICXML = ROOT / "fixtures" / "high_accuracy" / "stage56" / "stage56.musicxml"
PROFILE = ROOT / "tools" / "musescore-4.7.4" / "midi_import_options.xml"
EXTERNAL_READY = (
    resolve_musescore() is not None
    and resolve_notation_python().is_file()
    and PROFILE.is_file()
    and JIANPU.is_file()
    and LILYPOND.is_file()
)


def _stage56_inputs() -> tuple[tuple[NoteEvent, ...], MusicAnalysis]:
    payload = json.loads(STAGE56_INPUT.read_text(encoding="utf-8"))
    bpm = float(payload["source"]["tempo_bpm"])
    source_ticks = int(payload["source"]["ticks_per_quarter"])
    beat_times = [index * 60.0 / bpm for index in range(7)]
    analysis = MusicAnalysis(
        sample_rate=44_100,
        duration_sec=float(payload["source"]["duration_ticks"]) / source_ticks * 60.0 / bpm + 0.25,
        bpm=bpm,
        key=str(payload["source"]["key"]),
        time_signature=str(payload["source"]["time_signature"]),
        beat_times=beat_times,
        metadata={
            "beat_source": "beatnet",
            "beat_engine": "beatnet",
            "beatnet_version": "1.1.3",
            "beat_grid": {
                "beats": [
                    {"index": index, "time_sec": value, "downbeat": index % 3 == 0}
                    for index, value in enumerate(beat_times)
                ],
                "mapping": {"score_origin": {"downbeat_index": 0, "downbeat_sec": 0.0}},
            },
        },
    )
    events = tuple(
        NoteEvent(
            start_sec=float(note["start_tick"]) / source_ticks * 60.0 / bpm,
            end_sec=(int(note["start_tick"]) + int(note["duration_tick"])) / source_ticks * 60.0 / bpm,
            midi=int(note["midi"]),
            voice_id=f"voice-{note['voice']}",
            source="stage56-service-fixture",
        )
        for note in payload["notes"]
    )
    return events, analysis


def _valid_build_kwargs(tmp_path: Path, *, is_drum: bool = False) -> dict:
    events, analysis = _stage56_inputs()
    return {
        "instrument_id": "stage56-drums" if is_drum else "stage56",
        "title": "Stage 56 service fixture",
        "program": 0,
        "is_drum": is_drum,
        "events": events,
        "analysis": analysis,
        "output_dir": tmp_path / ("drums" if is_drum else "stage56"),
        "variant": "source",
    }


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_drum_build_is_midi_only_and_does_not_call_notation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_notation(*_args, **_kwargs):
        raise AssertionError("drum build must not enter notation")

    monkeypatch.setattr(service_module, "convert_performance_midi", fail_notation)
    monkeypatch.setattr(service_module, "standardize_musicxml", fail_notation)
    monkeypatch.setattr(service_module, "render_score", fail_notation)
    monkeypatch.setattr(service_module, "merge_svg_pages", fail_notation)

    kwargs = _valid_build_kwargs(tmp_path, is_drum=True)
    result = service_module.HighAccuracyArtifactService().build(**kwargs)

    assert result.status == "completed"
    assert result.jianpu_status == "midi_only"
    assert result.score is None
    assert result.alignment_report is None
    assert (result.output_dir / "stage56-drums.source.note-events.json").is_file()
    assert (result.output_dir / "stage56-drums.source.performance.mid").is_file()
    assert (result.output_dir / "stage56-drums.source.selected.mid").is_file()
    assert _manifest(result.output_dir)["stages"]["notation"]["status"] == "skipped"
    assert _manifest(result.output_dir)["jianpu_status"] == "midi_only"


def test_invalid_beatnet_writes_failure_manifest_and_preserves_raw_notes(tmp_path: Path) -> None:
    kwargs = _valid_build_kwargs(tmp_path, is_drum=True)
    analysis = kwargs["analysis"].model_copy(deep=True)
    analysis.metadata["beatnet_version"] = "0.0.0"
    kwargs["analysis"] = analysis

    with pytest.raises(service_module.HighAccuracyServiceError) as raised:
        service_module.HighAccuracyArtifactService().build(**kwargs)

    error = raised.value
    destination = Path(kwargs["output_dir"])
    assert error.stage == "validate"
    assert error.instrument_id == "stage56-drums"
    assert error.log_path is not None and error.log_path.is_file()
    assert error.manifest_path is not None and error.manifest_path.is_file()
    assert (destination / "stage56-drums.source.note-events.json").is_file()
    manifest = _manifest(destination)
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "validate"
    assert manifest["artifacts"]


def test_unicode_title_is_kept_in_service_metadata_and_manifest(tmp_path: Path) -> None:
    kwargs = _valid_build_kwargs(tmp_path, is_drum=True)
    kwargs["title"] = "中文歌曲・日本語"
    result = service_module.HighAccuracyArtifactService().build(**kwargs)

    metadata = json.loads((result.output_dir / "stage56-drums.source.performance.metadata.json").read_text(encoding="utf-8"))
    manifest = _manifest(result.output_dir)
    assert metadata["title"] == kwargs["title"]
    assert metadata["midi_track_names"]["conductor"].isascii()
    assert metadata["midi_track_names"]["instrument"].isascii()
    assert manifest["title"] == kwargs["title"]
    assert manifest["title_unicode"] == kwargs["title"]
    assert manifest["midi_track_names"] == metadata["midi_track_names"]


def test_musescore_failure_keeps_performance_prefix_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_musescore(*_args, **_kwargs):
        raise MuseScoreImportError("pinned MuseScore unavailable")

    monkeypatch.setattr(service_module, "convert_performance_midi", fail_musescore)
    kwargs = _valid_build_kwargs(tmp_path)

    with pytest.raises(service_module.HighAccuracyServiceError) as raised:
        service_module.HighAccuracyArtifactService().build(**kwargs)

    destination = Path(kwargs["output_dir"])
    assert raised.value.stage == "musescore_import"
    assert (destination / "stage56.source.note-events.json").is_file()
    assert (destination / "stage56.source.performance.mid").is_file()
    assert not list(destination.glob("*.score.json"))
    assert _manifest(destination)["failure"]["stage"] == "musescore_import"


def test_musicxml_standardization_failure_keeps_musicxml_and_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_convert(midi_path, musicxml_path, **_kwargs):
        shutil.copyfile(STAGE56_MUSICXML, musicxml_path)
        return MusicXMLArtifact(Path(midi_path), Path(musicxml_path), "stage56", ("fake-musescore",))

    def fail_standardize(*_args, **_kwargs):
        raise MusicXMLStandardizationError("notation worker failed")

    monkeypatch.setattr(service_module, "convert_performance_midi", fake_convert)
    monkeypatch.setattr(service_module, "standardize_musicxml", fail_standardize)
    kwargs = _valid_build_kwargs(tmp_path)

    with pytest.raises(service_module.HighAccuracyServiceError) as raised:
        service_module.HighAccuracyArtifactService().build(**kwargs)

    destination = Path(kwargs["output_dir"])
    assert raised.value.stage == "musicxml_standardize"
    assert (destination / "stage56.source.notated.musicxml").is_file()
    assert not list(destination.glob("*.score.json"))
    assert _manifest(destination)["failure"]["stage"] == "musicxml_standardize"


def test_render_failure_keeps_score_and_alignment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_convert(midi_path, musicxml_path, **_kwargs):
        shutil.copyfile(STAGE56_MUSICXML, musicxml_path)
        return MusicXMLArtifact(Path(midi_path), Path(musicxml_path), "stage56", ("fake-musescore",))

    score = Score.model_validate(json.loads(STAGE56_SCORE.read_text(encoding="utf-8")))
    report = {"source_note_count": 12, "source_to_score": []}

    monkeypatch.setattr(service_module, "convert_performance_midi", fake_convert)
    monkeypatch.setattr(service_module, "standardize_musicxml", lambda *_args, **_kwargs: (score, report))
    monkeypatch.setattr(service_module, "render_score", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("renderer down")))
    kwargs = _valid_build_kwargs(tmp_path)

    with pytest.raises(service_module.HighAccuracyServiceError) as raised:
        service_module.HighAccuracyArtifactService().build(**kwargs)

    destination = Path(kwargs["output_dir"])
    assert raised.value.stage == "render"
    assert (destination / "stage56.score.json").is_file()
    assert (destination / "stage56.alignment_report.json").is_file()
    assert _manifest(destination)["failure"]["stage"] == "render"


def test_output_overwrite_is_explicit_and_removes_stale_files(tmp_path: Path) -> None:
    kwargs = _valid_build_kwargs(tmp_path, is_drum=True)
    result = service_module.HighAccuracyArtifactService().build(**kwargs)
    stale = result.output_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        service_module.HighAccuracyArtifactService().build(**kwargs)

    kwargs["overwrite"] = True
    replaced = service_module.HighAccuracyArtifactService().build(**kwargs)
    assert replaced.status == "completed"
    assert not stale.exists()
    assert replaced.output_dir == result.output_dir
    assert len(list(replaced.output_dir.glob("*.manifest.json"))) == 0


def test_overwrite_refuses_unowned_directory_and_preserves_sentinel(tmp_path: Path) -> None:
    kwargs = _valid_build_kwargs(tmp_path, is_drum=True)
    destination = Path(kwargs["output_dir"])
    destination.mkdir(parents=True)
    sentinel = destination / "do-not-delete.txt"
    sentinel.write_text("keep", encoding="utf-8")
    kwargs["overwrite"] = True

    with pytest.raises(ValueError, match="unowned output directory"):
        service_module.HighAccuracyArtifactService().build(**kwargs)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (destination / "manifest.json").exists()


def test_overwrite_refuses_manifest_for_another_instrument(tmp_path: Path) -> None:
    kwargs = _valid_build_kwargs(tmp_path, is_drum=True)
    destination = Path(kwargs["output_dir"])
    destination.mkdir(parents=True)
    sentinel = destination / "do-not-delete.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": service_module.SERVICE_SCHEMA_VERSION,
                "instrument_id": "another-instrument",
                "variant": "source",
            }
        ),
        encoding="utf-8",
    )
    kwargs["overwrite"] = True

    with pytest.raises(ValueError, match="does not match"):
        service_module.HighAccuracyArtifactService().build(**kwargs)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (destination / "manifest.json").is_file()


@pytest.mark.skipif(not EXTERNAL_READY, reason="pinned notation toolchain is unavailable")
def test_real_stage56_service_produces_complete_bundle_and_verifies_final_midi(tmp_path: Path) -> None:
    kwargs = _valid_build_kwargs(tmp_path)
    result = service_module.HighAccuracyArtifactService().build(**kwargs)
    destination = result.output_dir

    required = [
        destination / "stage56.source.note-events.json",
        destination / "stage56.source.performance.mid",
        destination / "stage56.source.performance.metadata.json",
        destination / "stage56.source.notated.musicxml",
        destination / "stage56.score.json",
        destination / "stage56.alignment_report.json",
        destination / "stage56.score.jly",
        destination / "stage56.score.ly",
        destination / "stage56.score.svg",
        destination / "stage56.score.long.svg",
        destination / "stage56.score.mid",
        destination / "manifest.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    assert result.score is not None
    assert result.alignment_report is not None
    assert result.alignment_report["source_note_count"] == 12
    assert len(result.alignment_report["source_to_score"]) == 12
    assert all(item.get("reason") for item in result.alignment_report["source_to_score"])
    assert result.performance_metadata["note_count"] == 12
    assert (destination / "stage56.source.performance.mid").name != (destination / "stage56.score.mid").name

    score_midi = mido.MidiFile(destination / "stage56.score.mid")
    score_pitches = service_module._score_pitch_set(result.score)
    midi_pitches = {
        message.note
        for track in score_midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }
    assert midi_pitches == score_pitches
    assert result.alignment_report["source_note_count"] == len(
        json.loads((destination / "stage56.source.note-events.json").read_text(encoding="utf-8"))["events"]
    )

    manifest = _manifest(destination)
    assert manifest["status"] == "completed"
    assert manifest["notation_engine"] == "musescore-midi-import"
    assert manifest["beat_engine"] == "beatnet"
    assert manifest["beatnet_version"] == "1.1.3"
    assert manifest["musescore_version"] == "4.7.4"
    assert manifest["score_ticks_per_quarter"] == 48
    verification = manifest["stages"]["render"]["midi_verification"]
    assert verification["expected_note_intervals"] == verification["actual_note_intervals"]
    assert verification["expected_track_end_ticks"] == 2304
    by_path = {item["relative_path"]: item for item in manifest["artifacts"]}
    for artifact_path in required:
        if artifact_path.name == "manifest.json":
            continue
        relative = artifact_path.relative_to(destination).as_posix()
        assert relative in by_path
        assert by_path[relative]["bytes"] == artifact_path.stat().st_size
