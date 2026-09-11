from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import mido
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_asap_benchmark", ROOT / "scripts" / "prepare_asap_benchmark.py")
assert SPEC and SPEC.loader
asap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = asap
SPEC.loader.exec_module(asap)


def _annotation() -> dict[str, object]:
    beats = [index * 0.5 for index in range(17)]
    downbeats = [index * 2.0 for index in range(5)]
    return {
        "performance_beats": beats,
        "performance_downbeats": downbeats,
        "performance_beats_type": {str(value): "db" if value in downbeats else "b" for value in beats},
        "perf_time_signatures": {"0.0": ["4/4", 4]},
        "perf_key_signatures": {},
        "midi_score_beats": beats,
        "midi_score_downbeats": downbeats,
        "midi_score_beats_type": {},
        "midi_score_time_signatures": {"0.0": ["4/4", 4]},
        "midi_score_key_signatures": {},
        "downbeats_score_map": [0, 1, 2, 3, 4],
        "score_and_performance_aligned": True,
    }


def _midi_bytes(path: Path, *, tempo: int = 500_000) -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    midi.tracks.append(conductor)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480 * 16))
    midi.tracks.append(track)
    midi.save(path)
    return path.read_bytes()


def test_window_requires_complete_consecutive_score_mapped_measures() -> None:
    annotation = _annotation()
    window = asap._find_window(annotation, "4/4")
    assert window is not None
    assert window.performance_downbeats == (0.0, 2.0, 4.0, 6.0, 8.0)
    assert window.score_measure_indices == (0, 1, 2, 3, 4)
    annotation["downbeats_score_map"] = [0, 1, 3, 4, 5]
    assert asap._find_window(annotation, "4/4") is None


def test_window_rejects_br_beat() -> None:
    annotation = _annotation()
    annotation["performance_beats_type"]["1.0"] = "bR"
    assert asap._find_window(annotation, "4/4") is None


def test_prepare_verifies_direct_annotations_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    midi_path = tmp_path / "source.mid"
    midi = _midi_bytes(midi_path)
    annotation = _annotation()
    source_path = "Test/Piece/Player.mid"
    annotation_path = "Test/Piece/Player_annotations.txt"
    rows = [
        f"{value}\t{value}\t{'db,4/4' if value in annotation['performance_downbeats'] else 'b'}"
        for value in annotation["performance_beats"]
    ]
    archive = tmp_path / "asap.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(asap.ANNOTATIONS_MEMBER, json.dumps({source_path: annotation}))
        bundle.writestr(asap.LICENSE_MEMBER, "fixture license")
        bundle.writestr(asap.ARCHIVE_PREFIX + source_path, midi)
        bundle.writestr(asap.ARCHIVE_PREFIX + annotation_path, "\n".join(rows))
        bundle.writestr(asap.ARCHIVE_PREFIX + "Test/Piece/midi_score.mid", midi)
    monkeypatch.setattr(asap, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(asap, "ARCHIVE_SHA256", asap._sha256(archive).upper())

    def fake_render(source: Path, audio: Path, *, manifest_path: Path, overwrite: bool) -> dict[str, object]:
        audio.write_bytes(b"RIFF fixture")
        payload = {"verification": {"source_event_complete": True}}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(asap, "render_midi", fake_render)
    result = asap.prepare_archive(
        archive,
        output_root=tmp_path / "output",
        selection_plan=(("Test", "4/4"),),
    )
    record = result["cases"][0]
    assert record["score_and_performance_aligned"] is True
    assert record["complete_measure_count"] == 4
    assert record["source_score_measure_indices"] == [0, 1, 2, 3, 4]
    assert record["source_score"]["path"].endswith("sources/asap-v11-01/midi_score.mid")
    assert record["reference_midi"]["path"].endswith("clips/asap-v11-01.reference.mid")
    assert record["score_crop"]["time_basis"] == "score_quarters"
    assert record["score_crop"]["source_start_tick"] == 0.0
    assert record["score_crop"]["source_end_tick"] == 7680.0
    assert "score ticks" in record["score_crop"]["timing_policy"]
    grid = json.loads((tmp_path / "output" / record["beat_annotation"]["path"]).read_text(encoding="utf-8"))
    assert grid["source"] == "asap_v1.1_direct_performance_annotation"
    assert grid["beat_grid"]["annotation_policy"].startswith("direct crop")
    assert grid["beat_grid"]["source_annotation_sha256"] == record["source_annotation"]["sha256"]
    assert (tmp_path / "output" / "sources" / "LICENSE.md").read_text(encoding="utf-8") == "fixture license"


def test_score_crop_preserves_score_quarters_across_source_tempo(tmp_path: Path) -> None:
    source = tmp_path / "score.mid"
    _midi_bytes(source, tempo=666_667)
    destination = tmp_path / "score.reference.mid"
    crop = asap._crop_midi(
        source,
        destination,
        start_sec=0.0,
        end_sec=8.0,
        meter="3/4",
        time_basis="score_quarters",
    )
    assert crop["source_start_tick"] == 0.0
    assert crop["source_end_tick"] == 5760.0
    output = mido.MidiFile(destination)
    note_ticks = []
    tick = 0
    for message in output.tracks[-1]:
        tick += message.time
        if message.type in {"note_on", "note_off"}:
            note_ticks.append((message.type, tick))
    assert note_ticks == [("note_on", 0), ("note_off", 5760)]


def test_archive_member_traversal_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    monkeypatch.setattr(asap, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(asap, "ARCHIVE_SHA256", asap._sha256(archive).upper())
    with pytest.raises(ValueError, match="unsafe archive member"):
        asap.prepare_archive(archive, output_root=tmp_path / "output", selection_plan=())


def test_prepare_rejects_wrong_archive_hash(tmp_path: Path) -> None:
    archive = tmp_path / "wrong.zip"
    archive.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="size or SHA-256 mismatch"):
        asap.prepare_archive(archive, output_root=tmp_path / "output", selection_plan=(("Test", "4/4"),))
