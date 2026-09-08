from __future__ import annotations

from pathlib import Path

import mido
import pytest

from backend.jianpu_score.musescore_import import (
    MUSESCORE_ORIGIN_SENTINEL_CHANNEL,
    MUSESCORE_ORIGIN_SENTINEL_NAME,
    MUSESCORE_ORIGIN_SENTINEL_PITCH,
    MuseScoreImportError,
    _prepare_origin_sentinel,
    _strip_origin_sentinel_parts,
)


def _minimal_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="source", time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=480))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(path)


def test_origin_sentinel_copy_does_not_mutate_source(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _minimal_midi(source)
    before = source.read_bytes()

    sentinel, details = _prepare_origin_sentinel(source, tmp_path)
    try:
        assert source.read_bytes() == before
        midi = mido.MidiFile(sentinel)
        assert len(midi.tracks) == 2
        assert details["name"] == MUSESCORE_ORIGIN_SENTINEL_NAME
        assert details["channel"] == MUSESCORE_ORIGIN_SENTINEL_CHANNEL + 1
        messages = list(midi.tracks[-1])
        assert any(message.type == "track_name" and message.name == MUSESCORE_ORIGIN_SENTINEL_NAME for message in messages)
        assert any(
            message.type == "note_on"
            and message.note == MUSESCORE_ORIGIN_SENTINEL_PITCH
            and message.channel == MUSESCORE_ORIGIN_SENTINEL_CHANNEL
            for message in messages
        )
    finally:
        sentinel.unlink(missing_ok=True)


def test_origin_sentinel_musicxml_part_is_removed_and_audited(tmp_path: Path) -> None:
    musicxml = tmp_path / "source.musicxml"
    musicxml.write_text(
        "<score-partwise version='3.1'><part-list>"
        "<score-part id='P1'><part-name>source</part-name></score-part>"
        f"<score-part id='P2'><part-name>Grand Piano, {MUSESCORE_ORIGIN_SENTINEL_NAME}</part-name></score-part>"
        "</part-list><part id='P1'/><part id='P2'/></score-partwise>",
        encoding="utf-8",
    )

    details = _strip_origin_sentinel_parts(musicxml)

    assert details["removed_part_ids"] == ["P2"]
    assert details["remaining_part_ids"] == ["P1"]
    text = musicxml.read_text(encoding="utf-8")
    assert MUSESCORE_ORIGIN_SENTINEL_NAME not in text
    assert "id=\"P1\"" in text or "id='P1'" in text


def test_origin_sentinel_cleanup_rejects_untraceable_musicxml(tmp_path: Path) -> None:
    musicxml = tmp_path / "missing-sentinel.musicxml"
    musicxml.write_text(
        "<score-partwise version='3.1'><part-list>"
        "<score-part id='P1'><part-name>source</part-name></score-part>"
        "</part-list><part id='P1'/></score-partwise>",
        encoding="utf-8",
    )

    with pytest.raises(MuseScoreImportError, match="did not preserve the origin sentinel"):
        _strip_origin_sentinel_parts(musicxml)
