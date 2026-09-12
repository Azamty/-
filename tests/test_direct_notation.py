from collections import Counter

import pytest

from backend.jianpu_score.direct_notation import (
    DirectNotationOptions, KeyChange, build_direct_score, combine_direct_scores,
)
from backend.jianpu_score.domain import MusicAnalysis, NoteEvent, TempoEvent
from backend.jianpu_score.high_accuracy_service import _score_note_intervals, _verify_score_midi
from backend.jianpu_score.quantize import score_to_jianpu
from backend.jianpu_score.render import render_score


def analysis(**kwargs):
    return MusicAnalysis(sample_rate=22050, duration_sec=kwargs.pop("duration_sec", 8),
                         bpm=kwargs.pop("bpm", 120), key="D", metadata={"beat_source": "manual_bpm"}, **kwargs)


def note(p, start, end):
    return NoteEvent(midi=p, start_sec=start, end_sec=end)


def intervals(score):
    return Counter(tuple(v) for v in _score_note_intervals(score))


def test_polyphony_keeps_independent_releases_and_rearticulations():
    source = [note(38, 0, 3), note(62, 0, 1), note(66, 0, 1), note(78, 0, 1),
              note(70, .5, 2), note(74, 1, 1.5), note(74, 1.5, 2)]
    score, report = build_direct_score(source, analysis(), title="Piano")
    assert intervals(score) == Counter((n.midi, round(n.start_sec*96), round(n.end_sec*96)) for n in source)
    assert report["output_note_count"] == len(source)
    assert any(n.chord_pitches == [62, 66] for v in score.voices for n in v.events)
    assert any(n.end_tick == 288 for v in score.voices for n in v.events if n.midi == 38)


def test_duplicate_and_same_key_overlap_are_audited():
    score, report = build_direct_score([note(60, 0, 2), note(60, .01, 1), note(60, 1, 3)], analysis(), title="same key")
    assert intervals(score) == Counter([(60, 0, 96), (60, 96, 288)])
    assert {a["action"] for a in report["actions"]} == {"same_key_grid_duplicate", "same_key_rearticulation"}
    assert report["accounted_source_count"] == 3


def test_vocal_gaps_short_notes_and_collisions_do_not_affect_piano():
    source = [note(60, 0, .3), note(62, .41, .8), note(64, .81, .85), note(65, 1.5, 2)]
    score, report = build_direct_score(source, analysis(), title="voice", mode="vocal")
    assert len(score.voices) == 1
    assert report["output_note_count"] == 3
    assert {a["action"] for a in report["actions"]} == {"short_vocal_event", "vocal_gap"}
    assert any(n.is_rest and n.duration_tick >= 48 for n in score.voices[0].events)
    piano, piano_report = build_direct_score(source, analysis(), title="piano")
    assert piano_report["output_note_count"] == 4


def test_half_speed_changes_notation_but_not_playback_seconds():
    source = [note(60, .5, 1)]
    normal, _ = build_direct_score(source, analysis(), title="normal")
    half, _ = build_direct_score(source, analysis(), title="half", options=DirectNotationOptions(beat_divisor=2))
    assert intervals(normal) == Counter([(60, 48, 96)])
    assert intervals(half) == Counter([(60, 24, 48)])
    assert half.bpm == normal.bpm / 2
    assert half.tempo_events[0].bpm == normal.tempo_events[0].bpm / 2


def test_explicit_key_change_keeps_actual_pitch_and_validates_bars():
    options = DirectNotationOptions(key_changes=[KeyChange(bar=2, key="Eb")])
    score, _ = build_direct_score([note(62, 0, 1), note(63, 2, 3)], analysis(), title="key change", options=options)
    text = score_to_jianpu(score)
    assert "1=D" in text and "1=Eb" in text
    with pytest.raises(ValueError, match="distinct bars"):
        build_direct_score([note(62, 0, 1)], analysis(), title="bad", options=DirectNotationOptions(key_changes=[KeyChange(bar=20, key="C")]))


def test_stems_do_not_merge_and_composition_does_not_reselect_notes():
    source = [note(60, 0, 1).model_copy(update={"stem_id": s}) for s in ("piano", "guitar")]
    score, _ = build_direct_score(source, analysis(), title="two tracks")
    assert intervals(score) == Counter({(60, 0, 96): 2})
    other, _ = build_direct_score([note(43, 0, 2)], analysis(), title="bass")
    combined, _ = combine_direct_scores([("a", "keys", score), ("b", "bass", other)], title="all")
    assert intervals(combined) == intervals(score) + intervals(other)


def test_real_renderer_preserves_chromatic_chords_ties_and_key_changes(tmp_path):
    source = [note(38, 0, 3), note(54, 0, 1), note(58, 0, 1), note(61, 0, 1),
              note(74, .5, 2.5), note(63, 3, 4)]
    score, _ = build_direct_score(source, analysis(duration_sec=4), title="Direct regression",
        options=DirectNotationOptions(key_changes=[KeyChange(bar=2, key="Eb")] ))
    score.tempo_events = [TempoEvent(start_tick=0, bpm=120), TempoEvent(start_tick=192, bpm=110)]
    assert "4=110" not in score_to_jianpu(score)
    rendered = render_score(score, tmp_path)
    _verify_score_midi(score, rendered.midi_path)
    assert rendered.svg_paths
    assert rendered.pdf_path
    from pathlib import Path
    assert Path(rendered.pdf_path).read_bytes().startswith(b"%PDF-")


def test_direct_labels_keep_part_names_but_shorten_continuation_labels():
    from backend.jianpu_score.render import _melody_harmony_role_label
    assert _melody_harmony_role_label("钢琴 和声变音 chord lane 1") == ("钢琴 和声变音分1", "和声变音分1")
