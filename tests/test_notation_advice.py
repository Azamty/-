import json

from backend.jianpu_score.domain import MusicAnalysis
from backend.jianpu_score.notation_advice import recommend_notation


def analysis(bpm=176):
    return MusicAnalysis(sample_rate=22050, duration_sec=20, bpm=bpm, key="C",
                         metadata={"beat_grid": {"tempo": {"detected_bpm": bpm}}})


def test_dense_subdivision_does_not_force_double_tempo():
    notes = [{"pitch": 60+(i%3)*4, "start_sec": i*60/176/2, "end_sec": (i+1)*60/176/2} for i in range(80)]
    advice = recommend_notation(analysis(), notes)
    assert advice["bpm"] == 88
    json.dumps(advice)


def test_short_notes_can_outweigh_slow_tempo_prior():
    notes = [{"pitch": 60, "start_sec": i*.05, "end_sec": i*.05+.04} for i in range(40)]
    assert recommend_notation(analysis(120), notes)["bpm"] > 60


def test_key_evidence_transposes_with_notes_and_drums_are_ignored():
    notes = [{"pitch": p, "start_sec": i, "end_sec": i+.8} for i,p in enumerate([60,64,67,60]*5)]
    original = recommend_notation(analysis(100), notes)
    moved = recommend_notation(analysis(100), [{**n, "pitch": n["pitch"]+2} for n in notes])
    assert original["key"] == "C"
    assert moved["key"] == "D"
    assert recommend_notation(analysis(100), notes+[{"pitch": 42,"start_sec":0,"end_sec":20,"is_drum":True}])["key"] == "C"


def test_no_notes_retains_key_and_produces_json():
    result = recommend_notation(analysis(), [])
    assert result["key"] == "C"
    json.dumps(result)
