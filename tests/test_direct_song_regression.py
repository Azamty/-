"""Optional local full-song fixture; audio and recognition caches stay untracked."""
from collections import Counter
import json
from pathlib import Path

import pytest

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent
from backend.jianpu_score.direct_notation import build_direct_score
from backend.jianpu_score.notation_advice import recommend_notation
from backend.v2_job_manager import V2JobService

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "artifacts/review/direct-jianpu/jobs/1dfbe39f-01ce-4791-8a80-127e3c10af28"
REFERENCE = ROOT / "artifacts/pokolu-piano/score-data.json"


@pytest.mark.skipif(not (JOB / "job.json").is_file() or not REFERENCE.is_file(), reason="local song caches unavailable")
def test_automatic_piano_matches_original_pdf_note_intervals():
    state = json.loads((JOB / "job.json").read_text(encoding="utf-8"))
    analysis = MusicAnalysis.model_validate_json((JOB / state["v2"]["analysis_relative"]).read_text(encoding="utf-8"))
    notes = [n for n in state["v2"]["notes"] if n["instrument_group"] == "acoustic_piano"]
    advice = recommend_notation(analysis, notes)
    assert advice["bpm"] == pytest.approx(88.2353)
    assert advice["key"] == "Bm"
    analysis.metadata["notation_engine"] = "direct-jianpu"
    events = [NoteEvent(midi=n["pitch"], start_sec=n["start_sec"], end_sec=n["end_sec"]) for n in notes]
    selected = V2JobService._analysis_for_events(analysis, events,
        bpm=advice["bpm"], key=advice["key"], time_signature="4/4",
        bpm_manual=False, key_manual=False, time_signature_manual=False)
    score, report = build_direct_score(events, selected, title="regression")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert Counter(tuple(n) for n in report["quantized_intervals"]) == Counter(
        (n["p"], n["a"]*12, n["z"]*12) for n in reference["notes"])
    assert score.total_ticks == 55*192
