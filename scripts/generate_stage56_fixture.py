"""Regenerate the checked-in MuseScore/music21 stage 5/6 fixture artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.domain import MusicAnalysis, NoteEvent, Score  # noqa: E402
from backend.jianpu_score.musicxml_standardize import write_standardized_score  # noqa: E402
from backend.jianpu_score.musescore_import import convert_performance_midi  # noqa: E402
from backend.jianpu_score.performance_midi import build_performance_midi  # noqa: E402


FIXTURE_SOURCE = ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json"
OUTPUT_DIR = ROOT / "fixtures" / "high_accuracy" / "stage56"
MIDI = OUTPUT_DIR / "stage56.performance.mid"
MUSICXML = OUTPUT_DIR / "stage56.musicxml"
SCORE = OUTPUT_DIR / "stage56.score.json"
ALIGNMENT = OUTPUT_DIR / "stage56.alignment_report.json"
PERFORMANCE_METADATA = OUTPUT_DIR / "stage56.performance.metadata.json"
RELATIVE_MUSICXML = "fixtures/high_accuracy/stage56/stage56.musicxml"


def _normalize_paths(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if "source_musicxml" in document:
        document["source_musicxml"] = RELATIVE_MUSICXML
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        if "source_musicxml" in metadata:
            metadata["source_musicxml"] = RELATIVE_MUSICXML
        report = metadata.get("alignment_report")
        if isinstance(report, dict) and "source_musicxml" in report:
            report["source_musicxml"] = RELATIVE_MUSICXML
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_production_fixture(payload: dict) -> tuple[bytes, dict]:
    """Build the fixture through the same 480 PPQ performance-MIDI API."""

    bpm = float(payload["source"]["tempo_bpm"])
    ticks_per_quarter = int(payload["source"]["ticks_per_quarter"])
    # Six quarter-note beats cover the fixture's 2880 tick end.  The final
    # point closes the last interval without creating a tempo event beyond
    # the score timeline.
    beat_times = [index * 60.0 / bpm for index in range(7)]
    analysis = MusicAnalysis(
        sample_rate=44_100,
        duration_sec=float(payload["source"]["duration_ticks"]) / ticks_per_quarter * 60.0 / bpm + 0.25,
        bpm=bpm,
        key="D",
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
                "mapping": {
                    "manual_bpm_scale": 1.0,
                    "score_origin": {"downbeat_index": 0, "downbeat_sec": 0.0},
                },
            },
        },
    )
    events = tuple(
        NoteEvent(
            start_sec=float(note["start_tick"]) / ticks_per_quarter * 60.0 / bpm,
            end_sec=(int(note["start_tick"]) + int(note["duration_tick"])) / ticks_per_quarter * 60.0 / bpm,
            midi=int(note["midi"]),
            voice_id=f"voice-{note['voice']}",
            source="stage56-production-fixture",
        )
        for note in payload["notes"]
    )
    return build_performance_midi(
        events,
        analysis,
        instrument_group="stage56",
        program=0,
        title="Stage 56 boundary fixture",
    )


def check_fixture() -> int:
    required = (MIDI, MUSICXML, SCORE, ALIGNMENT, PERFORMANCE_METADATA)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"stage56 fixture is missing: {', '.join(missing)}")
    root_tag = ET.parse(MUSICXML).getroot().tag.rsplit("}", 1)[-1]
    if root_tag != "score-partwise":
        raise RuntimeError(f"stage56 MusicXML root must be score-partwise, got {root_tag!r}")
    score = Score.model_validate(json.loads(SCORE.read_text(encoding="utf-8")))
    report = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    metadata = json.loads(PERFORMANCE_METADATA.read_text(encoding="utf-8"))
    if score.key != "D" or score.time_signature != "6/8" or abs(score.bpm - 96.0) > 1e-6:
        raise RuntimeError(
            f"stage56 conductor mismatch: key={score.key}, meter={score.time_signature}, bpm={score.bpm}"
        )
    if report.get("source_note_count") != metadata.get("note_count") or report.get("source_note_count", 0) <= 0:
        raise RuntimeError(
            f"stage56 alignment source count mismatch: report={report.get('source_note_count')}, metadata={metadata.get('note_count')}"
        )
    if score.metadata.get("measure_total_ticks") != score.total_ticks:
        raise RuntimeError("stage56 measure timeline does not end at Score total_ticks")
    print(
        json.dumps(
            {
                "status": "ok",
                "check": True,
                "source_note_count": report["source_note_count"],
                "score_ticks_per_quarter": score.quarter_ticks,
                "total_ticks": score.total_ticks,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate checked-in artifacts without regenerating them")
    args = parser.parse_args()
    if args.check:
        return check_fixture()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(FIXTURE_SOURCE.read_text(encoding="utf-8"))
    midi_bytes, performance_metadata = build_production_fixture(payload)
    MIDI.write_bytes(midi_bytes)
    PERFORMANCE_METADATA.write_text(
        json.dumps(performance_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    convert_performance_midi(MIDI, MUSICXML, instrument_id="stage56", overwrite=True)
    write_standardized_score(
        MUSICXML,
        SCORE,
        alignment_report_path=ALIGNMENT,
        performance_metadata=performance_metadata,
        title="Stage 56 boundary fixture",
    )
    _normalize_paths(SCORE)
    _normalize_paths(ALIGNMENT)
    print(
        json.dumps(
            {
                "status": "ok",
                "midi": str(MIDI),
                "musicxml": str(MUSICXML),
                "score": str(SCORE),
                "alignment": str(ALIGNMENT),
                "performance_metadata": str(PERFORMANCE_METADATA),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
