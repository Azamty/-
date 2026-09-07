"""Regenerate the checked-in MuseScore/music21 stage 5/6 fixture artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.musicxml_standardize import write_standardized_score  # noqa: E402
from backend.jianpu_score.musescore_import import convert_performance_midi  # noqa: E402
from scripts.high_accuracy_fixture_smoke import _write_fixture_midi  # noqa: E402


FIXTURE_SOURCE = ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json"
OUTPUT_DIR = ROOT / "fixtures" / "high_accuracy" / "stage56"
MIDI = OUTPUT_DIR / "stage56.performance.mid"
MUSICXML = OUTPUT_DIR / "stage56.musicxml"
SCORE = OUTPUT_DIR / "stage56.score.json"
ALIGNMENT = OUTPUT_DIR / "stage56.alignment_report.json"
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(FIXTURE_SOURCE.read_text(encoding="utf-8"))
    _write_fixture_midi(MIDI, payload)
    convert_performance_midi(MIDI, MUSICXML, instrument_id="stage56", overwrite=True)
    write_standardized_score(MUSICXML, SCORE, alignment_report_path=ALIGNMENT, title="Stage 56 boundary fixture")
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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
