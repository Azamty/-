"""Run the installed tsumugi stem checkpoints and inspect MIDI parsing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.models.tsumugi import extract_tsumugi


def main() -> int:
    input_path = ROOT / "artifacts" / "review" / "scale_reference.wav"
    output = ROOT / "artifacts" / "review" / "stage3-tsumugi"
    results: dict[str, object] = {}
    for stem_id, model_type in (
        ("other", "other_v1_5"),
        ("bass", "bass_v2"),
        # The vocal checkpoint is intentionally reported as harmony output;
        # GAME remains the lead-vocal path in specialist routing.
        ("vocals", "vocal_harmony_v1_5"),
    ):
        result = extract_tsumugi(input_path, model_type=model_type, stem_id=stem_id)
        if not result.events:
            raise RuntimeError(f"tsumugi {model_type} smoke produced no events")
        results[stem_id] = {
            "engine": result.engine,
            "model": result.model,
            "metadata": result.metadata,
            "warnings": result.warnings,
            "events": [event.model_dump(mode="json") for event in result.events],
        }
    manifest = {"input": str(input_path.resolve()), "stems": results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
