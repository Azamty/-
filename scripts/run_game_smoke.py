"""Run the installed GAME checkpoint for the supported language requests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.models.game import extract_game


def main() -> int:
    input_path = ROOT / "artifacts" / "review" / "scale_reference.wav"
    output = ROOT / "artifacts" / "review" / "stage3-game"
    results: dict[str, object] = {}
    for language in ("zh", "ja"):
        result = extract_game(input_path, language=language, stem_id="vocals")
        if not result.events:
            raise RuntimeError(f"GAME {language} smoke produced no events")
        results[language] = {
            "engine": result.engine,
            "model": result.model,
            "metadata": result.metadata,
            "warnings": result.warnings,
            "events": [event.model_dump(mode="json") for event in result.events],
        }
    manifest = {"input": str(input_path.resolve()), "languages": results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
