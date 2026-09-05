"""Confirm a real silent WAV fails with an explicit NoNotes error."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.jianpu_score.pipeline import run_pipeline
from backend.jianpu_score.quantize import NoNotesError


def main() -> int:
    output = ROOT / "artifacts" / "review" / "stage2-no-notes"
    output.mkdir(parents=True, exist_ok=True)
    audio = output / "silence.wav"
    sf.write(audio, np.zeros(22050, dtype=np.float32), 22050)
    try:
        run_pipeline(audio, output / "job", engine="librosa", bpm_override=120, key_override="C", time_signature_override="4/4")
    except NoNotesError as error:
        manifest = {"input": str(audio), "expected_error": type(error).__name__, "message": str(error)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0
    raise RuntimeError("silent audio unexpectedly produced a score")


if __name__ == "__main__":
    raise SystemExit(main())
