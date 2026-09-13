"""Print the high-accuracy toolchain readiness as JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

from backend.jianpu_score.high_accuracy import get_high_accuracy_capabilities


if __name__ == "__main__":
    print(json.dumps(get_high_accuracy_capabilities(), ensure_ascii=False, indent=2))
