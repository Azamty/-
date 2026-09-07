from __future__ import annotations

import json
from pathlib import Path

from backend.jianpu_score.high_accuracy import (
    BEATNET_VERSION,
    MUSIC21_VERSION,
    MUSESCORE_RELEASE_SHA256,
    MUSESCORE_RELEASE_URL,
    MUSESCORE_IMPORT_PROFILE,
    MUSESCORE_IMPORT_PROFILE_SHA256,
    get_high_accuracy_capabilities,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage_a_fixture_is_deterministic_and_covers_notation_boundaries() -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "high_accuracy" / "beat_grid_fixture.json").read_text(encoding="utf-8")
    )
    grid = fixture["beat_grid"]["beats"]
    assert fixture["source"]["ticks_per_quarter"] == 480
    assert fixture["source"]["time_signature"] == "6/8"
    assert [item["time_sec"] for item in grid] == sorted(item["time_sec"] for item in grid)
    assert grid[0]["downbeat"] is True
    assert grid[3]["downbeat"] is True
    assert fixture["expected"] == {
        "requires_polyphony": True,
        "requires_ties": True,
        "requires_triplet": True,
        "triplet_pitches": [74, 76, 78],
        "requires_meter": "6/8",
        "requires_adaptive_quantization": True,
        "minimum_score_ticks_per_quarter": 48,
        "voice_count": 4,
        "musescore_profile_sha256": "86742B91922F921F725A1A5810572AB458EB7FB7AAC46FC683C92352B837C9FF",
        "disabled_musescore_tuplets": ["Quintuplets", "Septuplets", "Nonuplets"],
        "required_musescore_tuplet": [3, 2],
    }
    assert fixture["voice_stress"]["source"]["time_signature"] == "4/4"
    assert fixture["voice_stress"]["source"]["duration_ticks"] == 3840
    assert len(fixture["voice_stress"]["notes"]) == 20
    assert len({(item["start_tick"], item["midi"]) for item in fixture["notes"]}) == len(fixture["notes"])


def test_high_accuracy_toolchain_contract_is_pinned() -> None:
    capabilities = get_high_accuracy_capabilities()
    assert capabilities["beatnet"]["required_version"] == BEATNET_VERSION
    assert capabilities["music21"]["required_version"] == MUSIC21_VERSION
    assert capabilities["musescore"]["required_version"] == "4.7.4"
    assert capabilities["musescore"]["download_url"] == MUSESCORE_RELEASE_URL
    assert capabilities["musescore"]["download_sha256"] == MUSESCORE_RELEASE_SHA256
    assert capabilities["runtime"]["score_ticks_per_quarter"] == 48
    assert capabilities["runtime"]["musescore_import_profile"] == MUSESCORE_IMPORT_PROFILE
    assert capabilities["runtime"]["musescore_import_profile_sha256"] == MUSESCORE_IMPORT_PROFILE_SHA256
    assert capabilities["musescore"]["import_profile_available"] is True


def test_high_accuracy_capability_does_not_claim_ready_when_a_tool_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("JIANPU_BEATNET_PYTHON", str(ROOT / "missing-beatnet-python.exe"))
    monkeypatch.setenv("JIANPU_NOTATION_PYTHON", str(ROOT / "missing-notation-python.exe"))
    monkeypatch.setenv("JIANPU_MUSESCORE", str(ROOT / "missing-musescore.exe"))

    capabilities = get_high_accuracy_capabilities()

    assert capabilities["available"] is False
    assert capabilities["beatnet"]["available"] is False
    assert capabilities["music21"]["available"] is False
    assert capabilities["musescore"]["available"] is False
    assert "environment is missing" in capabilities["reason"]
