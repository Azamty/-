from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("ensure_musescore_3_6_2", ROOT / "scripts" / "ensure_musescore_3_6_2.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_existing_verified_asset_is_reused_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"pinned test asset"
    asset = tmp_path / "MuseScore-test.msi"
    asset.write_bytes(payload)
    monkeypatch.setattr(MODULE, "EXPECTED_BYTES", len(payload))
    monkeypatch.setattr(MODULE, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest())

    result = MODULE.download_asset(asset)

    assert result["downloaded"] is False
    assert result["asset"]["bytes"] == len(payload)
    assert result["asset"]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_asset_verification_rejects_wrong_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = tmp_path / "MuseScore-test.msi"
    asset.write_bytes(b"wrong")
    monkeypatch.setattr(MODULE, "EXPECTED_BYTES", 17)
    monkeypatch.setattr(MODULE, "EXPECTED_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="asset verification failed"):
        MODULE.verify_asset(asset)
