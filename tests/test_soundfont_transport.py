from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import backend.app as app_module
from backend.app import create_app


def test_soundfont_transport_supports_range_head_cache_and_no_local_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    soundfont = tmp_path / "MuseScore_General.sf3"
    soundfont.write_bytes(b"RIFFdemo")
    monkeypatch.setattr(app_module, "ensure_soundfont", lambda: soundfont)

    app = create_app(jobs_root=tmp_path / "jobs")
    with TestClient(app) as client:
        ranged = client.get(
            "/api/v2/soundfont",
            headers={"Range": "bytes=0-3", "Host": "tunnel.example"},
        )
        assert ranged.status_code == 206
        assert ranged.content == b"RIFF"
        assert ranged.headers["content-range"] == "bytes 0-3/8"
        assert ranged.headers["content-length"] == "4"
        assert ranged.headers["accept-ranges"] == "bytes"
        assert ranged.headers["cache-control"] == "public, max-age=604800, immutable"
        assert ranged.headers["x-content-type-options"] == "nosniff"

        head = client.head("/api/v2/soundfont", headers={"Host": "tunnel.example"})
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["content-length"] == "8"
        assert head.headers["accept-ranges"] == "bytes"

        status = client.get("/api/v2/soundfont/status", headers={"Host": "tunnel.example"})
        assert status.status_code == 200
        assert status.headers["cache-control"] == "no-store"
        payload = status.json()
        assert "cache_path" not in payload
        assert payload["download_url"] == "/api/v2/soundfont"
        assert payload["range_supported"] is True

        status_head = client.head("/api/v2/soundfont/status", headers={"Host": "tunnel.example"})
        assert status_head.status_code == 200
        assert status_head.content == b""


def test_vendor_processor_has_same_origin_cache_headers_and_head(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    vendor = dist / "vendor"
    vendor.mkdir(parents=True)
    (vendor / "spessasynth_processor.min.js").write_text("var processor = true;", encoding="utf-8")
    monkeypatch.setattr(app_module, "FRONTEND_DIST", dist)

    app = create_app(jobs_root=tmp_path / "jobs")
    with TestClient(app) as client:
        response = client.get(
            "/vendor/spessasynth_processor.min.js",
            headers={"Range": "bytes=0-3", "Host": "tunnel.example"},
        )
        assert response.status_code == 206
        assert response.content == b"var "
        assert response.headers["content-type"].split(";", 1)[0] in {"application/javascript", "text/javascript"}
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["x-content-type-options"] == "nosniff"

        head = client.head("/vendor/spessasynth_processor.min.js")
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["content-length"] == str(len("var processor = true;"))
