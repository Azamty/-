from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.summarize_asap_production_raw import summarize, write_reports


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_case(tmp_path: Path, *, route: str = "original_mix") -> tuple[Path, Path, str]:
    repo_root = tmp_path / "repo"
    result_root = repo_root / ".artifacts" / "review" / "asap-production-raw-v1"
    audio = repo_root / "clips" / "asap-v11-01.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"test-audio")
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    registry = repo_root / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
    _write_json(
        registry,
        {
            "cases": [
                {
                    "id": "asap-v11-01",
                    "title": "Fixture",
                    "category": "official_piano_aligned",
                    "source_kind": "official-score-performance-aligned",
                    "input": "clips/asap-v11-01.wav",
                    "input_sha256": audio_hash,
                    "production_gate_selected": True,
                }
            ]
        },
    )
    case_root = result_root / "asap-v11-01"
    identity = {
        "mode": "production",
        "implementation": "MuScriptor+Demucs+GAME+BeatNet",
        "version": "1.2",
    }
    raw = {
        "model_output": True,
        "notes": [{"start_sec": 0.0, "end_sec": 0.5, "midi": 60, "is_drum": False}],
        "provenance": {
            "recognizer_mode": "production",
            "recognizer_fingerprint": "f" * 64,
            "recognizer_identity": identity,
            "source_audio": str(audio.resolve()),
            "beat_engine": "beatnet",
            "beat_source": "original_mix",
            "beat_independent_of_reference": True,
            "route": {"engine": "muscriptor", "route_input": route},
        },
    }
    _write_json(case_root / "raw" / "recognition.json", raw)
    _write_json(
        case_root / "raw" / "beat_grid.json",
        {
            "engine": "beatnet",
            "mode": "offline",
            "beatnet": {"version": "1.1.3", "inference": "DBN"},
            "beats": [{"time_sec": 0.0, "downbeat": True}],
        },
    )
    (case_root / "raw" / "production-recognizer.log").write_text("", encoding="utf-8")
    (case_root / "raw" / "muscriptor" / "worker.log").parent.mkdir(parents=True, exist_ok=True)
    (case_root / "raw" / "muscriptor" / "worker.log").write_text("ok", encoding="utf-8")
    _write_json(
        case_root / "manifest.json",
        {
            "status": "success",
            "raw_only": True,
            "recognizer_mode": "production",
            "recognizer_fingerprint": "f" * 64,
            "raw": {"recognition": "raw/recognition.json", "beat_grid": "raw/beat_grid.json"},
        },
    )
    return repo_root, registry, result_root


def test_summary_accepts_raw_only_production_route_and_writes_manifest(tmp_path: Path) -> None:
    repo_root, registry, result_root = _make_case(tmp_path)

    summary = summarize(result_root, registry, case_ids=("asap-v11-01",), repo_root=repo_root)
    paths = write_reports(summary, result_root)

    assert summary["status"] == "success"
    assert summary["counts"] == {
        "requested": 1,
        "success": 1,
        "failed": 0,
        "model_output_true": 1,
        "beat_grid_present": 1,
        "pitched_events": 1,
    }
    assert summary["cases"][0]["input"]["sha256_match"] is True
    assert summary["cases"][0]["reference_annotation_used"] is False
    assert all(path.is_file() for path in paths)
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    assert not any(item["path"] == "artifact_manifest.json" for item in manifest["artifacts"])
    assert any(item["path"] == "summary.json" for item in manifest["artifacts"])


def test_summary_rejects_reference_route_even_when_payload_is_marked_model_output(tmp_path: Path) -> None:
    repo_root, registry, result_root = _make_case(tmp_path, route="reference_annotation")

    summary = summarize(result_root, registry, case_ids=("asap-v11-01",), repo_root=repo_root)

    assert summary["status"] == "failed"
    assert "production route provenance is incomplete or mismatched" in summary["cases"][0]["errors"]
