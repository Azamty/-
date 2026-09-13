from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_high_accuracy_inventory import build_inventory


def _write_raw(path: Path, *, source_audio: Path, model_output: bool, notes: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_output": model_output,
                "source": "production_audio_model" if model_output else "reference_midi_quantizer_isolation",
                "notes": notes,
                "beat_grid": {"beats": [{"downbeat": True}, {"downbeat": False}]},
                "provenance": {
                    "source_audio": str(source_audio),
                    "evaluation_scope": "production_end_to_end" if model_output else "quantizer_isolation",
                },
            }
        ),
        encoding="utf-8",
    )


def test_inventory_counts_declared_render_domains_when_model_raw_is_real(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"audio")
    annotation = tmp_path / "beats.json"
    annotation.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "real-case",
                        "category": "vocal",
                        "source_kind": "vocal",
                        "input": str(audio),
                        "beat_annotation": str(annotation),
                        "beat_annotation_independent": True,
                        "beat_annotation_source": "independent_audio_annotation",
                        "evaluation_scope": "production_end_to_end",
                    },
                    {
                        "id": "fixture-case",
                        "category": "synthetic_rendered",
                        "source_kind": "synthetic",
                        "input": str(audio),
                        "beat_annotation": str(annotation),
                        "beat_annotation_independent": True,
                        "beat_annotation_source": "deterministic_midi_render_ground_truth",
                        "evaluation_scope": "production_end_to_end",
                        "benchmark_role": "production_end_to_end",
                        "render_domain": "synthetic_local_midi_render",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    scan = tmp_path / "scan"
    _write_raw(scan / "real-case" / "production_raw.json", source_audio=audio, model_output=True, notes=[{"midi": 60}])
    _write_raw(scan / "fixture-case" / "production_raw.json", source_audio=audio, model_output=True, notes=[{"midi": 60}])
    inventory = build_inventory(registry, scan_roots=(scan,))
    assert inventory["production_candidate_ids"] == ["real-case", "fixture-case"]
    assert inventory["quantizer_fixture_model_smoke_ids"] == []
    assert inventory["cases"][1]["render_domain"] == "synthetic_local_midi_render"


def test_inventory_marks_reference_raw_without_model_output(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"audio")
    annotation = tmp_path / "beats.json"
    annotation.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "reference-only",
                        "input": str(audio),
                        "beat_annotation": str(annotation),
                        "beat_annotation_independent": True,
                        "evaluation_scope": "quantizer_isolation_fixture",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scan = tmp_path / "scan"
    _write_raw(scan / "reference-only" / "recognition.json", source_audio=audio, model_output=False, notes=[{"midi": 60}])
    inventory = build_inventory(registry, scan_roots=(scan,))
    record = inventory["cases"][0]
    assert inventory["production_candidate_count"] == 0
    assert record["reference_derived_raw_count"] == 1
    assert record["production_candidate"] is False


def test_inventory_excludes_reliable_diagnostic_case_from_gate(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"audio")
    annotation = tmp_path / "beats.json"
    annotation.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "diagnostic-reliable",
                        "input": str(audio),
                        "beat_annotation": str(annotation),
                        "beat_annotation_independent": True,
                        "evaluation_scope": "production_end_to_end",
                        "reference_midi_reliable": True,
                        "production_gate_selected": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scan = tmp_path / "scan"
    _write_raw(scan / "diagnostic-reliable" / "production_raw.json", source_audio=audio, model_output=True, notes=[{"midi": 60}])
    inventory = build_inventory(registry, scan_roots=(scan,))
    record = inventory["cases"][0]
    assert record["production_gate_selected"] is False
    assert record["production_candidate"] is False
    assert inventory["production_candidate_count"] == 0
