from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.job_manager import JobManager


def _score_job(
    tmp_path: Path,
    artifacts: list[dict[str, Any]],
    *,
    source_kind: str = "instrumental",
    selection_revision: int | None = None,
    nested_revision: int | None = None,
    score_refusal: dict[str, str] | None = None,
) -> tuple[Any, str]:
    app = create_app(jobs_root=tmp_path / "jobs")
    manager: JobManager = app.state.jobs
    job_id, input_path = manager.create_v2_job(original_name="fixture.wav", source_kind=source_kind, title="fixture")
    input_path.write_bytes(b"fixture")
    job_dir = manager._safe_job_dir(job_id)
    registered: list[dict[str, Any]] = []
    for index, spec in enumerate(artifacts):
        relative_path = Path(str(spec["relative_path"]))
        path = job_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec["payload"], ensure_ascii=False), encoding="utf-8")
        registered.append(
            manager._register(
                job_dir,
                path,
                artifact_id=str(spec["artifact_id"]),
                kind=str(spec["kind"]),
                label=str(spec.get("label", f"fixture {index}")),
                media_type="application/json",
            )
        )
    v2: dict[str, Any] = {"source_kind": source_kind, "score_refusal": score_refusal}
    if selection_revision is not None:
        v2["selection_revision"] = selection_revision
    if nested_revision is not None:
        v2["selection"] = {"revision": nested_revision}
    manager._update(
        job_id,
        status="completed",
        phase="completed",
        progress=1.0,
        v2=v2,
        artifacts=registered,
    )
    return app, job_id


def _artifact(artifact_id: str, kind: str, revision: int | None, marker: str) -> dict[str, Any]:
    path = f"output/selections/rev-{revision:04d}/score/{artifact_id}.json" if revision is not None else f"output/{artifact_id}.json"
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "relative_path": path,
        "payload": {"marker": marker},
    }


def test_v2_score_uses_current_split_when_current_combo_failed(tmp_path: Path) -> None:
    app, job_id = _score_job(
        tmp_path,
        [
            _artifact("v2-selection-r1-melody-harmony-score-json", "melody_harmony_score_json", 1, "r1-combo"),
            _artifact("v2-selection-r2-piano-score-json", "instrument_score_json", 2, "r2-split"),
        ],
        selection_revision=2,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v2/jobs/{job_id}/score")

    assert response.status_code == 200
    assert response.json() == {"marker": "r2-split"}


def test_v2_score_returns_409_for_current_drum_only_selection(tmp_path: Path) -> None:
    app, job_id = _score_job(
        tmp_path,
        [_artifact("v2-selection-r1-melody-harmony-score-json", "melody_harmony_score_json", 1, "r1-combo")],
        selection_revision=2,
        score_refusal={"code": "no_pitched_tracks", "message": "未选择有音高乐器"},
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v2/jobs/{job_id}/score")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "no_pitched_tracks"


def test_v2_score_prefers_current_combo_over_current_split(tmp_path: Path) -> None:
    app, job_id = _score_job(
        tmp_path,
        [
            _artifact("v2-selection-r2-piano-score-json", "instrument_score_json", 2, "r2-split"),
            _artifact("v2-selection-r2-melody-harmony-score-json", "melody_harmony_score_json", 2, "r2-combo"),
        ],
        nested_revision=2,
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v2/jobs/{job_id}/score")

    assert response.status_code == 200
    assert response.json() == {"marker": "r2-combo"}


def test_v2_score_keeps_unversioned_legacy_artifact_compatible(tmp_path: Path) -> None:
    app, job_id = _score_job(
        tmp_path,
        [_artifact("score-json", "score_json", None, "legacy")],
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v2/jobs/{job_id}/score")

    assert response.status_code == 200
    assert response.json() == {"marker": "legacy"}


def test_vocal_score_without_revision_keeps_legacy_priority(tmp_path: Path) -> None:
    app, job_id = _score_job(
        tmp_path,
        [
            _artifact("vocal-score", "vocal_score_json", None, "vocal"),
            _artifact("score-json", "score_json", None, "legacy"),
        ],
        source_kind="vocal",
    )

    with TestClient(app) as client:
        response = client.get(f"/api/v2/jobs/{job_id}/score")

    assert response.status_code == 200
    assert response.json() == {"marker": "legacy"}
