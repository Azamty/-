"""Pydantic response models for the local HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiError(BaseModel):
    code: str
    message: str


class ArtifactItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str
    kind: str
    label: str
    filename: str
    relative_path: str
    media_type: str
    size_bytes: int
    stem_id: str | None = None
    page: int | None = None
    url: str | None = None


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "1.0"
    id: str
    status: str
    phase: str
    phase_label: str
    phase_index: int | None = None
    progress: float | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    attempt: int = 1
    options: dict[str, Any] = {}
    input: dict[str, Any] = {}
    error: ApiError | None = None
    warnings: list[str] = []
    artifacts: list[ArtifactItem] = []
    summary: dict[str, Any] | None = None
    retryable: bool = False
    score_available: bool = False
    artifacts_available: bool = False


class ArtifactsResponse(BaseModel):
    job_id: str
    artifacts: list[ArtifactItem]
