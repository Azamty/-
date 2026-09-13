"""Common result and error contracts for isolated transcription engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain import NoteEvent


class EngineError(RuntimeError):
    """Base error for a requested engine that cannot produce a result."""


class EngineUnavailableError(EngineError):
    """Raised when an engine, environment, model, or route is unavailable."""


class EngineExecutionError(EngineError):
    """Raised when an available engine subprocess fails or emits bad output."""


@dataclass(frozen=True)
class EngineResult:
    """Normalized output shared by every recognizer adapter."""

    events: list[NoteEvent]
    engine: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class EngineAdapter(Protocol):
    """Minimal adapter shape used by the pipeline and future API layer."""

    engine_id: str

    def extract(self, audio_path: str, **kwargs: Any) -> EngineResult:
        ...


def run_engine(
    engine: str,
    audio_path: str,
    *,
    analysis: Any = None,
    samples: Any = None,
    stem_id: str | None = None,
    language: str = "mixed",
    model_type: str | None = None,
    trusted_internal: bool = False,
) -> EngineResult:
    """Dispatch one explicit adapter without falling back between engines."""

    if engine == "basic-pitch":
        from .basic_pitch import extract_basic_pitch

        events = extract_basic_pitch(audio_path, enforce_upload_size=not trusted_internal)
        return EngineResult(
            events=events,
            engine="basic-pitch",
            metadata={"engine": "basic-pitch", "kind": "baseline", "event_count": len(events)},
        )
    if engine == "librosa":
        if analysis is None or samples is None:
            raise EngineExecutionError("librosa adapter requires the shared audio analysis and samples")
        from ..analysis import extract_librosa_events

        events = extract_librosa_events(samples, analysis.sample_rate, source="librosa")
        return EngineResult(
            events=events,
            engine="librosa",
            metadata={"engine": "librosa", "kind": "fallback", "event_count": len(events)},
        )
    if engine == "game":
        from .game import extract_game

        return extract_game(
            audio_path,
            language=language,
            stem_id=stem_id,
            enforce_upload_size=not trusted_internal,
        )
    if engine == "tsumugi":
        from .tsumugi import extract_tsumugi, model_type_for_stem

        selected_model = model_type or (model_type_for_stem(stem_id) if stem_id else "other_v1_5")
        return extract_tsumugi(
            audio_path,
            model_type=selected_model,
            stem_id=stem_id,
            enforce_upload_size=not trusted_internal,
        )
    raise EngineUnavailableError(f"unknown transcription engine: {engine}")
