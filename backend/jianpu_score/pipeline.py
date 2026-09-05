"""Synchronous stage2 pipeline used by the CLI and later API workers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from .analysis import analyze_audio, load_audio
from .capabilities import get_capabilities
from .domain import MusicAnalysis, NoteEvent, Score
from .models.adapter import EngineResult, EngineUnavailableError, run_engine
from .models.demucs import separate_htdemucs
from .models.tsumugi import model_type_for_stem
from .quantize import NoNotesError, quantize_events
from .render import RenderArtifacts, render_score, write_score_json


SUPPORTED_ENGINES = ("basic-pitch", "librosa", "game", "tsumugi", "specialist")


def _engine_plan(
    requested: str,
    *,
    source_kind: str,
    voice_mode: str,
    stem_id: str,
) -> tuple[str, str | None]:
    """Resolve one requested engine to one concrete adapter/model."""

    if requested not in SUPPORTED_ENGINES:
        raise EngineUnavailableError(
            f"unknown transcription engine {requested!r}; choose one of {', '.join(SUPPORTED_ENGINES)}"
        )
    if requested == "specialist":
        if voice_mode == "monophonic" and source_kind == "vocal" and stem_id == "vocals":
            return "game", None
        if voice_mode == "monophonic" and source_kind == "instrumental" and stem_id == "other":
            return "tsumugi", model_type_for_stem(stem_id)
        if voice_mode == "polyphonic" and stem_id in {"vocals", "bass", "other"}:
            return "tsumugi", model_type_for_stem(stem_id)
        raise EngineUnavailableError(
            "specialist routing is explicit: use --source vocal/instrumental "
            "(or --source mixed with --separate) and a supported voice mode; "
            "mixed monophonic is ambiguous and is not silently sent to a model"
        )
    if requested == "game":
        if voice_mode != "monophonic":
            raise EngineUnavailableError("GAME is a lead-vocal monophonic adapter; use specialist/tsumugi for polyphonic stems")
        if stem_id not in {"mixed", "vocals"}:
            raise EngineUnavailableError(f"GAME cannot process stem {stem_id!r}; route vocals or an explicit mixed input")
        return "game", None
    if requested == "tsumugi":
        if stem_id == "mixed":
            raise EngineUnavailableError("tsumugi requires a routed stem; pass --separate so a vocals, bass, or other checkpoint is selected")
        return "tsumugi", model_type_for_stem(stem_id)
    return requested, None


def _ensure_requested_engine(engine: str) -> None:
    capabilities = get_capabilities()
    status = capabilities.get("engines", {}).get(engine)
    if status is None:
        raise EngineUnavailableError(f"engine {engine!r} is not registered")
    if not status.get("available"):
        raise EngineUnavailableError(str(status.get("reason") or f"engine {engine!r} is unavailable"))


def prepare_sources(
    input_path: str | Path,
    *,
    source_kind: str = "mixed",
    separate: bool = False,
    voice_mode: str = "monophonic",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Route each requested source to an independent recognizer input.

    The original mix remains the analysis source.  Demucs stems are only
    routed to recognizers; they are never mixed back together or truncated to
    the shortest stem.
    """

    if source_kind not in {"mixed", "vocal", "instrumental"}:
        raise ValueError("source_kind must be mixed, vocal or instrumental")
    if voice_mode not in {"monophonic", "polyphonic"}:
        raise ValueError("voice_mode must be monophonic or polyphonic")
    source = Path(input_path).resolve()
    if not separate or (source_kind == "mixed" and voice_mode == "monophonic"):
        # A baseline recognizer can work directly on the original mix.  The
        # source label still describes the user's intent; only an explicit
        # separation request (automatically enabled for specialist routes)
        # changes the recognizer input to a Demucs stem.
        return {"mixed": source}
    if output_dir is None:
        output_dir = source.parent / ".separated"
    stems = separate_htdemucs(source, output_dir)
    if voice_mode == "monophonic":
        if source_kind == "vocal":
            return {"vocals": stems["vocals"]}
        if source_kind == "instrumental":
            return {"other": stems["other"]}
        return {"mixed": source}
    if source_kind == "instrumental":
        names = ("bass", "other")
    else:
        names = ("vocals", "bass", "other")
    return {name: stems[name] for name in names}


def prepare_source(
    input_path: str | Path,
    *,
    source_kind: str = "mixed",
    separate: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    """Backward-compatible single-source view of :func:`prepare_sources`."""

    return next(iter(prepare_sources(input_path, source_kind=source_kind, separate=separate, output_dir=output_dir).values()))


def extract_events(
    source_path: str | Path,
    *,
    engine: str = "basic-pitch",
    analysis: MusicAnalysis | None = None,
    samples: np.ndarray | None = None,
    stem_id: str | None = None,
    language: str = "mixed",
    model_type: str | None = None,
) -> list[NoteEvent]:
    if engine == "specialist":
        raise EngineUnavailableError(
            "specialist needs source_kind and voice_mode for per-stem routing; use run_pipeline"
        )
    return run_engine(
        engine,
        os.fspath(Path(source_path).resolve()),
        analysis=analysis,
        samples=samples,
        stem_id=stem_id,
        language=language,
        model_type=model_type,
    ).events


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    engine: str = "basic-pitch",
    voice_mode: str = "monophonic",
    source_kind: str = "mixed",
    separate: bool = False,
    bpm_override: float | None = None,
    key_override: str | None = None,
    time_signature_override: str | None = None,
    language: str = "mixed",
    title: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[MusicAnalysis, Score, RenderArtifacts]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    original = Path(input_path).resolve()
    def report(phase: str) -> None:
        if progress_callback is not None:
            progress_callback(phase)

    report("probing")
    _ensure_requested_engine(engine)
    samples, analysis = analyze_audio(
        original,
        bpm_override=bpm_override,
        key_override=key_override,
        time_signature_override=time_signature_override,
    )
    # Source and voice choices determine routing even when a caller omits the
    # internal separation flag. This keeps API, CLI and future UI callers on
    # the same stem contract.
    requires_stems = source_kind in {"vocal", "instrumental"} or (
        voice_mode == "polyphonic" and source_kind == "mixed"
    )
    effective_separate = bool(separate or requires_stems)
    report("separating" if effective_separate else "recognizing")
    sources = prepare_sources(
        original,
        source_kind=source_kind,
        separate=effective_separate,
        voice_mode=voice_mode,
        output_dir=destination / "separation",
    )
    events: list[NoteEvent] = []
    skipped_stems: list[str] = []
    engine_by_stem: dict[str, str] = {}
    engine_details: dict[str, dict[str, object]] = {}
    engine_warnings: list[str] = []
    for stem_id, source in sources.items():
        report("recognizing")
        adapter_name, model_type = _engine_plan(
            engine,
            source_kind=source_kind,
            voice_mode=voice_mode,
            stem_id=stem_id,
        )
        stem_samples = samples if source == original else None
        if engine == "librosa" and stem_samples is None:
            stem_samples, _stem_rate = load_audio(source, sample_rate=analysis.sample_rate)
        result: EngineResult = run_engine(
            adapter_name,
            os.fspath(source),
            analysis=analysis,
            samples=stem_samples,
            stem_id=stem_id,
            language=language,
            model_type=model_type,
            trusted_internal=source != original,
        )
        engine_by_stem[stem_id] = result.engine
        engine_details[stem_id] = {
            **result.metadata,
            "requested_engine": engine,
            "adapter": adapter_name,
        }
        engine_warnings.extend(result.warnings)
        extracted = result.events
        if not extracted:
            skipped_stems.append(stem_id)
            continue
        events.extend(
            event.model_copy(
                update={
                    "stem_id": stem_id,
                    "metadata": {
                        **event.metadata,
                        "stem_id": stem_id,
                        "engine": result.engine,
                        **({"model": result.model} if result.model else {}),
                    },
                }
            )
            for event in extracted
        )
    if not events:
        raise NoNotesError("NoNotes: every selected source stem produced no usable note events")
    analysis = analysis.model_copy(
        update={
            "note_events": events,
            "metadata": {
                **analysis.metadata,
                "engine": engine,
                "language": language,
                "engine_by_stem": engine_by_stem,
                "engine_details": engine_details,
                "source_kind": source_kind,
                "source_stems": list(sources),
                "prepared_audio": {stem_id: os.fspath(path) for stem_id, path in sources.items()},
                "skipped_stems": skipped_stems,
            },
            "warnings": [
                *analysis.warnings,
                *engine_warnings,
                *[f"声部 {stem_id} 未识别到音符，已跳过并保留其他声部时间线" for stem_id in skipped_stems],
            ],
        }
    )
    report("quantizing")
    score = quantize_events(
        events,
        analysis,
        mode=voice_mode,
        title=title or Path(input_path).stem,
    )
    score = score.model_copy(
        update={
            "metadata": {
                **score.metadata,
                "source_audio": os.fspath(original),
                "prepared_audio": {stem_id: os.fspath(path) for stem_id, path in sources.items()},
                "engine_by_stem": engine_by_stem,
                "engine_details": engine_details,
                "skipped_stems": skipped_stems,
            }
        }
    )
    # Persist the analysis and quantized Score before invoking the external
    # renderer.  A converter/LilyPond failure must still leave the precise
    # engine output and score available for diagnosis or a renderer-only
    # retry; the renderer is not allowed to be the first durable write.
    (destination / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    write_score_json(score, destination / "score.json")
    report("rendering")
    artifacts = render_score(score, destination, basename="score")
    return analysis, score, artifacts
