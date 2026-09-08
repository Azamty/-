"""Resumable orchestration for the 30-case high-accuracy benchmark.

Recognition runs exactly once per case, stores its raw notes and beat grid
immutably, and passes independent copies of that same payload to the legacy
baseline and the new service.  The default production worker routes
instrumental audio through MuScriptor plus one original-mix BeatNet pass, and
vocal audio through Demucs, GAME cleanup, plus one original-mix BeatNet pass.
A missing adapter is a recorded failure, never a fabricated pass.  The
``--reference-isolation`` mode is intentionally marked as such and only
prepares a deterministic quantizer-isolation input from a reference MIDI.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / ".cache" / "high-accuracy-benchmarks" / "runs"
RUNNER_SCHEMA_VERSION = "1.1"
PRODUCTION_RECOGNIZER_VERSION = "1.2"
MUSCRIPTOR_BENCHMARK_SEED = 20260907

Adapter = Callable[[Mapping[str, Any], Mapping[str, Any], Path], Mapping[str, Any]]


class BatchRunError(RuntimeError):
    """A benchmark stage failed and was recorded in its case manifest."""

    def __init__(self, case_id: str, stage: str, cause: str):
        self.case_id = case_id
        self.stage = stage
        self.cause = cause
        super().__init__(f"{case_id}/{stage}: {cause}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> str:
    """Write an immutable JSON artifact, or prove an existing copy matches."""

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != encoded:
            raise ValueError(f"immutable benchmark artifact changed: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    return _sha256(path)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Replace a retryable pipeline manifest atomically (raw stays immutable)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return _sha256(path)


def _safe_case_dir(root: Path, case_id: str) -> Path:
    if not case_id or any(part in {"", ".", ".."} for part in Path(case_id).parts) or Path(case_id).name != case_id:
        raise ValueError(f"unsafe benchmark case id: {case_id!r}")
    case_dir = (root / case_id).resolve()
    if not case_dir.is_relative_to(root.resolve()):
        raise ValueError(f"benchmark case escaped result root: {case_id!r}")
    return case_dir


def _pipeline_roots(result_root: Path, baseline_root: Path | None, new_root: Path | None) -> tuple[Path, Path]:
    """Resolve independent pipeline roots and reject accidental aliasing."""

    result = result_root.resolve()
    baseline = (baseline_root or result / "baseline").resolve()
    new = (new_root or result / "new").resolve()
    if baseline in {result, new} or new == result:
        raise ValueError("baseline, new, and raw result roots must be independent")
    return baseline, new


def _recognizer_identity(recognizer: Adapter | None) -> dict[str, Any]:
    """Return the stable identity that owns an immutable raw payload."""

    if recognizer is None:
        return {"mode": "unconfigured", "implementation": "none", "version": RUNNER_SCHEMA_VERSION}
    declared = getattr(recognizer, "recognizer_identity", None)
    if callable(declared):
        declared = declared()
    if isinstance(declared, Mapping):
        identity = dict(declared)
    else:
        implementation = getattr(recognizer, "__qualname__", type(recognizer).__qualname__)
        module = getattr(recognizer, "__module__", type(recognizer).__module__)
        identity = {"mode": "custom", "implementation": f"{module}.{implementation}", "version": RUNNER_SCHEMA_VERSION}
    identity.setdefault("mode", "custom")
    identity.setdefault("version", RUNNER_SCHEMA_VERSION)
    return identity


def _recognizer_provenance(recognizer: Adapter | None) -> dict[str, Any]:
    identity = _recognizer_identity(recognizer)
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "mode": str(identity.get("mode") or "custom"),
        "identity": identity,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def _effective_evaluation_scope(case: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    """Derive the claim scope from recognition provenance, not registry intent."""

    provenance = raw.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    if provenance.get("reference_is_not_model_output") is True or raw.get("model_output") is False:
        return "quantizer_isolation"
    if raw.get("model_output") is True or provenance.get("model_output") is True:
        return "production_end_to_end"
    declared = provenance.get("effective_evaluation_scope") or provenance.get("evaluation_scope")
    if isinstance(declared, str) and declared.startswith("quantizer_isolation"):
        return "quantizer_isolation"
    return "quantizer_isolation"


def _annotate_raw_provenance(
    raw: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    recognizer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach immutable runner ownership and effective scope to one raw payload."""

    payload = dict(raw)
    declared = payload.get("provenance")
    provenance = dict(declared) if isinstance(declared, Mapping) else {}
    expected_mode = str(recognizer_provenance["mode"])
    expected_fingerprint = str(recognizer_provenance["fingerprint"])
    if provenance.get("recognizer_mode") not in {None, expected_mode} or provenance.get("recognizer_fingerprint") not in {None, expected_fingerprint}:
        raise ValueError("recognizer payload provenance does not match the requested recognizer identity")
    effective_scope = _effective_evaluation_scope(case, payload)
    provenance.update(
        {
            "recognizer_mode": expected_mode,
            "recognizer_identity": dict(recognizer_provenance["identity"]),
            "recognizer_fingerprint": expected_fingerprint,
            "model_output": payload.get("model_output") is True,
            "case_evaluation_scope": case.get("evaluation_scope"),
            "effective_evaluation_scope": effective_scope,
        }
    )
    payload["provenance"] = provenance
    return payload


def _validate_existing_raw_provenance(
    raw: Mapping[str, Any],
    *,
    case_id: str,
    recognizer_provenance: Mapping[str, Any],
) -> None:
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BatchRunError(case_id, "raw", "existing recognition.json has no recognizer provenance; use a new mode-isolated result root")
    expected_mode = str(recognizer_provenance["mode"])
    expected_fingerprint = str(recognizer_provenance["fingerprint"])
    actual_mode = provenance.get("recognizer_mode")
    actual_fingerprint = provenance.get("recognizer_fingerprint")
    if actual_mode != expected_mode or actual_fingerprint != expected_fingerprint:
        raise BatchRunError(
            case_id,
            "raw",
            "existing recognition.json belongs to a different recognizer "
            f"(found mode={actual_mode!r}, fingerprint={actual_fingerprint!r}; "
            f"requested mode={expected_mode!r}, fingerprint={expected_fingerprint!r}); "
            "use a new mode-isolated result root",
        )


def _reference_isolation_payload(case: Mapping[str, Any], *, root: Path) -> Mapping[str, Any]:
    """Return a reference-derived payload with explicit non-model provenance."""

    value = str(case.get("reference_midi") or "")
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"reference MIDI unavailable: {path}")
    midi = mido.MidiFile(path)
    notes: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        active: dict[tuple[int, int], list[int]] = {}
        for message in track:
            tick += int(message.time)
            channel = int(getattr(message, "channel", 0))
            key = (channel, int(getattr(message, "note", -1)))
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault(key, []).append(tick)
            elif message.type in {"note_on", "note_off"} and key in active and active[key]:
                start = active[key].pop(0)
                if tick > start:
                    notes.append({"start_quarter": start / midi.ticks_per_beat, "end_quarter": tick / midi.ticks_per_beat, "midi": key[1], "track_index": track_index, "channel": channel})
    beat_value = str(case.get("beat_annotation") or "")
    beat_path = Path(beat_value)
    if not beat_path.is_absolute():
        beat_path = (ROOT / beat_path).resolve()
    beat_grid: Mapping[str, Any] = {}
    if beat_path.is_file():
        payload = json.loads(beat_path.read_text(encoding="utf-8"))
        beat_grid = payload.get("beat_grid", payload) if isinstance(payload, Mapping) else {}
    beat_records = beat_grid.get("beats", []) if isinstance(beat_grid, Mapping) else []
    beat_times = [float(item.get("time_sec")) for item in beat_records if isinstance(item, Mapping) and item.get("time_sec") is not None]
    tempo = beat_grid.get("tempo") if isinstance(beat_grid, Mapping) and isinstance(beat_grid.get("tempo"), Mapping) else {}
    bpm = float(tempo.get("selected_bpm") or 120.0)
    if bpm == 120.0 and len(beat_times) >= 2 and beat_times[1] > beat_times[0]:
        bpm = 60.0 / (beat_times[1] - beat_times[0])
    time_signature = beat_grid.get("time_signature", "4/4") if isinstance(beat_grid, Mapping) else "4/4"
    if isinstance(time_signature, Mapping):
        time_signature = time_signature.get("selected", "4/4")
    return {
        "schema_version": "1.0",
        "source": "reference_midi_quantizer_isolation",
        "model_output": False,
        "reference_midi": str(path),
        "reference_sha256": _sha256(path),
        "notes": notes,
        "beat_grid": beat_grid,
        "analysis": {
            "sample_rate": 44_100,
            "duration_sec": max((float(item["end_quarter"]) * 60.0 / bpm for item in notes), default=(beat_times[-1] + 0.1 if beat_times else 0.1)),
            "bpm": bpm,
            "time_signature": str(time_signature),
            "key": "C",
            "metadata": {
                "beat_engine": "beatnet",
                "beatnet_version": "1.1.3",
                "beat_source": "reference_midi_quantizer_isolation",
                "reference_derived": True,
                "beat_grid": beat_grid,
            },
        },
        "provenance": {"evaluation_scope": "quantizer_isolation", "reference_is_not_model_output": True},
    }


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop a production recognizer and every model child it created."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, 15)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # A model may ignore SIGTERM.  The process group/taskkill branch above
        # should normally have removed the entire tree; this final kill keeps
        # the parent from returning while an orphaned child is still running.
        try:
            process.kill()
        except OSError:
            pass


class ProductionRecognizer:
    """Run the same model route used by V2 in a killable child process.

    The child writes a normalized raw payload containing the original audio
    BeatNet analysis.  Instrumental input uses MuScriptor on the original
    mix; vocal input uses Demucs then GAME plus the shared original-mix
    analysis.  The parent never imports either model runtime into the batch
    process and can terminate the whole child tree on timeout.
    """

    isolated_process = True

    @property
    def recognizer_identity(self) -> Mapping[str, Any]:
        return {
            "mode": "production",
            "implementation": "MuScriptor+Demucs+GAME+BeatNet",
            "version": PRODUCTION_RECOGNIZER_VERSION,
            "demucs_model": self.demucs_model or "htdemucs",
            "beat_route": "original_mix_once",
            "muscriptor_decode": "greedy_float32_deterministic",
            "muscriptor_seed": MUSCRIPTOR_BENCHMARK_SEED,
        }

    def __init__(self, *, timeout_sec: float = 1800.0, demucs_model: str | None = None) -> None:
        self.timeout_sec = float(timeout_sec)
        self.demucs_model = demucs_model

    def __call__(self, case: Mapping[str, Any], _raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
        audio_value = str(case.get("input") or "")
        audio = Path(audio_value)
        if not audio.is_absolute():
            audio = (ROOT / audio).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"production recognizer input is unavailable: {audio}")
        source_kind = "vocal" if str(case.get("source_kind")) == "vocal" else "instrumental"
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            os.fspath(ROOT / "scripts" / "high_accuracy_production_recognizer.py"),
            "--audio",
            os.fspath(audio),
            "--source-kind",
            source_kind,
            "--output",
            os.fspath(destination),
        ]
        if self.demucs_model:
            command.extend(("--demucs-model", self.demucs_model))
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        try:
            output, _ = process.communicate(timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                # ``taskkill /T`` is authoritative on Windows, and the POSIX
                # process-group kill above is authoritative on Unix.  Do not
                # leave a reader or model child behind if a pipe is stubborn.
                output = ""
            (destination / "production-recognizer.log").write_text(output or "", encoding="utf-8")
            raise TimeoutError(f"production recognizer timed out after {self.timeout_sec:g}s; process tree terminated") from exc
        (destination / "production-recognizer.log").write_text(output or "", encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"production recognizer failed ({process.returncode}): {(output or '')[-4000:]}")
        raw_path = destination / "production_raw.json"
        if not raw_path.is_file():
            raise RuntimeError("production recognizer completed without production_raw.json")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or raw.get("model_output") is not True:
            raise RuntimeError("production recognizer payload is not marked as model output")
        return raw


class ReferenceIsolationRecognizer:
    """Build raw from reference MIDI for quantizer diagnostics only."""

    @property
    def recognizer_identity(self) -> Mapping[str, Any]:
        return {
            "mode": "reference-isolation",
            "implementation": "reference-midi-quantizer-isolation",
            "version": "1.0",
        }

    def __call__(self, case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
        return _reference_isolation_payload(case, root=ROOT)


def _analysis_and_events_from_raw(raw: Mapping[str, Any], case: Mapping[str, Any]):
    """Build the production domain objects required by either score chain.

    A real recognizer returns second based note events and an analysis object.
    The explicit reference-isolation payload uses quarter positions and is
    converted with its declared fixed tempo; it remains marked as a
    quantizer-only provenance in the caller's manifest.
    """

    from backend.jianpu_score.domain import MusicAnalysis, NoteEvent, normalize_key, normalize_time_signature

    analysis_payload = raw.get("analysis") if isinstance(raw.get("analysis"), Mapping) else {}
    beat_grid = raw.get("beat_grid") if isinstance(raw.get("beat_grid"), Mapping) else {}
    tempo_payload = beat_grid.get("tempo") if isinstance(beat_grid.get("tempo"), Mapping) else {}
    bpm = float(analysis_payload.get("bpm") or tempo_payload.get("selected_bpm") or raw.get("bpm") or 120.0)
    meter = str(analysis_payload.get("time_signature") or beat_grid.get("time_signature") or case.get("time_signature") or "4/4")
    key = str(analysis_payload.get("key") or raw.get("key") or "C")
    try:
        meter = normalize_time_signature(meter)
        key = normalize_key(key)
    except ValueError as exc:
        raise ValueError(f"raw analysis has unsupported key or meter: {exc}") from exc
    beat_records = beat_grid.get("beats", []) if isinstance(beat_grid, Mapping) else []
    beat_times = [float(item.get("time_sec")) for item in beat_records if isinstance(item, Mapping) and item.get("time_sec") is not None]
    source_kind = str(raw.get("source_kind") or case.get("source_kind") or "")
    cleanup_report: Mapping[str, Any] | None = None
    raw_notes = raw.get("notes", [])
    if source_kind == "instrumental" and raw.get("model_output") is True:
        from backend.jianpu_score.instrumental_cleanup import clean_instrumental_model_notes

        cleanup = clean_instrumental_model_notes(raw_notes)
        prepared_notes = cleanup.events
        cleanup_report = cleanup.report
    else:
        prepared_notes = tuple(dict(item) for item in raw_notes)
    notes: list[NoteEvent] = []
    max_end = 0.0
    for index, item in enumerate(prepared_notes):
        if not isinstance(item, Mapping):
            raise ValueError(f"raw note {index} is not an object")
        if item.get("start_sec") is not None and item.get("end_sec") is not None:
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
        elif item.get("start_quarter") is not None and item.get("end_quarter") is not None:
            start_sec = float(item["start_quarter"]) * 60.0 / bpm
            end_sec = float(item["end_quarter"]) * 60.0 / bpm
        else:
            raise ValueError(f"raw note {index} lacks start_sec/end_sec")
        max_end = max(max_end, end_sec)
        channel = item.get("channel")
        is_drum = bool(item.get("is_drum")) or channel is not None and int(channel) == 9
        cleanup_lineage = item.get("_instrumental_cleanup")
        primary_source_index = index
        source_indices = [index]
        if isinstance(cleanup_lineage, Mapping):
            try:
                primary_source_index = int(cleanup_lineage["primary_source_index"])
                source_indices = [int(value) for value in cleanup_lineage.get("source_indices", [primary_source_index])]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"raw note {index} has invalid instrumental cleanup lineage") from exc
        event_metadata = {
            "raw_index": primary_source_index,
            "is_drum": is_drum,
            "channel": channel,
            "instrument_group": item.get("instrument_group"),
            "program": item.get("program"),
        }
        if isinstance(cleanup_lineage, Mapping):
            event_metadata["instrumental_cleanup"] = {
                "primary_source_index": primary_source_index,
                "source_indices": source_indices,
                "merged_source_indices": [
                    int(value) for value in cleanup_lineage.get("merged_source_indices", [])
                ],
            }
        notes.append(
            NoteEvent(
                start_sec=start_sec,
                end_sec=end_sec,
                midi=int(item["midi"]),
                confidence=item.get("confidence"),
                velocity=item.get("velocity"),
                raw_pitch=item.get("raw_pitch"),
                voice_id=str(item.get("voice_id", "voice-0")),
                source=str(item.get("source", "benchmark-raw")),
                metadata=event_metadata,
            )
        )
    duration_sec = max(float(analysis_payload.get("duration_sec") or 0.0), max_end, (beat_times[-1] if beat_times else 0.0) + 0.1, 0.1)
    sample_rate = int(analysis_payload.get("sample_rate") or raw.get("sample_rate") or 44_100)
    metadata = dict(analysis_payload.get("metadata") or {})
    metadata.setdefault("beat_source", "beatnet")
    metadata["beat_grid"] = beat_grid
    if cleanup_report is not None:
        metadata["instrumental_cleanup"] = dict(cleanup_report)
    analysis = MusicAnalysis(sample_rate=sample_rate, duration_sec=duration_sec, bpm=bpm, time_signature=meter, key=key, beat_times=beat_times, note_events=notes, warnings=list(analysis_payload.get("warnings") or []), metadata=metadata)
    return analysis, notes


def _pitched_events(events: Sequence[Any]) -> list[Any]:
    """Exclude drum events from numeric notation while retaining raw input."""

    return [event for event in events if not bool(getattr(event, "metadata", {}).get("is_drum"))]


def _pitched_analysis(analysis: Any, events: Sequence[Any]) -> Any:
    return analysis.model_copy(update={"note_events": list(events)})


def legacy_baseline_adapter(case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
    """Run the preserved uniform-grid quantizer for baseline comparison only."""

    from backend.jianpu_score.quantize import quantize_events
    from backend.jianpu_score.render import render_score

    analysis, all_events = _analysis_and_events_from_raw(raw, case)
    events = _pitched_events(all_events)
    if not events:
        raise ValueError("raw recognition contains no pitched events after excluding drums")
    analysis = _pitched_analysis(analysis, events)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cleanup_report = analysis.metadata.get("instrumental_cleanup")
    cleanup_path: Path | None = None
    if isinstance(cleanup_report, Mapping):
        cleanup_path = destination / "instrumental.cleanup.report.json"
        cleanup_path.write_text(
            json.dumps(cleanup_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    score = quantize_events(events, analysis, mode="polyphonic", title=str(case.get("title") or case["id"]))
    score_path = destination / "baseline.score.json"
    score_path.write_text(json.dumps(score.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rendered = render_score(score, destination / "render", basename="baseline")
    beat_path = destination / "beat_grid.json"
    beat_path.write_text(json.dumps(raw.get("beat_grid", {}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_midi = Path(rendered.midi_path).resolve() if rendered.midi_path else None
    if final_midi is None or not final_midi.is_file():
        raise RuntimeError("legacy baseline renderer did not produce final MIDI")
    result = {"engine": "legacy-uniform-grid", "score_json": str(score_path.relative_to(destination.resolve())), "render": rendered.model_dump(mode="json"), "final_midi": str(final_midi.relative_to(destination.resolve())), "beat_grid": "beat_grid.json", "profile": "legacy-uniform-grid"}
    if cleanup_path is not None:
        result["instrumental_cleanup"] = str(cleanup_path.relative_to(destination.resolve()))
    return result


def high_accuracy_service_adapter(case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
    """Run the production high-accuracy service on the shared raw payload."""

    from backend.jianpu_score.high_accuracy_service import build_high_accuracy_artifacts

    analysis, all_events = _analysis_and_events_from_raw(raw, case)
    events = _pitched_events(all_events)
    if not events:
        raise ValueError("raw recognition contains no pitched events after excluding drums")
    analysis = _pitched_analysis(analysis, events)
    destination = destination.resolve()
    service_output = destination / "service_output"
    cleanup_report = analysis.metadata.get("instrumental_cleanup")
    cleanup_path: Path | None = None
    if isinstance(cleanup_report, Mapping):
        destination.mkdir(parents=True, exist_ok=True)
        cleanup_path = destination / "instrumental.cleanup.report.json"
        cleanup_path.write_text(
            json.dumps(cleanup_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    source_kind = str(case.get("source_kind"))
    variant = "game-cleaned" if source_kind == "vocal" else "instrument-part"
    result = build_high_accuracy_artifacts(instrument_id=str(case["id"]), title=str(case.get("title") or case["id"]), program=int(case.get("program", 0)), is_drum=False, events=events, analysis=analysis, output_dir=service_output, variant=variant, overwrite=False)
    beat_path = destination / "beat_grid.json"
    beat_path.write_text(json.dumps(raw.get("beat_grid", {}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact_dicts = [artifact.as_dict() for artifact in result.artifacts]
    score_artifact = next((item for item in artifact_dicts if "score_midi" in str(item.get("kind", "")) or str(item.get("relative_path", "")).endswith("score.mid")), None)
    if score_artifact is None:
        raise RuntimeError("high-accuracy service did not register final score MIDI")
    final_midi = service_output / str(score_artifact["relative_path"])
    if not final_midi.is_file():
        raise RuntimeError(f"high-accuracy final MIDI is missing: {final_midi}")
    result_payload = {"engine": "musescore-midi-import", "variant": variant, "profile": variant, "manifest": str(result.manifest_path.relative_to(destination.resolve())), "status": result.status, "jianpu_status": result.jianpu_status, "artifacts": artifact_dicts, "final_midi": str(final_midi.relative_to(destination.resolve())), "beat_grid": "beat_grid.json"}
    if cleanup_path is not None:
        result_payload["instrumental_cleanup"] = str(cleanup_path.relative_to(destination.resolve()))
    return result_payload


class BenchmarkBatchRunner:
    """Run one registry through shared recognition, baseline and new adapters."""

    def __init__(
        self,
        *,
        recognizer: Adapter | None,
        baseline: Adapter | None,
        new_chain: Adapter | None,
        timeout_sec: float = 1800.0,
        raw_only: bool = False,
    ) -> None:
        self.recognizer = recognizer
        self.baseline = baseline
        self.new_chain = new_chain
        self.timeout_sec = float(timeout_sec)
        self.raw_only = bool(raw_only)

    def _call_with_timeout(self, adapter: Adapter, case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path, stage: str) -> Mapping[str, Any]:
        started = time.monotonic()
        try:
            # ProductionRecognizer owns a killable process tree.  Other
            # adapters run synchronously so a timeout never leaves a model or
            # external process running behind the resumable manifest.
            result = adapter(case, copy.deepcopy(raw), destination)
        except BatchRunError:
            raise
        except Exception as exc:
            raise BatchRunError(str(case["id"]), stage, f"{type(exc).__name__}: {exc}") from exc
        elapsed = time.monotonic() - started
        if elapsed > self.timeout_sec:
            raise BatchRunError(
                str(case["id"]),
                stage,
                f"adapter exceeded {self.timeout_sec:g}s after synchronous cleanup ({elapsed:.1f}s); no background worker was left running",
            )
        if not isinstance(result, Mapping):
            raise BatchRunError(str(case["id"]), stage, "adapter must return a JSON object")
        return result

    def run_case(
        self,
        case: Mapping[str, Any],
        *,
        result_root: Path,
        baseline_result_root: Path | None = None,
        new_result_root: Path | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        case_id = str(case["id"])
        case_dir = _safe_case_dir(result_root, case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        baseline_root, new_root = _pipeline_roots(result_root, baseline_result_root, new_result_root)
        recognizer_provenance = _recognizer_provenance(self.recognizer)
        manifest_path = case_dir / "manifest.json"
        state: dict[str, Any] = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "case_id": case_id,
            "status": "running",
            "evaluation_scope": None,
            "case_evaluation_scope": case.get("evaluation_scope"),
            "recognizer_mode": recognizer_provenance["mode"],
            "recognizer_fingerprint": recognizer_provenance["fingerprint"],
            "raw": None,
            "pipelines": {},
            "pipeline_roots": {"baseline": str(baseline_root), "new": str(new_root)},
            "error": None,
        }
        raw_dir = case_dir / "raw"
        raw_path = raw_dir / "recognition.json"
        try:
            if raw_path.is_file():
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping) or not isinstance(raw.get("notes"), list):
                    raise BatchRunError(case_id, "raw", "existing recognition.json is invalid")
                _validate_existing_raw_provenance(raw, case_id=case_id, recognizer_provenance=recognizer_provenance)
                raw_hash = _sha256(raw_path)
            else:
                if self.recognizer is None:
                    raise BatchRunError(case_id, "raw", "recognition adapter is not configured; no result fabricated")
                raw = self._call_with_timeout(self.recognizer, case, {}, raw_dir, "raw")
                if not isinstance(raw.get("notes"), list) or not isinstance(raw.get("beat_grid"), Mapping):
                    raise BatchRunError(case_id, "raw", "recognizer must return notes[] and beat_grid object")
                raw = _annotate_raw_provenance(raw, case=case, recognizer_provenance=recognizer_provenance)
                raw_hash = _write_json_once(raw_path, raw)
            effective_scope = _effective_evaluation_scope(case, raw)
            state["evaluation_scope"] = effective_scope
            beat_hash = _write_json_once(raw_dir / "beat_grid.json", raw.get("beat_grid", {}))
            state["raw"] = {
                "recognition": "raw/recognition.json",
                "recognition_sha256": raw_hash,
                "beat_grid": "raw/beat_grid.json",
                "beat_grid_sha256": beat_hash,
                "immutable": True,
                "model_output": raw.get("model_output") is True,
                "evaluation_scope": effective_scope,
                "case_evaluation_scope": case.get("evaluation_scope"),
                "recognizer_mode": recognizer_provenance["mode"],
                "recognizer_fingerprint": recognizer_provenance["fingerprint"],
            }
            if _sha256(raw_path) != raw_hash:
                raise BatchRunError(case_id, "raw", "raw recognition changed during pipeline")
            if self.raw_only:
                state["status"] = "success"
                state["raw_only"] = True
                state["finished_at"] = time.time()
                _write_json_once(manifest_path, state) if not manifest_path.exists() else manifest_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return state
            for name, adapter, pipeline_root in (
                ("baseline", self.baseline, baseline_root),
                ("new", self.new_chain, new_root),
            ):
                destination = _safe_case_dir(pipeline_root, case_id)
                pipeline_manifest = destination / "manifest.json"
                if resume and pipeline_manifest.is_file():
                    existing = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
                    if (
                        isinstance(existing, Mapping)
                        and existing.get("status") == "success"
                        and existing.get("raw_recognition_sha256") == raw_hash
                        and existing.get("recognizer_mode") == recognizer_provenance["mode"]
                        and existing.get("recognizer_fingerprint") == recognizer_provenance["fingerprint"]
                        and existing.get("effective_evaluation_scope") == effective_scope
                    ):
                        state["pipelines"][name] = existing
                        continue
                if adapter is None:
                    failure = {
                        "schema_version": RUNNER_SCHEMA_VERSION,
                        "case_id": case_id,
                        "pipeline": name,
                        "status": "failed",
                        "stage": name,
                        "error": "adapter is not configured; no result fabricated",
                        "evaluation_scope": effective_scope,
                        "effective_evaluation_scope": effective_scope,
                        "case_evaluation_scope": case.get("evaluation_scope"),
                        "raw_model_output": raw.get("model_output") is True,
                        "raw_recognition_sha256": raw_hash,
                        "recognizer_mode": recognizer_provenance["mode"],
                        "recognizer_fingerprint": recognizer_provenance["fingerprint"],
                    }
                    destination.mkdir(parents=True, exist_ok=True)
                    _replace_json(destination / "manifest.json", failure)
                    state["pipelines"][name] = failure
                    continue
                result = self._call_with_timeout(adapter, case, raw, destination, name)
                pipeline_state = {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "case_id": case_id,
                    "pipeline": name,
                    "status": "success",
                    "stage": name,
                    "evaluation_scope": effective_scope,
                    "effective_evaluation_scope": effective_scope,
                    "case_evaluation_scope": case.get("evaluation_scope"),
                    "source_kind": case.get("source_kind"),
                    "raw_model_output": raw.get("model_output") is True,
                    "raw_recognition_sha256": raw_hash,
                    "recognizer_mode": recognizer_provenance["mode"],
                    "recognizer_fingerprint": recognizer_provenance["fingerprint"],
                    "result": result,
                    "final_midi": result.get("final_midi"),
                    "beat_grid": result.get("beat_grid"),
                    "manifest_path": str(pipeline_manifest),
                }
                destination.mkdir(parents=True, exist_ok=True)
                pipeline_state["manifest_sha256"] = _replace_json(pipeline_manifest, pipeline_state)
                state["pipelines"][name] = pipeline_state
            pipeline_statuses = [item.get("status") for item in state["pipelines"].values()]
            if pipeline_statuses and all(status == "success" for status in pipeline_statuses):
                state["status"] = "success"
            elif any(status == "success" for status in pipeline_statuses):
                state["status"] = "partial"
                state["error"] = {"stage": "pipelines", "message": "one pipeline succeeded but the other is missing or failed"}
            else:
                state["status"] = "failed"
                state["error"] = {"stage": "pipelines", "message": "baseline and new adapters are both unavailable or failed"}
        except BatchRunError as exc:
            state.update({"status": "failed", "error": {"stage": exc.stage, "message": exc.cause}})
        except Exception as exc:
            state.update({"status": "failed", "error": {"stage": "runner", "message": f"{type(exc).__name__}: {exc}"}})
        state["finished_at"] = time.time()
        _write_json_once(manifest_path, state) if not manifest_path.exists() else manifest_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return state


def _load_registry(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"invalid benchmark registry: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-result-root", type=Path, help="baseline pipeline manifests/artifacts root")
    parser.add_argument("--new-result-root", type=Path, help="new pipeline manifests/artifacts root")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    recognition = parser.add_mutually_exclusive_group()
    recognition.add_argument("--reference-isolation", action="store_true", help="显式使用参考 MIDI 准备量化器隔离 raw；不代表模型识别")
    recognition.add_argument("--production-recognizer", action="store_true", help="显式运行可终止的 MuScriptor/Demucs/GAME/BeatNet production worker")
    parser.add_argument("--run-legacy-baseline", action="store_true", help="在已准备的 raw 上运行保留的旧均匀网格 baseline")
    parser.add_argument("--run-new-chain", action="store_true", help="在已准备的 raw 上运行当前高精度服务")
    parser.add_argument("--raw-only", action="store_true", help="只运行并登记 recognizer raw，不创建或运行 baseline/new score pipeline")
    parser.add_argument("--demucs-model", choices=("htdemucs", "htdemucs_ft"), help="vocal production route 的 Demucs model")
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.raw_only and (args.run_legacy_baseline or args.run_new_chain):
        parser.error("--raw-only cannot be combined with --run-legacy-baseline or --run-new-chain")
    registry = _load_registry(args.manifest.resolve())
    cases = [case for case in registry["cases"] if not args.case_ids or str(case.get("id")) in set(args.case_ids)]
    if not cases:
        raise SystemExit("no benchmark cases selected")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be greater than zero")
    recognizer: Adapter | None = None
    if args.reference_isolation:
        recognizer = ReferenceIsolationRecognizer()
    elif args.production_recognizer or args.raw_only or args.run_legacy_baseline or args.run_new_chain:
        recognizer = ProductionRecognizer(timeout_sec=args.timeout_sec, demucs_model=args.demucs_model)
    runner = BenchmarkBatchRunner(
        recognizer=recognizer,
        baseline=legacy_baseline_adapter if args.run_legacy_baseline else None,
        new_chain=high_accuracy_service_adapter if args.run_new_chain else None,
        timeout_sec=args.timeout_sec,
        raw_only=args.raw_only,
    )
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    baseline_root = args.baseline_result_root.resolve() if args.baseline_result_root else None
    new_root = args.new_result_root.resolve() if args.new_result_root else None
    outcomes = [
        runner.run_case(
            case,
            result_root=result_root,
            baseline_result_root=baseline_root,
            new_result_root=new_root,
            resume=not args.no_resume,
        )
        for case in cases
    ]
    print(
        json.dumps(
            {
                "result_root": str(result_root),
                "baseline_result_root": str(baseline_root or result_root / "baseline"),
                "new_result_root": str(new_root or result_root / "new"),
                "recognizer": "reference-isolation" if args.reference_isolation else "production" if recognizer is not None else None,
                "raw_only": args.raw_only,
                "cases": [{"id": item["case_id"], "status": item["status"], "error": item.get("error")} for item in outcomes],
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(item["status"] == "success" for item in outcomes) else 2


if __name__ == "__main__":
    sys.exit(main())
