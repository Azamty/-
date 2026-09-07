"""Resumable orchestration for the 30-case high-accuracy benchmark.

Recognition is an explicit dependency.  The runner calls it exactly once per
case, stores its raw notes and beat grid immutably, and passes independent
copies of that same payload to the legacy baseline and the new service.  A
missing adapter is a recorded failure, never a fabricated pass.  The
``--reference-isolation`` mode is intentionally marked as such and only
prepares a deterministic quantizer-isolation input from a reference MIDI.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / ".cache" / "high-accuracy-benchmarks" / "runs"
RUNNER_SCHEMA_VERSION = "1.0"

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
    return {
        "schema_version": "1.0",
        "source": "reference_midi_quantizer_isolation",
        "model_output": False,
        "reference_midi": str(path),
        "reference_sha256": _sha256(path),
        "notes": notes,
        "beat_grid": beat_grid,
        "provenance": {"evaluation_scope": "quantizer_isolation", "reference_is_not_model_output": True},
    }


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
    notes: list[NoteEvent] = []
    max_end = 0.0
    for index, item in enumerate(raw.get("notes", [])):
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
        notes.append(NoteEvent(start_sec=start_sec, end_sec=end_sec, midi=int(item["midi"]), confidence=item.get("confidence"), velocity=item.get("velocity"), raw_pitch=item.get("raw_pitch"), voice_id=str(item.get("voice_id", "voice-0")), source=str(item.get("source", "benchmark-raw")), metadata={"raw_index": index}))
    duration_sec = max(float(analysis_payload.get("duration_sec") or 0.0), max_end, (beat_times[-1] if beat_times else 0.0) + 0.1, 0.1)
    sample_rate = int(analysis_payload.get("sample_rate") or raw.get("sample_rate") or 44_100)
    metadata = dict(analysis_payload.get("metadata") or {})
    metadata.setdefault("beat_source", "beatnet")
    metadata["beat_grid"] = beat_grid
    analysis = MusicAnalysis(sample_rate=sample_rate, duration_sec=duration_sec, bpm=bpm, time_signature=meter, key=key, beat_times=beat_times, note_events=notes, warnings=list(analysis_payload.get("warnings") or []), metadata=metadata)
    return analysis, notes


def legacy_baseline_adapter(case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
    """Run the preserved uniform-grid quantizer for baseline comparison only."""

    from backend.jianpu_score.quantize import quantize_events
    from backend.jianpu_score.render import render_score

    analysis, events = _analysis_and_events_from_raw(raw, case)
    destination.mkdir(parents=True, exist_ok=True)
    score = quantize_events(events, analysis, mode="polyphonic", title=str(case.get("title") or case["id"]))
    score_path = destination / "baseline.score.json"
    score_path.write_text(json.dumps(score.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rendered = render_score(score, destination / "render", basename="baseline")
    return {"engine": "legacy-uniform-grid", "score_json": str(score_path.relative_to(destination)), "render": rendered.model_dump(mode="json")}


def high_accuracy_service_adapter(case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path) -> Mapping[str, Any]:
    """Run the production high-accuracy service on the shared raw payload."""

    from backend.jianpu_score.high_accuracy_service import build_high_accuracy_artifacts

    analysis, events = _analysis_and_events_from_raw(raw, case)
    service_output = destination / "service_output"
    result = build_high_accuracy_artifacts(instrument_id=str(case["id"]), title=str(case.get("title") or case["id"]), program=int(case.get("program", 0)), is_drum=False, events=events, analysis=analysis, output_dir=service_output, variant="benchmark-new", overwrite=False)
    return {"engine": "musescore-midi-import", "manifest": str(result.manifest_path.relative_to(destination)), "status": result.status, "jianpu_status": result.jianpu_status, "artifacts": [artifact.as_dict() for artifact in result.artifacts]}


class BenchmarkBatchRunner:
    """Run one registry through shared recognition, baseline and new adapters."""

    def __init__(
        self,
        *,
        recognizer: Adapter | None,
        baseline: Adapter | None,
        new_chain: Adapter | None,
        timeout_sec: float = 1800.0,
    ) -> None:
        self.recognizer = recognizer
        self.baseline = baseline
        self.new_chain = new_chain
        self.timeout_sec = float(timeout_sec)

    def _call_with_timeout(self, adapter: Adapter, case: Mapping[str, Any], raw: Mapping[str, Any], destination: Path, stage: str) -> Mapping[str, Any]:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"benchmark-{stage}")
        future: Future[Mapping[str, Any]] = executor.submit(adapter, case, copy.deepcopy(raw), destination)
        try:
            result = future.result(timeout=self.timeout_sec)
        except FutureTimeout as exc:
            future.cancel()
            raise BatchRunError(str(case["id"]), stage, f"timeout after {self.timeout_sec:g}s") from exc
        except BatchRunError:
            raise
        except Exception as exc:
            raise BatchRunError(str(case["id"]), stage, f"{type(exc).__name__}: {exc}") from exc
        finally:
            # A timed out worker may still be unwinding, but it cannot block
            # the resumable manifest writer or be mistaken for a successful
            # second pipeline.
            executor.shutdown(wait=False, cancel_futures=True)
        if not isinstance(result, Mapping):
            raise BatchRunError(str(case["id"]), stage, "adapter must return a JSON object")
        return result

    def run_case(self, case: Mapping[str, Any], *, result_root: Path, resume: bool = True) -> dict[str, Any]:
        case_id = str(case["id"])
        case_dir = _safe_case_dir(result_root, case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = case_dir / "manifest.json"
        state: dict[str, Any] = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "case_id": case_id,
            "status": "running",
            "evaluation_scope": case.get("evaluation_scope"),
            "raw": None,
            "pipelines": {},
            "error": None,
        }
        raw_dir = case_dir / "raw"
        raw_path = raw_dir / "recognition.json"
        try:
            if raw_path.is_file():
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping) or not isinstance(raw.get("notes"), list):
                    raise BatchRunError(case_id, "raw", "existing recognition.json is invalid")
                raw_hash = _sha256(raw_path)
            else:
                if self.recognizer is None:
                    raise BatchRunError(case_id, "raw", "recognition adapter is not configured; no result fabricated")
                raw = self._call_with_timeout(self.recognizer, case, {}, raw_dir, "raw")
                if not isinstance(raw.get("notes"), list) or not isinstance(raw.get("beat_grid"), Mapping):
                    raise BatchRunError(case_id, "raw", "recognizer must return notes[] and beat_grid object")
                raw_hash = _write_json_once(raw_path, raw)
            beat_hash = _write_json_once(raw_dir / "beat_grid.json", raw.get("beat_grid", {}))
            state["raw"] = {"recognition": "raw/recognition.json", "recognition_sha256": raw_hash, "beat_grid": "raw/beat_grid.json", "beat_grid_sha256": beat_hash, "immutable": True, "model_output": raw.get("model_output") is not False}
            if _sha256(raw_path) != raw_hash:
                raise BatchRunError(case_id, "raw", "raw recognition changed during pipeline")
            for name, adapter in (("baseline", self.baseline), ("new", self.new_chain)):
                destination = case_dir / name
                pipeline_manifest = destination / "manifest.json"
                if resume and pipeline_manifest.is_file():
                    existing = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
                    if isinstance(existing, Mapping) and existing.get("status") == "success":
                        state["pipelines"][name] = existing
                        continue
                if adapter is None:
                    failure = {"status": "failed", "stage": name, "error": "adapter is not configured; no result fabricated"}
                    destination.mkdir(parents=True, exist_ok=True)
                    _replace_json(destination / "manifest.json", failure)
                    state["pipelines"][name] = failure
                    continue
                result = self._call_with_timeout(adapter, case, raw, destination, name)
                pipeline_state = {"status": "success", "stage": name, "raw_recognition_sha256": raw_hash, "result": result}
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
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--reference-isolation", action="store_true", help="显式使用参考 MIDI 准备量化器隔离 raw；不代表模型识别")
    parser.add_argument("--run-legacy-baseline", action="store_true", help="在已准备的 raw 上运行保留的旧均匀网格 baseline")
    parser.add_argument("--run-new-chain", action="store_true", help="在已准备的 raw 上运行当前高精度服务")
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    registry = _load_registry(args.manifest.resolve())
    cases = [case for case in registry["cases"] if not args.case_ids or str(case.get("id")) in set(args.case_ids)]
    if not cases:
        raise SystemExit("no benchmark cases selected")
    recognizer: Adapter | None = None
    if args.reference_isolation:
        def reference_recognizer(case: Mapping[str, Any], _raw: Mapping[str, Any], _destination: Path) -> Mapping[str, Any]:
            return _reference_isolation_payload(case, root=ROOT)

        recognizer = reference_recognizer
    runner = BenchmarkBatchRunner(
        recognizer=recognizer,
        baseline=legacy_baseline_adapter if args.run_legacy_baseline else None,
        new_chain=high_accuracy_service_adapter if args.run_new_chain else None,
        timeout_sec=args.timeout_sec,
    )
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    outcomes = [runner.run_case(case, result_root=result_root, resume=not args.no_resume) for case in cases]
    print(json.dumps({"result_root": str(result_root), "cases": [{"id": item["case_id"], "status": item["status"], "error": item.get("error")} for item in outcomes]}, ensure_ascii=False))
    return 0 if all(item["status"] == "success" for item in outcomes) else 2


if __name__ == "__main__":
    sys.exit(main())
