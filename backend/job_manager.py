"""Persistent single worker for local audio transcription jobs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
from typing import Any, Callable
import uuid
import zipfile

from .jianpu_score.models.adapter import EngineUnavailableError
from .jianpu_score.pipeline import run_pipeline
from .jianpu_score.quantize import NoNotesError
from .jianpu_score.render import RenderArtifacts, render_score
from .v2_job_manager import V2JobService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = ROOT / "artifacts" / "jobs"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
CLEANUP_INTERVAL_SECONDS = 180.0
TERMINAL_STATUSES = frozenset({"completed", "selection_ready", "failed", "interrupted"})
PHASE_LABELS = {
    "uploading": "上传中",
    "queued": "排队中",
    "probing": "检查音频",
    "separating": "分离声部",
    "recognizing": "识别音符",
    "quantizing": "整理节拍",
    "rendering": "生成谱面",
    "packaging": "整理下载文件",
    "completed": "已完成",
    "selection_ready": "识别完成，等待选择",
    "exporting": "导出选择结果",
    "failed": "处理失败",
    "interrupted": "服务重启，中断待重试",
}
PHASE_ORDER = (
    "uploading",
    "queued",
    "probing",
    "separating",
    "recognizing",
    "quantizing",
    "rendering",
    "exporting",
    "packaging",
    "selection_ready",
    "completed",
)
STEM_LABELS = {"vocals": "人声", "bass": "低音", "other": "器乐", "mixed": "原音"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str, fallback: str = "audio") -> str:
    """Keep a user-visible name printable without using it as a filesystem path."""

    text = str(value or "").replace("\x00", "")
    text = "".join(char if char.isprintable() and char not in "\\/:*?\"<>|" else " " for char in text)
    text = " ".join(text.split()).strip(" .")
    return text or fallback


def _error_payload(exc: Exception) -> dict[str, str]:
    if isinstance(exc, NoNotesError):
        return {
            "code": "no_notes",
            "message": "没有识别到可记谱音符，请换一段有清晰旋律的音频或调整来源选项。",
        }
    if isinstance(exc, EngineUnavailableError):
        return {"code": "engine_unavailable", "message": str(exc)}
    if isinstance(exc, ValueError):
        return {"code": "invalid_audio", "message": str(exc)}
    return {"code": "processing_error", "message": f"处理失败：{exc}"}


class JobManager:
    """Own a durable queue and exactly one worker thread.

    The persisted JSON is the source of truth.  A process restart marks any
    running job as interrupted and leaves it available for an explicit retry.
    """

    def __init__(self, root: str | Path = DEFAULT_JOBS_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._current_job_id: str | None = None
        self._child_process: subprocess.Popen[bytes] | None = None
        self._pending_after_recovery: list[str] = []
        self.v2 = V2JobService(self)
        self._recover_running_jobs()
        self._cleanup_safely()

    def _cleanup_safely(self) -> list[str]:
        """Run retention maintenance without taking down the worker/server."""

        try:
            return self.cleanup_expired()
        except Exception:
            # A locked or concurrently changed old job must not stop the
            # single transcription worker.  The next maintenance tick retries.
            return []

    def _safe_job_dir(self, job_id: str) -> Path:
        if not JOB_ID_PATTERN.fullmatch(job_id.lower()):
            raise KeyError("invalid job id")
        candidate = (self.root / job_id).resolve()
        if candidate.parent != self.root:
            raise KeyError("invalid job path")
        return candidate

    def _job_json(self, job_id: str) -> Path:
        return self._safe_job_dir(job_id) / "job.json"

    def _read(self, job_id: str) -> dict[str, Any]:
        path = self._job_json(job_id)
        if not path.is_file():
            raise KeyError("job not found")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KeyError(f"job metadata is unreadable: {job_id}") from exc
        if not isinstance(value, dict) or value.get("id") != job_id:
            raise KeyError("job metadata is invalid")
        return value

    def _write(self, state: dict[str, Any]) -> None:
        directory = self._safe_job_dir(str(state["id"]))
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "job.json"
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)

    def _update(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read(job_id)
            state.update(updates)
            state["updated_at"] = utc_now()
            self._write(state)
            return state

    def _log(self, job_id: str, message: str) -> None:
        try:
            directory = self._safe_job_dir(job_id)
            with (directory / "job.log").open("a", encoding="utf-8") as handle:
                handle.write(f"[{utc_now()}] {message}\n")
        except (OSError, KeyError):
            # Logging must never terminate the worker or mask a job failure.
            return

    def _recover_running_jobs(self) -> None:
        now = utc_now()
        with self._lock:
            for directory in self.root.iterdir():
                if not directory.is_dir() or directory.is_symlink() or not JOB_ID_PATTERN.fullmatch(directory.name.lower()):
                    continue
                metadata = directory / "job.json"
                if not metadata.is_file():
                    continue
                try:
                    state = json.loads(metadata.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(state, dict) or state.get("id") != directory.name:
                    continue
                if state.get("status") == "running":
                    state.update(
                        {
                            "status": "interrupted",
                            "phase": "interrupted",
                            "finished_at": now,
                            "error": {
                                "code": "interrupted",
                                "message": "服务重启时任务被中断，请点击重试。",
                            },
                            "updated_at": now,
                        }
                    )
                    self._write(state)
                    self._log(directory.name, "recovered running job as interrupted")
                elif state.get("status") == "uploading":
                    state.update(
                        {
                            "status": "interrupted",
                            "phase": "interrupted",
                            "finished_at": now,
                            "error": {
                                "code": "upload_interrupted",
                                "message": "服务重启时上传尚未完成，文件不完整，请重新上传。",
                            },
                            "updated_at": now,
                        }
                    )
                    self._write(state)
                    self._log(directory.name, "recovered incomplete upload as interrupted")
                elif state.get("status") == "queued":
                    self._pending_after_recovery.append(directory.name)

    def cleanup_expired(self, *, now: datetime | None = None, max_age: timedelta = timedelta(hours=24)) -> list[str]:
        """Remove only terminal job directories older than the retention window."""

        current = now or datetime.now(timezone.utc)
        removed: list[str] = []
        root = self.root.resolve()
        with self._lock:
            for directory in list(root.iterdir()):
                if not directory.is_dir() or directory.is_symlink() or directory.resolve().parent != root:
                    continue
                if not JOB_ID_PATTERN.fullmatch(directory.name.lower()):
                    continue
                metadata = directory / "job.json"
                try:
                    state = json.loads(metadata.read_text(encoding="utf-8"))
                    finished = datetime.fromisoformat(str(state.get("finished_at", "")))
                except (OSError, json.JSONDecodeError, ValueError, TypeError):
                    continue
                if state.get("status") not in TERMINAL_STATUSES or current - finished <= max_age:
                    continue
                # Re-check the resolved target immediately before removal.
                target = directory.resolve()
                if target.parent != root:
                    continue
                shutil.rmtree(target)
                removed.append(directory.name)
        return removed

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            for job_id in self._pending_after_recovery:
                self._queue.put(job_id)
            self._pending_after_recovery.clear()
            self._worker = threading.Thread(target=self._worker_loop, name="jianpu-job-worker", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._stop.set()
        child = self._child_process
        if child is not None and child.poll() is None:
            self._terminate_child(child)
        self._queue.put(None)
        worker.join(timeout=5)
        self._worker = None

    @staticmethod
    def _terminate_child(process: subprocess.Popen[Any]) -> None:
        """Terminate an isolated model process and its descendants."""

        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
            else:
                process.terminate()
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass

    def _run_isolated_child(self, command: list[str], job_id: str, progress_path: Path | None = None) -> int:
        """Run one model child while allowing ``stop`` to terminate it."""

        directory = self._safe_job_dir(job_id)
        output = directory / "output"
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "v2-recognition" / "muscriptor-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        handle = log_path.open("ab")
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            self._child_process = process
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                if progress_path is not None and progress_path.is_file():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                        total = int(progress.get("total", 0))
                        completed = int(progress.get("completed", 0))
                        ratio = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
                        with self._lock:
                            state = self._read(job_id)
                            state["progress"] = ratio
                            state.setdefault("v2", {})["progress_detail"] = progress
                            state["updated_at"] = utc_now()
                            self._write(state)
                    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                        pass
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue
            return int(return_code)
        finally:
            handle.close()
            if process is not None and process.poll() is None:
                self._terminate_child(process)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            self._child_process = None

    def create_job(self, *, original_name: str, options: dict[str, Any]) -> tuple[str, Path]:
        job_id = str(uuid.uuid4())
        directory = self._safe_job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=False)
        suffix = Path(original_name).suffix.lower()
        input_path = directory / f"input{suffix}"
        now = utc_now()
        state: dict[str, Any] = {
            "schema_version": "1.0",
            "id": job_id,
            "status": "uploading",
            "phase": "uploading",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "attempt": 1,
            "options": deepcopy(options),
            "input": {
                "original_name": safe_filename(original_name),
                "stored_name": input_path.name,
                "bytes": None,
                "duration_sec": None,
            },
            "error": None,
            "warnings": [],
            "artifacts": [],
        }
        self._write(state)
        self._log(job_id, f"created: {state['input']['original_name']}")
        return job_id, input_path

    def create_v2_job(self, *, original_name: str, source_kind: str, title: str) -> tuple[str, Path]:
        """Create a V2 task while keeping its queue on this manager's worker."""

        return self.v2.create_job(original_name=original_name, source_kind=source_kind, title=title)

    def select_v2(self, job_id: str, selected_track_ids: list[str], merge_main_melody: bool = False) -> dict[str, Any]:
        return self.v2.select(job_id, selected_track_ids, merge_main_melody)

    def v2_tracks(self, job_id: str) -> dict[str, Any]:
        return self.v2.tracks(job_id)

    def input_path(self, job_id: str) -> Path:
        state = self._read(job_id)
        stored_name = str(state.get("input", {}).get("stored_name", ""))
        directory = self._safe_job_dir(job_id)
        path = directory / stored_name
        if path.parent != directory or path.name != stored_name or path.is_symlink() or path.resolve().parent != directory:
            raise KeyError("invalid input path")
        return path

    def set_input_info(self, job_id: str, *, bytes_count: int, duration_sec: float) -> None:
        with self._lock:
            state = self._read(job_id)
            state["input"] = {
                **state.get("input", {}),
                "bytes": int(bytes_count),
                "duration_sec": float(duration_sec),
            }
            state["updated_at"] = utc_now()
            self._write(state)

    def fail_immediate(self, job_id: str, error: dict[str, str]) -> None:
        self._update(
            job_id,
            status="failed",
            phase="failed",
            finished_at=utc_now(),
            error=error,
        )
        self._log(job_id, f"failed before queue: {error.get('code')}: {error.get('message')}")

    def enqueue(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._read(job_id)
            if state.get("status") != "uploading":
                raise ValueError("only complete uploads can be enqueued")
            state.update({"status": "queued", "phase": "queued", "updated_at": utc_now()})
            self._write(state)
            self._queue.put(job_id)
            self._log(job_id, "queued for the single worker")
            return state

    def retry(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._read(job_id)
            if state.get("status") not in {"failed", "interrupted"}:
                raise ValueError("only failed or interrupted jobs can be retried")
            error = state.get("error") or {}
            if error.get("code") == "upload_interrupted":
                raise ValueError("上传未完成且文件不完整，请重新上传")
            directory = self._safe_job_dir(job_id)
            output = directory / "output"
            retained_artifacts: list[dict[str, Any]] = []
            if output.is_symlink():
                raise ValueError("invalid job output path")
            if output.exists():
                if output.resolve().parent != directory:
                    raise ValueError("invalid job output path")
                if state.get("kind") == "v2" and state.get("v2", {}).get("stage") == "export":
                    selections = output / "selections"
                    if selections.is_symlink() or (selections.exists() and selections.resolve().parent != output):
                        raise ValueError("invalid V2 selection output path")
                    if selections.exists():
                        shutil.rmtree(selections)
                    state["artifacts"] = [
                        item for item in state.get("artifacts", [])
                        if not str(item.get("artifact_id", "")).startswith("v2-selection-")
                    ]
                    retained_artifacts = list(state["artifacts"])
                else:
                    shutil.rmtree(output)
            now = utc_now()
            state.update(
                {
                    "status": "queued",
                    "phase": "queued",
                    "started_at": None,
                    "finished_at": None,
                    "updated_at": now,
                    "attempt": int(state.get("attempt", 1)) + 1,
                    "error": None,
                    "warnings": [],
                    "artifacts": retained_artifacts,
                }
            )
            self._write(state)
            self._log(job_id, f"retry attempt {state['attempt']}")
            self._queue.put(job_id)
            return state

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public(self._read(job_id))

    def _public(self, state: dict[str, Any]) -> dict[str, Any]:
        public = deepcopy(state)
        phase = str(public.get("phase", public.get("status", "queued")))
        public["phase_label"] = PHASE_LABELS.get(phase, phase)
        public["phase_index"] = PHASE_ORDER.index(phase) if phase in PHASE_ORDER else None
        # Internal relative paths never become part of the user-facing API.
        public.pop("internal", None)
        public["progress"] = public.get("progress")
        error = public.get("error") or {}
        public["retryable"] = public.get("status") in {"failed", "interrupted"} and error.get("code") != "upload_interrupted"
        public["score_available"] = public.get("status") == "completed" and not bool(
            (public.get("v2") or {}).get("score_refusal")
        )
        public["artifacts_available"] = public.get("status") in {"selection_ready", "completed"} and bool(public.get("artifacts"))
        return public

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            state = self._read(job_id)
            return deepcopy(state.get("artifacts", []))

    def artifact_path(self, job_id: str, artifact_id: str) -> tuple[Path, dict[str, Any]]:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", artifact_id):
            raise KeyError("invalid artifact id")
        with self._lock:
            state = self._read(job_id)
            artifact = next((item for item in state.get("artifacts", []) if item.get("artifact_id") == artifact_id), None)
            if artifact is None:
                raise KeyError("artifact not found")
            relative = Path(str(artifact.get("relative_path", "")))
            directory = self._safe_job_dir(job_id)
            path = (directory / relative).resolve()
            if path == directory or directory not in path.parents or not path.is_file():
                raise KeyError("artifact path is invalid")
            return path, deepcopy(artifact)

    def _set_phase(self, job_id: str, phase: str) -> None:
        with self._lock:
            state = self._read(job_id)
            if state.get("status") != "running":
                return
            state["phase"] = phase
            state["updated_at"] = utc_now()
            self._write(state)
        self._log(job_id, f"phase={phase}")

    def _worker_loop(self) -> None:
        next_cleanup = time.monotonic() + CLEANUP_INTERVAL_SECONDS
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                if time.monotonic() >= next_cleanup:
                    self._cleanup_safely()
                    next_cleanup = time.monotonic() + CLEANUP_INTERVAL_SECONDS
                continue
            if job_id is None:
                self._queue.task_done()
                break
            self._current_job_id = job_id
            try:
                self._run_job(job_id)
            finally:
                self._current_job_id = None
                self._queue.task_done()
                self._cleanup_safely()

    def _run_job(self, job_id: str) -> None:
        try:
            with self._lock:
                state = self._read(job_id)
                if state.get("status") != "queued":
                    return
                now = utc_now()
                is_v2 = state.get("kind") == "v2"
                v2_stage = str(state.get("v2", {}).get("stage", "")) if is_v2 else ""
                initial_phase = "rendering" if is_v2 and v2_stage == "export" else "probing"
                state.update({"status": "running", "phase": initial_phase, "started_at": now, "finished_at": None, "error": None})
                self._write(state)
            self._log(job_id, "worker started")

            if is_v2:
                self.v2.run(job_id)
                return

            state = self._read(job_id)
            options = dict(state.get("options", {}))
            input_path = self.input_path(job_id)
            output_dir = self._safe_job_dir(job_id) / "output"
            if output_dir.is_symlink() or (output_dir.exists() and output_dir.resolve().parent != self._safe_job_dir(job_id)):
                raise ValueError("invalid job output path")
            output_dir.mkdir(parents=True, exist_ok=True)

            def progress(phase: str) -> None:
                self._set_phase(job_id, phase)

            analysis, score, render_artifacts = run_pipeline(
                input_path,
                output_dir,
                engine=str(options.get("engine", "basic-pitch")),
                voice_mode=str(options.get("voice_mode", "monophonic")),
                source_kind=str(options.get("source_kind", "mixed")),
                separate=bool(options.get("separate", False)),
                bpm_override=options.get("bpm_override"),
                key_override=options.get("key_override"),
                time_signature_override=options.get("time_signature_override"),
                language=str(options.get("language", "mixed")),
                title=options.get("title"),
                progress_callback=progress,
            )
            self._set_phase(job_id, "packaging")
            artifacts = self._package_artifacts(job_id, score, analysis, render_artifacts)
            warnings = list(dict.fromkeys([*analysis.warnings, *score.warnings]))
            summary = {
                "note_count": len(analysis.note_events),
                "voice_count": len(score.voices),
                "total_ticks": score.total_ticks,
                "bpm": score.bpm,
                "key": score.key,
                "time_signature": score.time_signature,
            }
            finished = utc_now()
            self._update(
                job_id,
                status="completed",
                phase="completed",
                finished_at=finished,
                error=None,
                warnings=warnings,
                artifacts=artifacts,
                summary=summary,
            )
            self._log(job_id, f"completed: {len(analysis.note_events)} notes, {len(artifacts)} artifacts")
        except Exception as exc:  # Keep one bad job from stopping the queue.
            if self._stop.is_set():
                error = {"code": "interrupted", "message": "服务停止时任务被中断，请重启服务后重试。"}
                failure_phase = "interrupted"
                failure_status = "interrupted"
            else:
                error = _error_payload(exc)
                failure_phase = "failed"
                failure_status = "failed"
            self._log(job_id, traceback.format_exc())
            try:
                self._update(job_id, status=failure_status, phase=failure_phase, finished_at=utc_now(), error=error)
            except Exception:
                return

    def _relative_artifact(self, job_dir: Path, path: Path) -> str:
        resolved_job = job_dir.resolve()
        resolved_path = path.resolve()
        if resolved_path == resolved_job or resolved_job not in resolved_path.parents or not resolved_path.is_file():
            raise ValueError(f"artifact is outside job directory: {path}")
        return resolved_path.relative_to(resolved_job).as_posix()

    def _register(
        self,
        job_dir: Path,
        path: Path,
        *,
        artifact_id: str,
        kind: str,
        label: str,
        media_type: str,
        stem_id: str | None = None,
        page: int | None = None,
    ) -> dict[str, Any]:
        relative = self._relative_artifact(job_dir, path)
        return {
            "artifact_id": artifact_id,
            "kind": kind,
            "label": label,
            "filename": path.name,
            "relative_path": relative,
            "media_type": media_type,
            "size_bytes": path.stat().st_size,
            "stem_id": stem_id,
            "page": page,
        }

    def _package_artifacts(
        self,
        job_id: str,
        score: Any,
        analysis: Any,
        render_artifacts: RenderArtifacts,
    ) -> list[dict[str, Any]]:
        job_dir = self._safe_job_dir(job_id)
        output = job_dir / "output"
        artifacts: list[dict[str, Any]] = []
        svg_paths: list[Path] = [Path(path).resolve() for path in render_artifacts.svg_paths]
        for index, path in enumerate(svg_paths, start=1):
            artifacts.append(
                self._register(
                    job_dir,
                    path,
                    artifact_id=f"score-svg-{index}",
                    kind="score_svg",
                    label=f"总谱第 {index} 页",
                    media_type="image/svg+xml",
                    page=index,
                )
            )
        if render_artifacts.midi_path:
            artifacts.append(
                self._register(
                    job_dir,
                    Path(render_artifacts.midi_path).resolve(),
                    artifact_id="score-midi",
                    kind="midi",
                    label="总谱 MIDI",
                    media_type="audio/midi",
                )
            )
        for path, artifact_id, kind, label, media_type in (
            (output / "score.json", "score-json", "score_json", "Score 数据", "application/json"),
            (output / "analysis.json", "analysis-json", "analysis_json", "分析数据", "application/json"),
            (output / "score.jly", "score-jly", "jianpu_source", "简谱源文本", "text/plain; charset=utf-8"),
            (output / "score.ly", "score-lilypond", "lilypond_source", "LilyPond 源文本", "text/plain; charset=utf-8"),
        ):
            if path.is_file():
                artifacts.append(self._register(job_dir, path, artifact_id=artifact_id, kind=kind, label=label, media_type=media_type))

        # Render one independently registered score for each source stem.  The
        # voices are already timeline-complete, so this keeps total and split
        # score timing identical without inventing another quantizer.
        stem_svg_paths: list[Path] = []
        for stem_id in sorted({voice.stem_id for voice in score.voices if voice.stem_id}):
            stem_voices = [voice for voice in score.voices if voice.stem_id == stem_id]
            if not stem_voices:
                continue
            stem_label = STEM_LABELS.get(str(stem_id), str(stem_id))
            stem_dir = output / "stems" / re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem_id)).strip("._")
            stem_dir.mkdir(parents=True, exist_ok=True)
            stem_score = score.model_copy(update={"voices": stem_voices, "source": str(stem_id)})
            stem_artifacts = render_score(stem_score, stem_dir, basename="score")
            for index, path_text in enumerate(stem_artifacts.svg_paths, start=1):
                path = Path(path_text).resolve()
                stem_svg_paths.append(path)
                artifacts.append(
                    self._register(
                        job_dir,
                        path,
                        artifact_id=f"stem-{stem_id}-svg-{index}",
                        kind="stem_svg",
                        label=f"{stem_label} 分谱第 {index} 页",
                        media_type="image/svg+xml",
                        stem_id=str(stem_id),
                        page=index,
                    )
                )
            if stem_artifacts.midi_path:
                artifacts.append(
                    self._register(
                        job_dir,
                        Path(stem_artifacts.midi_path).resolve(),
                        artifact_id=f"stem-{stem_id}-midi",
                        kind="stem_midi",
                        label=f"{stem_label} 分谱 MIDI",
                        media_type="audio/midi",
                        stem_id=str(stem_id),
                    )
                )

        all_svg = svg_paths + stem_svg_paths
        zip_path = output / "score-svg.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in all_svg:
                archive.write(path, arcname=path.relative_to(output).as_posix())
        artifacts.append(
            self._register(
                job_dir,
                zip_path,
                artifact_id="score-svg-zip",
                kind="svg_zip",
                label="全部 SVG 压缩包",
                media_type="application/zip",
            )
        )
        log_path = job_dir / "job.log"
        if log_path.is_file():
            artifacts.append(self._register(job_dir, log_path, artifact_id="job-log", kind="log", label="任务日志", media_type="text/plain; charset=utf-8"))
        return artifacts
