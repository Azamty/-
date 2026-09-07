"""V2 persistent task stages built on the existing single JobManager worker."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .jianpu_score.analysis import analyze_audio, load_audio, probe_audio
from .jianpu_score.beat_grid import beat_grid_onsets_from_notes
from .jianpu_score.domain import (
    MusicAnalysis,
    NoteEvent,
    normalize_key,
    normalize_time_signature,
)
from .jianpu_score.high_accuracy import BEATNET_VERSION, MUSESCORE_VERSION
from .jianpu_score.high_accuracy_service import (
    HighAccuracyArtifactService,
    HighAccuracyBuildResult,
    HighAccuracyServiceError,
)
from .jianpu_score.models.adapter import EngineResult, run_engine
from .jianpu_score.models.demucs import (
    DEFAULT_DEMUCS_MODEL,
    demucs_model_catalog,
    normalize_demucs_model,
    separate_htdemucs,
)
from .jianpu_score.quantize import NoNotesError
from .jianpu_score.render import (
    natural_svg_sort_key,
    render_score,  # noqa: F401 - legacy monkeypatch/import surface
)
from .jianpu_score.vocal_cleanup import VocalCleanupError, clean_vocal_events
from .muscriptor_v2 import (
    instrument_label_zh,
    stable_track_id,
    write_unquantized_midi,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PYTHON = ROOT / ".venv-model-muscriptor" / "Scripts" / "python.exe"
V2_SOURCE_KINDS = frozenset({"instrumental", "vocal"})
V2_SOURCE_LABELS = {"instrumental": "伴奏/纯音乐", "vocal": "人声"}
HIGH_ACCURACY_V2_METADATA = {
    "notation_engine": "musescore-midi-import",
    "beat_engine": "beatnet",
    "beatnet_version": BEATNET_VERSION,
    "musescore_version": MUSESCORE_VERSION,
    "score_ticks_per_quarter": 48,
}
V2_RECOGNITION_ARTIFACT_IDS = frozenset(
    {
        "v2-recognition-json",
        "v2-original-midi",
        "v2-analysis-suggestion",
        "v2-analysis-full",
        "v2-beat-grid",
        "v2-recognition-log",
    }
)
def _is_vocal_generation_artifact_id(artifact_id: str) -> bool:
    """Return whether an ID belongs to the replaceable latest vocal attempt."""

    return (
        artifact_id in {
            "v2-vocal-game-raw-notes",
            "v2-vocal-game-cleaned-notes",
            "v2-vocal-game-cleanup-failure",
            "v2-vocal-game-cleanup-report",
            "v2-vocal-analysis-cleaned",
        }
        or artifact_id.startswith("v2-vocal-high-accuracy-")
    )


def _is_historical_attempt_artifact_id(artifact_id: str) -> bool:
    return bool(re.search(r"-attempt-\d{4}(?:-\d+)?$", artifact_id))


def _analysis_suggestion(analysis: MusicAnalysis) -> dict[str, Any]:
    """Expose MusicAnalysis values and ranked candidates to the V2 client."""

    metadata = dict(analysis.metadata)
    beat_grid = metadata.get("beat_grid") if isinstance(metadata.get("beat_grid"), Mapping) else {}
    tempo_summary = deepcopy(dict(beat_grid.get("tempo") or {})) if isinstance(beat_grid, Mapping) else {}
    for candidate in tempo_summary.get("candidates", []):
        if isinstance(candidate, Mapping):
            candidate.pop("beat_times", None)
    return {
        "bpm": float(analysis.bpm),
        "key": analysis.key,
        "time_signature": analysis.time_signature,
        "candidates": {
            "bpm": list(metadata.get("bpm_candidates") or [float(analysis.bpm)]),
            "key": list(metadata.get("key_candidates") or [analysis.key]),
            "time_signature": list(metadata.get("time_signature_candidates") or [analysis.time_signature]),
        },
        "warnings": list(analysis.warnings),
        "beat_grid": {
            "engine": beat_grid.get("engine", metadata.get("beat_source")),
            "mode": beat_grid.get("mode", "offline"),
            "inference": beat_grid.get("inference", "DBN"),
            "beat_count": len(beat_grid.get("beats", [])) if isinstance(beat_grid, Mapping) else 0,
            "bar_count": len(beat_grid.get("bars", [])) if isinstance(beat_grid, Mapping) else 0,
            "time_signature": beat_grid.get("time_signature"),
            "tempo": tempo_summary,
            "warnings": list(beat_grid.get("warnings", [])) if isinstance(beat_grid, Mapping) else [],
        },
        "sources": {
            "bpm": metadata.get("beat_source"),
            "key": "librosa_chroma" if metadata.get("key_candidates") else "analysis",
            "time_signature": metadata.get("time_signature_source"),
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cached_model_path() -> Path | None:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--MuScriptor--muscriptor-medium" / "snapshots"
    candidates = [path / "model.safetensors" for path in cache.glob("*")]
    candidates = [path for path in candidates if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


class V2JobService:
    """Store and execute V2 stages while sharing JobManager's one worker."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def create_job(
        self,
        *,
        original_name: str,
        source_kind: str,
        title: str,
        separation_model: str | None = None,
    ) -> tuple[str, Path]:
        if source_kind not in V2_SOURCE_KINDS:
            raise ValueError("V2 source_kind must be instrumental or vocal")
        selected_model = normalize_demucs_model(separation_model)
        if source_kind != "vocal":
            selected_model = None
        job_id = str(uuid.uuid4())
        directory = self.manager._safe_job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=False)
        suffix = Path(original_name).suffix.lower()
        input_path = directory / f"input{suffix}"
        now = _utc_now()
        engine = "muscriptor" if source_kind == "instrumental" else "game"
        state: dict[str, Any] = {
            "schema_version": "2.0",
            "kind": "v2",
            "id": job_id,
            "status": "uploading",
            "phase": "uploading",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "attempt": 1,
            "options": {
                "engine": engine,
                "voice_mode": "monophonic",
                "source_kind": source_kind,
                "language": "mixed",
                "separate": False,
                "title": title,
                "separation_model": selected_model,
            },
            "input": {
                "original_name": self._filename(original_name),
                "stored_name": input_path.name,
                "bytes": None,
                "duration_sec": None,
            },
            "error": None,
            "warnings": [],
            "artifacts": [],
            "progress": None,
            "summary": None,
            "v2": {
                "stage": "recognize" if source_kind == "instrumental" else "vocal_separation",
                "source_kind": source_kind,
                "source_label": V2_SOURCE_LABELS[source_kind],
                "route": {
                    "engine": engine,
                    "use_demucs": source_kind == "vocal",
                    **(
                        {
                            "separation_engine": "demucs",
                            "separation_model": selected_model,
                        }
                        if source_kind == "vocal"
                        else {}
                    ),
                },
                "tracks": [],
                "notes": [],
                "analysis": None,
                "selection_revision": 0,
                "selection": None,
                "selection_history": [],
                "score_refusal": None,
                **deepcopy(HIGH_ACCURACY_V2_METADATA),
            },
        }
        self.manager._write(state)
        self.manager._log(job_id, f"created V2 task: {state['input']['original_name']}")
        return job_id, input_path

    @staticmethod
    def _filename(value: str) -> str:
        text = str(value or "").replace("\x00", "")
        text = "".join(char if char.isprintable() and char not in "\\/:*?\"<>|" else " " for char in text)
        return " ".join(text.split()).strip(" .") or "audio"

    def is_v2(self, job_id: str) -> bool:
        return self.manager._read(job_id).get("kind") == "v2"

    @staticmethod
    def _vocal_model(state: Mapping[str, Any]) -> str:
        """Read the durable model choice, with v2.0 compatibility fallback."""

        v2 = state.get("v2") or {}
        separation = v2.get("separation") or {}
        route = v2.get("route") or {}
        options = state.get("options") or {}
        return normalize_demucs_model(
            separation.get("model") or route.get("separation_model") or options.get("separation_model") or DEFAULT_DEMUCS_MODEL
        )

    @staticmethod
    def _vocal_route(model: str) -> dict[str, Any]:
        return {
            "engine": "game",
            "use_demucs": True,
            "separation_engine": "demucs",
            "separation_model": model,
        }

    @staticmethod
    def _vocal_model_info(model: str) -> dict[str, Any]:
        return next(item for item in demucs_model_catalog() if item["id"] == model)

    def generate_vocal(self, job_id: str) -> dict[str, Any]:
        """Queue GAME for a previously separated vocal stem.

        Separation is deliberately a separate durable state.  A repeated click
        while the generation is queued/running is idempotent, and a failed GAME
        attempt can be retried without invoking Demucs again.
        """

        with self.manager._lock:
            state = self.manager._read(job_id)
            if state.get("kind") != "v2" or state.get("v2", {}).get("source_kind") != "vocal":
                raise ValueError("只有 V2 人声任务可以生成人声简谱")
            status = str(state.get("status"))
            stage = str(state.get("v2", {}).get("stage"))
            if status == "completed" and stage == "vocal_complete":
                return state
            if status in {"queued", "running"} and stage == "vocal_generate":
                return state
            if status not in {"vocal_ready", "failed", "interrupted"}:
                raise ValueError("请先等待 Demucs 分离完成并试听人声")
            if status in {"failed", "interrupted"} and stage != "vocal_generate":
                raise ValueError("当前任务尚未得到可复用的人声分离结果，请先重试分离")
            prepared = self._prepared_vocal_path(job_id, state)
            self._validate_internal_vocal_path(job_id, prepared)
            v2 = dict(state.get("v2", {}))
            model = self._vocal_model(state)
            v2["stage"] = "vocal_generate"
            v2["route"] = self._vocal_route(model)
            v2["progress_detail"] = {
                "status": "queued",
                "engine": "game",
                "input": "v2-vocals-audio",
                "model": model,
                "separation_model": model,
            }
            state.update(
                {
                    "v2": v2,
                    "status": "queued",
                    "phase": "queued",
                    "started_at": None,
                    "finished_at": None,
                    "updated_at": _utc_now(),
                    "error": None,
                    "progress": 0.0,
                }
            )
            self.manager._write(state)
            self.manager._queue.put(job_id)
        self.manager._log(job_id, "queued GAME generation from persisted vocals stem")
        return state

    def select(
        self,
        job_id: str,
        selected_track_ids: Sequence[str],
        merge_main_melody: bool = False,
        *,
        bpm_override: float | None = None,
        key_override: str | None = None,
        time_signature_override: str | None = None,
    ) -> dict[str, Any]:
        with self.manager._lock:
            state = self.manager._read(job_id)
            if state.get("kind") != "v2":
                raise ValueError("不是 V2 任务")
            if state.get("v2", {}).get("source_kind") != "instrumental":
                raise ValueError("人声任务不需要选择乐器")
            if state.get("status") not in {"selection_ready", "completed"}:
                raise ValueError("识别尚未完成，暂不能选择乐器")
            tracks = list(state.get("v2", {}).get("tracks", []))
            known = {str(track.get("track_id")) for track in tracks}
            requested: list[str] = []
            for value in selected_track_ids:
                track_id = str(value).strip()
                if track_id and track_id not in requested:
                    requested.append(track_id)
            unknown = sorted(set(requested) - known)
            if unknown:
                raise ValueError("selected_track_ids 未识别：" + ", ".join(unknown))
            analysis = dict(state.get("v2", {}).get("analysis") or {})
            bpm = float(analysis.get("bpm", 120.0)) if bpm_override is None else float(bpm_override)
            if not math.isfinite(bpm) or bpm <= 0 or bpm > 400:
                raise ValueError("bpm_override 必须在 0 到 400 之间")
            key = normalize_key(key_override or str(analysis.get("key") or "C"))
            time_signature = normalize_time_signature(time_signature_override or str(analysis.get("time_signature") or "4/4"))
            revision = int(state.get("v2", {}).get("selection_revision", 0)) + 1
            selection = {
                "revision": revision,
                "selected_track_ids": requested,
                "merge_main_melody": bool(merge_main_melody),
                "bpm_override": bpm,
                "key_override": key,
                "time_signature_override": time_signature,
                # Keep the effective values above for API compatibility, but
                # persist whether the user actually supplied each override.
                # The high-accuracy pipeline uses these flags to preserve the
                # detected BeatNet source when a value was left untouched.
                "bpm_override_explicit": bpm_override is not None,
                "key_override_explicit": key_override is not None,
                "time_signature_override_explicit": time_signature_override is not None,
                "created_at": _utc_now(),
            }
            v2 = dict(state.get("v2", {}))
            v2["stage"] = "export"
            v2["selection_revision"] = revision
            v2["selection"] = selection
            v2["selection_history"] = [*v2.get("selection_history", []), deepcopy(selection)]
            state.update(
                {
                    "v2": v2,
                    "status": "queued",
                    "phase": "queued",
                    "started_at": None,
                    "finished_at": None,
                    "updated_at": _utc_now(),
                    "error": None,
                    "progress": 0.0,
                }
            )
            self.manager._write(state)
            self.manager._queue.put(job_id)
        self.manager._log(job_id, f"queued selection revision {revision}")
        return state

    def tracks(self, job_id: str) -> dict[str, Any]:
        state = self.manager._read(job_id)
        if state.get("kind") != "v2":
            raise KeyError("not a V2 job")
        v2 = state.get("v2", {})
        return {
            "job_id": job_id,
            "status": state.get("status"),
            "phase": state.get("phase"),
            "source_kind": v2.get("source_kind"),
            "source_label": v2.get("source_label"),
            "recognition_progress": {
                "value": state.get("progress"),
                "detail": v2.get("progress_detail"),
            },
            "count": len(v2.get("tracks", [])),
            "tracks": deepcopy(v2.get("tracks", [])),
            "analysis": deepcopy(v2.get("analysis")),
            "selection_revision": v2.get("selection_revision", 0),
            "selection": deepcopy(v2.get("selection")),
            "score_refusal": deepcopy(v2.get("score_refusal")),
            "metadata": {
                key: v2.get(key)
                for key in HIGH_ACCURACY_V2_METADATA
                if v2.get(key) is not None
            },
        }

    def run(self, job_id: str) -> None:
        state = self.manager._read(job_id)
        v2 = state.get("v2", {})
        stage = str(v2.get("stage"))
        if stage == "recognize":
            self._run_instrumental_recognition(job_id)
            return
        if stage == "vocal_separation":
            self._run_vocal_separation(job_id)
            return
        if stage == "vocal_generate":
            self._run_vocal_generation(job_id)
            return
        if stage == "export":
            self._run_instrumental_export(job_id)
            return
        raise ValueError("V2 task stage is invalid")

    def _output_dir(self, job_id: str) -> Path:
        directory = self.manager._safe_job_dir(job_id)
        output = directory / "output"
        if output.is_symlink() or (output.exists() and output.resolve().parent != directory):
            raise ValueError("invalid job output path")
        output.mkdir(parents=True, exist_ok=True)
        return output

    def _run_instrumental_recognition(self, job_id: str) -> None:
        state_at_start = self.manager._read(job_id)
        attempt = int(state_at_start.get("attempt", 1))
        recognition_root = self._output_dir(job_id) / "v2-recognition"
        if recognition_root.is_symlink() or (
            recognition_root.exists() and recognition_root.resolve().parent != self._output_dir(job_id)
        ):
            raise ValueError("invalid V2 recognition output path")
        output = recognition_root / f"attempt-{attempt:04d}"
        if output.is_symlink() or (output.exists() and output.resolve().parent != recognition_root):
            raise ValueError("invalid V2 recognition attempt output path")
        output.mkdir(parents=True, exist_ok=True)
        progress_path = output / "progress.json"
        self.manager._set_phase(job_id, "recognizing")
        command = [
            os.fspath(MODEL_PYTHON),
            os.fspath(ROOT / "scripts" / "run_muscriptor_job.py"),
            "--audio",
            os.fspath(self.manager.input_path(job_id)),
            "--output",
            os.fspath(output),
            "--progress",
            os.fspath(progress_path),
        ]
        cached = _cached_model_path()
        if cached is not None:
            command.extend(("--model", os.fspath(cached)))
        return_code = self.manager._run_isolated_child(command, job_id, progress_path)
        if return_code:
            raise RuntimeError(f"MuScriptor worker exited with code {return_code}")
        recognition_path = output / "recognition.json"
        midi_path = output / "original.mid"
        if not recognition_path.is_file() or not midi_path.is_file():
            raise RuntimeError("MuScriptor worker did not produce recognition.json and original.mid")
        recognition = json.loads(recognition_path.read_text(encoding="utf-8"))
        tracks = list(recognition.get("tracks", []))
        notes = list(recognition.get("notes", []))
        if not tracks and not notes:
            raise NoNotesError("NoNotes: MuScriptor returned no note events")
        _samples, analysis = analyze_audio(
            self.manager.input_path(job_id),
            source_onsets=beat_grid_onsets_from_notes(notes),
        )
        analysis_suggestion = _analysis_suggestion(analysis)
        analysis_path = output / "analysis-suggestion.json"
        _safe_json(analysis_path, analysis_suggestion)
        full_analysis_path = output / "analysis.json"
        full_analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        beat_grid = analysis.metadata.get("beat_grid")
        beat_grid_path = output / "beat_grid.json"
        if isinstance(beat_grid, Mapping):
            _safe_json(beat_grid_path, beat_grid)
        job_dir = self.manager._safe_job_dir(job_id)
        prior_artifacts = [
            item
            for item in state_at_start.get("artifacts", [])
            if str(item.get("artifact_id")) not in V2_RECOGNITION_ARTIFACT_IDS
            and str(item.get("artifact_id")) != "v2-source-audio"
        ]
        artifacts = [
            *prior_artifacts,
            self.manager._register(
                job_dir,
                self.manager.input_path(job_id),
                artifact_id="v2-source-audio",
                kind="source_audio",
                label="原始音频",
                media_type=self._audio_media_type(self.manager.input_path(job_id)),
            ),
            self.manager._register(
                job_dir,
                midi_path,
                artifact_id="v2-original-midi",
                kind="original_midi",
                label="完整识别 MIDI（原始时间）",
                media_type="audio/midi",
            ),
            self.manager._register(
                job_dir,
                recognition_path,
                artifact_id="v2-recognition-json",
                kind="recognition_json",
                label="乐器识别数据",
                media_type="application/json",
            ),
            self.manager._register(
                job_dir,
                analysis_path,
                artifact_id="v2-analysis-suggestion",
                kind="analysis_json",
                label="原音分析建议",
                media_type="application/json",
            ),
            self.manager._register(
                job_dir,
                full_analysis_path,
                artifact_id="v2-analysis-full",
                kind="analysis_full_json",
                label="BeatNet 完整分析数据",
                media_type="application/json",
            ),
        ]
        if beat_grid_path.is_file():
            artifacts.append(
                self.manager._register(
                    job_dir,
                    beat_grid_path,
                    artifact_id="v2-beat-grid",
                    kind="beat_grid_json",
                    label="BeatNet 拍点网格",
                    media_type="application/json",
                )
            )
        worker_log = output / "muscriptor-worker.log"
        if worker_log.is_file():
            artifacts.append(
                self.manager._register(
                    job_dir,
                    worker_log,
                    artifact_id="v2-recognition-log",
                    kind="log",
                    label="MuScriptor 识别日志",
                    media_type="text/plain; charset=utf-8",
                )
            )
        with self.manager._lock:
            state = self.manager._read(job_id)
            v2 = dict(state.get("v2", {}))
            v2.update(
                {
                    **HIGH_ACCURACY_V2_METADATA,
                    "stage": "selection_ready",
                    "tracks": tracks,
                    "notes": notes,
                    "recognition": {
                        "engine": "muscriptor",
                        "model": recognition.get("model", "medium"),
                        "device": recognition.get("device"),
                        "metadata": recognition.get("metadata", {}),
                    },
                    "analysis": analysis_suggestion,
                    "analysis_relative": full_analysis_path.relative_to(job_dir).as_posix(),
                    "beat_grid_relative": beat_grid_path.relative_to(job_dir).as_posix() if beat_grid_path.is_file() else None,
                    "progress_detail": recognition.get("progress", {}),
                    "score_refusal": None,
                }
            )
            state.update(
                {
                    "v2": v2,
                    "status": "selection_ready",
                    "phase": "selection_ready",
                    "finished_at": _utc_now(),
                    "error": None,
                    "warnings": [
                        "已完成一次 MuScriptor 全量识别；选择乐器不会重新运行模型。",
                        "识别 NoteEvent 不包含力度；MIDI 试听使用固定 playback_default=80。",
                    ],
                    "artifacts": artifacts,
                    "progress": 1.0,
                    "summary": {
                        **HIGH_ACCURACY_V2_METADATA,
                        "source_kind": "instrumental",
                        "route": {"engine": "muscriptor", "use_demucs": False},
                        "note_count": len(notes),
                        "track_count": len(tracks),
                        "instrument_counts": recognition.get("instrument_counts", {}),
                        "analysis": analysis_suggestion,
                        "selection_ready": True,
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(state)
        self.manager._log(job_id, f"MuScriptor selection_ready: {len(tracks)} tracks, {len(notes)} notes")

    def _safe_internal_path(self, job_id: str, path: Path | str) -> Path:
        """Resolve a model-produced path while keeping every parent in the job."""

        job_dir = self.manager._safe_job_dir(job_id)
        raw = Path(path)
        if not raw.is_absolute():
            raw = job_dir / raw
        raw = raw.absolute()
        current = raw
        while current != job_dir:
            if current.is_symlink():
                raise ValueError("人声分离结果路径包含不安全的符号链接")
            if current.parent == current:
                raise ValueError("人声分离结果路径不在任务目录内")
            current = current.parent
        resolved = raw.resolve()
        if resolved == job_dir or job_dir not in resolved.parents or not resolved.is_file():
            raise ValueError("人声分离结果路径不在任务目录内")
        return resolved

    def _validate_internal_vocal_path(self, job_id: str, path: Path) -> dict[str, Any]:
        """Apply the trusted internal-stem guard before GAME or browser serving."""

        safe_path = self._safe_internal_path(job_id, path)
        probe = probe_audio(safe_path, enforce_upload_size=False)
        samples, sample_rate = load_audio(safe_path, sample_rate=16000)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Demucs 分离后的人声为空或无效，无法进入 GAME，请更换音频。")
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if not math.isfinite(peak) or not math.isfinite(rms) or peak <= 1e-6 or rms <= 1e-7:
            raise ValueError("Demucs 分离后的人声为空或接近静音，无法进入 GAME，请更换音频。")
        return {
            "duration_sec": float(probe["duration_sec"]),
            "bytes": int(probe["bytes"]),
            "sample_rate": int(sample_rate),
            "peak": peak,
            "rms": rms,
        }

    def _prepared_vocal_path(self, job_id: str, state: Mapping[str, Any] | None = None) -> Path:
        current = state or self.manager._read(job_id)
        separation = dict((current.get("v2") or {}).get("separation") or {})
        relative = separation.get("prepared_vocals_relative")
        if not relative:
            artifact = next(
                (item for item in current.get("artifacts", []) if item.get("artifact_id") == "v2-vocals-audio"),
                None,
            )
            relative = artifact.get("relative_path") if artifact else None
        if not relative:
            raise ValueError("当前任务没有可复用的人声分离结果，请先完成 Demucs 分离")
        return self._safe_internal_path(job_id, Path(str(relative)))

    def _prepared_analysis(self, job_id: str, state: Mapping[str, Any]) -> MusicAnalysis:
        separation = dict((state.get("v2") or {}).get("separation") or {})
        relative = separation.get("analysis_relative")
        if not relative:
            raise ValueError("人声原音分析结果缺失，请重新分离")
        path = self._safe_internal_path(job_id, Path(str(relative)))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return MusicAnalysis.model_validate(value)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("人声原音分析结果无效，请重新分离") from exc

    @staticmethod
    def _analysis_has_beatnet(analysis: MusicAnalysis | None) -> bool:
        if analysis is None:
            return False
        metadata = analysis.metadata
        grid = metadata.get("beat_grid")
        return bool(
            metadata.get("beat_engine") == "beatnet"
            and metadata.get("beatnet_version") == BEATNET_VERSION
            and isinstance(grid, Mapping)
            and grid.get("beats")
            and len(analysis.beat_times) >= 2
        )

    @staticmethod
    def _event_payload(events: Sequence[NoteEvent], **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            **extra,
            "event_count": len(events),
            "events": [event.model_dump(mode="json") for event in events],
        }

    def _register_high_accuracy_result(
        self,
        job_id: str,
        result: HighAccuracyBuildResult,
        *,
        prefix: str,
        label: str,
        stem_id: str,
        family: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Register every service file with stable V2 IDs.

        The service manifest is intentionally registered separately because it
        cannot include its own hash without becoming self-referential.
        ``family`` controls the public artifact kinds while the service keeps
        its own generic manifest kinds.
        """

        if family not in {"vocal", "instrument", "main_melody"}:
            raise ValueError(f"unsupported high-accuracy artifact family: {family}")
        job_dir = self.manager._safe_job_dir(job_id)
        artifacts: list[dict[str, Any]] = []
        score_ids: list[str] = []
        page_number = 0
        used_ids: set[str] = set()

        def public_id(tag: str) -> str:
            candidate = f"{prefix}-{tag}"
            suffix = 2
            while candidate in used_ids:
                candidate = f"{prefix}-{tag}-{suffix}"
                suffix += 1
            used_ids.add(candidate)
            return candidate

        def family_kind(suffix: str) -> str:
            return f"{family}_{suffix}"

        def artifact_sort_key(service_artifact: Any) -> tuple[str, int, int, str]:
            relative = Path(service_artifact.relative_path)
            parent = relative.parent.as_posix().casefold()
            if relative.suffix.casefold() == ".svg":
                category, page, name = natural_svg_sort_key(relative)
                return parent, category, page, name
            return parent, 9, 0, relative.as_posix().casefold()

        for service_artifact in sorted(result.artifacts, key=artifact_sort_key):
            path = service_artifact.path.resolve()
            name = path.name.lower()
            page: int | None = None
            if name.endswith(".performance.mid"):
                tag, kind, item_label = "performance-midi", family_kind("performance_midi"), f"{label}性能 MIDI"
            elif name.endswith(".performance.metadata.json"):
                tag, kind, item_label = "performance-metadata", family_kind("performance_metadata"), f"{label}性能 MIDI 元数据"
            elif name.endswith(".selected.mid"):
                tag, kind, item_label = "selected-midi", family_kind("selected_midi"), f"{label}选中 MIDI"
            elif name.endswith((".score.mid", ".score.midi")):
                tag, kind, item_label = "score-midi", family_kind("score_midi"), f"{label}最终 Score MIDI"
                score_ids.append(f"{prefix}-{tag}")
            elif name.endswith(".notated.musicxml"):
                tag, kind, item_label = "musicxml", family_kind("musicxml"), f"{label} MusicXML"
            elif name.endswith(".alignment_report.json"):
                tag, kind, item_label = "alignment-report", family_kind("alignment_report"), f"{label}对齐报告"
            elif name.endswith(".score.json"):
                tag, kind, item_label = "score-json", family_kind("score_json"), f"{label}分谱数据"
                score_ids.append(f"{prefix}-{tag}")
            elif name.endswith(".note-events.json"):
                tag, kind, item_label = "service-note-events", family_kind("note_events"), f"{label}服务输入音符"
            elif name.endswith(".score.jly"):
                tag, kind, item_label = "jianpu-source", family_kind("jianpu_source"), f"{label}简谱源文本"
            elif name.endswith(".score.ly"):
                tag, kind, item_label = "lilypond-source", family_kind("lilypond_source"), f"{label} LilyPond 源文本"
            elif name.endswith(".long.svg"):
                tag, kind, item_label = "score-svg-long", family_kind("score_svg_long"), f"下载长图 SVG · {label}"
                score_ids.append(f"{prefix}-{tag}")
            elif name.endswith(".svg"):
                page_number += 1
                page = page_number
                tag, kind, item_label = f"score-svg-{page_number}", family_kind("score_svg"), f"{label}第 {page_number} 页"
                score_ids.append(f"{prefix}-{tag}")
            elif name.endswith(".log"):
                tag, kind, item_label = f"service-log-{len(artifacts) + 1}", "high_accuracy_log", f"{label}高精度处理日志"
            else:
                safe_name = re.sub(r"[^a-z0-9_.-]+", "-", service_artifact.relative_path.lower()).strip("-") or "file"
                tag, kind, item_label = f"service-{safe_name}", "high_accuracy_support", f"{label}高精度辅助文件"
            artifact_id = public_id(tag)
            registered = self.manager._register(
                job_dir,
                path,
                artifact_id=artifact_id,
                kind=kind,
                label=item_label,
                media_type=self._artifact_media_type(path),
                stem_id=stem_id,
                page=page,
            )
            artifacts.append(registered)
        manifest_id = public_id("manifest")
        artifacts.append(
            self.manager._register(
                job_dir,
                result.manifest_path,
                artifact_id=manifest_id,
                kind="high_accuracy_manifest",
                label=f"{label}高精度处理清单",
                media_type="application/json",
                stem_id=stem_id,
            )
        )
        return artifacts, score_ids

    def _register_high_accuracy_failure(
        self,
        job_id: str,
        error: HighAccuracyServiceError,
        *,
        prefix: str,
        label: str,
        stem_id: str,
    ) -> list[dict[str, Any]]:
        job_dir = self.manager._safe_job_dir(job_id)
        registered: list[dict[str, Any]] = []
        for suffix, path, kind, item_label in (
            ("manifest", error.manifest_path, "high_accuracy_manifest", f"{label}失败清单"),
            ("failure-log", error.log_path, "high_accuracy_log", f"{label}失败日志"),
        ):
            if path is None or not path.is_file():
                continue
            registered.append(
                self.manager._register(
                    job_dir,
                    path,
                    artifact_id=f"{prefix}-{suffix}",
                    kind=kind,
                    label=item_label,
                    media_type=self._artifact_media_type(path),
                    stem_id=stem_id,
                )
            )
        return registered

    @staticmethod
    def _artifact_media_type(path: Path) -> str:
        suffix = path.name.lower()
        if suffix.endswith(".mid"):
            return "audio/midi"
        if suffix.endswith(".musicxml"):
            return "application/vnd.recordare.musicxml+xml"
        if suffix.endswith(".svg"):
            return "image/svg+xml"
        if suffix.endswith(".json"):
            return "application/json"
        if suffix.endswith((".jly", ".ly", ".log")):
            return "text/plain; charset=utf-8"
        return "application/octet-stream"

    def _persist_generation_state(
        self,
        job_id: str,
        *,
        artifacts: Sequence[Mapping[str, Any]],
        generation: Mapping[str, Any],
    ) -> None:
        with self.manager._lock:
            state = self.manager._read(job_id)
            v2 = dict(state.get("v2", {}))
            v2["generation"] = deepcopy(dict(generation))
            v2.update(HIGH_ACCURACY_V2_METADATA)
            state["v2"] = v2
            state["artifacts"] = [dict(item) for item in artifacts]
            state["updated_at"] = _utc_now()
            self.manager._write(state)

    @staticmethod
    def _analysis_for_events(
        base_analysis: MusicAnalysis,
        events: Sequence[NoteEvent],
        *,
        bpm: float,
        key: str,
        time_signature: str,
        bpm_manual: bool,
        key_manual: bool,
        time_signature_manual: bool,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> MusicAnalysis:
        duration = max((event.end_sec for event in events), default=0.1)
        metadata = V2JobService._selection_analysis_metadata(
            base_analysis,
            bpm=bpm,
            key=key,
            time_signature=time_signature,
            bpm_manual=bpm_manual,
            key_manual=key_manual,
            time_signature_manual=time_signature_manual,
        )
        if metadata_extra:
            metadata.update(deepcopy(dict(metadata_extra)))
        return base_analysis.model_copy(
            update={
                "duration_sec": max(float(base_analysis.duration_sec), duration, 0.1),
                "bpm": float(bpm),
                "key": normalize_key(key),
                "time_signature": normalize_time_signature(time_signature),
                "note_events": list(events),
                "metadata": metadata,
            }
        )

    def _persisted_instrumental_analysis(self, job_id: str, state: Mapping[str, Any]) -> MusicAnalysis | None:
        """Load the full BeatNet analysis retained during MuScriptor decode.

        V2 selections are rendered later than recognition.  Reconstructing a
        new MusicAnalysis from only BPM/key suggestions here used to discard
        BeatNet's real beat_times and silently returned the old fixed grid.
        Legacy hand-authored states without the full artifact remain usable,
        but are explicitly represented as an analysis without a beat map.
        """

        v2 = state.get("v2") or {}
        relative = v2.get("analysis_relative")
        if not relative:
            artifact = next(
                (item for item in state.get("artifacts", []) if item.get("artifact_id") == "v2-analysis-full"),
                None,
            )
            relative = artifact.get("relative_path") if artifact else None
        if relative:
            path = self._safe_internal_path(job_id, Path(str(relative)))
            try:
                return MusicAnalysis.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("MuScriptor 的 BeatNet 完整分析数据无效，请重新识别") from exc
        suggestion = v2.get("analysis") or {}
        if not isinstance(suggestion, Mapping) or not suggestion:
            return None
        notes = list(v2.get("notes") or [])
        duration = max((float(note.get("end_sec", 0.0)) for note in notes), default=0.1)
        try:
            return MusicAnalysis(
                sample_rate=22050,
                duration_sec=max(duration, 0.1),
                bpm=float(suggestion.get("bpm", 120.0)),
                key=normalize_key(str(suggestion.get("key", "C"))),
                time_signature=normalize_time_signature(str(suggestion.get("time_signature", "4/4"))),
                warnings=list(suggestion.get("warnings") or []),
                metadata={"beat_source": "legacy_v2_suggestion"},
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("V2 分析建议无效，请重新识别") from exc

    def _run_vocal_separation(self, job_id: str) -> None:
        """Prepare and persist only the Demucs vocals stem."""

        self.manager._set_phase(job_id, "separating")
        output = self._output_dir(job_id) / "vocal-prep"
        if output.is_symlink() or (output.exists() and output.resolve().parent != self.manager._safe_job_dir(job_id) / "output"):
            raise ValueError("invalid V2 vocal preparation output path")
        output.mkdir(parents=True, exist_ok=True)
        input_path = self.manager.input_path(job_id)
        _samples, analysis = analyze_audio(input_path)
        state = self.manager._read(job_id)
        model = self._vocal_model(state)
        model_info = self._vocal_model_info(model)
        stems = separate_htdemucs(input_path, output / "demucs", model=model, process_holder=self.manager)
        vocal_path = stems.get("vocals") if isinstance(stems, Mapping) else None
        if vocal_path is None:
            raise ValueError("Demucs 未生成 vocals 人声结果，请更换音频后重试")
        vocal_path = self._safe_internal_path(job_id, Path(vocal_path))
        stats = self._validate_internal_vocal_path(job_id, vocal_path)
        # Only the requested stem is retained.  This prevents an accompaniment
        # stem from being accidentally routed into GAME or exposed as an audio
        # artifact on a later retry.
        for stem_id, stem_value in (stems.items() if isinstance(stems, Mapping) else []):
            if stem_id == "vocals":
                continue
            try:
                candidate = self._safe_internal_path(job_id, Path(stem_value))
            except ValueError:
                continue
            candidate.unlink(missing_ok=True)
        job_dir = self.manager._safe_job_dir(job_id)
        analysis_path = output / "original-analysis.json"
        analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        beat_grid = analysis.metadata.get("beat_grid")
        beat_grid_path = output / "beat_grid.json"
        if isinstance(beat_grid, Mapping):
            _safe_json(beat_grid_path, beat_grid)
        analysis_suggestion = _analysis_suggestion(analysis)
        artifacts = [
            self.manager._register(
                job_dir,
                input_path,
                artifact_id="v2-source-audio",
                kind="source_audio",
                label="原始人声音频",
                media_type=self._audio_media_type(input_path),
            ),
            self.manager._register(
                job_dir,
                vocal_path,
                artifact_id="v2-vocals-audio",
                kind="vocal_audio",
                label="Demucs 分离人声（试听）",
                media_type="audio/wav",
                stem_id="vocals",
            ),
            self.manager._register(
                job_dir,
                analysis_path,
                artifact_id="v2-vocal-analysis",
                kind="analysis_json",
                label="原音分析建议",
                media_type="application/json",
            ),
        ]
        if beat_grid_path.is_file():
            artifacts.append(
                self.manager._register(
                    job_dir,
                    beat_grid_path,
                    artifact_id="v2-beat-grid",
                    kind="beat_grid_json",
                    label="BeatNet 拍点网格",
                    media_type="application/json",
                )
            )
        separation = {
            "engine": "demucs",
            "model": model,
            "model_info": model_info,
            "source": "vocal",
            "stem": "vocals",
            "artifact_id": "v2-vocals-audio",
            "prepared_vocals_relative": vocal_path.relative_to(job_dir).as_posix(),
            "analysis_relative": analysis_path.relative_to(job_dir).as_posix(),
            "beat_grid_relative": beat_grid_path.relative_to(job_dir).as_posix() if beat_grid_path.is_file() else None,
            **stats,
            "warnings": ["这是模型分离结果，可能含伴奏残留；GAME 只处理 vocals stem。"],
        }
        with self.manager._lock:
            state = self.manager._read(job_id)
            v2 = dict(state.get("v2", {}))
            v2.update(
                {
                    **HIGH_ACCURACY_V2_METADATA,
                    "stage": "vocal_ready",
                    "route": self._vocal_route(model),
                    "analysis": analysis_suggestion,
                    "separation": separation,
                    "progress_detail": {"status": "vocal_ready", "engine": "demucs", "model": model, "separation_model": model},
                }
            )
            state.update(
                {
                    "v2": v2,
                    "status": "vocal_ready",
                    "phase": "vocal_ready",
                    "finished_at": _utc_now(),
                    "error": None,
                    "warnings": list(dict.fromkeys([*analysis.warnings, *separation["warnings"]])),
                    "artifacts": artifacts,
                    "progress": 1.0,
                    "summary": {
                        **HIGH_ACCURACY_V2_METADATA,
                        "source_kind": "vocal",
                        "route": self._vocal_route(model),
                        "stage": "vocal_ready",
                        "separation": separation,
                        "analysis": analysis_suggestion,
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(state)
        self.manager._log(job_id, f"V2 vocal_ready: Demucs vocals {stats['duration_sec']:.3f}s")

    def _run_vocal_generation(self, job_id: str) -> None:
        """Run GAME, conservatively clean its monophonic output, then use 9A."""

        state = self.manager._read(job_id)
        options = dict(state.get("options", {}))
        model = self._vocal_model(state)
        vocal_path = self._prepared_vocal_path(job_id, state)
        original_analysis = self._prepared_analysis(job_id, state)
        if not self._analysis_has_beatnet(original_analysis):
            raise ValueError("人声原曲的 BeatNet 拍点分析缺失或版本不匹配，无法生成高精度简谱")
        self.manager._set_phase(job_id, "recognizing")
        result: EngineResult = run_engine(
            "game",
            os.fspath(vocal_path),
            analysis=original_analysis,
            stem_id="vocals",
            language="mixed",
            trusted_internal=True,
        )
        if not result.events:
            raise ValueError("GAME 未在人声分离结果中识别到有效音符，请重试或更换音频。")
        raw_events = tuple(
            event.model_copy(
                update={
                    "stem_id": "vocals",
                    "metadata": {
                        **event.metadata,
                        "stem_id": "vocals",
                        "engine": result.engine,
                        **({"model": result.model} if result.model else {}),
                    },
                }
            )
            for event in result.events
        )
        output = self._output_dir(job_id)
        attempt = int(state.get("attempt", 1))
        generation_root = output / "vocal-generation" / f"attempt-{attempt:04d}"
        if generation_root.is_symlink() or (generation_root.exists() and generation_root.resolve().parent != output / "vocal-generation"):
            raise ValueError("invalid V2 vocal generation output path")
        generation_root.mkdir(parents=True, exist_ok=True)
        raw_path = generation_root / "game.raw.note-events.json"
        _safe_json(
            raw_path,
            self._event_payload(
                raw_events,
                engine=result.engine,
                model=result.model,
                source_artifact_id="v2-vocals-audio",
                immutable=True,
            ),
        )
        job_dir = self.manager._safe_job_dir(job_id)
        # Retry preserves prior attempts under unique historical IDs.  Carry
        # those registrations forward so diagnostics remain downloadable;
        # only the current attempt uses the stable latest IDs below.
        prior_artifacts = list(state.get("artifacts", []))
        raw_artifacts = [
            *prior_artifacts,
            self.manager._register(job_dir, raw_path, artifact_id="v2-vocal-game-raw-notes", kind="vocal_raw_notes", label="GAME 原始音符（保留）", media_type="application/json", stem_id="vocals"),
        ]
        self._persist_generation_state(
            job_id,
            artifacts=raw_artifacts,
            generation={
                "stage": "raw_events_persisted",
                "engine": "game",
                "model": result.model,
                "input_artifact_id": "v2-vocals-audio",
                "input_relative": vocal_path.relative_to(job_dir).as_posix(),
                "raw_relative": raw_path.relative_to(job_dir).as_posix(),
                "analysis_reused_from_original": True,
                "raw_count": len(raw_events),
            },
        )
        self.manager._set_phase(job_id, "quantizing")
        try:
            cleanup = clean_vocal_events(
                raw_events,
                bpm=original_analysis.bpm,
                beat_context={
                    "bpm": original_analysis.bpm,
                    "beat_times": list(original_analysis.beat_times),
                    "beat_grid": deepcopy(original_analysis.metadata.get("beat_grid", {})),
                },
            )
        except (VocalCleanupError, TypeError, ValueError) as exc:
            report = getattr(exc, "report", None)
            if isinstance(report, Mapping):
                failure_report_path = generation_root / "vocal-cleanup.failure.json"
                _safe_json(failure_report_path, dict(report))
                failure_artifacts = [
                    *raw_artifacts,
                    self.manager._register(job_dir, failure_report_path, artifact_id="v2-vocal-game-cleanup-failure", kind="vocal_cleanup_report", label="人声清理失败报告", media_type="application/json", stem_id="vocals"),
                ]
                self._persist_generation_state(
                    job_id,
                    artifacts=failure_artifacts,
                    generation={"stage": "cleanup", "status": "failed", "failure": {"stage": "vocal_cleanup", "cause": str(exc)}, "raw_count": len(raw_events)},
                )
                self.manager._update(
                    job_id,
                    status="failed",
                    phase="failed",
                    finished_at=_utc_now(),
                    error={"code": "vocal_cleanup_failed", "message": str(exc), "stage": "vocal_cleanup"},
                )
            else:
                self._persist_generation_state(
                    job_id,
                    artifacts=raw_artifacts,
                    generation={"stage": "cleanup", "status": "failed", "failure": {"stage": "vocal_cleanup", "cause": str(exc)}, "raw_count": len(raw_events)},
                )
                self.manager._update(
                    job_id,
                    status="failed",
                    phase="failed",
                    finished_at=_utc_now(),
                    error={"code": "vocal_cleanup_failed", "message": str(exc), "stage": "vocal_cleanup"},
                )
            # The failure state and diagnostic report are already durable.  Do
            # not re-raise here: the generic queue worker would replace the
            # structured vocal_cleanup_failed error with a less useful
            # ValueError payload.
            return
        cleaned_events = tuple(cleanup.events)
        cleaned_path = generation_root / "game.cleaned.note-events.json"
        cleanup_report_path = generation_root / "game.cleanup.report.json"
        _safe_json(cleaned_path, self._event_payload(cleaned_events, source_raw=raw_path.name, immutable=True))
        _safe_json(cleanup_report_path, cleanup.report)
        generation_artifacts = [
            *raw_artifacts,
            self.manager._register(job_dir, cleaned_path, artifact_id="v2-vocal-game-cleaned-notes", kind="vocal_cleaned_notes", label="GAME 清理后人声音符", media_type="application/json", stem_id="vocals"),
            self.manager._register(job_dir, cleanup_report_path, artifact_id="v2-vocal-game-cleanup-report", kind="vocal_cleanup_report", label="人声清理报告", media_type="application/json", stem_id="vocals"),
        ]
        generation = {
            "stage": "cleanup_completed",
            "engine": "game",
            "model": result.model,
            "input_artifact_id": "v2-vocals-audio",
            "input_relative": vocal_path.relative_to(job_dir).as_posix(),
            "raw_relative": raw_path.relative_to(job_dir).as_posix(),
            "cleaned_relative": cleaned_path.relative_to(job_dir).as_posix(),
            "cleanup_report_relative": cleanup_report_path.relative_to(job_dir).as_posix(),
            "analysis_reused_from_original": True,
            "raw_count": len(raw_events),
            "cleaned_count": len(cleaned_events),
            "cleanup_schema_version": cleanup.report.get("schema_version"),
        }
        self._persist_generation_state(job_id, artifacts=generation_artifacts, generation=generation)
        generation_analysis = original_analysis.model_copy(
            update={
                "note_events": list(cleaned_events),
                "metadata": {
                    **original_analysis.metadata,
                    "engine": "game",
                    "source_kind": "vocal",
                    "source_stems": ["vocals"],
                    "prepared_audio": {"vocals": os.fspath(vocal_path)},
                    "analysis_reused_from_original": True,
                    "engine_details": {"vocals": {**result.metadata, "adapter": "game"}},
                    "vocal_cleanup": cleanup.report,
                },
                "warnings": [*original_analysis.warnings, *result.warnings],
            }
        )
        analysis_path = generation_root / "analysis.cleaned.json"
        _safe_json(analysis_path, generation_analysis.model_dump(mode="json"))
        generation_artifacts.append(
            self.manager._register(job_dir, analysis_path, artifact_id="v2-vocal-analysis-cleaned", kind="analysis_full_json", label="清理后高精度分析", media_type="application/json", stem_id="vocals")
        )
        service_dir = generation_root / "high-accuracy"
        self.manager._set_phase(job_id, "rendering")
        service = HighAccuracyArtifactService()
        try:
            build = service.build(
                instrument_id="vocals",
                title=str(options.get("title") or "人声主旋律"),
                program=0,
                is_drum=False,
                events=cleaned_events,
                analysis=generation_analysis,
                output_dir=service_dir,
                variant="game-cleaned",
                overwrite=False,
            )
        except HighAccuracyServiceError as exc:
            failure_artifacts = [*generation_artifacts, *self._register_high_accuracy_failure(job_id, exc, prefix="v2-vocal-high-accuracy", label="人声高精度处理", stem_id="vocals")]
            failed_generation = {**generation, "stage": exc.stage, "status": "failed", "failure": {"instrument_id": exc.instrument_id, "stage": exc.stage, "cause": exc.cause}}
            self._persist_generation_state(job_id, artifacts=failure_artifacts, generation=failed_generation)
            self.manager._update(
                job_id,
                status="failed",
                phase="failed",
                finished_at=_utc_now(),
                error={"code": "high_accuracy_failed", "message": str(exc), "instrument_id": exc.instrument_id, "stage": exc.stage},
            )
            # The service failure manifest and structured task error are
            # already persisted.  Returning keeps the stage/instrument fields
            # visible to the V2 API instead of letting the queue worker
            # rewrite them as a generic exception.
            return
        except Exception as exc:  # noqa: BLE001 - persist a structured service failure
            failed_generation = {**generation, "stage": "service", "status": "failed", "failure": {"instrument_id": "vocals", "stage": "service", "cause": str(exc)}}
            self._persist_generation_state(job_id, artifacts=generation_artifacts, generation=failed_generation)
            self.manager._update(
                job_id,
                status="failed",
                phase="failed",
                finished_at=_utc_now(),
                error={"code": "high_accuracy_failed", "message": str(exc), "instrument_id": "vocals", "stage": "service"},
            )
            return
        service_artifacts, score_ids = self._register_high_accuracy_result(
            job_id,
            build,
            prefix="v2-vocal-high-accuracy",
            label="人声主旋律",
            stem_id="vocals",
            family="vocal",
        )
        artifacts = [*generation_artifacts, *service_artifacts]
        generation = {
            **generation,
            "stage": "completed",
            "status": "completed",
            "service_variant": build.variant,
            "service_relative": service_dir.relative_to(job_dir).as_posix(),
            "manifest_relative": build.manifest_path.relative_to(job_dir).as_posix(),
            "score_artifact_ids": score_ids,
        }
        separation = dict((state.get("v2") or {}).get("separation") or {})
        separation["model"] = model
        separation.setdefault("model_info", self._vocal_model_info(model))
        with self.manager._lock:
            current = self.manager._read(job_id)
            current_v2 = dict(current.get("v2", {}))
            current_v2.update(
                {
                    **HIGH_ACCURACY_V2_METADATA,
                    "stage": "vocal_complete",
                    "route": self._vocal_route(model),
                    "analysis": _analysis_suggestion(generation_analysis),
                    "generation": generation,
                    "separation": separation,
                    "progress_detail": {"status": "completed", "engine": "game", "stem": "vocals", "model": model, "separation_model": model},
                }
            )
            current.update(
                {
                    "v2": current_v2,
                    "status": "completed",
                    "phase": "completed",
                    "finished_at": _utc_now(),
                    "error": None,
                    "artifacts": artifacts,
                    "progress": 1.0,
                    "warnings": list(dict.fromkeys([*generation_analysis.warnings, *separation.get("warnings", [])])),
                    "summary": {
                        **HIGH_ACCURACY_V2_METADATA,
                        "source_kind": "vocal",
                        "route": self._vocal_route(model),
                        "stage": "completed",
                        "note_count": len(generation_analysis.note_events),
                        "raw_note_count": len(raw_events),
                        "cleanup_report": cleanup.report,
                        "score_artifact_ids": score_ids,
                        "generation": generation,
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(current)
        self.manager._log(job_id, f"V2 vocal completed from vocals stem: {len(cleaned_events)} cleaned notes, {len(artifacts)} artifacts")

    def _run_instrumental_export(self, job_id: str) -> None:
        state = self.manager._read(job_id)
        v2 = dict(state.get("v2", {}))
        tracks = list(v2.get("tracks", []))
        notes = list(v2.get("notes", []))
        selection = dict(v2.get("selection") or {})
        revision = int(selection.get("revision", v2.get("selection_revision", 0)))
        selected_ids = [str(value) for value in selection.get("selected_track_ids", [])]
        selected_set = set(selected_ids)
        track_by_id = {str(track.get("track_id")): track for track in tracks}
        selected_tracks = [track_by_id[track_id] for track_id in selected_ids if track_id in track_by_id]
        base_analysis = self._persisted_instrumental_analysis(job_id, state)
        bpm = float(selection.get("bpm_override") if selection.get("bpm_override") is not None else (base_analysis.bpm if base_analysis else 120.0))
        key = normalize_key(str(selection.get("key_override") or (base_analysis.key if base_analysis else "C")))
        time_signature = normalize_time_signature(
            str(selection.get("time_signature_override") or (base_analysis.time_signature if base_analysis else "4/4"))
        )
        bpm_manual = bool(selection.get("bpm_override_explicit", False))
        key_manual = bool(selection.get("key_override_explicit", False))
        time_signature_manual = bool(selection.get("time_signature_override_explicit", False))
        notes_with_ids: list[dict[str, Any]] = []
        for note in notes:
            track_key = (str(note.get("instrument_group")), int(note.get("program", 0)), bool(note.get("is_drum", False)))
            track_id = stable_track_id(*track_key)
            if track_id in selected_set:
                notes_with_ids.append({**note, "track_id": track_id})
        selected_pitched = [track for track in selected_tracks if not bool(track.get("is_drum"))]
        output = self._output_dir(job_id) / "selections" / f"rev-{revision:04d}"
        if output.exists() and output.is_symlink():
            raise ValueError("invalid selection output path")
        output.mkdir(parents=True, exist_ok=True)
        self.manager._set_phase(job_id, "rendering")
        title = str(state.get("options", {}).get("title") or "Untitled")
        midi_path = write_unquantized_midi(
            notes_with_ids,
            output / "selected.mid",
            title=f"{title} selection r{revision}",
            bpm=bpm,
        )
        job_dir = self.manager._safe_job_dir(job_id)
        artifacts: list[dict[str, Any]] = [
            self.manager._register(
                job_dir,
                midi_path,
                artifact_id=f"v2-selection-r{revision}-midi",
                kind="selected_midi",
                label=f"选择版本 {revision} MIDI（含鼓试听）",
                media_type="audio/midi",
            )
        ]
        score_artifact_ids: list[str] = []
        track_failures: list[dict[str, Any]] = []
        successful_pitched: list[dict[str, Any]] = []
        for track in selected_tracks:
            track_id = str(track["track_id"])
            label = str(track.get("label_zh") or instrument_label_zh(str(track.get("instrument_group"))))
            track_notes = [note for note in notes_with_ids if note.get("track_id") == track_id]
            try:
                track_midi = write_unquantized_midi(
                    track_notes,
                    output / f"{track_id}.mid",
                    title=f"{title} {label}",
                    bpm=bpm,
                )
                artifacts.append(
                    self.manager._register(
                        job_dir,
                        track_midi,
                        artifact_id=f"v2-selection-r{revision}-{track_id}-midi",
                        kind="instrument_preview_midi",
                        label=f"{label}试听 MIDI",
                        media_type="audio/midi",
                        stem_id=track_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve one-track continuation
                track_failures.append({"track_id": track_id, "label": label, "stage": "preview_midi", "error": str(exc)})
                continue
            if bool(track.get("is_drum")):
                continue
            try:
                rendered = self._render_track_score(
                    job_id,
                    output,
                    track,
                    track_notes,
                    title,
                    bpm=bpm,
                    key=key,
                    time_signature=time_signature,
                    base_analysis=base_analysis,
                    bpm_manual=bpm_manual,
                    key_manual=key_manual,
                    time_signature_manual=time_signature_manual,
                )
            except HighAccuracyServiceError as exc:
                track_failures.append({"track_id": track_id, "label": label, "stage": exc.stage, "error": exc.cause})
                artifacts.extend(list(getattr(exc, "v2_artifacts", [])))
                continue
            except Exception as exc:  # noqa: BLE001 - preserve one-track continuation
                track_failures.append({"track_id": track_id, "label": label, "stage": "service", "error": str(exc)})
                continue
            score_artifact_ids.extend(item["artifact_id"] for item in rendered if item.get("kind", "").endswith(("score_json", "score_midi", "score_svg", "score_svg_long")))
            artifacts.extend(rendered)
            successful_pitched.append(track)

        merged_artifacts: list[dict[str, Any]] = []
        merge_requested = bool(selection.get("merge_main_melody", False))
        if merge_requested and successful_pitched:
            merged_notes = self._main_melody_notes(notes_with_ids, {str(track["track_id"]) for track in successful_pitched})
            merged_track = {
                "track_id": "main-melody",
                "label_zh": "主旋律（合并）",
                "instrument_group": "main_melody",
                "program": 0,
                "is_drum": False,
            }
            try:
                merged_artifacts = self._render_track_score(
                    job_id,
                    output,
                    merged_track,
                    merged_notes,
                    title,
                    basename="main-melody",
                    label_override="主旋律（合并）",
                    bpm=bpm,
                    key=key,
                    time_signature=time_signature,
                    base_analysis=base_analysis,
                    bpm_manual=bpm_manual,
                    key_manual=key_manual,
                    time_signature_manual=time_signature_manual,
                )
                artifacts.extend(merged_artifacts)
                score_artifact_ids.extend(item["artifact_id"] for item in merged_artifacts if item.get("kind", "").endswith(("score_json", "score_midi", "score_svg", "score_svg_long")))
            except HighAccuracyServiceError as exc:
                track_failures.append({"track_id": "main-melody", "label": "主旋律（合并）", "stage": exc.stage, "error": exc.cause})
                artifacts.extend(list(getattr(exc, "v2_artifacts", [])))
            except Exception as exc:  # noqa: BLE001 - preserve main-melody continuation
                track_failures.append({"track_id": "main-melody", "label": "主旋律（合并）", "stage": "service", "error": str(exc)})

        score_refusal: dict[str, str] | None = None
        warnings: list[str] = []
        if not selected_pitched:
            score_refusal = {
                "code": "no_pitched_tracks",
                "message": "未选择有音高乐器，简谱已拒绝；仍可下载选中 MIDI 并试听鼓组。",
            }
            warnings.append(score_refusal["message"])
        elif not successful_pitched:
            score_refusal = {
                "code": "all_pitched_tracks_failed",
                "message": "所有选中的有音高乐器均未完成高精度谱面生成。",
            }
            warnings.append(score_refusal["message"])
        if track_failures:
            warnings.append(f"有 {len(track_failures)} 个乐器或产物阶段失败，详见任务结果中的 track_failures。")
        if merge_requested and successful_pitched:
            warnings.append("主旋律合并为单声部，结果会丢失和声；该产物不称为总谱。")

        page_artifacts = [item for item in artifacts if item.get("kind") in {"instrument_score_svg", "main_melody_score_svg"}]
        selection_zip_id: str | None = None
        if page_artifacts:
            selection_zip_id = f"v2-selection-r{revision}-svg-zip"
            zip_path = output / "selection-pages.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for item in page_artifacts:
                    page_path = (job_dir / str(item["relative_path"])).resolve()
                    if page_path == job_dir or job_dir not in page_path.parents or not page_path.is_file():
                        raise ValueError("选择版本分页 SVG 路径无效")
                    archive.write(page_path, arcname=page_path.relative_to(output).as_posix())
            artifacts.append(
                self.manager._register(
                    job_dir,
                    zip_path,
                    artifact_id=selection_zip_id,
                    kind="svg_zip",
                    label=f"选择版本 {revision} 全部分页 SVG 压缩包",
                    media_type="application/zip",
                )
            )

        overrides = {
            "bpm": bpm,
            "key": key,
            "time_signature": time_signature,
            "bpm_manual": bpm_manual,
            "key_manual": key_manual,
            "time_signature_manual": time_signature_manual,
            "sources": {
                "bpm": "manual" if bpm_manual else "beatnet",
                "key": "manual" if key_manual else "analysis",
                "time_signature": "manual" if time_signature_manual else "beatnet",
            },
        }
        selection_record = {
            "schema_version": "2.0",
            "revision": revision,
            "selected_track_ids": selected_ids,
            "selected_tracks": selected_tracks,
            "merge_main_melody": merge_requested,
            "overrides": overrides,
            "score_refusal": score_refusal,
            "track_failures": track_failures,
            "score_artifact_ids": score_artifact_ids,
            "midi_artifact_id": f"v2-selection-r{revision}-midi",
            "svg_zip_artifact_id": selection_zip_id,
            "metadata": {**HIGH_ACCURACY_V2_METADATA, "time_basis": "source_seconds", "velocity_policy": "playback_default", "note_event_velocity": None, "full_decode_reused": True},
        }
        selection_json = output / "selection.json"
        _safe_json(selection_json, selection_record)
        artifacts.append(
            self.manager._register(
                job_dir,
                selection_json,
                artifact_id=f"v2-selection-r{revision}-json",
                kind="selection_json",
                label=f"选择版本 {revision} 数据",
                media_type="application/json",
            )
        )
        with self.manager._lock:
            current = self.manager._read(job_id)
            current_v2 = dict(current.get("v2", {}))
            current_v2.update({**HIGH_ACCURACY_V2_METADATA, "score_refusal": score_refusal, "track_failures": track_failures, "progress_detail": {"status": "completed" if not score_refusal or score_refusal.get("code") == "no_pitched_tracks" else "failed", "revision": revision}})
            previous = [item for item in current.get("artifacts", []) if not str(item.get("artifact_id", "")).startswith(f"v2-selection-r{revision}-")]
            current.update(
                {
                    "status": "failed" if score_refusal and score_refusal.get("code") == "all_pitched_tracks_failed" else "completed",
                    "phase": "failed" if score_refusal and score_refusal.get("code") == "all_pitched_tracks_failed" else "completed",
                    "finished_at": _utc_now(),
                    "error": ({"code": "high_accuracy_all_tracks_failed", "message": score_refusal["message"], "track_failures": track_failures} if score_refusal and score_refusal.get("code") == "all_pitched_tracks_failed" else None),
                    "progress": 1.0,
                    "warnings": warnings,
                    "artifacts": [*previous, *artifacts],
                    "summary": {
                        **HIGH_ACCURACY_V2_METADATA,
                        "source_kind": "instrumental",
                        "route": {"engine": "muscriptor", "use_demucs": False},
                        "selection_revision": revision,
                        "selected_track_ids": selected_ids,
                        "selected_pitched_track_ids": [str(track["track_id"]) for track in selected_pitched],
                        "successful_pitched_track_ids": [str(track["track_id"]) for track in successful_pitched],
                        "drum_track_ids": [str(track["track_id"]) for track in selected_tracks if bool(track.get("is_drum"))],
                        "score_refusal": score_refusal,
                        "track_failures": track_failures,
                        "merge_main_melody": merge_requested,
                        "overrides": overrides,
                    },
                    "v2": {**current_v2, "stage": "export"},
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(current)
        # The explicit failed state and per-track diagnostics have already
        # been persisted above.  Returning prevents the generic worker from
        # replacing ``high_accuracy_all_tracks_failed`` with ``runtime_error``.
        self.manager._log(job_id, f"V2 selection revision {revision} completed: {len(artifacts)} new artifacts")

    def _render_track_score(
        self,
        job_id: str,
        output: Path,
        track: Mapping[str, Any],
        notes: Sequence[Mapping[str, Any]],
        title: str,
        *,
        basename: str | None = None,
        label_override: str | None = None,
        bpm: float = 120.0,
        key: str = "C",
        time_signature: str = "4/4",
        base_analysis: MusicAnalysis | None = None,
        bpm_manual: bool = False,
        key_manual: bool = False,
        time_signature_manual: bool = False,
    ) -> list[dict[str, Any]]:
        track_id = str(track.get("track_id"))
        label = label_override or str(track.get("label_zh") or instrument_label_zh(str(track.get("instrument_group"))))
        if not notes:
            raise NoNotesError(f"NoNotes: selected track {track_id} has no events")
        events = [
            NoteEvent(
                start_sec=float(note["start_sec"]),
                end_sec=float(note["end_sec"]),
                midi=int(note["pitch"]),
                confidence=float(note["confidence"]) if note.get("confidence") is not None else None,
                voice_id=str(note.get("voice_id") or f"{track_id}:voice-0"),
                source="muscriptor-main-melody" if track_id == "main-melody" else "muscriptor",
                velocity=int(note["velocity"]) if note.get("velocity") not in {None, 0} else None,
                stem_id=track_id,
                metadata={
                    "instrument_group": str(track.get("instrument_group", "unknown")),
                    "program": int(track.get("program", 0)),
                    "track_id": track_id,
                    "playback_default": 80,
                },
            )
            for note in notes
        ]
        if base_analysis is None:
            raise ValueError("高精度乐器分谱必须使用已持久化的 BeatNet 分析")
        analysis = self._analysis_for_events(
            base_analysis,
            events,
            bpm=bpm,
            key=key,
            time_signature=time_signature,
            bpm_manual=bpm_manual,
            key_manual=key_manual,
            time_signature_manual=time_signature_manual,
            metadata_extra={
                "instrument_group": str(track.get("instrument_group", "unknown")),
                "program": int(track.get("program", 0)),
                "track_id": track_id,
                "selection_score_kind": "main_melody" if track_id == "main-melody" else "instrument_part",
                "harmony_loss": track_id == "main-melody",
                "velocity_policy": "preserve_note_velocity",
            },
        )
        score_title = f"{title} {label}分谱"
        prefix = f"v2-selection-r{self._revision_from_path(output)}-{track_id}"
        family = "main_melody" if track_id == "main-melody" else "instrument"
        service_dir = output / ("main-melody" if track_id == "main-melody" else track_id) / "high-accuracy"
        try:
            result = HighAccuracyArtifactService().build(
                instrument_id=track_id,
                title=score_title,
                program=int(track.get("program", 0)),
                is_drum=False,
                events=events,
                analysis=analysis,
                output_dir=service_dir,
                variant="main-melody" if track_id == "main-melody" else "instrument-part",
                overwrite=True,
            )
        except HighAccuracyServiceError as exc:
            failure_artifacts = self._register_high_accuracy_failure(
                job_id,
                exc,
                prefix=prefix,
                label=label,
                stem_id=track_id,
            )
            exc.v2_artifacts = failure_artifacts
            raise
        artifacts, _score_ids = self._register_high_accuracy_result(
            job_id,
            result,
            prefix=prefix,
            label=label,
            stem_id=track_id,
            family=family,
        )
        return artifacts

    @staticmethod
    def _selection_analysis_metadata(
        base_analysis: MusicAnalysis,
        *,
        bpm: float,
        key: str | None = None,
        time_signature: str,
        bpm_manual: bool = False,
        key_manual: bool = False,
        time_signature_manual: bool = False,
    ) -> dict[str, Any]:
        """Apply effective selection values while preserving their sources."""

        metadata = deepcopy(dict(base_analysis.metadata))
        beat_grid = metadata.get("beat_grid")
        if isinstance(beat_grid, Mapping):
            updated_grid = deepcopy(dict(beat_grid))
            if bpm_manual:
                # Keep the detected beat phase/local shape, but scale score
                # beat positions only when the user explicitly changed BPM.
                tempo = deepcopy(dict(updated_grid.get("tempo") or {}))
                detected = float(base_analysis.bpm)
                scale = float(bpm) / detected if detected > 0 else 1.0
                tempo["selected_bpm"] = float(bpm)
                tempo["manual_bpm"] = float(bpm)
                tempo["manual_scale"] = scale
                tempo["selection_reason"] = "用户选择 BPM 优先；导出保留 BeatNet 首拍与局部拍点"
                updated_grid["tempo"] = tempo
                mapping = deepcopy(dict(updated_grid.get("mapping") or {}))
                mapping["manual_bpm_scale"] = scale
                updated_grid["mapping"] = mapping
            if time_signature_manual:
                meter = deepcopy(dict(updated_grid.get("time_signature") or {}))
                meter["selected"] = time_signature
                meter["source"] = "manual"
                updated_grid["time_signature"] = meter
            metadata["beat_grid"] = updated_grid
        selected_key = normalize_key(key or base_analysis.key)
        metadata.update(
            {
                "manual_bpm_override": bool(bpm_manual),
                "manual_key_override": bool(key_manual),
                "manual_time_signature_override": bool(time_signature_manual),
                "selected_bpm": float(bpm),
                "selected_key": selected_key,
                "selected_time_signature": time_signature,
                "override_sources": {
                    "bpm": "manual" if bpm_manual else metadata.get("beat_source", "beatnet"),
                    "key": "manual" if key_manual else metadata.get("key_source", "analysis"),
                    "time_signature": "manual" if time_signature_manual else metadata.get("time_signature_source", "beatnet"),
                },
            }
        )
        return metadata

    @staticmethod
    def _revision_from_path(path: Path) -> int:
        match = re.search(r"rev-(\d+)", path.as_posix())
        return int(match.group(1)) if match else 0

    @staticmethod
    def _main_melody_notes(notes: Sequence[Mapping[str, Any]], selected_ids: set[str]) -> list[dict[str, Any]]:
        pitched = [note for note in notes if str(note.get("track_id")) in selected_ids and not bool(note.get("is_drum"))]
        by_start: dict[float, dict[str, Any]] = {}
        for note in pitched:
            key = round(float(note["start_sec"]), 5)
            current = by_start.get(key)
            if current is None or int(note["pitch"]) > int(current["pitch"]):
                by_start[key] = note
        return sorted(by_start.values(), key=lambda item: (float(item["start_sec"]), int(item["pitch"])))

    @staticmethod
    def _audio_media_type(path: Path) -> str:
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
        }.get(path.suffix.lower(), "application/octet-stream")
