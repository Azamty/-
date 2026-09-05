"""V2 persistent task stages built on the existing single JobManager worker."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any, Mapping, Sequence

from .jianpu_score.domain import MusicAnalysis, NoteEvent, normalize_key, normalize_time_signature
from .jianpu_score.pipeline import run_pipeline
from .jianpu_score.quantize import NoNotesError, quantize_events
from .jianpu_score.render import render_score, write_score_json
from .muscriptor_v2 import (
    instrument_label_zh,
    stable_track_id,
    write_unquantized_midi,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PYTHON = ROOT / ".venv-model-muscriptor" / "Scripts" / "python.exe"
V2_SOURCE_KINDS = frozenset({"instrumental", "vocal"})
V2_SOURCE_LABELS = {"instrumental": "伴奏/纯音乐", "vocal": "人声"}


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

    def create_job(self, *, original_name: str, source_kind: str, title: str) -> tuple[str, Path]:
        if source_kind not in V2_SOURCE_KINDS:
            raise ValueError("V2 source_kind must be instrumental or vocal")
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
                "stage": "recognize",
                "source_kind": source_kind,
                "source_label": V2_SOURCE_LABELS[source_kind],
                "route": {"engine": engine, "use_demucs": False},
                "tracks": [],
                "notes": [],
                "selection_revision": 0,
                "selection": None,
                "selection_history": [],
                "score_refusal": None,
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
            bpm = 120.0 if bpm_override is None else float(bpm_override)
            if not math.isfinite(bpm) or bpm <= 0 or bpm > 400:
                raise ValueError("bpm_override 必须在 0 到 400 之间")
            key = normalize_key(key_override or "C")
            time_signature = normalize_time_signature(time_signature_override or "4/4")
            revision = int(state.get("v2", {}).get("selection_revision", 0)) + 1
            selection = {
                "revision": revision,
                "selected_track_ids": requested,
                "merge_main_melody": bool(merge_main_melody),
                "bpm_override": bpm,
                "key_override": key,
                "time_signature_override": time_signature,
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
            "selection_revision": v2.get("selection_revision", 0),
            "selection": deepcopy(v2.get("selection")),
            "score_refusal": deepcopy(v2.get("score_refusal")),
        }

    def run(self, job_id: str) -> None:
        state = self.manager._read(job_id)
        v2 = state.get("v2", {})
        if v2.get("stage") == "recognize":
            if v2.get("source_kind") == "vocal":
                self._run_vocal(job_id)
            else:
                self._run_instrumental_recognition(job_id)
            return
        if v2.get("stage") == "export":
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
        output = self._output_dir(job_id) / "v2-recognition"
        if output.exists() and output.is_symlink():
            raise ValueError("invalid V2 recognition output path")
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
        job_dir = self.manager._safe_job_dir(job_id)
        artifacts = [
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
        ]
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
                    "stage": "selection_ready",
                    "tracks": tracks,
                    "notes": notes,
                    "recognition": {
                        "engine": "muscriptor",
                        "model": recognition.get("model", "medium"),
                        "device": recognition.get("device"),
                        "metadata": recognition.get("metadata", {}),
                    },
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
                        "source_kind": "instrumental",
                        "route": {"engine": "muscriptor", "use_demucs": False},
                        "note_count": len(notes),
                        "track_count": len(tracks),
                        "instrument_counts": recognition.get("instrument_counts", {}),
                        "selection_ready": True,
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(state)
        self.manager._log(job_id, f"MuScriptor selection_ready: {len(tracks)} tracks, {len(notes)} notes")

    def _run_vocal(self, job_id: str) -> None:
        self.manager._set_phase(job_id, "recognizing")
        output = self._output_dir(job_id) / "vocal"
        options = dict(self.manager._read(job_id).get("options", {}))

        def progress(phase: str) -> None:
            # Passing mixed here is deliberate: V2 vocal is routed directly to
            # GAME and must never trigger the V1 Demucs stem preparation.
            self.manager._set_phase(job_id, phase)

        analysis, score, render_artifacts = run_pipeline(
            self.manager.input_path(job_id),
            output,
            engine="game",
            voice_mode="monophonic",
            source_kind="mixed",
            separate=False,
            language="mixed",
            title=options.get("title"),
            progress_callback=progress,
        )
        self.manager._set_phase(job_id, "packaging")
        artifacts = self.manager._package_artifacts(job_id, score, analysis, render_artifacts)
        job_dir = self.manager._safe_job_dir(job_id)
        artifacts.insert(
            0,
            self.manager._register(
                job_dir,
                self.manager.input_path(job_id),
                artifact_id="v2-source-audio",
                kind="source_audio",
                label="原始人声音频",
                media_type=self._audio_media_type(self.manager.input_path(job_id)),
            ),
        )
        for artifact in artifacts:
            if artifact.get("kind") in {"score_svg", "score_json", "jianpu_source", "lilypond_source", "midi"}:
                artifact["label"] = "人声主旋律" + (f"（{artifact['label']}）" if artifact.get("label") else "")
        with self.manager._lock:
            state = self.manager._read(job_id)
            state.update(
                {
                    "status": "completed",
                    "phase": "completed",
                    "finished_at": _utc_now(),
                    "error": None,
                    "artifacts": artifacts,
                    "progress": 1.0,
                    "warnings": list(dict.fromkeys([*analysis.warnings, *score.warnings])),
                    "summary": {
                        "source_kind": "vocal",
                        "route": {"engine": "game", "use_demucs": False},
                        "note_count": len(analysis.note_events),
                        "voice_count": len(score.voices),
                        "total_ticks": score.total_ticks,
                        "bpm": score.bpm,
                        "key": score.key,
                        "time_signature": score.time_signature,
                    },
                    "v2": {
                        **state.get("v2", {}),
                        "stage": "vocal_complete",
                        "progress_detail": {"status": "completed"},
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(state)
        self.manager._log(job_id, f"V2 vocal completed: {len(analysis.note_events)} notes, {len(artifacts)} artifacts")

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
        bpm = float(selection.get("bpm_override") or 120.0)
        key = normalize_key(str(selection.get("key_override") or "C"))
        time_signature = normalize_time_signature(str(selection.get("time_signature_override") or "4/4"))
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
        for track in selected_tracks:
            track_id = str(track["track_id"])
            track_notes = [note for note in notes_with_ids if note.get("track_id") == track_id]
            track_midi = write_unquantized_midi(
                track_notes,
                output / f"{track_id}.mid",
                title=f"{title} {track.get('label_zh') or instrument_label_zh(str(track.get('instrument_group')))}",
                bpm=bpm,
            )
            artifacts.append(
                self.manager._register(
                    job_dir,
                    track_midi,
                    artifact_id=f"v2-selection-r{revision}-{track_id}-midi",
                    kind="instrument_midi",
                    label=f"{track.get('label_zh') or instrument_label_zh(str(track.get('instrument_group')))} MIDI",
                    media_type="audio/midi",
                    stem_id=track_id,
                )
            )
            if bool(track.get("is_drum")):
                continue
            rendered = self._render_track_score(
                job_id,
                output,
                track,
                track_notes,
                title,
                bpm=bpm,
                key=key,
                time_signature=time_signature,
            )
            score_artifact_ids.extend(item["artifact_id"] for item in rendered)
            artifacts.extend(rendered)

        merged_artifacts: list[dict[str, Any]] = []
        merge_requested = bool(selection.get("merge_main_melody", False))
        if merge_requested and selected_pitched:
            merged_notes = self._main_melody_notes(notes_with_ids, selected_set)
            merged_track = {
                "track_id": "main-melody",
                "label_zh": "主旋律（合并）",
                "instrument_group": "main_melody",
                "program": 0,
                "is_drum": False,
            }
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
            )
            artifacts.extend(merged_artifacts)
            score_artifact_ids.extend(item["artifact_id"] for item in merged_artifacts)

        score_refusal: dict[str, str] | None = None
        warnings: list[str] = []
        if not selected_pitched:
            score_refusal = {
                "code": "no_pitched_tracks",
                "message": "未选择有音高乐器，简谱已拒绝；仍可下载选中 MIDI 并试听鼓组。",
            }
            warnings.append(score_refusal["message"])
        if merge_requested and selected_pitched:
            warnings.append("主旋律合并为单声部，结果会丢失和声；该产物不称为总谱。")

        selection_record = {
            "schema_version": "2.0",
            "revision": revision,
            "selected_track_ids": selected_ids,
            "selected_tracks": selected_tracks,
            "merge_main_melody": merge_requested,
            "overrides": {
                "bpm": bpm,
                "key": key,
                "time_signature": time_signature,
            },
            "score_refusal": score_refusal,
            "score_artifact_ids": score_artifact_ids,
            "midi_artifact_id": f"v2-selection-r{revision}-midi",
            "metadata": {
                "time_basis": "source_seconds",
                "velocity_policy": "playback_default",
                "note_event_velocity": None,
                "full_decode_reused": True,
            },
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
            current_v2["score_refusal"] = score_refusal
            current_v2["progress_detail"] = {"status": "completed", "revision": revision}
            current.update(
                {
                    "status": "completed",
                    "phase": "completed",
                    "finished_at": _utc_now(),
                    "error": None,
                    "progress": 1.0,
                    "warnings": warnings,
                    "artifacts": [*current.get("artifacts", []), *artifacts],
                    "summary": {
                        "source_kind": "instrumental",
                        "route": {"engine": "muscriptor", "use_demucs": False},
                        "selection_revision": revision,
                        "selected_track_ids": selected_ids,
                        "selected_pitched_track_ids": [str(track["track_id"]) for track in selected_pitched],
                        "drum_track_ids": [str(track["track_id"]) for track in selected_tracks if bool(track.get("is_drum"))],
                        "score_refusal": score_refusal,
                        "merge_main_melody": merge_requested,
                        "overrides": {
                            "bpm": bpm,
                            "key": key,
                            "time_signature": time_signature,
                        },
                    },
                    "v2": {**current_v2, "stage": "export"},
                    "updated_at": _utc_now(),
                }
            )
            self.manager._write(current)
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
    ) -> list[dict[str, Any]]:
        track_id = str(track.get("track_id"))
        label = label_override or str(track.get("label_zh") or instrument_label_zh(str(track.get("instrument_group"))))
        if not notes:
            raise NoNotesError(f"NoNotes: selected track {track_id} has no events")
        duration = max(float(note["end_sec"]) for note in notes)
        events = [
            NoteEvent(
                start_sec=float(note["start_sec"]),
                end_sec=float(note["end_sec"]),
                midi=int(note["pitch"]),
                confidence=None,
                voice_id=f"{track_id}:voice-0",
                source="muscriptor",
                velocity=None,
                stem_id=track_id,
                metadata={
                    "instrument_group": str(track.get("instrument_group", "unknown")),
                    "program": int(track.get("program", 0)),
                    "playback_default": 80,
                },
            )
            for note in notes
        ]
        analysis = MusicAnalysis(
            sample_rate=16000,
            duration_sec=max(duration, 0.1),
            bpm=bpm,
            key=key,
            time_signature=time_signature,
            note_events=events,
            metadata={
                "engine": "muscriptor",
                "source_kind": "instrumental",
                "time_basis": "source_seconds",
                "velocity_policy": "playback_default",
            },
        )
        score_title = f"{title} {label}分谱"
        score = quantize_events(events, analysis, mode="monophonic" if track_id == "main-melody" else "polyphonic", title=score_title)
        score = score.model_copy(
            update={
                "metadata": {
                    **score.metadata,
                    "instrument_group": str(track.get("instrument_group", "unknown")),
                    "program": int(track.get("program", 0)),
                    "track_id": track_id,
                    "selection_score_kind": "main_melody" if track_id == "main-melody" else "instrument_part",
                    "harmony_loss": track_id == "main-melody",
                    "velocity_policy": "playback_default",
                }
            }
        )
        track_dir = output / ("main-melody" if track_id == "main-melody" else track_id)
        track_dir.mkdir(parents=True, exist_ok=True)
        render = render_score(score, track_dir, basename=basename or "score")
        write_score_json(score, track_dir / "score.json")
        job_dir = self.manager._safe_job_dir(job_id)
        prefix = f"v2-selection-r{self._revision_from_path(output)}-{track_id}"
        artifacts: list[dict[str, Any]] = []
        for index, path_text in enumerate(render.svg_paths, start=1):
            path = Path(path_text).resolve()
            artifacts.append(
                self.manager._register(
                    job_dir,
                    path,
                    artifact_id=f"{prefix}-svg-{index}",
                    kind="instrument_score_svg" if track_id != "main-melody" else "main_melody_svg",
                    label=f"{label}分谱第 {index} 页" if track_id != "main-melody" else "主旋律（合并）第 {index} 页",
                    media_type="image/svg+xml",
                    stem_id=track_id,
                    page=index,
                )
            )
        for path, suffix, kind, media_type, item_label in (
            (track_dir / "score.json", "json", "instrument_score_json", "application/json", f"{label}分谱数据"),
            (track_dir / "score.jly", "jly", "instrument_jianpu_source", "text/plain; charset=utf-8", f"{label}分谱源文本"),
            (track_dir / "score.ly", "ly", "instrument_lilypond_source", "text/plain; charset=utf-8", f"{label}分谱 LilyPond"),
        ):
            if path.is_file():
                artifacts.append(
                    self.manager._register(
                        job_dir,
                        path,
                        artifact_id=f"{prefix}-{suffix}",
                        kind=kind if track_id != "main-melody" else f"main_melody_{suffix}",
                        label=item_label,
                        media_type=media_type,
                        stem_id=track_id,
                    )
                )
        return artifacts

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
