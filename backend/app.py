"""FastAPI application for the local jianpu score desk."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import re
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api_models import ArtifactsResponse, JobResponse, V2SelectionRequest
from .job_manager import JobManager, _error_payload, safe_filename
from .jianpu_score.analysis import MAX_AUDIO_BYTES, SUPPORTED_EXTENSIONS, probe_audio
from .jianpu_score.capabilities import get_capabilities
from .jianpu_score.domain import normalize_key, normalize_time_signature
from .jianpu_score.pipeline import SUPPORTED_ENGINES
from .soundfont import ensure_soundfont, soundfont_status


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
UPLOAD_CHUNK_BYTES = 1024 * 1024
LANGUAGES = frozenset({"zh", "ja", "mixed"})
VOICE_MODES = frozenset({"monophonic", "polyphonic"})
SOURCE_KINDS = frozenset({"mixed", "vocal", "instrumental"})
MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024


class _PayloadTooLarge(Exception):
    """Internal signal raised while an ASGI request body is being received."""


class UploadSizeLimitMiddleware:
    """Reject oversized multipart bodies while Starlette is still receiving them.

    ``UploadFile`` validation happens after the multipart parser has consumed the
    request. This wrapper bounds the request body first, then the endpoint
    applies the exact audio-file limit after parsing the form.
    """

    def __init__(
        self,
        app: Any,
        *,
        max_upload_bytes: int = MAX_AUDIO_BYTES,
        overhead_bytes: int = MULTIPART_OVERHEAD_BYTES,
    ) -> None:
        if isinstance(max_upload_bytes, bool) or not isinstance(max_upload_bytes, int) or max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be a positive integer")
        if isinstance(overhead_bytes, bool) or not isinstance(overhead_bytes, int) or overhead_bytes < 0:
            raise ValueError("overhead_bytes must be a non-negative integer")
        self.app = app
        self.max_upload_bytes = max_upload_bytes
        self.overhead_bytes = overhead_bytes

    async def _send_rejection(self, send: Any) -> None:
        body = json.dumps(
            {
                "detail": {
                    "code": "upload_too_large",
                    "message": "上传内容超过允许的大小，请选择不超过 100 MB 的音频文件。",
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method", "").upper() != "POST"
            or scope.get("path") not in {"/api/jobs", "/api/v2/jobs"}
        ):
            await self.app(scope, receive, send)
            return

        limit = self.max_upload_bytes + self.overhead_bytes
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                content_length = int(declared)
            except (TypeError, ValueError):
                content_length = None
            if content_length is not None and (content_length < 0 or content_length > limit):
                await self._send_rejection(send)
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    raise _PayloadTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _PayloadTooLarge:
            await self._send_rejection(send)


def _form_error(message: str, *, status_code: int = 422) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": "invalid_request", "message": message})


def _optional_float(value: str | None, label: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError:
        raise _form_error(f"{label} 必须是有限数字") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise _form_error(f"{label} 必须是大于 0 的有限数字")
    return parsed


def _status_response(manager: JobManager, job_id: str) -> JobResponse:
    try:
        return JobResponse.model_validate(manager.get(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "任务不存在"}) from None


def _safe_upload_extension(filename: str | None) -> str:
    if not filename:
        raise _form_error("请提供 MP3、WAV、FLAC 或 M4A 音频文件")
    suffix = Path(filename.replace("\x00", "")).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise _form_error("仅支持 MP3、WAV、FLAC 或 M4A 文件")
    return suffix


def _effective_options(
    *,
    engine: str,
    voice_mode: str,
    source_kind: str,
    language: str,
    separate: bool,
    bpm: str | None,
    key: str | None,
    time_signature: str | None,
    title: str | None,
) -> dict[str, Any]:
    if engine not in SUPPORTED_ENGINES:
        raise _form_error(f"不支持的识别引擎：{engine}")
    if voice_mode not in VOICE_MODES:
        raise _form_error("声部模式必须是主旋律或多声部")
    if source_kind not in SOURCE_KINDS:
        raise _form_error("来源必须是人声、纯音乐或混合")
    if language not in LANGUAGES:
        raise _form_error("语言必须是中文、日文或混合")
    if engine == "specialist" and voice_mode == "monophonic" and source_kind == "mixed":
        raise _form_error("专用模型的主旋律需要选择人声或纯音乐，混合来源请切换多声部")
    requires_stems = source_kind in {"vocal", "instrumental"} or (
        voice_mode == "polyphonic" and source_kind == "mixed"
    )
    if requires_stems or engine == "specialist":
        # Specialist engines need Demucs stem identity.  The UI need not
        # expose this implementation detail as a second fragile checkbox.
        separate = True
    try:
        bpm_value = _optional_float(bpm, "BPM")
        key_value = normalize_key(key) if key is not None and key.strip() else None
        meter_value = normalize_time_signature(time_signature) if time_signature is not None and time_signature.strip() else None
    except HTTPException:
        raise
    except ValueError as exc:
        raise _form_error(str(exc)) from exc
    clean_title = safe_filename(title) if title and title.strip() else None
    return {
        "engine": engine,
        "voice_mode": voice_mode,
        "source_kind": source_kind,
        "language": language,
        "separate": bool(separate),
        "bpm_override": bpm_value,
        "key_override": key_value,
        "time_signature_override": meter_value,
        "title": clean_title,
    }


async def _write_upload(
    manager: JobManager,
    job_id: str,
    upload: UploadFile,
    *,
    max_bytes: int = MAX_AUDIO_BYTES,
) -> tuple[int, Path]:
    input_path = manager.input_path(job_id)
    total = 0
    try:
        with input_path.open("wb") as destination:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={"code": "upload_too_large", "message": "音频文件超过上传大小限制"},
                    )
                destination.write(chunk)
    except HTTPException:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return total, input_path


def _artifact_response(manager: JobManager, job_id: str, *, prefix: str = "/api/jobs") -> ArtifactsResponse:
    try:
        items = manager.list_artifacts(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "任务不存在"}) from None
    for item in items:
        item["url"] = f"{prefix}/{job_id}/artifacts/{item['artifact_id']}"
    return ArtifactsResponse.model_validate({"job_id": job_id, "artifacts": items})


@asynccontextmanager
async def _lifespan(app: FastAPI):
    manager: JobManager = app.state.jobs
    manager.start()
    try:
        yield
    finally:
        manager.stop()


def create_app(
    *,
    jobs_root: str | Path | None = None,
    max_upload_bytes: int = MAX_AUDIO_BYTES,
    multipart_overhead_bytes: int = MULTIPART_OVERHEAD_BYTES,
) -> FastAPI:
    if isinstance(max_upload_bytes, bool) or not isinstance(max_upload_bytes, int) or max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be a positive integer")
    if isinstance(multipart_overhead_bytes, bool) or not isinstance(multipart_overhead_bytes, int) or multipart_overhead_bytes < 0:
        raise ValueError("multipart_overhead_bytes must be a non-negative integer")
    manager = JobManager(jobs_root or (ROOT / "artifacts" / "jobs"))
    app = FastAPI(title="谱面工作台", version="0.1.0", lifespan=_lifespan)
    app.state.jobs = manager
    app.add_middleware(
        UploadSizeLimitMiddleware,
        max_upload_bytes=max_upload_bytes,
        overhead_bytes=multipart_overhead_bytes,
    )

    if FRONTEND_DIST.is_dir():
        assets = FRONTEND_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "jianpu-score"}

    @app.get("/api/capabilities")
    async def capabilities() -> dict[str, Any]:
        value = get_capabilities()
        value["api"] = {
            "statuses": ["uploading", "queued", "probing", "separating", "vocal_ready", "recognizing", "quantizing", "rendering", "exporting", "packaging", "selection_ready", "completed", "failed", "interrupted"],
            "retention_hours": 24,
            "single_worker": True,
            "v2": {
                "sources": {"instrumental": "伴奏/纯音乐", "vocal": "人声"},
                "instrumental_engine": "MuScriptor medium CUDA",
                "vocal_engine": "GAME",
                "routes": {
                    "instrumental": {"engine": "muscriptor", "model": "medium", "use_demucs": False},
                    "vocal": {"engine": "game", "use_demucs": True, "separation_engine": "demucs", "separation_model": "htdemucs"},
                },
            },
            "upload_extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_upload_bytes": max_upload_bytes,
            "max_duration_sec": 15 * 60,
        }
        return value

    @app.post("/api/jobs", response_model=JobResponse, status_code=202)
    async def create_job(
        file: UploadFile = File(...),
        engine: str = Form("basic-pitch"),
        voice_mode: str = Form("monophonic"),
        source_kind: str = Form("instrumental"),
        language: str = Form("mixed"),
        separate: bool = Form(False),
        bpm: str | None = Form(None),
        key: str | None = Form(None),
        time_signature: str | None = Form(None),
        title: str | None = Form(None),
    ) -> JobResponse:
        suffix = _safe_upload_extension(file.filename)
        options = _effective_options(
            engine=engine,
            voice_mode=voice_mode,
            source_kind=source_kind,
            language=language,
            separate=separate,
            bpm=bpm,
            key=key,
            time_signature=time_signature,
            title=title or Path(safe_filename(file.filename)).stem,
        )
        job_id, input_path = manager.create_job(original_name=safe_filename(file.filename), options=options)
        try:
            bytes_count, input_path = await _write_upload(manager, job_id, file, max_bytes=max_upload_bytes)
            probe = await asyncio.to_thread(probe_audio, input_path)
            manager.set_input_info(job_id, bytes_count=bytes_count, duration_sec=float(probe["duration_sec"]))
            manager.enqueue(job_id)
            return _status_response(manager, job_id)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"code": "upload_rejected", "message": str(exc.detail)}
            manager.fail_immediate(job_id, {"code": str(detail.get("code", "upload_rejected")), "message": str(detail.get("message", "上传被拒绝"))})
            raise
        except Exception as exc:
            error = _error_payload(exc)
            manager.fail_immediate(job_id, error)
            try:
                input_path.unlink(missing_ok=True)
            except OSError:
                pass
            status = 413 if "exceeds" in str(exc).lower() else 422
            raise HTTPException(status_code=status, detail=error) from exc
        finally:
            await file.close()

    @app.post("/api/v2/jobs", response_model=JobResponse, status_code=202)
    async def create_v2_job(
        file: UploadFile = File(...),
        source_kind: str = Form("instrumental"),
        title: str | None = Form(None),
    ) -> JobResponse:
        suffix = _safe_upload_extension(file.filename)
        if source_kind not in {"instrumental", "vocal"}:
            raise _form_error("V2 来源只能是伴奏/纯音乐或人声")
        original_name = safe_filename(file.filename)
        clean_title = safe_filename(title) if title and title.strip() else Path(original_name).stem
        job_id, input_path = manager.create_v2_job(
            original_name=original_name,
            source_kind=source_kind,
            title=clean_title,
        )
        try:
            bytes_count, input_path = await _write_upload(manager, job_id, file, max_bytes=max_upload_bytes)
            probe = await asyncio.to_thread(probe_audio, input_path)
            manager.set_input_info(job_id, bytes_count=bytes_count, duration_sec=float(probe["duration_sec"]))
            manager.enqueue(job_id)
            return _status_response(manager, job_id)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"code": "upload_rejected", "message": str(exc.detail)}
            manager.fail_immediate(job_id, {"code": str(detail.get("code", "upload_rejected")), "message": str(detail.get("message", "上传被拒绝"))})
            raise
        except Exception as exc:
            error = _error_payload(exc)
            manager.fail_immediate(job_id, error)
            try:
                input_path.unlink(missing_ok=True)
            except OSError:
                pass
            status = 413 if "exceeds" in str(exc).lower() else 422
            raise HTTPException(status_code=status, detail=error) from exc
        finally:
            await file.close()

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str) -> JobResponse:
        return _status_response(manager, job_id)

    @app.get("/api/v2/jobs/{job_id}", response_model=JobResponse)
    async def get_v2_job(job_id: str) -> JobResponse:
        try:
            if not manager.v2.is_v2(job_id):
                raise KeyError("not a V2 task")
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        return _status_response(manager, job_id)

    @app.get("/api/v2/jobs/{job_id}/tracks")
    async def get_v2_tracks(job_id: str) -> dict[str, Any]:
        try:
            value = manager.v2_tracks(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        if value.get("source_kind") != "instrumental":
            raise HTTPException(status_code=409, detail={"code": "tracks_not_applicable", "message": "人声任务没有乐器选择阶段"})
        if value.get("status") not in {"selection_ready", "completed"}:
            raise HTTPException(status_code=409, detail={"code": "tracks_not_ready", "message": "全量乐器识别尚未完成"})
        return value

    @app.post("/api/v2/jobs/{job_id}/selection", response_model=JobResponse, status_code=202)
    @app.post("/api/v2/jobs/{job_id}/selection/export", response_model=JobResponse, status_code=202)
    async def select_v2_tracks(job_id: str, payload: V2SelectionRequest = Body(...)) -> JobResponse:
        try:
            manager.select_v2(
                job_id,
                payload.selected_track_ids,
                payload.merge_main_melody,
                bpm_override=payload.bpm_override,
                key_override=payload.key_override,
                time_signature_override=payload.time_signature_override,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "selection_not_ready", "message": str(exc)}) from None
        return _status_response(manager, job_id)

    @app.post("/api/v2/jobs/{job_id}/vocal/generate", response_model=JobResponse, status_code=202)
    async def generate_v2_vocal(job_id: str) -> JobResponse:
        try:
            manager.generate_vocal_v2(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "vocal_not_ready", "message": str(exc)}) from None
        return _status_response(manager, job_id)

    @app.post("/api/v2/jobs/{job_id}/retry", response_model=JobResponse, status_code=202)
    async def retry_v2_job(job_id: str) -> JobResponse:
        try:
            manager.retry(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "not_retryable", "message": str(exc)}) from None
        return _status_response(manager, job_id)

    @app.get("/api/v2/jobs/{job_id}/score")
    async def get_v2_score(job_id: str) -> JSONResponse:
        try:
            if not manager.v2.is_v2(job_id):
                raise KeyError("not a V2 task")
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        status = _status_response(manager, job_id)
        if status.status != "completed":
            raise HTTPException(status_code=409, detail={"code": "score_not_ready", "message": f"任务当前阶段：{status.phase_label}"})
        artifacts = manager.list_artifacts(job_id)
        score_artifact = next((item for item in reversed(artifacts) if item.get("artifact_id") == "score-json"), None)
        if score_artifact is None:
            score_artifact = next((item for item in reversed(artifacts) if item.get("kind") == "instrument_score_json"), None)
        if score_artifact is None:
            refusal = (manager.get(job_id).get("v2") or {}).get("score_refusal")
            detail = refusal or {"code": "score_not_found", "message": "当前选择没有可下载的简谱"}
            raise HTTPException(status_code=409, detail=detail)
        try:
            path, _artifact = manager.artifact_path(job_id, str(score_artifact["artifact_id"]))
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "score_not_found", "message": "谱面尚未生成"}) from None
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/v2/jobs/{job_id}/artifacts")
    async def list_v2_artifacts(job_id: str) -> ArtifactsResponse:
        try:
            if not manager.v2.is_v2(job_id):
                raise KeyError("not a V2 task")
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "V2 任务不存在"}) from None
        return _artifact_response(manager, job_id, prefix="/api/v2/jobs")

    @app.get("/api/v2/jobs/{job_id}/artifacts/{artifact_id}")
    async def download_v2_artifact(job_id: str, artifact_id: str) -> FileResponse:
        try:
            if not manager.v2.is_v2(job_id):
                raise KeyError("not a V2 task")
            path, artifact = manager.artifact_path(job_id, artifact_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "产物不存在"}) from None
        name = safe_filename(str(artifact.get("filename", path.name)), fallback="artifact")
        return FileResponse(
            path,
            media_type=str(artifact.get("media_type", "application/octet-stream")),
            headers={"Content-Disposition": f'inline; filename="{name}"'},
        )

    @app.get("/api/v2/soundfont/status")
    async def get_v2_soundfont_status() -> dict[str, Any]:
        return await asyncio.to_thread(soundfont_status)

    @app.get("/api/v2/soundfont")
    async def download_v2_soundfont() -> FileResponse:
        try:
            path = await asyncio.to_thread(ensure_soundfont)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "soundfont_unavailable", "message": str(exc)},
            ) from exc
        return FileResponse(
            path,
            media_type="audio/x-soundfont-sf3",
            headers={"Content-Disposition": 'inline; filename="MuseScore_General.sf3"'},
        )

    @app.post("/api/jobs/{job_id}/retry", response_model=JobResponse, status_code=202)
    async def retry_job(job_id: str) -> JobResponse:
        try:
            manager.retry(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "任务不存在"}) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "not_retryable", "message": str(exc)}) from None
        return _status_response(manager, job_id)

    @app.get("/api/jobs/{job_id}/score")
    async def get_score(job_id: str) -> JSONResponse:
        status = _status_response(manager, job_id)
        if status.status != "completed":
            raise HTTPException(status_code=409, detail={"code": "score_not_ready", "message": f"任务当前阶段：{status.phase_label}"})
        try:
            path, _artifact = manager.artifact_path(job_id, "score-json")
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "score_not_found", "message": "谱面尚未生成"}) from None
        import json

        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/jobs/{job_id}/artifacts")
    async def list_job_artifacts(job_id: str) -> ArtifactsResponse:
        return _artifact_response(manager, job_id)

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
    async def download_artifact(job_id: str, artifact_id: str) -> FileResponse:
        try:
            path, artifact = manager.artifact_path(job_id, artifact_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "产物不存在"}) from None
        name = safe_filename(str(artifact.get("filename", path.name)), fallback="artifact")
        return FileResponse(
            path,
            media_type=str(artifact.get("media_type", "application/octet-stream")),
            headers={"Content-Disposition": f'inline; filename="{name}"'},
        )

    @app.get("/{full_path:path}")
    async def frontend(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        if FRONTEND_DIST.is_dir():
            candidate = (FRONTEND_DIST / full_path).resolve()
            if FRONTEND_DIST.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            index = FRONTEND_DIST / "index.html"
            if index.is_file():
                return FileResponse(index)
        raise HTTPException(status_code=404, detail="frontend build not found")

    return app


app = create_app()
