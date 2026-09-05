from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_track_checkbox_is_the_single_selection_control() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert 'data-selection-policy="single-checkbox-controls-roll-synth-midi-score"' in source
    assert 'onChange={() => toggleTrack(track.track_id)}' in source
    assert "mutedTrackIds" not in source
    assert "toggleMute" not in source
    assert "mute-button" not in source


def test_vocal_ready_contract_uses_staged_generation_and_long_svg_first() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "vocal-ready" in source
    assert "下一步 · 生成人声简谱" in source
    assert "/vocal/generate" in source
    assert "Demucs 分离人声 → GAME" in source
    assert "score_svg_long" in source
    assert 'name="separation-model"' in source
    assert "htdemucs_ft" in source
    assert "切换只对新任务生效" in source


def test_remote_synth_contract_reports_secure_context_and_download_progress() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "window.isSecureContext" in source
    assert "secure_context_required" in source
    assert "audio_worklet_unsupported" in source
    assert "processor_load_failed" in source
    assert "cache: \"force-cache\"" in source
    assert "response.body.getReader()" in source
    assert "soundfont_incomplete" in source
    assert "soundfont_content_type" in source
    assert "soundfont_timeout" in source
    assert "fetchSoundfontRangeChunk" in source
    assert "SOUND_FONT_CHUNK_RETRIES" in source
    assert "soundfont_user_fallback" in source
    assert "轻量音色" in source
    assert "storeCachedSoundfont" in source
    assert "立即使用轻量试听" in source
    assert "重新加载高质量音色" in source
    assert "已改用轻量试听" in source
    assert "正在下载官方 MuseScore General SF3" in source
    assert "synth-diagnostic" in source
    assert "soundfontVersion" in source
    assert "soundfontCacheKey" in source
    assert "await storeCachedSoundfont" in source
    assert "X-Soundfont-SHA256" in source
    assert "高质量音色已缓存" in source
    assert "浏览器未能保存缓存，下次可能重新下载" in source
    assert "不会写入缓存" in source
