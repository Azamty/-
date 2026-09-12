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
    score_artifacts = (ROOT / "frontend" / "src" / "scoreArtifacts.ts").read_text(encoding="utf-8")

    assert "vocal-ready" in source
    assert "下一步 · 生成人声简谱" in source
    assert "/vocal/generate" in source
    assert "Demucs 分离人声 → GAME" in source
    assert "score_svg_long" in score_artifacts
    assert 'name="separation-model"' in source
    assert "htdemucs_ft" in source
    assert "modelChoiceHint" in source


def test_demucs_model_cards_hide_native_dot_but_keep_keyboard_radio_semantics() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    styles = (ROOT / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")
    model_choice = (ROOT / "frontend" / "src" / "modelChoice.ts").read_text(encoding="utf-8")

    assert 'type="radio" name="separation-model"' in source
    assert "model-choice-state" in source
    assert "modelChoiceDisabled(busy)" in source
    assert "modelChoiceHint(activeVocalJob ? actualDemucsModel : null)" in source
    assert "return busy;" in model_choice
    assert '.model-choice input[type="radio"]' in styles
    assert "clip-path: inset(50%)" in styles
    assert ".model-choice:focus-within" in styles


def test_selection_state_hydrates_before_persisting_ready_job_values() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    selection_state = (ROOT / "frontend" / "src" / "selectionState.ts").read_text(encoding="utf-8")

    assert "canPersistSelection(job.status, loadedJobRef.current === job.id)" in source
    assert "resolveSelectionValues(" in source
    assert "selectionStorageKey(jobId)" in source
    assert "server overrides first" in selection_state


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


def test_synth_pause_contract_destroys_worklet_queue_and_resumes_from_offset() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    # stopAll() alone leaves SpessaSynth's future event queue alive. The
    # browser contract must destroy that instance, preserve an offset, and
    # reject stale animation/timer callbacks before rebuilding on resume.
    assert "const [synthPaused, setSynthPaused]" in source
    assert "const synthPositionRef = useRef(0)" in source
    assert "const synthSessionRef = useRef(0)" in source
    assert "try { synth.disconnect(); }" in source
    assert "try { synth.destroy(); }" in source
    assert "soundfontAbortRef.current?.abort()" in source
    assert '"synth_cancelled"' in source
    assert "cloneSoundfontBuffer(soundfont)" in source
    assert "const soundfontForSynth = cloneSoundfontBuffer(soundfont)" in source
    assert "const pauseSynth = useCallback" in source
    assert 'haltSynth("pause", elapsed)' in source
    assert "synthSessionRef.current !== session" in source
    assert "const startOffset = synthPaused" in source
    assert "slicePlaybackNotes(playable, startOffset)" in source
    assert "playbackPosition(audioNow, synthOriginRef.current, synthDurationRef.current)" in source
    assert 'synthPaused ? "播放"' in source
    assert "synthPlaying || synthPaused ? synthTime : originalTime" in source


def test_result_contract_keeps_long_score_first_and_limits_downloads() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    score_artifacts = (ROOT / "frontend" / "src" / "scoreArtifacts.ts").read_text(encoding="utf-8")
    styles = (ROOT / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")

    assert "notation_engine" in source
    assert "beat_engine" in source
    assert "beatnet_version" in source
    assert "musescore_version" in source
    assert "score_ticks_per_quarter" in source
    assert "track_failures" in source
    assert "部分乐器未完成" in source
    assert "当前没有可生成的简谱" in source
    assert "isScoreDownload(item.kind)" in source
    assert "instrument_musicxml" not in source
    assert "instrument_alignment_report" not in source
    assert "instrument_performance_midi" not in source
    assert "main_melody_score_svg_long" in score_artifacts
    assert 'data-score-group="long"' in source
    assert 'data-score-group="paged"' in source
    assert "vocal_raw_notes" not in source
    assert "vocal_cleaned_notes" not in source
    assert "high_accuracy_manifest" not in source
    assert "high_accuracy_log" not in source
    assert "PDF · SVG · MIDI" in source
    assert "纵向长图 SVG · 默认预览" in source
    assert "artifact-download-panel" in styles
