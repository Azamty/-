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
