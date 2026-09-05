from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_track_checkbox_is_the_single_selection_control() -> None:
    source = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert 'data-selection-policy="single-checkbox-controls-roll-synth-midi-score"' in source
    assert 'onChange={() => toggleTrack(track.track_id)}' in source
    assert "mutedTrackIds" not in source
    assert "toggleMute" not in source
    assert "mute-button" not in source
