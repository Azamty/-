from __future__ import annotations

from pathlib import Path

import pytest

import backend.jianpu_score.models.demucs as demucs


def test_demucs_adapter_uses_selected_model_output_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fixture")
    captured: dict[str, object] = {}

    monkeypatch.setattr(demucs, "probe_audio", lambda *_args, **_kwargs: {"duration_sec": 1.0})
    monkeypatch.setattr(demucs, "resolve_ffmpeg", lambda: None)
    monkeypatch.setattr(demucs, "demucs_python", lambda: tmp_path / "python.exe")

    class Process:
        returncode = 0

        def communicate(self) -> tuple[str, str]:
            return "ok", ""

    def fake_popen(command: list[str], **_kwargs: object) -> Process:
        captured["command"] = command
        model = command[command.index("--name") + 1]
        output = Path(command[command.index("--out") + 1])
        stem_dir = output / model / audio.stem
        stem_dir.mkdir(parents=True)
        for name in ("vocals", "drums", "bass", "other"):
            (stem_dir / f"{name}.wav").write_bytes(b"stem")
        return Process()

    monkeypatch.setattr(demucs.subprocess, "Popen", fake_popen)
    stems = demucs.separate_htdemucs(audio, tmp_path / "demucs", model="htdemucs_ft")

    assert all(path.parent.parent.name == "htdemucs_ft" for path in stems.values())
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--name") + 1] == "htdemucs_ft"


def test_demucs_adapter_rejects_unknown_model_before_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False

    def fail_probe(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(demucs, "probe_audio", fail_probe)
    with pytest.raises(ValueError, match="只允许"):
        demucs.separate_htdemucs(tmp_path / "input.wav", tmp_path / "demucs", model="invalid")
    assert called is False
