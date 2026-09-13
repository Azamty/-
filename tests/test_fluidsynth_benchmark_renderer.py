from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess

import mido
import pytest
import soundfile as sf

from scripts import fluidsynth_benchmark_renderer as renderer


def _write_source_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    midi.tracks.append(conductor)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=37, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=240))
    track.append(mido.Message("note_on", channel=0, note=64, velocity=101, time=120))
    track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=240))
    midi.tracks.append(track)
    midi.save(path)


def test_source_summary_preserves_note_timing_velocity_and_meter(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _write_source_midi(source)

    summary = renderer.source_midi_summary(source)

    assert summary["note_count"] == 2
    assert summary["pitch_counts"] == {60: 1, 64: 1}
    assert summary["velocity_counts"] == {37: 1, 101: 1}
    assert summary["tempo_events"][0]["bpm"] == 120.0
    assert summary["meter_events"][0]["numerator"] == 3
    assert summary["meter_events"][0]["denominator"] == 4
    assert summary["note_event_fields"] == ["track", "channel", "pitch", "start_tick", "end_tick", "velocity"]
    assert len(summary["note_events_sha256"]) == 64


def test_event_comparison_requires_velocity_when_source_has_it() -> None:
    source = [{"pitch": 60, "velocity": 37}]
    complete = renderer.parse_event_dump(
        "event_post_noteon 0 60 37\n"
        "event_post_noteoff 0 60 0\n"
    )
    wrong_velocity = renderer.parse_event_dump(
        "event_post_noteon 0 60 38\n"
        "event_post_noteoff 0 60 0\n"
    )

    assert renderer.compare_event_dump(source, complete)["event_complete"] is True
    assert renderer.compare_event_dump(source, wrong_velocity)["event_complete"] is False


def test_trim_wave_pads_to_declared_fixed_boundary(tmp_path: Path) -> None:
    source = tmp_path / "short.wav"
    destination = tmp_path / "trimmed.wav"
    with renderer.wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(b"\x01\x02\x03\x04" * 2)

    info = renderer.trim_wave(source, destination, target_frames=5)

    assert info["source_frames"] == 2
    assert info["target_frames"] == 5
    assert info["final_frames"] == 5
    with renderer.wave.open(str(destination), "rb") as output:
        assert output.readframes(5) == b"\x01\x02\x03\x04" * 2 + b"\x00" * 12


def test_fluid_synth_timeout_terminates_tree_and_preserves_partial_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mid"
    destination = tmp_path / "partial.wav"
    source.write_bytes(b"midi")
    destination.write_bytes(b"partial")

    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self) -> None:
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            assert timeout == 0.25
            raise subprocess.TimeoutExpired(
                ["fluidsynth"], timeout, output="partial stdout", stderr="partial stderr"
            )

        def poll(self) -> None:
            return None

    process = FakeProcess()
    popen_calls: list[dict[str, object]] = []
    terminated: list[FakeProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        popen_calls.append({"command": command, **kwargs})
        return process

    def fake_terminate(value: FakeProcess) -> dict[str, object]:
        terminated.append(value)
        return {"pid": value.pid, "method": "taskkill_tree_force", "waited": True}

    monkeypatch.setattr(renderer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(renderer, "_terminate_process_tree", fake_terminate)

    result = renderer._run_fluid_synth(
        tmp_path / "fluidsynth.exe",
        tmp_path / "MS Basic.sf3",
        source,
        destination,
        timeout_sec=0.25,
    )

    assert result["timeout"] is True
    assert result["return_code"] is None
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr"
    assert result["partial_artifact"] == {
        "path": str(destination),
        "exists": True,
        "bytes": 7,
        "sha256": renderer.sha256(destination),
    }
    assert result["termination"] == {"pid": 4321, "method": "taskkill_tree_force", "waited": True}
    assert terminated == [process]
    assert popen_calls[0]["command"][-1] == str(source)
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_render_failure_writes_timeout_diagnostics_and_skips_second_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mid"
    output = tmp_path / "source.wav"
    manifest_path = tmp_path / "source.render_manifest.json"
    _write_source_midi(source)
    monkeypatch.setattr(renderer, "renderer_metadata", lambda **_kwargs: {"renderer_version": "test"})
    calls: list[Path] = []

    def fake_run(_executable: Path, _soundfont: Path, _source: Path, destination: Path, *, timeout_sec: float) -> dict[str, object]:
        calls.append(destination)
        return {
            "command": ["fake-fluidsynth"],
            "return_code": None,
            "timeout": True,
            "stdout": "partial stdout",
            "stderr": "partial stderr",
            "exists": False,
            "event_dump": {},
            "partial_artifact": {"path": str(destination), "exists": False},
            "termination": {"method": "taskkill_tree_force"},
        }

    monkeypatch.setattr(renderer, "_run_fluid_synth", fake_run)

    with pytest.raises(RuntimeError, match="diagnostics preserved"):
        renderer.render_midi(source, output, manifest_path=manifest_path, timeout_sec=1.0)

    assert len(calls) == 1
    diagnostic_path = tmp_path / ".source.fluidsynth-verification" / "render_failure.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["failure"]["runs"]["run_a"]["timeout"] is True
    assert diagnostic["failure"]["runs"]["run_b"]["skipped"] is True
    assert diagnostic["failure"]["partial_artifacts"]["run_a"]["exists"] is False


@pytest.mark.skipif(
    not renderer.DEFAULT_EXECUTABLE.is_file() or not renderer.DEFAULT_SOUNDFONT.is_file(),
    reason="pinned FluidSynth or MS Basic SoundFont is unavailable",
)
def test_direct_renderer_is_fixed_pcm_deterministic_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    output = tmp_path / "source.wav"
    manifest_path = tmp_path / "source.render_manifest.json"
    _write_source_midi(source)

    manifest = renderer.render_midi(source, output, manifest_path=manifest_path)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    audio = sf.info(output)

    assert json.loads(json.dumps(manifest)) == persisted
    assert manifest["renderer"]["renderer_version"] == renderer.RENDERER_VERSION
    assert manifest["renderer"]["sample_rate"] == 44_100
    assert manifest["renderer"]["channels"] == 2
    assert manifest["renderer"]["subtype"] == "PCM_16"
    assert manifest["renderer"]["effects"] == {"reverb": False, "chorus": False}
    assert manifest["renderer"]["gain"] == renderer.GAIN
    assert manifest["renderer"]["tail_sec"] == 0.25
    assert persisted["source_midi"]["note_count"] == 2
    assert persisted["source_midi"]["pitch_counts"] == {"60": 1, "64": 1}
    assert persisted["source_midi"]["velocity_counts"] == {"37": 1, "101": 1}
    assert persisted["source_midi"]["tempo_events"][0]["bpm"] == 120.0
    assert persisted["source_midi"]["meter_events"][0]["numerator"] == 3
    assert persisted["source_midi"]["last_note_end_sec"] == 0.625
    assert manifest["verification"]["byte_deterministic"] is True
    assert manifest["verification"]["source_event_complete"] is True
    assert audio.samplerate == 44_100
    assert audio.channels == 2
    assert audio.subtype == "PCM_16"
    assert manifest["output"]["sha256"] == renderer.sha256(output)
