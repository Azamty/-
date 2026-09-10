"""Verify and select ten reproducible MAESTRO MIDI benchmark members.

The default mode requires a user-provided archive path.  ``--download`` is
available for an explicitly authorized local run, but this task does not run
it.  The archive hash is checked before extraction; selected members and their
hashes are recorded so a later run cannot silently switch the public data.
Audio is rendered locally from deterministic source MIDI clips.  These clips
are valid pitch/event-time model cases in the benchmark's local-render domain:
the audio still goes through the production recognizer, while the domain
limits are disclosed explicitly.  MAESTRO performance MIDI does not provide
score-aligned beat labels; its fixed transport tick grid is retained only as a
diagnostic ruler and must not be scored as beat/downbeat ground truth.  The
pinned FluidSynth 2.6.0 binary receives the
source MIDI directly with MS Basic.sf3; it preserves MIDI pitch, timing,
tempo, meter, and note velocity but does not reproduce piano timbre, pedal
noise, room acoustics, or the original MAESTRO performance.  Clip boundaries
are selected from source MIDI timing before recognition and never from model
output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from scripts.fluidsynth_benchmark_renderer import RENDERER_VERSION, render_midi  # noqa: E402
DEFAULT_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
EXPECTED_SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
DEFAULT_OUTPUT = ROOT / ".cache" / "high-accuracy-benchmarks" / "maestro"
CLIP_BARS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def _midi_tracks(path: Path):
    from generate_high_accuracy_benchmarks import RenderNote, RenderTrack

    midi = mido.MidiFile(path)
    tracks: list[RenderTrack] = []
    tempo: dict[Fraction, float] = {}
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        program = 0
        channel = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        notes: list[RenderNote] = []
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                tempo[Fraction(tick, midi.ticks_per_beat)] = float(mido.tempo2bpm(message.tempo))
            elif message.type == "program_change":
                program = int(message.program)
                channel = int(message.channel)
            elif message.type == "note_on" and message.velocity > 0:
                channel = int(message.channel)
                active.setdefault((channel, int(message.note)), []).append((tick, int(message.velocity)))
            elif message.type in {"note_on", "note_off"}:
                key = (int(message.channel), int(message.note))
                if active.get(key):
                    start, velocity = active[key].pop(0)
                    if tick > start:
                        notes.append(
                            RenderNote(
                                Fraction(start, midi.ticks_per_beat),
                                Fraction(tick, midi.ticks_per_beat),
                                int(message.note),
                                velocity,
                                int(message.channel),
                            )
                        )
        if notes:
            tracks.append(RenderTrack(f"maestro track {track_index + 1}", program, tuple(notes), channel))
    return midi, tuple(tracks), tuple(sorted(tempo.items()) or ((Fraction(0), 120.0),))


def _midi_metadata(path: Path, midi: mido.MidiFile | None = None) -> dict[str, Any]:
    """Read source timing metadata without changing the note representation."""

    midi = midi or mido.MidiFile(path)
    tempo: list[tuple[Fraction, float]] = []
    meters: list[tuple[Fraction, tuple[int, int]]] = []
    keys: list[tuple[Fraction, str]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            quarter = Fraction(tick, midi.ticks_per_beat)
            if message.type == "set_tempo":
                tempo.append((quarter, float(mido.tempo2bpm(message.tempo))))
            elif message.type == "time_signature":
                meters.append((quarter, (int(message.numerator), int(message.denominator))))
            elif message.type == "key_signature":
                keys.append((quarter, str(message.key)))
    return {
        "tempo": tuple(sorted(tempo) or ((Fraction(0), 120.0),)),
        "meters": tuple(sorted(meters) or ((Fraction(0), (4, 4)),)),
        "keys": tuple(sorted(keys) or ((Fraction(0), "C"),)),
    }


def _value_at(points: Sequence[tuple[Fraction, Any]], position: Fraction, default: Any) -> Any:
    selected = default
    for point, value in sorted(points):
        if point > position:
            break
        selected = value
    return selected


def _clip_points(
    points: Sequence[tuple[Fraction, Any]],
    start: Fraction,
    end: Fraction,
    *,
    default: Any,
) -> tuple[tuple[Fraction, Any], ...]:
    """Shift a source map to clip-local quarter positions."""

    value = _value_at(points, start, default)
    clipped: list[tuple[Fraction, Any]] = [(Fraction(0), value)]
    for position, item in sorted(points):
        if start < position < end:
            clipped.append((position - start, item))
    return tuple(clipped)


def _clip_tracks(tracks: Sequence[Any], start: Fraction, end: Fraction) -> tuple[Any, ...]:
    """Crop notes to ``[start, end]`` and shift the result to zero."""

    from generate_high_accuracy_benchmarks import RenderNote, RenderTrack

    clipped_tracks: list[RenderTrack] = []
    for track in tracks:
        notes = []
        for note in track.notes:
            note_start = max(note.start, start)
            note_end = min(note.end, end)
            if note_end <= note_start:
                continue
            notes.append(
                RenderNote(
                    note_start - start,
                    note_end - start,
                    int(note.pitch),
                    int(note.velocity),
                    int(note.channel),
                )
            )
        if notes:
            clipped_tracks.append(RenderTrack(track.name, track.program, tuple(notes), track.channel))
    return tuple(clipped_tracks)


def _clip_window(tracks: Sequence[Any], meter: tuple[int, int]) -> tuple[Fraction, Fraction]:
    """Choose the first note's containing bar and at most eight bars."""

    notes = [note for track in tracks for note in track.notes]
    if not notes:
        raise ValueError("selected MAESTRO MIDI has no notes")
    bar_quarters = Fraction(int(meter[0]) * 4, int(meter[1]))
    first_note = min(note.start for note in notes)
    source_end = max(note.end for note in notes)
    start = (first_note // bar_quarters) * bar_quarters
    target_end = start + CLIP_BARS * bar_quarters
    if target_end > source_end:
        target_end = ((source_end + bar_quarters - Fraction(1, 10**6)) // bar_quarters) * bar_quarters
        if target_end <= start:
            target_end = start + bar_quarters
    return start, target_end


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def prepare_archive(archive: Path, *, output_root: Path = DEFAULT_OUTPUT, count: int = 10, overwrite: bool = False) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_sha = _sha256(archive)
    if actual_sha.lower() != EXPECTED_SHA256.lower():
        raise ValueError(f"MAESTRO archive SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha}")
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(_safe_member_name(info.filename) for info in bundle.infolist() if not info.is_dir() and info.filename.lower().endswith((".mid", ".midi")))
        if len(members) < count:
            raise ValueError(f"MAESTRO archive contains only {len(members)} MIDI members; need {count}")
        selected = members[:count]
        output_root = output_root.resolve()
        selected_root = output_root / "selected"
        rendered_root = output_root / "rendered"
        clip_root = rendered_root / "clips"
        selected_root.mkdir(parents=True, exist_ok=True)
        rendered_root.mkdir(parents=True, exist_ok=True)
        clip_root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index, member in enumerate(selected, start=1):
            normalized_id = f"maestro-midi-{index:02d}"
            midi_destination = selected_root / f"{normalized_id}.mid"
            if midi_destination.exists() and not overwrite:
                raise FileExistsError(f"selection exists; use --overwrite: {midi_destination}")
            with bundle.open(member) as source, midi_destination.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            midi, tracks, source_tempo = _midi_tracks(midi_destination)
            metadata = _midi_metadata(midi_destination, midi)
            source_meter = _value_at(metadata["meters"], Fraction(0), (4, 4))
            source_key = _value_at(metadata["keys"], Fraction(0), "C")
            start_q, end_q = _clip_window(tracks, source_meter)
            clip_end_q = end_q - start_q
            clip_tracks = _clip_tracks(tracks, start_q, end_q)
            clip_tempo = _clip_points(source_tempo, start_q, end_q, default=120.0)
            clip_meter = _clip_points(metadata["meters"], start_q, end_q, default=(4, 4))
            clip_key = _clip_points(metadata["keys"], start_q, end_q, default="C")

            from generate_high_accuracy_benchmarks import _beat_annotation

            full_audio_destination = rendered_root / f"{normalized_id}.wav"
            full_beat_destination = rendered_root / f"{normalized_id}.beat_grid.json"
            # Full renders are already deterministic cache products and can
            # be very large for MAESTRO.  Reuse an existing full render even
            # when clips are being regenerated; the clip itself is always
            # rewritten below under ``--overwrite``.
            full_render_manifest_path = rendered_root / f"{normalized_id}.render_manifest.json"
            full_render_manifest = render_midi(
                midi_destination,
                full_audio_destination,
                manifest_path=full_render_manifest_path,
                overwrite=True,
            )
            source_end_q = max((note.end for track in tracks for note in track.notes), default=Fraction(1))
            if overwrite or not full_beat_destination.exists():
                _beat_annotation(full_beat_destination, tempo=source_tempo, meter=source_meter, end_q=source_end_q, source="official_maestro_performance_midi_tick_grid_diagnostic_only")

            clip_midi_destination = clip_root / f"{normalized_id}.mid"
            clip_audio_destination = clip_root / f"{normalized_id}.wav"
            clip_beat_destination = clip_root / f"{normalized_id}.beat_grid.json"
            if not overwrite and clip_midi_destination.exists():
                raise FileExistsError(f"clip exists; use --overwrite: {clip_midi_destination}")
            from generate_high_accuracy_benchmarks import _midi_events

            clip_midi = _midi_events(clip_tracks, clip_tempo, source_meter, source_key)
            clip_midi.save(clip_midi_destination)
            clip_render_manifest_path = clip_root / f"{normalized_id}.render_manifest.json"
            clip_render_manifest = render_midi(
                clip_midi_destination,
                clip_audio_destination,
                manifest_path=clip_render_manifest_path,
                overwrite=True,
            )
            _beat_annotation(
                clip_beat_destination,
                tempo=clip_tempo,
                meter=source_meter,
                end_q=clip_end_q,
                source="official_maestro_performance_midi_tick_grid_diagnostic_only",
            )
            records.append(
                {
                    "id": normalized_id,
                    "archive_member": member,
                    "source_midi": {
                        "path": midi_destination.name,
                        "bytes": midi_destination.stat().st_size,
                        "sha256": _sha256(midi_destination),
                    },
                    "midi": _file_record(clip_midi_destination, relative_to=output_root),
                    "audio": _file_record(clip_audio_destination, relative_to=output_root),
                    "render_manifest": _file_record(clip_render_manifest_path, relative_to=output_root),
                    "beat_annotation": _file_record(clip_beat_destination, relative_to=output_root),
                    "beat_annotation_independent": False,
                    "beat_annotation_role": "performance_midi_tick_grid_diagnostic_only",
                    "full_render": {
                        "audio": _file_record(full_audio_destination, relative_to=output_root),
                        "render_manifest": _file_record(full_render_manifest_path, relative_to=output_root),
                        "beat_annotation": _file_record(full_beat_destination, relative_to=output_root),
                    },
                    "source_event_complete": bool(
                        clip_render_manifest["verification"]["source_event_complete"]
                        and full_render_manifest["verification"]["source_event_complete"]
                    ),
                    "ticks_per_beat": midi.ticks_per_beat,
                    "clip": {
                        "start_quarter": str(start_q),
                        "end_quarter": str(end_q),
                        "duration_quarter": str(clip_end_q),
                        "bar_count": CLIP_BARS if clip_end_q == CLIP_BARS * Fraction(source_meter[0] * 4, source_meter[1]) else str(clip_end_q / Fraction(source_meter[0] * 4, source_meter[1])),
                        "bar_quarters": str(Fraction(source_meter[0] * 4, source_meter[1])),
                        "source_end_quarter": str(source_end_q),
                        "policy": "first_source_note_aligned_to_containing_bar_then_eight_source_bars_or_final_complete_bars",
                        "policy_limitation": "quarter positions are performance transport ticks, not independently annotated musical bars",
                        "source_meter": f"{source_meter[0]}/{source_meter[1]}",
                        "source_key": source_key,
                        "tempo_points": [{"quarter": str(point), "bpm": bpm} for point, bpm in clip_tempo],
                        "meter_points": [{"quarter": str(point), "time_signature": f"{value[0]}/{value[1]}"} for point, value in clip_meter],
                        "key_points": [{"quarter": str(point), "key": value} for point, value in clip_key],
                    },
                }
            )
    manifest = {
        "schema_version": "1.0",
        "source_url": "https://magenta.tensorflow.org/datasets/maestro",
        "download_url": DEFAULT_URL,
        "license": "CC BY-NC-SA 4.0",
        "renderer": RENDERER_VERSION,
        "render_seed": None,
        "archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": actual_sha},
        "selection_rule": "sorted archive MIDI member names, first ten after hash verification",
        "render_domain": {
            "kind": "local_midi_render",
            "renderer": RENDERER_VERSION,
            "renderer_policy": "see each case render_manifest for pinned FluidSynth executable/SoundFont hashes, PCM16 format, gain, effects, and fixed tail",
            "preserves": ["MIDI pitch", "MIDI onset and duration", "MIDI note velocity", "tempo map", "source meter and key at clip start"],
            "does_not_model": ["piano timbre", "pedal noise", "room acoustics", "original performance nuance"],
            "velocity_policy": "preserve_source_midi",
            "evaluation_scope": "production_end_to_end",
            "benchmark_role": "production_end_to_end_render_domain",
            "production_end_to_end": True,
            "beat_evaluation_eligible": False,
            "beat_evaluation_reason": "MAESTRO performance MIDI has aligned key-event time but no score-aligned beat/downbeat labels",
            "clip_policy": f"deterministic source-MIDI transport-tick window: first note's containing {CLIP_BARS * 4}-quarter block, never model-output selected",
        },
        "cases": records,
    }
    selection_manifest = output_root / "selection_manifest.json"
    temporary = selection_manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(selection_manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="已下载的 MAESTRO MIDI zip；默认不联网")
    parser.add_argument("--download", action="store_true", help="显式下载官方 archive 后校验；不会自动运行")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--download-to", type=Path, default=ROOT / ".cache" / "packages" / "maestro-v3.0.0-midi.zip")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    archive = args.archive.resolve() if args.archive else None
    if args.download:
        archive = args.download_to.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".download", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                urllib.request.urlretrieve(args.url, temporary)
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
    if archive is None:
        raise SystemExit("provide --archive or explicitly pass --download")
    manifest = prepare_archive(archive, output_root=args.output_root, count=args.count, overwrite=args.overwrite)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "archive_sha256": manifest["archive"]["sha256"], "selected": [item["id"] for item in manifest["cases"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
