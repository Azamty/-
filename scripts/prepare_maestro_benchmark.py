"""Verify and select ten reproducible MAESTRO MIDI benchmark members.

The default mode requires a user-provided archive path.  ``--download`` is
available for an explicitly authorized local run, but this task does not run
it.  The archive hash is checked before extraction; selected members and their
hashes are recorded so a later run cannot silently switch the public data.
Audio is rendered locally from the selected MIDI, which makes these cases a
quantizer-isolation subset until a separately licensed performance recording
is supplied.
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
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DEFAULT_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
EXPECTED_SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
DEFAULT_OUTPUT = ROOT / ".cache" / "high-accuracy-benchmarks" / "maestro"


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
        active: dict[tuple[int, int], list[int]] = {}
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
                active.setdefault((channel, int(message.note)), []).append(tick)
            elif message.type in {"note_on", "note_off"}:
                key = (int(message.channel), int(message.note))
                if active.get(key):
                    start = active[key].pop(0)
                    if tick > start:
                        notes.append(RenderNote(Fraction(start, midi.ticks_per_beat), Fraction(tick, midi.ticks_per_beat), int(message.note), 80, int(message.channel)))
        if notes:
            tracks.append(RenderTrack(f"maestro track {track_index + 1}", program, tuple(notes), channel))
    return midi, tuple(tracks), tuple(sorted(tempo.items()) or ((Fraction(0), 120.0),))


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
        selected_root.mkdir(parents=True, exist_ok=True)
        rendered_root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        for index, member in enumerate(selected, start=1):
            normalized_id = f"maestro-midi-{index:02d}"
            midi_destination = selected_root / f"{normalized_id}.mid"
            if midi_destination.exists() and not overwrite:
                raise FileExistsError(f"selection exists; use --overwrite: {midi_destination}")
            with bundle.open(member) as source, midi_destination.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            midi, tracks, tempo = _midi_tracks(midi_destination)
            from generate_high_accuracy_benchmarks import _beat_annotation, _render_audio

            audio_destination = rendered_root / f"{normalized_id}.wav"
            beat_destination = rendered_root / f"{normalized_id}.beat_grid.json"
            _render_audio(audio_destination, tracks, tempo, seed=20260907 + index)
            end_q = max((note.end for track in tracks for note in track.notes), default=Fraction(1))
            _beat_annotation(beat_destination, tempo=tempo, meter=(4, 4), end_q=end_q, source="official_maestro_midi_rendered")
            records.append({"id": normalized_id, "archive_member": member, "midi": {"path": midi_destination.name, "bytes": midi_destination.stat().st_size, "sha256": _sha256(midi_destination)}, "audio": {"path": audio_destination.name, "bytes": audio_destination.stat().st_size, "sha256": _sha256(audio_destination)}, "beat_annotation": {"path": beat_destination.name, "bytes": beat_destination.stat().st_size, "sha256": _sha256(beat_destination)}, "ticks_per_beat": midi.ticks_per_beat})
    manifest = {"schema_version": "1.0", "source_url": "https://magenta.tensorflow.org/datasets/maestro", "download_url": DEFAULT_URL, "license": "CC BY-NC-SA 4.0", "render_seed": 20260907, "archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": actual_sha}, "selection_rule": "sorted archive MIDI member names, first ten after hash verification", "cases": records}
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
