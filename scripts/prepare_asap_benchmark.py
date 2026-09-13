"""Prepare ten reproducible ASAP v1.1 score/performance benchmark clips.

The official ASAP tag contains performance MIDI, score MIDI, and independent
performance beat/downbeat annotations.  This script verifies the pinned tag
archive, chooses a fixed composer/meter plan before model inference, crops four
complete annotated measures, renders only the performance MIDI with the pinned
project FluidSynth policy, and keeps the aligned score MIDI as reference.  The
performance clip preserves annotated wall-clock seconds; the score reference
uses the source score MIDI's quarter-note positions after inverting its tempo
map, so a 3/4 bar remains three quarters even when its score tempo is not 120.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fluidsynth_benchmark_renderer import (
    RENDERER_VERSION,
    _tempo_segments,
    midi_notes,
    render_midi,
)
from scripts.generate_high_accuracy_benchmarks import RenderNote, RenderTrack, _midi_events


VERSION = "v1.1"
COMMIT = "fad8d1e8078d0ae47ad2f280b5d022bd2de24784"
ARCHIVE_URL = "https://codeload.github.com/fosfrancesco/asap-dataset/zip/refs/tags/v1.1"
ARCHIVE_BYTES = 59_474_742
ARCHIVE_SHA256 = "CDE942FD71809C185DABCE0A095A56CD26ADDD20F8C6DC3B2BFE91B94BCC535B"
LICENSE = "CC BY-NC-SA 4.0"
REPOSITORY = "https://github.com/fosfrancesco/asap-dataset"
TAG_URL = "https://github.com/fosfrancesco/asap-dataset/releases/tag/v1.1"
ARCHIVE_PREFIX = "asap-dataset-1.1/"
ANNOTATIONS_MEMBER = ARCHIVE_PREFIX + "asap_annotations.json"
LICENSE_MEMBER = ARCHIVE_PREFIX + "LICENSE.md"
DEFAULT_ARCHIVE = ROOT / ".cache" / "packages" / "asap-dataset-v1.1.zip"
DEFAULT_OUTPUT = ROOT / ".cache" / "high-accuracy-benchmarks" / "asap-v1.1"
MEASURE_COUNT = 4
SELECTION_PLAN = (
    ("Bach", "4/4"),
    ("Balakirev", "6/8"),
    ("Beethoven", "2/4"),
    ("Brahms", "3/4"),
    ("Chopin", "6/8"),
    ("Haydn", "4/4"),
    ("Liszt", "6/8"),
    ("Mozart", "3/4"),
    ("Prokofiev", "2/4"),
    ("Schubert", "4/4"),
)


@dataclass(frozen=True)
class Window:
    downbeat_index: int
    start_sec: float
    end_sec: float
    meter: str
    beats_per_measure: int
    performance_beats: tuple[float, ...]
    performance_downbeats: tuple[float, ...]
    score_start_sec: float
    score_end_sec: float
    score_measure_indices: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_matches(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == ARCHIVE_BYTES and _sha256(path).upper() == ARCHIVE_SHA256


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix() if relative_to else str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _safe_member(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def _parse_measure(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else None


def _annotation_txt_beats(text: str) -> tuple[list[float], list[float]]:
    beats: list[float] = []
    downbeats: list[float] = []
    for raw in text.splitlines():
        fields = raw.strip().split("\t")
        if len(fields) < 3:
            continue
        label = fields[2].split(",", 1)[0]
        if label not in {"b", "db", "bR"}:
            continue
        value = float(fields[1])
        beats.append(value)
        if label == "db":
            downbeats.append(value)
    return beats, downbeats


def _same_times(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-6) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def _time_signature_points(annotation: Mapping[str, Any]) -> list[tuple[float, tuple[str, int]]]:
    return sorted(
        (float(key), (str(value[0]), int(value[1])))
        for key, value in annotation.get("perf_time_signatures", {}).items()
    )


def _find_window(annotation: Mapping[str, Any], meter: str, *, measure_count: int = MEASURE_COUNT) -> Window | None:
    beats = sorted(float(value) for value in annotation.get("performance_beats", []))
    downbeats = sorted(float(value) for value in annotation.get("performance_downbeats", []))
    score_downbeats = [float(value) for value in annotation.get("midi_score_downbeats", [])]
    score_map = list(annotation.get("downbeats_score_map", []))
    beat_types = {float(key): str(value) for key, value in annotation.get("performance_beats_type", {}).items()}
    signature_points = _time_signature_points(annotation)
    signature_times = [item[0] for item in signature_points]
    if len(downbeats) != len(score_downbeats) or len(downbeats) != len(score_map):
        return None
    for index in range(len(downbeats) - measure_count):
        selected_downbeats = downbeats[index : index + measure_count + 1]
        signature_index = bisect.bisect_right(signature_times, selected_downbeats[0] + 1e-6) - 1
        if signature_index < 0:
            continue
        selected_meter, beats_per_measure = signature_points[signature_index][1]
        next_change = signature_points[signature_index + 1][0] if signature_index + 1 < len(signature_points) else float("inf")
        if selected_meter != meter or next_change < selected_downbeats[-1] - 1e-6:
            continue
        selected_beats = [value for value in beats if selected_downbeats[0] - 1e-6 <= value <= selected_downbeats[-1] + 1e-6]
        if any(beat_types.get(value) == "bR" for value in selected_beats):
            continue
        counts = [
            sum(start - 1e-6 <= value < end - 1e-6 for value in selected_beats)
            for start, end in zip(selected_downbeats, selected_downbeats[1:])
        ]
        if counts != [beats_per_measure] * measure_count:
            continue
        measures = tuple(_parse_measure(value) for value in score_map[index : index + measure_count + 1])
        if any(value is None for value in measures):
            continue
        integer_measures = tuple(int(value) for value in measures if value is not None)
        if any(b != a + 1 for a, b in zip(integer_measures, integer_measures[1:])):
            continue
        return Window(
            downbeat_index=index,
            start_sec=selected_downbeats[0],
            end_sec=selected_downbeats[-1],
            meter=meter,
            beats_per_measure=beats_per_measure,
            performance_beats=tuple(selected_beats),
            performance_downbeats=tuple(selected_downbeats),
            score_start_sec=score_downbeats[index],
            score_end_sec=score_downbeats[index + measure_count],
            score_measure_indices=integer_measures,
        )
    return None


def _select(
    annotations: Mapping[str, Any],
    members: set[str],
    plan: Sequence[tuple[str, str]],
) -> list[tuple[str, Mapping[str, Any], Window]]:
    selected: list[tuple[str, Mapping[str, Any], Window]] = []
    for composer, meter in plan:
        match = None
        for path in sorted(annotations):
            annotation = annotations[path]
            annotation_path = path.removesuffix(".mid") + "_annotations.txt"
            if not path.startswith(composer + "/") or annotation.get("score_and_performance_aligned") is not True:
                continue
            try:
                performance_member = ARCHIVE_PREFIX + _safe_member(path)
                annotation_member = ARCHIVE_PREFIX + _safe_member(annotation_path)
            except ValueError:
                continue
            if performance_member not in members or annotation_member not in members:
                continue
            window = _find_window(annotation, meter)
            if window is not None:
                match = (path, annotation, window)
                break
        if match is None:
            raise ValueError(f"ASAP v1.1 has no eligible {composer} {meter} performance")
        selected.append(match)
    if len({path.split("/", 1)[0] for path, _annotation, _window in selected}) != len(plan):
        raise ValueError("selection plan must use distinct composers")
    return selected


def _seconds_to_tick(mid: mido.MidiFile, value_sec: float) -> float:
    """Invert the source MIDI tempo map for an annotation second value."""

    segments, _starts = _tempo_segments(mid)
    for index, (start_tick, start_sec, tempo) in enumerate(segments):
        next_sec = segments[index + 1][1] if index + 1 < len(segments) else float("inf")
        if value_sec <= next_sec + 1e-7:
            return start_tick + (value_sec - start_sec) * mid.ticks_per_beat * 1_000_000 / tempo
    start_tick, start_sec, tempo = segments[-1]
    return start_tick + (value_sec - start_sec) * mid.ticks_per_beat * 1_000_000 / tempo


def _snap_tick(value: float) -> float:
    rounded = round(value)
    # ASAP stores score seconds at six decimal places.  At a non-integral
    # microsecond tempo that rounding can move the inverted tick by a few
    # thousandths, while annotated downbeats are still integer MIDI ticks.
    return float(rounded) if abs(value - rounded) <= 1e-2 else value


def _crop_midi(
    source: Path,
    destination: Path,
    *,
    start_sec: float,
    end_sec: float,
    meter: str,
    time_basis: str = "seconds",
) -> dict[str, Any]:
    mid, notes = midi_notes(source)
    if time_basis not in {"seconds", "score_quarters"}:
        raise ValueError(f"unsupported MIDI crop time basis: {time_basis!r}")
    if time_basis == "score_quarters":
        start_position = _snap_tick(_seconds_to_tick(mid, start_sec))
        end_position = _snap_tick(_seconds_to_tick(mid, end_sec))
        if end_position <= start_position:
            raise ValueError(f"score MIDI crop has non-positive tick interval: {source}")
    else:
        start_position = start_sec
        end_position = end_sec
    numerator, denominator = (int(value) for value in meter.split("/", 1))
    cropped: list[RenderNote] = []
    for note in notes:
        if time_basis == "score_quarters":
            note_start = float(note["start_tick"])
            note_end = float(note["end_tick"])
        else:
            note_start = float(note["start_sec"])
            note_end = float(note["end_sec"])
        start = max(start_position, note_start)
        end = min(end_position, note_end)
        if end <= start:
            continue
        if time_basis == "score_quarters":
            start_quarter = (start - start_position) / mid.ticks_per_beat
            end_quarter = (end - start_position) / mid.ticks_per_beat
        else:
            start_quarter = (start - start_position) * 2.0
            end_quarter = (end - start_position) * 2.0
        cropped.append(
            RenderNote(
                Fraction(str(start_quarter)).limit_denominator(960_000),
                Fraction(str(end_quarter)).limit_denominator(960_000),
                int(note["pitch"]),
                int(note["velocity"]),
                int(note["channel"]),
            )
        )
    if not cropped:
        raise ValueError(f"MIDI crop has no notes: {source}")
    tracks = (RenderTrack("ASAP piano", 0, tuple(cropped), 0),)
    output = _midi_events(tracks, ((Fraction(0), 120.0),), (numerator, denominator), "C")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)
    return {
        "source_note_count": len(notes),
        "clip_note_count": len(cropped),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": end_sec - start_sec,
        "time_basis": time_basis,
        "source_start_tick": start_position if time_basis == "score_quarters" else None,
        "source_end_tick": end_position if time_basis == "score_quarters" else None,
        "timing_policy": (
            "source MIDI score ticks shifted to zero and encoded as quarter positions at fixed 120 BPM/480 PPQ"
            if time_basis == "score_quarters"
            else "source MIDI seconds shifted to zero and encoded at fixed 120 BPM/480 PPQ"
        ),
    }


def _write_beat_grid(path: Path, window: Window, *, source_path: str, annotation_hash: str) -> None:
    beats: list[dict[str, Any]] = []
    for index, value in enumerate(window.performance_beats):
        local = max(0.0, min(window.end_sec - window.start_sec, value - window.start_sec))
        beat_in_bar = index % window.beats_per_measure
        beats.append(
            {
                "index": index,
                "bar_index": index // window.beats_per_measure,
                "beat_index": beat_in_bar,
                "time_sec": local,
                "downbeat": beat_in_bar == 0,
            }
        )
    payload = {
        "schema_version": "1.0",
        "source": "asap_v1.1_direct_performance_annotation",
        "beat_grid": {
            "beats": beats,
            "downbeats": [item for item in beats if item["downbeat"]],
            "time_signature": window.meter,
            "annotation_policy": "direct crop of official ASAP performance beat/downbeat annotation; never MIDI-tempo-derived",
            "source_performance_path": source_path,
            "source_annotation_sha256": annotation_hash,
            "source_downbeat_index": window.downbeat_index,
            "complete_measure_count": MEASURE_COUNT,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_archive(
    archive: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
    selection_plan: Sequence[tuple[str, str]] = SELECTION_PLAN,
) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not _archive_matches(archive):
        raise ValueError("ASAP v1.1 archive size or SHA-256 mismatch")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sources_root = output_root / "sources"
    clips_root = output_root / "clips"
    sources_root.mkdir(exist_ok=True)
    clips_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = {_safe_member(info.filename) for info in bundle.infolist() if not info.is_dir()}
        for required in (ANNOTATIONS_MEMBER, LICENSE_MEMBER):
            if required not in members:
                raise ValueError(f"ASAP archive is missing {required}")
        annotations_bytes = bundle.read(ANNOTATIONS_MEMBER)
        annotations = json.loads(annotations_bytes)
        selected = _select(annotations, members, selection_plan)
        records: list[dict[str, Any]] = []
        for number, (performance_path, annotation, window) in enumerate(selected, start=1):
            case_id = f"asap-v11-{number:02d}"
            relative_annotation = performance_path.removesuffix(".mid") + "_annotations.txt"
            score_path = str(Path(performance_path).parent / "midi_score.mid").replace("\\", "/")
            try:
                archive_members = {
                    relative: ARCHIVE_PREFIX + _safe_member(relative)
                    for relative in (performance_path, relative_annotation, score_path)
                }
            except ValueError as exc:
                raise ValueError(f"unsafe ASAP selection path for {performance_path!r}") from exc
            for relative, member in archive_members.items():
                if member not in members:
                    raise ValueError(f"ASAP selection is missing {relative}")
            source_case_root = sources_root / case_id
            source_case_root.mkdir(parents=True, exist_ok=True)
            source_performance = source_case_root / "performance.mid"
            source_score = source_case_root / "midi_score.mid"
            source_annotation = source_case_root / "performance_annotations.txt"
            source_performance.write_bytes(bundle.read(archive_members[performance_path]))
            source_score.write_bytes(bundle.read(archive_members[score_path]))
            source_annotation.write_bytes(bundle.read(archive_members[relative_annotation]))
            txt_beats, txt_downbeats = _annotation_txt_beats(source_annotation.read_text(encoding="utf-8"))
            if not _same_times(txt_beats, annotation["performance_beats"]) or not _same_times(txt_downbeats, annotation["performance_downbeats"]):
                raise ValueError(f"{performance_path}: central JSON and performance annotation text disagree")

            performance_clip = clips_root / f"{case_id}.performance.mid"
            reference_clip = clips_root / f"{case_id}.reference.mid"
            audio = clips_root / f"{case_id}.wav"
            beat_grid = clips_root / f"{case_id}.beat_grid.json"
            render_manifest_path = clips_root / f"{case_id}.render_manifest.json"
            performance_crop = _crop_midi(source_performance, performance_clip, start_sec=window.start_sec, end_sec=window.end_sec, meter=window.meter)
            score_crop = _crop_midi(
                source_score,
                reference_clip,
                start_sec=window.score_start_sec,
                end_sec=window.score_end_sec,
                meter=window.meter,
                time_basis="score_quarters",
            )
            render_manifest = render_midi(performance_clip, audio, manifest_path=render_manifest_path, overwrite=True)
            _write_beat_grid(
                beat_grid,
                window,
                source_path=performance_path,
                annotation_hash=_sha256(source_annotation),
            )
            records.append(
                {
                    "id": case_id,
                    "composer": performance_path.split("/", 1)[0],
                    "source_performance_path": performance_path,
                    "source_score_path": score_path,
                    "source_annotation_path": relative_annotation,
                    "source_performance": _file_record(source_performance, relative_to=output_root),
                    "source_score": _file_record(source_score, relative_to=output_root),
                    "source_annotation": _file_record(source_annotation, relative_to=output_root),
                    "performance_midi": _file_record(performance_clip, relative_to=output_root),
                    "reference_midi": _file_record(reference_clip, relative_to=output_root),
                    "audio": _file_record(audio, relative_to=output_root),
                    "beat_annotation": _file_record(beat_grid, relative_to=output_root),
                    "render_manifest": _file_record(render_manifest_path, relative_to=output_root),
                    "source_event_complete": bool(render_manifest["verification"]["source_event_complete"]),
                    "score_and_performance_aligned": True,
                    "meter": window.meter,
                    "beats_per_measure": window.beats_per_measure,
                    "complete_measure_count": MEASURE_COUNT,
                    "source_downbeat_index": window.downbeat_index,
                    "source_score_measure_indices": list(window.score_measure_indices),
                    "performance_window_sec": [window.start_sec, window.end_sec],
                    "score_window_sec": [window.score_start_sec, window.score_end_sec],
                    "performance_crop": performance_crop,
                    "score_crop": score_crop,
                }
            )
        license_path = sources_root / "LICENSE.md"
        license_path.write_bytes(bundle.read(LICENSE_MEMBER))
    manifest = {
        "schema_version": "asap_benchmark_selection_v1",
        "source": {
            "name": "Aligned Scores and Performances (ASAP)",
            "repository": REPOSITORY,
            "tag": VERSION,
            "tag_url": TAG_URL,
            "commit": COMMIT,
            "archive_url": ARCHIVE_URL,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha256(archive),
            "license": LICENSE,
            "license_file": _file_record(license_path, relative_to=output_root),
            "annotations_member": ANNOTATIONS_MEMBER,
            "annotations_sha256": hashlib.sha256(annotations_bytes).hexdigest(),
        },
        "renderer": RENDERER_VERSION,
        "selection_policy": {
            "plan": [{"composer": composer, "meter": meter} for composer, meter in selection_plan],
            "performance": "lexicographically first score_and_performance_aligned path for each fixed composer/meter pair",
            "window": f"first {MEASURE_COUNT} consecutive complete score-mapped measures with stable meter and no bR beats",
            "model_output_used": False,
            "beat_annotation": "direct official ASAP performance annotation crop; MIDI tempo is never used to derive beats",
            "score_reference": "source midi_score.mid cropped at annotated score-second boundaries, inverted through the source MIDI tempo map to quarter positions, then encoded at fixed 120 BPM/480 PPQ",
        },
        "cases": records,
    }
    manifest_path = output_root / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    archive = args.archive.resolve()
    if args.download and not _archive_matches(archive):
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".download", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            urllib.request.urlretrieve(ARCHIVE_URL, temporary)
            if not _archive_matches(temporary):
                raise ValueError("downloaded ASAP v1.1 archive size or SHA-256 mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    elif archive.is_file() and not _archive_matches(archive):
        raise ValueError("cached ASAP v1.1 archive size or SHA-256 mismatch; rerun with --download")
    manifest = prepare_archive(archive, output_root=args.output_root, overwrite=args.overwrite)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "cases": [item["id"] for item in manifest["cases"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
