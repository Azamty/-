"""Render the stage1 artificial polyphony fixture through jianpu-ly and LilyPond."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "polyphony.jly"
JIANPU = ROOT / "vendor" / "jianpu-ly" / "jianpu-ly.py"
LILYPOND = ROOT / "tools" / "lilypond-2.24.4" / "bin" / "lilypond.exe"
OUT = ROOT / "artifacts" / "stage1"


def run(command: list[str], *, cwd: Path, stdout: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if stdout is not None:
        stdout.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result


def inspect_midi(path: Path) -> dict[str, object]:
    """Check timing, independent tracks and real chord events in the MIDI."""

    midi = mido.MidiFile(path)
    ticks_per_beat = midi.ticks_per_beat
    note_tracks: list[dict[str, object]] = []
    for index, track in enumerate(midi.tracks):
        absolute = 0
        starts: list[tuple[int, int]] = []
        active: dict[int, list[int]] = {}
        spans: list[tuple[int, int, int]] = []
        for message in track:
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                starts.append((absolute, message.note))
                active.setdefault(message.note, []).append(absolute)
            elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
                pending = active.get(message.note)
                if pending:
                    spans.append((pending.pop(0), absolute, message.note))
        if starts:
            note_tracks.append({"index": index, "total_ticks": absolute, "starts": starts, "spans": spans})

    if len(note_tracks) != 3:
        raise RuntimeError(f"expected three note tracks, got {len(note_tracks)}")
    totals = {int(track["total_ticks"]) for track in note_tracks}
    expected_total = 24 * 4 * ticks_per_beat
    chord_events = [
        (tick, sorted(note for start, note in track["starts"] if start == tick))
        for track in note_tracks
        for tick in sorted({start for start, _ in track["starts"]})
        if sum(1 for start, _ in track["starts"] if start == tick) >= 2
    ]
    triplet_offsets = {ticks_per_beat // 3, (2 * ticks_per_beat) // 3}
    off_grid_starts = [
        (start, note)
        for track in note_tracks
        for start, note in track["starts"]
        if start % ticks_per_beat in triplet_offsets
    ]
    cross_tie_spans = [
        (start, end, note)
        for track in note_tracks
        for start, end, note in track["spans"]
        if start == 11 * ticks_per_beat and end - start == 2 * ticks_per_beat
    ]
    evidence = {
        "ticks_per_beat": ticks_per_beat,
        "note_tracks": [
            {
                "index": track["index"],
                "total_ticks": track["total_ticks"],
                "note_on_count": len(track["starts"]),
            }
            for track in note_tracks
        ],
        "checks": {
            "three_note_tracks": len(note_tracks) == 3,
            "same_24_bar_duration": totals == {expected_total},
            "real_chord_event": bool(chord_events),
            "off_grid_tuplet_event": bool(off_grid_starts),
            "cross_bar_tie_duration": bool(cross_tie_spans),
        },
        "chord_events": chord_events[:4],
        "off_grid_starts": off_grid_starts[:8],
        "cross_tie_spans": cross_tie_spans,
    }
    if not all(evidence["checks"].values()):
        raise RuntimeError(f"MIDI checks failed: {evidence}")
    return evidence


def main() -> int:
    if not FIXTURE.exists():
        raise FileNotFoundError(FIXTURE)
    if not JIANPU.exists():
        raise FileNotFoundError(JIANPU)
    if not LILYPOND.exists():
        raise FileNotFoundError(LILYPOND)

    OUT.mkdir(parents=True, exist_ok=True)
    # Remove only artifacts produced by this fixture so a rerun cannot count
    # stale pages from a previous invocation.
    for stale in OUT.glob("polyphony*"):
        if stale.is_file():
            stale.unlink()
    source = FIXTURE.read_text(encoding="utf-8")
    (OUT / "polyphony.jly").write_text(source, encoding="utf-8")

    generated = run(
        [os.fspath(Path(os.sys.executable)), os.fspath(JIANPU), os.fspath(FIXTURE)],
        cwd=ROOT,
    )
    ly_path = OUT / "polyphony.ly"
    ly_path.write_text(generated.stdout, encoding="utf-8")

    # Keep output deterministic and request SVG + MIDI. LilyPond uses its own
    # bundled Guile/font runtime from the extracted directory.
    lily = run(
        [os.fspath(LILYPOND), "--svg", "-o", os.fspath(OUT / "polyphony"), os.fspath(ly_path)],
        cwd=ROOT,
        stdout=OUT / "lilypond.log",
    )
    del lily

    svg_paths = sorted(OUT.glob("polyphony*.svg"))
    midi_paths = sorted(OUT.glob("polyphony*.midi")) + sorted(OUT.glob("polyphony*.mid"))
    if not svg_paths:
        raise RuntimeError("LilyPond completed without an SVG output")
    if len(midi_paths) != 1:
        raise RuntimeError(f"expected one MIDI file, got {midi_paths}")

    text = generated.stdout
    checks = {
        "has_three_jianpu_staves": text.count("BEGIN JIANPU STAFF") == 3,
        "has_tuplet": "tuplet" in text.lower() or "Tuplet" in text,
        "has_tie": "~" in text,
        "multiple_svg_pages": len(svg_paths) >= 2,
        "automatic_page_break": "page-count" not in source and "set-paper-size" in source,
        "has_midi": bool(midi_paths),
    }
    # jianpu-ly translates each NextPart into separate score/part material;
    # retain an explicit source-level check as well so future syntax changes
    # cannot silently remove the intended polyphonic fixture.
    source_checks = {
        "source_has_three_parts": source.count("NextPart") == 2,
        "source_has_triplet": source.count("3[") >= 3,
        "source_has_tie": "~" in source,
    }
    midi_evidence = inspect_midi(midi_paths[0])
    manifest = {
        "fixture_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "jianpu_output_bytes": len(generated.stdout.encode("utf-8")),
        "svg_paths": [p.relative_to(ROOT).as_posix() for p in svg_paths],
        "midi_paths": [p.relative_to(ROOT).as_posix() for p in midi_paths],
        "checks": {**source_checks, **checks},
        "midi_evidence": midi_evidence,
        "lilypond": "2.24.4",
        "jianpu_ly": "1.889",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not all(manifest["checks"].values()):
        raise RuntimeError(f"fixture checks failed: {manifest['checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
