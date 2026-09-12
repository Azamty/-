"""Deterministic jianpu-ly and LilyPond renderer."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import mido
from pydantic import BaseModel, ConfigDict

from .domain import Score, TempoEvent
from .quantize import score_to_jianpu


ROOT = Path(__file__).resolve().parents[2]
JIANPU = ROOT / "vendor" / "jianpu-ly" / "jianpu-ly.py"
LILYPOND = ROOT / "tools" / "lilypond-2.24.4" / "bin" / "lilypond.exe"


class RenderArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    jly_path: str
    lilypond_path: str
    svg_paths: list[str]
    midi_path: str | None = None
    log_path: str


def _apply_score_tempos(midi_path: Path, score: Score) -> None:
    """Make the generated MIDI tempo map match Score.tempo_events."""

    midi = mido.MidiFile(os.fspath(midi_path))
    tempo_events = score.tempo_events or [TempoEvent(start_tick=0, bpm=score.bpm)]
    absolute_tempos = {
        int(round(event.start_tick * midi.ticks_per_beat / score.quarter_ticks)): event
        for event in tempo_events
    }
    if not midi.tracks:
        midi.tracks.append(mido.MidiTrack())
    tempo_track = midi.tracks[0]
    absolute_messages: list[tuple[int, int, mido.MetaMessage | mido.Message]] = []
    absolute = 0
    for message in tempo_track:
        absolute += message.time
        if message.type != "set_tempo":
            absolute_messages.append((absolute, 1, message.copy()))
    for tick, tempo in absolute_tempos.items():
        absolute_messages.append((tick, 0, mido.MetaMessage("set_tempo", tempo=int(round(60_000_000 / tempo.bpm)))) )
    absolute_messages.sort(key=lambda item: (item[0], item[1]))
    tempo_track[:] = []
    previous = 0
    for tick, _priority, message in absolute_messages:
        message.time = max(0, tick - previous)
        tempo_track.append(message)
        previous = tick
    midi.save(os.fspath(midi_path))


def _safe_basename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or "score"


_SVG_PAGE_SUFFIX = re.compile(r"(?:^|[-_.])(?:page[-_]?)?(\d+)\.svg$", re.IGNORECASE)
_LAYOUT_OPEN = re.compile(r"(?m)^(?P<indent>[ \t]*)\\layout[ \t]*\{(?P<eol>\r?\n|$)")
_INSTRUMENT_NAME = re.compile(
    r'(?m)^(?P<indent>[ \t]*)instrumentName[ \t]*=[ \t]*"(?P<label>(?:\\[^\r\n]|[^\r\n"])*)"[ \t]*(?P<eol>\r?\n|$)'
)
_REST_HACK_NOTE = re.compile(r'''\\note-mod "0" c(?=[,']|[0-9])''')
_COMPOSITION_REST_FILTER = """#(define (composition-filter-rest-heads grob)
  (let ((items (ly:grob-object grob 'items-worth-living)))
    (if (ly:grob-array? items)
      (ly:grob-set-object! grob 'items-worth-living
        (ly:grob-list->grob-array
          (filter (lambda (item)
            (not (assoc-get 'composition-rest (ly:grob-property item 'details '()) #f)))
            (ly:grob-array->list items)))))))
"""


def _is_melody_harmony_score(score: Score) -> bool:
    """Enable the combined-score layout only for its explicit metadata marker."""

    return isinstance(score.metadata.get("melody_harmony"), Mapping) or score.metadata.get("notation_engine") == "direct-jianpu"


def _melody_harmony_role_label(label: str) -> tuple[str, str] | None:
    """Map a composed lane label to stable user-facing long and short labels."""

    if any(role in label for role in ("高音", "和声", "低音", "人声")):
        label = re.sub(r" chord lane (\d+)", r"分\1", label)
        short_label = re.search(r"(?:高音|和声|低音|人声).*$", label)
        return label, short_label.group(0) if short_label else label
    if label.startswith("主旋律"):
        return "主旋律", "主旋律"
    if label.startswith("伴奏和弦"):
        return "伴奏和弦", "和弦"
    return None


def _prepare_melody_harmony_lilypond(text: str) -> str:
    """Tidy composed-score labels and hide empty lanes per LilyPond system.

    The jianpu serializer emits one RhythmicStaff per composed lane. Empty
    lanes are expected when a sparse lane has no event in a system, so the
    combined score opts into LilyPond's first-system-aware
    ``\\RemoveAllEmptyStaves`` command. This changes notation layout only;
    the separate MIDI score in the source remains untouched.
    """

    def replace_instrument_name(match: re.Match[str]) -> str:
        role = _melody_harmony_role_label(match.group("label"))
        if role is None:
            return match.group(0)
        long_label, short_label = role
        indent = match.group("indent")
        eol = match.group("eol") or "\n"
        return (
            f'{indent}instrumentName = "{long_label}"{eol}'
            f'{indent}shortInstrumentName = "{short_label}"{eol}'
        )

    prepared = _INSTRUMENT_NAME.sub(replace_instrument_name, text)
    # jianpu-ly's rest hack writes a short zero placeholder as a pitched
    # ``c`` note so it can keep beams intact.  LilyPond consequently treats
    # an otherwise empty lane as alive.  Mark those generated placeholders
    # so the VerticalAxisGroup callback below can remove them from
    # ``items-worth-living`` while leaving their notation and beams visible.
    prepared, _ = _REST_HACK_NOTE.subn(
        lambda match: (
            r"\tweak NoteHead.details #'((composition-rest . #t)) " + match.group(0)
        ),
        prepared,
    )
    layout_matches = list(_LAYOUT_OPEN.finditer(prepared))
    if not layout_matches:
        raise RuntimeError("combined score LilyPond source has no layout block")
    layout = layout_matches[-1]
    layout_instructions = (
        "  indent = 26\\mm\n"
        "  short-indent = 18\\mm\n"
        "  \\context {\n"
        "    \\RhythmicStaff\n"
        "    \\RemoveAllEmptyStaves\n"
        "    \\override VerticalAxisGroup.before-line-breaking = #composition-filter-rest-heads\n"
        "  }\n"
    )
    return (
        _COMPOSITION_REST_FILTER
        + prepared[: layout.end()]
        + layout_instructions
        + prepared[layout.end() :]
    )


def natural_svg_sort_key(path: str | Path) -> tuple[int, int, str]:
    """Return a stable page-aware key for LilyPond and service SVG names.

    LilyPond emits names such as ``score-1.svg`` and ``score-10.svg``.  A
    normal lexical sort puts page 10 before page 2, which also corrupts the
    composed long image and the public page numbers.  Numbered pages sort
    first by their integer suffix; a single unnumbered SVG is kept after any
    numbered pages and ``*.long.svg`` is kept after that.
    """

    name = Path(path).name
    folded = name.casefold()
    if folded.endswith(".long.svg"):
        return (2, 0, folded)
    match = _SVG_PAGE_SUFFIX.search(folded) if folded.endswith(".svg") else None
    if match:
        return (0, int(match.group(1)), folded)
    if folded.endswith(".svg"):
        return (1, 0, folded)
    return (3, 0, folded)


def render_score(score: Score, output_dir: str | Path, *, basename: str = "score") -> RenderArtifacts:
    """Render a Score and return only artifacts beneath the requested directory."""

    if not JIANPU.is_file():
        raise FileNotFoundError(JIANPU)
    if not LILYPOND.is_file():
        raise FileNotFoundError(LILYPOND)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_basename(basename)
    jly_path = destination / f"{safe_name}.jly"
    lilypond_path = destination / f"{safe_name}.ly"
    prefix = destination / safe_name
    log_path = destination / f"{safe_name}.lilypond.log"
    render_suffixes = {".jly", ".ly", ".svg", ".mid", ".midi", ".log"}
    for path in destination.glob(f"{safe_name}*"):
        # Keep adjacent score/diagnostic JSON and source artifacts.  The
        # service writes those before rendering so a later LilyPond failure
        # still leaves the completed standardization prefix available.
        if path.is_file() and path.suffix.lower() in render_suffixes:
            path.unlink()

    jly_text = score_to_jianpu(score)
    jly_path.write_text(jly_text, encoding="utf-8")
    converter = subprocess.run(
        [os.fspath(Path(sys.executable)), os.fspath(JIANPU), os.fspath(jly_path)],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if converter.returncode:
        raise RuntimeError(f"jianpu-ly failed ({converter.returncode}): {converter.stderr[-4000:]}")
    lilypond_text = converter.stdout
    if _is_melody_harmony_score(score):
        lilypond_text = _prepare_melody_harmony_lilypond(lilypond_text)
    lilypond_path.write_text(lilypond_text, encoding="utf-8")
    lilypond = subprocess.run(
        [os.fspath(LILYPOND), "--svg", "-o", os.fspath(prefix), os.fspath(lilypond_path)],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path.write_text(lilypond.stdout + lilypond.stderr, encoding="utf-8")
    if lilypond.returncode:
        raise RuntimeError(f"LilyPond failed ({lilypond.returncode}): {lilypond.stderr[-4000:]}")

    svg_paths = sorted(destination.glob(f"{safe_name}*.svg"), key=natural_svg_sort_key)
    midi_candidates = sorted(destination.glob(f"{safe_name}*.mid")) + sorted(destination.glob(f"{safe_name}*.midi"))
    if not svg_paths:
        raise RuntimeError("LilyPond completed without SVG artifacts")
    if midi_candidates:
        _apply_score_tempos(midi_candidates[0], score)
    for path in [*svg_paths, *midi_candidates, jly_path, lilypond_path, log_path]:
        if destination not in path.resolve().parents:
            raise RuntimeError(f"renderer produced an artifact outside output directory: {path}")
    return RenderArtifacts(
        output_dir=os.fspath(destination),
        jly_path=os.fspath(jly_path),
        lilypond_path=os.fspath(lilypond_path),
        svg_paths=[os.fspath(path) for path in svg_paths],
        midi_path=os.fspath(midi_candidates[0]) if midi_candidates else None,
        log_path=os.fspath(log_path),
    )


def write_score_json(score: Score, path: str | Path) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(score.model_dump(mode="json"), indent=2), encoding="utf-8")
