"""Isolated music21 worker for MusicXML -> versioned notation JSON.

The API environment must not import music21.  This process is launched with
``.venv-notation`` by ``musicxml_standardize.py`` and emits only JSON-safe
primitive values.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

EXPECTED_MUSIC21 = "9.9.2"
SCHEMA_VERSION = "1.0"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _offset_in_hierarchy(element: Any, parent: Any) -> float:
    try:
        return _number(element.getOffsetInHierarchy(parent))
    except Exception:  # noqa: BLE001 - music21 adapters may raise library-specific exceptions
        return _number(element.offset)


def _context(element: Any, class_name: str) -> Any | None:
    try:
        return element.getContextByClass(class_name)
    except Exception:  # noqa: BLE001 - music21 adapters may raise library-specific exceptions
        return None


def _pitch_midi(pitch: Any) -> int:
    return max(0, min(127, round(float(pitch.midi))))


def _tie_type(value: Any) -> str | None:
    tie = getattr(value, "tie", None)
    kind = getattr(tie, "type", None)
    return str(kind) if kind in {"start", "stop", "continue"} else None


def _ordered_pitch_ties(values: Any) -> tuple[list[int], list[str | None]]:
    """Sort chord pitches while keeping each pitch's tie slot attached."""

    pitch_ties: dict[int, str | None] = {}
    for value in values:
        pitch = _pitch_midi(value.pitch)
        tie = _tie_type(value)
        if pitch in pitch_ties:
            previous = pitch_ties[pitch]
            if previous is not None and tie is not None and previous != tie:
                raise ValueError(f"duplicate chord pitch {pitch} has conflicting tie types")
            if previous is None:
                pitch_ties[pitch] = tie
        else:
            pitch_ties[pitch] = tie
    pitches = sorted(pitch_ties)
    return pitches, [pitch_ties[pitch] for pitch in pitches]


def _duration_details(element: Any) -> dict[str, Any]:
    duration = element.duration
    tuplets = list(getattr(duration, "tuplets", ()) or ())
    tuplet = tuplets[0] if tuplets else None
    return {
        "duration_quarter": _number(duration.quarterLength),
        "dots": int(getattr(duration, "dots", 0) or 0),
        "tuplet_actual": int(getattr(tuplet, "numberNotesActual", 0) or 0) or None,
        "tuplet_normal": int(getattr(tuplet, "numberNotesNormal", 0) or 0) or None,
        "grace": bool(getattr(duration, "isGrace", False)),
    }


def _event_record(element: Any, *, part: Any, part_index: int, event_index: int) -> dict[str, Any] | None:
    details = _duration_details(element)
    duration_quarter = details["duration_quarter"]
    if duration_quarter <= 0:
        # Grace notes are not representable on the current positive-duration
        # ScoreNote contract.  Keep a structured diagnostic rather than
        # inventing a duration in the worker output.
        return {
            "event_id": f"p{part_index}:grace:{event_index}",
            "kind": "grace",
            "offset_quarter": _offset_in_hierarchy(element, part),
            **details,
            "pitches": [],
            "tie_types": [],
            "voice": "1",
            "staff": 1,
            "measure_number": None,
        }
    from music21 import chord, note

    if isinstance(element, note.Rest):
        kind = "rest"
        pitches: list[int] = []
        tie_types: list[str] = []
    elif isinstance(element, chord.Chord):
        kind = "chord"
        # Sort pitch and tie together.  A chord tie is attached to each
        # music21 note, so filtering empty ties after sorting would shift a
        # C-only tie onto the E or G slot.  Duplicate MIDI pitches are
        # collapsed only after their tie metadata has been reconciled.
        pitches, tie_types = _ordered_pitch_ties(element.notes)
    elif isinstance(element, note.Note):
        kind = "note"
        pitches = [_pitch_midi(element.pitch)]
        note_tie = _tie_type(element)
        tie_types = [] if note_tie is None else [note_tie]
    else:
        return None

    measure = _context(element, "Measure")
    voice = _context(element, "Voice")
    staff = _context(element, "Staff")
    voice_id = str(getattr(voice, "id", None) or "1")
    staff_id = getattr(staff, "id", None)
    try:
        staff_number = int(staff_id) if staff_id is not None else 1
    except (TypeError, ValueError):
        staff_number = 1
    measure_number = getattr(measure, "number", None)
    present_ties = [value for value in tie_types if value is not None]
    tie = present_ties[0] if present_ties and len(present_ties) == len(tie_types) and all(value == present_ties[0] for value in present_ties) else None
    return {
        "event_id": f"p{part_index}:e{event_index}",
        "kind": kind,
        "offset_quarter": _offset_in_hierarchy(element, part),
        **details,
        "pitches": pitches,
        "tie": tie,
        "tie_types": tie_types,
        "voice": voice_id,
        "staff": staff_number,
        "measure_number": int(measure_number) if isinstance(measure_number, int) else None,
    }


def _measure_records(part: Any, part_index: int) -> list[dict[str, Any]]:
    from music21 import stream

    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for measure in part.recurse().getElementsByClass(stream.Measure):
        identity = id(measure)
        if identity in seen:
            continue
        seen.add(identity)
        start = _offset_in_hierarchy(measure, part)
        bar_duration = getattr(getattr(measure, "barDuration", None), "quarterLength", None)
        actual_duration = _number(getattr(measure.duration, "quarterLength", 0.0))
        # ``barDuration`` is the nominal full-bar length.  For a numbered
        # zero pickup it would hide the real partial duration, so use the
        # measure's actual content span instead.
        duration = actual_duration if getattr(measure, "number", None) == 0 else _number(bar_duration, actual_duration)
        time_signatures = list(measure.recurse().getElementsByClass("TimeSignature"))
        ratio = str(time_signatures[0].ratioString) if time_signatures else None
        result.append(
            {
                "part_index": part_index,
                "number": int(measure.number) if isinstance(measure.number, int) else len(result) + 1,
                "start_quarter": start,
                "duration_quarter": duration,
                "end_quarter": start + duration,
                "time_signature": ratio,
                "is_pickup": bool(getattr(measure, "number", None) == 0),
            }
        )
    return result


def _tempo_records(score: Any) -> list[dict[str, Any]]:
    from music21 import tempo

    result: list[dict[str, Any]] = []
    for item in score.recurse().getElementsByClass(tempo.MetronomeMark):
        number = getattr(item, "number", None)
        if number is None:
            continue
        result.append({"offset_quarter": _offset_in_hierarchy(item, score), "bpm": _number(number, 120.0)})
    return sorted(result, key=lambda value: (value["offset_quarter"], value["bpm"]))


def _time_signature_records(score: Any) -> list[dict[str, Any]]:
    from music21 import meter

    result: list[dict[str, Any]] = []
    for item in score.recurse().getElementsByClass(meter.TimeSignature):
        result.append(
            {
                "offset_quarter": _offset_in_hierarchy(item, score),
                "ratio": str(item.ratioString),
                "numerator": int(item.numerator),
                "denominator": int(item.denominator),
            }
        )
    return sorted(result, key=lambda value: (value["offset_quarter"], value["ratio"]))


def _key_signature_records(score: Any) -> list[dict[str, Any]]:
    from music21 import key

    result: list[dict[str, Any]] = []
    for item in score.recurse().getElementsByClass(key.KeySignature):
        try:
            as_key = item.asKey()
            name = str(as_key.tonic.name) + ("m" if as_key.mode == "minor" else "")
        except Exception:  # noqa: BLE001 - malformed key objects remain diagnosable as text
            name = str(item)
        result.append(
            {
                "offset_quarter": _offset_in_hierarchy(item, score),
                "key": name,
                "sharps": int(getattr(item, "sharps", 0) or 0),
            }
        )
    return sorted(result, key=lambda value: (value["offset_quarter"], value["key"]))


def build_payload(input_path: Path) -> dict[str, Any]:
    installed = importlib.metadata.version("music21")
    if installed != EXPECTED_MUSIC21:
        raise RuntimeError(f"music21 version mismatch: expected {EXPECTED_MUSIC21}, got {installed}")
    from music21 import converter

    score = converter.parse(str(input_path))
    parts = list(score.parts)
    if not parts:
        parts = [score]
    part_records: list[dict[str, Any]] = []
    all_measures: list[dict[str, Any]] = []
    for part_index, part in enumerate(parts):
        events: list[dict[str, Any]] = []
        part_id = str(getattr(part, "id", None) or f"part-{part_index + 1}")
        # music21 splits a multi-staff MusicXML part into synthetic parts such
        # as ``P1-Staff1`` and ``P1-Staff2``.  The Staff context on notes is
        # not reliable after that split, so preserve the explicit suffix.
        staff_match = re.search(r"-Staff(\d+)$", part_id)
        derived_staff = int(staff_match.group(1)) if staff_match else None
        for event_index, element in enumerate(part.recurse().notesAndRests):
            record = _event_record(element, part=part, part_index=part_index, event_index=event_index)
            if record is not None:
                if derived_staff is not None:
                    record["staff"] = derived_staff
                events.append(record)
        measures = _measure_records(part, part_index)
        all_measures.extend(measures)
        instrument = _context(part, "Instrument")
        part_records.append(
            {
                "part_id": part_id,
                "name": str(getattr(part, "partName", None) or getattr(instrument, "partName", None) or f"Part {part_index + 1}"),
                "instrument": str(getattr(instrument, "instrumentName", None) or ""),
                "highest_time_quarter": _number(getattr(part, "highestTime", 0.0)),
                "events": sorted(events, key=lambda value: (value["offset_quarter"], value["staff"], value["voice"], value["event_id"])),
                "measures": measures,
            }
        )
    time_signatures = _time_signature_records(score)
    keys = _key_signature_records(score)
    tempos = _tempo_records(score)
    highest = max(
        [_number(getattr(score, "highestTime", 0.0))]
        + [float(part["highest_time_quarter"]) for part in part_records]
        + [float(measure["end_quarter"]) for measure in all_measures],
        default=0.0,
    )
    first_measure = min(all_measures, key=lambda value: value["start_quarter"], default=None)
    first_ratio = time_signatures[0]["ratio"] if time_signatures else (first_measure or {}).get("time_signature") or "4/4"
    expected_bar = None
    if "/" in first_ratio:
        numerator, denominator = (int(value) for value in first_ratio.split("/", 1))
        expected_bar = numerator * 4.0 / denominator
    pickup = {
        "is_pickup": bool(first_measure and (first_measure.get("is_pickup") or (expected_bar and first_measure["duration_quarter"] < expected_bar - 1e-6))),
        "duration_quarter": float(first_measure["duration_quarter"]) if first_measure else 0.0,
        "measure_number": first_measure.get("number") if first_measure else None,
    }
    metadata = getattr(score, "metadata", None)
    title = str(getattr(metadata, "title", None) or input_path.stem)
    return {
        "schema_version": SCHEMA_VERSION,
        "worker": "music21",
        "music21_version": installed,
        "source_path": str(input_path.resolve()),
        "title": title,
        "highest_time_quarter": highest,
        "parts": part_records,
        "measures": sorted(all_measures, key=lambda value: (value["start_quarter"], value["part_index"], value["number"])),
        "tempo_events": tempos,
        "time_signature_events": time_signatures,
        "key_signature_events": keys,
        "pickup": pickup,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    payload = build_payload(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "schema_version": SCHEMA_VERSION, "music21_version": EXPECTED_MUSIC21, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI must report any worker failure as JSON-process failure
        print(f"musicxml-worker-failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
