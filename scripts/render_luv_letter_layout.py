"""Deterministic jianpu display renderer for the frozen Luv Letter Score.

This sample consumes an existing Score JSON. It does not run recognition,
MuseScore, or jianpu-ly. The high and low display bands share one Score tick
axis; display-only MIDI 60 grouping never rewrites source staff or voice data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
HIGH_THRESHOLD = 61
TPQ = 48
MEASURE_TICKS = 192
PAGE_BARS = 4
PAGE_HEIGHT = 680
LEFT = 104
RIGHT = 38
BAR_PADDING = 18.0
HIGH_Y = 300
LOW_Y = 560
HIGH_PANEL = 100
LOW_PANEL = 370
PANEL_HEIGHT = 250


def svgtag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


KEY_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
RELATIVE_MAJOR = {
    "Am": "C",
    "Bbm": "Db",
    "Bm": "D",
    "Cm": "Eb",
    "Dm": "F",
    "Ebm": "Gb",
    "Em": "G",
    "Fm": "Ab",
    "F#m": "A",
    "Gm": "Bb",
}
SCALE = (0, 2, 4, 5, 7, 9, 11)


def jianpu_name(midi: int, key: str) -> str:
    root = RELATIVE_MAJOR.get(key, key)
    tonic = 60 + KEY_PC.get(root, 0)
    tonic -= 12 if tonic > 66 else 0
    options: list[tuple[int, int, int, int]] = []
    for degree, offset in enumerate(SCALE, 1):
        for octave in range(-12, 13):
            nominal = tonic + offset + 12 * octave
            options.append((abs(midi - nominal), midi - nominal, degree, octave))
    _distance, difference, degree, octave = min(
        options,
        key=lambda item: (item[0], abs(item[1]), abs(item[3]), item[2], item[3]),
    )
    accidental = "#" if difference > 0 else "b" if difference < 0 else ""
    marks = "'" * octave if octave > 0 else "," * (-octave) if octave < 0 else ""
    return f"{accidental}{degree}{marks}"


def pitches(event: dict[str, Any]) -> list[int]:
    values = event.get("chord_pitches")
    if values:
        return [int(item) for item in values]
    return [] if event.get("midi") is None else [int(event["midi"])]


def tie_types(event: dict[str, Any], count: int) -> list[str | None]:
    values = event.get("tie_types")
    if isinstance(values, list) and values:
        return [values[i] if i < len(values) else event.get("tie") for i in range(count)]
    return [event.get("tie") for _ in range(count)]


def canonical_event(voice_index: int, voice_id: str, event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_key": f"v{voice_index}-e{event_index}",
        "voice_index": voice_index,
        "event_index": event_index,
        "voice_id": voice_id,
        "staff": event.get("staff"),
        "source_voice": event.get("source_voice"),
        "measure_number": event.get("measure_number"),
        "start_tick": event.get("start_tick"),
        "duration_tick": event.get("duration_tick"),
        "midi": event.get("midi"),
        "chord_pitches": event.get("chord_pitches") or [],
        "tie": event.get("tie"),
        "tie_types": event.get("tie_types") or [],
        "pitches": pitches(event),
        "pitch_ties": tie_types(event, len(pitches(event))),
        "dots": int(event.get("dots") or 0),
        "tuplet_actual": event.get("tuplet_actual"),
        "tuplet_normal": event.get("tuplet_normal"),
        "source": event.get("source"),
    }


def collect(score: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    index = 0
    for voice_index, voice in enumerate(score.get("voices", [])):
        voice_id = str(voice.get("voice_id", voice_index))
        for event_index, event in enumerate(voice.get("events", [])):
            if not isinstance(event, dict):
                continue
            record = canonical_event(voice_index, voice_id, event_index, event)
            events.append(record)
            for pitch, tie_type in zip(record["pitches"], record["pitch_ties"]):
                notes.append(
                    {
                        "occurrence_index": index,
                        "source_key": record["source_key"],
                        "voice_index": voice_index,
                        "event_index": event_index,
                        "voice_id": voice_id,
                        "staff": event.get("staff"),
                        "source_voice": event.get("source_voice"),
                        "measure_number": event.get("measure_number"),
                        "pitch": int(pitch),
                        "start_tick": int(event.get("start_tick", 0)),
                        "duration_tick": int(event.get("duration_tick", 0)),
                        "tie": event.get("tie"),
                        "tie_type": tie_type,
                        "region": "高音区" if int(pitch) >= HIGH_THRESHOLD else "低音区",
                        "dots": int(event.get("dots") or 0),
                        "tuplet_actual": event.get("tuplet_actual"),
                        "tuplet_normal": event.get("tuplet_normal"),
                    }
                )
                index += 1
    return events, notes


def spans(score: dict[str, Any]) -> list[dict[str, Any]]:
    values = (score.get("metadata") or {}).get("timeline_measures") or []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        start = int(item.get("start_tick", index * MEASURE_TICKS))
        duration = int(item.get("duration_tick", MEASURE_TICKS))
        result.append(
            {
                "number": index + 1,
                "start_tick": start,
                "end_tick": int(item.get("end_tick", start + duration)),
                "duration_tick": duration,
            }
        )
    if result:
        return result
    total = int(score.get("total_ticks", 0))
    return [
        {
            "number": index + 1,
            "start_tick": index * MEASURE_TICKS,
            "end_tick": min(total, (index + 1) * MEASURE_TICKS),
            "duration_tick": min(MEASURE_TICKS, total - index * MEASURE_TICKS),
        }
        for index in range(max(1, math.ceil(total / MEASURE_TICKS)))
    ]


class XMap:
    def __init__(self, bars: list[dict[str, Any]], notes: list[dict[str, Any]]) -> None:
        self.bars = bars
        self.ranges: list[tuple[int, int, float, float]] = []
        self.anchors: list[tuple[list[int], list[float]]] = []
        self.time_positions: dict[int, float] = {}
        cursor = float(LEFT)
        self.boundary_positions: dict[int, float] = {}
        for bar_index, bar in enumerate(bars):
            start = int(bar["start_tick"])
            end = int(bar["end_tick"])
            times = {start, end, *(
                start + beat * TPQ
                for beat in range(1, max(1, (end - start) // TPQ))
                if start + beat * TPQ < end
            )}
            times.update(
                int(note["start_tick"])
                for note in notes
                if start <= int(note["start_tick"]) < end
            )
            times.update(
                min(end, int(note["start_tick"]) + int(note["duration_tick"]))
                for note in notes
                if start < int(note["start_tick"]) + int(note["duration_tick"]) <= end
            )
            ordered = sorted(times)
            if len(ordered) == 1:
                ordered.append(end)
            gaps = [
                max(34.0, 232.0 * (right - left) / max(1, end - start))
                for left, right in zip(ordered, ordered[1:])
            ]
            bar_left = cursor
            positions = [cursor + BAR_PADDING]
            for gap in gaps:
                positions.append(positions[-1] + gap)
            for tick, position in zip(ordered, positions):
                if tick not in {start, end}:
                    self.time_positions[tick] = position
            self.anchors.append((ordered, positions))
            self.boundary_positions[start] = positions[0]
            if bar_index == len(bars) - 1:
                self.boundary_positions[end] = positions[-1]
            bar_right = positions[-1] + BAR_PADDING
            self.ranges.append((start, end, bar_left, bar_right))
            cursor = bar_right
        self.width = max(1160.0, cursor + RIGHT)

    def x(self, tick: int) -> float:
        if tick in self.boundary_positions:
            return self.boundary_positions[tick]
        if tick in self.time_positions:
            return self.time_positions[tick]
        for (start, end, left, right), (ordered, positions) in zip(self.ranges, self.anchors):
            if start <= tick <= end:
                for index, (first, second) in enumerate(zip(ordered, ordered[1:])):
                    if first <= tick <= second:
                        span = max(1, second - first)
                        return positions[index] + (positions[index + 1] - positions[index]) * (tick - first) / span
                return right
        return self.ranges[-1][3]


def text(parent: ET.Element, x: float, y: float, value: str, **attrs: Any) -> ET.Element:
    element = ET.SubElement(
        parent,
        svgtag("text"),
        {"x": f"{x:.2f}", "y": f"{y:.2f}", **{key: str(value) for key, value in attrs.items()}},
    )
    element.text = value
    return element


def svg_line(parent: ET.Element, x1: float, y1: float, x2: float, y2: float, **attrs: Any) -> ET.Element:
    return ET.SubElement(
        parent,
        svgtag("line"),
        {
            "x1": f"{x1:.2f}",
            "y1": f"{y1:.2f}",
            "x2": f"{x2:.2f}",
            "y2": f"{y2:.2f}",
            **{key: str(value) for key, value in attrs.items()},
        },
    )


def note_y(region: str, stack_rank: int, stack_count: int) -> float:
    """Place one onset stack around the region baseline.

    Jianpu keeps a running melody on one horizontal line and uses octave marks
    on the number itself. Chord members are the only items stacked vertically,
    so the display does not turn into one staff per pitch.
    """

    baseline = HIGH_Y if region == "高音区" else LOW_Y
    # The lowest chord member stays on the region baseline; a single note is
    # therefore stable across the whole melody. Higher members rise in fixed
    # 40px steps so octave points and duration marks have their own space.
    return baseline - (stack_count - 1 - stack_rank) * 40.0


def duration_shape(duration: int, dots: int = 0) -> tuple[int, int]:
    """Return (underline count, short long-value dash count).

    The 48 TPQ grid makes common jianpu values exact: 24/12/6 ticks are
    eighth/sixteenth/32nd, while 36/18 are dotted eighth/dotted sixteenth.
    Values at or above a quarter use short local dash marks, never a line that
    spans the next onset.
    """

    if duration in {24}:
        return 1, 0
    if duration in {12}:
        return 2, 0
    if duration in {6}:
        return 3, 0
    if duration == 36:
        return 1, 0
    if duration == 18:
        return 2, 0
    if duration < TPQ:
        return 3, 0
    if duration == TPQ:
        return 0, 0
    if duration == 72:
        return 0, 0
    if duration >= TPQ:
        return 0, max(1, round(duration / TPQ) - 1)
    return 0, 0


def duration_lines(parent: ET.Element, x: float, y: float, duration: int, dots: int = 0) -> None:
    underlines, dash_count = duration_shape(duration, dots)
    for index in range(underlines):
        svg_line(parent, x - 10, y + 12 + index * 4, x + 10, y + 12 + index * 4, stroke="#183b5c", **{"stroke-width": 1.55})
    # Long values use independent, short dash marks after the number on the
    # number baseline. The
    # exact duration remains in the SVG title and event manifest.
    for index in range(dash_count):
        left = x + 8 + index * 12
        svg_line(parent, left, y + 1, left + 9, y + 1, stroke="#183b5c", **{"stroke-width": 1.55})
    # A dotted half (144 ticks) is shown as a half plus two extension dashes;
    # adding a second dot would count the same duration twice.
    if dots and duration != 144:
        ET.SubElement(parent, svgtag("circle"), {"cx": f"{x + 14:.2f}", "cy": f"{y - 1:.2f}", "r": "2.1", "fill": "#183b5c", "data-duration-dot": "true"})


def draw_jianpu(parent: ET.Element, x: float, y: float, value: str) -> None:
    """Draw a numbered note with octave points instead of apostrophe glyphs."""

    accidental = ""
    if value.startswith(("#", "b")):
        accidental, value = value[0], value[1:]
    digits = value.rstrip("',")
    marks = value[len(digits):]
    if accidental:
        text(parent, x - 14, y, accidental, **{"class": "note", "text-anchor": "middle"})
    text(parent, x, y, digits, **{"class": "note", "text-anchor": "middle"})
    above = marks.count("'")
    below = marks.count(",")
    for index in range(above):
        ET.SubElement(parent, svgtag("circle"), {"cx": f"{x:.2f}", "cy": f"{y - 24 - index * 6:.2f}", "r": "2.2", "fill": "#122b42", "data-octave": "up"})
    for index in range(below):
        ET.SubElement(parent, svgtag("circle"), {"cx": f"{x:.2f}", "cy": f"{y + 7 + index * 5:.2f}", "r": "2.2", "fill": "#122b42", "data-octave": "down"})


def tie_pairs(notes: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        grouped[(int(note["voice_index"]), int(note["pitch"]))].append(note)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for values in grouped.values():
        values.sort(key=lambda item: (item["start_tick"], item["event_index"], item["occurrence_index"]))
        for current, following in zip(values, values[1:]):
            current_state = current["tie_type"] or current["tie"]
            following_state = following["tie_type"] or following["tie"]
            if current_state in {"start", "continue"} and following_state in {"continue", "stop"} and following["start_tick"] == current["start_tick"] + current["duration_tick"]:
                pairs.append((current, following))
    return pairs


def page_svg(
    score: dict[str, Any],
    bars: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    start_bar: int,
    end_bar: int,
    page_number: int,
    page_count: int,
    label: str = "",
    region_mode: str = "pitch",
) -> tuple[ET.Element, dict[str, Any]]:
    page_bars = bars[start_bar:end_bar]
    first_tick = int(page_bars[0]["start_tick"])
    last_tick = int(page_bars[-1]["end_tick"])
    def display_region(item: dict[str, Any]) -> str:
        if region_mode == "staff":
            return "高音区" if item.get("staff") == 1 else "低音区"
        return str(item["region"])

    page_notes = [
        dict(item, display_region=display_region(item))
        for item in notes
        if first_tick <= item["start_tick"] < last_tick
    ]
    active_notes = [
        dict(item, display_region=display_region(item))
        for item in notes
        if item["start_tick"] < last_tick and item["start_tick"] + item["duration_tick"] > first_tick
    ]
    xmap = XMap(page_bars, page_notes)
    title = str(score.get("title") or "Luv Letter")
    page_label = f"小节 {page_bars[0]['number']}–{page_bars[-1]['number']}"
    if label:
        page_label += f" · {label}"
    root = ET.Element(
        svgtag("svg"),
        {
            "width": f"{xmap.width:g}",
            "height": str(PAGE_HEIGHT),
            "viewBox": f"0 0 {xmap.width:g} {PAGE_HEIGHT}",
            "role": "img",
        },
    )
    defs = ET.SubElement(root, svgtag("defs"))
    style = ET.SubElement(defs, svgtag("style"))
    style.text = ".title{font:700 23px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#162b3f}.subtitle,.legend{font:13px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#587087}.measure{font:700 12px 'Microsoft YaHei',sans-serif;fill:#31506a}.region{font:700 16px 'Microsoft YaHei',sans-serif;fill:#1f5f8b}.low{fill:#6f4c2d}.note{font:700 20px 'Microsoft YaHei',sans-serif;fill:#122b42}.rest{font:18px 'Microsoft YaHei',sans-serif;fill:#9aabb9}.tick,.duration{font:10px Consolas,'Microsoft YaHei',sans-serif;fill:#6b7f90}"
    text(root, 28, 34, title, **{"class": "title"})
    text(root, 28, 57, page_label, **{"class": "subtitle"})
    canvas = ET.SubElement(root, svgtag("g"), {"class": "canvas"})
    score_key = str(score.get("key") or "C")
    jianpu_key = RELATIVE_MAJOR.get(score_key, score_key)
    bpm = round(float(score.get("bpm", 0))) if score.get("bpm") is not None else ""
    text(canvas, 28, 80, f"1={jianpu_key}  ·  {score.get('time_signature', '4/4')}  ·  ♩={bpm}", **{"class": "legend"})
    text(canvas, xmap.width - 32, 80, "数字=音高  下方短线=时值  红色曲线=延音", **{"class": "legend", "text-anchor": "end"})
    for region, panel_y, baseline, region_class in (
        ("高音区", HIGH_PANEL, HIGH_Y, "region"),
        ("低音区", LOW_PANEL, LOW_Y, "region low"),
    ):
        ET.SubElement(canvas, svgtag("rect"), {"x": "24", "y": str(panel_y), "width": f"{xmap.width - 48:g}", "height": str(PANEL_HEIGHT), "rx": "10", "fill": "#f3f8fc" if region == "高音区" else "#fcf8f2", "stroke": "#d9e4ec"})
        text(canvas, 38, panel_y + 24, region, **{"class": region_class})
        text(canvas, 38, panel_y + 44, "共同时间轴 · 同起和弦按列叠放", **{"class": "legend"})
        svg_line(canvas, xmap.left if hasattr(xmap, "left") else LEFT, baseline, xmap.width - RIGHT, baseline, stroke="#b9cbd8", **{"stroke-width": 1.0, "stroke-dasharray": "3 5"})
    for bar, (_start, _end, left, right) in zip(page_bars, xmap.ranges):
        svg_line(canvas, left, 108, left, 620, stroke="#63829a", **{"stroke-width": 1.15})
        text(canvas, left + 8, 99, str(bar["number"]), **{"class": "measure"})
        for beat in range(1, 4):
            tick = int(bar["start_tick"]) + beat * TPQ
            if tick < int(bar["end_tick"]):
                beat_x = xmap.x(tick)
                svg_line(canvas, beat_x, 116, beat_x, 608, stroke="#d6e1e8", **{"stroke-width": 0.7, "stroke-dasharray": "2 6"})
        text(canvas, (left + right) / 2, 638, f"{_start}–{_end}t", **{"class": "tick", "text-anchor": "middle"})
    svg_line(canvas, xmap.ranges[-1][3], 108, xmap.ranges[-1][3], 620, stroke="#63829a", **{"stroke-width": 1.15})

    onset_counts: dict[tuple[str, int], int] = defaultdict(int)
    for note in page_notes:
        onset_counts[(note["display_region"], int(note["start_tick"]))] += 1
    ranks: dict[tuple[str, int], int] = defaultdict(int)
    positions: dict[int, tuple[float, float]] = {}
    render_order: list[tuple[dict[str, Any], float, float]] = []
    for note in sorted(page_notes, key=lambda item: (item["start_tick"], item["display_region"], -item["pitch"], item["occurrence_index"])):
        rank_key = (note["display_region"], int(note["start_tick"]))
        rank = ranks[rank_key]
        ranks[rank_key] += 1
        x = xmap.x(int(note["start_tick"]))
        y = note_y(note["display_region"], rank, onset_counts[rank_key])
        positions[int(note["occurrence_index"])] = (x, y)
        render_order.append((note, x, y))
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in page_notes:
        source_groups[note["source_key"]].append(note)
    duration_anchor: dict[int, bool] = {}
    for values in source_groups.values():
        anchor = max(values, key=lambda item: positions[int(item["occurrence_index"])][1])
        for value in values:
            duration_anchor[int(value["occurrence_index"])] = value is anchor

    for note, x, y in render_order:
        group = ET.SubElement(
            canvas,
            svgtag("g"),
            {
                "id": f"note-{note['occurrence_index']}",
                "data-source-key": note["source_key"],
                "data-occurrence-index": str(note["occurrence_index"]),
                "data-start-tick": str(note["start_tick"]),
                "data-duration-tick": str(note["duration_tick"]),
                "data-pitch": str(note["pitch"]),
                "data-region": note["display_region"],
            },
        )
        state = note["tie_type"] or note["tie"] or "none"
        title_node = ET.SubElement(group, svgtag("title"))
        title_node.text = f"staff {note['staff']} / source voice {note['source_voice']} / {note['voice_id']}; MIDI {note['pitch']}; start {note['start_tick']}t; duration {note['duration_tick']}t; tie {state}; measure {note['measure_number']}"
        draw_jianpu(group, x, y, jianpu_name(int(note["pitch"]), str(score.get("key") or "C")))
        if duration_anchor[int(note["occurrence_index"])]:
            duration_lines(group, x, y, int(note["duration_tick"]), int(note["dots"]))
        if int(note["duration_tick"]) not in {4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 144, 192}:
            text(group, x, y + 29, f"{note['duration_tick']}t", **{"class": "duration", "text-anchor": "middle"})
        if state in {"continue", "stop"} and int(note["start_tick"]) == first_tick:
            text(group, x - 15, y - 13, "~", **{"class": "duration", "text-anchor": "middle"})

    tie_groups: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for current, following in tie_pairs(notes):
        if current["occurrence_index"] in positions or following["occurrence_index"] in positions:
            tie_groups[(current["source_key"], following["source_key"])].append((current, following))
    for pairs in tie_groups.values():
        if len(pairs) > 1:
            start_coords = [positions[item[0]["occurrence_index"]] for item in pairs if item[0]["occurrence_index"] in positions]
            end_coords = [positions[item[1]["occurrence_index"]] for item in pairs if item[1]["occurrence_index"] in positions]
            if not start_coords:
                start_coords = [(xmap.x(first_tick), note_y("高音区", 0, 1))]
            if not end_coords:
                end_coords = [(xmap.x(last_tick), note_y("高音区", 0, 1))]
            start_x = min(item[0] for item in start_coords)
            end_x = max(item[0] for item in end_coords)
            if end_x <= start_x:
                continue
            outside_y = min([item[1] for item in start_coords + end_coords]) - 22
            control = outside_y - 16
            path = ET.SubElement(canvas, svgtag("path"), {"d": f"M {start_x + 12:.2f},{outside_y:.2f} C {start_x + (end_x - start_x) * .3:.2f},{control:.2f} {end_x - (end_x - start_x) * .3:.2f},{control:.2f} {end_x - 12:.2f},{outside_y:.2f}", "fill": "none", "stroke": "#b05f55", "stroke-width": "1.3", "stroke-linecap": "round", "data-tie": "true", "data-tie-chord": "true"})
            title_node = ET.SubElement(path, svgtag("title"))
            title_node.text = "chord tie MIDI " + ",".join(str(item[0]["pitch"]) for item in pairs)
            continue
        current, following = pairs[0]
        current_region = (("高音区" if current.get("staff") == 1 else "低音区") if region_mode == "staff" else current["region"])
        following_region = (("高音区" if following.get("staff") == 1 else "低音区") if region_mode == "staff" else following["region"])
        start_x, start_y = positions.get(current["occurrence_index"], (xmap.x(first_tick), note_y(current_region, 0, 1)))
        end_x, end_y = positions.get(following["occurrence_index"], (xmap.x(last_tick), note_y(following_region, 0, 1)))
        if end_x <= start_x:
            continue
        control = min(start_y, end_y) - 28
        path = ET.SubElement(canvas, svgtag("path"), {"d": f"M {start_x + 12:.2f},{start_y - 15:.2f} C {start_x + (end_x - start_x) * .3:.2f},{control:.2f} {end_x - (end_x - start_x) * .3:.2f},{control:.2f} {end_x - 12:.2f},{end_y - 15:.2f}", "fill": "none", "stroke": "#b05f55", "stroke-width": "1.25", "stroke-linecap": "round", "data-tie": "true"})
        title_node = ET.SubElement(path, svgtag("title"))
        title_node.text = f"tie MIDI {current['pitch']}: {current['start_tick']}–{following['start_tick'] + following['duration_tick']} ticks"

    for bar in page_bars:
        for region, baseline in (("高音区", HIGH_Y), ("低音区", LOW_Y)):
            start = int(bar["start_tick"])
            end = int(bar["end_tick"])
            active = sorted(
                (
                    max(start, int(item["start_tick"])),
                    min(end, int(item["start_tick"]) + int(item["duration_tick"])),
                )
                for item in active_notes
                if item["display_region"] == region
                and int(item["start_tick"]) < end
                and int(item["start_tick"]) + int(item["duration_tick"]) > start
            )
            merged: list[list[int]] = []
            for active_start, active_end in active:
                if not merged or active_start > merged[-1][1]:
                    merged.append([active_start, active_end])
                else:
                    merged[-1][1] = max(merged[-1][1], active_end)
            cursor = start

            def draw_rest(rest_start: int, rest_end: int, *, whole_bar: bool = False) -> None:
                if rest_end <= rest_start:
                    return
                rest_x = xmap.x((rest_start + rest_end) // 2) if whole_bar else xmap.x(rest_start)
                rest_group = ET.SubElement(canvas, svgtag("g"), {"data-rest-start-tick": str(rest_start), "data-rest-duration-tick": str(rest_end - rest_start), "data-region": region})
                rest_title = ET.SubElement(rest_group, svgtag("title"))
                rest_title.text = f"{region} rest; start {rest_start}t; duration {rest_end - rest_start}t"
                text(rest_group, rest_x, baseline, "0", **{"class": "rest", "text-anchor": "middle"})
                duration_lines(rest_group, rest_x, baseline, rest_end - rest_start)
                if whole_bar:
                    text(rest_group, rest_x, baseline + 27, "全小节休止", **{"class": "duration", "text-anchor": "middle"})
                elif rest_end - rest_start not in {4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 144, 192}:
                    text(rest_group, rest_x, baseline + 27, f"{rest_end - rest_start}t", **{"class": "duration", "text-anchor": "middle"})

            if not merged:
                draw_rest(start, end, whole_bar=True)
                continue
            for active_start, active_end in merged:
                if active_start > cursor:
                    draw_rest(cursor, active_start)
                cursor = max(cursor, active_end)
            if cursor < end:
                draw_rest(cursor, end)
    text(canvas, 28, PAGE_HEIGHT - 14, "排版样本：仅重排显示；不解决错音/节奏；score.mid 保留原始 MIDI。", **{"class": "legend"})
    info = {
        "page": page_number,
        "measure_start": page_bars[0]["number"],
        "measure_end": page_bars[-1]["number"],
        "start_tick": first_tick,
        "end_tick": last_tick,
        "width": xmap.width,
        "height": PAGE_HEIGHT,
        "source_event_keys": sorted({note["source_key"] for note in page_notes}),
        "note_occurrence_indices": sorted(int(note["occurrence_index"]) for note in page_notes),
    }
    return root, info


def write_svg(path: Path, root: ET.Element) -> None:
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def compose_long(paths: list[Path], destination: Path) -> dict[str, Any]:
    roots = [ET.parse(path).getroot() for path in paths]
    sizes = [(float(root.get("width", "1160")), float(root.get("height", PAGE_HEIGHT))) for root in roots]
    width = max(item[0] for item in sizes)
    gap = 18.0
    height = sum(item[1] for item in sizes) + max(0, len(paths) - 1) * gap
    root = ET.Element(svgtag("svg"), {"width": f"{width:g}", "height": f"{height:g}", "viewBox": f"0 0 {width:g} {height:g}", "role": "img"})
    cursor = 0.0
    for index, (source, (_source_width, source_height)) in enumerate(zip(roots, sizes), 1):
        wrapper = ET.SubElement(root, svgtag("g"), {"transform": f"translate(0 {cursor:g})", "data-page": str(index)})
        for child in list(source):
            wrapper.append(copy.deepcopy(child))
        cursor += source_height + gap
    write_svg(destination, root)
    return {"width": width, "height": height, "page_count": len(paths), "gap": gap}


def manifest(
    score_path: Path,
    mid_path: Path,
    copied_mid: Path,
    score: dict[str, Any],
    events: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    page_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    source_keys = sorted(item["source_key"] for item in events if item["pitches"])
    rendered_keys = sorted({key for info in page_infos for key in info["source_event_keys"]})
    source_notes: list[dict[str, Any]] = []
    occurrence_index = 0
    for event in events:
        for pitch, tie_type in zip(event["pitches"], event["pitch_ties"]):
            source_notes.append(
                {
                    "occurrence_index": occurrence_index,
                    "source_key": event["source_key"],
                    "voice_index": event["voice_index"],
                    "event_index": event["event_index"],
                    "voice_id": event["voice_id"],
                    "staff": event["staff"],
                    "source_voice": event["source_voice"],
                    "measure_number": event["measure_number"],
                    "pitch": int(pitch),
                    "start_tick": int(event["start_tick"]),
                    "duration_tick": int(event["duration_tick"]),
                    "tie": event["tie"],
                    "tie_type": tie_type,
                    "region": "高音区" if int(pitch) >= HIGH_THRESHOLD else "低音区",
                    "dots": event["dots"],
                    "tuplet_actual": event["tuplet_actual"],
                    "tuplet_normal": event["tuplet_normal"],
                }
            )
            occurrence_index += 1
    source_note_records = sorted(source_notes, key=lambda item: (item["voice_index"], item["event_index"], item["pitch"], item["occurrence_index"]))
    rendered_note_records = sorted(copy.deepcopy(notes), key=lambda item: (item["voice_index"], item["event_index"], item["pitch"], item["occurrence_index"]))
    return {
        "schema_version": "luv-letter-layout-v1",
        "purpose": "Display-only deterministic SVG layout for one frozen Score.",
        "source": {
            "score_path": os.fspath(score_path.resolve()),
            "mid_path": os.fspath(mid_path.resolve()),
            "mid_sha256": digest(mid_path),
            "copied_mid": os.fspath(copied_mid.resolve()),
            "copied_mid_sha256": digest(copied_mid),
            "sample_date": datetime.fromtimestamp(score_path.stat().st_mtime).isoformat(timespec="seconds"),
            "source_commit": source_commit(),
        },
        "policy": {
            "display_regions": {"低音区": "MIDI <= 60", "高音区": "MIDI >= 61"},
            "shared_time_axis": "Both display bands use Score start_tick and duration_tick.",
            "source_metadata": "Source staff/voice is retained per occurrence; display region is presentation-only.",
            "duration_and_ties": "Each pitch occurrence gets its own duration line; ties are paired by source voice and pitch without chord merging.",
        },
        "summary": {
            "key": score.get("key"),
            "title": score.get("title"),
            "score_voice_count": len(score.get("voices", [])),
            "source_event_count": len(events),
            "note_occurrence_count": len(notes),
            "chord_event_count": sum(1 for item in events if len(item["pitches"]) > 1),
            "measure_count": len(spans(score)),
            "total_ticks": score.get("total_ticks"),
            "rendered_page_count": len(page_infos),
        },
        "equivalence": {
            "note_occurrences_equal": source_note_records == rendered_note_records,
            "source_event_keys_equal": source_keys == rendered_keys,
            "missing_source_event_keys": sorted(set(source_keys) - set(rendered_keys)),
            "unexpected_svg_event_keys": sorted(set(rendered_keys) - set(source_keys)),
            "mid_sha256_equal": digest(mid_path) == digest(copied_mid),
        },
        "source_events": events,
        "note_occurrences": notes,
        "pages": page_infos,
    }


def write_readme(path: Path, data: dict[str, Any]) -> None:
    summary = data["summary"]
    source = data["source"]
    eq = data["equivalence"]
    path.write_text(
        f"""# Luv Letter layout-v1

此目录是冻结 Score 的确定性数字简谱排版样本。只改变 SVG 显示，不重跑模型、MuseScore 或 jianpu-ly，不修错音和节奏。两条显示带共用 Score tick 横轴；固定 MIDI 60 边界仅用于显示分区。

- 高音区：MIDI >= 61；低音区：MIDI <= 60。
- 同起音 pitch 在同一 x 垂直叠放；数字使用 {data['summary'].get('key', 'Score key')} 的简谱级数，撇号/逗号表示八度。
- 下划线和延音线按每个 pitch occurrence 独立绘制；红色曲线表示 tie。精确 start/duration/tie 在 layout-manifest.json 和 SVG 悬停说明中保留。
- 第一小节全休止以 0 显示，首休止没有从时间轴删除。原 staff/voice 仅作为来源元数据保留，不把显示区称为左右手。

Score voices={summary['score_voice_count']}，events（含 rest）={summary['source_event_count']}，pitch occurrences={summary['note_occurrence_count']}，chord events={summary['chord_event_count']}，分页={summary['rendered_page_count']}。

校验：note_occurrences_equal={eq['note_occurrences_equal']}，source_event_keys_equal={eq['source_event_keys_equal']}，mid_sha256_equal={eq['mid_sha256_equal']}。

原始 Score：{source['score_path']}
原始 MIDI：{source['mid_path']}
原始 MIDI SHA-256：{source['mid_sha256']}
样本日期：{source['sample_date']}
生成时源 commit：{source['source_commit'] or 'unavailable'}
""",
        encoding="utf-8",
    )


def write_index(path: Path, score: dict[str, Any], data: dict[str, Any]) -> None:
    cards = []
    for info in data["pages"]:
        name = f"luv-letter-layout-{info['page']:03d}.svg"
        cards.append(f'<article><h3>第 {info["page"]} 页 · 小节 {info["measure_start"]}–{info["measure_end"]}</h3><a href="{name}"><img loading="lazy" src="{name}" alt="小节 {info["measure_start"]} 到 {info["measure_end"]}"></a><p>{info["start_tick"]}–{info["end_tick"]} ticks</p></article>')
    path.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(score.get('title', 'Luv Letter'))} layout-v1</title><style>body{{margin:0;background:#eef3f6;color:#172c40;font-family:system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}.notice,article,.preview{{background:#fff;border-radius:9px;padding:14px;box-shadow:0 2px 10px #cfd9df}}.notice{{border-left:4px solid #2c739f}}.links a{{display:inline-block;margin:10px 10px 10px 0;padding:8px 12px;border-radius:6px;background:#1f5f8b;color:#fff;text-decoration:none}}.preview img{{width:100%;height:auto;border:1px solid #d8e2e8}}.pages{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:22px}}article img{{width:100%;height:auto;border:1px solid #d8e2e8}}article h3{{font-size:15px;margin:2px 0 8px}}article p{{font-size:12px;color:#587087}}</style></head><body><main><h1>{esc(score.get('title', 'Luv Letter'))} · 排版样本</h1><div class="notice"><b>本次审查范围：前 4 小节。</b>下方整曲分页/长图是实验输出，尚未逐页审查。此目录不重跑模型、MuseScore 或 jianpu-ly，不解决错音/节奏；两区共用 tick 横轴，按 MIDI 60 仅作高低音区显示；原 staff/voice 保留。</div><p class="links"><a href="preview-4-bars.svg">前 4 小节预览</a><a href="preview-8-bars.svg">前 8 小节预览</a><a href="luv-letter-layout.long.svg">整曲长图 SVG（实验）</a><a href="score.mid">原 score.mid</a><a href="layout-manifest.json">事件 manifest</a><a href="README.md">说明</a></p><section class="preview"><h2>已审查预览 · 前 4 小节</h2><img src="preview-4-bars.svg" alt="前 4 小节预览"></section><h2>整曲实验分页（未逐页审查）</h2><section class="pages">{''.join(cards)}</section></main></body></html>""",
        encoding="utf-8",
    )


def render(args: argparse.Namespace) -> Path:
    score_path = Path(args.score).resolve()
    mid_path = Path(args.mid).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    events, notes = collect(score)
    all_bars = spans(score)
    per_page = max(1, int(args.bars_per_page))
    page_ranges = [(start, min(len(all_bars), start + per_page)) for start in range(0, len(all_bars), per_page)]
    page_paths: list[Path] = []
    page_infos: list[dict[str, Any]] = []
    for page_number, (start, end) in enumerate(page_ranges, 1):
        root, info = page_svg(score, all_bars, notes, start, end, page_number, len(page_ranges))
        destination = output / f"luv-letter-layout-{page_number:03d}.svg"
        write_svg(destination, root)
        page_paths.append(destination)
        page_infos.append(info)
    if not page_paths:
        raise RuntimeError("Score has no timeline bars")
    root, _ = page_svg(score, all_bars, notes, 0, min(4, len(all_bars)), 1, 1, "前 4 小节预览", region_mode="staff")
    write_svg(output / "preview-4-bars.svg", root)
    root, _ = page_svg(score, all_bars, notes, 0, min(8, len(all_bars)), 1, 1, "前 8 小节预览", region_mode="staff")
    write_svg(output / "preview-8-bars.svg", root)
    compose_long(page_paths, output / "luv-letter-layout.long.svg")
    copied_mid = output / "score.mid"
    shutil.copyfile(mid_path, copied_mid)
    data = manifest(score_path, mid_path, copied_mid, score, events, notes, page_infos)
    data["long_svg"] = {"path": os.fspath((output / "luv-letter-layout.long.svg").resolve())}
    (output / "layout-manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(output / "README.md", data)
    write_index(output / "index.html", score, data)
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", required=True)
    parser.add_argument("--mid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bars-per-page", type=int, default=PAGE_BARS)
    return parser.parse_args()


if __name__ == "__main__":
    render(arguments())
