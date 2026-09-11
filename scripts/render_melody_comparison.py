"""Render an auditable old/new main-melody comparison from frozen recognition notes.

The script never runs recognition or MuseScore. It selects the full source note
set first, crops the selected notes to ``--seconds`` afterwards, writes MIDI
with source-second timing, and creates a display-only pitch-order SVG. The
SVG has no duration marks and is not a formal rhythm score.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.muscriptor_v2 import stable_track_id, write_unquantized_midi  # noqa: E402
from backend.v2_job_manager import V2JobService  # noqa: E402
from scripts.render_luv_letter_layout import jianpu_name  # noqa: E402


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
SVG_COLUMNS = 16
SVG_CELL_WIDTH = 92
SVG_ROW_HEIGHT = 96
SVG_MARGIN_X = 30
SVG_HEADER_HEIGHT = 132
SVG_FOOTER_HEIGHT = 54


def _tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _normalise_notes(recognition: Mapping[str, Any]) -> list[dict[str, Any]]:
    tracks = [item for item in recognition.get("tracks", []) if isinstance(item, Mapping)]
    by_identity: dict[tuple[str, bool], Mapping[str, Any]] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for track in tracks:
        group = str(track.get("instrument_group") or "unknown")
        is_drum = _as_bool(track.get("is_drum"))
        by_identity.setdefault((group, is_drum), track)
        if track.get("track_id") is not None:
            by_id[str(track["track_id"])] = track

    normalised: list[dict[str, Any]] = []
    for raw_index, raw in enumerate(recognition.get("notes", [])):
        if not isinstance(raw, Mapping):
            raise ValueError(f"notes[{raw_index}] 不是对象")
        group = str(raw.get("instrument_group") or "unknown")
        is_drum = _as_bool(raw.get("is_drum"))
        raw_track_id = raw.get("track_id")
        track = by_id.get(str(raw_track_id)) if raw_track_id is not None else None
        track = track or by_identity.get((group, is_drum))
        program = int(raw.get("program", track.get("program", 0) if track else 0))
        track_id = str(raw_track_id or (track.get("track_id") if track else "") or stable_track_id(group, program, is_drum))
        try:
            pitch = int(raw["pitch"])
            start_sec = float(raw["start_sec"])
            end_sec = float(raw["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"notes[{raw_index}] 缺少有效 pitch/start_sec/end_sec") from exc
        if not all(math.isfinite(value) for value in (start_sec, end_sec)) or end_sec <= start_sec:
            raise ValueError(f"notes[{raw_index}] 的时间范围无效")
        normalised.append(
            {
                **dict(raw),
                "track_id": track_id,
                "instrument_group": group,
                "program": program,
                "is_drum": is_drum,
                "pitch": pitch,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "source_note_index": raw_index,
            }
        )
    return normalised


def _selected_track_ids(
    recognition: Mapping[str, Any],
    notes: Sequence[Mapping[str, Any]],
    requested: Sequence[str] | None,
) -> set[str]:
    track_rows = [item for item in recognition.get("tracks", []) if isinstance(item, Mapping)]
    known = {
        str(item.get("track_id")): _as_bool(item.get("is_drum"))
        for item in track_rows
        if item.get("track_id") is not None
    }
    note_tracks = {
        str(note.get("track_id")): _as_bool(note.get("is_drum"))
        for note in notes
        if note.get("track_id") is not None
    }
    known.update(note_tracks)
    if requested:
        selected = {str(value).strip() for value in requested if str(value).strip()}
        unknown = sorted(selected - set(known))
        if unknown:
            raise ValueError("--track-id 未在 recognition tracks/notes 中找到：" + ", ".join(unknown))
        drums = sorted(track_id for track_id in selected if known.get(track_id, False))
        if drums:
            raise ValueError("--track-id 只能选择有音高轨道，鼓组已排除：" + ", ".join(drums))
        return selected
    return {track_id for track_id, is_drum in known.items() if not is_drum}


def _old_baseline(notes: Sequence[Mapping[str, Any]], selected_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce the pre-accuracy V2 selector from git 08bf188 exactly."""

    pitched = [
        note
        for note in notes
        if str(note.get("track_id")) in selected_ids and not _as_bool(note.get("is_drum"))
    ]
    by_start: dict[float, Mapping[str, Any]] = {}
    for note in pitched:
        onset = round(float(note["start_sec"]), 5)
        current = by_start.get(onset)
        if current is None or int(note["pitch"]) > int(current["pitch"]):
            by_start[onset] = note
    selected = sorted(by_start.values(), key=lambda item: (float(item["start_sec"]), int(item["pitch"])))
    return [dict(note) for note in selected], {
        "schema_version": "melody-comparison-old-baseline-v1",
        "selector_version": "git-08bf188-v2-main-melody-notes",
        "selected_track_ids": sorted(selected_ids),
        "source_note_count": len(pitched),
        "baseline_count": len(selected),
        "selected_note_count": len(selected),
        "policy": "highest pitch per round(start_sec, 5) onset; ties keep first source note",
        "selected_source_note_indices": [note.get("source_note_index") for note in selected],
    }


def _crop(notes: Sequence[Mapping[str, Any]], seconds: float) -> list[dict[str, Any]]:
    cropped: list[dict[str, Any]] = []
    for note in notes:
        start_sec = float(note["start_sec"])
        if start_sec >= seconds:
            continue
        source_end = float(note["end_sec"])
        end_sec = min(source_end, seconds)
        if end_sec <= start_sec:
            continue
        cropped.append(
            {
                **dict(note),
                "end_sec": end_sec,
                "cropped_at_limit": end_sec != source_end,
            }
        )
    return cropped


def _event_records(notes: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, note in enumerate(notes, 1):
        start_sec = float(note["start_sec"])
        end_sec = float(note["end_sec"])
        records.append(
            {
                "sequence_index": index,
                "pitch": int(note["pitch"]),
                "jianpu": jianpu_name(int(note["pitch"]), key),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "cropped_at_limit": bool(note.get("cropped_at_limit")),
                "source_note_index": note.get("source_note_index"),
                "track_id": str(note.get("track_id")),
                "instrument_group": str(note.get("instrument_group", "unknown")),
                "program": int(note.get("program", 0)),
                "is_drum": _as_bool(note.get("is_drum")),
            }
        )
    return records


def _parse_jianpu(value: str) -> tuple[str, str, int, int]:
    accidental = value[0] if value.startswith(("#", "b")) else ""
    body = value[len(accidental):]
    digits = body.rstrip("',")
    marks = body[len(digits):]
    return accidental, digits, marks.count("'"), marks.count(",")


def _svg_text(parent: ET.Element, x: float, y: float, value: str, **attrs: Any) -> ET.Element:
    element = ET.SubElement(
        parent,
        _tag("text"),
        {"x": f"{x:.2f}", "y": f"{y:.2f}", **{key: str(item) for key, item in attrs.items()}},
    )
    element.text = value
    return element


def _render_guide(path: Path, label: str, records: Sequence[Mapping[str, Any]], key: str, seconds: float) -> None:
    rows = max(1, math.ceil(len(records) / SVG_COLUMNS))
    width = SVG_MARGIN_X * 2 + SVG_COLUMNS * SVG_CELL_WIDTH
    height = SVG_HEADER_HEIGHT + rows * SVG_ROW_HEIGHT + SVG_FOOTER_HEIGHT
    root = ET.Element(
        _tag("svg"),
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "role": "img",
            "aria-labelledby": "title subtitle",
        },
    )
    defs = ET.SubElement(root, _tag("defs"))
    style = ET.SubElement(defs, _tag("style"))
    style.text = (
        ".title{font:700 25px Arial,'Microsoft YaHei',sans-serif;fill:#17324b}"
        ".subtitle{font:14px Arial,'Microsoft YaHei',sans-serif;fill:#4b6578}"
        ".legend{font:12px Arial,'Microsoft YaHei',sans-serif;fill:#5b7484}"
        ".index{font:10px Consolas,monospace;fill:#79909d}"
        ".note{font:700 28px Arial,'Microsoft YaHei',sans-serif;fill:#172d43}"
        ".accidental{font:700 18px Arial,sans-serif;fill:#b35a48}"
        ".octave{fill:#b35a48}.cell{fill:#fffdf8;stroke:#d7e0e3;stroke-width:1}"
        ".empty{font:16px Arial,'Microsoft YaHei',sans-serif;fill:#81949f}"
    )
    ET.SubElement(root, _tag("rect"), {"x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "#eef3f5"})
    _svg_text(root, SVG_MARGIN_X, 38, f"{label}主旋律音高顺序谱", id="title", **{"class": "title"})
    _svg_text(
        root,
        SVG_MARGIN_X,
        66,
        f"1={key} · 八度用圆点 · 不标时值 · 非正式节奏简谱 · 前 {seconds:g} 秒",
        id="subtitle",
        **{"class": "subtitle"},
    )
    _svg_text(root, SVG_MARGIN_X, 91, "每行约 16 音；顺序只用于比较候选选择，不代表正式谱面。", **{"class": "legend"})
    _svg_text(root, width - SVG_MARGIN_X, 91, f"音符数 {len(records)}", **{"class": "legend", "text-anchor": "end"})
    if not records:
        _svg_text(root, SVG_MARGIN_X, SVG_HEADER_HEIGHT + 55, "当前选择没有音符", **{"class": "empty"})
    for index, record in enumerate(records):
        row, column = divmod(index, SVG_COLUMNS)
        left = SVG_MARGIN_X + column * SVG_CELL_WIDTH
        top = SVG_HEADER_HEIGHT + row * SVG_ROW_HEIGHT
        center_x = left + SVG_CELL_WIDTH / 2
        baseline = top + 63
        group = ET.SubElement(
            root,
            _tag("g"),
            {
                "data-sequence-index": str(record["sequence_index"]),
                "data-midi": str(record["pitch"]),
                "data-jianpu": str(record["jianpu"]),
            },
        )
        ET.SubElement(group, _tag("rect"), {"x": str(left + 2), "y": str(top + 2), "width": str(SVG_CELL_WIDTH - 4), "height": str(SVG_ROW_HEIGHT - 4), "rx": "7", "class": "cell"})
        _svg_text(group, left + 9, top + 18, str(record["sequence_index"]), **{"class": "index"})
        accidental, digits, above, below = _parse_jianpu(str(record["jianpu"]))
        if accidental:
            _svg_text(group, center_x - 22, baseline, accidental, **{"class": "accidental", "text-anchor": "middle"})
        _svg_text(group, center_x, baseline, digits, **{"class": "note", "text-anchor": "middle"})
        for mark_index in range(above):
            ET.SubElement(group, _tag("circle"), {"cx": f"{center_x:.2f}", "cy": f"{baseline - 31 - mark_index * 7:.2f}", "r": "3", "class": "octave", "data-octave": "up"})
        for mark_index in range(below):
            ET.SubElement(group, _tag("circle"), {"cx": f"{center_x:.2f}", "cy": f"{baseline + 14 + mark_index * 7:.2f}", "r": "3", "class": "octave", "data-octave": "down"})
        title = ET.SubElement(group, _tag("title"))
        title.text = f"序号 {record['sequence_index']} · MIDI {record['pitch']} · {record['jianpu']}"
    _svg_text(root, SVG_MARGIN_X, height - 20, "显示层只表达音高顺序；精确起止秒数保存在对应 events.json。", **{"class": "legend"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _module_root(path: Path) -> Path | None:
    candidate = path.resolve()
    for item in [candidate, *candidate.parents]:
        package = item / "package.json" if item.is_dir() else item
        if package.name != "package.json":
            continue
        try:
            data = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("name") == "sharp":
            return package.parent
    return None


def _find_sharp_module(explicit: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for name in ("SHARP_MODULE", "CODEX_SHARP_MODULE"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    candidates.extend((ROOT / "node_modules" / "sharp", ROOT / "frontend" / "node_modules" / "sharp"))
    cache_root = Path.home() / ".cache" / "codex-runtimes"
    if cache_root.is_dir():
        candidates.extend(cache_root.glob("**/node_modules/sharp"))
    for candidate in candidates:
        root = _module_root(candidate)
        if root:
            return root
    node = shutil.which("node")
    if node:
        for cwd in (ROOT, ROOT / "frontend"):
            if not cwd.is_dir():
                continue
            result = subprocess.run(
                [node, "-e", "try { console.log(require.resolve('sharp')) } catch (_) {}"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            resolved = result.stdout.strip()
            if resolved:
                root = _module_root(Path(resolved))
                if root:
                    return root
    return None


def _rasterize_with_sharp(svg_path: Path, png_path: Path, explicit_module: str | None) -> dict[str, Any]:
    node = shutil.which("node")
    module = _find_sharp_module(explicit_module)
    if not node or module is None:
        return {"status": "skipped", "reason": "本机没有可定位的 Node.js sharp 模块"}
    package = json.loads((module / "package.json").read_text(encoding="utf-8"))
    entry = (module / str(package.get("main") or "dist/index.cjs")).resolve()
    js = (
        "const sharp=require(process.argv[1]);"
        "sharp(process.argv[2]).png().toFile(process.argv[3])"
        ".then(info=>process.stdout.write(JSON.stringify(info)))"
        ".catch(error=>{console.error(error.stack||error);process.exit(1)});"
    )
    result = subprocess.run(
        [node, "-e", js, os.fspath(entry), os.fspath(svg_path), os.fspath(png_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"status": "failed", "module": os.fspath(module), "error": result.stderr.strip()[-2000:]}
    info: dict[str, Any] = {}
    try:
        info = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        info = {"sharp_output": result.stdout.strip()}
    return {"status": "rendered", "module": os.fspath(module), "png": os.fspath(png_path.resolve()), **info}


def _write_index(path: Path, audit: Mapping[str, Any], names: Mapping[str, str]) -> None:
    source = audit["source"]
    old = audit["variants"]["old"]
    new = audit["variants"]["new"]
    embedded = html.escape(
        json.dumps(
            {
                "source_sha256": source["sha256"],
                "source_commit": audit.get("source_commit"),
                "selected_track_ids": audit["selected_track_ids"],
                "seconds": audit["seconds"],
                "key": audit["key"],
                "variants": {
                    "old": {
                        "selector_version": old["selector"].get("selector_version"),
                        "full_selected_note_count": old["full_selected_note_count"],
                        "cropped_note_count": old["cropped_note_count"],
                    },
                    "new": {
                        "selector_version": new["selector"].get("selector_version"),
                        "full_selected_note_count": new["full_selected_note_count"],
                        "cropped_note_count": new["cropped_note_count"],
                        "skipped_count": len(new["selector"].get("skipped", [])),
                    },
                },
                "rasterization": audit.get("rasterization", {}),
                "audit_file": "comparison.audit.json",
            },
            ensure_ascii=False,
            indent=2,
        ),
        quote=False,
    )
    raster_rows = "".join(
        f"<li>{html.escape(label)}：{html.escape(str(info.get('status')))}"
        f"{(' · ' + html.escape(str(info.get('png')))) if info.get('png') else ''}</li>"
        for label, info in audit.get("rasterization", {}).items()
    )
    audio_dir = path.parent.parent / "melody-baseline-audit"
    audio_links = []
    for filename, label in (
        ("baseline-v2-first45.mp3", "旧规则试听"),
        ("candidate-first45.mp3", "候选试听"),
        ("original-first45.mp3", "原曲试听"),
    ):
        if (audio_dir / filename).is_file():
            href = f"../melody-baseline-audit/{filename}"
            audio_links.append(f'<li>{html.escape(label)}：<a href="{href}">{filename}</a><br><audio controls preload="none" src="{href}"></audio></li>')
    audio_section = f"<section class=\"card audio\"><h2>已有试听</h2><ul>{''.join(audio_links)}</ul></section>" if audio_links else ""
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>主旋律候选比较 · 前 {audit['seconds']:g} 秒</title>
<style>body{{margin:0;background:#eef3f5;color:#17324b;font-family:system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}.notice,.card,.audit{{background:#fff;border-radius:10px;padding:18px;box-shadow:0 2px 12px #cbd7dc;margin:0 0 20px}}.notice{{border-left:5px solid #26739a}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}}.card h2{{margin:0 0 12px;font-size:18px}}.card img{{display:block;width:100%;height:auto;border:1px solid #d7e0e3;background:#fffdf8}}.links a{{display:inline-block;margin:8px 8px 0 0;padding:7px 11px;border-radius:5px;background:#26739a;color:#fff;text-decoration:none}}.audit{{font-size:13px;line-height:1.7;color:#516b7b}}.audit summary{{cursor:pointer;color:#26739a;font-weight:700}}.audit pre{{max-height:420px;overflow:auto;font:11px/1.45 Consolas,monospace;white-space:pre-wrap}}</style></head>
<body><main><h1>主旋律候选比较 · 前 {audit['seconds']:g} 秒</h1>
<section class="notice"><strong>审查说明：</strong>两份 MIDI 都先在全曲候选音符上选择，再裁到前 {audit['seconds']:g} 秒。SVG 仅显示音高顺序、八度圆点和 1={html.escape(str(audit['key']))}，不标时值，也不是正式节奏简谱。</section>
{audio_section}<div class="grid"><article class="card"><h2>新候选 · 主旋律音高顺序谱</h2><img src="{html.escape(names['new_svg'])}" alt="新候选主旋律音高顺序谱"><p>前 {new['cropped_note_count']} 音 · 全曲选择保留 {new['full_selected_note_count']} 音</p><p class="links"><a href="{html.escape(names['new_svg'])}">SVG</a><a href="{html.escape(names['new_png'])}">PNG</a><a href="{html.escape(names['new_mid'])}">MIDI</a><a href="{html.escape(names['new_events'])}">events JSON</a></p></article>
<article class="card"><h2>旧规则 · 主旋律音高顺序谱</h2><img src="{html.escape(names['old_svg'])}" alt="旧规则主旋律音高顺序谱"><p>前 {old['cropped_note_count']} 音 · 全曲选择保留 {old['full_selected_note_count']} 音</p><p class="links"><a href="{html.escape(names['old_svg'])}">SVG</a><a href="{html.escape(names['old_png'])}">PNG</a><a href="{html.escape(names['old_mid'])}">MIDI</a><a href="{html.escape(names['old_events'])}">events JSON</a></p></article></div>
<section class="audit"><h2>审查记录</h2><p><a href="comparison.audit.json">下载完整 comparison.audit.json</a></p><details><summary>显示审计摘要</summary><p>recognition SHA-256：<code>{html.escape(str(source['sha256']))}</code></p><p>track：<code>{html.escape(', '.join(audit['selected_track_ids']))}</code> · 鼓组排除 · 脚本 SHA-256：<code>{html.escape(str(audit['script_sha256']))}</code></p><p>生成 commit：<code>{html.escape(str(audit.get('source_commit') or 'unavailable'))}</code></p><ul>{raster_rows}</ul><pre>{embedded}</pre></details></section></main></body></html>
""",
        encoding="utf-8",
    )


def render(args: argparse.Namespace) -> Path:
    recognition_path = Path(args.recognition).resolve()
    output = Path(args.output).resolve()
    seconds = float(args.seconds)
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("--seconds 必须是正数")
    bpm = float(args.bpm)
    if bpm <= 0 or not math.isfinite(bpm):
        raise ValueError("--bpm 必须是正数")
    recognition = json.loads(recognition_path.read_text(encoding="utf-8"))
    if not isinstance(recognition, Mapping):
        raise ValueError("recognition JSON 顶层必须是对象")
    notes = _normalise_notes(recognition)
    selected_ids = _selected_track_ids(recognition, notes, args.track_id)
    if not selected_ids:
        raise ValueError("没有可选择的有音高轨道")
    old_full, old_audit = _old_baseline(notes, selected_ids)
    new_full, new_audit = V2JobService._select_main_melody_notes(notes, selected_ids)
    old_crop = _crop(old_full, seconds)
    new_crop = _crop(new_full, seconds)
    old_records = _event_records(old_crop, args.key)
    new_records = _event_records(new_crop, args.key)
    output.mkdir(parents=True, exist_ok=True)
    tag = str(int(seconds)) if seconds.is_integer() else str(seconds).replace(".", "p")
    names = {
        "old_mid": f"old{tag}.mid",
        "new_mid": f"new{tag}.mid",
        "old_events": f"old{tag}.events.json",
        "new_events": f"new{tag}.events.json",
        "old_svg": f"old{tag}.svg",
        "new_svg": f"new{tag}.svg",
        "old_png": f"old{tag}.png",
        "new_png": f"new{tag}.png",
    }
    for crop, name in ((old_crop, names["old_mid"]), (new_crop, names["new_mid"])):
        write_unquantized_midi(crop, output / name, title=f"Melody comparison {name}", bpm=bpm)
    _render_guide(output / names["old_svg"], "旧规则", old_records, args.key, seconds)
    _render_guide(output / names["new_svg"], "新候选", new_records, args.key, seconds)
    rasterization = {
        "old": _rasterize_with_sharp(output / names["old_svg"], output / names["old_png"], args.sharp_module),
        "new": _rasterize_with_sharp(output / names["new_svg"], output / names["new_png"], args.sharp_module),
    }
    source = {
        "path": os.fspath(recognition_path),
        "sha256": _sha256(recognition_path),
        "schema_version": recognition.get("schema_version"),
        "engine": recognition.get("engine"),
        "source_kind": recognition.get("source_kind"),
        "note_count": len(notes),
        "tracks": recognition.get("tracks", []),
    }
    variant_data = {
        "old": {
            "selector": old_audit,
            "full_selected_note_count": len(old_full),
            "cropped_note_count": len(old_crop),
            "events_path": names["old_events"],
            "midi_path": names["old_mid"],
            "svg_path": names["old_svg"],
            "png_path": names["old_png"],
            "notes": old_records,
        },
        "new": {
            "selector": new_audit,
            "full_selected_note_count": len(new_full),
            "cropped_note_count": len(new_crop),
            "events_path": names["new_events"],
            "midi_path": names["new_mid"],
            "svg_path": names["new_svg"],
            "png_path": names["new_png"],
            "notes": new_records,
        },
    }
    common = {
        "schema_version": "melody-comparison-v1",
        "source": source,
        "source_commit": _source_commit(),
        "script_path": os.fspath(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "selected_track_ids": sorted(selected_ids),
        "seconds": seconds,
        "key": args.key,
        "bpm": bpm,
        "crop_policy": "full selection first; retain selected notes with start_sec < seconds and clip end_sec to seconds",
        "midi_policy": "write_unquantized_midi using source seconds; fixed playback velocity only",
        "display_policy": "pitch order only; no duration marks; informal jianpu guide; octave circles; 16 columns",
    }
    for variant, data in variant_data.items():
        _json_write(
            output / (names[f"{variant}_events"]),
            {
                **common,
                "variant": variant,
                "selector": data["selector"],
                "full_selected_note_count": data["full_selected_note_count"],
                "cropped_note_count": data["cropped_note_count"],
                "notes": data["notes"],
            },
        )
    audit = {
        **common,
        "variants": variant_data,
        "rasterization": rasterization,
        "outputs": names,
    }
    _json_write(output / "comparison.audit.json", audit)
    _write_index(output / "index.html", audit, names)
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition", required=True, help="Frozen recognition.json")
    parser.add_argument("--output", required=True, help="Comparison output directory")
    parser.add_argument("--seconds", type=float, default=45.0, help="Crop horizon after full selection (default: 45)")
    parser.add_argument("--key", default="C", help="Display key used by jianpu_name (default: C)")
    parser.add_argument("--track-id", action="append", help="Pitched track id; repeat for multiple tracks; default: all non-drum tracks")
    parser.add_argument("--bpm", type=float, default=120.0, help="Playback tempo used to encode source seconds in MIDI")
    parser.add_argument("--sharp-module", help="Optional path to a local sharp package for SVG -> PNG confirmation")
    return parser.parse_args()


if __name__ == "__main__":
    render(arguments())
