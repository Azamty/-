"""Attach one full-track BeatNet decode to the five CCMusic windows.

The CCMusic recognizer raw already contains the real GAME output for each
12-second window.  This script deliberately does not run GAME or Demucs
again.  It runs BeatNet once on the complete aligned Yueding mix, then maps
the resulting absolute beat times into each window.  The full-track audio
and beat-grid hashes, the absolute/local time map, and one beat on either side
of every window are retained for audit.  Boundary beats are kept in context
metadata; the window's ``beats``/``downbeats`` arrays contain only beats in
the scored local interval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_MANIFEST = ROOT / ".cache" / "high-accuracy-benchmarks" / "ccmusic-yueding" / "selection_manifest.json"
DEFAULT_RAW_ROOT = ROOT / ".artifacts" / "review" / "ccmusic-production-v2"
DEFAULT_OUTPUT_ROOT = ROOT / ".artifacts" / "review" / "ccmusic-production-context-v1"
CONTEXT_SCHEMA_VERSION = "1.0"
BEATNET_VERSION = "1.1.3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _audio_record(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    return {
        "path": os.fspath(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": float(info.frames / info.samplerate),
    }


def _full_grid_record(payload: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    beats = payload.get("beats")
    if not isinstance(beats, list) or len(beats) < 2:
        raise ValueError("full-track BeatNet grid must contain at least two beats")
    return {
        "path": os.fspath(path),
        "sha256": _sha256(path),
        "beat_count": len(beats),
        "downbeat_count": sum(bool(item.get("downbeat")) for item in beats if isinstance(item, Mapping)),
        "beatnet_version": payload.get("beatnet", {}).get("version", BEATNET_VERSION)
        if isinstance(payload.get("beatnet"), Mapping)
        else BEATNET_VERSION,
        "engine": payload.get("engine", "beatnet"),
        "mode": payload.get("mode", "offline"),
        "inference": payload.get("inference", "DBN"),
    }


def _beat_time(record: Mapping[str, Any]) -> float:
    try:
        value = float(record["time_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"BeatNet beat has no finite time_sec: {record!r}") from exc
    if not value == value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"BeatNet beat has non-finite time_sec: {record!r}")
    return value


def _context_beat(
    record: Mapping[str, Any],
    *,
    full_index: int,
    local_time: float,
    in_window: bool,
    local_index: int | None = None,
    local_bar_index: int | None = None,
    local_beat_number: int | None = None,
    local_quarter_position: float | None = None,
) -> dict[str, Any]:
    """Return a beat with local coordinates and explicit full-track audit data."""

    value = dict(record)
    # The old fields silently exposed full-track coordinates after cropping.
    # Preserve them under names that cannot be mistaken for window values.
    for field in ("index", "bar_index", "beat_number", "quarter_position"):
        if field in value:
            value[f"full_track_{field}"] = value[field]
            value.pop(field, None)
    if "time_sec" in value:
        value["full_track_time_sec"] = value["time_sec"]
        value.pop("time_sec", None)
    if "absolute_time_sec" in value:
        value["full_track_absolute_time_sec"] = value["absolute_time_sec"]
        value.pop("absolute_time_sec", None)
    value["time_sec"] = round(float(local_time), 9)
    value["absolute_time_sec"] = round(_beat_time(record), 9)
    value["full_track_time_sec"] = round(_beat_time(record), 9)
    value["full_track_index"] = int(full_index)
    value["index"] = int(local_index) if local_index is not None else None
    value["bar_index"] = int(local_bar_index) if local_bar_index is not None else None
    value["beat_number"] = int(local_beat_number) if local_beat_number is not None else None
    value["quarter_position"] = (
        round(float(local_quarter_position), 9)
        if local_quarter_position is not None
        else None
    )
    value["in_window"] = bool(in_window)
    value["include_in_evaluation"] = bool(in_window)
    value["context_only"] = not bool(in_window)
    return value


def _local_bars(
    full_grid: Mapping[str, Any],
    inside: Sequence[tuple[int, Mapping[str, Any]]],
    local_records: Sequence[Mapping[str, Any]],
    *,
    absolute_start_sec: float,
    absolute_end_sec: float,
    quarter_origin: float,
    window_end_quarter: float,
) -> list[dict[str, Any]]:
    """Filter and rebase full-track bars to the local beat index space."""

    full_bars = full_grid.get("bars")
    if not isinstance(full_bars, list):
        return []
    full_to_local = {int(full_index): local_index for local_index, (full_index, _item) in enumerate(inside)}
    local_bars: list[dict[str, Any]] = []
    for full_bar in full_bars:
        if not isinstance(full_bar, Mapping):
            continue
        try:
            full_start = int(full_bar["start_beat_index"])
            full_end = int(full_bar["end_beat_index"])
        except (KeyError, TypeError, ValueError):
            continue
        overlap = [
            local_index
            for full_index, local_index in full_to_local.items()
            if full_start <= full_index < full_end
        ]
        if not overlap:
            continue
        local_start = min(overlap)
        local_end = max(overlap) + 1
        item = dict(full_bar)
        for field in (
            "index",
            "bar_index",
            "start_beat_index",
            "end_beat_index",
            "start_sec",
            "end_sec",
            "start_quarter",
            "end_quarter",
            "beat_count",
            "duration_quarters",
        ):
            if field in item:
                item[f"full_track_{field}"] = item[field]
                item.pop(field, None)
        item["index"] = len(local_bars)
        item["start_beat_index"] = local_start
        item["end_beat_index"] = local_end
        item["beat_count"] = local_end - local_start
        full_start_sec = float(item.get("full_track_start_sec", _beat_time(inside[local_start][1])))
        full_end_sec = float(item.get("full_track_end_sec", _beat_time(inside[local_end - 1][1])))
        local_start_sec = max(0.0, min(absolute_end_sec - absolute_start_sec, full_start_sec - absolute_start_sec))
        local_end_sec = max(0.0, min(absolute_end_sec - absolute_start_sec, full_end_sec - absolute_start_sec))
        # A beat exactly on the closed window end belongs to the beat metric,
        # but its following bar has no duration inside this window.
        if local_end_sec <= local_start_sec:
            continue
        has_quarter_bounds = (
            "full_track_start_quarter" in item
            and "full_track_end_quarter" in item
        )
        if has_quarter_bounds:
            full_start_quarter = float(item["full_track_start_quarter"])
            full_end_quarter = float(item["full_track_end_quarter"])
            local_start_quarter = max(0.0, min(window_end_quarter, full_start_quarter - quarter_origin))
            local_end_quarter = max(0.0, min(window_end_quarter, full_end_quarter - quarter_origin))
            clipped_at_start = full_start_sec < absolute_start_sec or full_start_quarter < quarter_origin
            clipped_at_end = full_end_sec > absolute_end_sec or full_end_quarter - quarter_origin > window_end_quarter
        else:
            # Without both source endpoints, omit local quarter bounds rather
            # than treating the last visible beat as the bar endpoint.
            clipped_at_start = full_start_sec < absolute_start_sec
            clipped_at_end = full_end_sec > absolute_end_sec
            local_start_quarter = local_end_quarter = None
        partial = bool(
            clipped_at_start
            or clipped_at_end
            or (local_start == 0 and not bool(local_records[0].get("downbeat")))
        )
        item["start_sec"] = round(local_start_sec, 9)
        item["end_sec"] = round(local_end_sec, 9)
        # Local quarter zero is anchored at the first visible beat, not at an
        # arbitrary crop boundary.  A partial bar can therefore extend before
        # zero or beyond its last visible pulse.  Preserve the source bounds as
        # audit evidence, but omit local quarter bounds instead of clamping or
        # inventing a duration for a truncated bar.
        if has_quarter_bounds and not partial:
            item["start_quarter"] = round(float(local_start_quarter), 9)
            item["end_quarter"] = round(float(local_end_quarter), 9)
            item["duration_quarters"] = round(
                float(local_end_quarter) - float(local_start_quarter), 9
            )
        item["partial_window"] = partial
        if partial:
            item["stable"] = False
        local_bars.append(item)
    return local_bars


def crop_full_track_beat_grid(
    full_grid: Mapping[str, Any],
    *,
    absolute_start_sec: float,
    duration_sec: float,
    full_audio: Mapping[str, Any],
    full_grid_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one auditable local window view of a full-track grid.

    ``beats`` and ``downbeats`` are the scored local window.  The nearest
    preceding and following beats are retained under ``context.boundary_beats``
    so quantizer/debug tooling can see the interpolation context without
    counting those beats as false positives in window metrics.
    """

    start = float(absolute_start_sec)
    duration = float(duration_sec)
    end = start + duration
    if start < 0 or duration <= 0:
        raise ValueError("window start must be non-negative and duration must be positive")
    records = full_grid.get("beats")
    if not isinstance(records, list):
        raise ValueError("full-track BeatNet grid has no beats list")
    ordered = sorted(
        ((index, item) for index, item in enumerate(records) if isinstance(item, Mapping)),
        key=lambda pair: _beat_time(pair[1]),
    )
    if len(ordered) < 2:
        raise ValueError("full-track BeatNet grid has fewer than two valid beats")
    before = [(index, item) for index, item in ordered if _beat_time(item) < start]
    after = [(index, item) for index, item in ordered if _beat_time(item) > end]
    inside = [(index, item) for index, item in ordered if start <= _beat_time(item) <= end]
    if not before or not after or len(inside) < 2:
        raise ValueError(
            "BeatNet full-track grid cannot provide a complete window context: "
            f"before={len(before)}, inside={len(inside)}, after={len(after)}, start={start}, end={end}"
        )
    before_index, before_record = before[-1]
    after_index, after_record = after[0]
    selected = [(before_index, before_record), *inside, (after_index, after_record)]

    # Rebase all derived beat coordinates to the scored window.  The full
    # record index/time remain available on each record for audit only.
    full_quarters: list[float] = []
    for full_index, item in inside:
        try:
            quarter = float(item["quarter_position"])
        except (KeyError, TypeError, ValueError):
            quarter = float(full_index)
        if quarter != quarter or quarter in {float("inf"), float("-inf")}:
            quarter = float(full_index)
        full_quarters.append(quarter)
    quarter_origin = full_quarters[0]
    downbeat_indices = [
        index for index, (_full_index, item) in enumerate(inside) if bool(item.get("downbeat"))
    ]
    bar_starts = [0, *[index for index in downbeat_indices if index > 0]]
    bar_starts = sorted(set(bar_starts))
    local_inside: list[dict[str, Any]] = []
    for local_index, ((full_index, item), quarter) in enumerate(zip(inside, full_quarters)):
        local_bar_index = max(
            (bar for bar, bar_start in enumerate(bar_starts) if bar_start <= local_index),
            default=0,
        )
        local_inside.append(
            _context_beat(
                item,
                full_index=full_index,
                local_time=_beat_time(item) - start,
                in_window=True,
                local_index=local_index,
                local_bar_index=local_bar_index,
                # beat_number is the musical beat within the source bar.  A
                # cropped partial bar must not be renumbered from one.
                local_beat_number=(
                    int(item["beat_number"])
                    if item.get("beat_number") is not None
                    else None
                ),
                local_quarter_position=quarter - quarter_origin,
            )
        )
    local_inside_downbeats = [item for item in local_inside if bool(item.get("downbeat"))]

    window_end_quarter = local_inside[-1]["quarter_position"]
    if after_record is not None and _beat_time(after_record) > _beat_time(inside[-1][1]):
        try:
            after_quarter = float(after_record["quarter_position"])
        except (KeyError, TypeError, ValueError):
            after_quarter = float(after_index)
        if after_quarter != after_quarter or after_quarter in {float("inf"), float("-inf")}:
            after_quarter = float(after_index)
        interval = _beat_time(after_record) - _beat_time(inside[-1][1])
        if interval > 0:
            window_end_quarter = round(
                (
                    full_quarters[-1]
                    + (end - _beat_time(inside[-1][1]))
                    / interval
                    * (after_quarter - full_quarters[-1])
                )
                - quarter_origin,
                9,
            )
    window_end_quarter = max(float(local_inside[-1]["quarter_position"]), float(window_end_quarter))

    full_mapping = full_grid.get("mapping") if isinstance(full_grid.get("mapping"), Mapping) else {}
    mapping = copy.deepcopy(dict(full_mapping))
    if isinstance(full_mapping.get("beat_times"), list):
        mapping["full_track_beat_times"] = list(full_mapping["beat_times"])
    if "first_beat_sec" in full_mapping:
        mapping["full_track_first_beat_sec"] = full_mapping["first_beat_sec"]
    if isinstance(full_mapping.get("score_origin"), Mapping):
        mapping["full_track_score_origin"] = copy.deepcopy(dict(full_mapping["score_origin"]))
    mapping.pop("score_origin", None)
    mapping["beat_times"] = [float(item["time_sec"]) for item in local_inside]
    mapping["first_beat_sec"] = float(local_inside[0]["time_sec"])
    mapping["window_absolute_start_sec"] = start
    mapping["window_absolute_end_sec"] = end
    mapping["absolute_to_local"] = "local_sec = absolute_sec - window.absolute_start_sec"
    first_downbeat_index = downbeat_indices[0] if downbeat_indices else None
    first_is_downbeat = bool(local_inside and local_inside[0].get("downbeat"))
    full_downbeat_index = int(inside[first_downbeat_index][0]) if first_downbeat_index is not None else None
    local_origin = {
        "strategy": (
            "local_first_downbeat"
            if first_is_downbeat
            else "local_window_partial_bar"
            if first_downbeat_index is not None
            else "local_downbeat_unavailable"
        ),
        "timeline_scope": "ccmusic_local_window",
        "downbeat_status": "aligned" if first_is_downbeat else "undetermined",
        "downbeat_index": first_downbeat_index,
        "downbeat_sec": (
            float(local_inside[first_downbeat_index]["time_sec"])
            if first_downbeat_index is not None
            else None
        ),
        "downbeat_score_beat": 0.0 if first_is_downbeat else None,
        "pickup_candidate": False,
        "pickup_beats": 0.0,
        "origin_shift_beats": 0.0,
        "warning": (
            None
            if first_is_downbeat
            else "裁剪窗口从不完整小节开始；局部重拍阶段未确定，保持局部原点"
        ),
        "full_track_downbeat_index": full_downbeat_index,
        "absolute_downbeat_sec": (
            round(_beat_time(inside[first_downbeat_index][1]), 9)
            if first_downbeat_index is not None
            else None
        ),
        "origin_basis": "inside_window_downbeats_only",
    }
    mapping["score_origin"] = local_origin
    mapping["timeline_scope"] = "ccmusic_local_window"
    mapping["boundary_beats_excluded_from_evaluation"] = True
    grid = copy.deepcopy(dict(full_grid))
    grid["duration_sec"] = duration
    grid["beats"] = local_inside
    grid["downbeats"] = local_inside_downbeats
    grid["bars"] = _local_bars(
        full_grid,
        inside,
        local_inside,
        absolute_start_sec=start,
        absolute_end_sec=end,
        quarter_origin=quarter_origin,
        window_end_quarter=window_end_quarter,
    )
    grid["mapping"] = mapping
    if local_origin["warning"]:
        warnings = [str(item) for item in (grid.get("warnings") or []) if item]
        warnings.append(str(local_origin["warning"]))
        grid["warnings"] = list(dict.fromkeys(warnings))
    grid["context"] = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "mode": "full_track_absolute_window",
        "full_track_audio": dict(full_audio),
        "full_track_beat_grid": dict(full_grid_record),
        "full_track_beat_count": len(records),
        "full_track_downbeat_count": sum(bool(item.get("downbeat")) for item in records if isinstance(item, Mapping)),
        "window": {
            "absolute_start_sec": start,
            "absolute_end_sec": end,
            "duration_sec": duration,
            "local_start_sec": 0.0,
            "local_end_sec": duration,
            "absolute_to_local_offset_sec": -start,
            "absolute_to_local": "local_sec = absolute_sec - absolute_start_sec",
            "inside_full_track_indices": [int(index) for index, _item in inside],
            "selected_full_track_indices": [int(index) for index, _item in selected],
            "boundary_beats": {
                "before": [
                    _context_beat(
                        before_record,
                        full_index=before_index,
                        local_time=_beat_time(before_record) - start,
                        in_window=False,
                        local_index=-1,
                    )
                ],
                "after": [
                    _context_beat(
                        after_record,
                        full_index=after_index,
                        local_time=_beat_time(after_record) - start,
                        in_window=False,
                        local_index=len(local_inside),
                    )
                ],
            },
            "boundary_policy": "nearest one beat before and after; retained for audit/context and excluded from window beat metrics",
            "boundary_beats_include_in_evaluation": False,
            "coordinate_scope": "window_local",
        },
        "evaluation": {
            "beat_records": "beat_grid.beats (window-local; boundary beats are metadata only)",
            "downbeat_records": "beat_grid.downbeats (window-local; boundary beats are metadata only)",
            "boundary_beats_excluded_from_metrics": True,
            "reference_grid_not_used": True,
        },
    }
    # Keep the explicit local arrays easy to inspect while retaining boundary
    # records in the context object.  The evaluator reads these arrays.
    grid["beats"] = local_inside
    grid["downbeats"] = local_inside_downbeats
    return grid


def _case_index(selection: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = selection.get("cases")
    if not isinstance(cases, list):
        raise ValueError("CCMusic selection manifest has no cases list")
    result: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str):
            raise ValueError("CCMusic selection manifest has an invalid case")
        result[str(case["case_id"])] = case
    return result


def _context_raw(
    raw: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    beat_grid: Mapping[str, Any],
    full_audio: Mapping[str, Any],
    full_grid_record: Mapping[str, Any],
    source_raw_path: Path,
) -> dict[str, Any]:
    if raw.get("model_output") is not True:
        raise ValueError(f"{case.get('case_id')} raw is not marked model_output=true")
    if not isinstance(raw.get("notes"), list):
        raise ValueError(f"{case.get('case_id')} raw has no notes list")
    payload = copy.deepcopy(dict(raw))
    payload["beat_grid"] = dict(beat_grid)
    analysis = dict(payload.get("analysis") or {})
    tempo = beat_grid.get("tempo") if isinstance(beat_grid.get("tempo"), Mapping) else {}
    meter = beat_grid.get("time_signature") if isinstance(beat_grid.get("time_signature"), Mapping) else {}
    selected_bpm = float(tempo.get("selected_bpm") or analysis.get("bpm") or 120.0)
    selected_meter = str(meter.get("selected") or analysis.get("time_signature") or "4/4")
    analysis["bpm"] = selected_bpm
    analysis["time_signature"] = selected_meter
    analysis["duration_sec"] = float(case["duration_sec"])
    metadata = dict(analysis.get("metadata") or {})
    metadata["beat_grid"] = dict(beat_grid)
    metadata["beat_source"] = "beatnet_full_track_context"
    metadata["beat_context_mode"] = "full_track_absolute_window"
    metadata["beat_audio_path"] = str(full_audio["path"])
    metadata["beat_audio_sha256"] = str(full_audio["sha256"])
    metadata["beat_grid_full_track_sha256"] = str(full_grid_record["sha256"])
    metadata["beatnet_version"] = str(beat_grid.get("beatnet", {}).get("version", BEATNET_VERSION)) if isinstance(beat_grid.get("beatnet"), Mapping) else BEATNET_VERSION
    metadata["beatnet_mode"] = str(beat_grid.get("mode") or "offline")
    metadata["beatnet_inference"] = str(beat_grid.get("inference") or "DBN")
    metadata["bpm_candidates"] = [float(item["bpm"]) for item in (tempo.get("candidates") or []) if isinstance(item, Mapping) and item.get("bpm") is not None]
    metadata["time_signature_candidates"] = [str(item["value"]) for item in (meter.get("candidates") or []) if isinstance(item, Mapping) and item.get("value")]
    analysis["metadata"] = metadata
    payload["analysis"] = analysis
    provenance = dict(payload.get("provenance") or {})
    provenance["beat_source"] = "original_mix_full_track_context"
    provenance["beat_engine"] = "beatnet"
    provenance["beatnet_version"] = beat_grid.get("beatnet", {}).get("version", BEATNET_VERSION) if isinstance(beat_grid.get("beatnet"), Mapping) else BEATNET_VERSION
    provenance["beat_independent_of_reference"] = True
    provenance["beat_context"] = dict(beat_grid.get("context") or {})
    provenance["beat_source_audio"] = dict(full_audio)
    provenance["beat_source_grid"] = dict(full_grid_record)
    original_evidence = provenance.get("beat_onset_evidence")
    provenance["beat_onset_evidence"] = {
        "sources": list((tempo.get("evidence_sources") or [])),
        "counts": {},
        "meter_inference_uses_independent_accents": False,
        "full_track_context": True,
        "original_clip_evidence": original_evidence if isinstance(original_evidence, Mapping) else None,
    }
    provenance["note_source_raw"] = {
        "path": os.fspath(source_raw_path),
        "sha256": _sha256(source_raw_path),
        "model_output": True,
        "note_count": len(raw["notes"]),
        "source": "immutable_GAME_raw_notes; no note regeneration",
    }
    payload["provenance"] = provenance
    return payload


def prepare_context(
    *,
    selection_manifest: Path = DEFAULT_SELECTION_MANIFEST,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    full_audio: Path | None = None,
    overwrite: bool = False,
    rerun_full_track: bool = False,
) -> dict[str, Any]:
    selection_path = selection_manifest.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping):
        raise ValueError("selection manifest is not an object")
    cases = _case_index(selection)
    full_mix_record = selection.get("alignment", {}).get("full_mix") if isinstance(selection.get("alignment"), Mapping) else None
    if not isinstance(full_mix_record, Mapping):
        raise ValueError("selection manifest has no alignment.full_mix record")
    full_audio_path = full_audio.resolve() if full_audio else _resolve(str(full_mix_record.get("path")), base=selection_path.parent)
    if not full_audio_path.is_file():
        raise FileNotFoundError(f"full-track audio is unavailable: {full_audio_path}")
    actual_audio = _audio_record(full_audio_path)
    expected_audio_hash = str(full_mix_record.get("sha256") or "")
    if expected_audio_hash and actual_audio["sha256"] != expected_audio_hash:
        raise ValueError(f"full-track audio hash mismatch: expected {expected_audio_hash}, got {actual_audio['sha256']}")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    full_dir = output_root / "full_track"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_grid_path = full_dir / "beat_grid.json"
    if full_grid_path.is_file() and not rerun_full_track:
        full_grid = json.loads(full_grid_path.read_text(encoding="utf-8"))
    else:
        from backend.jianpu_score.beatnet import analyze_with_beatnet

        full_grid = analyze_with_beatnet(full_audio_path, duration_sec=float(actual_audio["duration_sec"]))
        full_grid_path.write_text(json.dumps(full_grid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not isinstance(full_grid, Mapping):
        raise ValueError("full-track BeatNet did not return a grid object")
    full_grid_record = _full_grid_record(full_grid, path=full_grid_path)
    if full_grid_record["beat_count"] != 179:
        raise ValueError(f"unexpected full-track BeatNet beat count: {full_grid_record['beat_count']} (expected 179)")
    output_cases: list[dict[str, Any]] = []
    raw_root = raw_root.resolve()
    for case_id in sorted(cases):
        case = cases[case_id]
        source_raw_path = raw_root / case_id / "raw" / "recognition.json"
        if not source_raw_path.is_file():
            raise FileNotFoundError(f"immutable production raw is unavailable: {source_raw_path}")
        raw = json.loads(source_raw_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"raw is not an object: {source_raw_path}")
        case_root = output_root / case_id
        if case_root.exists() and not overwrite:
            raise FileExistsError(f"context case exists; use --overwrite: {case_root}")
        case_root.mkdir(parents=True, exist_ok=True)
        context_grid = crop_full_track_beat_grid(
            full_grid,
            absolute_start_sec=float(case["audio_start_sec"]),
            duration_sec=float(case["duration_sec"]),
            full_audio=actual_audio,
            full_grid_record=full_grid_record,
        )
        context_raw = _context_raw(
            raw,
            case=case,
            beat_grid=context_grid,
            full_audio=actual_audio,
            full_grid_record=full_grid_record,
            source_raw_path=source_raw_path,
        )
        destination = case_root / "raw" / "recognition.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(context_raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (case_root / "raw" / "beat_grid.json").write_text(json.dumps(context_grid, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written = json.loads(destination.read_text(encoding="utf-8"))
        if written.get("notes") != raw.get("notes"):
            raise AssertionError(f"context transformation changed GAME notes for {case_id}")
        output_cases.append(
            {
                "case_id": case_id,
                "source_raw": os.fspath(source_raw_path),
                "source_raw_sha256": _sha256(source_raw_path),
                "context_raw": os.fspath(destination),
                "context_raw_sha256": _sha256(destination),
                "note_count": len(raw["notes"]),
                "window": context_grid["context"]["window"],
            }
        )
    manifest = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "mode": "ccmusic_full_track_beatnet_absolute_windows",
        "selection_manifest": os.fspath(selection_path),
        "source_raw_root": os.fspath(raw_root),
        "full_track_audio": actual_audio,
        "full_track_beat_grid": full_grid_record,
        "full_track_beat_grid_sha256": full_grid_record["sha256"],
        "full_track_beat_count": full_grid_record["beat_count"],
        "reference_grid_used_for_model": False,
        "notes_policy": "reuse immutable per-window production GAME notes exactly; no recognition rerun",
        "cases": output_cases,
    }
    manifest_path = output_root / "context_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--full-audio", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-full-track", action="store_true", help="rerun BeatNet even when an immutable full-track grid already exists")
    args = parser.parse_args(argv)
    manifest = prepare_context(
        selection_manifest=args.selection_manifest,
        raw_root=args.raw_root,
        output_root=args.output_root,
        full_audio=args.full_audio,
        overwrite=args.overwrite,
        rerun_full_track=args.rerun_full_track,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "full_track_beat_count": manifest["full_track_beat_count"],
                "full_track_beat_grid_sha256": manifest["full_track_beat_grid_sha256"],
                "cases": [item["case_id"] for item in manifest["cases"]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
