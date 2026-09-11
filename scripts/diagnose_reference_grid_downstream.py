"""Run an offline downstream oracle with exact reference beat grids.

The production-acceptance-v3 raw recognition payloads are immutable inputs to
this diagnostic.  Only their beat grid is replaced with the corresponding
reference annotation.  Each case then goes through the same seconds-to-beats,
performance MIDI, MuseScore, MusicXML standardization, and jianpu rendering
service used by the v3 chain.

This script is deliberately diagnostic-only.  It writes a separate artifact
root, never changes the production acceptance directory, and does not expose
the oracle grid to a runtime route.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.jianpu_score.quantize import _build_beat_mapper  # noqa: E402
from backend.jianpu_score.high_accuracy import resolve_notation_python  # noqa: E402
from backend.jianpu_score.musicxml_standardize import (  # noqa: E402
    _estimate_source_alignment,
    _logical_pitch_units,
    _source_notes,
    _worker_raw_events,
    run_musicxml_worker,
    standardize_musicxml,
)
from backend.jianpu_score.render import render_score  # noqa: E402
from backend.jianpu_score.svg_long import merge_svg_pages  # noqa: E402
from backend.jianpu_score.quantize import jianpu_serialization_diagnostics  # noqa: E402
from scripts.high_accuracy_benchmark import (  # noqa: E402
    _midi_notes,
    chord_retention,
    pitch_metrics,
    rhythm_error,
)
from scripts.run_high_accuracy_batch import (  # noqa: E402
    _analysis_and_events_from_raw,
    _pitched_events,
    high_accuracy_service_adapter,
)


DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "reference-grid-downstream-oracle-v1"
TOLERANCE_QUARTERS = Fraction(1, 16)
SCHEMA_VERSION = "reference_grid_downstream_oracle_v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_times(records: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for item in records:
        try:
            value = float(item["time_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _reference_grid(annotation: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a reference annotation to the production beat-grid contract."""

    source_beats = annotation.get("beats")
    if not isinstance(source_beats, list) or len(source_beats) < 2:
        raise ValueError("reference annotation must contain at least two beats")
    beats = [copy.deepcopy(item) for item in source_beats if isinstance(item, Mapping)]
    times = _finite_times(beats)
    if len(times) < 2 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("reference annotation beat times must be strictly increasing")
    downbeats = annotation.get("downbeats")
    if not isinstance(downbeats, list):
        downbeats = [item for item in beats if bool(item.get("downbeat"))]
    downbeats = [copy.deepcopy(item) for item in downbeats if isinstance(item, Mapping)]
    intervals = [right - left for left, right in zip(times, times[1:])]
    median_interval = statistics.median(intervals)
    if median_interval <= 0:
        raise ValueError("reference annotation has no positive beat interval")
    first_downbeat = next((index for index, item in enumerate(beats) if bool(item.get("downbeat"))), None)
    if first_downbeat is None:
        first_downbeat = 0
    meter = str(annotation.get("time_signature") or "4/4")
    raw_duration = float((raw.get("analysis") or {}).get("duration_sec") or 0.0)
    duration = max(raw_duration, times[-1], 0.1)
    annotation_bpm = annotation.get("tempo_bpm")
    try:
        bpm = float(annotation_bpm) if annotation_bpm is not None else 60.0 / median_interval
    except (TypeError, ValueError):
        bpm = 60.0 / median_interval
    if not math.isfinite(bpm) or bpm <= 0:
        bpm = 60.0 / median_interval
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "reference_grid_downstream_oracle",
        "engine": "reference_annotation",
        "source": "production_acceptance_v3_reference_beat_annotation",
        "beats": beats,
        "downbeats": downbeats,
        "bars": [],
        "duration_sec": duration,
        "time_signature": meter,
        "tempo": {
            "detected_bpm": bpm,
            "selected_bpm": bpm,
            "selected_factor": 1.0,
            "manual_bpm": None,
            "reference_median_interval_sec": median_interval,
        },
        "mapping": {
            "manual_bpm_scale": 1.0,
            "score_origin": {
                "downbeat_index": first_downbeat,
                "source": "reference_annotation",
            },
        },
        "oracle": {
            "reference_time_signature": meter,
            "reference_annotation_policy": annotation.get("annotation_policy"),
            "reference_beat_count": len(beats),
            "reference_downbeat_count": len(downbeats),
            "reference_tempo_bpm": bpm,
        },
        "warnings": [],
    }


def _as_fraction(value: float) -> Fraction:
    return Fraction(str(round(float(value), 9))).limit_denominator(1_000_000)


def _continuous_note_events(events: Sequence[Any], analysis: Any) -> list[tuple[int, Fraction, Fraction]]:
    """Represent mapped raw events before MIDI tick rounding.

    This is the seconds-to-beats boundary.  Comparing it with the reference
    MIDI helps distinguish map/tick effects from later MuseScore/normalizer
    changes.  It intentionally keeps the immutable recognized event set.
    """

    mapper = _build_beat_mapper(analysis, list(events))
    return sorted(
        (
            int(event.midi),
            _as_fraction(mapper.seconds_to_beat(event.start_sec)),
            _as_fraction(mapper.seconds_to_beat(event.end_sec)),
        )
        for event in events
    )


def _metric_bundle(reference: Sequence[tuple[int, Fraction, Fraction]], predicted: Sequence[tuple[int, Fraction, Fraction]]) -> dict[str, Any]:
    return {
        "pitch_f1": pitch_metrics(reference, predicted, tolerance_quarters=TOLERANCE_QUARTERS),
        "rhythm_error": rhythm_error(reference, predicted, tolerance_quarters=TOLERANCE_QUARTERS),
        "chord_retention": chord_retention(reference, predicted, tolerance_quarters=TOLERANCE_QUARTERS),
    }


def _metric_value(metrics: Mapping[str, Any], name: str, field: str) -> float | None:
    value = metrics.get(name)
    if not isinstance(value, Mapping):
        return None
    number = value.get(field)
    return float(number) if isinstance(number, (int, float)) and math.isfinite(float(number)) else None


def _mean(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return statistics.fmean(available) if available else None


def _improvement_percent(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def _find_artifact(result_payload: Mapping[str, Any], *, suffix: str) -> str:
    for item in result_payload.get("artifacts", []):
        if isinstance(item, Mapping) and str(item.get("relative_path", "")).lower().endswith(suffix.lower()):
            return str(item["relative_path"])
    raise FileNotFoundError(f"service result did not register {suffix}")


def _diagnostic_artifacts(service_output: Path) -> list[dict[str, Any]]:
    kind_by_suffix = {
        ".json": "metadata",
        ".mid": "midi",
        ".musicxml": "musicxml",
        ".jly": "jianpu-ly",
        ".ly": "lilypond",
        ".svg": "svg",
        ".log": "log",
    }
    artifacts: list[dict[str, Any]] = []
    for path in sorted(item for item in service_output.rglob("*") if item.is_file()):
        relative = path.relative_to(service_output).as_posix()
        digest = _sha256(path)
        artifacts.append(
            {
                "artifact_id": relative.replace("/", "_"),
                "kind": kind_by_suffix.get(path.suffix.lower(), "file"),
                "relative_path": relative,
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    return artifacts


def _midi_tempo_points(path: Path) -> list[dict[str, Any]]:
    """Read the tempo events that actually reached MuseScore's MIDI input."""

    midi = mido.MidiFile(os.fspath(path))
    points: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type != "set_tempo":
                continue
            points.append(
                {
                    "track_index": track_index,
                    "tick_480": absolute_tick,
                    "bpm": float(mido.tempo2bpm(message.tempo)),
                    "microseconds_per_beat": int(message.tempo),
                }
            )
    return sorted(points, key=lambda item: (int(item["tick_480"]), int(item["track_index"])))


def _xml_import_shape(path: Path) -> dict[str, Any]:
    """Extract the imported score origin facts without changing MusicXML."""

    tree = ET.parse(path)
    root = tree.getroot()

    def tag(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    parts = [element for element in root if tag(element) == "part"]
    first_part = parts[0] if parts else None
    measures = [element for element in first_part or [] if tag(element) == "measure"]
    first_measure = measures[0] if measures else None
    meter: str | None = None
    if first_measure is not None:
        for attributes in first_measure:
            if tag(attributes) != "attributes":
                continue
            time = next((child for child in attributes if tag(child) == "time"), None)
            if time is not None:
                beats = next((child.text for child in time if tag(child) == "beats"), None)
                beat_type = next((child.text for child in time if tag(child) == "beat-type"), None)
                if beats and beat_type:
                    meter = f"{beats}/{beat_type}"
                break
    return {
        "part_count": len(parts),
        "first_measure_number": first_measure.attrib.get("number") if first_measure is not None else None,
        "first_measure_implicit": (first_measure.attrib.get("implicit") if first_measure is not None else None),
        "first_measure_meter": meter,
        "measure_count": len(measures),
    }


def _alignment_diagnostics(
    *,
    case: Mapping[str, Any],
    case_output: Path,
    performance_path: Path,
    oracle_grid: Mapping[str, Any],
    cause: BaseException,
) -> dict[str, Any]:
    """Explain strict source-tempo/score-origin failures without relaxing them."""

    service_output = case_output / "service_output"
    performance_name = performance_path.name
    metadata_name = performance_name.replace(".performance.mid", ".performance.metadata.json")
    metadata_path = performance_path.with_name(metadata_name)
    xml_candidates = sorted(service_output.glob(f"{case['id']}.*.notated.musicxml"))
    diagnostic: dict[str, Any] = {
        "schema_version": "reference_grid_alignment_diagnostic_v1",
        "case_id": str(case["id"]),
        "reference_time_signature": oracle_grid.get("time_signature"),
        "strict_failure": str(cause),
        "strict_failure_stage": getattr(cause, "stage", None),
        "performance_metadata": str(metadata_path.relative_to(case_output)).replace("\\", "/"),
        "performance_midi": str(performance_path.relative_to(case_output)).replace("\\", "/"),
        "musicxml": str(xml_candidates[0].relative_to(case_output)).replace("\\", "/") if xml_candidates else None,
    }
    if not metadata_path.is_file() or not xml_candidates:
        diagnostic.update(
            {
                "diagnosis": "insufficient_artifacts_for_source_score_probe",
                "conclusion": "The strict failure is preserved, but the generated artifacts do not permit a source-tempo/origin comparison.",
            }
        )
        return diagnostic
    try:
        metadata = _load(metadata_path)
        xml_path = xml_candidates[0]
        worker_payload = run_musicxml_worker(xml_path, notation_python=resolve_notation_python(), timeout_sec=180)
        raw_events, worker_diagnostics = _worker_raw_events(worker_payload)
        source_notes = _source_notes(metadata)
        _hints, alignment = _estimate_source_alignment(raw_events, source_notes)
        units = _logical_pitch_units(raw_events)
        pitched_events = [event for event in raw_events if event.pitches]
        first_source = min(source_notes, key=lambda item: (int(item["start_tick"]), int(item["source_index"]))) if source_notes else None
        first_unit = min(units, key=lambda item: (int(item.start_tick), int(item.unit_id))) if units else None
        last_source = max(source_notes, key=lambda item: (int(item["end_tick"]), int(item["source_index"]))) if source_notes else None
        last_unit = max(units, key=lambda item: (int(item.end_tick), int(item.unit_id))) if units else None
        models = [item for item in alignment.get("models", []) if isinstance(item, Mapping)]
        diagnostic.update(
            {
                "source_time_signature": metadata.get("time_signature"),
                "source_score_origin": metadata.get("score_origin"),
                "source_score_origin_audio_sec": metadata.get("score_origin_audio_sec"),
                "source_tempo_points_metadata": metadata.get("tempo_points", []),
                "source_tempo_points_midi": _midi_tempo_points(performance_path),
                "musicxml_import_shape": _xml_import_shape(xml_path),
                "musicxml_worker_highest_time_quarter": worker_payload.highest_time_quarter,
                "musicxml_worker_event_count": len(raw_events),
                "musicxml_worker_diagnostics": worker_diagnostics,
                "musicxml_first_pitched_event": {
                    "start_tick_48": int(min(pitched_events, key=lambda event: event.start_tick).start_tick),
                    "end_tick_48": int(min(pitched_events, key=lambda event: event.start_tick).end_tick),
                    "pitches": list(min(pitched_events, key=lambda event: event.start_tick).pitches),
                }
                if pitched_events
                else None,
                "source_first_note": first_source,
                "source_last_note": last_source,
                "musicxml_first_logical_pitch_unit": {
                    "unit_id": int(first_unit.unit_id),
                    "pitch": int(first_unit.pitch),
                    "start_tick_48": int(first_unit.start_tick),
                    "end_tick_48": int(first_unit.end_tick),
                }
                if first_unit is not None
                else None,
                "musicxml_last_logical_pitch_unit": {
                    "unit_id": int(last_unit.unit_id),
                    "pitch": int(last_unit.pitch),
                    "start_tick_48": int(last_unit.start_tick),
                    "end_tick_48": int(last_unit.end_tick),
                }
                if last_unit is not None
                else None,
                "source_to_musicxml_alignment_probe": alignment,
                "alignment_models": models,
            }
        )
        if first_source is not None and first_unit is not None:
            diagnostic["first_note_coordinate_difference_ticks_48"] = {
                "start": int(first_unit.start_tick) - int(first_source["start_tick"]),
                "end": int(first_unit.end_tick) - int(first_source["end_tick"]),
            }
        if last_source is not None and last_unit is not None:
            diagnostic["last_note_coordinate_difference_ticks_48"] = {
                "start": int(last_unit.start_tick) - int(last_source["start_tick"]),
                "end": int(last_unit.end_tick) - int(last_source["end_tick"]),
            }
        reference_meter = str(oracle_grid.get("time_signature") or "")
        source_meter = str(metadata.get("time_signature") or "")
        imported_meter = str((diagnostic.get("musicxml_import_shape") or {}).get("first_measure_meter") or "")
        reason = str(alignment.get("reason") or "")
        midi_tempos = diagnostic.get("source_tempo_points_midi") or []
        metadata_tempos = diagnostic.get("source_tempo_points_metadata") or []
        diagnostic["source_tempo_origin_probe"] = {
            "metadata_point_count": len(metadata_tempos),
            "midi_point_count": len(midi_tempos),
            "metadata_tick_zero_present": any(int(item.get("tick", -1)) == 0 for item in metadata_tempos if isinstance(item, Mapping)),
            "midi_tick_zero_present": any(int(item.get("tick_480", -1)) == 0 for item in midi_tempos if isinstance(item, Mapping)),
            "first_metadata_point": metadata_tempos[0] if metadata_tempos else None,
            "first_midi_point": midi_tempos[0] if midi_tempos else None,
            "declared_score_origin": metadata.get("score_origin"),
            "first_source_note_tick_48": int(first_source["start_tick"]) if first_source is not None else None,
            "first_musicxml_note_tick_48": int(first_unit.start_tick) if first_unit is not None else None,
            "source_tempo_tick_conversion": "tick_480 / 10 = nominal score tick_48 before any proven source-to-score model",
        }
        if source_meter != reference_meter:
            diagnosis = "oracle_grid_meter_not_propagated_to_performance_metadata"
            conclusion = (
                f"The performance metadata still declares {source_meter or 'no meter'} while the oracle grid declares {reference_meter}; "
                "this is an oracle input propagation defect, so source-tempo/score-origin correspondence is not a valid production comparison."
            )
        elif imported_meter != reference_meter:
            diagnosis = "musescore_import_meter_inference_mismatch"
            conclusion = (
                f"The performance MIDI and metadata carry {reference_meter} from tick zero, but MuseScore MusicXML imports {imported_meter or 'no meter'} in its first measure. "
                "That changes the score bar/origin interpretation before normalization, so the source tempo points cannot be proven against the imported Score origin or first-note sequence; this is a MuseScore import mismatch, not a missing oracle grid field."
            )
        elif "mapped production tempo lies before" in str(cause):
            diagnosis = "production_standardizer_tempo_origin_rejection"
            conclusion = (
                "The oracle meter and performance metadata agree, but the standardizer maps a source tempo before the imported Score origin. "
                "This is a production source-tempo/score-origin reconciliation failure; the fail-closed rejection must remain until a universal proof is available."
            )
        elif reason in {
            "source_and_musicxml_pitch_counts_differ",
            "source_and_musicxml_pitch_sets_differ",
            "provisional_alignment_not_unique_same_pitch_timing",
            "provisional_alignment_global_order_reversed",
            "source_alignment_movement_exceeds_bound",
            "source_midi_lane_alignment_residual_exceeds_bound",
            "alignment_residual_exceeds_bound",
        }:
            diagnosis = "musescore_import_or_score_coordinate_mismatch"
            conclusion = (
                f"The oracle meter is propagated ({reference_meter}), but MuseScore MusicXML produces a non-identity source-to-score relation ({reason}). "
                "The exact source tempo points cannot be proven against the imported Score origin/first-note sequence; this is downstream importer/normalizer evidence, not a missing beat-grid field."
            )
        else:
            diagnosis = "production_source_to_score_alignment_rejected"
            conclusion = (
                f"The oracle meter is propagated ({reference_meter}), but the strict source-to-score proof is rejected as {reason or 'unspecified'}. "
                "No broad alignment relaxation is justified by this diagnostic."
            )
        diagnostic.update({"diagnosis": diagnosis, "conclusion": conclusion})
    except Exception as exc:
        diagnostic.update(
            {
                "diagnosis": "alignment_probe_failed",
                "probe_error": f"{type(exc).__name__}: {exc}",
                "conclusion": "The strict failure remains fail-closed because the source-tempo/score-origin probe itself did not complete.",
            }
        )
    return diagnostic


def _diagnostic_standardize_fallback(
    *,
    case: Mapping[str, Any],
    case_output: Path,
    cause: BaseException,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finish notation after strict source-tempo alignment rejects the oracle.

    Some exact reference grids produce a tempo point at every beat while the
    MuseScore human-performance importer moves the corresponding first note.
    The production service correctly fails closed in that situation.  For this
    offline attribution, standardize the already generated MusicXML without
    source performance metadata, then record the fallback explicitly.  This
    preserves MuseScore's actual import output and measures the downstream
    notation result without inventing a source-to-score alignment.
    """

    source_kind = str(case.get("source_kind") or "")
    variant = "game-cleaned" if source_kind == "vocal" else "instrument-part"
    service_output = case_output / "service_output"
    instrument = str(case["id"])
    musicxml_path = service_output / f"{instrument}.{variant}.notated.musicxml"
    if not musicxml_path.is_file():
        raise RuntimeError(f"diagnostic fallback requires completed MuseScore MusicXML: {musicxml_path}") from cause
    score, alignment_report = standardize_musicxml(
        musicxml_path,
        performance_metadata=None,
        title=str(case.get("title") or instrument),
        notation_python=resolve_notation_python(),
        timeout_sec=180,
    )
    score_path = service_output / f"{instrument}.score.json"
    alignment_path = service_output / f"{instrument}.alignment_report.json"
    _write_json(score_path, score.model_dump(mode="json"))
    _write_json(alignment_path, alignment_report)
    serialization = jianpu_serialization_diagnostics(score)
    score = score.model_copy(update={"metadata": {**score.metadata, "jianpu_serialization": serialization}})
    _write_json(score_path, score.model_dump(mode="json"))
    rendered = render_score(score, service_output, basename=f"{instrument}.score")
    long_svg = service_output / f"{instrument}.score.long.svg"
    merge_svg_pages(rendered.svg_paths, long_svg)
    existing_manifest_path = service_output / "manifest.json"
    existing_manifest = _load(existing_manifest_path) if existing_manifest_path.is_file() else {}
    oracle_manifest = copy.deepcopy(existing_manifest)
    oracle_manifest.update(
        {
            "status": "completed",
            "jianpu_status": "completed",
            "diagnostic_tempo_fallback": {
                "schema_version": SCHEMA_VERSION,
                "source_service_failure": str(cause),
                "source_service_stage": getattr(cause, "stage", None),
                "performance_metadata_passed_to_standardizer": False,
                "reason": "strict_source_to_score_tempo_alignment_unproven_for_reference_grid",
            },
            "stages": {
                **(oracle_manifest.get("stages") or {}),
                "musicxml_standardize": {
                    "status": "completed",
                    "diagnostic_fallback": True,
                    "source_performance_metadata": False,
                    "alignment_source_count": int(alignment_report.get("source_note_count", 0)),
                    "alignment_accounted_source_count": int(alignment_report.get("accounted_source_count", 0)),
                    "alignment_unresolved_count": int(alignment_report.get("unresolved_count", 0)),
                    "jianpu_serialization": serialization,
                },
                "render": {"status": "completed", "diagnostic_fallback": True},
            },
        }
    )
    oracle_manifest_path = service_output / "manifest.oracle.json"
    _write_json(oracle_manifest_path, oracle_manifest)
    performance_rel = f"{instrument}.{variant}.performance.mid"
    performance_path = service_output / performance_rel
    if not performance_path.is_file():
        raise RuntimeError(f"diagnostic fallback requires performance MIDI: {performance_path}") from cause
    final_path = Path(rendered.midi_path).resolve() if rendered.midi_path else None
    if final_path is None or not final_path.is_file():
        raise RuntimeError(f"diagnostic fallback renderer produced no final MIDI for {instrument}") from cause
    result_payload = {
        "engine": "musescore-midi-import",
        "variant": variant,
        "profile": variant,
        "manifest": "service_output/manifest.oracle.json",
        "status": "completed_with_diagnostic_tempo_fallback",
        "jianpu_status": "completed",
        "artifacts": _diagnostic_artifacts(service_output),
        "final_midi": str(final_path.relative_to(case_output.resolve())),
        "beat_grid": "raw/beat_grid.json",
        "diagnostic_tempo_fallback": True,
    }
    fallback_info = {
        "used": True,
        "cause": str(cause),
        "stage": getattr(cause, "stage", None),
        "standardizer_source_performance_metadata": False,
        "original_service_manifest": str(existing_manifest_path.relative_to(case_output.resolve())),
        "oracle_service_manifest": str(oracle_manifest_path.relative_to(case_output.resolve())),
    }
    return result_payload, fallback_info


def _baseline_index(batch_root: Path) -> dict[str, Mapping[str, Any]]:
    report = _load(batch_root / "production-report-v3.json")
    return {str(item["id"]): item for item in report.get("cases", []) if isinstance(item, Mapping) and item.get("id")}


def _case_record(
    *,
    batch_root: Path,
    output_root: Path,
    case: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(case["id"])
    source_root = batch_root / case_id
    raw_path = source_root / "raw" / "recognition.json"
    raw = _load(raw_path)
    annotation_record = case.get("beat_annotation")
    if not isinstance(annotation_record, Mapping):
        raise FileNotFoundError(f"{case_id}: evaluator has no beat annotation")
    annotation_path = Path(str(annotation_record["path"])).expanduser().resolve()
    annotation_payload = _load(annotation_path)
    annotation = annotation_payload.get("beat_grid", annotation_payload)
    if not isinstance(annotation, Mapping):
        raise ValueError(f"{case_id}: reference annotation payload is not an object")
    oracle_grid = _reference_grid(annotation, raw)
    oracle_raw = copy.deepcopy(raw)
    oracle_raw["beat_grid"] = oracle_grid
    # The v3 raw analysis may carry a BeatNet meter candidate that differs
    # from the exact annotation.  ``_analysis_and_events_from_raw`` gives the
    # analysis payload precedence over beat_grid, so an oracle substitution
    # must update both fields or it would still run the old meter/origin.
    oracle_raw.setdefault("analysis", {})["time_signature"] = oracle_grid["time_signature"]
    oracle_raw["analysis"]["bpm"] = oracle_grid["tempo"]["selected_bpm"]
    oracle_raw.setdefault("analysis", {})["metadata"] = dict((oracle_raw.get("analysis") or {}).get("metadata") or {})
    oracle_raw["analysis"]["metadata"]["beat_grid_oracle"] = True
    case_output = output_root / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    raw_output = case_output / "raw"
    raw_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, raw_output / "recognition.json")
    original_grid_path = source_root / "raw" / "beat_grid.json"
    if original_grid_path.is_file():
        shutil.copy2(original_grid_path, raw_output / "source_beat_grid.json")
    _write_json(raw_output / "beat_grid.json", oracle_grid)
    _write_json(raw_output / "oracle_recognition.json", oracle_raw)
    _write_json(
        raw_output / "oracle_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "source_recognition": str(raw_path.resolve()),
            "source_recognition_sha256": _sha256(raw_path),
            "source_beat_grid": str(original_grid_path.resolve()),
            "source_beat_grid_sha256": _sha256(original_grid_path) if original_grid_path.is_file() else None,
            "reference_annotation": str(annotation_path),
            "reference_annotation_sha256": _sha256(annotation_path),
            "recognition_unchanged": True,
            "reference_grid_only_substitution": True,
        },
    )
    service_output = case_output / "service_output"
    failed_service_manifest = service_output / "manifest.json"
    if failed_service_manifest.is_file():
        try:
            failed_payload = _load(failed_service_manifest)
        except (OSError, ValueError, TypeError):
            failed_payload = {}
        if isinstance(failed_payload, Mapping) and str(failed_payload.get("status", "")).lower() == "failed":
            archive = case_output / "service_output.strict-failure"
            if archive.exists():
                shutil.rmtree(archive)
            shutil.move(str(service_output), str(archive))
    started = time.perf_counter()
    fallback_info: dict[str, Any] = {"used": False}
    try:
        result_payload = high_accuracy_service_adapter(case, oracle_raw, case_output)
    except Exception as exc:
        if getattr(exc, "stage", None) != "musicxml_standardize":
            raise
        result_payload, fallback_info = _diagnostic_standardize_fallback(
            case=case,
            case_output=case_output,
            cause=exc,
        )
    elapsed = time.perf_counter() - started
    performance_rel = _find_artifact(result_payload, suffix=".performance.mid")
    final_rel = str(result_payload["final_midi"])
    performance_path = case_output / "service_output" / performance_rel
    final_path = case_output / final_rel
    reference_path = Path(str(case["reference_midi"]["path"])).expanduser().resolve()
    _, reference_notes = _midi_notes(reference_path, exclude_drum_channel=True)
    _, performance_notes = _midi_notes(performance_path, exclude_drum_channel=True)
    _, final_notes = _midi_notes(final_path, exclude_drum_channel=True)
    analysis, all_events = _analysis_and_events_from_raw(oracle_raw, case)
    events = _pitched_events(all_events)
    analysis = analysis.model_copy(update={"note_events": events})
    continuous_notes = _continuous_note_events(events, analysis)
    metrics_continuous = _metric_bundle(reference_notes, continuous_notes)
    metrics_performance = _metric_bundle(reference_notes, performance_notes)
    metrics_final = _metric_bundle(reference_notes, final_notes)
    notation_delta = _metric_bundle(performance_notes, final_notes)
    alignment_diagnostics_rel: str | None = None
    if bool(fallback_info.get("used")):
        alignment_diagnostics = _alignment_diagnostics(
            case=case,
            case_output=case_output,
            performance_path=performance_path,
            oracle_grid=oracle_grid,
            cause=RuntimeError(str(fallback_info.get("cause") or "strict standardization failed")),
        )
        alignment_diagnostics_path = case_output / "alignment_diagnostics.json"
        _write_json(alignment_diagnostics_path, alignment_diagnostics)
        alignment_diagnostics_rel = str(alignment_diagnostics_path.relative_to(output_root)).replace("\\", "/")
        fallback_info = {**fallback_info, "diagnosis": alignment_diagnostics.get("diagnosis"), "alignment_diagnostics": alignment_diagnostics_rel}
    result_payload["artifacts"] = _diagnostic_artifacts(service_output)
    # A diagnostic fallback deliberately leaves the strict failed manifest in
    # place for auditability and writes a separate manifest.oracle.json.  The
    # root manifest/result must reference the manifest that describes the
    # artifact set being scored; embedding the failed strict payload here
    # makes a completed diagnostic look like a failed service run.
    service_manifest_rel = str(result_payload.get("manifest") or "service_output/manifest.json").replace("\\", "/")
    service_manifest_path = case_output / service_manifest_rel
    if not service_manifest_path.is_file():
        raise FileNotFoundError(f"service result manifest is missing: {service_manifest_path}")
    service_manifest = _load(service_manifest_path)
    result_payload = dict(result_payload)
    result_payload["manifest"] = service_manifest_rel
    root_manifest = {
        "schema_version": "1.1",
        "status": "success",
        "case_id": case_id,
        "pipeline": "reference-grid-downstream-oracle-v1",
        "stage": "oracle",
        "source_kind": case.get("source_kind"),
        "effective_evaluation_scope": "reference_grid_downstream_oracle",
        "raw_model_output": True,
        "reference_grid_substituted": True,
        "raw_recognition_sha256": _sha256(raw_path),
        "beat_grid": "raw/beat_grid.json",
        "final_midi": final_rel,
        "diagnostic_tempo_fallback": fallback_info,
        "result": result_payload,
        "service_manifest": service_manifest_rel,
        "service_manifest_payload": service_manifest,
        "alignment_diagnostics": alignment_diagnostics_rel,
    }
    _write_json(case_output / "manifest.json", root_manifest)
    _write_json(case_output / "metrics.json", {
        "continuous_mapping": metrics_continuous,
        "performance_mapping": metrics_performance,
        "final_score": metrics_final,
        "notation_delta_performance_to_final": notation_delta,
    })
    baseline_metrics = ((baseline.get("baseline") or {}).get("metrics") or {})
    return {
        "id": case_id,
        "category": case.get("category"),
        "source_kind": case.get("source_kind"),
        "status": "success",
        "elapsed_sec": elapsed,
        "reference_annotation": str(annotation_path),
        "reference_grid": {
            "beat_count": len(oracle_grid["beats"]),
            "downbeat_count": len(oracle_grid["downbeats"]),
            "time_signature": oracle_grid["time_signature"],
            "median_interval_sec": oracle_grid["tempo"]["reference_median_interval_sec"],
        },
        "artifacts": {
            "performance_midi": str(performance_path.relative_to(output_root)),
            "score_midi": str(final_path.relative_to(output_root)),
            "service_manifest": str(service_manifest_path.relative_to(output_root)).replace("\\", "/"),
            "alignment_diagnostics": alignment_diagnostics_rel,
        },
        "continuous_mapping": metrics_continuous,
        "performance_mapping": metrics_performance,
        "final_score": metrics_final,
        "notation_delta_performance_to_final": notation_delta,
        "diagnostic_tempo_fallback": fallback_info,
        "baseline": baseline_metrics,
    }


def _fallback_used(item: Mapping[str, Any]) -> bool:
    fallback = item.get("diagnostic_tempo_fallback")
    return isinstance(fallback, Mapping) and bool(fallback.get("used"))


def _baseline_metrics(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("baseline") or {}
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("metrics")
    return nested if isinstance(nested, Mapping) else value


def _group_summary(items: Sequence[Mapping[str, Any]], *, label: str, gate_status: str) -> dict[str, Any]:
    """Aggregate one auditable evaluation population.

    The strict group is the only population that can support a production
    gate.  Fallback and all-diagnostic populations are still useful for
    attribution, but their metrics must never be presented as a strict
    production result.
    """

    successful = [item for item in items if item.get("status") == "success"]
    baseline_rhythm = _mean([
        _metric_value(_baseline_metrics(item), "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
        for item in successful
    ])
    final_rhythm = _mean([
        _metric_value(item.get("final_score") or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
        for item in successful
    ])
    mapping_continuous = _mean([
        _metric_value(item.get("continuous_mapping") or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
        for item in successful
    ])
    mapping_performance = _mean([
        _metric_value(item.get("performance_mapping") or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
        for item in successful
    ])
    notation = _mean([
        _metric_value(item.get("notation_delta_performance_to_final") or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
        for item in successful
    ])
    final_pitch = _mean([_metric_value(item.get("final_score") or {}, "pitch_f1", "f1") for item in successful])
    final_chord = _mean([_metric_value(item.get("final_score") or {}, "chord_retention", "retention") for item in successful])
    target_met = bool(final_rhythm is not None and baseline_rhythm is not None and final_rhythm <= baseline_rhythm * 0.8)
    return {
        "label": label,
        "case_count": len(items),
        "successful_count": len(successful),
        "failed_count": len(items) - len(successful),
        "baseline_mean_fixed_total_rhythm": baseline_rhythm,
        "continuous_mapping_mean_fixed_total_rhythm": mapping_continuous,
        "performance_mapping_mean_fixed_total_rhythm": mapping_performance,
        "final_score_mean_fixed_total_rhythm": final_rhythm,
        "notation_delta_mean_fixed_total_rhythm": notation,
        "final_score_mean_pitch_f1": final_pitch,
        "final_score_mean_chord_retention": final_chord,
        "continuous_mapping_improvement_vs_baseline_percent": _improvement_percent(baseline_rhythm, mapping_continuous),
        "performance_mapping_improvement_vs_baseline_percent": _improvement_percent(baseline_rhythm, mapping_performance),
        "final_score_improvement_vs_baseline_percent": _improvement_percent(baseline_rhythm, final_rhythm),
        "rhythm_20_percent_target": baseline_rhythm * 0.8 if baseline_rhythm is not None else None,
        "rhythm_20_percent_target_met": target_met,
        "gate_status": gate_status,
        "gate_eligible": gate_status == "strict_complete" and len(items) > 0 and len(successful) == len(items),
    }


def _aggregate(cases: Sequence[Mapping[str, Any]], baseline_report: Mapping[str, Any]) -> dict[str, Any]:
    # Keep the old top-level metric names for consumers of v1, while making
    # the population used for each metric explicit and preventing a fallback
    # success from silently becoming a production gate result.
    strict = [item for item in cases if item.get("status") == "success" and not _fallback_used(item)]
    fallback = [item for item in cases if item.get("status") == "success" and _fallback_used(item)]
    all_diagnostic = list(cases)
    strict_complete = bool(
        all_diagnostic
        and len(strict) == len(all_diagnostic)
        and not fallback
        and all(item.get("status") == "success" for item in all_diagnostic)
    )
    strict_gate_status = "strict_complete" if strict_complete else "strict_incomplete"
    groups = {
        "strict_production_compatible": _group_summary(
            strict,
            label="strict_production_compatible",
            gate_status=strict_gate_status,
        ),
        "diagnostic_tempo_fallback": _group_summary(fallback, label="diagnostic_tempo_fallback", gate_status="diagnostic_only"),
        "all_diagnostic": _group_summary(all_diagnostic, label="all_diagnostic", gate_status="provisional"),
    }
    all_summary = groups["all_diagnostic"]
    strict_summary = groups["strict_production_compatible"]
    fallback_summary = groups["diagnostic_tempo_fallback"]
    if strict_complete:
        reference_grid_sufficient = bool(strict_summary["rhythm_20_percent_target_met"])
        reference_grid_sufficiency_status = "strict_complete"
        reference_grid_sufficiency_reason = (
            f"all {len(all_diagnostic)} cases completed through the strict production-compatible path; "
            f"the 20% rhythm target was {'met' if reference_grid_sufficient else 'not met'} numerically."
        )
    else:
        reference_grid_sufficient = None
        reference_grid_sufficiency_status = "provisional_only_strict_population_incomplete"
        reference_grid_sufficiency_reason = (
            f"{len(fallback)} of {len(cases)} cases required diagnostic tempo fallback and "
            f"{sum(item.get('status') != 'success' for item in all_diagnostic)} failed; "
            "the all-diagnostic threshold cannot establish production sufficiency."
        )
    return {
        **all_summary,
        "groups": groups,
        "strict_production_compatible_count": len(strict),
        "diagnostic_tempo_fallback_count": len(fallback),
        "all_diagnostic_count": len(all_diagnostic),
        "strict_gate_status": strict_gate_status,
        "all_diagnostic_gate_status": "provisional",
        "fallback_gate_status": "diagnostic_only",
        "strict_gate_eligible": strict_complete,
        "reference_grid_sufficient_for_rhythm_target": reference_grid_sufficient,
        "reference_grid_sufficiency_status": reference_grid_sufficiency_status,
        "reference_grid_sufficiency_reason": reference_grid_sufficiency_reason,
        # Explicit aliases make the separate populations easy to consume from
        # JSON without requiring callers to understand the compatibility keys.
        "strict_production_compatible": strict_summary,
        "diagnostic_tempo_fallback": fallback_summary,
        "all_diagnostic": all_summary,
    }


def _by_category(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in cases:
        grouped.setdefault(str(item.get("category") or "uncategorized"), []).append(item)
    result: dict[str, Any] = {}
    for category, items in grouped.items():
        result[category] = {
            "count": len(items),
            "successful_count": sum(item.get("status") == "success" for item in items),
            "final_score_mean_fixed_total_rhythm": _mean([
                _metric_value(item.get("final_score") or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter") for item in items
            ]),
            "performance_mapping_mean_fixed_total_rhythm": _mean([
                _metric_value(item.get("performance_mapping") or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter") for item in items
            ]),
            "final_score_mean_pitch_f1": _mean([
                _metric_value(item.get("final_score") or {}, "pitch_f1", "f1") for item in items
            ]),
            "final_score_mean_chord_retention": _mean([
                _metric_value(item.get("final_score") or {}, "chord_retention", "retention") for item in items
            ]),
        }
    return result


def _write_markdown_legacy_v1(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Reference-grid downstream oracle v1",
        "",
        "This offline diagnostic keeps the production-v3 model recognition events immutable and substitutes only exact reference beat/downbeat annotations before running the complete downstream chain.",
        "",
        f"- Cases: **{summary['successful_count']}/{summary['case_count']} succeeded**; failures: **{summary['failed_count']}**.",
        f"- 20% rhythm target: **{str(summary['rhythm_20_percent_target_met']).lower()}**.",
        f"- Reference beat grid alone sufficient for target: **{str(summary['reference_grid_sufficient_for_rhythm_target']).lower()}**.",
        "",
        "## Aggregate",
        "",
        "| metric | value | improvement vs baseline |",
        "|---|---:|---:|",
        f"| baseline fixed-total rhythm | {summary['baseline_mean_fixed_total_rhythm']:.6f} | — |",
        f"| continuous seconds→beats mapping | {summary['continuous_mapping_mean_fixed_total_rhythm']:.6f} | {summary['continuous_mapping_improvement_vs_baseline_percent']:.3f}% |",
        f"| 480 PPQ performance MIDI | {summary['performance_mapping_mean_fixed_total_rhythm']:.6f} | {summary['performance_mapping_improvement_vs_baseline_percent']:.3f}% |",
        f"| final Score/MIDI/SVG chain | {summary['final_score_mean_fixed_total_rhythm']:.6f} | {summary['final_score_improvement_vs_baseline_percent']:.3f}% |",
        f"| performance→final notation delta | {summary['notation_delta_mean_fixed_total_rhythm']:.6f} | — |",
        f"| final pitch F1 | {summary['final_score_mean_pitch_f1']:.6f} | — |",
        f"| final chord retention | {summary['final_score_mean_chord_retention']:.6f} | — |",
        "",
        "## By category",
        "",
        "| category | n | performance rhythm | final rhythm | final pitch F1 | final chord retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in report["by_category"].items():
        lines.append(
            f"| {category} | {values['count']} | {values['performance_mapping_mean_fixed_total_rhythm']:.6f} | {values['final_score_mean_fixed_total_rhythm']:.6f} | {values['final_score_mean_pitch_f1']:.6f} | {values['final_score_mean_chord_retention'] if values['final_score_mean_chord_retention'] is not None else '—'} |"
        )
    lines += [
        "",
        "## Per case",
        "",
        "| case | category | performance rhythm | final rhythm | notation delta | pitch F1 | chord retention |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["cases"]:
        if item.get("status") != "success":
            lines.append(f"| {item['id']} | {item.get('category')} | failed | failed | failed | failed | failed |")
            continue
        values = {
            key: _metric_value(item.get(key) or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
            for key in ("performance_mapping", "final_score", "notation_delta_performance_to_final")
        }
        pitch = _metric_value(item["final_score"], "pitch_f1", "f1")
        chord = _metric_value(item["final_score"], "chord_retention", "retention")
        lines.append(
            f"| {item['id']} | {item.get('category')} | {values['performance_mapping']:.6f} | {values['final_score']:.6f} | {values['notation_delta_performance_to_final']:.6f} | {pitch:.6f} | {chord if chord is not None else '—'} |"
        )
    lines += [
        "",
        "The continuous mapping row is computed from the immutable recognized events before 480-PPQ rounding. The performance row includes that rounding and tempo-map MIDI encoding. The notation delta compares the performance MIDI with the final score MIDI after MuseScore import, MusicXML normalization, jianpu serialization, and LilyPond rendering.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    """Write a report that keeps strict and diagnostic populations separate."""

    summary = report["summary"]
    groups = summary.get("groups") or {}

    def fmt(value: Any, digits: int = 6) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Reference-grid downstream oracle v1",
        "",
        "This offline diagnostic keeps the production-v3 model recognition events immutable and substitutes only exact reference beat/downbeat annotations before running the complete downstream chain.",
        "",
        f"- Strict production-compatible cases: **{summary['strict_production_compatible_count']}/{summary['all_diagnostic_count']}**.",
        f"- Diagnostic tempo-fallback cases: **{summary['diagnostic_tempo_fallback_count']}/{summary['all_diagnostic_count']}**.",
        f"- All diagnostic cases: **{summary['successful_count']}/{summary['case_count']} succeeded**; failures: **{summary['failed_count']}**.",
        f"- Strict gate: **{summary['strict_gate_status']}**; all-diagnostic gate: **{summary['all_diagnostic_gate_status']}**.",
        f"- All-diagnostic 20% rhythm target: **{str(summary['rhythm_20_percent_target_met']).lower()} (provisional)**.",
        f"- Reference beat grid sufficiency: **{summary['reference_grid_sufficiency_status']}**.",
        f"- Reason: {summary['reference_grid_sufficiency_reason']}",
        "",
        "## Population comparison",
        "",
        "| population | n | status | baseline rhythm | final rhythm | improvement | final pitch F1 | final chord retention | target |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("strict_production_compatible", "diagnostic_tempo_fallback", "all_diagnostic"):
        values = groups.get(key) or {}
        target = values.get("rhythm_20_percent_target_met")
        target_text = f"{str(target).lower()} (provisional)" if key == "all_diagnostic" else str(target).lower()
        lines.append(
            f"| {key} | {values.get('case_count', 0)} | {values.get('gate_status', '-')} | "
            f"{fmt(values.get('baseline_mean_fixed_total_rhythm'))} | {fmt(values.get('final_score_mean_fixed_total_rhythm'))} | "
            f"{fmt(values.get('final_score_improvement_vs_baseline_percent'), 3)}% | "
            f"{fmt(values.get('final_score_mean_pitch_f1'))} | {fmt(values.get('final_score_mean_chord_retention'))} | {target_text} |"
        )
    lines += [
        "",
        "## All-diagnostic stage attribution",
        "",
        "| stage | fixed-total rhythm | improvement vs baseline |",
        "|---|---:|---:|",
        f"| baseline | {fmt(summary['baseline_mean_fixed_total_rhythm'])} | - |",
        f"| continuous seconds-to-beats mapping | {fmt(summary['continuous_mapping_mean_fixed_total_rhythm'])} | {fmt(summary['continuous_mapping_improvement_vs_baseline_percent'], 3)}% |",
        f"| 480 PPQ performance MIDI | {fmt(summary['performance_mapping_mean_fixed_total_rhythm'])} | {fmt(summary['performance_mapping_improvement_vs_baseline_percent'], 3)}% |",
        f"| final Score/MIDI/SVG chain | {fmt(summary['final_score_mean_fixed_total_rhythm'])} | {fmt(summary['final_score_improvement_vs_baseline_percent'], 3)}% |",
        f"| performance-to-final notation delta | {fmt(summary['notation_delta_mean_fixed_total_rhythm'])} | - |",
        f"| final pitch F1 | {fmt(summary['final_score_mean_pitch_f1'])} | - |",
        f"| final chord retention | {fmt(summary['final_score_mean_chord_retention'])} | - |",
        "",
        "## By category",
        "",
        "| category | n | performance rhythm | final rhythm | final pitch F1 | final chord retention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in report["by_category"].items():
        lines.append(
            f"| {category} | {values['count']} | {fmt(values.get('performance_mapping_mean_fixed_total_rhythm'))} | {fmt(values.get('final_score_mean_fixed_total_rhythm'))} | {fmt(values.get('final_score_mean_pitch_f1'))} | {fmt(values.get('final_score_mean_chord_retention'))} |"
        )
    lines += [
        "",
        "## Per case",
        "",
        "| case | category | route | performance rhythm | final rhythm | notation delta | pitch F1 | chord retention |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["cases"]:
        if item.get("status") != "success":
            lines.append(f"| {item['id']} | {item.get('category')} | failed | failed | failed | failed | failed | failed |")
            continue
        values = {
            key: _metric_value(item.get(key) or {}, "rhythm_error", "mean_fixed_total_assignment_rhythm_error_quarter")
            for key in ("performance_mapping", "final_score", "notation_delta_performance_to_final")
        }
        pitch = _metric_value(item["final_score"], "pitch_f1", "f1")
        chord = _metric_value(item["final_score"], "chord_retention", "retention")
        route = "fallback" if _fallback_used(item) else "strict"
        lines.append(
            f"| {item['id']} | {item.get('category')} | {route} | {fmt(values['performance_mapping'])} | {fmt(values['final_score'])} | {fmt(values['notation_delta_performance_to_final'])} | {fmt(pitch)} | {fmt(chord)} |"
        )
    lines += [
        "",
        "The continuous mapping row is computed from the immutable recognized events before 480-PPQ rounding. The performance row includes that rounding and tempo-map MIDI encoding. The notation delta compares the performance MIDI with the final score MIDI after MuseScore import, MusicXML normalization, jianpu serialization, and LilyPond rendering.",
        "The fallback population is scored only after standardizing MuseScore MusicXML without source performance metadata. Each original failed strict manifest remains alongside manifest.oracle.json. Fallback rows are diagnostic attribution, not strict production evidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _repair_existing_case_manifest(output_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Repair v1 reused records so fallback manifests remain auditable.

    ``--allow-existing`` is intentionally resumable.  Older partial oracle
    runs embedded the failed strict manifest in their root manifest; repair
    that metadata before reusing the case instead of forcing an expensive
    MuseScore rerun.
    """

    repaired = dict(record)
    case_id = str(record.get("id"))
    case_output = output_root / case_id
    root_path = case_output / "manifest.json"
    if not root_path.is_file():
        return repaired
    try:
        root_manifest = _load(root_path)
    except (OSError, ValueError, TypeError):
        return repaired
    fallback = record.get("diagnostic_tempo_fallback")
    fallback_used = isinstance(fallback, Mapping) and bool(fallback.get("used"))
    if fallback_used:
        oracle_rel = str(fallback.get("oracle_service_manifest") or "service_output/manifest.oracle.json").replace("\\", "/")
        service_rel = oracle_rel if (case_output / oracle_rel).is_file() else "service_output/manifest.json"
    else:
        result = root_manifest.get("result") if isinstance(root_manifest, Mapping) else {}
        result_rel = result.get("manifest") if isinstance(result, Mapping) else None
        service_rel = str(result_rel or "service_output/manifest.json").replace("\\", "/")
        if not (case_output / service_rel).is_file():
            service_rel = "service_output/manifest.json"
    service_path = case_output / service_rel
    if not service_path.is_file():
        return repaired
    service_payload = _load(service_path)
    root_manifest["service_manifest"] = service_rel
    root_manifest["service_manifest_payload"] = service_payload
    if isinstance(root_manifest.get("result"), Mapping):
        root_manifest["result"] = {**root_manifest["result"], "manifest": service_rel}
    _write_json(root_path, root_manifest)
    artifacts = dict(repaired.get("artifacts") or {})
    artifacts["service_manifest"] = str(service_path.relative_to(output_root)).replace("\\", "/")
    repaired["artifacts"] = artifacts
    return repaired


def _refresh_existing_alignment_diagnostic(output_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Fill diagnostics for fallback records created before the probe helper."""

    repaired = dict(record)
    fallback = record.get("diagnostic_tempo_fallback")
    if not isinstance(fallback, Mapping) or not bool(fallback.get("used")):
        return repaired
    case_id = str(record.get("id"))
    case_output = output_root / case_id
    existing_rel = str(fallback.get("alignment_diagnostics") or "").replace("\\", "/")
    existing_path = output_root / existing_rel if existing_rel else None
    if existing_path is not None and existing_path.is_file():
        try:
            existing_payload = _load(existing_path)
        except (OSError, ValueError, TypeError):
            existing_payload = {}
        if (
            isinstance(existing_payload, Mapping)
            and "source_to_musicxml_alignment_probe" in existing_payload
            and "source_tempo_origin_probe" in existing_payload
        ):
            return repaired
    artifacts = record.get("artifacts") or {}
    performance_rel = str(artifacts.get("performance_midi") or "").replace("\\", "/")
    performance_path = output_root / performance_rel
    grid_path = case_output / "raw" / "beat_grid.json"
    if not performance_path.is_file() or not grid_path.is_file():
        return repaired
    diagnostic = _alignment_diagnostics(
        case={"id": case_id},
        case_output=case_output,
        performance_path=performance_path,
        oracle_grid=_load(grid_path),
        cause=RuntimeError(str(fallback.get("cause") or "strict standardization failed")),
    )
    diagnostic_path = case_output / "alignment_diagnostics.json"
    _write_json(diagnostic_path, diagnostic)
    diagnostic_rel = str(diagnostic_path.relative_to(output_root)).replace("\\", "/")
    repaired["diagnostic_tempo_fallback"] = {**fallback, "diagnosis": diagnostic.get("diagnosis"), "alignment_diagnostics": diagnostic_rel}
    repaired_artifacts = dict(artifacts)
    repaired_artifacts["alignment_diagnostics"] = diagnostic_rel
    repaired["artifacts"] = repaired_artifacts
    root_path = case_output / "manifest.json"
    if root_path.is_file():
        root_manifest = _load(root_path)
        root_manifest["alignment_diagnostics"] = diagnostic_rel
        _write_json(root_path, root_manifest)
    return repaired


def run(batch_root: Path, output_root: Path, *, allow_existing: bool = False) -> dict[str, Any]:
    batch_root = batch_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(f"refusing to overwrite non-empty oracle output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    production_report = _load(batch_root / "production-report-v3.json")
    selected_payload = _load(batch_root / "raw-selection.json")
    selected = selected_payload.get("selected")
    selected_ids = [str(item.get("case_id")) for item in selected if isinstance(item, Mapping)] if isinstance(selected, list) else []
    if len(selected_ids) != 30 or len(set(selected_ids)) != 30:
        raise ValueError("v3 raw selection must contain exactly 30 unique cases")
    evaluator_cases = evaluator.get("cases")
    if not isinstance(evaluator_cases, list):
        raise ValueError("v3 evaluator has no cases list")
    evaluator_by_id = {str(item.get("id")): item for item in evaluator_cases if isinstance(item, Mapping) and item.get("id")}
    missing = [case_id for case_id in selected_ids if case_id not in evaluator_by_id]
    if missing:
        raise ValueError(f"v3 evaluator is missing selected cases: {', '.join(missing)}")
    cases = [evaluator_by_id[case_id] for case_id in selected_ids]
    baseline = _baseline_index(batch_root)
    existing_records: dict[str, Mapping[str, Any]] = {}
    existing_report_path = output_root / "oracle-report.json"
    if allow_existing and existing_report_path.is_file():
        previous = _load(existing_report_path)
        if isinstance(previous, Mapping) and isinstance(previous.get("cases"), list):
            existing_records = {
                str(item.get("id")): item
                for item in previous["cases"]
                if isinstance(item, Mapping) and item.get("id") and item.get("status") == "success"
            }
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id"))
        if case_id in existing_records and (output_root / case_id / "manifest.json").is_file():
            reused = _repair_existing_case_manifest(output_root, existing_records[case_id])
            records.append(_refresh_existing_alignment_diagnostic(output_root, reused))
            print(json.dumps({"case": case_id, "status": "reused"}, ensure_ascii=False), flush=True)
            continue
        try:
            records.append(_case_record(batch_root=batch_root, output_root=output_root, case=case, baseline=baseline[case_id]))
            print(json.dumps({"case": case_id, "status": "success"}, ensure_ascii=False), flush=True)
        except Exception as exc:
            records.append({"id": case_id, "category": case.get("category"), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"case": case_id, "status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
    summary = _aggregate(records, production_report)
    report = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "production_path_untouched": True,
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "source_v3_evaluator": str((batch_root / "evaluator-report-v3.json").resolve()),
        "source_v3_report": str((batch_root / "production-report-v3.json").resolve()),
        "source_recognition_immutable": True,
        "reference_grid_used_only_for_diagnostic": True,
        "cases": records,
        "summary": summary,
        "by_category": _by_category(records),
    }
    _write_json(output_root / "oracle-report.json", report)
    _write_markdown(report, output_root / "oracle-report.md")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-existing", action="store_true", help="allow a pre-existing empty/partial diagnostic root")
    args = parser.parse_args(argv)
    report = run(args.batch_root, args.output_root, allow_existing=args.allow_existing)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
