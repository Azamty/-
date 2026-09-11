"""Reference-free integer downbeat-phase diagnostic for Beat Transformer.

The official Beat Transformer pilot already contains the only model outputs used
by this diagnostic.  This module keeps those beat events byte-for-byte and only
tests an integer phase over the fixed DBN period.  Audio observations are used
to decide whether a phase is stable enough to change; annotations are opened
only by :func:`score_development` after ``raw-decision-freeze.json`` exists.

This is a review artifact, not a production routing helper.  In particular it
does not search tempo, offsets, meters, beat times, or model parameters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These helpers are intentionally shared with the earlier reference-free
# refinement pilot.  _select_downbeats is not called: that pilot searches
# meters and adds a meter prior, both of which are forbidden here.
from scripts.pilot_beatnet_observable_refinement import (  # noqa: E402
    _audio_features,
    _contrast,
    _sample_envelope,
)


BT_ROOT = ROOT / ".artifacts" / "review" / "beat-transformer-official-pilot-v1"
RECOGNITION_ROOT = ROOT / ".artifacts" / "review" / "production-acceptance-v4-origin-fix"
DEFAULT_INPUT_MANIFEST = BT_ROOT / "frozen-input-manifest.json"
DEFAULT_RAW_DIR = BT_ROOT / "raw"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beat-transformer-integer-phase-pilot-v2"
DEFAULT_EVALUATION = BT_ROOT / "evaluation.json"
FEATURE_HELPER_PATH = ROOT / "scripts" / "pilot_beatnet_observable_refinement.py"

PHASE_WEIGHTS: dict[str, float] = {
    "onset_contrast": 0.40,
    "low_frequency_contrast": 0.25,
    "bass_alignment": 0.20,
    "bar_stability": 0.15,
}
MIN_COMPLETE_BARS = 3
CROP_MIN_COMPLETE_BARS = 2
MIN_MARGIN = 0.10
TOLERANCE_SEC = 0.07
SCHEMA_VERSION = "bt_integer_phase_diagnostic_v2"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def _beat_output_invariant(
    frozen_values: Sequence[float], frozen_sha256: str, actual_values: Sequence[float]
) -> dict[str, Any]:
    """Compare actual output against the separately frozen input manifest."""

    frozen = [float(value) for value in frozen_values]
    actual = [float(value) for value in actual_values]
    actual_sha256 = _sha256_bytes(np.asarray(actual, dtype=np.float64).tobytes())
    values_equal = actual == frozen
    count_equal = len(actual) == len(frozen)
    hash_equal = actual_sha256 == str(frozen_sha256)
    return {
        "frozen_input_sha256": str(frozen_sha256),
        "actual_output_sha256": actual_sha256,
        "frozen_input_count": len(frozen),
        "actual_output_count": len(actual),
        "float_values_equal_to_frozen_input": values_equal,
        "count_equal_to_frozen_input": count_equal,
        "hash_equal_to_frozen_input": hash_equal,
        "invariant_passed": values_equal and count_equal and hash_equal,
    }


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _local_contrast(values: Sequence[float], selected: Sequence[int], segment: Sequence[int]) -> float:
    """Use the earlier pilot's contrast definition on one complete-bar span."""

    segment_set = set(int(index) for index in segment)
    selected_local = {int(index) for index in selected if int(index) in segment_set}
    if not selected_local:
        return 0.0
    local_values = [float(values[index]) for index in segment]
    local_positions = {position for position, index in enumerate(segment) if index in selected_local}
    return float(_contrast(local_values, local_positions))


def _median_interval(beats: Sequence[float]) -> float:
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    return statistics.median(intervals) if intervals else 0.5


def _bass_alignment(
    downbeat_times: Sequence[float], bass_onsets: Sequence[float], interval: float
) -> float:
    """Return the fixed-pilot alignment score without reweighting missing bass."""

    if not downbeat_times or not bass_onsets or interval <= 0:
        return 0.0
    distances = [min(abs(float(onset) - float(beat)) for beat in downbeat_times) / interval for onset in bass_onsets]
    return _clamp(1.0 - statistics.fmean(min(value, 1.0) for value in distances), 0.0, 1.0)


def _complete_bar_starts(length: int, period: int, phase: int) -> list[int]:
    return [start for start in range(int(phase), length, int(period)) if start + int(period) < length]


def _bar_span(starts: Sequence[int], period: int, length: int) -> list[int]:
    if not starts:
        return []
    first = int(starts[0])
    last = min(length, int(starts[-1]) + int(period))
    return list(range(first, last))


def _bar_time_window(starts: Sequence[int], period: int, beats: Sequence[float]) -> tuple[float, float] | None:
    """Return the half-open time window covered by complete bars."""

    if not starts:
        return None
    first = int(starts[0])
    end_index = int(starts[-1]) + int(period)
    if first < 0 or end_index >= len(beats):
        return None
    return float(beats[first]), float(beats[end_index])


def _phase_record(
    *,
    phase: int,
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    bar_numbers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Score one integer phase over the requested complete bars."""

    starts_all = _complete_bar_starts(len(beats), period, phase)
    if bar_numbers is None:
        starts = starts_all
    else:
        starts = [starts_all[index] for index in bar_numbers if 0 <= int(index) < len(starts_all)]
    span = _bar_span(starts, period, len(beats))
    selected = list(starts)
    interval = _median_interval(beats)
    accents = [float(onset_values[index]) + 0.5 * float(low_values[index]) for index in starts]
    time_window = _bar_time_window(starts, period, beats)
    bass_in_window = (
        [float(onset) for onset in bass_onsets if time_window[0] <= float(onset) < time_window[1]]
        if time_window is not None
        else []
    )
    components = {
        "onset_contrast": _local_contrast(onset_values, selected, span),
        "low_frequency_contrast": _local_contrast(low_values, selected, span),
        "bass_alignment": _bass_alignment([beats[index] for index in starts], bass_in_window, interval),
        "bar_stability": max(0.0, 1.0 - statistics.pstdev(accents)) if len(accents) > 1 else 0.5,
    }
    score = sum(PHASE_WEIGHTS[key] * float(components[key]) for key in PHASE_WEIGHTS)
    return {
        "phase_index": int(phase),
        "complete_bar_count": len(starts_all),
        "scored_bar_count": len(starts),
        "bar_start_indices": [int(index) for index in starts],
        "bar_time_window_sec": list(time_window) if time_window is not None else None,
        "bass_onset_count_in_window": len(bass_in_window),
        "components": {key: float(value) for key, value in components.items()},
        "score_higher_is_better": float(score),
    }


def _unique_winner(records: Sequence[Mapping[str, Any]]) -> tuple[int | None, float | None, float | None]:
    if not records:
        return None, None, None
    ordered = sorted(records, key=lambda item: (-float(item["score_higher_is_better"]), int(item["phase_index"])))
    top = float(ordered[0]["score_higher_is_better"])
    second = float(ordered[1]["score_higher_is_better"]) if len(ordered) > 1 else None
    if second is not None and math.isclose(top, second, rel_tol=0.0, abs_tol=1e-12):
        return None, top, second
    return int(ordered[0]["phase_index"]), top, second


def _eligible_phase_records(
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    minimum_complete_bars: int,
) -> list[dict[str, Any]]:
    records = [
        _phase_record(
            phase=phase,
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=period,
        )
        for phase in range(period)
    ]
    return [record for record in records if record["complete_bar_count"] >= minimum_complete_bars]


def _crop_equivariance_check(
    *,
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    original_phase: int,
    selected_phase: int,
) -> dict[str, Any]:
    """Check index-phase covariance after deleting the first beat.

    This deliberately does not claim robustness to a real PCM crop. A leading
    index deletion translates a phase by ``-crop_start`` modulo the fixed
    period. The check is weaker on short fixtures (two complete bars) because
    the full decision still requires three; it prevents the check from being
    vacuous for the 13-beat cases.
    """

    crop_start = 1
    crop_end = len(beats)
    if len(beats) - crop_start < (CROP_MIN_COMPLETE_BARS * period + 1):
        return {"available": False, "passed": False, "reason": "insufficient_beats_for_crop"}
    crop_beats = list(beats[crop_start:crop_end])
    crop_onset = list(onset_values[crop_start:crop_end])
    crop_low = list(low_values[crop_start:crop_end])
    crop_original = (int(original_phase) - crop_start) % period
    crop_result = _select_integer_phase(
        beats=crop_beats,
        onset_values=crop_onset,
        low_values=crop_low,
        bass_onsets=bass_onsets,
        period=period,
        original_phase=crop_original,
        minimum_complete_bars=CROP_MIN_COMPLETE_BARS,
        check_crop=False,
    )
    expected = (int(selected_phase) - crop_start) % period
    passed = crop_result["selected_phase_index"] == expected and crop_result["decision"] != "abstain"
    return {
        "available": True,
        "passed": bool(passed),
        "crop_start_beat_index": crop_start,
        "crop_end_beat_index_exclusive": crop_end,
        "expected_phase_index": expected,
        "cropped_original_phase_index": crop_original,
        "cropped_selected_phase_index": crop_result["selected_phase_index"],
        "cropped_decision": crop_result["decision"],
        "cropped_reason": crop_result["reason"],
    }


def _select_integer_phase(
    *,
    beats: Sequence[float],
    onset_values: Sequence[float],
    low_values: Sequence[float],
    bass_onsets: Sequence[float],
    period: int,
    original_phase: int | None,
    minimum_complete_bars: int = MIN_COMPLETE_BARS,
    check_crop: bool = True,
) -> dict[str, Any]:
    """Select one fixed-period integer phase, or abstain to the DBN phase."""

    if len(beats) != len(onset_values) or len(beats) != len(low_values):
        raise ValueError("beats and all audio feature samples must have equal length")
    eligible = _eligible_phase_records(
        beats, onset_values, low_values, bass_onsets, period, minimum_complete_bars
    )
    all_records = [
        _phase_record(
            phase=phase,
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=period,
        )
        for phase in range(period)
    ]
    winner, top_score, second_score = _unique_winner(eligible)
    margin = (top_score - second_score) if top_score is not None and second_score is not None else None

    half_records: dict[str, list[dict[str, Any]]] = {"first": [], "last": []}
    for record in eligible:
        phase = int(record["phase_index"])
        bars = _complete_bar_starts(len(beats), period, phase)
        # Use disjoint end sets. With three bars this scores bar 0 against
        # bar 2 and intentionally leaves the middle bar out of both halves.
        half_size = len(bars) // 2
        if half_size < 1:
            continue
        half_records["first"].append(
            _phase_record(
                phase=phase,
                beats=beats,
                onset_values=onset_values,
                low_values=low_values,
                bass_onsets=bass_onsets,
                period=period,
                bar_numbers=list(range(half_size)),
            )
        )
        half_records["last"].append(
            _phase_record(
                phase=phase,
                beats=beats,
                onset_values=onset_values,
                low_values=low_values,
                bass_onsets=bass_onsets,
                period=period,
                bar_numbers=list(range(len(bars) - half_size, len(bars))),
            )
        )
    first_winner, first_top, first_second = _unique_winner(half_records["first"])
    last_winner, last_top, last_second = _unique_winner(half_records["last"])

    reason = "selected"
    proposed_phase: int | None = winner
    if original_phase is None:
        proposed_phase = None
        reason = "missing_original_phase"
    elif winner is None:
        proposed_phase = None
        reason = "no_unique_global_winner"
    elif second_score is None:
        proposed_phase = None
        reason = "insufficient_competing_phases"
    elif margin < MIN_MARGIN:
        proposed_phase = None
        reason = "winner_margin_below_0_10"
    elif first_winner is None or last_winner is None or first_winner != last_winner or first_winner != winner:
        proposed_phase = None
        reason = "first_last_half_winner_disagreement"

    crop = {"available": False, "passed": None, "reason": "not_run"}
    if proposed_phase is not None and check_crop and original_phase is not None:
        crop = _crop_equivariance_check(
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=period,
            original_phase=int(original_phase),
            selected_phase=int(proposed_phase),
        )
        if not crop["passed"]:
            proposed_phase = None
            reason = "crop_equivariance_failed"

    if proposed_phase is None:
        selected_phase = int(original_phase) if original_phase is not None else None
        decision = "abstain"
    elif int(proposed_phase) == int(original_phase):
        selected_phase = int(proposed_phase)
        decision = "keep_original"
    else:
        selected_phase = int(proposed_phase)
        decision = "changed"

    return {
        "period_beats": int(period),
        "original_phase_index": int(original_phase) if original_phase is not None else None,
        "selected_phase_index": selected_phase,
        "proposed_phase_index": proposed_phase,
        "decision": decision,
        "reason": reason,
        "minimum_complete_bars": int(minimum_complete_bars),
        "eligible_phase_count": len(eligible),
        "phase_records": all_records,
        "winner_phase_index": winner,
        "winner_score": top_score,
        "runner_up_score": second_score,
        "winner_margin": margin,
        "half_winners": {
            "first": first_winner,
            "last": last_winner,
            "first_score": first_top,
            "first_runner_up_score": first_second,
            "last_score": last_top,
            "last_runner_up_score": last_second,
        },
        "half_phase_records": half_records,
        "crop_equivariance": crop,
    }


def _derive_joint_period(downbeats: np.ndarray, case_id: str) -> dict[str, Any]:
    """Derive and strictly validate the period from joint DBN positions."""

    if downbeats.ndim != 2 or downbeats.shape[1] < 2:
        raise ValueError(f"unexpected joint DBN position schema for {case_id}")
    raw_positions = np.asarray(downbeats[:, 1], dtype=float)
    rounded = np.rint(raw_positions).astype(int)
    if np.any(~np.isfinite(raw_positions)) or np.any(raw_positions != rounded) or np.any(rounded < 1):
        raise ValueError(f"joint DBN positions are not positive integers for {case_id}")
    maximum = int(np.max(rounded))
    candidates: list[int] = []
    for period in range(2, max(12, maximum) + 1):
        expected = ((int(rounded[0]) - 1 + np.arange(len(rounded))) % period) + 1
        if np.array_equal(expected, rounded):
            candidates.append(period)
    if len(candidates) != 1:
        raise ValueError(f"joint DBN positions do not identify one period for {case_id}: {rounded.tolist()}")
    return {
        "period_beats": int(candidates[0]),
        "position_values": rounded.tolist(),
        "candidate_periods": candidates,
        "validation": "exact contiguous cyclic position sequence",
    }


def _map_joint_downbeats_to_beat_phase(
    beats: np.ndarray, downbeats: np.ndarray, period: int, case_id: str
) -> dict[str, Any]:
    """Map joint DBN downbeat times to beat-event indices by nearest time.

    The phase is the mode of mapped ``nearest_index % period`` values. This is
    intentionally independent of the joint event row number, which may differ
    from the Beat Transformer beat-event array length.
    """

    positions = np.rint(np.asarray(downbeats[:, 1], dtype=float)).astype(int)
    downbeat_rows = np.flatnonzero(positions == 1)
    intervals = np.diff(beats)
    intervals = intervals[intervals > 0]
    median_interval = float(np.median(intervals)) if len(intervals) else 0.5
    max_allowed = max(0.5 * median_interval, 0.08)
    mapped: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    phase_votes: dict[int, list[float]] = {phase: [] for phase in range(period)}
    for row in downbeat_rows:
        time_sec = float(downbeats[row, 0])
        nearest = int(np.argmin(np.abs(beats - time_sec)))
        deviation = abs(float(beats[nearest]) - time_sec)
        record = {
            "joint_row_index": int(row),
            "joint_time_sec": time_sec,
            "nearest_beat_index": nearest,
            "nearest_beat_time_sec": float(beats[nearest]),
            "deviation_sec": float(deviation),
        }
        if deviation <= max_allowed:
            record["phase_index"] = int(nearest % period)
            mapped.append(record)
            phase_votes[int(nearest % period)].append(float(deviation))
        else:
            unmapped.append(record)
    ranked = sorted(
        ((phase, deviations) for phase, deviations in phase_votes.items() if deviations),
        key=lambda item: (-len(item[1]), float(np.median(item[1])), int(item[0])),
    )
    phase = int(ranked[0][0]) if ranked else None
    chosen_deviations = ranked[0][1] if ranked else []
    return {
        "case_id": case_id,
        "rule": "nearest beat-event index for each position-1 joint time, then mode of index modulo derived period; ties median deviation then lower phase",
        "position_one_joint_row_count": int(len(downbeat_rows)),
        "mapped_count": int(len(mapped)),
        "unmapped_count": int(len(unmapped)),
        "mapping_threshold_sec": float(max_allowed),
        "median_beat_interval_sec": median_interval,
        "mapped": mapped,
        "unmapped": unmapped,
        "phase_votes": {str(phase_index): [float(value) for value in deviations] for phase_index, deviations in phase_votes.items()},
        "phase_index": phase,
        "phase_vote_count": len(chosen_deviations),
        "phase_consistency": (len(chosen_deviations) / len(mapped)) if mapped else 0.0,
        "mapped_deviation_mean_sec": float(np.mean(chosen_deviations)) if chosen_deviations else None,
        "mapped_deviation_max_sec": float(np.max(chosen_deviations)) if chosen_deviations else None,
        "phase_mapping_available": bool(phase is not None),
    }


def _raw_case(raw_path: Path, case_id: str) -> dict[str, Any]:
    with np.load(raw_path, allow_pickle=False) as raw:
        beats = np.asarray(raw["beat_events"], dtype=np.float64).copy()
        downbeats = np.asarray(raw["downbeat_events"], dtype=np.float64).copy()
    if beats.ndim != 1 or downbeats.ndim != 2 or downbeats.shape[1] < 2:
        raise ValueError(f"unexpected Beat Transformer raw schema for {case_id}")
    period_info = _derive_joint_period(downbeats, case_id)
    mapping = _map_joint_downbeats_to_beat_phase(beats, downbeats, int(period_info["period_beats"]), case_id)
    positions = np.rint(downbeats[:, 1]).astype(int)
    downbeat_rows = np.flatnonzero(positions == 1)
    if len(downbeat_rows) == 0:
        raise ValueError(f"raw DBN output has no position-1 row for {case_id}")
    return {
        "case_id": case_id,
        "period_beats": int(period_info["period_beats"]),
        "period_derivation": period_info,
        "beat_events": beats,
        "downbeat_events": downbeats,
        "original_downbeat_times": [float(downbeats[index, 0]) for index in downbeat_rows],
        "original_phase_index": mapping["phase_index"],
        "joint_to_beat_mapping": mapping,
        "beat_event_count": int(len(beats)),
        "downbeat_event_count": int(len(downbeats)),
        "beat_events_sha256": _sha256_bytes(beats.tobytes()),
        "downbeat_events_sha256": _sha256_bytes(downbeats.tobytes()),
    }


def _extract_bass_onsets(recognition_path: Path, audio_path: Path) -> dict[str, Any]:
    """Read only raw note fields needed for bass onset evidence.

    The recognition artifact also contains beat grids and other derived data.
    They are deliberately not accessed here.  Chord notes at the same onset
    are collapsed to one six-decimal onset, matching the existing onset-group
    contract.
    """

    if not recognition_path.is_file():
        return {
            "available": False,
            "reason": "missing_recognition",
            "path": str(recognition_path),
            "sha256": None,
            "audio_path_in_recognition": None,
            "audio_path_matches_frozen_input": None,
            "raw_note_count": 0,
            "bass_candidate_count": 0,
            "bass_onsets": [],
            "used_fields": ["start_sec", "midi", "instrument_group", "voice_id", "is_drum"],
        }
    raw_bytes = recognition_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError(f"recognition notes must be a list: {recognition_path}")
    candidates: list[float] = []
    for note in notes:
        if not isinstance(note, Mapping) or bool(note.get("is_drum")):
            continue
        try:
            onset = float(note["start_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        midi_value: int | None
        try:
            midi_value = int(note["midi"])
        except (KeyError, TypeError, ValueError):
            midi_value = None
        instrument_group = str(note.get("instrument_group") or "").lower()
        voice_id = str(note.get("voice_id") or "").lower()
        explicit_bass = "bass" in instrument_group or "bass" in voice_id
        low_non_drum = midi_value is not None and midi_value < 48
        if explicit_bass or low_non_drum:
            candidates.append(round(onset, 6))
    bass_onsets = sorted(set(candidates))
    provenance = payload.get("provenance")
    audio_in_recognition = provenance.get("source_audio") if isinstance(provenance, Mapping) else None
    return {
        "available": True,
        "reason": "raw_note_fields_only",
        "path": str(recognition_path),
        "sha256": _sha256_bytes(raw_bytes),
        "audio_path_in_recognition": str(audio_in_recognition) if audio_in_recognition else None,
        "audio_path_matches_frozen_input": (
            Path(str(audio_in_recognition)).resolve() == audio_path.resolve() if audio_in_recognition else None
        ),
        "raw_note_count": len(notes),
        "bass_candidate_count": len(candidates),
        "bass_onset_count": len(bass_onsets),
        "bass_onsets": bass_onsets,
        "used_fields": ["start_sec", "midi", "instrument_group", "voice_id", "is_drum"],
        "deduplication": "sorted unique round(start_sec, 6), including chord/onset collapse",
    }


def _input_case_manifest(
    case: Mapping[str, Any], raw_dir: Path, raw_freeze_cases: Mapping[str, Mapping[str, Any]], recognition_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(case["id"])
    audio_path = Path(str(case["path"]))
    raw_path = raw_dir / f"{case_id}.npz"
    raw_json_path = raw_dir / f"{case_id}.json"
    if not audio_path.is_file() or not raw_path.is_file() or not raw_json_path.is_file():
        raise FileNotFoundError(f"missing frozen input/raw for {case_id}")
    if _sha256(audio_path) != str(case.get("sha256")) or _sha256(audio_path) != str(case.get("registry_sha256")):
        raise ValueError(f"frozen input audio hash mismatch for {case_id}")
    expected_raw = raw_freeze_cases.get(case_id)
    if expected_raw is None:
        raise ValueError(f"raw freeze manifest has no case {case_id}")
    actual_raw_sha = _sha256(raw_path)
    actual_metadata_sha = _sha256(raw_json_path)
    if actual_raw_sha != str(expected_raw.get("npz_sha256")) or actual_metadata_sha != str(expected_raw.get("metadata_sha256")):
        raise ValueError(f"upstream raw freeze hash mismatch for {case_id}")
    raw = _raw_case(raw_path, case_id)
    recognition = _extract_bass_onsets(recognition_root / case_id / "raw" / "recognition.json", audio_path)
    manifest = {
        "id": case_id,
        "category": str(case.get("category", "unknown")),
        "audio_path": str(audio_path),
        "audio_bytes": audio_path.stat().st_size,
        "audio_sha256": _sha256(audio_path),
        "raw_npz_path": str(raw_path),
        "raw_npz_bytes": raw_path.stat().st_size,
        "raw_npz_sha256": actual_raw_sha,
        "raw_metadata_path": str(raw_json_path),
        "raw_metadata_bytes": raw_json_path.stat().st_size,
        "raw_metadata_sha256": actual_metadata_sha,
        "period_beats": int(raw["period_beats"]),
        "period_derivation": raw["period_derivation"],
        "beat_event_count": raw["beat_event_count"],
        "downbeat_event_count": raw["downbeat_event_count"],
        "beat_events_sha256": raw["beat_events_sha256"],
        "downbeat_events_sha256": raw["downbeat_events_sha256"],
        "original_phase_index": raw["original_phase_index"],
        "joint_to_beat_mapping": raw["joint_to_beat_mapping"],
        "beat_event_values": [float(value) for value in raw["beat_events"]],
        "recognition": recognition,
        "upstream_expected_hashes": {
            "audio_sha256": str(case.get("sha256")),
            "audio_registry_sha256": str(case.get("registry_sha256")),
            "raw_npz_sha256": str(expected_raw.get("npz_sha256")),
            "raw_metadata_sha256": str(expected_raw.get("metadata_sha256")),
        },
    }
    return manifest, raw


def _reference_f1(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    used: set[int] = set()
    errors: list[float] = []
    for value in predicted:
        candidates = [
            (abs(float(value) - float(item)), index)
            for index, item in enumerate(reference)
            if index not in used and abs(float(value) - float(item)) <= TOLERANCE_SEC
        ]
        if candidates:
            error, index = min(candidates)
            used.add(index)
            errors.append(error)
    tp = len(errors)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": len(predicted),
        "reference": len(reference),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_error_sec": statistics.fmean(errors) if errors else None,
    }


def _annotation_times(path: Path) -> tuple[list[float], list[float]]:
    payload = _load_json(path)
    records = list(payload["beat_grid"]["beats"])
    beats = [float(item["time_sec"]) for item in records]
    downbeats = [float(item["time_sec"]) for item in records if bool(item.get("downbeat"))]
    return beats, downbeats


def _candidate_downbeats(raw: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    """Build the actual output grid and record its source explicitly."""

    if decision["decision"] != "changed":
        values = [float(value) for value in raw["original_downbeat_times"]]
        source = "joint_dbn_downbeat_events"
    else:
        period = int(decision["period_beats"])
        phase = int(decision["selected_phase_index"])
        beats = np.asarray(raw["beat_events"], dtype=np.float64)
        values = [float(value) for value in beats[phase::period]]
        source = "beat_events_integer_phase_slice"
    return {
        "times": values,
        "source": source,
        "count": len(values),
        "sha256": _sha256_bytes(np.asarray(values, dtype=np.float64).tobytes()),
    }


def _score_development(
    *,
    evaluation_path: Path,
    input_cases: Sequence[Mapping[str, Any]],
    raw_cases: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Open labels only after the raw decision freeze has been written."""

    evaluation = _load_json(evaluation_path)
    evaluation_by_id = {str(item["id"]): item for item in evaluation["cases"]}
    rows: list[dict[str, Any]] = []
    for input_case in input_cases:
        case_id = str(input_case["id"])
        entry = evaluation_by_id[case_id]
        annotation_path = Path(str(entry["annotation_path"]))
        reference_beats, reference_downbeats = _annotation_times(annotation_path)
        raw = raw_cases[case_id]
        decision = decisions[case_id]
        frozen_beat_values = [float(value) for value in input_case["beat_event_values"]]
        predicted_beats = [float(value) for value in raw["beat_events"]]
        beat_output = _beat_output_invariant(frozen_beat_values, str(input_case["beat_events_sha256"]), predicted_beats)
        output_grid = _candidate_downbeats(raw, decision)
        predicted_downbeats = output_grid["times"]
        original_downbeats = [float(value) for value in raw["original_downbeat_times"]]
        original_beat_metric = _reference_f1(reference_beats, predicted_beats)
        original_downbeat_metric = _reference_f1(reference_downbeats, original_downbeats)
        candidate_downbeat_metric = _reference_f1(reference_downbeats, predicted_downbeats)
        rows.append(
            {
                "id": case_id,
                "category": input_case.get("category"),
                "annotation_path": str(annotation_path),
                "annotation_sha256": _sha256(annotation_path),
                "reference_meter": entry.get("reference_meter"),
                "original": {
                    "beat": original_beat_metric,
                    "downbeat": original_downbeat_metric,
                },
                "candidate": {
                    "downbeat": candidate_downbeat_metric,
                    "predicted_downbeat_times": predicted_downbeats,
                    "output_grid": output_grid,
                    "output_source_is_phase_changed_candidate": output_grid["source"] == "beat_events_integer_phase_slice",
                },
                "beat_preserved": bool(beat_output["invariant_passed"]),
                "beat_count_preserved": bool(beat_output["count_equal_to_frozen_input"]),
                "beat_output": beat_output,
                "joint_vs_beat_grid": raw["joint_to_beat_mapping"],
                "decision": decision["decision"],
            }
        )
    candidate_db = [float(row["candidate"]["downbeat"]["f1"]) for row in rows]
    non_asap = [row for row in rows if not row["id"].startswith("asap-")]
    original_perfect = [row for row in rows if math.isclose(float(row["original"]["downbeat"]["f1"]), 1.0, rel_tol=0.0, abs_tol=1e-12)]
    original_perfect_preserved = all(
        float(row["candidate"]["downbeat"]["f1"]) + 1e-12 >= float(row["original"]["downbeat"]["f1"])
        for row in original_perfect
    )
    all_beats_exact = all(row["beat_preserved"] and row["beat_count_preserved"] for row in rows)
    summary = {
        "case_count": len(rows),
        "downbeat_f1_macro": statistics.fmean(candidate_db) if candidate_db else None,
        "non_asap_downbeat_f1_macro": statistics.fmean(
            float(row["candidate"]["downbeat"]["f1"]) for row in non_asap
        ) if non_asap else None,
        "original_downbeat_f1_macro": statistics.fmean(
            float(row["original"]["downbeat"]["f1"]) for row in rows
        ) if rows else None,
        "original_downbeat_f1_equals_1_count": len(original_perfect),
        "original_perfect_cases": [row["id"] for row in original_perfect],
        "original_perfect_not_decreased": original_perfect_preserved,
        "all_beats_float_and_count_exact": all_beats_exact,
    }
    gate = {
        "downbeat_macro_at_least_0_75": summary["downbeat_f1_macro"] is not None and summary["downbeat_f1_macro"] >= 0.75,
        "non_asap_downbeat_macro_at_least_0_80": summary["non_asap_downbeat_f1_macro"] is not None and summary["non_asap_downbeat_f1_macro"] >= 0.80,
        "exactly_two_original_downbeat_f1_equals_1": summary["original_downbeat_f1_equals_1_count"] == 2,
        "original_f1_equals_1_not_decreased": bool(original_perfect_preserved),
        "beat_float_and_count_unchanged": all_beats_exact,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "labels_read_after_raw_decision_freeze": True,
        "tolerance_sec": TOLERANCE_SEC,
        "cases": rows,
        "summary": summary,
        "development_gate": gate,
        "development_gate_passed": all(gate.values()),
        "decision": "eligible_for_heldout8" if all(gate.values()) else "stop_after_development8",
    }


def _artifact_manifest(output_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        files.append({"path": str(path.relative_to(output_root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    tracked_sources: list[dict[str, Any]] = []
    for source in (
        ROOT / "scripts" / "pilot_bt_integer_phase_diagnostic.py",
        ROOT / "tests" / "test_pilot_bt_integer_phase_diagnostic.py",
        FEATURE_HELPER_PATH,
    ):
        if source.is_file():
            tracked_sources.append({"path": str(source.relative_to(ROOT)).replace("\\", "/"), "bytes": source.stat().st_size, "sha256": _sha256(source)})
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(output_root),
        "files": files,
        "tracked_sources": tracked_sources,
    }


def run(
    *,
    input_manifest_path: Path = DEFAULT_INPUT_MANIFEST,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_root: Path = DEFAULT_OUTPUT,
    evaluation_path: Path = DEFAULT_EVALUATION,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    input_source = _load_json(input_manifest_path)
    source_cases = list(input_source["cases"])
    raw_freeze_path = BT_ROOT / "raw-freeze-manifest.json"
    raw_freeze_source = _load_json(raw_freeze_path)
    raw_freeze_cases = {str(item["id"]): item for item in raw_freeze_source["files"]}
    feature_helper_sha256 = _sha256(FEATURE_HELPER_PATH)
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "audit_correction_v2_frozen_before_annotation_access",
        "audit_correction": {
            "corrects_v1": True,
            "v1_artifact_is_audio_only_and_omitted_available_recognition": True,
            "v1_labels_are_already_seen": True,
            "weights_and_gate_unchanged": True,
        },
        "selection": {
            "case_ids": [str(case["id"]) for case in source_cases],
            "source": str(input_manifest_path),
            "source_sha256": _sha256(input_manifest_path),
            "raw_dir": str(raw_dir),
            "recognition_root": str(RECOGNITION_ROOT),
            "raw_freeze_manifest": str(raw_freeze_path),
            "raw_freeze_manifest_sha256": _sha256(raw_freeze_path),
        },
        "phase_policy": {
            "period_beats": "derived from each joint DBN position sequence and exact-cycle validated",
            "enumeration": "integer phases 0..period-1 only",
            "weights": PHASE_WEIGHTS,
            "minimum_complete_bars": MIN_COMPLETE_BARS,
            "first_last_half_same_winner": True,
            "half_bar_sets": "disjoint first floor(n/2) and last floor(n/2) complete bars; middle bars omitted when odd",
            "winner_margin_at_least": MIN_MARGIN,
            "crop_equivariance": "delete one leading beat index; this is index-phase covariance, not PCM-crop robustness",
        },
        "forbidden_searches": ["beat_time", "beat_count", "tempo", "offset_sec", "meter", "segment_phase", "velocity", "drum", "chroma", "oracle", "reference_labels"],
        "observable_sources": {
            "audio": "raw frozen WAV clips",
            "beat_events": "Beat Transformer official raw NPZ, unchanged",
            "downbeat_period": "derived and exact-cycle validated from each joint DBN position sequence",
            "joint_downbeat_phase": "nearest beat-event index mapping of position-1 joint times, phase mode with deviation report",
            "muscriptor_game_notes": "raw recognition start_sec/midi/instrument_group/voice_id/is_drum only",
            "bass_alignment": "explicit bass instrument or non-drum midi<48; chord/onset de-duplicated; missing stays zero without reweighting",
        },
        "feature_helper_source": {"path": str(FEATURE_HELPER_PATH), "sha256": feature_helper_sha256},
        "post_hoc_labels": "opened only after raw-decision-freeze.json is written",
    }
    _write_json(output_root / "preregistration.json", preregistration)

    input_cases: list[dict[str, Any]] = []
    raw_cases: dict[str, dict[str, Any]] = {}
    for case in source_cases:
        case_id = str(case["id"])
        manifest, raw = _input_case_manifest(case, raw_dir, raw_freeze_cases, RECOGNITION_ROOT)
        input_cases.append(manifest)
        raw_cases[case_id] = raw
    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "annotation_accessed": False,
        "source_manifest_path": str(input_manifest_path),
        "source_manifest_sha256": _sha256(input_manifest_path),
        "raw_directory": str(raw_dir),
        "raw_freeze_source": str(raw_freeze_path),
        "raw_freeze_source_sha256": _sha256(raw_freeze_path),
        "upstream_hash_validation": "source frozen-input manifest and upstream raw-freeze manifest expected hashes verified before feature extraction",
        "feature_helper_source": {"path": str(FEATURE_HELPER_PATH), "sha256": feature_helper_sha256},
        "cases": input_cases,
    }
    _write_json(output_root / "input-manifest.json", input_manifest)

    decisions: dict[str, dict[str, Any]] = {}
    raw_case_records: list[dict[str, Any]] = []
    for input_case in input_cases:
        case_id = str(input_case["id"])
        raw = raw_cases[case_id]
        features = _audio_features(Path(str(input_case["audio_path"])))
        beats = [float(value) for value in raw["beat_events"]]
        onset_values = _sample_envelope(features["onset"], beats)
        low_values = _sample_envelope(features["low_onset"], beats)
        bass_onsets = list(input_case["recognition"].get("bass_onsets", []))
        decision = _select_integer_phase(
            beats=beats,
            onset_values=onset_values,
            low_values=low_values,
            bass_onsets=bass_onsets,
            period=int(input_case["period_beats"]),
            original_phase=(int(raw["original_phase_index"]) if raw["original_phase_index"] is not None else None),
        )
        frozen_beat_values = [float(value) for value in input_case["beat_event_values"]]
        actual_beat_values = [float(value) for value in raw["beat_events"]]
        beat_output = _beat_output_invariant(frozen_beat_values, str(input_case["beat_events_sha256"]), actual_beat_values)
        phase_output = _candidate_downbeats(raw, decision)
        decision = {
            **decision,
            "audio_observations": {
                "duration_sec": float(features["duration_sec"]),
                "onset_sample_count": len(onset_values),
                "low_frequency_sample_count": len(low_values),
                "bass_onsets_available": bool(input_case["recognition"].get("bass_onsets")),
                "bass_onset_count": len(bass_onsets),
                "bass_recognition": input_case["recognition"],
                "bass_reweighted": False,
                "velocity_used": False,
                "drum_used": False,
                "chroma_used": False,
            },
            "beat_output": beat_output,
            "phase_output_contract": phase_output,
            "joint_to_beat_mapping": raw["joint_to_beat_mapping"],
            "raw_sources": {
                "raw_npz_sha256": input_case["raw_npz_sha256"],
                "raw_metadata_sha256": input_case["raw_metadata_sha256"],
                "audio_sha256": input_case["audio_sha256"],
                "recognition_sha256": input_case["recognition"].get("sha256"),
                "feature_helper_sha256": feature_helper_sha256,
            },
        }
        decisions[case_id] = decision
        raw_case_records.append(
            {
                "id": case_id,
                "raw_npz_sha256": input_case["raw_npz_sha256"],
                "audio_sha256": input_case["audio_sha256"],
                "decision_sha256": _sha256_bytes(_canonical_bytes(decision)),
                "beat_count": len(beats),
                "beat_events_sha256": raw["beat_events_sha256"],
                "recognition_sha256": input_case["recognition"].get("sha256"),
                "feature_helper_sha256": feature_helper_sha256,
            }
        )

    raw_decision_freeze = {
        "schema_version": SCHEMA_VERSION,
        "annotation_accessed_before_freeze": False,
        "freeze_created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": _sha256(output_root / "preregistration.json"),
        "input_manifest_sha256": _sha256(output_root / "input-manifest.json"),
        "source_raw_freeze_manifest_sha256": _sha256(BT_ROOT / "raw-freeze-manifest.json"),
        "feature_helper_source": {"path": str(FEATURE_HELPER_PATH), "sha256": feature_helper_sha256},
        "upstream_hash_validation": input_manifest["upstream_hash_validation"],
        "case_count": len(raw_case_records),
        "cases": raw_case_records,
        "decisions": decisions,
        "decision_semantics": "only this frozen decision may be scored; no annotation/oracle value participated",
    }
    _write_json(output_root / "raw-decision-freeze.json", raw_decision_freeze)

    # This is the first point where this diagnostic reads any annotation path
    # or label content.  The freeze above is durable before this call starts.
    evaluation = _score_development(
        evaluation_path=evaluation_path,
        input_cases=input_cases,
        raw_cases=raw_cases,
        decisions=decisions,
    )
    evaluation["raw_decision_freeze_sha256"] = _sha256(output_root / "raw-decision-freeze.json")
    _write_json(output_root / "evaluation.json", evaluation)

    decision = evaluation["decision"]
    if evaluation["development_gate_passed"]:
        heldout_status = {
            "attempted": False,
            "status": "blocked_pending_strict_asap_discovery_implementation",
            "reason": "The development gate unexpectedly passed; no heldout labels or model run were started by this bounded diagnostic.",
        }
    else:
        heldout_status = {
            "attempted": False,
            "status": "stopped_after_development_failure",
            "reason": "The preregistered development gate failed; no new ASAP works were opened or selected.",
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "development_decision": decision,
        "heldout8": heldout_status,
        "provenance": {
            "labels_already_present_before_experiment": True,
            "weights_and_labels_are_theoretical_diagnostic_only": True,
            "production_modified": False,
        },
        "evaluation_summary": evaluation["summary"],
        "gate": evaluation["development_gate"],
    }
    _write_json(output_root / "report.json", report)
    _write_json(output_root / "artifact-manifest.json", _artifact_manifest(output_root))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    report = run(
        input_manifest_path=args.input_manifest,
        raw_dir=args.raw_dir,
        output_root=args.output,
        evaluation_path=args.evaluation,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
