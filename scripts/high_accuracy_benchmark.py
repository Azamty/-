"""Register and evaluate high-accuracy transcription benchmark inputs.

The registry is intentionally small and honest.  Running this script without
``--result-root`` only records which local inputs are available; it does not
invent accuracy numbers.  A result directory is evaluated only when it has a
service manifest and a final score MIDI.  Audio and corpus files stay outside
the repository and are referenced by path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "review" / "high-accuracy-benchmark" / "latest.json"
MidiNote = tuple[int, Fraction, Fraction]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _file_record(value: str | None, *, required: bool = False) -> dict[str, Any] | None:
    path = _resolve_path(value)
    if path is None:
        return None
    record: dict[str, Any] = {"path": str(path), "required": required, "available": path.is_file()}
    if path.is_file():
        stat = path.stat()
        record.update({"bytes": stat.st_size, "sha256": _sha256(path)})
    else:
        record["reason"] = "文件不在本机；请按来源说明准备后再运行"
    return record


def _load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {"1.0", "2.0"} or not isinstance(payload.get("cases"), list):
        raise ValueError(f"benchmark registry is not a supported schema: {path}")
    ids: set[str] = set()
    for item in payload["cases"]:
        if not isinstance(item, Mapping) or not item.get("id") or item["id"] in ids:
            raise ValueError("benchmark cases require unique non-empty ids")
        ids.add(str(item["id"]))
        if not item.get("input"):
            raise ValueError(f"benchmark case {item['id']} has no input")
        if payload.get("schema_version") == "2.0":
            if not item.get("category") or not item.get("source_kind"):
                raise ValueError(f"benchmark case {item['id']} requires category and source_kind")
            if item.get("source_id") and item["source_id"] not in payload.get("sources", {}):
                raise ValueError(f"benchmark case {item['id']} refers to an unknown source_id")
    if payload.get("schema_version") == "2.0":
        reliable_count = sum(1 for item in payload["cases"] if item.get("reference_midi_reliable") is True and item.get("evaluation_policy") == "reference_metrics")
        if reliable_count != 30:
            raise ValueError(f"benchmark schema 2.0 requires exactly 30 reliable cases, found {reliable_count}")
    return payload


def _midi_notes(path: Path, *, exclude_drum_channel: bool = False) -> tuple[int, list[MidiNote]]:
    midi = mido.MidiFile(path)
    notes: list[MidiNote] = []
    for track in midi.tracks:
        tick = 0
        active: dict[tuple[int, int], list[int]] = defaultdict(list)
        for message in track:
            tick += int(message.time)
            channel = int(getattr(message, "channel", 0))
            if exclude_drum_channel and channel == 9 and message.type in {"note_on", "note_off"}:
                continue
            if message.type == "note_on" and message.velocity > 0:
                active[(channel, int(message.note))].append(tick)
            elif message.type in {"note_off", "note_on"}:
                key = (channel, int(message.note))
                starts = active.get(key, [])
                if starts:
                    start = starts.pop(0)
                    if tick > start:
                        notes.append((int(message.note), Fraction(start, midi.ticks_per_beat), Fraction(tick, midi.ticks_per_beat)))
    return int(midi.ticks_per_beat), sorted(notes, key=lambda value: (value[1], value[0], value[2]))


def _f1(precision_count: int, recall_count: int, predicted_count: int, reference_count: int) -> dict[str, float | int | None]:
    precision = precision_count / predicted_count if predicted_count else None
    recall = recall_count / reference_count if reference_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
        if precision == 0.0 and recall == 0.0
        else None
    )
    return {"true_positive": precision_count, "predicted": predicted_count, "reference": reference_count, "precision": precision, "recall": recall, "f1": f1}


def pitch_metrics(reference: Sequence[MidiNote], predicted: Sequence[MidiNote], *, tolerance_quarters: Fraction) -> dict[str, Any]:
    used: set[int] = set()
    matched: list[tuple[int, int]] = []
    for pred_index, (pitch, start, _end) in enumerate(predicted):
        candidates = [
            (abs(start - ref_start), ref_index)
            for ref_index, (ref_pitch, ref_start, _ref_end) in enumerate(reference)
            if ref_index not in used and ref_pitch == pitch and abs(start - ref_start) <= tolerance_quarters
        ]
        if candidates:
            _, ref_index = min(candidates)
            used.add(ref_index)
            matched.append((ref_index, pred_index))
    return _f1(len(matched), len(matched), len(predicted), len(reference))


def rhythm_error(reference: Sequence[MidiNote], predicted: Sequence[MidiNote], *, tolerance_quarters: Fraction) -> dict[str, Any]:
    used: set[int] = set()
    onset_errors: list[Fraction] = []
    duration_errors: list[Fraction] = []
    for pitch, start, end in predicted:
        candidates = [
            (abs(start - ref_start), ref_index)
            for ref_index, (ref_pitch, ref_start, _ref_end) in enumerate(reference)
            if ref_index not in used and ref_pitch == pitch and abs(start - ref_start) <= tolerance_quarters
        ]
        if candidates:
            _onset_error, ref_index = min(candidates)
            used.add(ref_index)
            _, ref_start, ref_end = reference[ref_index]
            onset_errors.append(abs(start - ref_start))
            duration_errors.append(abs((end - start) - (ref_end - ref_start)))
    mean_onset = sum(onset_errors, Fraction(0)) / len(onset_errors) if onset_errors else None
    mean_duration = sum(duration_errors, Fraction(0)) / len(duration_errors) if duration_errors else None
    combined = [onset + duration for onset, duration in zip(onset_errors, duration_errors, strict=True)]
    mean_combined = sum(combined, Fraction(0)) / len(combined) if combined else None
    return {
        "matched_notes": len(onset_errors),
        "mean_onset_error_quarter": float(mean_onset) if mean_onset is not None else None,
        "mean_duration_error_quarter": float(mean_duration) if mean_duration is not None else None,
        "mean_rhythm_error_quarter": float(mean_combined) if mean_combined is not None else None,
        "unit": "quarter_note",
    }


def chord_retention(reference: Sequence[MidiNote], predicted: Sequence[MidiNote], *, tolerance_quarters: Fraction) -> dict[str, Any]:
    def groups(notes: Sequence[MidiNote]) -> list[tuple[Fraction, frozenset[int]]]:
        grouped: dict[Fraction, set[int]] = defaultdict(set)
        for pitch, start, _end in notes:
            grouped[start].add(pitch)
        return sorted((start, frozenset(pitches)) for start, pitches in grouped.items() if len(pitches) > 1)

    predicted_groups = groups(predicted)
    retained = 0
    for ref_start, ref_pitches in groups(reference):
        if any(abs(pred_start - ref_start) <= tolerance_quarters and ref_pitches.issubset(pred_pitches) for pred_start, pred_pitches in predicted_groups):
            retained += 1
    total = len(groups(reference))
    return {"reference_chords": total, "retained_chords": retained, "retention": retained / total if total else None}


def _read_time_points(path: Path | None, *, downbeats: bool = False) -> list[float]:
    if path is None or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    values: list[float] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("beat_grid"), Mapping):
        payload = payload["beat_grid"]
    explicit_downbeat_records = bool(downbeats and isinstance(payload, Mapping) and "downbeats" in payload)
    if isinstance(payload, Mapping):
        if downbeats:
            # A beat list without explicit downbeat flags is not a downbeat
            # annotation.  Falling back to every beat would inflate the
            # downbeat F1 and hide missing bar-position information.
            has_downbeat_array = "downbeats" in payload
            records = payload.get("downbeats") if has_downbeat_array else payload.get("beats", [])
        else:
            records = payload.get("beats") or payload.get("beat_times") or []
    else:
        records = payload
    if isinstance(records, list):
        for item in records:
            value = (
                item.get("time_sec", item.get("time", item.get("start_sec")))
                if isinstance(item, Mapping)
                else item
            )
            if downbeats and not explicit_downbeat_records and isinstance(item, Mapping) and not item.get("downbeat", False):
                continue
            if downbeats and not explicit_downbeat_records and not isinstance(item, Mapping):
                # Numeric fallback records came from a beat list, not an
                # explicit downbeat list, and therefore carry no evidence.
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
    return values


def beat_f1(reference: Sequence[float], predicted: Sequence[float], *, tolerance_sec: float = 0.07) -> dict[str, Any]:
    used: set[int] = set()
    true_positive = 0
    for value in predicted:
        candidates = [(abs(value - item), index) for index, item in enumerate(reference) if index not in used and abs(value - item) <= tolerance_sec]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            true_positive += 1
    return _f1(true_positive, true_positive, len(predicted), len(reference))


def _manifest_artifacts(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only artifacts explicitly declared by the selected manifest."""

    values: list[Mapping[str, Any]] = []
    for owner in (manifest, manifest.get("result") if isinstance(manifest.get("result"), Mapping) else None):
        if isinstance(owner, Mapping) and isinstance(owner.get("artifacts"), list):
            values.extend(item for item in owner["artifacts"] if isinstance(item, Mapping))
    return values


def _safe_manifest_path(result_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = (result_root / value).resolve()
    if not candidate.is_relative_to(result_root.resolve()) or not candidate.is_file():
        return None
    return candidate


def _find_result_artifact(result_root: Path, manifest: Mapping[str, Any], suffixes: Iterable[str]) -> Path | None:
    """Resolve one artifact from manifest declarations; never scan by suffix.

    A benchmark must fail closed when a manifest names multiple candidates.  A
    recursive directory scan can silently select baseline/performance output
    from a neighboring pipeline and is therefore not an acceptable fallback.
    """

    suffix_values = tuple(str(suffix).lower() for suffix in suffixes)
    candidates: list[Path] = []
    owners: list[Mapping[str, Any]] = [manifest]
    nested = manifest.get("result")
    if isinstance(nested, Mapping):
        owners.append(nested)
    for owner in owners:
        for key in ("final_midi", "beat_grid", "beat_grid_path"):
            path = _safe_manifest_path(result_root, owner.get(key))
            if path is not None and (
                any(str(path).lower().endswith(suffix) for suffix in suffix_values)
                or key == "final_midi" and any("midi" in suffix for suffix in suffix_values)
                or key.startswith("beat_grid") and any("beat_grid" in suffix for suffix in suffix_values)
            ):
                candidates.append(path)
    for artifact in _manifest_artifacts(manifest):
        relative = artifact.get("relative_path") or artifact.get("path")
        kind = str(artifact.get("kind", "")).lower()
        path = _safe_manifest_path(result_root, relative)
        kind_tokens = {suffix.lstrip(".").replace(".", "_") for suffix in suffix_values}
        kind_tokens.update(
            token.replace("_mid", "_midi")
            for token in kind_tokens
            if token.endswith("_mid")
        )
        if path is not None and (
            any(str(path).lower().endswith(suffix) for suffix in suffix_values)
            or any(token in kind for token in kind_tokens)
        ):
            candidates.append(path)
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _manifest_effective_scope(manifest: Mapping[str, Any], *, raw_model_output: bool | None) -> str:
    """Read the scope selected by the runner, with a safe legacy fallback."""

    owners: list[Mapping[str, Any]] = [manifest]
    nested = manifest.get("result")
    if isinstance(nested, Mapping):
        owners.append(nested)
    for owner in owners:
        value = owner.get("effective_evaluation_scope")
        if isinstance(value, str) and value.strip():
            return value
    if raw_model_output is not None:
        return "production_end_to_end" if raw_model_output is True else "quantizer_isolation"
    for owner in owners:
        value = owner.get("evaluation_scope")
        if isinstance(value, str) and value.strip():
            return value
    return "quantizer_isolation"


def evaluate_case(case: Mapping[str, Any], *, result_root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": str(case["id"]),
        "title": case.get("title", case["id"]),
        "language": case.get("language"),
        "category": case.get("category"),
        "source_kind": case.get("source_kind"),
        "source_id": case.get("source_id"),
        "license": case.get("license"),
        "input": _file_record(str(case["input"]), required=True),
        "reference_midi": _file_record(case.get("reference_midi"), required=False),
        "reference_midi_reliable": bool(case.get("reference_midi_reliable", False)),
        "beat_annotation": _file_record(case.get("beat_annotation"), required=False),
        "beat_annotation_independent": case.get("beat_annotation_independent") is True,
        "evaluation_policy": case.get("evaluation_policy", "reference_metrics"),
        "evaluation_scope": case.get("evaluation_scope"),
        "case_evaluation_scope": case.get("evaluation_scope"),
        "status": "registered",
        "crash": None,
        "metrics": {"pitch_f1": None, "chord_retention": None, "rhythm_error": None, "beat_f1": None, "downbeat_f1": None},
        "beat_metrics_eligible": False,
        "beat_metrics_reason": None,
        "notes": case.get("notes", ""),
    }
    if result_root is None:
        return result
    case_root = (result_root / str(case["id"])).resolve()
    manifest_path = case_root / "manifest.json"
    if not manifest_path.is_file():
        result.update({"status": "not_evaluated", "crash": False, "reason": "result manifest not found"})
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("result manifest is not an object")
        failed = str(manifest.get("status", "")).lower() in {"failed", "error"} or bool(manifest.get("error"))
        result["crash"] = failed
        if failed:
            result.update({"status": "crashed", "reason": manifest.get("error", "service manifest reports failure")})
            return result
        raw_model_output = manifest.get("raw_model_output")
        if raw_model_output is None and isinstance(manifest.get("result"), Mapping):
            raw_model_output = manifest["result"].get("raw_model_output")
        raw_model_output = raw_model_output if isinstance(raw_model_output, bool) else None
        effective_scope = _manifest_effective_scope(manifest, raw_model_output=raw_model_output)
        result["evaluation_scope"] = effective_scope
        if result["evaluation_policy"] != "reference_metrics" or not result["reference_midi_reliable"]:
            result.update({"status": "integrity_only", "reason": "此样本没有可用于准确率结论的可靠参考标注"})
            return result
        predicted_path = _find_result_artifact(case_root, manifest, (".score.mid", ".score.midi"))
        reference_value = case.get("reference_midi")
        reference_path = _resolve_path(reference_value)
        if predicted_path is None or reference_path is None or not reference_path.is_file():
            result.update({"status": "not_evaluated", "reason": "final MIDI or reliable reference is unavailable"})
            return result
        pred_ppq, predicted = _midi_notes(predicted_path, exclude_drum_channel=True)
        ref_ppq, reference = _midi_notes(reference_path, exclude_drum_channel=True)
        # _midi_notes already converts each SMF's integer ticks to exact
        # quarter-note Fractions.  The tolerance is therefore independent of
        # whether one file uses 96, 480, or another PPQ.
        tolerance_quarters = Fraction(1, 16)
        result["metrics"] = {
            "pitch_f1": pitch_metrics(reference, predicted, tolerance_quarters=tolerance_quarters),
            "chord_retention": chord_retention(reference, predicted, tolerance_quarters=tolerance_quarters),
            "rhythm_error": rhythm_error(reference, predicted, tolerance_quarters=tolerance_quarters),
            "beat_f1": None,
            "downbeat_f1": None,
        }
        result.update({"status": "evaluated", "result_midi": str(predicted_path), "reference_ppq": ref_ppq, "result_ppq": pred_ppq})
        beat_path = _resolve_path(case.get("beat_annotation"))
        scope = str(effective_scope or "")
        case_scope = str(case.get("evaluation_scope") or "")
        reference_derived = (
            "reference_derived" in case_scope
            or str(case.get("beat_annotation_source") or "").casefold() in {"reference_midi", "reference-derived", "same_reference"}
        )
        quantizer_isolation = scope.startswith("quantizer_isolation")
        beat_eligible = bool(beat_path) and not reference_derived and not quantizer_isolation and case.get("beat_annotation_independent") is True and raw_model_output is True
        result["beat_metrics_eligible"] = beat_eligible
        if not beat_path:
            result["beat_metrics_reason"] = "beat annotation unavailable"
        elif reference_derived:
            result["beat_metrics_reason"] = "beat annotation is derived from the reference MIDI; excluded from independent BeatNet F1"
        elif quantizer_isolation:
            result["beat_metrics_reason"] = "quantizer-isolation case; beat grid is not an independent production recognition result"
        elif raw_model_output is not True:
            result["beat_metrics_reason"] = "result manifest is reference-derived rather than model output"
        elif case.get("beat_annotation_independent") is not True:
            result["beat_metrics_reason"] = "case does not declare an independent beat annotation"
        if beat_eligible:
            predicted_beat_path = _find_result_artifact(case_root, manifest, ("beat_grid.json",))
            result["metrics"]["beat_f1"] = beat_f1(_read_time_points(beat_path), _read_time_points(predicted_beat_path))
            result["metrics"]["downbeat_f1"] = beat_f1(_read_time_points(beat_path, downbeats=True), _read_time_points(predicted_beat_path, downbeats=True))
    except Exception as exc:  # benchmark must record a crash instead of hiding it
        result.update({"status": "crashed", "crash": True, "reason": f"{type(exc).__name__}: {exc}"})
    return result


def _metric_f1(case: Mapping[str, Any], name: str, field: str) -> float | None:
    value = (case.get("metrics") or {}).get(name)
    if not isinstance(value, Mapping):
        return None
    number = value.get(field)
    return float(number) if isinstance(number, (int, float)) and math.isfinite(float(number)) else None


def _mean_metric(cases: Sequence[Mapping[str, Any]], name: str, field: str) -> float | None:
    values = [value for case in cases if (value := _metric_f1(case, name, field)) is not None]
    return sum(values) / len(values) if values else None


def _case_id_index(
    cases: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> tuple[dict[str, Mapping[str, Any]], list[str], list[str]]:
    """Index reliable cases while retaining ID integrity diagnostics."""

    all_ids: list[str] = []
    missing: list[str] = []
    for index, case in enumerate(cases):
        value = case.get("id")
        if value is None or not str(value).strip():
            missing.append(f"{label}[{index}]")
            continue
        all_ids.append(str(value))
    duplicates = sorted(value for value, count in Counter(all_ids).items() if count > 1)
    reliable: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        value = case.get("id")
        if value is None or not str(value).strip():
            continue
        case_id = str(value)
        if (
            case.get("status") == "evaluated"
            and case.get("evaluation_policy") == "reference_metrics"
            and case.get("reference_midi_reliable") is True
        ):
            reliable.setdefault(case_id, case)
    return reliable, missing, duplicates


def assess_accuracy_claim(
    cases: Sequence[Mapping[str, Any]],
    baseline_cases: Sequence[Mapping[str, Any]] | None,
    *,
    minimum_cases: int = 30,
    require_beat_metrics: bool = True,
    minimum_beat_cases: int | None = None,
    _include_scopes: bool = True,
) -> dict[str, Any]:
    """Apply the stated accuracy gate without filling missing results.

    The gate is deliberately separate from per-case scoring so a future run
    can supply a baseline produced by the old chain without changing this
    registry or pretending that unrun cases passed.
    """

    reasons: list[str] = []
    new_by_id, new_missing, new_duplicates = _case_id_index(cases, label="new")
    if new_missing:
        reasons.append(f"新链路存在缺失 case ID：{', '.join(new_missing)}")
    if new_duplicates:
        reasons.append(f"新链路存在重复 case ID：{', '.join(new_duplicates)}")
    if any(case.get("crash") is True for case in cases):
        reasons.append("新链路存在崩溃样本")

    baseline_by_id: dict[str, Mapping[str, Any]] = {}
    baseline_missing: list[str] = []
    baseline_duplicates: list[str] = []
    if baseline_cases is None:
        reasons.append("缺少 baseline 结果")
    else:
        baseline_by_id, baseline_missing, baseline_duplicates = _case_id_index(baseline_cases, label="baseline")
        if baseline_missing:
            reasons.append(f"baseline 存在缺失 case ID：{', '.join(baseline_missing)}")
        if baseline_duplicates:
            reasons.append(f"baseline 存在重复 case ID：{', '.join(baseline_duplicates)}")

    new_ids = set(new_by_id)
    baseline_ids = set(baseline_by_id)
    if baseline_cases is None:
        shared_ids: list[str] = []
    else:
        missing_from_baseline = sorted(new_ids - baseline_ids)
        missing_from_new = sorted(baseline_ids - new_ids)
        if missing_from_baseline:
            reasons.append(f"baseline 缺少新链路可靠 case ID：{', '.join(missing_from_baseline)}")
        if missing_from_new:
            reasons.append(f"新链路缺少 baseline 可靠 case ID：{', '.join(missing_from_new)}")
        # Keep the new report's registry order in diagnostics; the set below
        # is only used for membership, so IDs such as case-10 do not sort
        # before case-2 merely because they are strings.
        shared_ids = [case_id for case_id in new_by_id if case_id in baseline_by_id]
    reliable = [new_by_id[case_id] for case_id in shared_ids]
    baseline_reliable = [baseline_by_id[case_id] for case_id in shared_ids]
    if len(shared_ids) < minimum_cases:
        reasons.append(f"相同可靠 case ID 只有 {len(shared_ids)}/{minimum_cases} 个")

    new_beat = _mean_metric(reliable, "beat_f1", "f1")
    new_downbeat = _mean_metric(reliable, "downbeat_f1", "f1")
    beat_case_count = sum(
        case.get("beat_metrics_eligible", True) is not False
        and _metric_f1(case, "beat_f1", "f1") is not None
        and _metric_f1(case, "downbeat_f1", "f1") is not None
        for case in reliable
    )
    if require_beat_metrics:
        if minimum_beat_cases is not None and beat_case_count < minimum_beat_cases:
            reasons.append(f"独立 BeatNet 拍点/重拍指标只有 {beat_case_count}/{minimum_beat_cases} 个 case")
        if new_beat is None or new_beat < 0.85:
            reasons.append(f"拍点 F1 不足 0.85（当前 {new_beat if new_beat is not None else '缺失'}）")
        if new_downbeat is None or new_downbeat < 0.75:
            reasons.append(f"重拍 F1 不足 0.75（当前 {new_downbeat if new_downbeat is not None else '缺失'}）")

    baseline_rhythm = _mean_metric(baseline_reliable, "rhythm_error", "mean_rhythm_error_quarter")
    new_rhythm = _mean_metric(reliable, "rhythm_error", "mean_rhythm_error_quarter")
    baseline_pitch = _mean_metric(baseline_reliable, "pitch_f1", "f1")
    new_pitch = _mean_metric(reliable, "pitch_f1", "f1")
    baseline_chord = _mean_metric(baseline_reliable, "chord_retention", "retention")
    new_chord = _mean_metric(reliable, "chord_retention", "retention")
    if baseline_rhythm is None or new_rhythm is None:
        reasons.append("新链路或 baseline 缺少可比较的四分音符节奏误差")
    elif baseline_rhythm <= 0:
        reasons.append("baseline 节奏误差为零，无法计算20%下降")
    elif new_rhythm > baseline_rhythm * 0.8:
        reasons.append(f"节奏误差未下降至少20%（新 {new_rhythm:.6f}，baseline {baseline_rhythm:.6f}）")
    if baseline_pitch is None or new_pitch is None:
        reasons.append("新链路或 baseline 缺少 pitch F1")
    elif new_pitch + 1e-9 < baseline_pitch - 0.01:
        reasons.append(f"pitch F1 下降超过0.01（新 {new_pitch:.6f}，baseline {baseline_pitch:.6f}）")
    if baseline_chord is None or new_chord is None:
        reasons.append("新链路或 baseline 缺少和弦保留率")
    elif new_chord < baseline_chord:
        reasons.append(f"和弦保留率下降（新 {new_chord:.6f}，baseline {baseline_chord:.6f}）")
    result = {
        "ready": not reasons,
        "reason": "; ".join(reasons) if reasons else "已满足可靠样本、节奏、音高、和弦和无崩溃门槛" if not require_beat_metrics else "已满足30个可靠样本、拍点/重拍、节奏、音高、和弦和无崩溃门槛",
        "minimum_cases": minimum_cases,
        "require_beat_metrics": require_beat_metrics,
        "minimum_beat_cases": minimum_beat_cases,
        "beat_cases_with_metrics": beat_case_count,
        "new_reliable_count": len(reliable),
        "baseline_reliable_count": len(baseline_reliable),
        "new_reliable_total": len(new_by_id),
        "baseline_reliable_total": len(baseline_by_id),
        "shared_case_ids": shared_ids,
        "new_mean_beat_f1": new_beat,
        "new_mean_downbeat_f1": new_downbeat,
        "new_mean_rhythm_error_quarter": new_rhythm,
        "baseline_mean_rhythm_error_quarter": baseline_rhythm,
        "new_mean_pitch_f1": new_pitch,
        "baseline_mean_pitch_f1": baseline_pitch,
        "new_mean_chord_retention": new_chord,
        "baseline_mean_chord_retention": baseline_chord,
    }
    if _include_scopes:
        quantizer_cases = [case for case in cases if str(case.get("evaluation_scope") or "").startswith("quantizer_isolation")]
        quantizer_baseline = (
            [case for case in baseline_cases if str(case.get("evaluation_scope") or "").startswith("quantizer_isolation")]
            if baseline_cases is not None
            else None
        )
        production_cases = [
            case
            for case in cases
            if str(case.get("evaluation_scope") or "").startswith(("end_to_end", "production_end_to_end"))
        ]
        production_baseline = (
            [
                case
                for case in baseline_cases
                if str(case.get("evaluation_scope") or "").startswith(("end_to_end", "production_end_to_end"))
            ]
            if baseline_cases is not None
            else None
        )
        scopes: dict[str, Any] = {}
        if quantizer_cases:
            scopes["quantizer_isolation_overall"] = assess_accuracy_claim(
                quantizer_cases,
                quantizer_baseline,
                minimum_cases=len(quantizer_cases),
                require_beat_metrics=False,
                _include_scopes=False,
            )
        if production_cases:
            expected_beat_cases = sum(case.get("beat_annotation_independent") is True for case in production_cases)
            scopes["production_end_to_end_subset"] = assess_accuracy_claim(
                production_cases,
                production_baseline,
                minimum_cases=len(production_cases),
                require_beat_metrics=True,
                minimum_beat_cases=expected_beat_cases,
                _include_scopes=False,
            )
        result["scopes"] = scopes
    return result


def build_report(
    registry: Mapping[str, Any],
    *,
    result_root: Path | None = None,
    baseline_root: Path | None = None,
    baseline_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = [evaluate_case(item, result_root=result_root) for item in registry["cases"]]
    evaluated = [item for item in cases if item["status"] == "evaluated"]
    crashed = [item for item in cases if item["crash"] is True]
    baseline_cases = (
        list(baseline_report.get("cases", []))
        if baseline_report is not None and isinstance(baseline_report.get("cases"), list)
        else [evaluate_case(item, result_root=baseline_root) for item in registry["cases"]]
        if baseline_root is not None
        else None
    )
    production_cases = [
        case
        for case in cases
        if str(case.get("evaluation_scope") or "").startswith("production_end_to_end")
    ]
    production_baseline = (
        [
            case
            for case in baseline_cases
            if str(case.get("evaluation_scope") or "").startswith("production_end_to_end")
        ]
        if baseline_cases is not None
        else None
    )
    # The acceptance gate is always the shared 30-case production gate.  The
    # quantizer and production subset claims below are diagnostics and cannot
    # make a partial or reference-derived run acceptable.
    claim = assess_accuracy_claim(
        production_cases,
        production_baseline,
        minimum_cases=30,
        require_beat_metrics=True,
        minimum_beat_cases=30,
        _include_scopes=False,
    )
    diagnostic_claim = assess_accuracy_claim(cases, baseline_cases)
    scoped_claims = diagnostic_claim.get("scopes", {})
    accuracy_ready = claim["ready"]
    accuracy_reason = claim["reason"]
    return {
        "schema_version": "2.0",
        "registry": str(DEFAULT_REGISTRY),
        "result_root": str(result_root) if result_root else None,
        "registered_count": len(cases),
        "evaluated_count": len(evaluated),
        "crash_count": len(crashed),
        "baseline_result_root": str(baseline_root) if baseline_root else None,
        "accuracy_claim_ready": accuracy_ready,
        "accuracy_claim_reason": accuracy_reason,
        "accuracy_gate": claim,
        "accuracy_gate_scopes": scoped_claims,
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--result-root", type=Path, help="按 case id/manifest.json 提供已完成服务结果")
    parser.add_argument("--baseline-result-root", type=Path, help="按 case id/manifest.json 提供旧链路对照结果")
    parser.add_argument("--baseline-report", type=Path, help="读取已有 benchmark 报告作为 baseline 对照")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-inputs", action="store_true", help="缺少登记的本机输入时以失败退出")
    parser.add_argument("--check", action="store_true", help="只登记并校验输入，不评估服务结果")
    args = parser.parse_args(argv)
    registry = _load_registry(args.manifest.resolve())
    baseline_report = json.loads(args.baseline_report.resolve().read_text(encoding="utf-8")) if args.baseline_report else None
    report = build_report(
        registry,
        result_root=None if args.check else (args.result_root.resolve() if args.result_root else None),
        baseline_root=None if args.check else (args.baseline_result_root.resolve() if args.baseline_result_root else None),
        baseline_report=baseline_report,
    )
    missing = [item["id"] for item in report["cases"] if not item["input"]["available"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "registered": report["registered_count"], "evaluated": report["evaluated_count"], "crashed": report["crash_count"], "missing_inputs": missing}, ensure_ascii=False))
    return 2 if args.strict_inputs and missing else 0


if __name__ == "__main__":
    sys.exit(main())
