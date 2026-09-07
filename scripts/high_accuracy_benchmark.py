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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mido

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "review" / "high-accuracy-benchmark" / "latest.json"
MIDI_SUFFIXES = (".mid", ".midi")


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
    if payload.get("schema_version") != "1.0" or not isinstance(payload.get("cases"), list):
        raise ValueError(f"benchmark registry is not schema 1.0: {path}")
    ids: set[str] = set()
    for item in payload["cases"]:
        if not isinstance(item, Mapping) or not item.get("id") or item["id"] in ids:
            raise ValueError("benchmark cases require unique non-empty ids")
        ids.add(str(item["id"]))
        if not item.get("input"):
            raise ValueError(f"benchmark case {item['id']} has no input")
    return payload


def _midi_notes(path: Path) -> tuple[int, list[tuple[int, int, int]]]:
    midi = mido.MidiFile(path)
    notes: list[tuple[int, int, int]] = []
    for track in midi.tracks:
        tick = 0
        active: dict[tuple[int, int], list[int]] = defaultdict(list)
        for message in track:
            tick += int(message.time)
            if message.type == "note_on" and message.velocity > 0:
                active[(int(getattr(message, "channel", 0)), int(message.note))].append(tick)
            elif message.type in {"note_off", "note_on"}:
                key = (int(getattr(message, "channel", 0)), int(message.note))
                starts = active.get(key, [])
                if starts:
                    start = starts.pop(0)
                    if tick > start:
                        notes.append((int(message.note), start, tick))
    return int(midi.ticks_per_beat), sorted(notes, key=lambda value: (value[1], value[0], value[2]))


def _f1(precision_count: int, recall_count: int, predicted_count: int, reference_count: int) -> dict[str, float | int | None]:
    precision = precision_count / predicted_count if predicted_count else None
    recall = recall_count / reference_count if reference_count else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {"true_positive": precision_count, "predicted": predicted_count, "reference": reference_count, "precision": precision, "recall": recall, "f1": f1}


def pitch_metrics(reference: Sequence[tuple[int, int, int]], predicted: Sequence[tuple[int, int, int]], *, tolerance_ticks: int) -> dict[str, Any]:
    used: set[int] = set()
    matched: list[tuple[int, int]] = []
    for pred_index, (pitch, start, _end) in enumerate(predicted):
        candidates = [
            (abs(start - ref_start), ref_index)
            for ref_index, (ref_pitch, ref_start, _ref_end) in enumerate(reference)
            if ref_index not in used and ref_pitch == pitch and abs(start - ref_start) <= tolerance_ticks
        ]
        if candidates:
            _, ref_index = min(candidates)
            used.add(ref_index)
            matched.append((ref_index, pred_index))
    return _f1(len(matched), len(matched), len(predicted), len(reference))


def rhythm_error(reference: Sequence[tuple[int, int, int]], predicted: Sequence[tuple[int, int, int]], *, tolerance_ticks: int) -> dict[str, Any]:
    used: set[int] = set()
    errors: list[int] = []
    for pitch, start, _end in predicted:
        candidates = [
            (abs(start - ref_start), ref_index)
            for ref_index, (ref_pitch, ref_start, _ref_end) in enumerate(reference)
            if ref_index not in used and ref_pitch == pitch and abs(start - ref_start) <= tolerance_ticks
        ]
        if candidates:
            error, ref_index = min(candidates)
            used.add(ref_index)
            errors.append(error)
    return {"matched_notes": len(errors), "mean_onset_error_ticks": (sum(errors) / len(errors) if errors else None)}


def chord_retention(reference: Sequence[tuple[int, int, int]], predicted: Sequence[tuple[int, int, int]], *, tolerance_ticks: int) -> dict[str, Any]:
    def groups(notes: Sequence[tuple[int, int, int]]) -> list[tuple[int, frozenset[int]]]:
        grouped: dict[int, set[int]] = defaultdict(set)
        for pitch, start, _end in notes:
            grouped[start].add(pitch)
        return sorted((start, frozenset(pitches)) for start, pitches in grouped.items() if len(pitches) > 1)

    predicted_groups = groups(predicted)
    retained = 0
    for ref_start, ref_pitches in groups(reference):
        if any(abs(pred_start - ref_start) <= tolerance_ticks and ref_pitches.issubset(pred_pitches) for pred_start, pred_pitches in predicted_groups):
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
    if isinstance(payload, Mapping):
        records = payload.get("downbeats" if downbeats else "beats") or payload.get("beat_times") or []
    else:
        records = payload
    if isinstance(records, list):
        for item in records:
            value = item.get("time") if isinstance(item, Mapping) else item
            if downbeats and isinstance(item, Mapping) and not item.get("downbeat", False):
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


def _find_result_artifact(result_root: Path, manifest: Mapping[str, Any], suffixes: Iterable[str]) -> Path | None:
    artifacts = manifest.get("artifacts", [])
    for artifact in artifacts if isinstance(artifacts, list) else []:
        if not isinstance(artifact, Mapping):
            continue
        kind = str(artifact.get("kind", ""))
        relative = artifact.get("relative_path") or artifact.get("path")
        if relative and any(suffix in kind or str(relative).lower().endswith(suffix) for suffix in suffixes):
            candidate = (result_root / str(relative)).resolve()
            if candidate.is_file() and candidate.is_relative_to(result_root.resolve()):
                return candidate
    for suffix in suffixes:
        matches = sorted(result_root.rglob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def evaluate_case(case: Mapping[str, Any], *, result_root: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": str(case["id"]),
        "title": case.get("title", case["id"]),
        "language": case.get("language"),
        "license": case.get("license"),
        "input": _file_record(str(case["input"]), required=True),
        "reference_midi": _file_record(case.get("reference_midi"), required=False),
        "reference_midi_reliable": bool(case.get("reference_midi_reliable", False)),
        "beat_annotation": _file_record(case.get("beat_annotation"), required=False),
        "evaluation_policy": case.get("evaluation_policy", "reference_metrics"),
        "status": "registered",
        "crash": None,
        "metrics": {"pitch_f1": None, "chord_retention": None, "rhythm_error": None, "beat_f1": None, "downbeat_f1": None},
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
        if result["evaluation_policy"] != "reference_metrics" or not result["reference_midi_reliable"]:
            result.update({"status": "integrity_only", "reason": "此样本没有可用于准确率结论的可靠参考标注"})
            return result
        predicted_path = _find_result_artifact(case_root, manifest, (".score.mid", ".score.midi"))
        reference_value = case.get("reference_midi")
        reference_path = _resolve_path(reference_value)
        if predicted_path is None or reference_path is None or not reference_path.is_file():
            result.update({"status": "not_evaluated", "reason": "final MIDI or reliable reference is unavailable"})
            return result
        pred_ppq, predicted = _midi_notes(predicted_path)
        ref_ppq, reference = _midi_notes(reference_path)
        tolerance = max(2, round(max(pred_ppq, ref_ppq) * 0.08))
        result["metrics"] = {
            "pitch_f1": pitch_metrics(reference, predicted, tolerance_ticks=tolerance),
            "chord_retention": chord_retention(reference, predicted, tolerance_ticks=tolerance),
            "rhythm_error": rhythm_error(reference, predicted, tolerance_ticks=tolerance),
            "beat_f1": None,
            "downbeat_f1": None,
        }
        result.update({"status": "evaluated", "result_midi": str(predicted_path), "reference_ppq": ref_ppq, "result_ppq": pred_ppq})
        beat_path = _resolve_path(case.get("beat_annotation"))
        result["metrics"]["beat_f1"] = beat_f1(_read_time_points(beat_path), _read_time_points(_find_result_artifact(case_root, manifest, ("beat_grid.json",)))) if beat_path else None
        result["metrics"]["downbeat_f1"] = beat_f1(_read_time_points(beat_path, downbeats=True), _read_time_points(_find_result_artifact(case_root, manifest, ("beat_grid.json",)), downbeats=True)) if beat_path else None
    except Exception as exc:  # benchmark must record a crash instead of hiding it
        result.update({"status": "crashed", "crash": True, "reason": f"{type(exc).__name__}: {exc}"})
    return result


def build_report(registry: Mapping[str, Any], *, result_root: Path | None = None) -> dict[str, Any]:
    cases = [evaluate_case(item, result_root=result_root) for item in registry["cases"]]
    evaluated = [item for item in cases if item["status"] == "evaluated"]
    crashed = [item for item in cases if item["crash"] is True]
    return {
        "schema_version": "1.0",
        "registry": str(DEFAULT_REGISTRY),
        "result_root": str(result_root) if result_root else None,
        "registered_count": len(cases),
        "evaluated_count": len(evaluated),
        "crash_count": len(crashed),
        "accuracy_claim_ready": False,
        "accuracy_claim_reason": "本机只登记了少量候选；没有完整30段结果，不据此声称节奏误差下降20%。",
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--result-root", type=Path, help="按 case id/manifest.json 提供已完成服务结果")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-inputs", action="store_true", help="缺少登记的本机输入时以失败退出")
    parser.add_argument("--check", action="store_true", help="只登记并校验输入，不评估服务结果")
    args = parser.parse_args(argv)
    registry = _load_registry(args.manifest.resolve())
    report = build_report(registry, result_root=None if args.check else (args.result_root.resolve() if args.result_root else None))
    missing = [item["id"] for item in report["cases"] if not item["input"]["available"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "registered": report["registered_count"], "evaluated": report["evaluated_count"], "crashed": report["crash_count"], "missing_inputs": missing}, ensure_ascii=False))
    return 2 if args.strict_inputs and missing else 0


if __name__ == "__main__":
    sys.exit(main())
