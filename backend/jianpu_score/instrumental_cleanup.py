"""Strict postprocessing for duplicated instrumental model events.

MuScriptor can emit the same short event more than once.  This module only
coalesces events whose complete JSON semantics match after a very small,
explicit timestamp normalization.  It does not merge nearby notes, overlap
notes with different fields, or alter the immutable model raw payload.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


INSTRUMENTAL_CLEANUP_SCHEMA_VERSION = "1.0"
TIMESTAMP_DECIMAL_PLACES = 9


@dataclass(frozen=True)
class InstrumentalCleanupResult:
    """Cleaned model events and their source-accounting report."""

    events: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _normalized_timestamp(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"instrumental note {label} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"instrumental note {label} must be finite and non-negative")
    return float(round(result, TIMESTAMP_DECIMAL_PLACES))


def _semantic_key(note: Mapping[str, Any], *, start_sec: float, end_sec: float) -> str:
    """Build a stable key from every model field except lineage internals."""

    value = {
        str(key): deepcopy(item)
        for key, item in note.items()
        if not str(key).startswith("_instrumental_cleanup")
    }
    value["start_sec"] = start_sec
    value["end_sec"] = end_sec
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("instrumental model note contains a non-JSON semantic field") from exc
    return encoded


def clean_instrumental_model_notes(notes: Iterable[Mapping[str, Any]]) -> InstrumentalCleanupResult:
    """Coalesce only exact model duplicates and retain complete lineage.

    Timestamp keys are rounded to nine decimal places solely to remove binary
    floating-point spelling differences.  Events separated by at least one
    nanosecond, or by any semantic field difference (including voice,
    instrument, program, confidence, velocity, and drum status), remain
    independent.
    """

    materialized: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    normalized: dict[int, tuple[float, float]] = {}
    fingerprints: dict[int, str] = {}
    for source_index, raw in enumerate(notes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"instrumental model note {source_index} is not an object")
        item = deepcopy(dict(raw))
        start_sec = _normalized_timestamp(item.get("start_sec"), label="start_sec")
        end_sec = _normalized_timestamp(item.get("end_sec"), label="end_sec")
        if end_sec <= start_sec:
            raise ValueError(
                f"instrumental model note {source_index} has non-positive interval "
                f"{start_sec}:{end_sec}"
            )
        if item.get("midi") is None:
            raise ValueError(f"instrumental model note {source_index} lacks midi")
        try:
            int(item["midi"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"instrumental model note {source_index} has invalid midi") from exc
        key = _semantic_key(item, start_sec=start_sec, end_sec=end_sec)
        materialized.append(item)
        normalized[source_index] = (start_sec, end_sec)
        fingerprints[source_index] = hashlib.sha256(key.encode("utf-8")).hexdigest()
        groups[key].append(source_index)

    cleaned: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for indices in sorted(groups.values(), key=lambda values: values[0]):
        primary_index = indices[0]
        duplicate_indices = indices[1:]
        primary = deepcopy(materialized[primary_index])
        start_sec, end_sec = normalized[primary_index]
        primary["_instrumental_cleanup"] = {
            "primary_source_index": primary_index,
            "source_indices": list(indices),
            "merged_source_indices": list(duplicate_indices),
        }
        cleaned.append(primary)
        if not duplicate_indices:
            continue
        duplicate_groups.append(
            {
                "primary_source_index": primary_index,
                "merged_source_indices": list(duplicate_indices),
                "source_indices": list(indices),
                "reason": "exact_model_duplicate",
                "normalized_start_sec": start_sec,
                "normalized_end_sec": end_sec,
                "midi": int(primary["midi"]),
                "semantic_fingerprint": fingerprints[primary_index],
            }
        )
        for duplicate_index in duplicate_indices:
            duplicate_start, duplicate_end = normalized[duplicate_index]
            merged.append(
                {
                    "source_index": duplicate_index,
                    "primary_source_index": primary_index,
                    "reason": "exact_model_duplicate",
                    "normalized_start_sec": duplicate_start,
                    "normalized_end_sec": duplicate_end,
                    "midi": int(materialized[duplicate_index]["midi"]),
                    "semantic_fingerprint": fingerprints[duplicate_index],
                }
            )

    source_count = len(materialized)
    cleaned_count = len(cleaned)
    merged_count = len(merged)
    if cleaned_count + merged_count != source_count:
        raise RuntimeError("instrumental cleanup source accounting is inconsistent")
    report = {
        "schema_version": INSTRUMENTAL_CLEANUP_SCHEMA_VERSION,
        "policy": "exact_model_duplicate_only",
        "timestamp_normalization": {
            "decimal_places": TIMESTAMP_DECIMAL_PLACES,
            "method": "round_for_key_only_preserve_primary_raw_values",
        },
        "source_note_count": source_count,
        "cleaned_note_count": cleaned_count,
        "matched_count": cleaned_count,
        "merged_count": merged_count,
        "accounted_source_count": source_count,
        "unresolved_count": 0,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "merged": merged,
        "source_accounting": {
            "source_note_count": source_count,
            "primary_note_count": cleaned_count,
            "merged_note_count": merged_count,
            "accounted_source_count": source_count,
            "unresolved_count": 0,
        },
    }
    return InstrumentalCleanupResult(tuple(cleaned), report)
