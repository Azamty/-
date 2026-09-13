"""BeatNet beat-grid contracts and deterministic seconds-to-beats mapping.

BeatNet's DBN output contains beat times and a beat number within the bar.  It
does not expose calibrated confidence, and its 1.1.3 DBN decoder only knows
2, 3 and 4 beats per bar.  This module keeps those facts explicit: meter and
confidence are derived from the returned beat sequence and optional onset
evidence, while every seconds-to-beat operation remains a pure piecewise
linear interpolation (with linear extrapolation outside the observed range).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .domain import normalize_time_signature


BEATNET_MODE = "offline"
BEATNET_INFERENCE = "DBN"
TIME_SIGNATURE_CANDIDATES = ("2/4", "3/4", "4/4", "6/8")
METER_BEAT_COUNTS = {"2/4": 2, "3/4": 3, "4/4": 4, "6/8": 6}
CONFIDENCE_WARNING_THRESHOLD = 0.65
TEMPO_FACTORS = (0.5, 1.0, 2.0)
ONSET_WEIGHTS = {"drums": 1.5, "bass": 1.25, "full_track": 1.0, "all": 1.0}
BEAT_UNIT_SCHEMA_VERSION = "1.0"

# A BeatNet row is a pulse observation.  Its numeric index is not, by itself,
# a duration in quarter notes.  Simple meters have an unambiguous quarter-note
# pulse here.  Compound 6/8 needs an explicit caller definition because the
# same notation can be tracked as six eighth-note pulses or two dotted-quarter
# pulses.
BEAT_UNIT_ALIASES = {
    "quarter": "quarter",
    "quarter_note": "quarter",
    "quarter-pulse": "quarter",
    "quarter_pulse": "quarter",
    "eighth": "eighth",
    "eighth_note": "eighth",
    "eighth-pulse": "eighth",
    "eighth_pulse": "eighth",
    "six_eighth_pulses": "eighth",
    "dotted_quarter": "dotted_quarter",
    "dotted-quarter": "dotted_quarter",
    "dotted_quarter_note": "dotted_quarter",
    "dotted_quarter_pulse": "dotted_quarter",
    "two_dotted_quarter_pulses": "dotted_quarter",
}


class BeatGridError(ValueError):
    """Raised when a BeatNet result cannot form a valid beat grid."""


def _normalize_beat_unit_name(value: Any, *, label: str = "beat unit") -> str:
    if not isinstance(value, str):
        raise BeatGridError(f"{label} must be one of quarter, eighth, dotted_quarter")
    normalized = value.strip().casefold().replace(" ", "_")
    try:
        return BEAT_UNIT_ALIASES[normalized]
    except KeyError as exc:
        raise BeatGridError(
            f"{label} must be one of quarter, eighth, dotted_quarter"
        ) from exc


def resolve_beat_unit(
    time_signature: str,
    beat_unit_definition: str | None = None,
    *,
    legacy_compat: bool = False,
) -> dict[str, Any]:
    """Resolve a beat row's explicit duration in score quarter notes.

    The resolver never uses an observed interval to choose a compound pulse.
    For 6/8, callers must declare either six eighth-note pulses or two
    dotted-quarter pulses.  ``legacy_compat`` is reserved for reading an old
    grid that had no unit field; it preserves its historical quarter-index
    mapping and marks the result as unproven so an evaluator can fail closed.
    """

    meter = normalize_time_signature(time_signature)
    if meter != "6/8":
        if beat_unit_definition is not None and _normalize_beat_unit_name(beat_unit_definition) != "quarter":
            raise BeatGridError(f"{meter} requires a quarter-note beat unit")
        beats_per_bar = METER_BEAT_COUNTS[meter]
        return {
            "schema_version": BEAT_UNIT_SCHEMA_VERSION,
            "beat_unit": "quarter",
            "beat_duration_quarters": 1.0,
            "beats_per_bar": beats_per_bar,
            "bar_duration_quarters": float(beats_per_bar),
            "source": "standard_meter_definition",
            "proven": True,
            "legacy_default_applied": False,
            "pulse_definition": "quarter_note_pulse",
        }

    if beat_unit_definition is None:
        if not legacy_compat:
            raise BeatGridError(
                "6/8 beat unit is not provable; explicitly declare "
                "eighth_pulse or dotted_quarter_pulse"
            )
        # Old beat_grid payloads treated every row as one quarter-note index.
        # Keep that mapping readable for old artifacts, but make its semantic
        # status explicit so it cannot pass a new meter/unit gate.
        return {
            "schema_version": BEAT_UNIT_SCHEMA_VERSION,
            "beat_unit": "legacy_quarter_index",
            "beat_duration_quarters": 1.0,
            "beats_per_bar": METER_BEAT_COUNTS["6/8"],
            "bar_duration_quarters": float(METER_BEAT_COUNTS["6/8"]),
            "source": "legacy_schema_default",
            "proven": False,
            "legacy_default_applied": True,
            "pulse_definition": None,
            "warning": "旧 beat_grid 未声明 6/8 脉冲单位，保留历史四分拍索引；无法作为语义正确的 6/8 通过验收",
        }

    unit = _normalize_beat_unit_name(beat_unit_definition)
    if unit == "eighth":
        return {
            "schema_version": BEAT_UNIT_SCHEMA_VERSION,
            "beat_unit": "eighth",
            "beat_duration_quarters": 0.5,
            "beats_per_bar": 6,
            "bar_duration_quarters": 3.0,
            "source": "explicit_candidate_definition",
            "proven": True,
            "legacy_default_applied": False,
            "pulse_definition": "six_eighth_pulses",
        }
    if unit == "dotted_quarter":
        return {
            "schema_version": BEAT_UNIT_SCHEMA_VERSION,
            "beat_unit": "dotted_quarter",
            "beat_duration_quarters": 1.5,
            "beats_per_bar": 2,
            "bar_duration_quarters": 3.0,
            "source": "explicit_candidate_definition",
            "proven": True,
            "legacy_default_applied": False,
            "pulse_definition": "two_dotted_quarter_pulses",
        }
    raise BeatGridError(
        "6/8 beat unit must be eighth_pulse or dotted_quarter_pulse"
    )


def beat_unit_from_grid(
    grid: Mapping[str, Any],
    *,
    time_signature: str | None = None,
    legacy_compat: bool = True,
) -> dict[str, Any]:
    """Read beat-unit semantics from a grid, with an auditable old-grid path."""

    if not isinstance(grid, Mapping):
        raise BeatGridError("beat_grid must be an object")
    raw_meter = time_signature
    if raw_meter is None:
        raw_meter = grid.get("time_signature", "4/4")
        if isinstance(raw_meter, Mapping):
            raw_meter = raw_meter.get("selected", "4/4")
    meter = normalize_time_signature(str(raw_meter))
    mapping = grid.get("mapping")
    mapping = mapping if isinstance(mapping, Mapping) else {}
    raw_unit = grid.get("beat_unit_definition", grid.get("beat_unit"))
    if raw_unit is None:
        raw_unit = mapping.get("beat_unit_definition", mapping.get("beat_unit"))
    if raw_unit is None:
        return resolve_beat_unit(meter, legacy_compat=legacy_compat)
    resolved = resolve_beat_unit(meter, str(raw_unit), legacy_compat=False)
    raw_duration = grid.get("beat_duration_quarters", mapping.get("beat_duration_quarters"))
    if raw_duration is not None:
        duration = _finite_float(raw_duration, label="beat_duration_quarters")
        if abs(duration - float(resolved["beat_duration_quarters"])) > 1e-9:
            raise BeatGridError(
                "beat_grid beat_duration_quarters conflicts with its explicit beat unit"
            )
    raw_source = grid.get("beat_unit_source")
    if raw_source is None:
        raw_source = mapping.get("beat_unit_source")
    if raw_source is not None:
        resolved = {**resolved, "source": str(raw_source)}
    raw_state_count = grid.get("dbn_position_count")
    if raw_state_count is None:
        raw_state_count = mapping.get("dbn_position_count")
    if raw_state_count is not None:
        try:
            resolved = {**resolved, "state_position_count": int(raw_state_count)}
        except (TypeError, ValueError) as exc:
            raise BeatGridError("dbn_position_count must be an integer") from exc
    return resolved


@dataclass(frozen=True)
class BeatObservation:
    """One normalized BeatNet observation."""

    time_sec: float
    beat_number: int | None = None
    downbeat: bool = False


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BeatGridError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise BeatGridError(f"{label} must be a finite number")
    return result


def normalize_beat_observations(values: Iterable[Any]) -> tuple[BeatObservation, ...]:
    """Normalize BeatNet rows or fixture mappings without hiding bad data.

    BeatNet 1.1.3 returns rows shaped ``(time_sec, beat_number)``.  Fixtures
    may use mappings so downbeat and meter behavior can be tested directly.
    """

    observations: list[BeatObservation] = []
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            raw_time = value.get("time_sec", value.get("time"))
            raw_number = value.get("beat_number", value.get("beat"))
            raw_downbeat = value.get("downbeat", raw_number == 1)
        else:
            try:
                raw_time = value[0]
                raw_number = value[1] if len(value) > 1 else None
            except (IndexError, TypeError, KeyError) as exc:
                raise BeatGridError(f"invalid BeatNet observation at index {index}") from exc
            raw_downbeat = raw_number == 1
        time_sec = _finite_float(raw_time, label=f"beat[{index}].time_sec")
        if time_sec < 0:
            raise BeatGridError(f"beat[{index}].time_sec cannot be negative")
        beat_number: int | None
        if raw_number is None:
            beat_number = None
        else:
            try:
                beat_number = int(raw_number)
            except (TypeError, ValueError) as exc:
                raise BeatGridError(f"beat[{index}].beat_number must be an integer") from exc
            if beat_number <= 0:
                raise BeatGridError(f"beat[{index}].beat_number must be positive")
        observations.append(BeatObservation(time_sec, beat_number, bool(raw_downbeat)))
    if len(observations) < 2:
        raise BeatGridError("BeatNet must return at least two beat observations")
    for left, right in zip(observations, observations[1:]):
        if right.time_sec <= left.time_sec:
            raise BeatGridError("BeatNet beat times must be strictly increasing")
    return tuple(observations)


def _as_times(values: Sequence[float]) -> tuple[float, ...]:
    times = tuple(_finite_float(value, label="beat time") for value in values)
    if len(times) < 2:
        raise BeatGridError("at least two beat times are required")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise BeatGridError("beat times must be strictly increasing")
    if times[0] < 0:
        raise BeatGridError("beat times cannot be negative")
    return times


def _intervals(times: Sequence[float]) -> tuple[float, ...]:
    return tuple(right - left for left, right in zip(times, times[1:]))


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    centre = sum(values) / len(values)
    if centre <= 0:
        return 1.0
    return math.sqrt(sum((value - centre) ** 2 for value in values) / len(values)) / centre


def _dbn_meter_state_definition(
    observations: Sequence[BeatObservation],
) -> dict[str, Any] | None:
    """Return a compound pulse definition only from DBN position state.

    BeatNet's state labels are the auditable source of pulse cardinality.  A
    wall-clock interval is intentionally never consulted here: six positions
    mean six eighth pulses, while two positions mean two dotted-quarter
    pulses.  Other state shapes cannot prove a 6/8 pulse unit.
    """

    runs: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for item in observations:
        number = item.beat_number
        # A missing DBN state cannot be treated as an invisible interior gap.
        # The pulse unit is only proven when every observed state participates
        # in one of the auditable boundary or complete runs below.
        if number is None:
            return None
        if current and (item.downbeat or number == 1 or (previous is not None and number <= previous)):
            runs.append(current)
            current = []
        current.append(number)
        previous = number
    if current:
        runs.append(current)
    if not runs:
        return None

    def is_complete(run: list[int], expected: list[int]) -> bool:
        return run == expected

    def is_suffix(run: list[int], expected: list[int]) -> bool:
        return bool(run) and len(run) <= len(expected) and run == expected[-len(run) :]

    def is_prefix(run: list[int], expected: list[int]) -> bool:
        return bool(run) and len(run) <= len(expected) and run == expected[: len(run)]

    def supports(expected: list[int]) -> bool:
        # A sole run has no boundary context: it must itself be a complete
        # cycle.  With multiple runs, only the first and last may be partial;
        # every interior run must be complete.  Requiring a complete run also
        # prevents two compatible-looking boundary fragments from proving a
        # pulse unit on their own.
        if len(runs) == 1:
            return is_complete(runs[0], expected)
        return (
            any(is_complete(run, expected) for run in runs)
            and (is_complete(runs[0], expected) or is_suffix(runs[0], expected))
            and (is_complete(runs[-1], expected) or is_prefix(runs[-1], expected))
            and all(is_complete(run, expected) for run in runs[1:-1])
        )

    definitions = (
        (list(range(1, 7)), "eighth_pulse", "eighth", 6),
        ([1, 2], "dotted_quarter_pulse", "dotted_quarter", 2),
    )
    for expected, definition, unit, position_count in definitions:
        if supports(expected):
            return {
                "beat_unit_definition": definition,
                "beat_unit": unit,
                "position_count": position_count,
                "source": "dbn_meter_state_definition",
            }
    return None


def infer_time_signature(
    observations: Sequence[BeatObservation],
    *,
    meter_hint: str | None = None,
    independent_accent_times: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Infer meter candidates from bar resets and explicit compound accents.

    A 3-beat reset is intentionally ambiguous between 3/4 and 6/8 for the
    BeatNet 1.1.3 DBN output.  Accent evidence at the midpoint of a six-beat
    group is required to prefer 6/8; otherwise both candidates remain visible
    and the confidence is reduced.
    """

    state_definition = _dbn_meter_state_definition(observations)
    if meter_hint is not None:
        selected = normalize_time_signature(meter_hint)
        return {
            "selected": selected,
            "confidence": 1.0,
            "source": "manual",
            "candidates": [
                {
                    "value": value,
                    "score": 1.0 if value == selected else 0.0,
                    "beat_unit_definitions": (
                        ["eighth_pulse", "dotted_quarter_pulse"]
                        if value == "6/8"
                        else ["quarter_pulse"]
                    ),
                    "dbn_position_count": (
                        state_definition["position_count"]
                        if value == "6/8" and state_definition
                        else None
                    ),
                    "beat_unit_definition": (
                        state_definition["beat_unit_definition"]
                        if value == "6/8" and state_definition
                        else None
                    ),
                }
                for value in TIME_SIGNATURE_CANDIDATES
            ],
            "warning": None,
            "beat_unit_definition": (
                state_definition["beat_unit_definition"]
                if selected == "6/8" and state_definition
                else None
            ),
            "beat_unit_source": (
                state_definition["source"]
                if selected == "6/8" and state_definition
                else None
            ),
            "dbn_position_count": (
                state_definition["position_count"]
                if selected == "6/8" and state_definition
                else None
            ),
        }

    times = tuple(item.time_sec for item in observations)
    downbeats = [index for index, item in enumerate(observations) if item.downbeat or item.beat_number == 1]
    intervals = _intervals(times)
    stability = max(0.0, 1.0 - min(1.0, _coefficient_of_variation(intervals)))
    reset_distances = [right - left for left, right in zip(downbeats, downbeats[1:])]
    # BeatNet's own downbeat labels are not independent audio evidence.  The
    # caller must explicitly provide an independent accent stream before a
    # meter can receive confidence above the warning threshold.
    accent_set = tuple(
        sorted(_finite_float(value, label="independent accent time") for value in (independent_accent_times or ()))
    )

    scores: dict[str, float] = {}
    compound_accent_evidence = False
    for value in TIME_SIGNATURE_CANDIDATES:
        count = METER_BEAT_COUNTS[value]
        if reset_distances:
            # A reset every six beats is evidence for 6/8, while a reset every
            # four beats is evidence for 4/4.  Treating ``distance % count``
            # as zero would incorrectly make every divisor look perfect and
            # would silently turn 4/4 into 2/4.
            distance_error = sum(abs(distance - count) for distance in reset_distances) / len(reset_distances)
            reset_score = max(0.0, 1.0 - distance_error / max(count, 1))
        else:
            reset_score = 0.35 if value == "4/4" else 0.25
        score = 0.75 * reset_score + 0.25 * stability
        if value == "6/8" and reset_distances and all(distance in {3, 6} for distance in reset_distances):
            # BeatNet may expose dotted-quarter pulses (distance 3) for
            # compound meter.  Without a secondary accent this is still only
            # a candidate, not evidence of a direct 6/8 prediction.
            midpoint_accents = 0
            for downbeat_index, index in enumerate(downbeats[:-1]):
                next_downbeat = downbeats[downbeat_index + 1]
                midpoint = times[index] + (times[next_downbeat] - times[index]) / 2.0
                if any(abs(accent - midpoint) <= 0.12 for accent in accent_set):
                    midpoint_accents += 1
            required_accents = max(2, math.ceil(0.6 * max(1, len(downbeats) - 1)))
            if midpoint_accents >= required_accents:
                compound_accent_evidence = True
                score = min(1.0, score + 0.45 * midpoint_accents / max(1, len(downbeats) - 1))
                if "3/4" in scores:
                    score = max(score, scores["3/4"] + 0.01)
        scores[value] = max(0.0, min(1.0, score))

    ranked = sorted(scores.items(), key=lambda item: (-item[1], TIME_SIGNATURE_CANDIDATES.index(item[0])))
    selected, top_score = ranked[0]
    meter_warning: str | None = None
    if selected == "6/8" and not compound_accent_evidence:
        # BeatNet 1.1.3 cannot natively distinguish a compound 6/8 bar from
        # other beat-number interpretations.  Keep 6/8 in candidates but do
        # not silently select it without independent compound-accent support.
        selected = "4/4"
        top_score = scores[selected]
        meter_warning = "BeatNet 未获得跨多数小节的独立复合拍重音证据，6/8 仅保留为候选"
    if compound_accent_evidence and "6/8" in scores and "3/4" in scores and scores["6/8"] >= scores["3/4"]:
        selected, top_score = "6/8", scores["6/8"]
    tied = [value for value, score in ranked if abs(score - top_score) < 0.08]
    if selected == "6/8" and compound_accent_evidence:
        tied = ["6/8"]
    confidence = top_score
    warning: str | None = meter_warning
    compound_ambiguous = (
        selected == "3/4"
        and reset_distances
        and all(distance == 3 for distance in reset_distances)
        and not accent_set
    )
    if compound_ambiguous:
        confidence = min(confidence, 0.55)
        tied = ["3/4", "6/8"]
        warning = "BeatNet 的三拍重拍同时可能是 3/4 或 6/8；缺少复合拍重音证据，请确认拍号"
    if len(tied) > 1:
        confidence = min(confidence, 0.55)
        if warning is None:
            warning = "BeatNet 的重拍证据无法区分 " + "、".join(tied) + "；请确认拍号"
    if not accent_set:
        confidence = min(confidence, 0.55)
        if warning is None:
            warning = "未提供独立音频重音证据；BeatNet 拍号仅作候选，请确认拍号"
    elif confidence < CONFIDENCE_WARNING_THRESHOLD:
        warning = f"拍号识别置信度较低（{confidence:.2f}），请确认 {selected}"
    return {
        "selected": selected,
        "confidence": round(confidence, 4),
        "source": "beatnet_derived",
        "candidates": [
            {
                "value": value,
                "score": round(scores[value], 4),
                "beat_unit_definitions": (
                    ["eighth_pulse", "dotted_quarter_pulse"]
                    if value == "6/8"
                    else ["quarter_pulse"]
                ),
                "dbn_position_count": (
                    state_definition["position_count"]
                    if value == "6/8" and state_definition
                    else None
                ),
                "beat_unit_definition": (
                    state_definition["beat_unit_definition"]
                    if value == "6/8" and state_definition
                    else None
                ),
            }
            for value in TIME_SIGNATURE_CANDIDATES
        ],
        "warning": warning,
        "beat_unit_definition": (
            state_definition["beat_unit_definition"]
            if selected == "6/8" and state_definition
            else None
        ),
        "beat_unit_source": (
            state_definition["source"]
            if selected == "6/8" and state_definition
            else None
        ),
        "dbn_position_count": (
            state_definition["position_count"]
            if selected == "6/8" and state_definition
            else None
        ),
    }


def _candidate_times(times: Sequence[float], factor: float) -> tuple[float, ...]:
    """Build half/original/double beat grids while preserving the first beat."""

    source = _as_times(times)
    if factor == 1.0:
        return source
    if factor == 0.5:
        values = source[::2]
        if values[-1] != source[-1]:
            values += (source[-1],)
        return values if len(values) >= 2 else source
    if factor == 2.0:
        values: list[float] = []
        for left, right in zip(source, source[1:]):
            values.extend((left, (left + right) / 2.0))
        values.append(source[-1])
        return tuple(values)
    raise BeatGridError(f"unsupported tempo candidate factor: {factor}")


def _nearest_distance(value: float, points: Sequence[float]) -> float:
    return min(abs(value - point) for point in points)


def _onset_evidence(onsets: Mapping[str, Sequence[float]] | Sequence[float] | None) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if onsets is None:
        return ()
    if isinstance(onsets, Mapping):
        result: list[tuple[str, tuple[float, ...]]] = []
        for source, values in onsets.items():
            cleaned = tuple(sorted(_finite_float(value, label=f"{source} onset") for value in values))
            result.append((str(source), cleaned))
        return tuple(result)
    cleaned = tuple(sorted(_finite_float(value, label="onset") for value in onsets))
    return (("all", cleaned),)


def _score_candidate(
    beat_times: Sequence[float],
    onsets: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    beats_per_bar: int,
    factor: float,
    strong_octave_evidence: bool,
) -> tuple[float, float, float, float, float]:
    if not onsets:
        prior = 0.0 if factor == 1.0 else (0.18 if strong_octave_evidence else 0.6)
        return prior, 0.0, 0.0, 1.0, 1.0
    intervals = _intervals(beat_times)
    base_interval = float(median(intervals))
    dedupe_tolerance = max(0.035, base_interval * 0.08)
    deduped = tuple(
        (source, _dedupe_onsets(values, tolerance=dedupe_tolerance))
        for source, values in onsets
    )
    support_tolerance = max(0.06, base_interval * 0.16)
    weighted_error = 0.0
    total_weight = 0.0
    supported_weight = 0.0
    for source, values in deduped:
        weight = ONSET_WEIGHTS.get(source, 1.0)
        for onset in values:
            distance = _nearest_distance(onset, beat_times)
            weighted_error += weight * (distance / max(base_interval, 1e-9))
            if distance <= support_tolerance:
                supported_weight += weight
            total_weight += weight
    onset_error = weighted_error / total_weight if total_weight else 0.0

    combined = [value for _source, values in deduped for value in values]
    if combined:
        lower = min(combined) - support_tolerance
        upper = max(combined) + support_tolerance
        candidate_points = [value for value in beat_times if lower <= value <= upper]
    else:
        candidate_points = []
    supported_points = sum(
        _nearest_distance(point, combined) <= support_tolerance for point in candidate_points
    )
    precision = supported_points / len(candidate_points) if candidate_points else 1.0
    coverage = supported_weight / total_weight if total_weight else 1.0

    # Compare the amount of onset error in each bar.  This favors a candidate
    # whose alignment is stable across the track instead of one that wins only
    # because of a dense opening.
    bar_errors: list[float] = []
    for start in range(0, len(beat_times) - 1, beats_per_bar):
        end = min(len(beat_times), start + beats_per_bar)
        points = beat_times[start:end]
        if not points:
            continue
        bar_onsets = [value for value in combined if points[0] - base_interval / 2 <= value <= points[-1] + base_interval / 2]
        if bar_onsets:
            bar_errors.append(sum(_nearest_distance(value, points) / max(base_interval, 1e-9) for value in bar_onsets) / len(bar_onsets))
    stability = _coefficient_of_variation([max(0.0, 1.0 - error) for error in bar_errors]) if len(bar_errors) > 1 else 0.0
    prior = 0.0 if factor == 1.0 else (0.18 if strong_octave_evidence else 0.6)
    # A denser candidate must explain its additional grid points.  The prior
    # blocks full-track eighth-note subdivisions from flipping to double time;
    # drums/bass evidence is allowed to overcome it when it supports the
    # octave change consistently.
    score = onset_error + 0.5 * (1.0 - precision) + 0.4 * (1.0 - coverage) + 0.25 * stability + prior
    return score, onset_error, stability, precision, coverage


def _dedupe_onsets(values: Sequence[float], *, tolerance: float) -> tuple[float, ...]:
    """Collapse chord/onset clusters so one chord cannot dominate a score."""

    ordered = sorted(values)
    if not ordered:
        return ()
    result = [ordered[0]]
    for value in ordered[1:]:
        if value - result[-1] > tolerance:
            result.append(value)
    return tuple(result)


def choose_tempo_candidate(
    beat_times: Sequence[float],
    *,
    source_onsets: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    time_signature: str = "4/4",
    manual_bpm: float | None = None,
    beat_duration_quarters: float = 1.0,
    beats_per_bar: int | None = None,
) -> dict[str, Any]:
    """Rank half/original/double tempo interpretations from onset evidence."""

    source = _as_times(beat_times)
    normalized_meter = normalize_time_signature(time_signature)
    duration_quarters = _finite_float(beat_duration_quarters, label="beat_duration_quarters")
    if duration_quarters <= 0:
        raise BeatGridError("beat_duration_quarters must be greater than zero")
    resolved_beats_per_bar = int(beats_per_bar if beats_per_bar is not None else METER_BEAT_COUNTS[normalized_meter])
    if resolved_beats_per_bar <= 0:
        raise BeatGridError("beats_per_bar must be greater than zero")
    evidence = _onset_evidence(source_onsets)
    strong_octave_evidence = any(source in {"drums", "bass"} and values for source, values in evidence)
    intervals = _intervals(source)
    detected_bpm = 60.0 * duration_quarters / float(median(intervals))
    candidates: list[dict[str, Any]] = []
    for factor in TEMPO_FACTORS:
        candidate_grid = _candidate_times(source, factor)
        score, onset_error, stability, precision, coverage = _score_candidate(
            candidate_grid,
            evidence,
            beats_per_bar=resolved_beats_per_bar,
            factor=factor,
            strong_octave_evidence=strong_octave_evidence,
        )
        label = "half" if factor == 0.5 else "original" if factor == 1.0 else "double"
        candidates.append(
            {
                "label": label,
                "factor": factor,
                "bpm": round(detected_bpm * factor, 4),
                "beat_duration_quarters": duration_quarters,
                "beats_per_bar": resolved_beats_per_bar,
                "bar_duration_quarters": round(duration_quarters * resolved_beats_per_bar, 9),
                "score": round(score, 6),
                "onset_error": round(onset_error, 6),
                "bar_stability": round(stability, 6),
                "grid_precision": round(precision, 6),
                "onset_coverage": round(coverage, 6),
                "prior_penalty": round(0.0 if factor == 1.0 else (0.18 if strong_octave_evidence else 0.6), 6),
                "selected": False,
                "beat_times": list(candidate_grid),
            }
        )
    selected = min(candidates, key=lambda item: (float(item["score"]), abs(float(item["factor"]) - 1.0)))
    if manual_bpm is not None:
        selected_bpm = _finite_float(manual_bpm, label="manual BPM")
        if selected_bpm <= 0:
            raise BeatGridError("manual BPM must be greater than zero")
        # Manual BPM changes the scale of the observed beat positions while
        # retaining BeatNet's first beat and local timing shape.
        selected = next(item for item in candidates if item["factor"] == 1.0)
        selected_bpm = selected_bpm
        manual_scale = selected_bpm / detected_bpm
        selection_reason = "用户手动 BPM 优先；保留 BeatNet 首拍与局部拍点并按手动 BPM 缩放"
    else:
        selected_bpm = float(selected["bpm"])
        manual_scale = 1.0
        if not evidence:
            selection_reason = "无 MuScriptor/全轨 onset 证据，保留原速 BeatNet 网格"
        elif not strong_octave_evidence:
            selection_reason = (
                "仅有全轨/非鼓贝斯 onset；候选拍点精确率与原速先验阻止仅凭细分翻倍"
            )
        else:
            selection_reason = (
                f"按鼓/贝斯/全轨 onset 对齐误差、候选精确率与小节稳定性选择 {selected['label']} 速度"
            )
    for item in candidates:
        item["selected"] = item is selected
        item["rationale"] = (
            "用户 BPM 覆盖"
            if manual_bpm is not None and item is selected
            else "候选速度，分数越低越好"
        )
    return {
        "detected_bpm": round(detected_bpm, 4),
        "selected_bpm": round(selected_bpm, 4),
        "selected_factor": selected["factor"],
        "manual_bpm": manual_bpm,
        "manual_scale": round(manual_scale, 8),
        "beat_duration_quarters": duration_quarters,
        "beats_per_bar": resolved_beats_per_bar,
        "bar_duration_quarters": round(duration_quarters * resolved_beats_per_bar, 9),
        "selection_reason": selection_reason,
        "candidates": candidates,
        "evidence_sources": [source for source, values in evidence if values],
    }


def _bar_records(
    beat_times: Sequence[float],
    downbeats: Sequence[bool],
    *,
    beats_per_bar: int,
    beat_duration_quarters: float = 1.0,
    beat_scale: float = 1.0,
) -> list[dict[str, Any]]:
    starts = [index for index, is_downbeat in enumerate(downbeats) if is_downbeat]
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    starts = sorted(set(index for index in starts if 0 <= index < len(beat_times)))
    bars: list[dict[str, Any]] = []
    for bar_index, start_index in enumerate(starts):
        next_explicit = starts[bar_index + 1] if bar_index + 1 < len(starts) else None
        end_index = next_explicit if next_explicit is not None else min(len(beat_times), start_index + beats_per_bar)
        if end_index <= start_index:
            continue
        local = _intervals(beat_times[start_index : min(len(beat_times), end_index + 1)])
        local_bpm = (
            60.0 * beat_duration_quarters * beat_scale / float(median(local))
            if local
            else 0.0
        )
        bars.append(
            {
                "index": bar_index,
                "start_beat_index": start_index,
                "end_beat_index": end_index,
                "start_sec": round(beat_times[start_index], 9),
                "end_sec": round(beat_times[min(end_index, len(beat_times) - 1)], 9),
                "beat_count": end_index - start_index,
                "beat_duration_quarters": beat_duration_quarters,
                "duration_quarters": round(
                    (end_index - start_index) * beat_duration_quarters * beat_scale,
                    9,
                ),
                "start_quarter": round(start_index * beat_duration_quarters * beat_scale, 9),
                "end_quarter": round(end_index * beat_duration_quarters * beat_scale, 9),
                "local_bpm": round(local_bpm, 4),
                "stable": _coefficient_of_variation(local) <= 0.12 if local else False,
            }
        )
    return bars


def build_beat_grid(
    observations: Iterable[Any],
    *,
    duration_sec: float | None = None,
    meter_hint: str | None = None,
    manual_bpm: float | None = None,
    manual_time_signature: str | None = None,
    source_onsets: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    independent_accent_times: Sequence[float] | None = None,
    beat_unit_definition: str | None = None,
    engine: str = "beatnet",
) -> dict[str, Any]:
    """Build the durable ``beat_grid.json`` representation."""

    normalized = normalize_beat_observations(observations)
    meter = infer_time_signature(
        normalized,
        meter_hint=manual_time_signature or meter_hint,
        independent_accent_times=independent_accent_times,
    )
    selected_meter = normalize_time_signature(manual_time_signature or meter["selected"])
    # A newly built 6/8 grid must carry a pulse definition from the DBN meter
    # state (or an explicit fixture/candidate declaration).  Old persisted
    # grids are handled by beat_unit_from_grid(legacy_compat=True) when they
    # are read by downstream consumers.
    derived_definition = meter.get("beat_unit_definition")
    selected_definition = beat_unit_definition or (
        str(derived_definition) if selected_meter == "6/8" and derived_definition else None
    )
    beat_unit = resolve_beat_unit(
        selected_meter,
        selected_definition,
        legacy_compat=False,
    )
    if beat_unit_definition is None and selected_definition is not None and selected_meter == "6/8":
        beat_unit = {
            **beat_unit,
            "source": str(meter.get("beat_unit_source") or "dbn_meter_state_definition"),
            "state_position_count": meter.get("dbn_position_count"),
        }
    tempo = choose_tempo_candidate(
        tuple(item.time_sec for item in normalized),
        source_onsets=source_onsets,
        time_signature=selected_meter,
        manual_bpm=manual_bpm,
        beat_duration_quarters=float(beat_unit["beat_duration_quarters"]),
        beats_per_bar=int(beat_unit["beats_per_bar"]),
    )
    selected_factor = float(tempo["selected_factor"])
    selected_times = tuple(float(value) for value in next(item for item in tempo["candidates"] if item["selected"])["beat_times"])
    raw_downbeats = [item.downbeat or item.beat_number == 1 for item in normalized]
    raw_times = tuple(item.time_sec for item in normalized)
    if selected_factor == 1.0:
        selected_downbeats = raw_downbeats
        selected_numbers = [item.beat_number for item in normalized]
    elif selected_factor == 0.5:
        selected_downbeats = [False] * len(selected_times)
        selected_numbers = []
        for time_sec in selected_times:
            nearest_index = min(range(len(raw_times)), key=lambda index: abs(raw_times[index] - time_sec))
            selected_downbeats[len(selected_numbers)] = raw_downbeats[nearest_index]
            selected_numbers.append(normalized[nearest_index].beat_number)
    else:
        selected_downbeats = [flag for flag in raw_downbeats for _ in (0, 1)]
        selected_downbeats = selected_downbeats[: len(selected_times)]
        selected_numbers = []
        for item in normalized:
            selected_numbers.extend((item.beat_number, None))
        selected_numbers = selected_numbers[: len(selected_times)]
    beats_per_bar = int(beat_unit["beats_per_bar"])
    bars = _bar_records(
        selected_times,
        selected_downbeats,
        beats_per_bar=beats_per_bar,
        beat_duration_quarters=float(beat_unit["beat_duration_quarters"]),
        beat_scale=float(tempo["manual_scale"]),
    )
    downbeat_starts = [index for index, is_downbeat in enumerate(selected_downbeats) if is_downbeat]
    if not downbeat_starts or downbeat_starts[0] != 0:
        downbeat_starts = [0, *downbeat_starts]
    downbeat_starts = sorted(set(downbeat_starts))
    first_downbeat_index = next(
        (index for index, value in enumerate(selected_downbeats) if value),
        None,
    )
    first_downbeat_sec = (
        selected_times[first_downbeat_index]
        if first_downbeat_index is not None
        else selected_times[0]
    )
    duration = None if duration_sec is None else _finite_float(duration_sec, label="duration_sec")
    warnings = [str(meter["warning"])] if meter.get("warning") else []
    if not source_onsets:
        warnings.append("未提供 MuScriptor 鼓/贝斯/全轨 onset；速度候选依据仅为 BeatNet 网格")
    if manual_time_signature is None and meter["confidence"] < CONFIDENCE_WARNING_THRESHOLD:
        warnings.append("拍号未获得足够重拍证据，页面应显示候选而不是静默固定拍号")
    beat_records: list[dict[str, Any]] = []
    for index, time_sec in enumerate(selected_times):
        interval = selected_times[index + 1] - time_sec if index + 1 < len(selected_times) else (time_sec - selected_times[index - 1] if index else 0.0)
        current_bar = max((bar for bar, start in enumerate(downbeat_starts) if start <= index), default=0)
        beat_number = selected_numbers[index] if index < len(selected_numbers) else None
        beat_records.append(
            {
                "index": index,
                "time_sec": round(time_sec, 9),
                "beat_number": int(beat_number) if beat_number is not None else index % beats_per_bar + 1,
                "downbeat": bool(selected_downbeats[index]) if index < len(selected_downbeats) else index % beats_per_bar == 0,
                "bar_index": current_bar,
                "beat_duration_quarters": beat_unit["beat_duration_quarters"],
                "quarter_position": round(
                    index
                    * float(beat_unit["beat_duration_quarters"])
                    * float(tempo["manual_scale"]),
                    9,
                ),
                "local_bpm": round(
                    60.0
                    * float(beat_unit["beat_duration_quarters"])
                    * float(tempo["manual_scale"])
                    / interval,
                    4,
                )
                if interval > 0
                else 0.0,
            }
        )
    return {
        "schema_version": "1.1",
        "beat_unit_schema_version": BEAT_UNIT_SCHEMA_VERSION,
        "engine": engine,
        "mode": BEATNET_MODE,
        "inference": BEATNET_INFERENCE,
        "beats": beat_records,
        "bars": bars,
        "time_signature": {
            **meter,
            "selected": selected_meter,
            "source": "manual" if manual_time_signature else meter["source"],
            "confidence_source": "manual_override" if manual_time_signature else "derived_from_downbeat_periodicity_and_compound_accent",
        },
        "beat_unit_definition": beat_unit["beat_unit"],
        "beat_unit": beat_unit["beat_unit"],
        "beat_duration_quarters": beat_unit["beat_duration_quarters"],
        "beats_per_bar": beat_unit["beats_per_bar"],
        "bar_duration_quarters": beat_unit["bar_duration_quarters"],
        "beat_unit_source": beat_unit["source"],
        "beat_unit_proven": beat_unit["proven"],
        "dbn_position_count": beat_unit.get("state_position_count"),
        "beat_unit_semantics": dict(beat_unit),
        "tempo": tempo,
        "duration_sec": duration,
        "mapping": {
            "beat_times": list(selected_times),
            "seconds_to_beat": "piecewise_linear_with_linear_extrapolation",
            "seconds_to_quarter": "beat_index_times_beat_duration_quarters",
            "position_unit": "quarter_note",
            "manual_bpm_scale": tempo["manual_scale"],
            "beat_unit_definition": beat_unit["beat_unit"],
            "beat_unit": beat_unit["beat_unit"],
            "beat_duration_quarters": beat_unit["beat_duration_quarters"],
            "beats_per_bar": beat_unit["beats_per_bar"],
            "bar_duration_quarters": beat_unit["bar_duration_quarters"],
            "beat_unit_source": beat_unit["source"],
            "beat_unit_proven": beat_unit["proven"],
            "dbn_position_count": beat_unit.get("state_position_count"),
            "first_beat_sec": selected_times[0],
            "score_origin": {
                "strategy": "first_downbeat" if first_downbeat_index is not None else "first_beat_fallback",
                "downbeat_index": first_downbeat_index,
                "downbeat_sec": first_downbeat_sec,
                "pickup_candidate": bool(first_downbeat_index and first_downbeat_index > 0),
                "downbeat_score_beat": (
                    0.0 if first_downbeat_index is not None and not (first_downbeat_index and first_downbeat_index > 0) else None
                ),
                "pickup_beats": 0.0,
                "origin_shift_beats": (
                    -first_downbeat_index * float(tempo["manual_scale"])
                    if first_downbeat_index is not None
                    else 0.0
                ),
            },
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


def seconds_to_beat(
    seconds: float,
    beat_times: Sequence[float],
    *,
    scale: float = 1.0,
    beat_duration_quarters: float = 1.0,
) -> float:
    """Map seconds to a beat position measured in score quarter notes.

    The default duration of one quarter preserves the historical fractional
    beat-index API.  Compound grids pass their explicit pulse duration so a
    six-eighth-pulse bar occupies three, rather than six, score quarters.
    """

    value = _finite_float(seconds, label="seconds")
    times = _as_times(beat_times)
    scale_value = _finite_float(scale, label="beat scale")
    if scale_value <= 0:
        raise BeatGridError("beat scale must be greater than zero")
    duration_value = _finite_float(beat_duration_quarters, label="beat_duration_quarters")
    if duration_value <= 0:
        raise BeatGridError("beat_duration_quarters must be greater than zero")
    if value <= times[0]:
        interval = times[1] - times[0]
        position = (value - times[0]) / interval
    elif value >= times[-1]:
        interval = times[-1] - times[-2]
        position = len(times) - 1 + (value - times[-1]) / interval
    else:
        right = 1
        while times[right] <= value:
            right += 1
        left = right - 1
        position = left + (value - times[left]) / (times[right] - times[left])
    return position * scale_value * duration_value


def map_note_seconds(
    start_sec: float,
    end_sec: float,
    beat_times: Sequence[float],
    *,
    scale: float = 1.0,
    beat_duration_quarters: float = 1.0,
) -> tuple[float, float]:
    """Map a note interval and reject inverted or zero-length source notes."""

    start = _finite_float(start_sec, label="note start_sec")
    end = _finite_float(end_sec, label="note end_sec")
    if end <= start:
        raise BeatGridError("note end_sec must be greater than start_sec")
    mapped_start = seconds_to_beat(
        start,
        beat_times,
        scale=scale,
        beat_duration_quarters=beat_duration_quarters,
    )
    mapped_end = seconds_to_beat(
        end,
        beat_times,
        scale=scale,
        beat_duration_quarters=beat_duration_quarters,
    )
    if mapped_end <= mapped_start:
        raise BeatGridError("mapped note interval is not increasing")
    return mapped_start, mapped_end


def beat_grid_onsets_from_notes(notes: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Extract weighted MuScriptor onset evidence from normalized note records."""

    result: dict[str, list[float]] = {"full_track": []}
    for note in notes:
        start = _finite_float(note.get("start_sec"), label="note start_sec")
        group = str(note.get("instrument_group", "unknown"))
        result["full_track"].append(start)
        if bool(note.get("is_drum")) or group == "drums":
            result.setdefault("drums", []).append(start)
        if "bass" in group:
            result.setdefault("bass", []).append(start)
    return {key: sorted(values) for key, values in result.items() if values}
