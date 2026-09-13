"""Reference-free BeatNet refinement pilot over production-acceptance-v3.

Candidate generation and selection use audio features, the immutable BeatNet
grid, and immutable model-note onsets only. Reference files are opened after a
candidate is selected and are used solely for post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_beatnet_v3 import _f1, _load, _note_groups, _times  # noqa: E402
from scripts.high_accuracy_benchmark import _midi_notes, rhythm_error  # noqa: E402


DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "beatnet-observable-refinement-pilot-v1"
DEFAULT_ORACLE_REPORT = ROOT / ".artifacts" / "review" / "beatnet-v3-error-attribution" / "attribution.json"
SR = 22050
HOP = 512
BEAT_TOLERANCE_SEC = 0.07

POLICY: dict[str, Any] = {
    "schema_version": "observable_refinement_policy_1",
    "reference_free_selection": True,
    "audio": {"sample_rate": SR, "hop_length": HOP, "mono": True},
    "tempo_peak_bpm_range": [40.0, 240.0],
    "tempo_peak_count": 6,
    "affine_scale_range": [0.5, 2.0],
    "affine_scale_perturbations": [-0.02, 0.0, 0.02],
    "phase_offsets_per_period": 25,
    "beat_candidate_weights": {
        "audio_onset_hit": 0.30,
        "tempo_periodicity": 0.20,
        "note_onset_alignment": 0.20,
        "note_onset_coverage": 0.10,
        "audio_peak_coverage": 0.10,
        "local_smoothness": 0.10,
    },
    "downbeat_periods": [2, 3, 4, 6],
    "downbeat_weights": {
        "audio_accent_contrast": 0.40,
        "low_frequency_contrast": 0.25,
        "bass_note_alignment": 0.20,
        "bar_accent_stability": 0.15,
    },
    "meter_theory_prior": {"2": 0.0, "3": 0.015, "4": 0.03, "6": 0.015},
    "weight_origin": "fixed_theory_before_post_hoc_reference_evaluation",
    "rhythm_simulation": "cleaned_service_note_events_piecewise_map_round_48tpq_without_musescore_rerender",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def _sample_envelope(envelope: np.ndarray, times: Sequence[float]) -> list[float]:
    if envelope.size == 0:
        return []
    frames = np.clip(np.rint(np.asarray(times) * SR / HOP).astype(int), 0, len(envelope) - 1)
    return [float(envelope[index]) for index in frames]


def _audio_features(path: Path) -> dict[str, Any]:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    duration = len(audio) / SR
    onset = _normalize(librosa.onset.onset_strength(y=audio, sr=SR, hop_length=HOP))
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=HOP))
    frequencies = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low_energy = np.mean(spectrum[frequencies <= 250.0], axis=0)
    low_onset = _normalize(np.maximum(0.0, np.diff(low_energy, prepend=low_energy[0])))
    peaks = librosa.util.peak_pick(onset, pre_max=2, post_max=2, pre_avg=4, post_avg=4, delta=0.08, wait=2)
    peak_times = librosa.frames_to_time(peaks, sr=SR, hop_length=HOP).tolist()
    tempogram = librosa.feature.tempogram(onset_envelope=onset, sr=SR, hop_length=HOP)
    strengths = np.mean(tempogram, axis=1)
    bpms = librosa.tempo_frequencies(tempogram.shape[0], sr=SR, hop_length=HOP)
    valid = np.flatnonzero((bpms >= 40.0) & (bpms <= 240.0) & np.isfinite(bpms))
    ranked = sorted(valid.tolist(), key=lambda index: (-float(strengths[index]), float(bpms[index])))
    tempo_peaks: list[dict[str, float]] = []
    for index in ranked:
        bpm = float(bpms[index])
        if all(abs(math.log2(bpm / item["bpm"])) > 0.025 for item in tempo_peaks):
            tempo_peaks.append({"bpm": bpm, "strength": float(strengths[index])})
        if len(tempo_peaks) == POLICY["tempo_peak_count"]:
            break
    return {
        "audio": audio,
        "duration_sec": duration,
        "onset": onset,
        "low_onset": low_onset,
        "peak_times": peak_times,
        "tempo_peaks": tempo_peaks,
    }


def _median_interval(beats: Sequence[float]) -> float:
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    return statistics.median(intervals) if intervals else 0.5


def _regularity(beats: Sequence[float]) -> tuple[float, float]:
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    if not intervals:
        return 0.0, 0.0
    mean = statistics.fmean(intervals)
    cv = statistics.pstdev(intervals) / mean if len(intervals) > 1 and mean else 0.0
    smooth = statistics.fmean(abs(c - 2 * b + a) / mean for a, b, c in zip(beats, beats[1:], beats[2:])) if len(beats) > 2 and mean else 0.0
    return cv, smooth


def _nearest_distances(points: Sequence[float], grid: Sequence[float], interval: float) -> list[float]:
    if not points or not grid or interval <= 0:
        return []
    return [min(abs(point - beat) for beat in grid) / interval for point in points]


def _periodicity(onset: np.ndarray, interval_sec: float) -> float:
    lag = round(interval_sec * SR / HOP)
    if lag <= 0 or lag >= len(onset):
        return 0.0
    left, right = onset[:-lag], onset[lag:]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _score_beats(beats: Sequence[float], features: Mapping[str, Any], note_onsets: Sequence[float]) -> dict[str, Any]:
    interval = _median_interval(beats)
    sampled = _sample_envelope(features["onset"], beats)
    audio_hit = statistics.fmean(sampled) if sampled else 0.0
    note_distances = _nearest_distances(note_onsets, beats, interval)
    note_alignment = 1.0 - statistics.fmean(min(value, 1.0) for value in note_distances) if note_distances else 0.0
    note_coverage = sum(value <= 0.18 for value in note_distances) / len(note_distances) if note_distances else 0.0
    peak_distances = _nearest_distances(beats, features["peak_times"], interval)
    audio_peak_coverage = sum(value <= 0.18 for value in peak_distances) / len(peak_distances) if peak_distances else 0.0
    cv, smooth = _regularity(beats)
    local_smoothness = max(0.0, 1.0 - min(1.0, 0.5 * cv + 0.5 * smooth))
    components = {
        "audio_onset_hit": audio_hit,
        "tempo_periodicity": _periodicity(features["onset"], interval),
        "note_onset_alignment": note_alignment,
        "note_onset_coverage": note_coverage,
        "audio_peak_coverage": audio_peak_coverage,
        "local_smoothness": local_smoothness,
    }
    score = sum(POLICY["beat_candidate_weights"][key] * value for key, value in components.items())
    return {"score_higher_is_better": score, "components": components, "median_interval_sec": interval, "bpm": 60.0 / interval if interval else None}


def _clip_grid(beats: Sequence[float], duration: float) -> list[float]:
    return sorted({round(float(value), 9) for value in beats if -0.07 <= value <= duration + 0.07})


def _candidate_grids(current: Sequence[float], features: Mapping[str, Any]) -> list[dict[str, Any]]:
    duration = float(features["duration_sec"])
    candidates: list[dict[str, Any]] = [{"source": "beatnet_current", "scale": 1.0, "offset_sec": 0.0, "beats": _clip_grid(current, duration)}]
    current_bpm = 60.0 / _median_interval(current)
    scales = {1.0}
    for peak in features["tempo_peaks"]:
        base = current_bpm / peak["bpm"]
        for perturbation in POLICY["affine_scale_perturbations"]:
            scales.add(round(base * (1.0 + perturbation), 8))
    for scale in sorted(value for value in scales if 0.5 <= value <= 2.0):
        period = _median_interval(current) * scale
        for offset in np.linspace(-period / 2.0, period / 2.0, POLICY["phase_offsets_per_period"]):
            beats = _clip_grid([scale * value + float(offset) for value in current], duration)
            if len(beats) >= 3:
                candidates.append({"source": "beatnet_affine_audio_proposal", "scale": scale, "offset_sec": float(offset), "beats": beats})
    tempo_values = [peak["bpm"] for peak in features["tempo_peaks"][:3]] or [current_bpm]
    for bpm in tempo_values:
        _tempo, frames = librosa.beat.beat_track(onset_envelope=features["onset"], sr=SR, hop_length=HOP, bpm=float(bpm), units="frames")
        beats = _clip_grid(librosa.frames_to_time(frames, sr=SR, hop_length=HOP).tolist(), duration)
        if len(beats) >= 3:
            candidates.append({"source": "librosa_beat", "seed_bpm": float(bpm), "beats": beats})
    win_length = max(32, min(384, len(features["onset"])))
    pulse = librosa.beat.plp(onset_envelope=features["onset"], sr=SR, hop_length=HOP, tempo_min=40, tempo_max=240, win_length=win_length)
    maxima = np.flatnonzero(librosa.util.localmax(pulse) & (pulse >= np.quantile(pulse, 0.6)))
    plp_beats = _clip_grid(librosa.frames_to_time(maxima, sr=SR, hop_length=HOP).tolist(), duration)
    if len(plp_beats) >= 3:
        candidates.append({"source": "librosa_plp", "beats": plp_beats})
    unique: dict[tuple[float, ...], dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(tuple(candidate["beats"]), candidate)
    return list(unique.values())


def _contrast(values: Sequence[float], indices: set[int]) -> float:
    selected = [value for index, value in enumerate(values) if index in indices]
    others = [value for index, value in enumerate(values) if index not in indices]
    if not selected:
        return 0.0
    return max(-1.0, min(1.0, statistics.fmean(selected) - (statistics.fmean(others) if others else 0.0)))


def _select_downbeats(beats: Sequence[float], features: Mapping[str, Any], bass_onsets: Sequence[float]) -> dict[str, Any]:
    onset_values = _sample_envelope(features["onset"], beats)
    low_values = _sample_envelope(features["low_onset"], beats)
    interval = _median_interval(beats)
    options = []
    for period in POLICY["downbeat_periods"]:
        for phase in range(period):
            indices = set(range(phase, len(beats), period))
            bass_distances = _nearest_distances(bass_onsets, [beats[index] for index in sorted(indices)], interval)
            bass_alignment = 1.0 - statistics.fmean(min(value, 1.0) for value in bass_distances) if bass_distances else 0.0
            bar_accents = [onset_values[index] + 0.5 * low_values[index] for index in sorted(indices)]
            stability = max(0.0, 1.0 - statistics.pstdev(bar_accents)) if len(bar_accents) > 1 else 0.5
            components = {
                "audio_accent_contrast": _contrast(onset_values, indices),
                "low_frequency_contrast": _contrast(low_values, indices),
                "bass_note_alignment": bass_alignment,
                "bar_accent_stability": stability,
            }
            score = sum(POLICY["downbeat_weights"][key] * value for key, value in components.items()) + POLICY["meter_theory_prior"][str(period)]
            options.append({"period_beats": period, "phase_index": phase, "score_higher_is_better": score, "components": components})
    selected = max(options, key=lambda item: (item["score_higher_is_better"], item["period_beats"] == 4, -item["phase_index"]))
    return {**selected, "downbeats": [beats[index] for index in range(selected["phase_index"], len(beats), selected["period_beats"])], "options": options}


def _seconds_to_beat(value: float, beats: Sequence[float]) -> float:
    if len(beats) < 2:
        return value / 0.5
    if value <= beats[0]:
        return (value - beats[0]) / (beats[1] - beats[0])
    if value >= beats[-1]:
        return len(beats) - 1 + (value - beats[-1]) / (beats[-1] - beats[-2])
    index = int(np.searchsorted(beats, value, side="right")) - 1
    return index + (value - beats[index]) / (beats[index + 1] - beats[index])


def _mapping_rhythm(events: Sequence[Mapping[str, Any]], beats: Sequence[float], downbeat: Mapping[str, Any], reference_midi: Path) -> dict[str, Any]:
    period, phase = int(downbeat["period_beats"]), int(downbeat["phase_index"])
    mapped = [(_seconds_to_beat(float(item["start_sec"]), beats), _seconds_to_beat(float(item["end_sec"]), beats), int(item["midi"])) for item in events]
    earliest = min((start for start, _end, _pitch in mapped), default=0.0)
    origin = -float(phase)
    if earliest + origin < -1 / 48:
        origin += period * math.ceil(-(earliest + origin) / period)
    predicted = []
    for start, end, pitch in mapped:
        start_tick = round((start + origin) * 48)
        end_tick = max(start_tick + 1, round((end + origin) * 48))
        predicted.append((pitch, Fraction(start_tick, 48), Fraction(end_tick, 48)))
    _ppq, reference = _midi_notes(reference_midi, exclude_drum_channel=True)
    return rhythm_error(reference, predicted, tolerance_quarters=Fraction(1, 16))


def _current_downbeat_config(grid: Mapping[str, Any]) -> dict[str, Any]:
    meter = str(grid.get("time_signature", {}).get("selected") or "4/4")
    period = {"2/4": 2, "3/4": 3, "4/4": 4, "6/8": 3}.get(meter, 4)
    records = list(grid.get("beats", []))
    phase = next((index for index, item in enumerate(records) if item.get("downbeat")), 0)
    return {"period_beats": period, "phase_index": phase % period}


def _service_events(case_root: Path) -> list[dict[str, Any]]:
    paths = list((case_root / "service_output").glob("*.note-events.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one service note-event artifact in {case_root}, found {len(paths)}")
    return list(_load(paths[0]).get("events", []))


def run(batch_root: Path, output_root: Path) -> dict[str, Any]:
    selection = _load(batch_root / "raw-selection.json")
    evaluator = _load(batch_root / "evaluator-report-v3.json")
    evaluated = {item["id"]: item for item in evaluator["cases"]}
    oracle_path = DEFAULT_ORACLE_REPORT
    oracle_cases = {item["case_id"]: item for item in _load(oracle_path)["cases"]} if oracle_path.is_file() else {}
    cases = []
    for selected in selection["selected"]:
        case_id = selected["case_id"]
        registry_case = evaluated[case_id]
        audio_path = Path(registry_case["input"]["path"])
        raw_root = batch_root / case_id / "raw"
        grid = _load(raw_root / "beat_grid.json")
        recognition = _load(raw_root / "recognition.json")
        current_beats = _times(grid.get("beats", []))
        groups = _note_groups(recognition.get("notes", []))
        features = _audio_features(audio_path)
        candidates = _candidate_grids(current_beats, features)
        for candidate in candidates:
            candidate["reference_free_score"] = _score_beats(candidate["beats"], features, groups["all"])
        refined = max(candidates, key=lambda item: (item["reference_free_score"]["score_higher_is_better"], item["source"] == "beatnet_current"))
        downbeat = _select_downbeats(refined["beats"], features, groups["bass"])

        # Reference access begins here. No value below participates in selection.
        annotation_path = Path(registry_case["beat_annotation"]["path"])
        annotation = _load(annotation_path)["beat_grid"]
        reference_beats = _times(annotation["beats"])
        reference_downbeats = _times(annotation["beats"], downbeats=True)
        reference_midi = Path(registry_case["reference_midi"]["path"])
        rhythm = _mapping_rhythm(_service_events(batch_root / "new" / case_id), refined["beats"], downbeat, reference_midi)
        current_mapping_rhythm = _mapping_rhythm(_service_events(batch_root / "new" / case_id), current_beats, _current_downbeat_config(grid), reference_midi)
        current_rhythm = registry_case["metrics"]["rhythm_error"]
        oracle = oracle_cases.get(case_id)
        summaries = sorted(candidates, key=lambda item: item["reference_free_score"]["score_higher_is_better"], reverse=True)[:12]
        cases.append({
            "case_id": case_id,
            "category": registry_case["category"],
            "input": {"path": str(audio_path), "sha256": _sha256(audio_path)},
            "observable_inputs": {"beatnet_activation_available": False, "beatnet_activation_reason": "immutable v3 raw stores decoded grid but no activation tensor", "all_onsets": len(groups["all"]), "bass_onsets": len(groups["bass"]), "drum_onsets": len(groups["drum"])},
            "audio_observations": {"duration_sec": features["duration_sec"], "onset_peak_count": len(features["peak_times"]), "tempo_peaks": features["tempo_peaks"]},
            "candidate_count": len(candidates),
            "selected": {key: value for key, value in refined.items() if key != "beats"} | {"beat_count": len(refined["beats"]), "beat_times": refined["beats"]},
            "downbeat_selected": downbeat,
            "top_candidates": [{key: value for key, value in item.items() if key != "beats"} | {"beat_count": len(item["beats"])} for item in summaries],
            "post_hoc_reference_evaluation": {
                "current_beat": _f1(reference_beats, current_beats),
                "refined_beat": _f1(reference_beats, refined["beats"]),
                "current_downbeat": registry_case["metrics"]["downbeat_f1"],
                "refined_downbeat": _f1(reference_downbeats, downbeat["downbeats"]),
                "current_new_fixed_total_rhythm": current_rhythm,
                "mapping_only_current_fixed_total_rhythm": current_mapping_rhythm,
                "mapping_only_refined_fixed_total_rhythm": rhythm,
                "reference_oracle_gap": {
                    "affine_beat_f1": oracle["oracles_reference_only"]["single_affine_tempo_offset"]["metrics"]["f1"] if oracle else None,
                    "refined_to_affine_beat_f1": oracle["oracles_reference_only"]["single_affine_tempo_offset"]["metrics"]["f1"] - _f1(reference_beats, refined["beats"])["f1"] if oracle else None,
                    "affine_aligned_meter_downbeat_f1": oracle["oracles_reference_only"]["affine_aligned_meter_downbeat"]["metrics"]["f1"] if oracle else None,
                    "refined_to_affine_meter_downbeat_f1": oracle["oracles_reference_only"]["affine_aligned_meter_downbeat"]["metrics"]["f1"] - _f1(reference_downbeats, downbeat["downbeats"])["f1"] if oracle else None,
                },
            },
        })

    baseline_mean = float(evaluator["accuracy_gate"]["baseline_mean_rhythm_error_quarter"])
    summary = {
        "case_count": len(cases),
        "current_mean_beat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["current_beat"]["f1"] for item in cases),
        "refined_mean_beat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["refined_beat"]["f1"] for item in cases),
        "current_mean_downbeat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["current_downbeat"]["f1"] for item in cases),
        "refined_mean_downbeat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["refined_downbeat"]["f1"] for item in cases),
        "current_new_mean_fixed_total_rhythm": float(evaluator["accuracy_gate"]["new_mean_rhythm_error_quarter"]),
        "mapping_only_current_mean_fixed_total_rhythm": statistics.fmean(item["post_hoc_reference_evaluation"]["mapping_only_current_fixed_total_rhythm"]["mean_fixed_total_assignment_rhythm_error_quarter"] for item in cases),
        "mapping_only_refined_mean_fixed_total_rhythm": statistics.fmean(item["post_hoc_reference_evaluation"]["mapping_only_refined_fixed_total_rhythm"]["mean_fixed_total_assignment_rhythm_error_quarter"] for item in cases),
        "baseline_mean_fixed_total_rhythm": baseline_mean,
    }
    summary["mapping_only_rhythm_improvement_vs_baseline_percent"] = 100.0 * (baseline_mean - summary["mapping_only_refined_mean_fixed_total_rhythm"]) / baseline_mean
    summary["mapping_only_refinement_change_vs_mapping_current_percent"] = 100.0 * (summary["mapping_only_current_mean_fixed_total_rhythm"] - summary["mapping_only_refined_mean_fixed_total_rhythm"]) / summary["mapping_only_current_mean_fixed_total_rhythm"]
    if oracle_cases:
        summary["reference_oracle_comparison"] = {
            "mean_affine_beat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["reference_oracle_gap"]["affine_beat_f1"] for item in cases),
            "observable_refined_gap_to_affine_beat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["reference_oracle_gap"]["refined_to_affine_beat_f1"] for item in cases),
            "mean_affine_aligned_meter_downbeat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["reference_oracle_gap"]["affine_aligned_meter_downbeat_f1"] for item in cases),
            "observable_refined_gap_to_affine_meter_downbeat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["reference_oracle_gap"]["refined_to_affine_meter_downbeat_f1"] for item in cases),
            "source": str(oracle_path),
            "source_sha256": _sha256(oracle_path),
        }
    summary["beat_target_0_85_met"] = summary["refined_mean_beat_f1"] >= 0.85
    summary["downbeat_target_0_75_met"] = summary["refined_mean_downbeat_f1"] >= 0.75
    summary["rhythm_20_percent_target_met"] = summary["mapping_only_rhythm_improvement_vs_baseline_percent"] >= 20.0
    source_counts: dict[str, int] = {}
    for item in cases:
        source = item["selected"]["source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    summary["selected_source_counts"] = source_counts
    category_metrics = {}
    for category in sorted({item["category"] for item in cases}):
        subset = [item for item in cases if item["category"] == category]
        category_metrics[category] = {
            "count": len(subset),
            "current_beat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["current_beat"]["f1"] for item in subset),
            "refined_beat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["refined_beat"]["f1"] for item in subset),
            "current_downbeat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["current_downbeat"]["f1"] for item in subset),
            "refined_downbeat_f1": statistics.fmean(item["post_hoc_reference_evaluation"]["refined_downbeat"]["f1"] for item in subset),
        }
    report = {"schema_version": "beatnet_observable_refinement_pilot_1", "diagnostic_only": True, "runtime_consumed": False, "policy": POLICY, "summary": summary, "category_metrics": category_metrics, "cases": cases}
    output_root.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_root / "pilot-report.json", output_root / "pilot-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    manifest = {"schema_version": "diagnostic_artifact_manifest_1", "artifacts": [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in (json_path, md_path)]}
    (output_root / "artifact-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# BeatNet observable refinement pilot v1", "",
        "This is an offline pilot. Candidate selection is reference-free; reference annotations are loaded only after selection for evaluation.", "",
        "## Fixed policy", "", "```json", json.dumps(report["policy"], ensure_ascii=False, indent=2), "```", "",
        "## Aggregate", "", "| metric | before | refined | target |", "|---|---:|---:|---:|",
        f"| beat F1 | {summary['current_mean_beat_f1']:.6f} | {summary['refined_mean_beat_f1']:.6f} | 0.85 |",
        f"| downbeat F1 | {summary['current_mean_downbeat_f1']:.6f} | {summary['refined_mean_downbeat_f1']:.6f} | 0.75 |",
        f"| fixed-total rhythm (production current / mapping pilot) | {summary['current_new_mean_fixed_total_rhythm']:.6f} | {summary['mapping_only_refined_mean_fixed_total_rhythm']:.6f} | <= {summary['baseline_mean_fixed_total_rhythm'] * 0.8:.6f} |", "",
        f"Mapping-only refined rhythm change versus baseline: {summary['mapping_only_rhythm_improvement_vs_baseline_percent']:.3f}%.",
        f"Mapping-only current/refined values are {summary['mapping_only_current_mean_fixed_total_rhythm']:.6f}/{summary['mapping_only_refined_mean_fixed_total_rhythm']:.6f}, a {summary['mapping_only_refinement_change_vs_mapping_current_percent']:.3f}% change under the same simulator.",
        f"Observable refined beat/downbeat gaps to affine reference-only oracle are {summary.get('reference_oracle_comparison', {}).get('observable_refined_gap_to_affine_beat_f1', float('nan')):.6f}/{summary.get('reference_oracle_comparison', {}).get('observable_refined_gap_to_affine_meter_downbeat_f1', float('nan')):.6f}.",
        f"Selected sources: {json.dumps(summary['selected_source_counts'], ensure_ascii=False, sort_keys=True)}.", "",
        "## By category", "", "| category | n | beat before | beat refined | downbeat before | downbeat refined |", "|---|---:|---:|---:|---:|---:|",
    ]
    for category, values in report["category_metrics"].items():
        lines.append(f"| {category} | {values['count']} | {values['current_beat_f1']:.3f} | {values['refined_beat_f1']:.3f} | {values['current_downbeat_f1']:.3f} | {values['refined_downbeat_f1']:.3f} |")
    lines += ["", "## Per case", "", "| case | source | candidates | beat before | beat refined | downbeat before | downbeat refined | mapping rhythm |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in report["cases"]:
        metrics = item["post_hoc_reference_evaluation"]
        lines.append(f"| {item['case_id']} | {item['selected']['source']} | {item['candidate_count']} | {metrics['current_beat']['f1']:.3f} | {metrics['refined_beat']['f1']:.3f} | {metrics['current_downbeat']['f1']:.3f} | {metrics['refined_downbeat']['f1']:.3f} | {metrics['mapping_only_refined_fixed_total_rhythm']['mean_fixed_total_assignment_rhythm_error_quarter']:.3f} |")
    lines += ["", "## Decision", "", f"Beat target met: **{str(summary['beat_target_0_85_met']).lower()}**.", f"Downbeat target met: **{str(summary['downbeat_target_0_75_met']).lower()}**.", f"Mapping-only rhythm target met: **{str(summary['rhythm_20_percent_target_met']).lower()}**.", "", "The rhythm figure is a mapping-only estimate using the existing cleaned service note events and 48-TPQ rounding. It does not claim MuseScore import/render equivalence and does not alter the production gate.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.batch_root.resolve(), args.output_root.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
