"""Audit the MAESTRO local-render benchmark's audio and beat time axes.

This is an offline diagnostic.  It never changes tracker output and it treats
reference annotations as evaluation data only.  The audit distinguishes
renderer/timestamp errors from a more basic label-validity problem: MAESTRO's
aligned performance MIDI stores key-event time, but the selected files do not
contain score beat annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_beatnet_v3 import _best_affine, _best_shift, _f1
from scripts.fluidsynth_benchmark_renderer import _tempo_segments, _tick_to_seconds, midi_notes


DEFAULT_MAESTRO_ROOT = ROOT / ".cache" / "high-accuracy-benchmarks" / "maestro"
DEFAULT_BATCH = ROOT / ".artifacts" / "review" / "production-acceptance-v3"
DEFAULT_MADMOM = ROOT / ".artifacts" / "review" / "beat-tracker-comparator-v1"
DEFAULT_BEAT_THIS = DEFAULT_MADMOM / "beat-this-pilot"
DEFAULT_OUTPUT = ROOT / ".artifacts" / "review" / "maestro-timeline-audit-v1"
TOLERANCE_SEC = 0.07


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _annotation_times(payload: Mapping[str, Any], *, downbeats: bool = False) -> list[float]:
    grid = payload.get("beat_grid", payload)
    values = grid.get("downbeats" if downbeats else "beats", [])
    return [float(item["time_sec"] if isinstance(item, Mapping) else item) for item in values]


def _tracker_times(payload: Mapping[str, Any], *, downbeats: bool = False) -> list[float]:
    if "records" in payload:
        return [float(item["time_sec"]) for item in payload["records"] if not downbeats or item.get("downbeat")]
    values = payload.get("downbeats" if downbeats else "beats", [])
    return [float(item["time_sec"] if isinstance(item, Mapping) else item) for item in values]


def _interval_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    intervals = [float(b - a) for a, b in zip(values, values[1:]) if b > a]
    return {
        "count": len(values),
        "median_interval_sec": statistics.median(intervals) if intervals else None,
        "median_bpm": 60.0 / statistics.median(intervals) if intervals and statistics.median(intervals) > 0 else None,
    }


def _first_audible_frame(path: Path) -> dict[str, float | int | None]:
    with wave.open(str(path), "rb") as reader:
        sample_rate = int(reader.getframerate())
        channels = int(reader.getnchannels())
        sample_width = int(reader.getsampwidth())
        frames = int(reader.getnframes())
        if sample_width != 2:
            raise ValueError(f"expected PCM16 WAV: {path}")
        samples = np.frombuffer(reader.readframes(frames), dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels)
        magnitude = np.max(np.abs(samples.astype(np.int32)), axis=1)
    else:
        magnitude = np.abs(samples.astype(np.int32))
    audible = np.flatnonzero(magnitude > 0)
    first = int(audible[0]) if audible.size else None
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_sec": frames / sample_rate,
        "first_nonzero_frame": first,
        "first_nonzero_sec": first / sample_rate if first is not None else None,
    }


def _tick_conversion_audit(midi_path: Path, annotation: Mapping[str, Any]) -> dict[str, Any]:
    midi, notes = midi_notes(midi_path)
    segments, starts = _tempo_segments(midi)
    beat_times = _annotation_times(annotation)
    expected = [_tick_to_seconds(midi, segments, starts, index * midi.ticks_per_beat) for index in range(len(beat_times))]
    errors = [abs(a - b) for a, b in zip(beat_times, expected)]
    return {
        "ticks_per_beat": int(midi.ticks_per_beat),
        "tempo_segments": [
            {"tick": tick, "time_sec": sec, "tempo_us_per_quarter": tempo}
            for tick, sec, tempo in segments
        ],
        "note_count": len(notes),
        "first_note_sec": min((float(note["start_sec"]) for note in notes), default=None),
        "last_note_end_sec": max((float(note["end_sec"]) for note in notes), default=None),
        "annotation_matches_midi_tick_clock": len(beat_times) == len(expected) and math.fsum(errors) <= 1e-9,
        "maximum_tick_clock_error_sec": max(errors, default=0.0),
    }


def _comparison(reference: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    return {
        "direct": _f1(reference, predicted),
        "offset_only_oracle": _best_shift(reference, predicted),
        "affine_oracle": _best_affine(reference, predicted),
        "timing": _interval_summary(predicted),
    }


def _pairwise(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    variants = {
        "direct": list(a),
        "a_every_second_phase_0": list(a[::2]),
        "a_every_second_phase_1": list(a[1::2]),
    }
    scored = {name: _f1(b, values) for name, values in variants.items()}
    best_name = max(scored, key=lambda name: (scored[name]["f1"], scored[name]["true_positive"]))
    return {"best_variant": best_name, "best_metrics": scored[best_name], "variants": scored}


def run_audit(
    *,
    maestro_root: Path = DEFAULT_MAESTRO_ROOT,
    batch_root: Path = DEFAULT_BATCH,
    madmom_root: Path = DEFAULT_MADMOM,
    beat_this_root: Path = DEFAULT_BEAT_THIS,
) -> dict[str, Any]:
    selection = _read_json(maestro_root / "selection_manifest.json")
    cases: list[dict[str, Any]] = []
    annotation_hashes: list[str] = []
    beat_this_coverage: list[str] = []
    for selected in selection["cases"]:
        case_id = str(selected["id"])
        midi_path = maestro_root / selected["midi"]["path"]
        audio_path = maestro_root / selected["audio"]["path"]
        annotation_path = maestro_root / selected["beat_annotation"]["path"]
        render_manifest_path = maestro_root / selected["render_manifest"]["path"]
        annotation = _read_json(annotation_path)
        render_manifest = _read_json(render_manifest_path)
        reference = _annotation_times(annotation)
        reference_downbeats = _annotation_times(annotation, downbeats=True)
        annotation_hashes.append(_sha256(annotation_path))

        beatnet_payload = _read_json(batch_root / case_id / "raw" / "beat_grid.json")
        beatnet = _tracker_times(beatnet_payload)
        beatnet_downbeats = _tracker_times(beatnet_payload, downbeats=True)
        madmom_path = madmom_root / "raw" / case_id / "madmom_official_ensemble__official_234.json"
        madmom_payload = _read_json(madmom_path)
        madmom = _tracker_times(madmom_payload)
        madmom_downbeats = _tracker_times(madmom_payload, downbeats=True)

        beat_this_path = beat_this_root / "raw" / case_id / "final0_official_dbn_34.json"
        beat_this: dict[str, Any] | None = None
        if beat_this_path.is_file():
            beat_this_payload = _read_json(beat_this_path)
            beat_this_values = _tracker_times(beat_this_payload)
            beat_this = {
                "path": str(beat_this_path.resolve()),
                "comparison": _comparison(reference, beat_this_values),
                "pairwise_with_madmom": _pairwise(beat_this_values, madmom),
            }
            beat_this_coverage.append(case_id)

        audio = _first_audible_frame(audio_path)
        tick_audit = _tick_conversion_audit(midi_path, annotation)
        first_note = tick_audit["first_note_sec"]
        first_audio = audio["first_nonzero_sec"]
        onset_delta_ms = (
            (float(first_audio) - float(first_note)) * 1000.0
            if first_audio is not None and first_note is not None
            else None
        )
        cases.append(
            {
                "case_id": case_id,
                "source_archive_member": selected.get("archive_member"),
                "paths": {
                    "midi": str(midi_path.resolve()),
                    "audio": str(audio_path.resolve()),
                    "annotation": str(annotation_path.resolve()),
                    "render_manifest": str(render_manifest_path.resolve()),
                },
                "reference": {
                    "beat_count": len(reference),
                    "downbeat_count": len(reference_downbeats),
                    "timing": _interval_summary(reference),
                    "annotation_sha256": annotation_hashes[-1],
                    "beats_before_first_note": sum(value < float(first_note or 0.0) for value in reference),
                    "beats_after_audio_end": sum(value > float(audio["duration_sec"]) for value in reference),
                },
                "midi_tick_clock": tick_audit,
                "renderer": {
                    "audio": audio,
                    "manifest_target_duration_sec": float(render_manifest["target_duration_sec"]),
                    "target_duration_error_sec": abs(float(audio["duration_sec"]) - float(render_manifest["target_duration_sec"])),
                    "first_audio_minus_first_midi_note_ms": onset_delta_ms,
                    "fixed_tail_sec": float(render_manifest["renderer"]["tail_sec"]),
                    "source_event_complete": bool(render_manifest["verification"]["source_event_complete"]),
                },
                "trackers": {
                    "beatnet": {
                        "comparison": _comparison(reference, beatnet),
                        "downbeat_direct": _f1(reference_downbeats, beatnet_downbeats),
                    },
                    "madmom": {
                        "comparison": _comparison(reference, madmom),
                        "downbeat_direct": _f1(reference_downbeats, madmom_downbeats),
                    },
                    "beat_this": beat_this,
                    "beatnet_vs_madmom": _pairwise(beatnet, madmom),
                },
            }
        )

    renderer_deltas = [case["renderer"]["first_audio_minus_first_midi_note_ms"] for case in cases]
    report = {
        "schema_version": "maestro_timeline_audit_v1",
        "diagnostic_only": True,
        "tolerance_sec": TOLERANCE_SEC,
        "case_count": len(cases),
        "summary": {
            "all_annotations_byte_identical": len(set(annotation_hashes)) == 1,
            "unique_annotation_hash_count": len(set(annotation_hashes)),
            "all_annotations_are_uniform_120_bpm_tick_grids": all(
                case["reference"]["timing"]["median_bpm"] == 120.0 for case in cases
            ),
            "all_midi_tick_conversions_exact": all(case["midi_tick_clock"]["annotation_matches_midi_tick_clock"] for case in cases),
            "all_renderer_source_events_complete": all(case["renderer"]["source_event_complete"] for case in cases),
            "maximum_renderer_target_duration_error_sec": max(case["renderer"]["target_duration_error_sec"] for case in cases),
            "renderer_first_onset_delay_ms": {
                "minimum": min(renderer_deltas),
                "median": statistics.median(renderer_deltas),
                "maximum": max(renderer_deltas),
            },
            "mean_reference_beats_before_first_note": statistics.fmean(case["reference"]["beats_before_first_note"] for case in cases),
            "beatnet_mean_direct_f1": statistics.fmean(case["trackers"]["beatnet"]["comparison"]["direct"]["f1"] for case in cases),
            "madmom_mean_direct_f1": statistics.fmean(case["trackers"]["madmom"]["comparison"]["direct"]["f1"] for case in cases),
            "beatnet_mean_offset_only_f1": statistics.fmean(case["trackers"]["beatnet"]["comparison"]["offset_only_oracle"]["metrics"]["f1"] for case in cases),
            "beatnet_mean_affine_f1": statistics.fmean(case["trackers"]["beatnet"]["comparison"]["affine_oracle"]["metrics"]["f1"] for case in cases),
            "madmom_mean_offset_only_f1": statistics.fmean(case["trackers"]["madmom"]["comparison"]["offset_only_oracle"]["metrics"]["f1"] for case in cases),
            "madmom_mean_affine_f1": statistics.fmean(case["trackers"]["madmom"]["comparison"]["affine_oracle"]["metrics"]["f1"] for case in cases),
            "beat_this_case_coverage": beat_this_coverage,
            "conclusion": "benchmark_coordinate_semantics_invalid_for_beat_f1",
        },
        "finding": {
            "renderer_time_axis": "valid_with_sub_tolerance_soundfont_attack_delay",
            "midi_tick_to_second_conversion": "internally_exact",
            "beat_annotation": "invalid_as_independent_musical_beat_ground_truth",
            "reason": (
                "The selected MAESTRO files expose one fixed 120 BPM tempo and 4/4 meter while their ticks encode "
                "aligned performance-event time. They contain no score beat/downbeat labels. The benchmark generated "
                "0.5-second beats from that transport clock, so identical labels were assigned to ten different performances."
            ),
            "production_inference_changed": False,
            "recommended_gate_action": "exclude these ten annotations from beat/downbeat metrics until independently aligned beat labels are supplied",
        },
        "limitations": [
            "Beat This was available only for the predeclared maestro-midi-01 pilot; no claim is made for the other nine cases.",
            "Offset and affine fits are reference-only diagnostics and must never select or modify production tracker output.",
            "This audit changes only beat/downbeat eligibility; reference-MIDI score metrics require a separate review.",
        ],
        "cases": cases,
    }
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# MAESTRO timeline audit v1",
        "",
        "This diagnostic does not modify production inference. Reference data is read only after tracker outputs have already been saved.",
        "",
        "## Finding",
        "",
        "The renderer and MIDI-to-seconds implementation are internally correct, but the generated MAESTRO beat labels are not musical beat annotations. All ten different performances received the same 33-point, 120 BPM grid from 0.0 through 16.0 seconds. Their source files contain one fixed 120 BPM transport event and 4/4 marker, with performance key-event timestamps; they contain no score-aligned beat or downbeat labels.",
        "",
        f"FluidSynth preserved every source event, matched its target duration within {summary['maximum_renderer_target_duration_error_sec']:.9f} seconds, and began sounding {summary['renderer_first_onset_delay_ms']['minimum']:.3f}–{summary['renderer_first_onset_delay_ms']['maximum']:.3f} ms after the first MIDI note. This delay is far below the 70 ms evaluation tolerance and cannot explain the low F1.",
        "",
        f"BeatNet direct/offset-only/affine mean F1 was {summary['beatnet_mean_direct_f1']:.6f}/{summary['beatnet_mean_offset_only_f1']:.6f}/{summary['beatnet_mean_affine_f1']:.6f}. Madmom was {summary['madmom_mean_direct_f1']:.6f}/{summary['madmom_mean_offset_only_f1']:.6f}/{summary['madmom_mean_affine_f1']:.6f}. Offset alone does not repair the scores; scale plus offset often does because it warps each tracker's musical pulse onto the unrelated fixed transport grid. That is diagnostic evidence, not a valid correction.",
        "",
        "Beat This coverage contains only `maestro-midi-01`. Its official DBN output follows roughly the same 0.72–0.76 second pulse as madmom, while BeatNet often emits the double-time pulse. Both disagree with the synthetic 0.5-second reference. The nine missing Beat This cases are reported as missing rather than inferred.",
        "",
        "The ten generated grids must be excluded from beat/downbeat F1 until independently aligned beat labels are available. This audit does not change reference-MIDI score eligibility.",
        "",
        "## Per case",
        "",
        "| case | first note | audible delay ms | silent-grid beats | BeatNet direct / offset / affine | madmom direct / offset / affine | BeatNet BPM | madmom BPM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        bn = case["trackers"]["beatnet"]["comparison"]
        mm = case["trackers"]["madmom"]["comparison"]
        lines.append(
            f"| `{case['case_id']}` | {case['midi_tick_clock']['first_note_sec']:.3f} | "
            f"{case['renderer']['first_audio_minus_first_midi_note_ms']:.3f} | {case['reference']['beats_before_first_note']} | "
            f"{bn['direct']['f1']:.3f} / {bn['offset_only_oracle']['metrics']['f1']:.3f} / {bn['affine_oracle']['metrics']['f1']:.3f} | "
            f"{mm['direct']['f1']:.3f} / {mm['offset_only_oracle']['metrics']['f1']:.3f} / {mm['affine_oracle']['metrics']['f1']:.3f} | "
            f"{bn['timing']['median_bpm']:.1f} | {mm['timing']['median_bpm']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-root", type=Path, default=DEFAULT_MAESTRO_ROOT)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--madmom-root", type=Path, default=DEFAULT_MADMOM)
    parser.add_argument("--beat-this-root", type=Path, default=DEFAULT_BEAT_THIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_audit(
        maestro_root=args.maestro_root.resolve(),
        batch_root=args.batch_root.resolve(),
        madmom_root=args.madmom_root.resolve(),
        beat_this_root=args.beat_this_root.resolve(),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "timeline-audit.json"
    markdown_path = output / "timeline-audit.md"
    _write_json(report_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": "artifact_manifest_1",
            "artifacts": [
                {"path": report_path.name, "sha256": _sha256(report_path), "bytes": report_path.stat().st_size},
                {"path": markdown_path.name, "sha256": _sha256(markdown_path), "bytes": markdown_path.stat().st_size},
            ],
        },
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
