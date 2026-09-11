from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mido
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("high_accuracy_benchmark", ROOT / "scripts" / "high_accuracy_benchmark.py")
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _write_midi(path: Path, *, pitch: int = 60, ppq: int = 480, pitches: tuple[int, ...] | None = None) -> None:
    midi = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="fixture"))
    for index, note in enumerate(pitches or (pitch,)):
        track.append(mido.Message("note_on", note=note, velocity=80, time=0 if index else 0))
    for index, note in enumerate(pitches or (pitch,)):
        track.append(mido.Message("note_off", note=note, velocity=0, time=ppq if index == 0 else 0))
    midi.tracks.append(track)
    midi.save(path)


def _rhythm_metric(value: float) -> dict[str, object]:
    return {
        "mean_rhythm_error_quarter": value,
        "mean_fixed_total_assignment_rhythm_error_quarter": value,
        "metric_schema": benchmark.RHYTHM_METRIC_SCHEMA,
    }


def test_registry_records_pjs_and_marks_luv_letter_manual_only() -> None:
    registry = benchmark._load_registry(ROOT / "fixtures" / "high_accuracy" / "benchmark_manifest.json")
    assert registry["schema_version"] == "2.0"
    assert len(registry["cases"]) == 46
    assert sum(item.get("reference_midi_reliable") is True for item in registry["cases"]) == 40
    assert sum(item.get("production_gate_selected") is True for item in registry["cases"]) == 30
    assert all(isinstance(item.get("production_gate_selected"), bool) for item in registry["cases"])
    assert {item["id"] for item in registry["cases"] if item["category"] == "vocal"} == {
        "ccmusic-yueding-01",
        "ccmusic-yueding-02",
        "ccmusic-yueding-03",
        "ccmusic-yueding-04",
        "ccmusic-yueding-05",
    }
    assert {item["id"] for item in registry["cases"] if item["category"] == "vocal_diagnostic"} == {
        "pjs001",
        "pjs002",
        "pjs003",
        "pjs004",
        "pjs005",
    }
    assert sum(item["category"] == "synthetic_rendered" for item in registry["cases"]) == 10
    assert sum(item["category"] == "official_piano_rendered" for item in registry["cases"]) == 10
    assert sum(item["category"] == "official_piano_aligned" for item in registry["cases"]) == 10
    assert sum(item["category"] == "specialized_fixture" for item in registry["cases"]) == 5
    reliable = [item for item in registry["cases"] if item.get("reference_midi_reliable") is True]
    selected = [item for item in registry["cases"] if item.get("production_gate_selected") is True]
    assert {item["category"] for item in selected} == {
        "synthetic_rendered",
        "official_piano_aligned",
        "vocal",
        "specialized_fixture",
    }
    assert all(item.get("reference_midi_reliable") is True for item in selected)
    assert all(item.get("evaluation_policy") == "reference_metrics" for item in selected)
    assert all(item.get("evaluation_scope") == "production_end_to_end" for item in selected)
    assert all(item.get("benchmark_role") == "production_end_to_end" for item in selected)
    assert {item.get("render_domain") for item in reliable} == {
        "synthetic_local_midi_render",
        "maestro_local_midi_render",
        "asap_performance_midi_render",
        "research_mixed_song",
        "specialized_local_midi_render",
    }
    assert sum(item.get("beat_annotation_independent") is True for item in reliable) == 30
    assert sum(item.get("beat_annotation_independent") is False for item in reliable) == 10
    assert all(
        item.get("beat_annotation_source") in {"deterministic_midi_render_ground_truth", "ccmusic_musicxml_score_ground_truth", "asap_v1.1_direct_performance_annotation"}
        for item in reliable
        if item.get("beat_annotation_independent") is True
    )
    assert all(item.get("reference_midi_reliable") is False for item in registry["cases"] if item["category"] == "vocal_diagnostic")
    maestro = [item for item in registry["cases"] if item["category"] == "official_piano_rendered"]
    assert all(item["beat_annotation_source"] == "maestro_performance_midi_tick_grid_diagnostic_only" for item in maestro)
    assert all(item["reference_midi_reliable"] is True for item in maestro)
    assert all(item["production_gate_selected"] is False for item in maestro)
    assert all(item["evaluation_policy"] == "diagnostic_only" for item in maestro)
    assert all(item["evaluation_scope"] == "diagnostic_only" for item in maestro)
    assert all(item["benchmark_role"] == "diagnostic_only" for item in maestro)
    asap = [item for item in selected if item["category"] == "official_piano_aligned"]
    assert len(asap) == 10
    assert len({item["composer"] for item in asap}) == 10
    assert {item["time_signature"] for item in asap} == {"2/4", "3/4", "4/4", "6/8"}
    assert all(item["beat_annotation_source"] == "asap_v1.1_direct_performance_annotation" for item in asap)
    assert registry["sources"]["asap-v1.1"]["commit"] == "fad8d1e8078d0ae47ad2f280b5d022bd2de24784"
    assert all(item.get("evaluation_policy") == "diagnostic_only" for item in registry["cases"] if item["category"] == "vocal_diagnostic")
    assert registry["sources"]["ccmusic-demo"]["archive_sha256"] == "477b5466936eec40cef7dfd43205900e3e4a651b8ec671fdccaff48910523053"
    luv = next(item for item in registry["cases"] if item["id"] == "luv-letter")
    assert luv["evaluation_policy"] == "integrity_and_manual_listening_only"
    assert luv["reference_midi_reliable"] is False

    report = benchmark.build_report(registry)
    assert report["registered_count"] == 46
    assert report["evaluated_count"] == 0
    assert report["accuracy_claim_ready"] is False
    assert all(item["metrics"]["pitch_f1"] is None for item in report["cases"])
    assert all(item["metrics"]["note_onset_f1"] is None for item in report["cases"])
    assert all(item["metrics"]["pitch_multiset_f1"] is None for item in report["cases"])


def test_reliable_diagnostic_case_is_excluded_from_main_gate_index() -> None:
    cases = [
        {
            "id": "maestro-diagnostic",
            "status": "evaluated",
            "evaluation_policy": "diagnostic_only",
            "reference_midi_reliable": True,
            "production_gate_selected": False,
        },
        {
            "id": "asap-production",
            "status": "evaluated",
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "production_gate_selected": True,
        },
    ]
    indexed, missing, duplicates = benchmark._case_id_index(cases, label="new")
    assert set(indexed) == {"asap-production"}
    assert missing == []
    assert duplicates == []


def test_evaluator_normalizes_different_ppq_and_does_not_invent_missing_beats(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mid"
    _write_midi(reference, ppq=480, pitches=(60, 64, 67))
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fixture")
    result_root = tmp_path / "results" / "case-1"
    result_root.mkdir(parents=True)
    result_midi = result_root / "case-1.score.mid"
    _write_midi(result_midi, ppq=96, pitches=(60, 64, 67))
    (result_root / "manifest.json").write_text(
        json.dumps({"status": "success", "artifacts": [{"kind": "instrument_score_midi", "relative_path": result_midi.name}]}),
        encoding="utf-8",
    )
    case = {
        "id": "case-1",
        "title": "fixture",
        "input": str(input_audio),
        "reference_midi": str(reference),
        "reference_midi_reliable": True,
        "beat_annotation": None,
        "evaluation_policy": "reference_metrics",
    }
    evaluated = benchmark.evaluate_case(case, result_root=tmp_path / "results")
    assert evaluated["status"] == "evaluated"
    assert evaluated["crash"] is False
    assert evaluated["metrics"]["pitch_f1"]["f1"] == 1.0
    assert evaluated["metrics"]["note_onset_f1"] == evaluated["metrics"]["pitch_f1"]
    assert evaluated["metrics"]["note_onset_f1"]["metric_name"] == "note_onset_f1"
    assert evaluated["metrics"]["note_onset_f1"]["definition"] == benchmark.NOTE_ONSET_F1_DEFINITION
    assert evaluated["metrics"]["pitch_multiset_f1"]["f1"] == 1.0
    assert evaluated["metrics"]["pitch_multiset_f1"]["diagnostic_only"] is True
    assert evaluated["metrics"]["pitch_multiset_f1"]["formal_accuracy_metric"] is False
    assert evaluated["metrics"]["chord_retention"]["retention"] == 1.0
    assert evaluated["metrics"]["rhythm_error"]["mean_onset_error_quarter"] == 0.0
    assert evaluated["metrics"]["rhythm_error"]["mean_duration_error_quarter"] == 0.0
    assert evaluated["metrics"]["rhythm_error"]["mean_rhythm_error_quarter"] == 0.0
    assert evaluated["metrics"]["beat_f1"] is None


def test_pitch_multiset_diagnostic_ignores_time_order_and_duration() -> None:
    reference = [
        (60, benchmark.Fraction(0), benchmark.Fraction(1)),
        (60, benchmark.Fraction(1), benchmark.Fraction(2)),
        (64, benchmark.Fraction(2), benchmark.Fraction(3)),
    ]
    predicted = [
        (64, benchmark.Fraction(8), benchmark.Fraction(9)),
        (60, benchmark.Fraction(10), benchmark.Fraction(12)),
        (60, benchmark.Fraction(14), benchmark.Fraction(14) + benchmark.Fraction(1, 4)),
    ]
    note_onset = benchmark.pitch_metrics(
        reference, predicted, tolerance_quarters=benchmark.Fraction(1, 16)
    )
    multiset = benchmark.pitch_multiset_metrics(reference, predicted)
    assert note_onset["f1"] == 0.0
    assert multiset["f1"] == 1.0
    assert "onset" in multiset["definition"]
    assert multiset["diagnostic_only"] is True
    assert multiset["formal_accuracy_metric"] is False


def test_rhythm_assignment_penalizes_unmatched_notes_and_zero_match_cases() -> None:
    reference = [
        (60, benchmark.Fraction(0), benchmark.Fraction(1)),
        (62, benchmark.Fraction(1), benchmark.Fraction(2)),
    ]
    predicted = [
        (60, benchmark.Fraction(0), benchmark.Fraction(1)),
        (64, benchmark.Fraction(3), benchmark.Fraction(4)),
    ]
    metric = benchmark.rhythm_error(reference, predicted, tolerance_quarters=benchmark.Fraction(1, 16))
    assert metric["matched_notes"] == 1
    assert metric["false_negative_unmatched_reference"] == 1
    assert metric["false_positive_unmatched_prediction"] == 1
    assert metric["mean_matched_rhythm_error_quarter"] == 0.0
    assert metric["fixed_total_assignment_cost_quarter"] == 2.0
    assert metric["mean_fixed_total_assignment_rhythm_error_quarter"] == 1.0
    assert metric["mean_rhythm_error_quarter"] == 1.0
    assert metric["metric_schema"] == benchmark.RHYTHM_METRIC_SCHEMA

    zero_match = benchmark.rhythm_error(
        reference,
        [(65, benchmark.Fraction(3), benchmark.Fraction(4))],
        tolerance_quarters=benchmark.Fraction(1, 16),
    )
    assert zero_match["matched_notes"] == 0
    assert zero_match["mean_matched_rhythm_error_quarter"] is None
    assert zero_match["mean_rhythm_error_quarter"] == 1.5


def test_rhythm_gate_prefers_fixed_total_metric_when_report_has_both_fields() -> None:
    case = {
        "metrics": {
            "rhythm_error": {
                "mean_rhythm_error_quarter": 0.01,
                "mean_fixed_total_assignment_rhythm_error_quarter": 1.25,
                "metric_schema": benchmark.RHYTHM_METRIC_SCHEMA,
            }
        }
    }
    assert benchmark._metric_f1(case, "rhythm_error", "mean_rhythm_error_quarter") == 1.25

    legacy = {"metrics": {"rhythm_error": {"mean_rhythm_error_quarter": 0.01}}}
    assert benchmark._metric_f1(legacy, "rhythm_error", "mean_rhythm_error_quarter") is None
    assert benchmark._has_fixed_total_rhythm_metric(legacy) is False


def test_note_onset_gate_accepts_legacy_pitch_f1_and_canonical_name() -> None:
    legacy = {"metrics": {"pitch_f1": {"f1": 0.71}}}
    canonical = {"metrics": {"note_onset_f1": {"f1": 0.72}}}
    assert benchmark._metric_f1(legacy, "note_onset_f1", "f1") == 0.71
    assert benchmark._metric_f1(canonical, "pitch_f1", "f1") == 0.72


def test_accuracy_gate_rejects_legacy_rhythm_report_instead_of_falling_back() -> None:
    def legacy_case(case_id: str) -> dict[str, object]:
        return {
            "id": case_id,
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "metrics": {
                "pitch_f1": {"f1": 0.9},
                "chord_retention": {"retention": 0.9},
                "rhythm_error": {"mean_rhythm_error_quarter": 0.01},
            },
        }

    claim = benchmark.assess_accuracy_claim(
        [legacy_case("case-1"), legacy_case("case-2")],
        [legacy_case("case-1"), legacy_case("case-2")],
        minimum_cases=2,
        require_beat_metrics=False,
    )
    assert claim["ready"] is False
    assert "fixed-total rhythm metric" in claim["reason"]
    assert "需重跑" in claim["reason"]


def test_beat_grid_time_sec_and_downbeat_metrics_are_read_correctly(tmp_path: Path) -> None:
    beat_grid = tmp_path / "beat_grid.json"
    beat_grid.write_text(
        json.dumps({"beat_grid": {"beats": [
            {"time_sec": 0.0, "downbeat": True},
            {"time_sec": 0.5, "downbeat": False},
            {"time_sec": 1.0, "downbeat": True},
        ]}}),
        encoding="utf-8",
    )
    assert benchmark._read_time_points(beat_grid) == [0.0, 0.5, 1.0]
    assert benchmark._read_time_points(beat_grid, downbeats=True) == [0.0, 1.0]
    assert benchmark.beat_f1([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])["f1"] == 1.0
    assert benchmark.beat_f1([0.0, 1.0], [0.0, 1.0])["f1"] == 1.0


def test_zero_true_positives_report_zero_f1_instead_of_missing() -> None:
    metric = benchmark._f1(0, 0, predicted_count=3, reference_count=4)
    assert metric["precision"] == 0.0
    assert metric["recall"] == 0.0
    assert metric["f1"] == 0.0
    assert benchmark.beat_f1([0.0], [1.0])["f1"] == 0.0


def test_numeric_midi_metrics_ignore_general_midi_drum_channel(tmp_path: Path) -> None:
    path = tmp_path / "mixed.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    pitched = mido.MidiTrack()
    pitched.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    pitched.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
    drums = mido.MidiTrack()
    drums.append(mido.Message("note_on", channel=9, note=36, velocity=80, time=0))
    drums.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=480))
    midi.tracks.extend((pitched, drums))
    midi.save(path)
    _, all_notes = benchmark._midi_notes(path)
    _, pitched_notes = benchmark._midi_notes(path, exclude_drum_channel=True)
    assert {note[0] for note in all_notes} == {36, 60}
    assert [note[0] for note in pitched_notes] == [60]


def test_downbeat_reader_does_not_treat_unmarked_beats_as_downbeats(tmp_path: Path) -> None:
    beat_grid = tmp_path / "beat_grid.json"
    beat_grid.write_text(
        json.dumps({"beat_grid": {"beats": [{"time_sec": 0.0}, {"time_sec": 0.5}, {"time_sec": 1.0}]}}),
        encoding="utf-8",
    )
    assert benchmark._read_time_points(beat_grid) == [0.0, 0.5, 1.0]
    assert benchmark._read_time_points(beat_grid, downbeats=True) == []

    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"downbeats": [{"time_sec": 0.0}, {"time_sec": 2.0}]}), encoding="utf-8")
    assert benchmark._read_time_points(explicit, downbeats=True) == [0.0, 2.0]


@pytest.mark.parametrize(
    ("unit", "duration", "positions"),
    [("eighth", 0.5, 6), ("dotted_quarter", 1.5, 2)],
)
def test_beat_unit_semantic_gate_accepts_explicit_six_eight_state(
    tmp_path: Path,
    unit: str,
    duration: float,
    positions: int,
) -> None:
    path = tmp_path / f"{unit}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "time_signature": {"selected": "6/8"},
                "beat_unit_definition": unit,
                "beat_unit_source": "dbn_meter_state_definition",
                "beat_unit_proven": True,
                "beat_duration_quarters": duration,
                "beats_per_bar": positions,
                "bar_duration_quarters": 3.0,
                "dbn_position_count": positions,
                "bars": [{"beat_count": positions, "duration_quarters": 3.0}],
            }
        ),
        encoding="utf-8",
    )
    gate = benchmark.beat_unit_semantic_gate(
        {"music_context_policy": {"meter": "6/8"}},
        path,
    )
    assert gate["valid"] is True
    assert gate["errors"] == []


def test_beat_unit_semantic_gate_rejects_legacy_six_eight_and_three_four_misreport(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "time_signature": {"selected": "6/8"},
                "beats": [{"time_sec": 0.0, "downbeat": True}, {"time_sec": 0.5}],
            }
        ),
        encoding="utf-8",
    )
    gate = benchmark.beat_unit_semantic_gate(
        {"music_context_policy": {"meter": "6/8"}},
        legacy,
    )
    assert gate["valid"] is False
    assert any("not proven" in error for error in gate["errors"])

    misreported = tmp_path / "misreported.json"
    misreported.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "time_signature": {"selected": "3/4"},
                "beat_unit_definition": "quarter",
                "beat_unit_source": "standard_meter_definition",
                "beat_unit_proven": True,
                "beat_duration_quarters": 1.0,
                "beats_per_bar": 3,
                "bar_duration_quarters": 3.0,
            }
        ),
        encoding="utf-8",
    )
    mismatch = benchmark.beat_unit_semantic_gate(
        {"music_context_policy": {"meter": "6/8"}},
        misreported,
    )
    assert mismatch["valid"] is False
    assert any("does not match expected meter" in error for error in mismatch["errors"])


def test_beat_semantic_gate_uses_reference_annotation_meter_only_for_evaluation(tmp_path: Path) -> None:
    reference_annotation = tmp_path / "reference-beats.json"
    reference_annotation.write_text(
        json.dumps({"beat_grid": {"time_signature": {"selected": "3/4"}}}),
        encoding="utf-8",
    )
    predicted = tmp_path / "predicted-beats.json"
    predicted.write_text(
        json.dumps(
            {
                "time_signature": {"selected": "4/4"},
                "beat_unit_definition": "quarter",
                "beat_unit_source": "standard_meter_definition",
                "beat_unit_proven": True,
                "beat_duration_quarters": 1.0,
                "beats_per_bar": 4,
                "bar_duration_quarters": 4.0,
            }
        ),
        encoding="utf-8",
    )
    case = {"beat_annotation": str(reference_annotation)}
    gate = benchmark.beat_unit_semantic_gate(case, predicted)
    assert gate["expected_meter"] == "3/4"
    assert gate["expected_meter_source"] == "reference_beat_annotation.time_signature.selected"
    assert gate["valid"] is False
    assert any("does not match expected meter" in error for error in gate["errors"])
    # An explicit manifest expectation remains authoritative over the
    # evaluator-only reference fallback.
    explicit = benchmark.beat_unit_semantic_gate(
        {"expected_meter": "4/4", "beat_annotation": str(reference_annotation)}, predicted
    )
    assert explicit["valid"] is True
    assert explicit["expected_meter_source"] == "case.expected_meter"


def test_evaluator_never_scans_neighbor_pipeline_artifacts(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mid"
    _write_midi(reference)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fixture")
    root = tmp_path / "results"
    case_root = root / "case-1"
    case_root.mkdir(parents=True)
    # This is the counterexample that the old rglob fallback selected from a
    # neighboring pipeline when the selected manifest did not declare an
    # artifact.  A missing declaration must remain not_evaluated.
    (case_root / "baseline").mkdir()
    _write_midi(case_root / "baseline" / "wrong.score.mid")
    (case_root / "manifest.json").write_text(json.dumps({"status": "success", "artifacts": []}), encoding="utf-8")
    case = {
        "id": "case-1",
        "title": "fixture",
        "input": str(input_audio),
        "reference_midi": str(reference),
        "reference_midi_reliable": True,
        "beat_annotation": None,
        "evaluation_policy": "reference_metrics",
    }
    evaluated = benchmark.evaluate_case(case, result_root=root)
    assert evaluated["status"] == "not_evaluated"
    assert "final MIDI" in evaluated["reason"]


def test_reference_derived_pjs_beats_are_excluded_from_beatnet_f1(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mid"
    _write_midi(reference)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fixture")
    beat_annotation = tmp_path / "reference-derived-beats.json"
    beat_annotation.write_text(json.dumps({"beat_grid": {"beats": [{"time_sec": 0.0, "downbeat": True}]}}), encoding="utf-8")
    root = tmp_path / "results"
    case_root = root / "pjs001"
    case_root.mkdir(parents=True)
    result_midi = case_root / "pjs001.score.mid"
    _write_midi(result_midi)
    beat_grid = case_root / "beat_grid.json"
    beat_grid.write_text(json.dumps({"beats": [{"time_sec": 0.0, "downbeat": True}]}), encoding="utf-8")
    (case_root / "manifest.json").write_text(
        json.dumps({"status": "success", "artifacts": [{"kind": "score_midi", "relative_path": result_midi.name}, {"kind": "beat_grid_json", "relative_path": beat_grid.name}]}),
        encoding="utf-8",
    )
    case = {
        "id": "pjs001",
        "title": "PJS",
        "input": str(input_audio),
        "reference_midi": str(reference),
        "reference_midi_reliable": True,
        "beat_annotation": str(beat_annotation),
        "evaluation_policy": "reference_metrics",
        "evaluation_scope": "end_to_end_pitch_rhythm_reference_derived_beats",
    }
    evaluated = benchmark.evaluate_case(case, result_root=root)
    assert evaluated["status"] == "evaluated"
    assert evaluated["beat_metrics_eligible"] is False
    assert evaluated["metrics"]["beat_f1"] is None
    assert "reference MIDI" in evaluated["beat_metrics_reason"]


def test_evaluator_uses_selected_manifest_scope_over_static_case_scope(tmp_path: Path) -> None:
    reference = tmp_path / "reference.mid"
    _write_midi(reference)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fixture")
    beat_annotation = tmp_path / "independent-beats.json"
    beat_annotation.write_text(json.dumps({"beat_grid": {"beats": [{"time_sec": 0.0, "downbeat": True}]}}), encoding="utf-8")
    case = {
        "id": "scope-case",
        "title": "scope",
        "input": str(input_audio),
        "reference_midi": str(reference),
        "reference_midi_reliable": True,
        "beat_annotation": str(beat_annotation),
        "beat_annotation_independent": True,
        "evaluation_policy": "reference_metrics",
        "evaluation_scope": "quantizer_isolation_fixture",
    }

    def write_result(root: Path, *, model_output: bool, scope: str) -> None:
        case_root = root / "scope-case"
        case_root.mkdir(parents=True)
        result_midi = case_root / "score.mid"
        _write_midi(result_midi)
        beat_grid = case_root / "beat_grid.json"
        beat_grid.write_text(json.dumps({"beats": [{"time_sec": 0.0, "downbeat": True}]}), encoding="utf-8")
        (case_root / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "effective_evaluation_scope": scope,
                    "raw_model_output": model_output,
                    "final_midi": result_midi.name,
                    "beat_grid": beat_grid.name,
                }
            ),
            encoding="utf-8",
        )

    production_root = tmp_path / "production"
    reference_root = tmp_path / "reference-results"
    write_result(production_root, model_output=True, scope="production_end_to_end")
    write_result(reference_root, model_output=False, scope="quantizer_isolation")
    production = benchmark.evaluate_case(case, result_root=production_root)
    reference_result = benchmark.evaluate_case(case, result_root=reference_root)
    assert production["evaluation_scope"] == "production_end_to_end"
    assert production["beat_metrics_eligible"] is True
    assert reference_result["evaluation_scope"] == "quantizer_isolation"
    assert reference_result["beat_metrics_eligible"] is False
    assert reference_result["metrics"]["beat_f1"] is None


def test_accuracy_gate_reports_quantizer_and_production_scopes_separately() -> None:
    def case(case_id: str, scope: str, rhythm: float) -> dict[str, object]:
        return {
            "id": case_id,
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "evaluation_scope": scope,
            "metrics": {
                "pitch_f1": {"f1": 0.9},
                "chord_retention": {"retention": 0.9},
                "rhythm_error": _rhythm_metric(rhythm),
                "beat_f1": None,
                "downbeat_f1": None,
            },
        }

    new = [case("quantizer", "quantizer_isolation_fixture", 0.1), case("production", "end_to_end_pitch_rhythm_reference_derived_beats", 0.1)]
    baseline = [case("quantizer", "quantizer_isolation_fixture", 0.2), case("production", "end_to_end_pitch_rhythm_reference_derived_beats", 0.2)]
    claim = benchmark.assess_accuracy_claim(new, baseline, minimum_cases=2)
    assert claim["scopes"]["quantizer_isolation_overall"]["ready"] is True
    assert claim["scopes"]["production_end_to_end_subset"]["ready"] is False
    assert claim["scopes"]["production_end_to_end_subset"]["require_beat_metrics"] is True


def test_manual_only_reference_never_becomes_accuracy_result(tmp_path: Path) -> None:
    case = {
        "id": "manual",
        "title": "manual",
        "input": str(tmp_path / "input.wav"),
        "reference_midi": str(tmp_path / "reference.mid"),
        "reference_midi_reliable": False,
        "evaluation_policy": "integrity_and_manual_listening_only",
    }
    result_root = tmp_path / "results" / "manual"
    result_root.mkdir(parents=True)
    (result_root / "manifest.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    evaluated = benchmark.evaluate_case(case, result_root=tmp_path / "results")
    assert evaluated["status"] == "integrity_only"
    assert evaluated["metrics"]["pitch_f1"] is None


def test_accuracy_gate_requires_real_baseline_and_accepts_synthetic_passing_fixture() -> None:
    def case(case_id: int, rhythm: float, pitch: float = 0.9, chord: float = 0.9) -> dict[str, object]:
        return {
            "id": f"case-{case_id}",
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "metrics": {
                "pitch_f1": {"f1": pitch},
                "chord_retention": {"retention": chord},
                "rhythm_error": _rhythm_metric(rhythm),
                "beat_f1": {"f1": 0.9},
                "downbeat_f1": {"f1": 0.8},
            },
        }

    new = [case(index, 0.1) for index in range(30)]
    baseline = [case(index, 0.2, pitch=0.91, chord=0.89) for index in range(30)]
    gate = benchmark.assess_accuracy_claim(new, baseline)
    assert gate["ready"] is True
    assert abs(float(gate["new_mean_rhythm_error_quarter"]) - 0.1) < 1e-9
    assert gate["new_mean_note_onset_f1"] == pytest.approx(0.9)
    assert gate["new_mean_note_onset_f1"] == gate["new_mean_pitch_f1"]
    assert gate["baseline_mean_note_onset_f1"] == pytest.approx(0.91)
    assert gate["baseline_mean_note_onset_f1"] == gate["baseline_mean_pitch_f1"]
    assert gate["shared_case_ids"] == [f"case-{index}" for index in range(30)]

    insufficient = benchmark.assess_accuracy_claim(new[:29], baseline)
    assert insufficient["ready"] is False
    assert "30" in insufficient["reason"]


def test_accuracy_gate_rejects_disjoint_case_ids_even_when_each_side_has_thirty() -> None:
    def case(case_id: int) -> dict[str, object]:
        return {
            "id": f"case-{case_id}",
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "metrics": {
                "pitch_f1": {"f1": 0.9},
                "chord_retention": {"retention": 0.9},
                "rhythm_error": _rhythm_metric(0.1),
                "beat_f1": {"f1": 0.9},
                "downbeat_f1": {"f1": 0.8},
            },
        }

    disjoint = benchmark.assess_accuracy_claim([case(index) for index in range(30)], [case(index) for index in range(30, 60)])
    assert disjoint["ready"] is False
    assert disjoint["shared_case_ids"] == []
    assert "相同可靠 case ID" in disjoint["reason"]


def test_accuracy_gate_requires_beat_coverage_in_addition_to_mean_f1() -> None:
    def case(index: int, *, with_beats: bool) -> dict[str, object]:
        return {
            "id": f"case-{index}",
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "beat_metrics_eligible": with_beats,
            "metrics": {
                "pitch_f1": {"f1": 0.9},
                "chord_retention": {"retention": 0.9},
                "rhythm_error": _rhythm_metric(0.1),
                "beat_f1": {"f1": 0.95} if with_beats else None,
                "downbeat_f1": {"f1": 0.85} if with_beats else None,
            },
        }

    baseline_25 = [case(index, with_beats=True) for index in range(25)]
    partial_25 = [case(index, with_beats=index < 24) for index in range(25)]
    partial_claim = benchmark.assess_accuracy_claim(
        partial_25,
        baseline_25,
        minimum_cases=25,
        minimum_beat_cases=25,
    )
    assert partial_claim["ready"] is False
    assert partial_claim["beat_cases_with_metrics"] == 24
    assert "24/25" in partial_claim["reason"]

    baseline_30 = [case(index, with_beats=True) for index in range(30)]
    sparse_30 = [case(index, with_beats=index == 0) for index in range(30)]
    sparse_claim = benchmark.assess_accuracy_claim(
        sparse_30,
        baseline_30,
        minimum_cases=30,
        minimum_beat_cases=30,
    )
    assert sparse_claim["ready"] is False
    assert sparse_claim["beat_cases_with_metrics"] == 1
    assert "1/30" in sparse_claim["reason"]


def test_build_report_requires_thirty_shared_production_cases(monkeypatch) -> None:
    def evaluated(case_id: str, scope: str, *, beat: float | None) -> dict[str, object]:
        return {
            "id": case_id,
            "status": "evaluated",
            "crash": False,
            "evaluation_policy": "reference_metrics",
            "reference_midi_reliable": True,
            "evaluation_scope": scope,
            "beat_metrics_eligible": beat is not None,
            "metrics": {
                "pitch_f1": {"f1": 0.9},
                "chord_retention": {"retention": 0.9},
                "rhythm_error": _rhythm_metric(0.1),
                "beat_f1": {"f1": beat} if beat is not None else None,
                "downbeat_f1": {"f1": 0.8} if beat is not None else None,
            },
        }

    values = {
        "quantizer": evaluated("quantizer", "quantizer_isolation", beat=None),
        "production": evaluated("production", "production_end_to_end", beat=0.9),
    }
    baseline_values = {
        **values,
        "quantizer": {**values["quantizer"], "metrics": {**values["quantizer"]["metrics"], "rhythm_error": _rhythm_metric(0.2)}},
        "production": {**values["production"], "metrics": {**values["production"]["metrics"], "rhythm_error": _rhythm_metric(0.2)}},
    }
    monkeypatch.setattr(benchmark, "evaluate_case", lambda case, result_root=None: values[str(case["id"])])
    report = benchmark.build_report(
        {"cases": [{"id": "quantizer"}, {"id": "production"}]},
        baseline_report={"cases": [baseline_values["quantizer"], baseline_values["production"]]},
    )
    assert report["accuracy_claim_ready"] is False
    assert "30" in report["accuracy_claim_reason"]
    assert report["accuracy_gate"]["new_reliable_count"] == 1
    assert report["accuracy_gate_scopes"]["quantizer_isolation_overall"]["ready"] is True
    assert report["accuracy_gate_scopes"]["production_end_to_end_subset"]["ready"] is True
