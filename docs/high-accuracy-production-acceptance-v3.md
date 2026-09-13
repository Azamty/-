# High accuracy production acceptance v3

This tracked document describes the v3 acceptance snapshot generated on 2026-09-09. The machine-readable raw selection and result report are under `.artifacts/review/production-acceptance-v3/`; those local artifacts remain outside git because they contain model outputs and generated score files.

The gate population is exactly 30 reliable production cases from the current registry:

- 22 unchanged local MIDI render-domain cases from `fluidsynth-production-raw-v1`;
- 3 updated specialized fixtures (`special-pickup-3-4`, `special-triplet`, and `special-complex-chord`) from `fluidsynth-special-context-production-raw-v1`;
- 5 CCMusic Yueding clips from `ccmusic-production-context-v1`, whose BeatNet grid was computed from the complete original-song mix and then cropped by absolute window.

PJS, Luv Letter, reference-isolation payloads, old oscillator renders, and empty model payloads are excluded. Every selected raw record is `model_output=true`, has effective scope `production_end_to_end`, contains a pitched event, and matches recognizer fingerprint `2d529aba92c9c9930945eaec5fe4e0ab46226fa68ce46d2956d6645049039ff3`. Each v3 case stores the exact recognition payload and a source-file SHA-256 inventory in `raw/source_provenance.json`.

The batch used the same immutable raw payload for the legacy baseline and the current MuseScore high accuracy chain. The runner writes a failed pipeline manifest when an adapter raises, including the case, stage, raw hash, recognizer identity, effective scope, and error. Resuming a failed pipeline archives its prior service output under `retry_history` and retries only that pipeline; raw and a successful sibling pipeline are not rerun.

The v3 results are [production-report-v3.md](../.artifacts/review/production-acceptance-v3/production-report-v3.md) and [production-report-v3.json](../.artifacts/review/production-acceptance-v3/production-report-v3.json). The production report contains exactly 30 top-level gate cases and records both chain statuses and `fixed_total_assignment_v1` metrics. The independent evaluator snapshot is [evaluator-report-v3.json](../.artifacts/review/production-acceptance-v3/evaluator-report-v3.json); it evaluates all 30 reliable production cases and retains the six diagnostic registry cases in its registry snapshot.

The latest measured v3 run completed baseline and new successfully for all 30 cases, with zero crashes. Every new case produced its final Score MIDI, SVG, JLY, Score JSON, and alignment report, and all 30 immutable raw hashes and provenance records still match `raw-selection.json`. Independent BeatNet coverage is 30/30. Mean beat F1 is 0.512941 and mean downbeat F1 is 0.400670. The `fixed_total_assignment_v1` rhythm error is 1.767684 quarter notes for new versus 1.753007 for baseline, a -0.837259% change, so it does not improve by 20%. Mean pitch F1 is 0.162190 for new versus 0.164064 for baseline (change -0.001874, within the 0.01 limit), and chord retention is 0.233974 versus 0.203297 (change +0.030678). The formal gate remains false because the beat F1, downbeat F1, and 20% rhythm-improvement conditions fail; all 30 cases remain in the denominator.

An earlier partial snapshot recorded 23/30 new-chain successes and seven failed adapters while the MusicXML repair work was in progress. That 23/30 result is retained as historical diagnosis only; it is not mixed into the latest acceptance metrics.

The v2 snapshot remains historical in [high-accuracy-production-acceptance-v2.md](high-accuracy-production-acceptance-v2.md). Its raw selection and metrics are not mixed into v3.
