# High accuracy production acceptance v3

This tracked document describes the v3 acceptance snapshot generated on 2026-09-09. The machine-readable raw selection and result report are under `.artifacts/review/production-acceptance-v3/`; those local artifacts remain outside git because they contain model outputs and generated score files.

The gate population is exactly 30 reliable production cases from the current registry:

- 22 unchanged local MIDI render-domain cases from `fluidsynth-production-raw-v1`;
- 3 updated specialized fixtures (`special-pickup-3-4`, `special-triplet`, and `special-complex-chord`) from `fluidsynth-special-context-production-raw-v1`;
- 5 CCMusic Yueding clips from `ccmusic-production-context-v1`, whose BeatNet grid was computed from the complete original-song mix and then cropped by absolute window.

PJS, Luv Letter, reference-isolation payloads, old oscillator renders, and empty model payloads are excluded. Every selected raw record is `model_output=true`, has effective scope `production_end_to_end`, contains a pitched event, and matches recognizer fingerprint `2d529aba92c9c9930945eaec5fe4e0ab46226fa68ce46d2956d6645049039ff3`. Each v3 case stores the exact recognition payload and a source-file SHA-256 inventory in `raw/source_provenance.json`.

The batch used the same immutable raw payload for the legacy baseline and the current MuseScore high accuracy chain. The runner writes a failed pipeline manifest when an adapter raises, including the case, stage, raw hash, recognizer identity, effective scope, and error. Resuming a failed pipeline archives its prior service output under `retry_history` and retries only that pipeline; raw and a successful sibling pipeline are not rerun.

The v3 results are [production-report-v3.md](../.artifacts/review/production-acceptance-v3/production-report-v3.md) and [production-report-v3.json](../.artifacts/review/production-acceptance-v3/production-report-v3.json). The report contains exactly 30 top-level gate cases and records both chain statuses and fixed-total assignment metrics. The independent evaluator snapshot is [evaluator-report-v3.json](../.artifacts/review/production-acceptance-v3/evaluator-report-v3.json).

The measured v3 run has 23/30 new-chain successes and 7 failed new adapters; baseline completed 30/30. The failed cases are `synthetic-bass-01`, `maestro-midi-01`, `maestro-midi-04`, `maestro-midi-05`, `maestro-midi-07`, `maestro-midi-08`, and `special-complex-chord`. Their failure stages and preserved logs are recorded per case in the report and pipeline directories. The formal gate is false because the shared successful population is 23/30, independent BeatNet coverage is 23/30, mean beat F1 is 0.575020, mean downbeat F1 is 0.467376, and the shared fixed-total rhythm error does not improve by 20% (new 1.631318 versus baseline 1.623445 quarter notes). No failed case is silently averaged away.

The v2 snapshot remains historical in [high-accuracy-production-acceptance-v2.md](high-accuracy-production-acceptance-v2.md). Its raw selection and metrics are not mixed into v3.
