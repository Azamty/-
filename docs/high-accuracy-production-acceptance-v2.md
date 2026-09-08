# High accuracy production acceptance v2

This is the reproducible acceptance snapshot made on 2026-09-09. It reuses immutable production raw payloads and does not run MuScriptor or GAME again. A payload was selected only when its recognizer identity was production 1.2, its model_output provenance was true, its source audio path and hash matched the registry input, and it contained a pitched event. Reference-isolation and empty model payloads were not substituted.

The selected raw manifest contains 30 cases under .artifacts/review/production-acceptance-v2/raw-selection.json. It uses recognizer fingerprint 2d529aba92c9c9930945eaec5fe4e0ab46226fa68ce46d2956d6645049039ff3. The 30 consist of 26 production cases with pitched raw plus PJS002–005, which are preserved as diagnostic integrity-only cases because their reference annotation is not reliable for an accuracy claim.

The unified baseline/new pipeline results are in [production-report-v2.md](../.artifacts/review/production-acceptance-v2/production-report-v2.md) and [production-report-v2.json](../.artifacts/review/production-acceptance-v2/production-report-v2.json). Baseline completed 30/30. The new chain completed 29/30; PJS003 failed in the new render stage because LilyPond rejected the fine-grid token e'62 as “not a duration”, with a barcheck warning at 15/32. The failure is retained as a crash and did not enter the accuracy averages. The current report therefore has 26 evaluated production cases, 3 PJS integrity-only cases, and 1 crash.

The required 30-case gate is false for the measured data:

| quantity | new | baseline |
|---|---:|---:|
| shared reliable production cases | 26/30 | 26/30 |
| independent BeatNet metric cases | 26/30 | 26/30 |
| mean beat F1 | 0.566663 | — |
| mean downbeat F1 | 0.487190 | — |
| fixed-total rhythm error (quarter notes) | 1.884942 | 1.871697 |
| pitch F1 | 0.253112 | 0.188642 |
| chord retention | 0.278409 | 0.136364 |

The gate requires 30 shared reliable cases, beat F1 at least 0.85, downbeat F1 at least 0.75, and at least 20% lower fixed-total rhythm error. None of the missing-case failures were hidden in the report.

## Missing pitched-raw diagnostics

The read-only inputs, reference MIDI, beat grid, render manifests, audio features, model progress, generated empty original.mid, and logs are in [missing-case-diagnostics.md](../.artifacts/review/production-acceptance-v2/missing-case-diagnostics.md) and [missing-case-diagnostics.json](../.artifacts/review/production-acceptance-v2/missing-case-diagnostics.json). All four model runs reached selection_ready with deterministic settings (seed=20260907, greedy decoder, sampling disabled, NumPy/Torch/CUDA deterministic flags enabled) and emitted zero notes.

| case | audio | reference MIDI | reference density | max simultaneous | model progress |
|---|---:|---:|---:|---:|---|
| synthetic-bass-02 | 9.300 s, median centroid 90.3 Hz | 8 notes | 0.889/s | 1 | 0 notes, 2/2 |
| special-triplet | 2.455 s, median centroid 653.0 Hz | 12 notes | 5.500/s | 2 | 0 notes, 1/1 |
| special-complex-chord | 2.596 s, median centroid 226.0 Hz | 8 notes | 3.467/s | 8 | 0 notes, 1/1 |
| maestro-midi-07 | 16.000 s, median centroid 176.7 Hz | 288 notes | 18.124/s | 7 | 0 notes, 4/4 |

The generated synthetic cases use deterministic_harmonic_oscillator_v1; their manifests preserve MIDI pitch and timing and apply fixed notated downbeat velocity accents. The bass case is intentionally low-register and has only one monophonic line. The triplet and complex-chord cases are short, while the complex chord has eight simultaneous notes. MAESTRO-07 is a deterministic first-note-aligned eight-bar local render with source velocities preserved; its 288 notes are spread over the 16-second clip and the local renderer does not model piano timbre, pedal, room acoustics, or performance nuance. These facts describe the fixture domain and do not turn the empty model result into a success.

## Pre-recognition improvement proposal

The next production-change phase should use one fixed policy chosen before recognition:

- Prefer a pinned standard GM SoundFont/MuseScore headless render for generated and local-MIDI domains, preserving MIDI pitch, onset, duration, velocity, tempo and meter. Record renderer, SoundFont/version, hashes, and velocity policy in the case manifest.
- If the deterministic oscillator remains, use one pinned program-family preset with fixed attack/release and harmonic partials, applied uniformly to each family. Keep score-derived downbeat accents deterministic and independent of model output.
- Keep source-MIDI windowing deterministic: the first note's containing bar plus at most eight complete bars, or the complete short fixture. Any maximum note-density rule must be fixed in registry/source preparation and record the original and discarded-window hashes. Never select or shorten a window after inspecting model notes or scores.
- Re-rendering or window changes must create new input hashes and new production raw; the current empty raw remains immutable evidence.

No production renderer, registry label, threshold, or acceptance gate was changed in this snapshot.
