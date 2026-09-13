# High accuracy production acceptance v2

This is the reproducible acceptance snapshot made on 2026-09-09. It reuses immutable production raw payloads and does not run MuScriptor or GAME again. A payload was selected only when its recognizer identity was production 1.2, its model_output provenance was true, its source audio path and hash matched the registry input, and it contained a pitched event. Reference-isolation and empty model payloads were not substituted.

The snapshot below preserves the earlier oscillator-render input results as historical evidence. After this snapshot, all 15 generated/special MIDI cases and all 10 MAESTRO clip inputs were regenerated under the single direct FluidSynth policy `fluidsynth_direct_ms_basic_v1`: pinned FluidSynth 2.6.0 plus the pinned MS Basic SoundFont, 44.1 kHz stereo PCM16, gain 0.2, reverb/chorus disabled, and a fixed 0.25 second tail. The source MIDI is sent directly to FluidSynth, and each cache product carries a render manifest with executable/SoundFont hashes, source event accounting, and two-run byte determinism. This renderer replacement changes input hashes, so it requires a new production raw run; it does not reinterpret the historical raw or gate metrics in this document.

The selected raw manifest contains 30 cases under .artifacts/review/production-acceptance-v2/raw-selection.json. It uses recognizer fingerprint 2d529aba92c9c9930945eaec5fe4e0ab46226fa68ce46d2956d6645049039ff3. The 30 consist of 26 production cases with pitched raw plus PJS002–005, which are preserved as diagnostic integrity-only cases because their reference annotation is not reliable for an accuracy claim.

The unified baseline/new pipeline results are in [production-report-v2.md](../.artifacts/review/production-acceptance-v2/production-report-v2.md) and [production-report-v2.json](../.artifacts/review/production-acceptance-v2/production-report-v2.json). Baseline completed 30/30. The historical new chain completed 29/30; PJS003 failed in the new render stage because LilyPond rejected the fine-grid token e'62 as “not a duration”, with a barcheck warning at 15/32. That failure was caused by a vendor tied-note collapse regex matching the suffix of a longer duration token; the boundary fix is covered by the stage 7 render regression. A standalone rerun using the same immutable PJS003 production raw completed baseline and new successfully under `.artifacts/review/production-acceptance-v2-pjs003-fix/`; PJS remains diagnostic-only and does not alter the 30-case gate snapshot below.

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

The historical generated synthetic cases used `deterministic_harmonic_oscillator_v1`; their manifests preserved MIDI pitch and timing and applied fixed notated downbeat velocity accents. The regenerated production inputs use the direct FluidSynth policy described above. The bass case is intentionally low-register and has only one monophonic line. The current specialized fixtures use a fixed four-complete-measure context rule; the older short triplet/complex-chord measurements in this historical snapshot are not the current production inputs. The complex chord retains eight simultaneous notes per chord unit. MAESTRO-07 is a deterministic first-note-aligned eight-bar local render with source velocities preserved; its 288 notes are spread over the 16-second clip and the local renderer does not model piano timbre, pedal, room acoustics, or performance nuance. These facts describe the fixture domain and do not turn an empty model result into a success.

## Current renderer policy

The registry now records the direct renderer policy and input hashes for the 25 local MIDI cases. Generated fixtures, MAESTRO clips, and the five special fixtures use the same renderer parameters; no case-specific model result is used to select a renderer or shorten a clip. The MAESTRO selection manifest remains hash-verified and records the original archive member, fixed eight-bar source interval, clip MIDI, clip beat grid, full render, clip render, and source-event completeness. Re-rendering or window changes create new input hashes and require new production raw. CCMusic mixed-song inputs and PJS diagnostic audio are outside this renderer policy.
