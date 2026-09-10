# High accuracy benchmark inventory

This report is generated from the registry and local result roots. `production_model_raw` requires `model_output=true`, a registry input hash/path match, and at least one non-drum event. Reference-isolation payloads remain diagnostic and never satisfy that status.

- Registry cases: **36**
- Strict production-scope cases with independent beat annotation and successful model raw: **26**
- Quantizer-fixture model smokes (diagnostic only): **0**
- Categories: `{'local_manual_only': 1, 'official_piano_rendered': 10, 'specialized_fixture': 5, 'synthetic_rendered': 10, 'vocal': 5, 'vocal_diagnostic': 5}`
- Case roles: `{'diagnostic_only': 5, 'manual_only': 1, 'production_scope': 30}`
- Production cases still needed to reach 30: **4**

| Case | Category | Render domain | Role | Input | Beat eligible | Raw status | Pitched events | Effective scopes seen |
|---|---|---|---|---:|---:|---|---:|---|
| `synthetic-piano-01` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 23 | production_end_to_end |
| `synthetic-piano-02` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `synthetic-piano-03` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 32 | production_end_to_end |
| `synthetic-guitar-01` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 15 | production_end_to_end |
| `synthetic-guitar-02` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 14 | production_end_to_end |
| `synthetic-bass-01` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 33 | production_end_to_end |
| `synthetic-bass-02` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_failed` | — | — |
| `synthetic-multitrack-01` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 24 | production_end_to_end |
| `synthetic-multitrack-02` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 25 | production_end_to_end |
| `synthetic-multitrack-03` | `synthetic_rendered` | `synthetic_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 37 | production_end_to_end |
| `maestro-midi-01` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 123 | production_end_to_end |
| `maestro-midi-02` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 176 | production_end_to_end |
| `maestro-midi-03` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 169 | production_end_to_end |
| `maestro-midi-04` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 310 | production_end_to_end |
| `maestro-midi-05` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 217 | production_end_to_end |
| `maestro-midi-06` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 27 | production_end_to_end |
| `maestro-midi-07` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_failed` | — | — |
| `maestro-midi-08` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 2748 | production_end_to_end |
| `maestro-midi-09` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 51 | production_end_to_end |
| `maestro-midi-10` | `official_piano_rendered` | `maestro_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 14 | production_end_to_end |
| `pjs001` | `vocal_diagnostic` | `isolated_vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 35 | production_end_to_end |
| `pjs002` | `vocal_diagnostic` | `isolated_vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 32 | production_end_to_end |
| `pjs003` | `vocal_diagnostic` | `isolated_vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 21 | production_end_to_end |
| `pjs004` | `vocal_diagnostic` | `isolated_vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 30 | production_end_to_end |
| `pjs005` | `vocal_diagnostic` | `isolated_vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 44 | production_end_to_end |
| `ccmusic-yueding-01` | `vocal` | `research_mixed_song` | `production_scope` | yes | yes | `production_model_raw` | 21 | production_end_to_end |
| `ccmusic-yueding-02` | `vocal` | `research_mixed_song` | `production_scope` | yes | yes | `production_model_raw` | 21 | production_end_to_end |
| `ccmusic-yueding-03` | `vocal` | `research_mixed_song` | `production_scope` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `ccmusic-yueding-04` | `vocal` | `research_mixed_song` | `production_scope` | yes | yes | `production_model_raw` | 21 | production_end_to_end |
| `ccmusic-yueding-05` | `vocal` | `research_mixed_song` | `production_scope` | yes | yes | `production_model_raw` | 26 | production_end_to_end |
| `special-pickup-3-4` | `specialized_fixture` | `specialized_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 9 | production_end_to_end |
| `special-6-8` | `specialized_fixture` | `specialized_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `special-triplet` | `specialized_fixture` | `specialized_local_midi_render` | `production_scope` | yes | yes | `production_model_failed` | — | — |
| `special-tempo-change` | `specialized_fixture` | `specialized_local_midi_render` | `production_scope` | yes | yes | `production_model_raw` | 16 | production_end_to_end |
| `special-complex-chord` | `specialized_fixture` | `specialized_local_midi_render` | `production_scope` | yes | yes | `production_model_failed` | — | — |
| `luv-letter` | `local_manual_only` | `manual_local_candidate` | `manual_only` | yes | no | `no_raw` | — | — |

## Special context v1 follow-up

The dedicated raw-only rerun in `.artifacts/review/fluidsynth-special-context-production-raw-v1` uses the registry hashes after applying `minimum_four_complete_notated_measures_v1`; it does not overwrite the historical raw roots used for the table above. All three changed fixtures completed the real MuScriptor 1.2 + BeatNet route with `model_output=true`: `special-pickup-3-4` returned 13 pitched events, `special-triplet` returned 47, and `special-complex-chord` returned 20. `special-6-8` and `special-tempo-change` were unchanged and were not rerun. No baseline or new score pipeline was run in this follow-up.

## Gate interpretation

The registry's 30-case production composition contains 10 deterministic known-MIDI renders, 10 MAESTRO performance-MIDI renders, 5 CCMusic mixed-song segments, and 5 specialized deterministic segments. The MAESTRO reference-MIDI score eligibility is unchanged by this audit, but its fixed 120 BPM transport grid is not independent musical beat ground truth. Those ten annotations are diagnostic-only and excluded from beat/downbeat F1, leaving the 30-case beat gate explicitly incomplete until independent annotations or replacements are added.
The local MAESTRO archive/render cache is hash-verified by its selection manifest. Each selected case now uses a deterministic eight-bar source-MIDI clip, with the original member hash, source quarter interval, clip hash, tempo/meter, and independent beat grid recorded in provenance. The 10 MAESTRO clips and the 15 generated/special MIDI cases use the single direct FluidSynth 2.6.0/MS Basic renderer; the registry records the fixed 44.1 kHz stereo PCM16 policy, renderer/SoundFont hashes, source-event completeness, and per-input render-manifest hashes. Historical oscillator artifacts are not production inputs.

## CCMusic context audit

The five CCMusic production raw payloads in the scanned roots were produced from 12-second clip inputs, so their BeatNet calls were clip-context calls. The full aligned Yueding mix is available locally. The compliant reproducible path is one BeatNet call on that complete mix, retaining the absolute output times and hash, then a deterministic crop to each segment's `[audio_start_sec, audio_start_sec + duration_sec]` window with one boundary beat on each side for interpolation and a recorded absolute-to-local offset. The score-derived beat grid remains an evaluation annotation only; it must not be passed as a tempo, phase, or meter override. Existing full-window diagnostic output should be cited separately from the clip raw and must not be silently merged into it.

## Failure semantics

A model failure, empty pitched-event result, absent independent beat annotation, or input hash mismatch remains visible in the inventory. It is not converted into a successful case by copying a reference raw payload.
