# High accuracy benchmark inventory

This report is generated from the registry and local result roots. `production_model_raw` requires `model_output=true`, a registry input hash/path match, and at least one non-drum event. Reference-isolation payloads remain diagnostic and never satisfy that status.

- Registry cases: **36**
- Strict production-scope cases with independent beat annotation and successful model raw: **5**
- Quantizer-fixture model smokes (diagnostic only): **16**
- Categories: `{'local_manual_only': 1, 'official_piano_rendered': 10, 'specialized_fixture': 5, 'synthetic_rendered': 10, 'vocal': 5, 'vocal_diagnostic': 5}`
- Case roles: `{'diagnostic_only': 5, 'production_scope': 5, 'quantizer_fixture': 25, 'unclassified': 1}`
- Production cases still needed to reach 30: **25**

| Case | Category | Role | Input | Beat eligible | Raw status | Pitched events | Effective scopes seen |
|---|---|---|---:|---:|---|---:|---|
| `synthetic-piano-01` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 23 | production_end_to_end |
| `synthetic-piano-02` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `synthetic-piano-03` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 32 | production_end_to_end |
| `synthetic-guitar-01` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 15 | production_end_to_end |
| `synthetic-guitar-02` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 14 | production_end_to_end |
| `synthetic-bass-01` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 33 | production_end_to_end |
| `synthetic-bass-02` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_failed` | — | — |
| `synthetic-multitrack-01` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 24 | production_end_to_end |
| `synthetic-multitrack-02` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 25 | production_end_to_end |
| `synthetic-multitrack-03` | `synthetic_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 37 | production_end_to_end |
| `maestro-midi-01` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `no_raw` | — | — |
| `maestro-midi-02` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 35075 | production_end_to_end |
| `maestro-midi-03` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 27217 | production_end_to_end |
| `maestro-midi-04` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 5257 | production_end_to_end |
| `maestro-midi-05` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `no_raw` | — | — |
| `maestro-midi-06` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `production_model_raw` | 47760 | production_end_to_end |
| `maestro-midi-07` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `no_raw` | — | — |
| `maestro-midi-08` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `no_raw` | — | — |
| `maestro-midi-09` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `no_raw` | — | — |
| `maestro-midi-10` | `official_piano_rendered` | `quantizer_fixture` | yes | yes | `no_raw` | — | — |
| `pjs001` | `vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 35 | production_end_to_end |
| `pjs002` | `vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 32 | production_end_to_end |
| `pjs003` | `vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 21 | production_end_to_end |
| `pjs004` | `vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 30 | production_end_to_end |
| `pjs005` | `vocal_diagnostic` | `diagnostic_only` | yes | no | `production_model_raw` | 44 | production_end_to_end |
| `ccmusic-yueding-01` | `vocal` | `production_scope` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `ccmusic-yueding-02` | `vocal` | `production_scope` | yes | yes | `production_model_raw` | 21 | production_end_to_end |
| `ccmusic-yueding-03` | `vocal` | `production_scope` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `ccmusic-yueding-04` | `vocal` | `production_scope` | yes | yes | `production_model_raw` | 21 | production_end_to_end |
| `ccmusic-yueding-05` | `vocal` | `production_scope` | yes | yes | `production_model_raw` | 26 | production_end_to_end |
| `special-pickup-3-4` | `specialized_fixture` | `quantizer_fixture` | yes | yes | `production_model_raw` | 9 | production_end_to_end |
| `special-6-8` | `specialized_fixture` | `quantizer_fixture` | yes | yes | `production_model_raw` | 22 | production_end_to_end |
| `special-triplet` | `specialized_fixture` | `quantizer_fixture` | yes | yes | `production_model_failed` | — | — |
| `special-tempo-change` | `specialized_fixture` | `quantizer_fixture` | yes | yes | `production_model_raw` | 16 | production_end_to_end |
| `special-complex-chord` | `specialized_fixture` | `quantizer_fixture` | yes | yes | `production_model_failed` | — | — |
| `luv-letter` | `local_manual_only` | `unclassified` | yes | no | `no_raw` | — | — |

## Gate interpretation

The registry's 30-case composition is currently a fixture selection policy: 10 generated rendered instruments, 10 MAESTRO MIDI renders, 5 CCMusic segments, and 5 specialized generated fixtures. The generated and MAESTRO cases remain quantizer-isolation fixtures even when a real model raw payload is available; the audit never relabels them. On the current local inputs, only the five CCMusic records have the registry's production scope plus independent beat annotation, so the strict production path is short by 25 cases. Those additional cases require fresh mixed-song audio with independent beat/downbeat annotations. A reference-derived payload is never substituted for a missing model result. PJS remains a five-case vocal diagnostic set because its beat annotation is derived from the same reference MIDI and is not independent.
The local MAESTRO archive/render cache is present and hash-verified by its selection manifest; this makes the ten files executable for quantizer diagnostics, but its render manifest explicitly records `production_end_to_end=false`.

## CCMusic context audit

The five original CCMusic production raw payloads were produced from 12-second clip inputs, so their first BeatNet calls were clip-context calls. A reproducible full-track context run is now recorded under `.artifacts/review/ccmusic-production-context-v1`: the complete aligned Yueding mix was decoded once, producing 179 beats, and each case carries the full-track audio/grid hashes, absolute/local window mapping, and the nearest preceding/following boundary beat. The copied raw notes are byte-for-byte the original GAME-cleaned notes; only beat analysis is replaced by the full-track window view. The score-derived beat grid remains an evaluation annotation only and is never passed as a tempo, phase, or meter override. The five windows still share one recording and one BeatNet decode, so this context run does not increase independent sample count.

## Failure semantics

A model failure, empty pitched-event result, absent independent beat annotation, or input hash mismatch remains visible in the inventory. It is not converted into a successful case by copying a reference raw payload.
