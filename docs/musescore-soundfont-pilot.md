# MuseScore Basic SoundFont pilot

The pilot used only local pinned files and did not change the registry or launch MuScriptor/GAME/BeatNet:

- MuseScore Studio 4.7.4, build 7688c00, executable SHA-256 65F3868FE4421E9223B641664298F5286EAFD426F440CCAC1FDD04B4F24F8C82.
- MuseScore Basic SoundFont MS Basic.sf3, 51,278,610 bytes, SHA-256 5EA2375E8BD7D8E71DEF1036978C1621E85B66934169B6A2744B27B9B3C2D99C.
- MIDI import profile tools/musescore-4.7.4/midi_import_options.xml.
- The runner used --factory-settings --test-mode -M ... --sound-profile "MuseScore Basic" -o ... input.mid through a waited child process, so no background MuseScore process was left behind.

The six samples were fixed before any model run: the four empty-output cases (synthetic-bass-02, special-triplet, special-complex-chord, maestro-midi-07) and the two controls (synthetic-piano-01, maestro-midi-01). Each MIDI was exported to WAV twice, exported back to MIDI once, and exported to MusicXML once. Full machine-readable output is in .artifacts/review/musescore-soundfont-pilot-v3/pilot-report.json.

All 12 WAV exports were byte-identical between repetitions. Every WAV was 44,100 Hz, stereo, and deterministic. The source-preservation and duration checks were:

| case | source/imported notes | max onset delta | max end delta | WAV duration vs MIDI | import timing |
|---|---:|---:|---:|---:|---|
| synthetic-bass-02 | 8/8 | 0.000 s | 0.0013 s | 12.599 vs 9.000 s (+3.599) | pass |
| special-triplet | 12/12 | 0.000 s | 0.0011 s | 5.181 vs 2.182 s (+2.999) | pass |
| special-complex-chord | 8/8 | 0.000 s | 0.0589 s | 6.461 vs 2.308 s (+4.153) | fail |
| maestro-midi-07 | 288/288 | 1.097 s | 1.088 s | 19.615 vs 15.891 s (+3.724) | fail |
| synthetic-piano-01 | 28/27 | 0.000 s | 0.1064 s | 11.421 vs 8.421 s (+3.000) | fail |
| maestro-midi-01 | 130/130 | 0.409 s | 0.248 s | 20.560 vs 16.000 s (+4.560) | fail |

The pilot policy required the source pitch multiset and every matched onset/end to remain within 10 ms. Therefore pilot_ok=false. The two short monophonic controls passed the importer comparison, but the full fixed set did not. MuseScore's WAV determinism is reproducible, yet its MIDI import can lose a piano note, shorten long notes, or shift dense MAESTRO events by hundreds of milliseconds. The exported WAV also contains a deterministic tail beyond the source MIDI event end, ranging from about 3.0 to 4.6 seconds, so file duration cannot be treated as the score duration.

No MuScriptor or BeatNet smoke was run because the fixed pilot did not satisfy the precondition. This avoids comparing model output from an input renderer that already changed source timing or note content.

A direct FluidSynth pilot is now recorded separately in docs/fluidsynth-soundfont-pilot.md. It consumes the original MIDI directly and proves byte determinism and source event completeness for six fixed cases, but it does not change this MuseScore pilot's result or the benchmark registry. Any adoption for the wider render domain remains a separate production change requiring regenerated input hashes and manifests.
