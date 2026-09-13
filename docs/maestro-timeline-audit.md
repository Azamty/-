# MAESTRO timeline audit

The ten local MAESTRO renders preserve source MIDI pitch and event time, but their generated beat grids are not valid musical beat annotations. Each source file has one fixed 120 BPM transport event and a 4/4 marker. The MIDI provides aligned performance key-event timestamps, not score-aligned beat/downbeat labels. The previous fixture builder interpreted every 480 transport ticks as one musical beat and therefore assigned the same 33-point grid from 0.0 through 16.0 seconds to ten different performances.

The v1 audit in `.artifacts/review/maestro-timeline-audit-v1` rules out a renderer or crop offset:

- all MIDI tick-to-second conversions reproduce the saved grid exactly;
- the annotation is generated from the cropped, zero-based clip MIDI, not the untrimmed MIDI's absolute time;
- FluidSynth preserves every source note and the output duration exactly matches the source last-note time plus the fixed 0.25-second release tail;
- the first nonzero audio sample follows the first MIDI note by only 1.623–3.006 ms, below the 70 ms beat tolerance;
- each grid includes an average of 2.6 synthetic beats before the first played note because the so-called bar boundary is only a transport-tick boundary.

Against the invalid fixed grid, MAESTRO-only mean beat F1 is 0.216374 for BeatNet and 0.176824 for madmom. A global offset alone raises them only to 0.377307 and 0.323408. An affine reference-only fit raises them further because it scales each detected musical pulse onto the unrelated 120 BPM ruler; this is an oracle diagnostic and cannot be used in production. The available Beat This pilot covers only `maestro-midi-01`; its official DBN pulse is close to madmom's half-time interpretation and also disagrees with the 0.5-second grid.

The registry now marks these ten tick grids `beat_annotation_independent=false`. Their reference-MIDI score eligibility is unchanged because that question is outside this audit. The 30-case beat gate is consequently unmet until ten independently aligned beat annotations or replacement public beat-annotated cases are supplied. Production inference and all saved tracker outputs are unchanged.
