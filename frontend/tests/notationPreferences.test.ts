import assert from "node:assert/strict";
import test from "node:test";
import { NOTATION_KEYS, keyLabel, isScoreDownload } from "../src/notationPreferences.ts";

test("all minor keys have selectable options and explain numbered key", () => {
  assert.ok(NOTATION_KEYS.includes("F#m"));
  assert.equal(keyLabel("F#m"), "F♯小调（F#m）· 1=A");
  assert.equal(keyLabel("Bm"), "B小调（Bm）· 1=D");
  assert.equal(NOTATION_KEYS.length, 28);
});
test("only rendered PDF, SVG and MIDI appear in downloads", () => {
  for (const family of ["instrument", "vocal", "main_melody", "melody_harmony"]) {
    for (const format of ["pdf", "midi", "svg", "svg_long"]) assert.ok(isScoreDownload(`${family}_score_${format}`));
    for (const format of ["json", "musicxml", "jly"]) assert.ok(!isScoreDownload(`${family}_score_${format}`));
  }
  for (const kind of ["selected_midi", "original_midi", "high_accuracy_manifest", "instrument_performance_midi", "svg_zip"]) assert.ok(!isScoreDownload(kind));
});
