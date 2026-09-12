import assert from "node:assert/strict";
import test from "node:test";
import { classifyMelodyHarmonyScoreArtifacts, classifyScoreArtifacts, findMelodyHarmonyMidi } from "../src/scoreArtifacts.ts";

test("main melody score long SVG is in the default long view", () => {
  const result = classifyScoreArtifacts([
    { artifact_id: "page-1", kind: "main_melody_score_svg", page: 1 },
    { artifact_id: "long", kind: "main_melody_score_svg_long" },
  ], "instrument");

  assert.deepEqual(result.long.map((item) => item.artifact_id), ["long"]);
  assert.deepEqual(result.paged.map((item) => item.artifact_id), ["page-1"]);
});

test("instrument score page stays in the collapsible paged bucket", () => {
  const result = classifyScoreArtifacts([
    { artifact_id: "page-2", kind: "instrument_score_svg", page: 2 },
    { artifact_id: "page-1", kind: "instrument_score_svg", page: 1 },
  ], "instrument");

  assert.deepEqual(result.long, []);
  assert.deepEqual(result.paged.map((item) => item.artifact_id), ["page-1", "page-2"]);
});

test("vocal classifier accepts the production family kind", () => {
  const result = classifyScoreArtifacts([
    { artifact_id: "vocal-long", kind: "vocal_score_svg_long" },
    { artifact_id: "vocal-page", kind: "vocal_score_svg", page: 1 },
  ], "vocal");

  assert.equal(result.long[0]?.artifact_id, "vocal-long");
  assert.equal(result.paged[0]?.artifact_id, "vocal-page");
});

test("melody harmony classifier puts the combined long SVG first and keeps pages", () => {
  const artifacts = [
    { artifact_id: "v2-selection-r3-melody-harmony-score-svg-2", kind: "melody_harmony_score_svg", page: 2 },
    { artifact_id: "v2-selection-r3-melody-harmony-score-svg-long", kind: "melody_harmony_score_svg_long" },
    { artifact_id: "v2-selection-r3-melody-harmony-score-svg-1", kind: "melody_harmony_score_svg", page: 1 },
    { artifact_id: "v2-selection-r3-melody-harmony-score-midi", kind: "melody_harmony_score_midi" },
  ];

  const result = classifyMelodyHarmonyScoreArtifacts(artifacts, 3);
  assert.deepEqual(result.long.map((item) => item.artifact_id), ["v2-selection-r3-melody-harmony-score-svg-long"]);
  assert.deepEqual(result.paged.map((item) => item.artifact_id), [
    "v2-selection-r3-melody-harmony-score-svg-1",
    "v2-selection-r3-melody-harmony-score-svg-2",
  ]);
  assert.equal(findMelodyHarmonyMidi(artifacts, 3)?.artifact_id, "v2-selection-r3-melody-harmony-score-midi");
});
