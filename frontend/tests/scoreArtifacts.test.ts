import assert from "node:assert/strict";
import test from "node:test";
import { classifyScoreArtifacts } from "../src/scoreArtifacts.ts";

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
