import assert from "node:assert/strict";
import test from "node:test";
import { classifyMelodyHarmonyScoreArtifacts, classifyScoreArtifacts, filterArtifactsForSelectionRevision, findMelodyHarmonyMidi, selectionArtifactRevision } from "../src/scoreArtifacts.ts";

test("R2 score view keeps the current part and hides the retained R1 main melody", () => {
  const result = classifyScoreArtifacts([
    {
      artifact_id: "v2-selection-r1-main-melody-score-svg-long",
      kind: "main_melody_score_svg_long",
      relative_path: "output/selections/rev-0001/main-melody/high-accuracy/main-melody.long.svg",
    },
    {
      artifact_id: "v2-selection-r2-piano-score-svg-long",
      kind: "instrument_score_svg_long",
      relative_path: "output/selections/rev-0002/piano/high-accuracy/piano.long.svg",
    },
    {
      artifact_id: "v2-selection-r2-piano-score-svg-1",
      kind: "instrument_score_svg",
      page: 1,
      relative_path: "output/selections/rev-0002/piano/high-accuracy/piano-1.svg",
    },
  ], "instrument", 2);

  assert.deepEqual(result.long.map((item) => item.artifact_id), ["v2-selection-r2-piano-score-svg-long"]);
  assert.deepEqual(result.paged.map((item) => item.artifact_id), ["v2-selection-r2-piano-score-svg-1"]);
});

test("R2 combined score view hides retained R1 harmony pages and MIDI", () => {
  const artifacts = [
    {
      artifact_id: "v2-selection-r1-melody-harmony-score-svg-long",
      kind: "melody_harmony_score_svg_long",
      relative_path: "output/selections/rev-0001/melody-harmony/melody-harmony.long.svg",
    },
    {
      artifact_id: "v2-selection-r1-melody-harmony-score-midi",
      kind: "melody_harmony_score_midi",
      relative_path: "output/selections/rev-0001/melody-harmony/melody-harmony.score.mid",
    },
    {
      artifact_id: "v2-selection-r2-melody-harmony-score-svg-long",
      kind: "melody_harmony_score_svg_long",
      relative_path: "output/selections/rev-0002/melody-harmony/melody-harmony.long.svg",
    },
    {
      artifact_id: "v2-selection-r2-melody-harmony-score-svg-1",
      kind: "melody_harmony_score_svg",
      page: 1,
      relative_path: "output/selections/rev-0002/melody-harmony/melody-harmony-1.svg",
    },
    {
      artifact_id: "v2-selection-r2-melody-harmony-score-midi",
      kind: "melody_harmony_score_midi",
      relative_path: "output/selections/rev-0002/melody-harmony/melody-harmony.score.mid",
    },
  ];

  const result = classifyMelodyHarmonyScoreArtifacts(artifacts, 2);
  assert.deepEqual(result.long.map((item) => item.artifact_id), ["v2-selection-r2-melody-harmony-score-svg-long"]);
  assert.deepEqual(result.paged.map((item) => item.artifact_id), ["v2-selection-r2-melody-harmony-score-svg-1"]);
  assert.equal(findMelodyHarmonyMidi(artifacts, 2)?.artifact_id, "v2-selection-r2-melody-harmony-score-midi");
});

test("artifacts without a revision marker remain visible for legacy compatibility", () => {
  const legacy = {
    artifact_id: "score-svg-long",
    kind: "main_melody_score_svg_long",
    relative_path: "output/score-long.svg",
  };

  assert.equal(selectionArtifactRevision(legacy), null);
  assert.deepEqual(classifyScoreArtifacts([legacy], "instrument", 2).long.map((item) => item.artifact_id), ["score-svg-long"]);
});

test("a pending R2 does not expose R1 selection downloads but keeps original artifacts", () => {
  const artifacts = [
    { artifact_id: "v2-selection-r1-main-melody-score-midi", kind: "main_melody_score_midi" },
    { artifact_id: "v2-selection-r1-main-melody-musicxml", kind: "main_melody_musicxml" },
    { artifact_id: "v2-selection-r1-main-melody-score-json", kind: "main_melody_score_json" },
    { artifact_id: "v2-original-midi", kind: "original_midi" },
    { artifact_id: "v2-source-audio", kind: "source_audio" },
  ];

  assert.deepEqual(
    filterArtifactsForSelectionRevision(artifacts, 2).map((item) => item.artifact_id),
    ["v2-original-midi", "v2-source-audio"],
  );
});
