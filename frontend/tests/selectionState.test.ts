import assert from "node:assert/strict";
import test from "node:test";
import { canPersistSelection, resolveSelectionValues, selectionStorageKey } from "../src/selectionState.ts";

test("a first ready instrumental job uses the analysis suggestion", () => {
  assert.deepEqual(resolveSelectionValues(
    { bpm: 96, key: "Am", time_signature: "3/4" },
    null,
    null,
  ), { bpm: "96", key: "Am", meter: "3/4" });
});

test("saved user values survive refresh when no server override exists", () => {
  assert.deepEqual(resolveSelectionValues(
    { bpm: 96, key: "Am", time_signature: "3/4" },
    { bpm: 88, key: "G", meter: "4/4" },
    null,
  ), { bpm: "88", key: "G", meter: "4/4" });
});

test("submitted selection overrides are always the highest priority", () => {
  assert.deepEqual(resolveSelectionValues(
    { bpm: 96, key: "Am", time_signature: "3/4" },
    { bpm: 88, key: "G", meter: "4/4" },
    { bpm: 110, key: "D", time_signature: "6/8" },
  ), { bpm: "110", key: "D", meter: "6/8" });
});

test("selection drafts are persisted only after recognition is ready and stay job scoped", () => {
  assert.equal(canPersistSelection("queued", false), false);
  assert.equal(canPersistSelection("recognizing", false), false);
  assert.equal(canPersistSelection("selection_ready", false), false);
  assert.equal(canPersistSelection("selection_ready", true), true);
  assert.equal(canPersistSelection("completed", true), true);
  assert.equal(selectionStorageKey("job-a"), "jianpu-v2-selection:job-a");
  assert.notEqual(selectionStorageKey("job-a"), selectionStorageKey("job-b"));
});
