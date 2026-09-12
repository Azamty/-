import assert from "node:assert/strict";
import test from "node:test";
import { canPersistSelection, isValidJobId, parseSelectionDraft, resolveInitialJobId, resolveSelectionValues, SELECTION_DRAFT_SCHEMA, SELECTION_DRAFT_VERSION, selectionStorageKey } from "../src/selectionState.ts";

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
    { schema: SELECTION_DRAFT_SCHEMA, version: SELECTION_DRAFT_VERSION, job_id: "job-a", hydrated: true, bpm: 88, key: "G", meter: "4/4", manual: { bpm: true, key: true, meter: true } },
    null,
  ), { bpm: "88", key: "G", meter: "4/4" });
});

test("legacy unmarked values cannot override a new analysis suggestion", () => {
  assert.deepEqual(resolveSelectionValues(
    { bpm: 96, key: "Am", time_signature: "3/4" },
    { bpm: 120, key: "C", meter: "4/4" },
    null,
  ), { bpm: "96", key: "Am", meter: "3/4" });
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

test("selection draft schema rejects legacy and cross-job records", () => {
  const marked = JSON.stringify({ schema: SELECTION_DRAFT_SCHEMA, version: SELECTION_DRAFT_VERSION, job_id: "job-a", hydrated: true, manual: { bpm: true }, bpm: 88 });
  assert.equal(parseSelectionDraft(marked, "job-a")?.bpm, 88);
  assert.equal(parseSelectionDraft(marked, "job-b"), null);
  assert.equal(parseSelectionDraft(JSON.stringify({ bpm: 120, key: "C", meter: "4/4" }), "job-a"), null);
});

test("direct job links are validated and take priority over the saved job", () => {
  const direct = "a7428e72-c181-48cd-9953-f19247955d8e";
  const saved = "7acc1e55-86ea-445a-bb8f-59f4478a0b2a";
  assert.equal(isValidJobId(direct), true);
  assert.equal(resolveInitialJobId(`?job=${direct}`, saved).jobId, direct);
  assert.equal(resolveInitialJobId("", saved).jobId, saved);
});

test("an invalid direct job link does not fall back to an unrelated saved job", () => {
  const result = resolveInitialJobId("?job=not-a-uuid", "7acc1e55-86ea-445a-bb8f-59f4478a0b2a");
  assert.equal(result.jobId, null);
  assert.equal(result.invalidQuery, true);
  assert.equal(resolveInitialJobId("", "legacy-job-id").jobId, null);
});
