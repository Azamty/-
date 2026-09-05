import assert from "node:assert/strict";
import test from "node:test";
import { clampPlaybackOffset, cloneSoundfontBuffer, playbackPosition, slicePlaybackNotes } from "../src/synthPlayback.ts";

test("cloneSoundfontBuffer remains independent after a worklet transfer", () => {
  const original = new Uint8Array([82, 73, 70, 70]);
  const copy = new Uint8Array(cloneSoundfontBuffer(original.buffer));
  copy[0] = 0;
  assert.equal(original[0], 82);
  assert.notStrictEqual(copy.buffer, original.buffer);
});

test("clampPlaybackOffset keeps invalid and out of range positions safe", () => {
  assert.equal(clampPlaybackOffset(-2, 8), 0);
  assert.equal(clampPlaybackOffset(12, 8), 8);
  assert.equal(clampPlaybackOffset(Number.NaN, 8), 0);
  assert.equal(clampPlaybackOffset(2.5, 8), 2.5);
});

test("playbackPosition follows the audio clock without wall clock drift", () => {
  assert.ok(Math.abs(playbackPosition(13.4, 10, 20) - 3.4) < 1e-9);
  assert.equal(playbackPosition(4, 10, 20), 0);
  assert.equal(playbackPosition(40, 10, 20), 20);
});

test("slicePlaybackNotes shifts only notes after the resume point", () => {
  const notes = [
    { id: "before", start_sec: 0, end_sec: 1, track_id: "piano" },
    { id: "crossing", start_sec: 1.5, end_sec: 3, track_id: "piano" },
    { id: "after", start_sec: 4, end_sec: 5, track_id: "drums" },
  ];
  assert.deepEqual(slicePlaybackNotes(notes, 2), [
    { id: "crossing", start_sec: 0, end_sec: 1, track_id: "piano" },
    { id: "after", start_sec: 2, end_sec: 3, track_id: "drums" },
  ]);
});
