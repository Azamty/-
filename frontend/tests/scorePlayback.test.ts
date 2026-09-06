import assert from "node:assert/strict";
import test from "node:test";
import { scoreToPlaybackData } from "../src/scorePlayback.ts";

test("score playback uses quantized ticks and ignores rests", () => {
  const result = scoreToPlaybackData({
    bpm: 120,
    quarter_ticks: 12,
    total_ticks: 24,
    tempo_events: [],
    voices: [{ events: [
      { start_tick: 0, duration_tick: 12, midi: 60, velocity: null },
      { start_tick: 12, duration_tick: 4, midi: null },
      { start_tick: 16, duration_tick: 8, midi: 64, velocity: 96 },
    ] }],
    metadata: { program: 40 },
  });
  assert.equal(result.duration_sec, 1);
  assert.deepEqual(result.notes, [
    { midi: 60, start_sec: 0, end_sec: 0.5, velocity: 80 },
    { midi: 64, start_sec: 2 / 3, end_sec: 1, velocity: 96 },
  ]);
  assert.equal(result.program, 40);
});

test("score playback follows tempo changes on the final score timeline", () => {
  const result = scoreToPlaybackData({
    bpm: 120,
    quarter_ticks: 12,
    total_ticks: 24,
    tempo_events: [{ start_tick: 12, bpm: 60 }],
    voices: [{ events: [
      { start_tick: 0, duration_tick: 12, midi: 60 },
      { start_tick: 12, duration_tick: 12, midi: 62 },
    ] }],
  });
  assert.equal(result.notes[0].end_sec, 0.5);
  assert.equal(result.notes[1].start_sec, 0.5);
  assert.equal(result.notes[1].end_sec, 1.5);
  assert.equal(result.duration_sec, 1.5);
});
