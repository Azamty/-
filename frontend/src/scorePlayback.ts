export type ScorePlaybackNote = {
  midi: number;
  start_sec: number;
  end_sec: number;
  velocity: number;
};

export type ScorePlaybackData = {
  notes: ScorePlaybackNote[];
  duration_sec: number;
  program: number;
};

type ScoreTempo = {
  start_tick?: unknown;
  bpm?: unknown;
};

type ScoreEvent = {
  start_tick?: unknown;
  duration_tick?: unknown;
  midi?: unknown;
  velocity?: unknown;
  metadata?: Record<string, unknown>;
};

type ScoreVoice = {
  events?: unknown;
};

export type ScoreJson = {
  bpm?: unknown;
  quarter_ticks?: unknown;
  total_ticks?: unknown;
  tempo_events?: unknown;
  voices?: unknown;
  metadata?: Record<string, unknown>;
};

const positiveNumber = (value: unknown, fallback: number) => {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
};

const nonNegativeInteger = (value: unknown) => {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) && number >= 0 ? Math.round(number) : null;
};

const clampMidi = (value: unknown) => {
  if (value === null || value === undefined || value === "") return null;
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) return null;
  const midi = Math.round(number);
  return midi >= 0 && midi <= 127 ? midi : null;
};

const clampVelocity = (value: unknown) => {
  if (value === null || value === undefined || value === "") return 80;
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) return 80;
  return Math.max(1, Math.min(127, Math.round(number)));
};

/**
 * Convert the final quantized Score timeline into seconds for browser playback.
 * This deliberately reads Score events rather than the raw model note events.
 */
export function scoreToPlaybackData(input: ScoreJson): ScorePlaybackData {
  const quarterTicks = positiveNumber(input.quarter_ticks, 12);
  const baseBpm = positiveNumber(input.bpm, 120);
  const totalTicks = nonNegativeInteger(input.total_ticks);
  if (totalTicks === null || totalTicks <= 0) {
    throw new Error("简谱数据缺少有效的 total_ticks，无法播放。 ");
  }

  const tempoEvents = (Array.isArray(input.tempo_events) ? input.tempo_events : [])
    .map((item) => item as ScoreTempo)
    .map((item) => {
      const startTick = nonNegativeInteger(item.start_tick);
      const bpm = positiveNumber(item.bpm, 0);
      return startTick === null || bpm <= 0 ? null : { start_tick: startTick, bpm };
    })
    .filter((item): item is { start_tick: number; bpm: number } => item !== null)
    .filter((item) => item.start_tick <= totalTicks)
    .sort((left, right) => left.start_tick - right.start_tick);

  const tickToSeconds = (targetTick: number) => {
    const target = Math.max(0, Math.min(totalTicks, targetTick));
    let cursor = 0;
    let bpm = baseBpm;
    let seconds = 0;
    for (const tempo of tempoEvents) {
      if (tempo.start_tick <= cursor) {
        bpm = tempo.bpm;
        continue;
      }
      if (tempo.start_tick >= target) break;
      seconds += ((tempo.start_tick - cursor) / quarterTicks) * (60 / bpm);
      cursor = tempo.start_tick;
      bpm = tempo.bpm;
    }
    if (target > cursor) seconds += ((target - cursor) / quarterTicks) * (60 / bpm);
    return seconds;
  };

  const voices = Array.isArray(input.voices) ? input.voices : [];
  const notes: ScorePlaybackNote[] = [];
  voices.forEach((voice) => {
    const events = (voice as ScoreVoice)?.events;
    if (!Array.isArray(events)) return;
    events.forEach((event) => {
      const item = event as ScoreEvent;
      const startTick = nonNegativeInteger(item.start_tick);
      const durationTick = nonNegativeInteger(item.duration_tick);
      const midi = clampMidi(item.midi);
      if (startTick === null || durationTick === null || durationTick <= 0 || midi === null) return;
      const endTick = Math.min(totalTicks, startTick + durationTick);
      if (endTick <= startTick) return;
      notes.push({
        midi,
        start_sec: tickToSeconds(startTick),
        end_sec: tickToSeconds(endTick),
        velocity: clampVelocity(item.velocity),
      });
    });
  });
  notes.sort((left, right) => left.start_sec - right.start_sec || left.midi - right.midi);

  const metadataProgram = input.metadata?.program;
  const programNumber = typeof metadataProgram === "number" ? metadataProgram : Number(metadataProgram);
  const program = Number.isFinite(programNumber) ? Math.max(0, Math.min(127, Math.round(programNumber))) : 0;
  return {
    notes,
    duration_sec: Math.max(0.01, tickToSeconds(totalTicks)),
    program,
  };
}
