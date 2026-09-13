export type TimedPlaybackNote = {
  start_sec: number;
  end_sec: number;
};

/** SpessaSynth transfers its input buffer to the worklet; retain an owned copy. */
export function cloneSoundfontBuffer(buffer: ArrayBuffer): ArrayBuffer {
  return buffer.slice(0);
}

/** Keep a seek/pause position inside the current song timeline. */
export function clampPlaybackOffset(position: number, duration: number): number {
  const safeDuration = Number.isFinite(duration) ? Math.max(0, duration) : 0;
  const safePosition = Number.isFinite(position) ? position : 0;
  return Math.max(0, Math.min(safeDuration, safePosition));
}

/** Return the current song position from an audio-clock origin. */
export function playbackPosition(audioNow: number, audioOrigin: number, duration: number): number {
  const safeNow = Number.isFinite(audioNow) ? audioNow : audioOrigin;
  return clampPlaybackOffset(safeNow - audioOrigin, duration);
}

/** Shift notes into the timeline that starts at a pause/resume offset. */
export function slicePlaybackNotes<T extends TimedPlaybackNote>(notes: readonly T[], offset: number): T[] {
  const resumeAt = Math.max(0, Number.isFinite(offset) ? offset : 0);
  return notes
    .filter((note) => Number.isFinite(note.start_sec) && Number.isFinite(note.end_sec) && note.end_sec > resumeAt)
    .map((note) => ({
      ...note,
      start_sec: Math.max(0, note.start_sec - resumeAt),
      end_sec: Math.max(0, note.end_sec - resumeAt),
    }));
}
