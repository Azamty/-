export const FIXTURE_JOB = {
  id: "fixture-multitrack",
  status: "selection_ready",
  phase: "selection_ready",
  phase_label: "识别完成，等待选择",
  progress: 1,
  attempt: 1,
  input: { original_name: "controlled-mix.wav", duration_sec: 6 },
  warnings: ["受控 UI fixture：不调用模型。"],
  v2: { source_kind: "instrumental", source_label: "伴奏/纯音乐", selection_revision: 0, selection: null },
} as const;

export const FIXTURE_TRACKS = [
  { track_id: "fixture-piano", instrument_group: "acoustic_piano", label_zh: "原声钢琴", program: 0, is_drum: false, note_count: 4, duration_sec: 5.2, preview_available: true },
  { track_id: "fixture-violin", instrument_group: "violin", label_zh: "小提琴", program: 40, is_drum: false, note_count: 4, duration_sec: 5.7, preview_available: true },
  { track_id: "fixture-drums", instrument_group: "drums", label_zh: "鼓组", program: 0, is_drum: true, note_count: 5, duration_sec: 5.9, preview_available: true },
] as const;

export const FIXTURE_NOTES = [
  { track_id: "fixture-piano", instrument_group: "acoustic_piano", program: 0, is_drum: false, pitch: 60, start_sec: 0.2, end_sec: 1.1 },
  { track_id: "fixture-piano", instrument_group: "acoustic_piano", program: 0, is_drum: false, pitch: 64, start_sec: 1.4, end_sec: 2.2 },
  { track_id: "fixture-piano", instrument_group: "acoustic_piano", program: 0, is_drum: false, pitch: 67, start_sec: 2.8, end_sec: 3.7 },
  { track_id: "fixture-piano", instrument_group: "acoustic_piano", program: 0, is_drum: false, pitch: 72, start_sec: 4.1, end_sec: 5.2 },
  { track_id: "fixture-violin", instrument_group: "violin", program: 40, is_drum: false, pitch: 72, start_sec: 0.5, end_sec: 1.6 },
  { track_id: "fixture-violin", instrument_group: "violin", program: 40, is_drum: false, pitch: 76, start_sec: 1.8, end_sec: 3.0 },
  { track_id: "fixture-violin", instrument_group: "violin", program: 40, is_drum: false, pitch: 79, start_sec: 3.2, end_sec: 4.3 },
  { track_id: "fixture-violin", instrument_group: "violin", program: 40, is_drum: false, pitch: 81, start_sec: 4.5, end_sec: 5.7 },
  { track_id: "fixture-drums", instrument_group: "drums", program: 0, is_drum: true, pitch: 36, start_sec: 0.0, end_sec: 0.08 },
  { track_id: "fixture-drums", instrument_group: "drums", program: 0, is_drum: true, pitch: 42, start_sec: 1.0, end_sec: 1.08 },
  { track_id: "fixture-drums", instrument_group: "drums", program: 0, is_drum: true, pitch: 36, start_sec: 2.0, end_sec: 2.08 },
  { track_id: "fixture-drums", instrument_group: "drums", program: 0, is_drum: true, pitch: 42, start_sec: 3.0, end_sec: 3.08 },
  { track_id: "fixture-drums", instrument_group: "drums", program: 0, is_drum: true, pitch: 36, start_sec: 4.0, end_sec: 4.08 },
] as const;
