export const FIXTURE_VOCAL_JOB = {
  id: "fixture-vocal-ready",
  status: "vocal_ready",
  phase: "vocal_ready",
  phase_label: "人声已分离，等待试听",
  progress: 1,
  attempt: 1,
  input: { original_name: "vocal-mix.wav", duration_sec: 6 },
  warnings: ["这是受控页面数据：模型分离结果可能含伴奏残留。"],
  artifacts: [
    { artifact_id: "v2-source-audio", kind: "source_audio", label: "原始人声音频", filename: "input.wav", relative_path: "input.wav", media_type: "audio/wav", size_bytes: 1200, url: "#" },
    { artifact_id: "v2-vocals-audio", kind: "vocal_audio", label: "Demucs 分离人声（试听）", filename: "vocals.wav", relative_path: "output/vocal-prep/vocals.wav", media_type: "audio/wav", size_bytes: 980, stem_id: "vocals", url: "#" },
  ],
  v2: {
    source_kind: "vocal",
    source_label: "人声",
    route: { engine: "game", use_demucs: true, separation_engine: "demucs", separation_model: "htdemucs" },
    analysis: {
      bpm: 92,
      key: "Am",
      time_signature: "4/4",
      candidates: { bpm: [92, 46, 184], key: ["Am", "C"], time_signature: ["4/4", "3/4", "6/8"] },
      warnings: ["自动拍号识别尚未启用，暂按 4/4；生成后请确认"],
      sources: { bpm: "fixture", key: "fixture", time_signature: "fallback" },
    },
    separation: { artifact_id: "v2-vocals-audio", duration_sec: 5.98, warnings: ["这是模型分离结果，可能含伴奏残留。"] },
  },
} as const;
