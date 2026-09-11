import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { WorkletSynthesizer } from "spessasynth_lib";
import { FIXTURE_JOB, FIXTURE_NOTES, FIXTURE_TRACKS } from "./fixtures/multitrack";
import { FIXTURE_VOCAL_JOB } from "./fixtures/vocal";
import { modelChoiceDisabled, modelChoiceHint } from "./modelChoice";
import { canPersistSelection, parseSelectionDraft, resolveSelectionValues, SELECTION_DRAFT_SCHEMA, SELECTION_DRAFT_VERSION, selectionStorageKey, type ManualSelectionFields } from "./selectionState";
import { classifyScoreArtifacts, filterArtifactsForSelectionRevision } from "./scoreArtifacts";
import { clampPlaybackOffset, cloneSoundfontBuffer, playbackPosition, slicePlaybackNotes } from "./synthPlayback";

type SourceKind = "instrumental" | "vocal";
type DemucsModel = "htdemucs" | "htdemucs_ft";

type Artifact = {
  artifact_id: string;
  kind: string;
  label: string;
  filename: string;
  media_type?: string;
  size_bytes: number;
  stem_id?: string | null;
  page?: number | null;
  relative_path?: string | null;
  url?: string | null;
};

type HighAccuracyMetadata = {
  notation_engine?: string;
  beat_engine?: string;
  beatnet_version?: string;
  musescore_version?: string;
  score_ticks_per_quarter?: number;
};

type TrackFailure = {
  track_id?: string;
  instrument_id?: string;
  stage?: string;
  message?: string;
  error?: string;
};

type Job = {
  id: string;
  status: string;
  phase: string;
  phase_label: string;
  progress?: number | null;
  attempt?: number;
  input?: { original_name?: string; bytes?: number | null; duration_sec?: number | null };
  error?: { code?: string; message?: string } | null;
  warnings?: readonly string[];
  artifacts?: Artifact[];
  summary?: Record<string, unknown> | null;
  v2?: {
    source_kind?: SourceKind;
    source_label?: string;
    analysis?: AnalysisSuggestion | null;
    route?: { engine?: string; use_demucs?: boolean; separation_engine?: string; separation_model?: string };
    separation?: { artifact_id?: string; model?: string; duration_sec?: number; warnings?: readonly string[] } | null;
    selection_revision?: number;
    selection?: SelectionSnapshot | null;
    score_refusal?: { code?: string; message?: string } | null;
    track_failures?: TrackFailure[];
    generation?: { stage?: string; status?: string; service_variant?: string; score_artifact_ids?: string[] };
    notation_engine?: string;
    beat_engine?: string;
    beatnet_version?: string;
    musescore_version?: string;
    score_ticks_per_quarter?: number;
  };
  retryable?: boolean;
};

type Track = {
  track_id: string;
  instrument_group: string;
  label_zh: string;
  program: number;
  is_drum: boolean;
  note_count: number;
  duration_sec: number;
  preview_available?: boolean;
};

type RollNote = {
  track_id?: string;
  instrument_group: string;
  program: number;
  is_drum: boolean;
  pitch: number;
  start_sec: number;
  end_sec: number;
};

type SelectionSnapshot = {
  revision?: number;
  selected_track_ids?: string[];
  merge_main_melody?: boolean;
  bpm_override?: number;
  key_override?: string;
  time_signature_override?: string;
  overrides?: { bpm?: number; key?: string; time_signature?: string };
};

type Capabilities = {
  engines?: Record<string, { available?: boolean; reason?: string | null }>;
  api?: { max_duration_sec?: number; retention_hours?: number };
  hardware?: { cuda?: boolean; gpu?: string };
};

type SoundfontStatus = {
  available: boolean;
  status: "ready" | "missing" | "invalid";
  filename?: string;
  size_bytes?: number | null;
  sha256?: string | null;
  expected_sha256?: string;
  source?: string;
  license?: string;
  message?: string;
  media_type?: string;
  download_url?: string;
  range_supported?: boolean;
};

type AnalysisSuggestion = {
  bpm: number;
  key: string;
  time_signature: string;
  candidates?: { bpm?: readonly number[]; key?: readonly string[]; time_signature?: readonly string[] };
  warnings?: readonly string[];
  sources?: { bpm?: string | null; key?: string | null; time_signature?: string | null };
};

const DEMUCS_MODEL_OPTIONS: Array<{
  id: DemucsModel;
  label: string;
  speed: string;
  quality: string;
}> = [
  { id: "htdemucs", label: "快速", speed: "基准速度", quality: "官方默认模型" },
  { id: "htdemucs_ft", label: "质量优先", speed: "官方说明约慢 4 倍", quality: "fine-tuned，可能略好" },
];
const demucsModelOption = (value?: string | null) => DEMUCS_MODEL_OPTIONS.find((item) => item.id === value) || DEMUCS_MODEL_OPTIONS[0];

const PHASES: Array<[string, string]> = [
  ["uploading", "上传"], ["queued", "排队"], ["probing", "检查"], ["separating", "分离"], ["vocal_ready", "试听"], ["recognizing", "识别"],
  ["selection_ready", "选择"], ["rendering", "渲染"], ["packaging", "整理"], ["completed", "完成"],
];
const LANE_COLORS = ["#f07858", "#e4b94d", "#75a99b", "#8e91d7", "#d78fba", "#79a8d1"];
const KEYS = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "Gb", "G", "Ab", "A", "Bb", "B", "Am", "Dm", "Em", "Fm", "Gm", "Bm"];
const METERS = ["2/4", "3/4", "4/4", "6/8"];

const formatBytes = (value?: number | null) => {
  if (!value) return "—";
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
};

const HIGH_ACCURACY_PRIMARY_KINDS = new Set([
  "instrument_score_midi", "vocal_score_midi", "main_melody_score_midi",
  "instrument_musicxml", "vocal_musicxml", "main_melody_musicxml",
  "instrument_alignment_report", "vocal_alignment_report", "main_melody_alignment_report",
  "instrument_performance_midi", "vocal_performance_midi", "main_melody_performance_midi",
  "high_accuracy_manifest", "vocal_raw_notes", "vocal_cleaned_notes", "vocal_cleanup_report",
  "instrument_score_json", "vocal_score_json", "main_melody_score_json",
  "instrument_jianpu_source", "vocal_jianpu_source", "main_melody_jianpu_source",
  "instrument_lilypond_source", "vocal_lilypond_source", "main_melody_lilypond_source",
]);

const HIGH_ACCURACY_SUPPORT_KINDS = new Set(["high_accuracy_log", "high_accuracy_support", "analysis_full_json"]);

const highAccuracyArtifactLabel = (artifact: Artifact) => {
  const labels: Record<string, string> = {
    instrument_score_midi: "最终简谱 MIDI",
    vocal_score_midi: "最终人声 MIDI",
    main_melody_score_midi: "主旋律 MIDI",
    instrument_musicxml: "MusicXML 分谱",
    vocal_musicxml: "MusicXML 人声分谱",
    main_melody_musicxml: "MusicXML 主旋律",
    instrument_alignment_report: "音符对齐报告",
    vocal_alignment_report: "人声音符对齐报告",
    main_melody_alignment_report: "主旋律对齐报告",
    instrument_performance_midi: "性能 MIDI（MuseScore 输入）",
    vocal_performance_midi: "人声性能 MIDI（MuseScore 输入）",
    main_melody_performance_midi: "主旋律性能 MIDI",
    high_accuracy_manifest: "高精度处理清单",
    vocal_raw_notes: "GAME 原始音符",
    vocal_cleaned_notes: "清理后人声音符",
    vocal_cleanup_report: "人声清理报告",
    instrument_score_json: "分谱数据 JSON",
    vocal_score_json: "人声分谱数据 JSON",
    main_melody_score_json: "主旋律分谱数据 JSON",
    instrument_jianpu_source: "简谱源文本",
    vocal_jianpu_source: "人声简谱源文本",
    main_melody_jianpu_source: "主旋律简谱源文本",
    instrument_lilypond_source: "LilyPond 源文本",
    vocal_lilypond_source: "人声 LilyPond 源文本",
    main_melody_lilypond_source: "主旋律 LilyPond 源文本",
  };
  return labels[artifact.kind] || artifact.label;
};

class SynthPlaybackError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "SynthPlaybackError";
    this.code = code;
  }
}

type BrowserAudioWindow = Window & { webkitAudioContext?: typeof AudioContext };

const isLocalBrowserHost = () => {
  const hostname = window.location.hostname.toLowerCase();
  return hostname === "localhost"
    || hostname === "127.0.0.1"
    || hostname === "[::1]"
    || hostname.endsWith(".localhost");
};

const getAudioContextConstructor = (): typeof AudioContext | undefined => {
  const browserWindow = window as BrowserAudioWindow;
  const nativeAudioContext = (browserWindow as unknown as { AudioContext?: typeof AudioContext }).AudioContext;
  return nativeAudioContext || browserWindow.webkitAudioContext;
};

const isSoundfontContentType = (value: string | null) => {
  if (!value) return true;
  const mediaType = value.split(";", 1)[0].trim().toLowerCase();
  return mediaType === "audio/x-soundfont-sf3"
    || mediaType === "application/octet-stream"
    || mediaType === "audio/sf2";
};

const isRiffContainer = (buffer: ArrayBuffer) => {
  if (buffer.byteLength < 12) return false;
  const header = new Uint8Array(buffer, 0, 4);
  const form = new Uint8Array(buffer, 8, 4);
  return String.fromCharCode(...header) === "RIFF"
    && ["sfbk", "sfen"].includes(String.fromCharCode(...form).toLowerCase());
};

async function readSoundfontResponse(
  response: Response,
  expectedBytes: number | null | undefined,
  onProgress: (received: number, total: number | null) => void,
): Promise<ArrayBuffer> {
  const rawLength = response.headers.get("content-length");
  const declaredBytes = rawLength ? Number.parseInt(rawLength, 10) : null;
  const total = declaredBytes && Number.isFinite(declaredBytes) ? declaredBytes : (expectedBytes || null);
  const expectedBodyBytes = expectedBytes || (declaredBytes && Number.isFinite(declaredBytes) ? declaredBytes : null);
  if (declaredBytes && expectedBytes && declaredBytes !== expectedBytes) {
    throw new SynthPlaybackError(
      "soundfont_size_mismatch",
      `音色库响应大小异常（声明 ${formatBytes(declaredBytes)}，应为约 ${formatBytes(expectedBytes)}）。隧道可能截断了大文件。`,
    );
  }

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress(buffer.byteLength, total);
    if (expectedBodyBytes && buffer.byteLength !== expectedBodyBytes) {
      throw new SynthPlaybackError(
        "soundfont_incomplete",
        `音色库只收到 ${formatBytes(buffer.byteLength)}，应为约 ${formatBytes(expectedBodyBytes)}。请检查隧道的大文件传输。`,
      );
    }
    if (!isRiffContainer(buffer)) {
      throw new SynthPlaybackError("soundfont_invalid_body", "音色库响应不是有效 SF3 文件，隧道可能返回了 HTML 登录页或错误页。");
    }
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const result = await reader.read();
    if (result.done) break;
    if (!result.value?.byteLength) continue;
    const chunk = new Uint8Array(result.value);
    chunks.push(chunk);
    received += chunk.byteLength;
    onProgress(received, total);
  }

  if (expectedBodyBytes && received !== expectedBodyBytes) {
    throw new SynthPlaybackError(
      "soundfont_incomplete",
      `音色库只收到 ${formatBytes(received)}，应为约 ${formatBytes(expectedBodyBytes)}。请检查隧道的大文件传输。`,
    );
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  chunks.forEach((chunk) => {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  });
  const buffer = bytes.buffer;
  if (!isRiffContainer(buffer)) {
    throw new SynthPlaybackError("soundfont_invalid_body", "音色库响应不是有效 SF3 文件，隧道可能返回了 HTML 登录页或错误页。");
  }
  return buffer;
}

const SOUND_FONT_CACHE_NAME = "jianpu-v2-soundfont-v2";
const SOUND_FONT_CHUNK_BYTES = 512 * 1024;
const SOUND_FONT_CHUNK_RETRIES = 2;
const SOUND_FONT_CHUNK_TIMEOUT_MS = 15_000;

type SoundfontCacheReadResult = {
  buffer: ArrayBuffer | null;
  notice?: string;
};

type SoundfontCacheWriteResult = {
  saved: boolean;
  notice: string;
};

const soundfontVersion = (status: SoundfontStatus | null) => {
  const candidate = (status?.expected_sha256 || status?.sha256 || "").trim().toLowerCase();
  return /^[a-f0-9]{64}$/.test(candidate) ? candidate : null;
};

const soundfontCacheKey = (version: string | null) => (
  version ? `/api/v2/soundfont?sha256=${version}` : null
);

const cacheFailureReason = (caught: unknown) => {
  if (caught instanceof DOMException) {
    if (caught.name === "QuotaExceededError") return "浏览器存储空间不足";
    if (caught.name === "SecurityError") return "浏览器隐私或安全策略阻止缓存";
    if (caught.name === "NotAllowedError") return "浏览器未允许缓存";
    return `浏览器缓存不可用（${caught.name || "DOMException"}）`;
  }
  return "浏览器缓存不可用";
};

const waitForRetry = (milliseconds: number, signal: AbortSignal) => new Promise<void>((resolve, reject) => {
  if (signal.aborted) {
    reject(new DOMException("Aborted", "AbortError"));
    return;
  }
  const timer = window.setTimeout(() => {
    signal.removeEventListener("abort", onAbort);
    resolve();
  }, milliseconds);
  const onAbort = () => {
    window.clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
    reject(new DOMException("Aborted", "AbortError"));
  };
  signal.addEventListener("abort", onAbort, { once: true });
});

async function fetchSoundfontRangeChunk(
  start: number,
  end: number,
  expectedTotal: number | null,
  outerSignal: AbortSignal,
  onProgress: (received: number, total: number) => void,
): Promise<{ bytes: Uint8Array; total: number }> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt <= SOUND_FONT_CHUNK_RETRIES; attempt += 1) {
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    outerSignal.addEventListener("abort", forwardAbort, { once: true });
    let timeoutId = window.setTimeout(() => controller.abort(), SOUND_FONT_CHUNK_TIMEOUT_MS);
    const refreshTimeout = () => {
      window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(() => controller.abort(), SOUND_FONT_CHUNK_TIMEOUT_MS);
    };
    try {
      const response = await fetch("/api/v2/soundfont", {
        cache: "no-store",
        headers: { Range: `bytes=${start}-${end}` },
        signal: controller.signal,
      });
      if (response.status !== 206) {
        throw new SynthPlaybackError("soundfont_range_unsupported", `隧道未返回分块响应（HTTP ${response.status}），无法稳定传输 SF3。`);
      }
      if (!isSoundfontContentType(response.headers.get("content-type"))) {
        throw new SynthPlaybackError("soundfont_content_type", "SF3 分块响应不是音频类型，隧道可能返回了 HTML 登录页。");
      }
      const contentRange = response.headers.get("content-range")?.match(/^bytes\s+(\d+)-(\d+)\/(\d+)$/);
      if (!contentRange) {
        throw new SynthPlaybackError("soundfont_range_invalid", "SF3 分块响应缺少有效 Content-Range，无法安全拼接音色库。");
      }
      const responseStart = Number(contentRange[1]);
      const responseEnd = Number(contentRange[2]);
      const total = Number(contentRange[3]);
      if (responseStart !== start || responseEnd !== end || !Number.isFinite(total) || total <= end || (expectedTotal && total !== expectedTotal)) {
        throw new SynthPlaybackError("soundfont_range_invalid", "SF3 分块范围与总大小不一致，已停止拼接以避免损坏音色库。");
      }
      const bytes: Uint8Array[] = [];
      let received = 0;
      if (response.body) {
        const reader = response.body.getReader();
        while (true) {
          const result = await reader.read();
          if (result.done) break;
          if (!result.value?.byteLength) continue;
          const chunk = new Uint8Array(result.value);
          bytes.push(chunk);
          received += chunk.byteLength;
          onProgress(start + received, total);
          refreshTimeout();
        }
      } else {
        const buffer = await response.arrayBuffer();
        const chunk = new Uint8Array(buffer);
        bytes.push(chunk);
        received = chunk.byteLength;
        onProgress(start + received, total);
        refreshTimeout();
      }
      const merged = new Uint8Array(received);
      let offset = 0;
      bytes.forEach((chunk) => {
        merged.set(chunk, offset);
        offset += chunk.byteLength;
      });
      const bytesReceived = merged;
      if (bytesReceived.byteLength !== end - start + 1) {
        throw new SynthPlaybackError("soundfont_incomplete", `SF3 分块只收到 ${formatBytes(bytesReceived.byteLength)}，预期 ${formatBytes(end - start + 1)}。`);
      }
      return { bytes: bytesReceived, total };
    } catch (caught) {
      if (outerSignal.aborted) throw caught;
      lastError = caught;
      if (attempt < SOUND_FONT_CHUNK_RETRIES) await waitForRetry(350 * (attempt + 1), outerSignal);
    } finally {
      window.clearTimeout(timeoutId);
      outerSignal.removeEventListener("abort", forwardAbort);
    }
  }
  if (lastError instanceof SynthPlaybackError) throw lastError;
  throw new SynthPlaybackError("soundfont_chunk_failed", "SF3 分块下载多次失败，已准备切换轻量试听。");
}

async function downloadSoundfontInChunks(
  expectedTotal: number | null,
  onProgress: (received: number, total: number | null) => void,
  externalSignal: AbortSignal,
): Promise<ArrayBuffer> {
  const chunks: Uint8Array[] = [];
  let received = 0;
  let total = expectedTotal;
  try {
    onProgress(0, total);
    while (total === null || received < total) {
      const end = total === null
        ? received + SOUND_FONT_CHUNK_BYTES - 1
        : Math.min(total - 1, received + SOUND_FONT_CHUNK_BYTES - 1);
      const chunk = await fetchSoundfontRangeChunk(received, end, total, externalSignal, (progressReceived, progressTotal) => {
        onProgress(progressReceived, progressTotal);
      });
      total = chunk.total;
      if (total > 128 * 1024 * 1024) {
        throw new SynthPlaybackError("soundfont_size_mismatch", "隧道报告的 SF3 大小超过安全上限，已停止下载。");
      }
      chunks.push(chunk.bytes);
      received += chunk.bytes.byteLength;
      onProgress(received, total);
    }
  } catch (caught) {
    if (externalSignal.aborted) {
      throw new SynthPlaybackError("soundfont_user_fallback", "已停止高质量音色加载，正在改用轻量试听；本次不会写入缓存。");
    }
    if (caught instanceof SynthPlaybackError) throw caught;
    throw new SynthPlaybackError("soundfont_chunk_failed", `SF3 分块传输中断（已收到 ${formatBytes(received)}），已准备切换轻量试听。`);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  chunks.forEach((chunk) => {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  });
  const buffer = bytes.buffer;
  if (!isRiffContainer(buffer)) throw new SynthPlaybackError("soundfont_invalid_body", "分块拼接后不是有效 SF3，已停止加载。");
  return buffer;
}

async function readCachedSoundfont(
  expectedBytes: number | null | undefined,
  cacheKey: string | null,
  expectedVersion: string | null,
  onProgress: (received: number, total: number | null) => void,
): Promise<SoundfontCacheReadResult> {
  if (!cacheKey || !expectedVersion) {
    return { buffer: null, notice: "音色版本校验信息不可用，已跳过浏览器缓存。" };
  }
  if (!("caches" in window) || typeof window.caches?.open !== "function") {
    return { buffer: null, notice: "当前浏览器不支持 Cache Storage，音色不会持久化。" };
  }
  try {
    const cache = await window.caches.open(SOUND_FONT_CACHE_NAME);
    const cached = await cache.match(cacheKey);
    if (!cached) return { buffer: null };
    const contentType = cached.headers.get("content-type");
    const declaredBytes = Number.parseInt(cached.headers.get("content-length") || "", 10);
    const cachedVersion = cached.headers.get("x-soundfont-sha256");
    if (
      cachedVersion !== expectedVersion
      || !contentType
      || !isSoundfontContentType(contentType)
      || !Number.isFinite(declaredBytes)
      || declaredBytes <= 0
    ) {
      await cache.delete(cacheKey).catch(() => undefined);
      return { buffer: null, notice: "浏览器缓存校验失败，已清理并重新下载。" };
    }
    try {
      const buffer = await readSoundfontResponse(cached, expectedBytes, () => undefined);
      if (
        buffer.byteLength !== declaredBytes
        || (expectedBytes !== null && expectedBytes !== undefined && buffer.byteLength !== expectedBytes)
        || !isRiffContainer(buffer)
      ) {
        throw new Error("cache_validation_failed");
      }
      onProgress(buffer.byteLength, expectedBytes || declaredBytes);
      return { buffer };
    } catch {
      await cache.delete(cacheKey).catch(() => undefined);
      return { buffer: null, notice: "浏览器缓存内容校验失败，已清理并重新下载。" };
    }
  } catch (caught) {
    return { buffer: null, notice: `浏览器缓存读取失败（${cacheFailureReason(caught)}），将重新下载。` };
  }
}

async function storeCachedSoundfont(
  buffer: ArrayBuffer,
  expectedBytes: number | null | undefined,
  cacheKey: string | null,
  expectedVersion: string | null,
): Promise<SoundfontCacheWriteResult> {
  if (!cacheKey || !expectedVersion) {
    return { saved: false, notice: "浏览器未能保存缓存，下次可能重新下载（音色版本校验信息不可用）。" };
  }
  if (!("caches" in window) || typeof window.caches?.open !== "function") {
    return { saved: false, notice: "浏览器未能保存缓存，下次可能重新下载（当前浏览器不支持 Cache Storage）。" };
  }
  try {
    const cache = await window.caches.open(SOUND_FONT_CACHE_NAME);
    await cache.put(cacheKey, new Response(buffer.slice(0), {
      headers: {
        "Content-Length": String(buffer.byteLength),
        "Content-Type": "audio/x-soundfont-sf3",
        "X-Soundfont-SHA256": expectedVersion,
      },
    }));
    const cached = await cache.match(cacheKey);
    if (!cached) {
      return { saved: false, notice: "浏览器未能保存缓存，下次可能重新下载（写入后无法回读）。" };
    }
    const contentType = cached.headers.get("content-type");
    const declaredBytes = Number.parseInt(cached.headers.get("content-length") || "", 10);
    if (
      cached.headers.get("x-soundfont-sha256") !== expectedVersion
      || !contentType
      || !isSoundfontContentType(contentType)
      || !Number.isFinite(declaredBytes)
      || declaredBytes !== buffer.byteLength
    ) {
      await cache.delete(cacheKey).catch(() => undefined);
      return { saved: false, notice: "浏览器未能保存缓存，下次可能重新下载（写入后的类型或大小校验失败）。" };
    }
    try {
      const roundTrip = await readSoundfontResponse(cached, expectedBytes, () => undefined);
      if (roundTrip.byteLength !== buffer.byteLength || !isRiffContainer(roundTrip)) {
        throw new Error("cache_validation_failed");
      }
    } catch {
      await cache.delete(cacheKey).catch(() => undefined);
      return { saved: false, notice: "浏览器未能保存缓存，下次可能重新下载（写入后的 RIFF/SF3 校验失败）。" };
    }
    return { saved: true, notice: "高质量音色已缓存，可在此浏览器和域名复用。" };
  } catch (caught) {
    return { saved: false, notice: `浏览器未能保存缓存，下次可能重新下载（${cacheFailureReason(caught)}）。` };
  }
}

const formatDuration = (value?: number | null) => {
  if (!value || !Number.isFinite(value)) return "—";
  return `${Math.floor(value / 60)}:${Math.floor(value % 60).toString().padStart(2, "0")}`;
};
const sourceLabel = (source: SourceKind) => source === "instrumental" ? "伴奏 / 纯音乐" : "人声";
const shortStatus = (job: Job | null) => {
  if (!job) return "等待音频";
  if (job.status === "vocal_ready") return "人声已分离 · 等待试听";
  if (job.status === "selection_ready") return "识别完成 · 等待选择";
  if (job.status === "completed") return "产物已就绪";
  if (job.status === "failed") return "任务失败";
  if (job.status === "interrupted") return "任务中断";
  return job.phase_label || "处理中";
};

async function readResponse(response: Response): Promise<any> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || "请求没有完成");
  }
  return payload;
}

function App() {
  const [source, setSource] = useState<SourceKind>("instrumental");
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [notes, setNotes] = useState<RollNote[]>([]);
  const [selectedTrackIds, setSelectedTrackIds] = useState<string[]>([]);
  const [rollVisible, setRollVisible] = useState(true);
  const [rollZoom, setRollZoom] = useState(1);
  const [bpm, setBpm] = useState("120");
  const [key, setKey] = useState("C");
  const [meter, setMeter] = useState("4/4");
  const [separationModel, setSeparationModel] = useState<DemucsModel>("htdemucs");
  const [analysisSuggestion, setAnalysisSuggestion] = useState<AnalysisSuggestion | null>(null);
  const [mergeMainMelody, setMergeMainMelody] = useState(false);
  const [manualSelectionFields, setManualSelectionFields] = useState<ManualSelectionFields>({});
  const [originalTime, setOriginalTime] = useState(0);
  const [synthTime, setSynthTime] = useState(0);
  const [synthPlaying, setSynthPlaying] = useState(false);
  const [synthPaused, setSynthPaused] = useState(false);
  const [synthLoading, setSynthLoading] = useState(false);
  const [soundfontDownloading, setSoundfontDownloading] = useState(false);
  const [lightweightActive, setLightweightActive] = useState(false);
  const [synthResource, setSynthResource] = useState("正在检查官方音色库…");
  const [synthError, setSynthError] = useState("");
  const [soundfontCacheNotice, setSoundfontCacheNotice] = useState("");
  const [soundfontStatus, setSoundfontStatus] = useState<SoundfontStatus | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const loadedJobRef = useRef("");
  const audioContextRef = useRef<AudioContext | null>(null);
  const synthRef = useRef<WorkletSynthesizer | null>(null);
  const lightweightNodesRef = useRef<OscillatorNode[]>([]);
  const soundfontAbortRef = useRef<AbortController | null>(null);
  const synthLoadRef = useRef<Promise<WorkletSynthesizer> | null>(null);
  const synthOriginRef = useRef<number | null>(null);
  const synthPositionRef = useRef(0);
  const synthDurationRef = useRef(0);
  const synthSessionRef = useRef(0);
  const synthBufferRef = useRef<{ version: string | null; buffer: ArrayBuffer } | null>(null);
  const synthTimerRef = useRef<number | null>(null);
  const synthStopTimerRef = useRef<number | null>(null);
  const fixtureValue = new URLSearchParams(window.location.search).get("fixture");
  const fixtureMode = fixtureValue === "multitrack" || fixtureValue === "vocal-ready";
  const vocalFixtureMode = fixtureValue === "vocal-ready";
  const isInstrumental = source === "instrumental";
  const activeVocalJob = job?.v2?.source_kind === "vocal";
  const actualDemucsModel = activeVocalJob ? demucsModelOption(job?.v2?.route?.separation_model) : demucsModelOption(separationModel);
  const isReady = job?.status === "selection_ready" || job?.status === "completed";
  const pitchedSelected = tracks.some((track) => selectedTrackIds.includes(track.track_id) && !track.is_drum);
  const selectedTracks = tracks.filter((track) => selectedTrackIds.includes(track.track_id));
  const activeSelectionRevision = isInstrumental
    ? (job?.v2?.selection_revision ?? job?.v2?.selection?.revision)
    : undefined;
  const currentArtifacts = filterArtifactsForSelectionRevision(artifacts, activeSelectionRevision);
  const instrumentScoreArtifacts = classifyScoreArtifacts(currentArtifacts, "instrument");
  const longScoreArtifacts = instrumentScoreArtifacts.long;
  const pagedScoreArtifacts = instrumentScoreArtifacts.paged;
  const vocalScoreArtifacts = classifyScoreArtifacts(currentArtifacts, "vocal");
  const vocalLongScoreArtifacts = vocalScoreArtifacts.long;
  const vocalPagedScoreArtifacts = vocalScoreArtifacts.paged;
  const originalArtifact = currentArtifacts.find((item) => item.kind === "source_audio");
  const vocalArtifact = currentArtifacts.find((item) => item.kind === "vocal_audio" || item.artifact_id === "v2-vocals-audio");
  const selectedMidi = [...currentArtifacts].reverse().find((item) => item.kind === "selected_midi");
  const selectedZip = [...currentArtifacts].reverse().find((item) => item.kind === "svg_zip");
  const originalMidi = currentArtifacts.find((item) => item.kind === "original_midi");
  const vocalMidi = currentArtifacts.find((item) => ["vocal_score_midi", "main_melody_score_midi", "midi"].includes(item.kind));
  const highAccuracyMetadata: HighAccuracyMetadata = {
    notation_engine: job?.v2?.notation_engine,
    beat_engine: job?.v2?.beat_engine,
    beatnet_version: job?.v2?.beatnet_version,
    musescore_version: job?.v2?.musescore_version,
    score_ticks_per_quarter: job?.v2?.score_ticks_per_quarter,
  };
  const primaryArtifactDownloads = currentArtifacts
    .filter((item) => HIGH_ACCURACY_PRIMARY_KINDS.has(item.kind) && item.url)
    .sort((left, right) => highAccuracyArtifactLabel(left).localeCompare(highAccuracyArtifactLabel(right)));
  const supportArtifactDownloads = currentArtifacts
    .filter((item) => HIGH_ACCURACY_SUPPORT_KINDS.has(item.kind) && item.url)
    .sort((left, right) => left.artifact_id.localeCompare(right.artifact_id));
  const trackFailures = job?.v2?.track_failures || [];

  const renderHighAccuracyMeta = () => {
    const hasMetadata = Boolean(highAccuracyMetadata.notation_engine || highAccuracyMetadata.beat_engine || highAccuracyMetadata.score_ticks_per_quarter);
    if (!hasMetadata && !trackFailures.length && !job?.v2?.score_refusal) return null;
    return <div className="high-accuracy-meta" aria-label="高精度处理信息">
      {hasMetadata && <div className="engine-readout"><span>高精度链路</span><strong>{highAccuracyMetadata.notation_engine || "—"}</strong><small>{highAccuracyMetadata.beat_engine || "—"} · BeatNet {highAccuracyMetadata.beatnet_version || "—"} · MuseScore {highAccuracyMetadata.musescore_version || "—"} · {highAccuracyMetadata.score_ticks_per_quarter || "—"} TPQ</small></div>}
      {trackFailures.length > 0 && <div className="track-failure-list" role="alert"><strong>部分乐器未完成</strong>{trackFailures.map((failure, index) => <span key={`${failure.track_id || failure.instrument_id || "track"}-${index}`}>{failure.track_id || failure.instrument_id || "未命名轨道"} · {failure.stage || "处理"} · {failure.message || failure.error || "失败"}</span>)}</div>}
      {job?.v2?.score_refusal && <div className="score-refusal" role="alert"><strong>当前没有可生成的简谱</strong><span>{job.v2.score_refusal.message || job.v2.score_refusal.code || "所有有音高轨道均未成功"}</span></div>}
    </div>;
  };

  const renderArtifactDownloads = () => {
    if (!primaryArtifactDownloads.length && !supportArtifactDownloads.length) return null;
    return <div className="artifact-download-panel">
      <div className="subhead"><div><span className="section-kicker">FILES</span><strong>高精度产物</strong></div><span className="score-warning">最终文件与诊断文件分开</span></div>
      <div className="artifact-download-grid">{primaryArtifactDownloads.map((artifact) => <a className="artifact-download-card" key={artifact.artifact_id} href={artifact.url || "#"} download={artifact.filename}><strong>{highAccuracyArtifactLabel(artifact)}</strong><small>{artifact.filename} · {formatBytes(artifact.size_bytes)}</small></a>)}</div>
      {supportArtifactDownloads.length > 0 && <details className="support-artifacts"><summary>日志与辅助 JSON（{supportArtifactDownloads.length}）</summary><div className="artifact-download-grid">{supportArtifactDownloads.map((artifact) => <a className="artifact-download-card support" key={artifact.artifact_id} href={artifact.url || "#"} download={artifact.filename}><strong>{artifact.label}</strong><small>{artifact.filename} · {formatBytes(artifact.size_bytes)}</small></a>)}</div></details>}
    </div>;
  };

  const teardownSoundfontSynth = useCallback(() => {
    const synth = synthRef.current;
    synthRef.current = null;
    if (!synth) return;
    // stopAll() only clears active voices. WorkletSynthesizer also keeps future
    // noteOn/noteOff calls in an internal queue, so destroy the old instance
    // before a pause/resume cycle to prevent stale events from resurfacing.
    try { synth.stopAll(true); } catch { /* processor may already be gone */ }
    try { synth.disconnect(); } catch { /* already disconnected */ }
    try { synth.destroy(); } catch { /* already destroyed */ }
  }, []);

  const haltSynth = useCallback((mode: "stop" | "pause", position = 0) => {
    synthSessionRef.current += 1;
    soundfontAbortRef.current?.abort();
    soundfontAbortRef.current = null;
    teardownSoundfontSynth();
    lightweightNodesRef.current.forEach((node) => {
      try { node.stop(); } catch { /* already stopped */ }
      try { node.disconnect(); } catch { /* already disconnected */ }
    });
    lightweightNodesRef.current = [];
    if (synthTimerRef.current !== null) window.cancelAnimationFrame(synthTimerRef.current);
    if (synthStopTimerRef.current !== null) window.clearTimeout(synthStopTimerRef.current);
    synthTimerRef.current = null;
    synthStopTimerRef.current = null;
    synthOriginRef.current = null;
    setSynthPlaying(false);
    if (mode === "pause") {
      const bounded = Math.max(0, Math.min(synthDurationRef.current, position));
      synthPositionRef.current = bounded;
      setSynthTime(bounded);
      setSynthPaused(true);
    } else {
      synthPositionRef.current = 0;
      synthDurationRef.current = 0;
      setSynthTime(0);
      setSynthPaused(false);
    }
  }, [teardownSoundfontSynth]);

  const stopSynth = useCallback(() => haltSynth("stop"), [haltSynth]);

  const pauseSynth = useCallback(() => {
    if (!synthPlaying) return;
    const audioNow = audioContextRef.current?.currentTime;
    const elapsed = synthOriginRef.current === null || audioNow === undefined
      ? synthPositionRef.current
      : playbackPosition(audioNow, synthOriginRef.current, synthDurationRef.current);
    haltSynth("pause", elapsed);
  }, [haltSynth, synthPlaying]);

  const loadSoundfontSynth = useCallback(async (): Promise<WorkletSynthesizer> => {
    if (synthRef.current) return synthRef.current;
    if (synthLoadRef.current) return synthLoadRef.current;
    const lifecycle = synthSessionRef.current;
    const load = (async () => {
      let receivedBytes = 0;
      let downloadTimedOut = false;
      const AudioContextConstructor = getAudioContextConstructor();
      if (!AudioContextConstructor) {
        throw new SynthPlaybackError("audio_context_unsupported", "当前浏览器没有 AudioContext，无法播放本机合成音频。请改用最新版 Chrome、Edge 或 Safari。 ");
      }

      let context: AudioContext;
      try {
        context = audioContextRef.current || new AudioContextConstructor();
      } catch {
        throw new SynthPlaybackError("audio_context_create", "浏览器无法创建音频上下文，请检查设备声音权限后重试。");
      }
      audioContextRef.current = context;
      try {
        await context.resume();
      } catch {
        throw new SynthPlaybackError("audio_context_blocked", "浏览器没有在播放按钮手势中启用音频，请再次点击播放并允许声音。");
      }
      if (context.state !== "running") {
        throw new SynthPlaybackError("audio_context_blocked", "浏览器仍将音频上下文保持为暂停，请再次点击播放并检查设备声音权限。");
      }
      if (window.isSecureContext === false && !isLocalBrowserHost()) {
        throw new SynthPlaybackError(
          "secure_context_required",
          "真实 SF3 合成需要 HTTPS：当前地址不是安全上下文，浏览器禁用了 AudioWorklet；将尝试轻量音色试听。请使用 HTTPS 隧道获得 MuseScore 音色。",
        );
      }
      if (!context.audioWorklet || typeof context.audioWorklet.addModule !== "function") {
        throw new SynthPlaybackError("audio_worklet_unsupported", "当前浏览器不支持 AudioWorklet（部分旧版 iOS/Safari 会受限），将尝试轻量音色试听。使用最新版 Chrome、Edge 或 Safari 可获得真实 SF3 音色。 ");
      }
      try {
        await context.audioWorklet.addModule("/vendor/spessasynth_processor.min.js");
      } catch {
        throw new SynthPlaybackError(
          "processor_load_failed",
          "SpessaSynth 处理器脚本加载失败，将尝试轻量音色试听。请确认 HTTPS 隧道同源转发了 /vendor/spessasynth_processor.min.js，且响应不是登录页或 HTML。",
        );
      }

      const expectedBytes = soundfontStatus?.size_bytes || null;
      const expectedVersion = soundfontVersion(soundfontStatus);
      const cacheKey = soundfontCacheKey(expectedVersion);
      setSoundfontCacheNotice("");
      const reportProgress = (received: number, total: number | null) => {
        receivedBytes = received;
        const denominator = total || expectedBytes || 0;
        const percent = denominator ? ` · ${Math.min(100, Math.round((received / denominator) * 100))}%` : "";
        setSynthResource(`正在下载官方 MuseScore General SF3 · ${formatBytes(received)}${denominator ? ` / ${formatBytes(denominator)}` : ""}${percent}`);
      };
      const memoryCandidate = expectedVersion && synthBufferRef.current?.version === expectedVersion
        ? synthBufferRef.current.buffer
        : null;
      let soundfont = memoryCandidate && memoryCandidate.byteLength > 0
        && (!expectedBytes || memoryCandidate.byteLength === expectedBytes)
        ? memoryCandidate
        : null;
      const memoryHit = Boolean(soundfont);
      if (!soundfont) {
        const cachedResult = await readCachedSoundfont(expectedBytes, cacheKey, expectedVersion, reportProgress);
        if (cachedResult.notice) setSoundfontCacheNotice(cachedResult.notice);
        soundfont = cachedResult.buffer;
      }
      if (!soundfont) {
        if (!isLocalBrowserHost()) {
          setSynthResource("正在分块下载官方 MuseScore General SF3…");
          const remoteController = new AbortController();
          soundfontAbortRef.current = remoteController;
          setSoundfontDownloading(true);
          try {
            soundfont = await downloadSoundfontInChunks(expectedBytes, reportProgress, remoteController.signal);
          } finally {
            setSoundfontDownloading(false);
            if (soundfontAbortRef.current === remoteController) soundfontAbortRef.current = null;
          }
        } else {
          setSynthResource("正在下载官方 MuseScore General SF3…");
          const timeoutMs = 180_000;
          const controller = new AbortController();
          soundfontAbortRef.current = controller;
          const timeoutId = window.setTimeout(() => {
            downloadTimedOut = true;
            controller.abort();
          }, timeoutMs);
          try {
            let response: Response;
            try {
              response = await fetch("/api/v2/soundfont", { cache: "force-cache", signal: controller.signal });
            } catch (caught) {
              if (downloadTimedOut || (caught instanceof DOMException && caught.name === "AbortError")) {
                throw new SynthPlaybackError("soundfont_timeout", `音色库下载超时（已收到 ${formatBytes(receivedBytes)}），请保持页面打开重试。`);
              }
              throw new SynthPlaybackError("soundfont_network", "音色库网络请求失败，请确认本机服务仍在线并转发 /api/v2/soundfont。");
            }
            if (!response.ok) {
              const detail = await response.json().catch(() => ({}));
              throw new SynthPlaybackError("soundfont_http", `音色库请求失败（HTTP ${response.status}）。${detail?.detail?.message || "请确认服务转发 /api/v2/soundfont。"}`);
            }
            if (!isSoundfontContentType(response.headers.get("content-type"))) {
              throw new SynthPlaybackError("soundfont_content_type", `音色库响应类型为 ${response.headers.get("content-type") || "未知"}，不是 SF3；服务可能返回了 HTML 错误页。`);
            }
            try {
              soundfont = await readSoundfontResponse(response, expectedBytes, reportProgress);
            } catch (caught) {
              if (downloadTimedOut || (caught instanceof DOMException && caught.name === "AbortError")) {
                throw new SynthPlaybackError("soundfont_timeout", `音色库下载超时（已收到 ${formatBytes(receivedBytes)}），请保持页面打开重试。`);
              }
              if (caught instanceof SynthPlaybackError) throw caught;
              throw new SynthPlaybackError("soundfont_network", `音色库传输中断（已收到 ${formatBytes(receivedBytes)}），请确认本机服务仍在线。`);
            }
          } finally {
            window.clearTimeout(timeoutId);
            if (soundfontAbortRef.current === controller) soundfontAbortRef.current = null;
          }
        }
        const cacheResult = await storeCachedSoundfont(soundfont, expectedBytes, cacheKey, expectedVersion);
        setSoundfontCacheNotice(cacheResult.notice);
      } else if (memoryHit) {
        setSoundfontCacheNotice("高质量音色已从当前页面内存缓存读取，可直接复用。");
        setSynthResource("已从当前页面内存缓存读取 MuseScore General SF3");
      } else {
        setSoundfontCacheNotice("高质量音色已从本机浏览器缓存读取，可直接复用。");
        setSynthResource("已从本机浏览器缓存读取 MuseScore General SF3");
      }
      if (soundfont && expectedVersion) {
        // SpessaSynth transfers the buffer to its AudioWorklet. Keep a private
        // copy so pause/resume can initialize a fresh synth without re-fetching.
        synthBufferRef.current = { version: expectedVersion, buffer: cloneSoundfontBuffer(soundfont) };
      }
      if (synthSessionRef.current !== lifecycle) {
        throw new SynthPlaybackError("synth_cancelled", "合成试听已停止，不再启动旧的音频实例。");
      }
      const soundfontForSynth = cloneSoundfontBuffer(soundfont);
      setSynthResource("SF3 下载完成，正在初始化 SpessaSynth…");
      let synth: WorkletSynthesizer | null = null;
      try {
        synth = new WorkletSynthesizer(context, { eventsEnabled: false });
        synth.connect(context.destination);
        await synth.soundBankManager.addSoundBank(soundfontForSynth, "MuseScore_General");
        await synth.isReady;
        if (synthSessionRef.current !== lifecycle) {
          throw new SynthPlaybackError("synth_cancelled", "合成试听已停止，不再启动旧的音频实例。");
        }
      } catch (caught) {
        if (synth) {
          try { synth.disconnect(); } catch { /* processor may be incomplete */ }
          try { synth.destroy(); } catch { /* processor may be incomplete */ }
        }
        if (caught instanceof SynthPlaybackError && caught.code === "synth_cancelled") throw caught;
        throw new SynthPlaybackError("synth_init_failed", "SF3 已下载，但浏览器初始化 SpessaSynth 失败；请尝试最新版 Chrome/Edge 或释放设备内存后重试。");
      }
      if (!synth) throw new SynthPlaybackError("synth_init_failed", "SF3 合成器没有创建成功，请重试。");
      synthRef.current = synth;
      setLightweightActive(false);
      setSoundfontStatus((current) => current ? { ...current, available: true, status: "ready" } : current);
      setSynthResource("MuseScore General · SpessaSynth / SF3");
      return synth;
    })();
    synthLoadRef.current = load;
    try {
      const result = await load;
      if (synthLoadRef.current === load) synthLoadRef.current = null;
      return result;
    } catch (caught) {
      if (synthLoadRef.current === load) synthLoadRef.current = null;
      setSynthResource("音色库不可用 · 无法合成试听");
      throw caught;
    }
  }, [soundfontStatus]);

  const loadJob = useCallback(async (jobId: string) => {
    const next = await readResponse(await fetch(`/api/v2/jobs/${jobId}`)) as Job;
    setJob(next);
    setSource(next.v2?.source_kind || "instrumental");
    setAnalysisSuggestion(next.v2?.analysis || null);
    if (next.v2?.source_kind === "vocal") {
      const model = next.v2.route?.separation_model;
      if (model === "htdemucs" || model === "htdemucs_ft") setSeparationModel(model);
    }
    if (next.status === "selection_ready" || next.status === "vocal_ready" || next.status === "completed" || next.status === "failed") {
      const artifactResult = await readResponse(await fetch(`/api/v2/jobs/${jobId}/artifacts`));
      setArtifacts(artifactResult.artifacts || []);
      if (next.v2?.source_kind === "instrumental" && loadedJobRef.current !== jobId) {
        const trackResult = await readResponse(await fetch(`/api/v2/jobs/${jobId}/tracks`));
        setTracks(trackResult.tracks || []);
        const suggestion = (trackResult.analysis || next.v2?.analysis || null) as AnalysisSuggestion | null;
        setAnalysisSuggestion(suggestion);
        const storageKey = selectionStorageKey(jobId);
        const stored = window.localStorage.getItem(storageKey);
        const saved = parseSelectionDraft(stored, jobId);
        if (stored && !saved) window.localStorage.removeItem(storageKey);
        const selection = trackResult.selection || next.v2?.selection || null;
        const selectionOverrides = selection?.overrides || {};
        const suggestionBpm = suggestion?.bpm ?? 120;
        const suggestionKey = suggestion?.key ?? "C";
        const suggestionMeter = suggestion?.time_signature ?? "4/4";
        const resolvedValues = resolveSelectionValues(
          { bpm: suggestionBpm, key: suggestionKey, time_signature: suggestionMeter },
          saved,
          {
            bpm: selectionOverrides.bpm ?? selection?.bpm_override,
            key: selectionOverrides.key ?? selection?.key_override,
            time_signature: selectionOverrides.time_signature ?? selection?.time_signature_override,
          },
        );
        const selected = selection?.selected_track_ids || saved?.selectedTrackIds || (trackResult.tracks || []).map((item: Track) => item.track_id);
        setSelectedTrackIds(selected);
        setManualSelectionFields(saved?.manual || {});
        if (saved?.rollVisible !== undefined) setRollVisible(Boolean(saved.rollVisible));
        if (selection?.merge_main_melody !== undefined) setMergeMainMelody(Boolean(selection.merge_main_melody));
        setBpm(resolvedValues.bpm);
        setKey(resolvedValues.key);
        setMeter(resolvedValues.meter);
        const recognition = (artifactResult.artifacts || []).find((item: Artifact) => item.artifact_id === "v2-recognition-json");
        if (recognition?.url) {
          const recognitionPayload = await readResponse(await fetch(recognition.url));
          const recognizedTracks = trackResult.tracks || [];
          setNotes((recognitionPayload.notes || []).map((note: RollNote) => ({
            ...note,
            track_id: note.track_id || recognizedTracks.find((track: Track) => (
              track.instrument_group === note.instrument_group
              && Number(track.program) === Number(note.program)
              && Boolean(track.is_drum) === Boolean(note.is_drum)
            ))?.track_id,
          })));
        }
        loadedJobRef.current = jobId;
      }
    }
    return next;
  }, []);

  useEffect(() => {
    fetch("/api/capabilities").then(readResponse).then(setCapabilities).catch(() => setCapabilities(null));
    fetch("/api/v2/soundfont/status").then(readResponse).then((value) => {
      const status = value as SoundfontStatus;
      setSoundfontStatus(status);
      setSynthResource(status.available ? "MuseScore General · SpessaSynth / SF3" : "首次播放时缓存官方 SF3 音色库");
    }).catch(() => setSynthResource("音色库状态不可用"));
    if (vocalFixtureMode) {
      setJob(FIXTURE_VOCAL_JOB as unknown as Job);
      setSource("vocal");
      setArtifacts(FIXTURE_VOCAL_JOB.artifacts as unknown as Artifact[]);
      setTracks([]);
      setNotes([]);
      setSeparationModel((FIXTURE_VOCAL_JOB.v2.route?.separation_model || "htdemucs") as DemucsModel);
      setAnalysisSuggestion(FIXTURE_VOCAL_JOB.v2.analysis || null);
      setBpm(String(FIXTURE_VOCAL_JOB.v2.analysis?.bpm ?? 120));
      setKey(String(FIXTURE_VOCAL_JOB.v2.analysis?.key ?? "C"));
      setMeter(String(FIXTURE_VOCAL_JOB.v2.analysis?.time_signature ?? "4/4"));
      setNotice("受控人声已分离页已载入：可验证双音频试听与下一步按钮。");
      loadedJobRef.current = FIXTURE_VOCAL_JOB.id;
      return;
    }
    if (fixtureMode) {
      setJob(FIXTURE_JOB as unknown as Job);
      setSource("instrumental");
      setTracks(FIXTURE_TRACKS as unknown as Track[]);
      setNotes(FIXTURE_NOTES as unknown as RollNote[]);
      setAnalysisSuggestion(FIXTURE_JOB.v2.analysis || null);
      setBpm(String(FIXTURE_JOB.v2.analysis?.bpm ?? 120));
      setKey(String(FIXTURE_JOB.v2.analysis?.key ?? "C"));
      setMeter(String(FIXTURE_JOB.v2.analysis?.time_signature ?? "4/4"));
      setSelectedTrackIds(FIXTURE_TRACKS.map((track) => track.track_id));
      setNotice("受控多轨数据已载入：可验证勾选、鼓组排除简谱与本机合成试听。");
      loadedJobRef.current = FIXTURE_JOB.id;
      return;
    }
    const saved = window.localStorage.getItem("jianpu-v2-job-id");
    if (saved) void loadJob(saved).catch((caught) => setError(caught instanceof Error ? caught.message : "任务状态读取失败"));
  }, [fixtureMode, loadJob, vocalFixtureMode]);

  useEffect(() => {
    if (!job || fixtureMode || ["selection_ready", "vocal_ready", "completed", "failed", "interrupted"].includes(job.status)) return undefined;
    const timer = window.setInterval(() => void loadJob(job.id).catch((caught) => setError(caught instanceof Error ? caught.message : "任务状态读取失败")), 1400);
    return () => window.clearInterval(timer);
  }, [fixtureMode, job?.id, job?.status, loadJob]);

  useEffect(() => {
    if (!job || fixtureMode || !isInstrumental || !canPersistSelection(job.status, loadedJobRef.current === job.id)) return;
    const value = {
      schema: SELECTION_DRAFT_SCHEMA,
      version: SELECTION_DRAFT_VERSION,
      job_id: job.id,
      hydrated: true,
      selectedTrackIds,
      rollVisible,
      bpm,
      key,
      meter,
      mergeMainMelody,
      manual: manualSelectionFields,
    };
    window.localStorage.setItem(selectionStorageKey(job.id), JSON.stringify(value));
  }, [bpm, fixtureMode, isInstrumental, job?.id, job?.status, key, manualSelectionFields, mergeMainMelody, meter, rollVisible, selectedTrackIds]);

  useEffect(() => () => {
    stopSynth();
  }, [stopSynth]);

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    setError(""); setNotice(""); setFile(candidate);
    if (!title) setTitle(candidate.name.replace(/\.[^/.]+$/, ""));
  };
  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => chooseFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); chooseFile(event.dataTransfer.files?.[0]); };
  const updateSelectionField = (field: keyof ManualSelectionFields, value: string) => {
    setManualSelectionFields((current) => ({ ...current, [field]: true }));
    if (field === "bpm") setBpm(value);
    if (field === "key") setKey(value);
    if (field === "meter") setMeter(value);
  };

  const submit = async () => {
    if (!file) { setError("先放入一段音频，再开始识别。"); return; }
    setBusy(true); setError(""); setNotice(""); stopSynth();
    setAnalysisSuggestion(null); setBpm("120"); setKey("C"); setMeter("4/4"); setRollVisible(true); setMergeMainMelody(false); setManualSelectionFields({});
    try {
      const form = new FormData(); form.append("file", file); form.append("source_kind", source); if (title.trim()) form.append("title", title.trim()); if (source === "vocal") form.append("separation_model", separationModel);
      const next = await readResponse(await fetch("/api/v2/jobs", { method: "POST", body: form })) as Job;
      setJob(next); setArtifacts([]); setTracks([]); setNotes([]); setSelectedTrackIds([]); loadedJobRef.current = "";
      if (source === "vocal") {
        const model = next.v2?.route?.separation_model;
        if (model === "htdemucs" || model === "htdemucs_ft") setSeparationModel(model);
      }
      window.localStorage.setItem("jianpu-v2-job-id", next.id);
      setNotice(source === "instrumental" ? "已送入 MuScriptor 全量识别；完成后再选择乐器。" : "已送入 Demucs 人声分离；完成后先试听，再生成简谱。");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "任务没有创建成功"); }
    finally { setBusy(false); }
  };

  const retry = async () => {
    if (!job || fixtureMode) return;
    setBusy(true); setError("");
    try {
      const next = await readResponse(await fetch(`/api/v2/jobs/${job.id}/retry`, { method: "POST" })) as Job;
      setJob(next); setArtifacts([]); setTracks([]); setNotes([]); loadedJobRef.current = "";
      if (next.v2?.source_kind === "vocal") {
        const model = next.v2.route?.separation_model;
        if (model === "htdemucs" || model === "htdemucs_ft") setSeparationModel(model);
      }
    } catch (caught) { setError(caught instanceof Error ? caught.message : "重试没有开始"); }
    finally { setBusy(false); }
  };

  const generateVocal = async () => {
    if (!job || fixtureMode) {
      setNotice("受控预览只演示分离完成后的下一步；真实任务会在这里排入 GAME。");
      return;
    }
    if (source !== "vocal" || job.status !== "vocal_ready") {
      setError("请先等待人声分离完成，再生成简谱。");
      return;
    }
    setBusy(true); setError(""); setNotice("正在把已分离人声送入 GAME；不会重新运行 Demucs。");
    try {
      const next = await readResponse(await fetch(`/api/v2/jobs/${job.id}/vocal/generate`, { method: "POST" })) as Job;
      setJob(next); setNotice("GAME 已排队；将只处理已分离的人声 stem。");
      window.localStorage.setItem("jianpu-v2-job-id", next.id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "人声简谱没有开始生成"); }
    finally { setBusy(false); }
  };
  const toggleTrack = (trackId: string) => {
    if (synthPlaying || synthPaused) stopSynth();
    setSelectedTrackIds((current) => current.includes(trackId) ? current.filter((id) => id !== trackId) : [...current, trackId]);
  };

  const startSynthClock = (maxEnd: number, startOffset: number, audioStartAt: number) => {
    const session = synthSessionRef.current + 1;
    synthSessionRef.current = session;
    const boundedOffset = clampPlaybackOffset(startOffset, maxEnd);
    synthDurationRef.current = maxEnd;
    synthPositionRef.current = boundedOffset;
    synthOriginRef.current = audioStartAt - boundedOffset;
    setSynthTime(boundedOffset);
    setSynthPlaying(true);
    setSynthPaused(false);
    const tick = () => {
      if (synthSessionRef.current !== session || synthOriginRef.current === null) return;
      const audioNow = audioContextRef.current?.currentTime;
      if (audioNow === undefined) return;
      const rawElapsed = audioNow - synthOriginRef.current;
      if (rawElapsed >= maxEnd + 0.15) {
        stopSynth();
        return;
      }
      const position = playbackPosition(audioNow, synthOriginRef.current, maxEnd);
      synthPositionRef.current = position;
      setSynthTime(position);
      synthTimerRef.current = window.requestAnimationFrame(tick);
    };
    synthTimerRef.current = window.requestAnimationFrame(tick);
    synthStopTimerRef.current = window.setTimeout(() => {
      if (synthSessionRef.current === session) stopSynth();
    }, Math.max(0, maxEnd - boundedOffset + 0.35) * 1000);
  };

  const playLightweightSynth = async (
    playable: RollNote[],
    fallbackReason = "真实 SF3 未在当前网络或浏览器中加载",
    startOffset = 0,
  ) => {
    const AudioContextConstructor = getAudioContextConstructor();
    if (!AudioContextConstructor) throw new SynthPlaybackError("lightweight_unsupported", "当前浏览器没有 Web Audio，无法启动轻量试听。");
    let context = audioContextRef.current;
    if (!context) {
      try {
        context = new AudioContextConstructor();
        audioContextRef.current = context;
      } catch {
        throw new SynthPlaybackError("lightweight_unsupported", "浏览器无法创建轻量试听的音频上下文，请检查设备声音权限。");
      }
    }
    try {
      await context.resume();
    } catch {
      throw new SynthPlaybackError("audio_context_blocked", "浏览器没有启用轻量试听，请再次点击播放并允许声音。");
    }
    if (context.state !== "running") throw new SynthPlaybackError("audio_context_blocked", "浏览器仍将音频上下文保持为暂停，请再次点击播放。");
    const trackById = new Map(tracks.map((track) => [track.track_id, track]));
    const maxEnd = Math.max(...playable.map((note) => note.end_sec), 0);
    const resumeAt = clampPlaybackOffset(startOffset, maxEnd);
    const scheduledNotes = slicePlaybackNotes(playable, resumeAt);
    const start = context.currentTime + 0.08;
    const waveforms: OscillatorType[] = ["sine", "triangle", "square", "sawtooth"];
    scheduledNotes.forEach((note) => {
      const track = note.track_id ? trackById.get(note.track_id) : undefined;
      if (!track) return;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const noteStart = note.start_sec;
      const noteEnd = Math.max(noteStart + 0.06, note.end_sec);
      const when = start + noteStart;
      const until = start + noteEnd;
      const frequency = track.is_drum
        ? Math.max(80, Math.min(900, 55 * Math.pow(2, (note.pitch - 36) / 12)))
        : Math.max(55, Math.min(1800, 440 * Math.pow(2, (note.pitch - 69) / 12)));
      oscillator.type = track.is_drum ? "square" : waveforms[Math.abs(track.program) % waveforms.length];
      oscillator.frequency.setValueAtTime(frequency, when);
      gain.gain.setValueAtTime(0.0001, when);
      gain.gain.exponentialRampToValueAtTime(track.is_drum ? 0.07 : 0.11, when + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, Math.max(when + 0.035, until));
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(when);
      oscillator.stop(Math.max(when + 0.05, until + 0.04));
      lightweightNodesRef.current.push(oscillator);
    });
    if (!lightweightNodesRef.current.length) throw new SynthPlaybackError("lightweight_empty", "当前选择在暂停位置后没有可播放的音符。");
    const hasDrums = playable.some((note) => note.track_id && trackById.get(note.track_id)?.is_drum);
    setLightweightActive(true);
    setSynthError(`${fallbackReason} 已改用轻量试听；音高、节奏和所选轨道仍保持。${hasDrums ? "鼓组使用电子近似音色。" : ""}`);
    setSynthResource("轻量音色 · Web Audio 波形 · 选择轨道已应用");
    startSynthClock(maxEnd, resumeAt, start);
  };

  const submitSelection = async (kind: "midi" | "score") => {
    if (!job || fixtureMode) { setNotice(kind === "midi" ? "受控预览只演示选择逻辑；真实任务会在这里生成选择版本 MIDI。" : "受控预览只演示选择逻辑；真实任务会在这里生成分谱。"); return; }
    if (!selectedTrackIds.length) { setError("至少选择一条轨道；全不选时不会生成 MIDI 或简谱。"); return; }
    setBusy(true); setError(""); setNotice(kind === "midi" ? "正在登记选择版本并生成 MIDI…" : "正在登记选择版本并生成分谱…");
    try {
      const payload = { selected_track_ids: selectedTrackIds, merge_main_melody: mergeMainMelody, bpm_override: Number(bpm) || 120, key_override: key, time_signature_override: meter };
      const next = await readResponse(await fetch(`/api/v2/jobs/${job.id}/selection/export`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })) as Job;
      setJob(next); setArtifacts([]); setNotice("选择版本已排队；识别缓存会复用，不会再次运行模型。");
      window.localStorage.setItem("jianpu-v2-job-id", next.id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "选择版本没有创建成功"); }
    finally { setBusy(false); }
  };

  const playSynth = async () => {
    if (synthPlaying) { pauseSynth(); return; }
    const playable = notes.filter((note) => note.track_id && selectedTrackIds.includes(note.track_id));
    if (!playable.length) { setError("当前没有可试听的已选轨道；鼓组也可以试听，但需先勾选。"); return; }
    if (synthLoading) return;
    const maxEnd = Math.max(...playable.map((note) => note.end_sec), 0);
    const playbackRequest = synthSessionRef.current;
    const startOffset = synthPaused && synthPositionRef.current < maxEnd
      ? clampPlaybackOffset(synthPositionRef.current, maxEnd)
      : 0;
    setError("");
    setSynthError("");
    setSynthLoading(true);
    try {
      const synth = await loadSoundfontSynth();
      if (synthSessionRef.current !== playbackRequest) return;
      const context = audioContextRef.current;
      if (!context) throw new SynthPlaybackError("audio_context_missing", "音频上下文不可用，请重新点击播放。");
      await context.resume();
      if (context.state !== "running") throw new SynthPlaybackError("audio_context_blocked", "浏览器仍将音频上下文保持为暂停，请再次点击播放并检查设备声音权限。");
      synth.stopAll(true);
      const selectedPlayableTracks = tracks.filter((track) => selectedTrackIds.includes(track.track_id) && playable.some((note) => note.track_id === track.track_id));
      const channels = new Map<string, number>();
      let nextChannel = 0;
      selectedPlayableTracks.forEach((track) => {
        if (track.is_drum) {
          channels.set(track.track_id, 9);
          synth.midiChannels[9]?.setDrums(true);
          return;
        }
        while (nextChannel === 9) nextChannel += 1;
        const channel = nextChannel % 16;
        nextChannel += 1;
        channels.set(track.track_id, channel);
        synth.midiChannels[channel]?.setDrums(false);
        synth.programChange(channel, Math.max(0, Math.min(127, track.program)));
      });
      const start = context.currentTime + 0.08;
      const scheduledNotes = slicePlaybackNotes(playable, startOffset);
      scheduledNotes.forEach((note) => {
        const channel = note.track_id ? channels.get(note.track_id) : undefined;
        if (channel === undefined) return;
        const noteStart = note.start_sec;
        const noteEnd = Math.max(noteStart + 0.04, note.end_sec);
        const when = start + noteStart;
        const until = start + noteEnd;
        synth.noteOn(channel, Math.max(0, Math.min(127, note.pitch)), 80, { time: when });
        synth.noteOff(channel, Math.max(0, Math.min(127, note.pitch)), { time: until });
      });
      setSynthResource("MuseScore General · SpessaSynth / SF3 · program/channel 已应用");
      startSynthClock(maxEnd, startOffset, start);
    } catch (caught) {
      if (synthSessionRef.current !== playbackRequest) return;
      let fallbackDiagnostic = "";
      const fallbackCodes = new Set([
        "secure_context_required",
        "audio_worklet_unsupported",
        "processor_load_failed",
        "soundfont_range_unsupported",
        "soundfont_range_invalid",
        "soundfont_chunk_failed",
        "soundfont_user_fallback",
        "soundfont_timeout",
        "soundfont_network",
        "soundfont_http",
        "soundfont_content_type",
        "soundfont_incomplete",
        "soundfont_invalid_body",
        "soundfont_size_mismatch",
        "synth_init_failed",
      ]);
      if (caught instanceof SynthPlaybackError && fallbackCodes.has(caught.code)) {
        try {
          await playLightweightSynth(playable, caught.message, startOffset);
          setError("");
          return;
        } catch (lightweightCaught) {
          const reason = lightweightCaught instanceof Error ? lightweightCaught.message : "未知原因";
          fallbackDiagnostic = `${caught.message} 轻量试听也未能启动：${reason}`;
        }
      }
      const diagnostic = caught instanceof SynthPlaybackError
        ? (fallbackDiagnostic || caught.message)
        : (caught instanceof Error ? `合成试听初始化失败：${caught.message}` : "音色库不可用，无法合成试听。");
      setSynthError(diagnostic);
      setError(diagnostic);
      setSynthResource("音色库不可用 · 无法合成试听");
    } finally {
      setSynthLoading(false);
    }
  };

  const useLightweightNow = () => {
    if (!soundfontAbortRef.current) return;
    setSynthError("正在停止高质量音色加载，准备轻量试听；本次不会写入缓存…");
    soundfontAbortRef.current.abort();
  };

  const reloadHighQuality = () => {
    stopSynth();
    setLightweightActive(false);
    setSynthError("");
    setSynthResource("正在重新加载高质量 MuseScore SF3…");
    void playSynth();
  };

  const activeTime = synthPlaying || synthPaused ? synthTime : originalTime;
  const duration = Math.max(job?.input?.duration_sec || 0, notes.reduce((max, note) => Math.max(max, note.end_sec), 0), 1);
  const visibleTracks = tracks.filter((track) => selectedTrackIds.includes(track.track_id));
  const rollWidth = Math.max(780, duration * 116 * rollZoom); const laneHeight = 84; const rollHeight = Math.max(150, visibleTracks.length * laneHeight);
  const maxPitch = Math.max(84, ...notes.map((note) => note.pitch)); const minPitch = Math.min(28, ...notes.map((note) => note.pitch));

  const renderPianoRoll = () => {
    if (!rollVisible) return <div className="roll-hidden">钢琴卷帘已隐藏。再次打开即可继续查看轨道与播放头。</div>;
    if (!visibleTracks.length) return <div className="roll-hidden">没有已选轨道；勾选乐器后这里会显示时间同步的音符。</div>;
    return <div className="roll-scroll" aria-label="时间同步钢琴卷帘"><svg className="piano-roll" width={rollWidth} height={rollHeight + 28} role="img" aria-label="按乐器颜色显示的钢琴卷帘"><rect width={rollWidth} height={rollHeight + 28} fill="#172b31" />{Array.from({ length: Math.ceil(duration) + 1 }, (_, second) => <g key={`grid-${second}`}><line x1={(second / duration) * rollWidth} y1="0" x2={(second / duration) * rollWidth} y2={rollHeight} stroke="#36535a" strokeWidth="1" /><text x={(second / duration) * rollWidth + 5} y="18" fill="#a9c1b7" fontSize="11">{second}s</text></g>)}{visibleTracks.map((track, lane) => { const color = LANE_COLORS[tracks.findIndex((item) => item.track_id === track.track_id) % LANE_COLORS.length]; return <g key={track.track_id}><rect x="0" y={lane * laneHeight} width={rollWidth} height={laneHeight} fill={lane % 2 ? "#1b3339" : "#172b31"} /><text x="10" y={lane * laneHeight + 34} fill={color} fontSize="12" fontWeight="700">{track.label_zh}</text>{notes.filter((note) => note.track_id === track.track_id).map((note, index) => { const x = (note.start_sec / duration) * rollWidth; const width = Math.max(3, ((note.end_sec - note.start_sec) / duration) * rollWidth); const y = lane * laneHeight + laneHeight - 14 - ((note.pitch - minPitch) / Math.max(1, maxPitch - minPitch)) * (laneHeight - 31); return <rect key={`${track.track_id}-${index}`} x={x} y={y} width={width} height="9" rx="3" fill={color} opacity="0.9" />; })}</g>; })}<line className="playhead" x1={(activeTime / duration) * rollWidth} y1="0" x2={(activeTime / duration) * rollWidth} y2={rollHeight} stroke="#fff5d6" strokeWidth="2" /></svg></div>;
  };

  const trackCountLabel = tracks.length ? `${tracks.length} 条轨道 · ${notes.length} 个事件` : "等待完整识别";
  const statusIndex = Math.max(0, PHASES.findIndex(([phase]) => phase === job?.status || phase === job?.phase));

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="谱面工作台首页"><span className="brand-glyph">∿</span><span><strong>谱面</strong><small>LOCAL SCORE DESK · V2</small></span></a><div className="topbar-meta"><span className="local-dot"><i />仅限本机</span><span className="version-mark">MuScriptor / GAME</span></div></header>
    <main className="workspace">
      <section className="intro-block"><div className="intro-copy"><p className="eyebrow"><span className="eyebrow-rule" /> SIGNAL → SCORE</p><h1>把每条声部<br /><em>留在可读的轨道里。</em></h1><p className="intro-text">本机识别、逐轨试听、选择后生成分谱。原音与合成试听始终分开。</p></div><div className="notation-stamp" aria-hidden="true"><span className="stamp-label">LOCAL<br />TRANSCRIPTION</span><span className="stamp-note">♩</span><span className="stamp-tick" /><span className="stamp-caption">127.0.0.1 / NO CLOUD</span></div></section>
      <section className="source-panel panel" aria-labelledby="source-heading"><div className="panel-heading"><div><p className="section-kicker">01 · SOURCE</p><h2 id="source-heading">先告诉我音频来自哪里</h2></div><span className="step-badge">V2</span></div><div className="source-cards" role="group" aria-label="音频来源"><button type="button" className={`source-card ${source === "instrumental" ? "selected" : ""}`} onClick={() => setSource("instrumental")}><span className="source-mark">M</span><span><strong>伴奏 / 纯音乐</strong><small>MuScriptor · 全量识别后再选乐器</small></span><b>{source === "instrumental" ? "已选" : ""}</b></button><button type="button" className={`source-card ${source === "vocal" ? "selected" : ""}`} onClick={() => setSource("vocal")}><span className="source-mark vocal-mark">G</span><span><strong>人声</strong><small>Demucs 分离人声 → GAME · 先试听再生成</small></span><b>{source === "vocal" ? "已选" : ""}</b></button></div></section>
      <div className="desk-grid upload-grid"><section className="control-panel panel"><div className="panel-heading"><div><p className="section-kicker">02 · INPUT</p><h2>放入一段音频</h2></div><span className="step-badge">{source === "instrumental" ? "M" : "G"}</span></div><div className={`dropzone ${file ? "has-file" : ""}`} onClick={() => inputRef.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} onDragOver={(event) => event.preventDefault()} onDrop={onDrop} role="button" tabIndex={0} aria-label="选择音频文件"><input ref={inputRef} type="file" accept=".mp3,.wav,.flac,.m4a,audio/*" onChange={onFileChange} hidden /><span className="drop-orbit">{file ? "✓" : "↑"}</span>{file ? <><strong>{file.name}</strong><small>{formatBytes(file.size)} · 点击更换</small></> : <><strong>拖入一段音频</strong><small>或点击选择 · MP3 / WAV / FLAC / M4A</small></>}</div><div className="field-stack"><label htmlFor="title">任务标题 <small>可留空，使用文件名</small></label><input id="title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：夜行片段" /></div>{source === "vocal" && <fieldset className="demucs-options" disabled={modelChoiceDisabled(busy)}><legend>人声分离模型</legend><div className="model-choice-list">{DEMUCS_MODEL_OPTIONS.map((option) => { const selected = separationModel === option.id; return <label className={`model-choice ${selected ? "selected" : ""}`} key={option.id}><input type="radio" name="separation-model" value={option.id} checked={selected} onChange={() => setSeparationModel(option.id)} /><span className="model-choice-copy"><strong>{option.label}</strong><small>{option.speed} · {option.quality}</small></span><span className="model-choice-state" aria-hidden="true">{selected ? "已选" : ""}</span></label>; })}</div><p className="model-note">{modelChoiceHint(activeVocalJob ? actualDemucsModel : null)}</p></fieldset>}{error && <div className="error-message" role="alert"><span>!</span>{error}</div>}{notice && <div className="notice-message" role="status"><span>✦</span>{notice}</div>}<button type="button" className="primary-action" onClick={() => void submit()} disabled={busy || !file}><span>{busy ? "正在送入队列…" : "开始本机识别"}</span><strong>↗</strong></button><p className="privacy-note"><span>⌁</span> 文件只保留在本机任务目录，完成后按服务策略清理。</p></section>
        <section className="status-panel panel" aria-live="polite"><div className="panel-heading"><div><p className="section-kicker">03 · QUEUE</p><h2>{shortStatus(job)}</h2></div>{job && <span className={`status-chip ${job.status}`}><i />{job.status === "completed" ? "完成" : job.status === "failed" ? "失败" : job.status === "selection_ready" ? "可选择" : job.status === "vocal_ready" ? "可生成" : "处理中"}</span>}</div>{job ? <><div className="job-meta"><span><b>{job.input?.original_name || sourceLabel(source)}</b><small>{sourceLabel(job.v2?.source_kind || source)} · 第 {job.attempt || 1} 次</small>{activeVocalJob && <small className="model-readout">Demucs · {actualDemucsModel.label} · {actualDemucsModel.id}</small>}</span><span className="job-id">{job.id.slice(0, 8)}</span></div><div className="phase-rail">{PHASES.map(([phase, label], index) => <span key={phase} className={`${index < statusIndex ? "done" : ""} ${index === statusIndex ? "active" : ""}`} title={label}><i /></span>)}</div><div className="phase-labels">{PHASES.map(([phase, label]) => <span key={phase} className={phase === job.status || phase === job.phase ? "current" : ""}>{label}</span>)}</div>{job.progress !== null && job.progress !== undefined && <div className="progress-meter"><span style={{ width: `${Math.max(2, Math.min(100, job.progress * 100))}%` }} /></div>}{(job.status === "failed" || job.status === "interrupted") && <div className="failure-state"><span className="failure-mark">×</span><div><strong>{job.status === "interrupted" ? "任务中断" : "这次识别没有完成"}</strong><p>{job.error?.message || "可以保留当前选择并重试。"}</p><button type="button" className="outline-action" onClick={() => void retry()} disabled={!job.retryable || busy}>重新排队</button></div></div>}</> : <div className="empty-state"><span className="empty-wave">∿</span><p>上传后，这里会显示识别进度与可用产物。</p><small>{capabilities?.hardware?.cuda ? `CUDA · ${capabilities.hardware.gpu || "本机 GPU"}` : "本机任务队列"}</small></div>}</section></div>
      {job?.status === "failed" && renderHighAccuracyMeta()}
      {job?.status === "failed" && (primaryArtifactDownloads.length > 0 || supportArtifactDownloads.length > 0) && <section className="failure-artifacts panel" aria-label="失败诊断产物"><div className="panel-heading"><div><p className="section-kicker">DIAGNOSTICS</p><h2>失败诊断与已保留文件</h2><p className="heading-note">本次失败前已经生成的原始文件、清单和日志仍可下载。</p></div></div>{renderArtifactDownloads()}</section>}

      {isInstrumental && isReady && <section className="instrument-workspace panel" aria-labelledby="workspace-heading" data-selection-policy="single-checkbox-controls-roll-synth-midi-score"><div className="panel-heading workspace-title"><div><p className="section-kicker">04 · SELECTION READY</p><h2 id="workspace-heading">全量识别完成，选出要留下的轨道</h2><p className="heading-note">{trackCountLabel} · 勾选变化不会重新运行 MuScriptor。</p></div><span className="selection-revision">R{job?.v2?.selection_revision || 0}</span></div><div className="listen-strip"><div className="listen-card original-listen"><span className="listen-icon">◉</span><div><strong>原曲试听</strong><small>音频文件的真实播放</small></div>{originalArtifact?.url ? <audio ref={audioRef} controls src={originalArtifact.url} onTimeUpdate={(event) => setOriginalTime(event.currentTarget.currentTime)} onEnded={() => setOriginalTime(0)} /> : <span className="listen-pending">识别后可用</span>}</div><div className="listen-card synth-listen"><span className="listen-icon">⌁</span><div><strong>本机合成试听</strong><small title={soundfontStatus?.sha256 || "官方 SF3 音色库"} aria-live="polite">{synthResource}</small></div>{soundfontCacheNotice && <small className="synth-cache-note" aria-live="polite">{soundfontCacheNotice}</small>}<button type="button" className={`listen-button ${synthPlaying ? "playing" : ""}`} onClick={() => void playSynth()} disabled={synthLoading}>{synthLoading ? "准备试听…" : synthPlaying ? "暂停" : synthPaused ? "播放" : "播放选中轨"}</button>{synthLoading && soundfontDownloading && <button type="button" className="lightweight-action" onClick={useLightweightNow}>立即使用轻量试听</button>}<span className="synth-time">{formatDuration(synthTime)} / {formatDuration(duration)}</span>{synthError && <p className="synth-diagnostic" role="alert">{synthError}</p>}{lightweightActive && !synthLoading && <button type="button" className="lightweight-action" onClick={reloadHighQuality}>重新加载高质量音色</button>}</div></div><div className="workspace-grid"><div className="roll-column"><div className="subhead"><div><span className="section-kicker">PIANO ROLL</span><strong>时间同步试听轨</strong></div><div className="roll-tools"><label><input type="checkbox" checked={rollVisible} onChange={(event) => setRollVisible(event.target.checked)} />显示</label><label>缩放 <input aria-label="钢琴卷帘缩放" type="range" min="1" max="4" step="0.5" value={rollZoom} onChange={(event) => setRollZoom(Number(event.target.value))} /></label></div></div>{renderPianoRoll()}<p className="roll-caption">播放原曲或本机合成时，米白播放头沿原始秒数移动。颜色只表示轨道，音符时间不经过简谱量化。</p></div><aside className="track-roster"><div className="subhead"><div><span className="section-kicker">TRACK ROSTER</span><strong>乐器清单</strong></div><span className="roster-count">{selectedTrackIds.length}/{tracks.length}</span></div><div className="track-list">{tracks.map((track, index) => { const color = LANE_COLORS[index % LANE_COLORS.length]; const selected = selectedTrackIds.includes(track.track_id); return <div className={`track-row ${selected ? "selected" : ""}`} key={track.track_id}><span className="track-color" style={{ background: color }} /><label className="track-check"><input type="checkbox" checked={selected} onChange={() => toggleTrack(track.track_id)} /><span><strong>{track.label_zh}</strong><small>{track.instrument_group} · {track.note_count} notes · {track.is_drum ? "不生成简谱" : `program ${track.program}`}</small></span></label></div>; })}</div>{selectedTracks.some((track) => track.is_drum) && <p className="drum-note">鼓组保留在试听与选择 MIDI 中，但不会生成简谱。</p>}{!pitchedSelected && selectedTrackIds.length > 0 && <p className="drum-note">当前只选中鼓组：可以下载 MIDI，简谱按钮会保持禁用。</p>}</aside></div><div className="export-rail"><div className="analysis-advice" role="status"><div><strong>识别建议，可修改</strong><span>{analysisSuggestion ? `${analysisSuggestion.bpm} BPM · ${analysisSuggestion.key} · ${analysisSuggestion.time_signature}` : "等待原音分析"}</span></div>{analysisSuggestion?.candidates && <small>候选：BPM {analysisSuggestion.candidates.bpm?.join(" / ") || "—"} · 调性 {analysisSuggestion.candidates.key?.join(" / ") || "—"} · 拍号 {analysisSuggestion.candidates.time_signature?.join(" / ") || "—"}</small>}{analysisSuggestion?.warnings?.map((warning, index) => <small key={`analysis-warning-${index}`}>{warning}</small>)}</div><div className="override-fields"><div><label htmlFor="export-bpm">BPM</label><input id="export-bpm" inputMode="decimal" value={bpm} onChange={(event) => updateSelectionField("bpm", event.target.value)} /></div><div><label htmlFor="export-key">调性</label><select id="export-key" value={key} onChange={(event) => updateSelectionField("key", event.target.value)}>{KEYS.map((item) => <option key={item}>{item}</option>)}</select></div><div><label htmlFor="export-meter">拍号</label><select id="export-meter" value={meter} onChange={(event) => updateSelectionField("meter", event.target.value)}>{METERS.map((item) => <option key={item}>{item}</option>)}</select></div></div><label className="merge-option"><input type="checkbox" checked={mergeMainMelody} onChange={(event) => setMergeMainMelody(event.target.checked)} />另外生成单声部主旋律 <small>会丢失和声</small></label><div className="export-actions"><button type="button" className="outline-action" onClick={() => void submitSelection("midi")} disabled={busy || selectedTrackIds.length === 0}>下载所选 MIDI</button><button type="button" className="primary-action compact" onClick={() => void submitSelection("score")} disabled={busy || !pitchedSelected}><span>生成分谱</span><strong>↗</strong></button></div></div>{renderHighAccuracyMeta()}{(longScoreArtifacts.length > 0 || pagedScoreArtifacts.length > 0) && <div className="score-deck"><div className="subhead"><div><span className="section-kicker">SCORE PAGES</span><strong>已选有音高乐器分谱</strong></div><span className="score-warning">鼓组不在分谱中</span></div><div className="score-grid" data-score-group="long">{longScoreArtifacts.map((artifact) => <a className="score-card score-long-card" key={artifact.artifact_id} href={artifact.url || "#"} target="_blank" rel="noreferrer"><span className="score-thumb"><img src={artifact.url || ""} alt={artifact.label} /></span><strong>{artifact.label}</strong><small>纵向长图 SVG · 默认预览</small></a>)}</div>{pagedScoreArtifacts.length > 0 && <details className="score-pages" data-score-group="paged"><summary>分页 SVG（{pagedScoreArtifacts.length} 页）</summary><div className="score-grid">{pagedScoreArtifacts.map((artifact) => <a className="score-card" key={artifact.artifact_id} href={artifact.url || "#"} target="_blank" rel="noreferrer"><span className="score-thumb"><img src={artifact.url || ""} alt={artifact.label} /></span><strong>{artifact.label}</strong><small>分页 SVG · 下载或新窗口查看</small></a>)}</div></details>}{longScoreArtifacts.map((artifact) => <a className="download-line" key={`${artifact.artifact_id}-download`} href={artifact.url || "#"}>↓ 下载长图 SVG · {artifact.label}</a>)}{selectedZip?.url && <a className="download-line" href={selectedZip.url}>↓ 下载全部分页 SVG ZIP</a>}{selectedMidi && <a className="download-line" href={selectedMidi.url || "#"}>↓ 下载选择版本 MIDI · R{job?.v2?.selection_revision || 0}</a>}</div>}{renderArtifactDownloads()}{originalMidi && <a className="download-line muted-line" href={originalMidi.url || "#"}>↓ 完整识别 MIDI（原始时间） · 含全部轨道</a>}</section>}

      {source === "vocal" && job?.status === "vocal_ready" && <section className="vocal-result panel" aria-labelledby="vocal-ready-heading"><div className="panel-heading"><div><p className="section-kicker">04 · VOCALS READY</p><h2 id="vocal-ready-heading">人声已经分离，可以先试听</h2><p className="heading-note">这是 Demucs 模型分离结果，可能含伴奏残留；下一步才会交给 GAME。当前模型：{actualDemucsModel.label}（{actualDemucsModel.id}）。</p></div><span className="status-chip vocal_ready"><i />可生成</span></div><div className="vocal-listen"><div><strong>原曲试听</strong>{originalArtifact?.url && <audio controls src={originalArtifact.url} />}</div><div><strong>已分离人声</strong>{vocalArtifact?.url && <audio controls src={vocalArtifact.url} />}</div></div><button type="button" className="primary-action" onClick={() => void generateVocal()} disabled={busy}><span>{busy ? "正在排队…" : "下一步 · 生成人声简谱"}</span><strong>↗</strong></button></section>}
      {source === "vocal" && job?.status === "completed" && <section className="vocal-result panel" aria-labelledby="vocal-heading"><div className="panel-heading"><div><p className="section-kicker">04 · GAME RESULT</p><h2 id="vocal-heading">人声主旋律已经展开</h2><p className="heading-note">GAME 只处理已分离的人声，结果保留为单一主旋律谱面。分离模型：{actualDemucsModel.label}（{actualDemucsModel.id}）。</p></div><span className="status-chip completed"><i />完成</span></div>{renderHighAccuracyMeta()}<div className="vocal-listen"><div><strong>原曲试听</strong>{originalArtifact?.url && <audio controls src={originalArtifact.url} />}</div><div><strong>已分离人声</strong>{vocalArtifact?.url && <audio controls src={vocalArtifact.url} />}</div></div>{vocalLongScoreArtifacts.length > 0 && <div className="score-grid" data-score-group="long">{vocalLongScoreArtifacts.map((artifact) => <a className="score-card score-long-card" key={artifact.artifact_id} href={artifact.url || "#"} target="_blank" rel="noreferrer"><span className="score-thumb"><img src={artifact.url || ""} alt={artifact.label} /></span><strong>{artifact.label}</strong><small>纵向长图 SVG · 默认预览</small></a>)}</div>}{vocalPagedScoreArtifacts.length > 0 && <details className="score-pages" data-score-group="paged"><summary>分页 SVG（{vocalPagedScoreArtifacts.length} 页）</summary><div className="score-grid">{vocalPagedScoreArtifacts.map((artifact) => <a className="score-card" key={artifact.artifact_id} href={artifact.url || "#"} target="_blank" rel="noreferrer"><span className="score-thumb"><img src={artifact.url || ""} alt={artifact.label} /></span><strong>{artifact.label}</strong><small>分页 SVG · 下载或新窗口查看</small></a>)}</div></details>}<div className="download-stack">{vocalLongScoreArtifacts.map((artifact) => <a className="download-line" key={`${artifact.artifact_id}-download`} href={artifact.url || "#"} download={artifact.filename}>↓ 下载长图 SVG · {artifact.label}</a>)}</div>{renderArtifactDownloads()}</section>}      {!fixtureMode && job && (job.status === "selection_ready" || job.status === "vocal_ready" || job.status === "completed") && <p className="retention-note">任务 {job.id} · 人声分离结果与生成阶段可在刷新后从本机任务目录恢复。</p>}
    </main><footer className="footer"><span>谱面工作台 / V2</span><span>{capabilities?.api?.retention_hours ? `本机保留 ${capabilities.api.retention_hours} 小时` : "LOCAL ONLY"}</span></footer>
  </div>;
}

export default App;
