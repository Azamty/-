import { ChangeEvent, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { WorkletSynthesizer } from "spessasynth_lib";
import { FIXTURE_JOB, FIXTURE_NOTES, FIXTURE_TRACKS } from "./fixtures/multitrack";

type SourceKind = "instrumental" | "vocal";

type Artifact = {
  artifact_id: string;
  kind: string;
  label: string;
  filename: string;
  media_type?: string;
  size_bytes: number;
  stem_id?: string | null;
  page?: number | null;
  url?: string | null;
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
  warnings?: string[];
  artifacts?: Artifact[];
  summary?: Record<string, unknown> | null;
  v2?: {
    source_kind?: SourceKind;
    source_label?: string;
    selection_revision?: number;
    selection?: SelectionSnapshot | null;
    score_refusal?: { code?: string; message?: string } | null;
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
};

const PHASES: Array<[string, string]> = [
  ["uploading", "上传"], ["queued", "排队"], ["probing", "检查"], ["recognizing", "识别"],
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
const formatDuration = (value?: number | null) => {
  if (!value || !Number.isFinite(value)) return "—";
  return `${Math.floor(value / 60)}:${Math.floor(value % 60).toString().padStart(2, "0")}`;
};
const sourceLabel = (source: SourceKind) => source === "instrumental" ? "伴奏 / 纯音乐" : "人声";
const shortStatus = (job: Job | null) => {
  if (!job) return "等待音频";
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
  const [mutedTrackIds, setMutedTrackIds] = useState<string[]>([]);
  const [rollVisible, setRollVisible] = useState(true);
  const [rollZoom, setRollZoom] = useState(1);
  const [bpm, setBpm] = useState("120");
  const [key, setKey] = useState("C");
  const [meter, setMeter] = useState("4/4");
  const [mergeMainMelody, setMergeMainMelody] = useState(false);
  const [originalTime, setOriginalTime] = useState(0);
  const [synthTime, setSynthTime] = useState(0);
  const [synthPlaying, setSynthPlaying] = useState(false);
  const [synthResource, setSynthResource] = useState("正在检查官方音色库…");
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
  const synthLoadRef = useRef<Promise<WorkletSynthesizer> | null>(null);
  const synthOriginRef = useRef<number | null>(null);
  const synthTimerRef = useRef<number | null>(null);
  const synthStopTimerRef = useRef<number | null>(null);
  const fixtureMode = new URLSearchParams(window.location.search).get("fixture") === "multitrack";
  const isInstrumental = source === "instrumental";
  const isReady = job?.status === "selection_ready" || job?.status === "completed";
  const pitchedSelected = tracks.some((track) => selectedTrackIds.includes(track.track_id) && !track.is_drum);
  const selectedTracks = tracks.filter((track) => selectedTrackIds.includes(track.track_id));
  const scoreArtifacts = artifacts.filter((item) => item.kind === "instrument_score_svg" || item.kind === "main_melody_svg");
  const originalArtifact = artifacts.find((item) => item.kind === "source_audio");
  const selectedMidi = [...artifacts].reverse().find((item) => item.kind === "selected_midi");
  const originalMidi = artifacts.find((item) => item.kind === "original_midi");
  const vocalMidi = artifacts.find((item) => item.kind === "midi");

  const stopSynth = useCallback(() => {
    synthRef.current?.stopAll(true);
    if (synthTimerRef.current !== null) window.cancelAnimationFrame(synthTimerRef.current);
    if (synthStopTimerRef.current !== null) window.clearTimeout(synthStopTimerRef.current);
    synthTimerRef.current = null;
    synthStopTimerRef.current = null;
    synthOriginRef.current = null;
    setSynthPlaying(false);
  }, []);

  const loadSoundfontSynth = useCallback(async (): Promise<WorkletSynthesizer> => {
    if (synthRef.current) return synthRef.current;
    if (synthLoadRef.current) return synthLoadRef.current;
    const load = (async () => {
      setSynthResource("首次缓存官方 MuseScore General SF3…");
      const context = audioContextRef.current || new AudioContext();
      audioContextRef.current = context;
      await context.resume();
      await context.audioWorklet.addModule("/vendor/spessasynth_processor.min.js");
      const response = await fetch("/api/v2/soundfont");
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail?.detail?.message || "音色库不可用");
      }
      const soundfont = await response.arrayBuffer();
      const synth = new WorkletSynthesizer(context, { eventsEnabled: false });
      synth.connect(context.destination);
      await synth.soundBankManager.addSoundBank(soundfont, "MuseScore_General");
      await synth.isReady;
      synthRef.current = synth;
      setSoundfontStatus((current) => current ? { ...current, available: true, status: "ready" } : current);
      setSynthResource("MuseScore General · SpessaSynth / SF3");
      return synth;
    })();
    synthLoadRef.current = load;
    try {
      return await load;
    } catch (caught) {
      synthLoadRef.current = null;
      setSynthResource("音色库不可用 · 无法合成试听");
      throw caught;
    }
  }, []);

  const loadJob = useCallback(async (jobId: string) => {
    const next = await readResponse(await fetch(`/api/v2/jobs/${jobId}`)) as Job;
    setJob(next);
    setSource(next.v2?.source_kind || "instrumental");
    if (next.status === "selection_ready" || next.status === "completed") {
      const artifactResult = await readResponse(await fetch(`/api/v2/jobs/${jobId}/artifacts`));
      setArtifacts(artifactResult.artifacts || []);
      if (next.v2?.source_kind === "instrumental" && loadedJobRef.current !== jobId) {
        const trackResult = await readResponse(await fetch(`/api/v2/jobs/${jobId}/tracks`));
        setTracks(trackResult.tracks || []);
        const stored = window.localStorage.getItem(`jianpu-v2-selection:${jobId}`);
        const saved = stored ? JSON.parse(stored) : null;
        const selected = trackResult.selection?.selected_track_ids || saved?.selectedTrackIds || (trackResult.tracks || []).map((item: Track) => item.track_id);
        setSelectedTrackIds(selected);
        if (saved?.mutedTrackIds) setMutedTrackIds(saved.mutedTrackIds);
        if (saved?.rollVisible !== undefined) setRollVisible(Boolean(saved.rollVisible));
        if (saved?.bpm) setBpm(String(saved.bpm));
        if (saved?.key) setKey(String(saved.key));
        if (saved?.meter) setMeter(String(saved.meter));
        if (trackResult.selection?.merge_main_melody !== undefined) setMergeMainMelody(Boolean(trackResult.selection.merge_main_melody));
        const selectionOverrides = trackResult.selection?.overrides || {};
        if (selectionOverrides.bpm || trackResult.selection?.bpm_override) setBpm(String(selectionOverrides.bpm || trackResult.selection?.bpm_override));
        if (selectionOverrides.key || trackResult.selection?.key_override) setKey(String(selectionOverrides.key || trackResult.selection?.key_override));
        if (selectionOverrides.time_signature || trackResult.selection?.time_signature_override) setMeter(String(selectionOverrides.time_signature || trackResult.selection?.time_signature_override));
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
    if (fixtureMode) {
      setJob(FIXTURE_JOB as unknown as Job);
      setSource("instrumental");
      setTracks(FIXTURE_TRACKS as unknown as Track[]);
      setNotes(FIXTURE_NOTES as unknown as RollNote[]);
      setSelectedTrackIds(FIXTURE_TRACKS.map((track) => track.track_id));
      setNotice("受控多轨数据已载入：可验证勾选、鼓组排除简谱与本机合成试听。");
      loadedJobRef.current = FIXTURE_JOB.id;
      return;
    }
    const saved = window.localStorage.getItem("jianpu-v2-job-id");
    if (saved) void loadJob(saved).catch((caught) => setError(caught instanceof Error ? caught.message : "任务状态读取失败"));
  }, [fixtureMode, loadJob]);

  useEffect(() => {
    if (!job || fixtureMode || ["selection_ready", "completed", "failed", "interrupted"].includes(job.status)) return undefined;
    const timer = window.setInterval(() => void loadJob(job.id).catch((caught) => setError(caught instanceof Error ? caught.message : "任务状态读取失败")), 1400);
    return () => window.clearInterval(timer);
  }, [fixtureMode, job?.id, job?.status, loadJob]);

  useEffect(() => {
    if (!job || !isInstrumental) return;
    const value = { selectedTrackIds, mutedTrackIds, rollVisible, bpm, key, meter, mergeMainMelody };
    window.localStorage.setItem(`jianpu-v2-selection:${job.id}`, JSON.stringify(value));
  }, [bpm, isInstrumental, job?.id, key, mergeMainMelody, meter, mutedTrackIds, rollVisible, selectedTrackIds]);

  useEffect(() => () => {
    stopSynth();
    synthRef.current?.destroy();
    synthRef.current = null;
  }, [stopSynth]);

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    setError(""); setNotice(""); setFile(candidate);
    if (!title) setTitle(candidate.name.replace(/\.[^/.]+$/, ""));
  };
  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => chooseFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); chooseFile(event.dataTransfer.files?.[0]); };

  const submit = async () => {
    if (!file) { setError("先放入一段音频，再开始识别。"); return; }
    setBusy(true); setError(""); setNotice(""); stopSynth();
    try {
      const form = new FormData(); form.append("file", file); form.append("source_kind", source); if (title.trim()) form.append("title", title.trim());
      const next = await readResponse(await fetch("/api/v2/jobs", { method: "POST", body: form })) as Job;
      setJob(next); setArtifacts([]); setTracks([]); setNotes([]); setSelectedTrackIds([]); loadedJobRef.current = "";
      window.localStorage.setItem("jianpu-v2-job-id", next.id);
      setNotice(source === "instrumental" ? "已送入 MuScriptor 全量识别；完成后再选择乐器。" : "已送入 GAME 人声识别；完成后直接查看主旋律谱面。");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "任务没有创建成功"); }
    finally { setBusy(false); }
  };

  const retry = async () => {
    if (!job || fixtureMode) return;
    setBusy(true); setError("");
    try {
      const next = await readResponse(await fetch(`/api/v2/jobs/${job.id}/retry`, { method: "POST" })) as Job;
      setJob(next); setArtifacts([]); setTracks([]); setNotes([]); loadedJobRef.current = "";
    } catch (caught) { setError(caught instanceof Error ? caught.message : "重试没有开始"); }
    finally { setBusy(false); }
  };
  const toggleTrack = (trackId: string) => setSelectedTrackIds((current) => current.includes(trackId) ? current.filter((id) => id !== trackId) : [...current, trackId]);
  const toggleMute = (trackId: string) => setMutedTrackIds((current) => current.includes(trackId) ? current.filter((id) => id !== trackId) : [...current, trackId]);

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
    if (synthPlaying) { stopSynth(); return; }
    const playable = notes.filter((note) => note.track_id && selectedTrackIds.includes(note.track_id) && !mutedTrackIds.includes(note.track_id));
    if (!playable.length) { setError("当前没有可试听的已选轨道；鼓组也可以试听，但需先勾选。"); return; }
    setError("");
    try {
      const synth = await loadSoundfontSynth();
      const context = audioContextRef.current;
      if (!context) throw new Error("音频上下文不可用");
      await context.resume();
      synth.stopAll(true);
      const selectedPlayableTracks = tracks.filter((track) => selectedTrackIds.includes(track.track_id) && !mutedTrackIds.includes(track.track_id) && playable.some((note) => note.track_id === track.track_id));
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
      const maxEnd = Math.max(...playable.map((note) => note.end_sec), 0);
      playable.forEach((note) => {
        const channel = note.track_id ? channels.get(note.track_id) : undefined;
        if (channel === undefined) return;
        const when = start + Math.max(0, note.start_sec);
        const until = start + Math.max(note.start_sec + 0.04, note.end_sec);
        synth.noteOn(channel, Math.max(0, Math.min(127, note.pitch)), 80, { time: when });
        synth.noteOff(channel, Math.max(0, Math.min(127, note.pitch)), { time: until });
      });
      synthOriginRef.current = performance.now();
      setSynthPlaying(true);
      setSynthResource("MuseScore General · SpessaSynth / SF3 · program/channel 已应用");
      const tick = () => {
        const elapsed = synthOriginRef.current === null ? 0 : (performance.now() - synthOriginRef.current) / 1000;
        setSynthTime(Math.min(maxEnd, elapsed));
        if (elapsed < maxEnd + 0.15) synthTimerRef.current = window.requestAnimationFrame(tick);
        else stopSynth();
      };
      synthTimerRef.current = window.requestAnimationFrame(tick);
      synthStopTimerRef.current = window.setTimeout(stopSynth, (maxEnd + 0.35) * 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "音色库不可用，无法合成试听。");
      setSynthResource("音色库不可用 · 无法合成试听");
    }
  };

  const activeTime = synthPlaying ? synthTime : originalTime;
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
      <section className="source-panel panel" aria-labelledby="source-heading"><div className="panel-heading"><div><p className="section-kicker">01 · SOURCE</p><h2 id="source-heading">先告诉我音频来自哪里</h2></div><span className="step-badge">V2</span></div><div className="source-cards" role="group" aria-label="音频来源"><button type="button" className={`source-card ${source === "instrumental" ? "selected" : ""}`} onClick={() => setSource("instrumental")}><span className="source-mark">M</span><span><strong>伴奏 / 纯音乐</strong><small>MuScriptor · 全量识别后再选乐器</small></span><b>{source === "instrumental" ? "已选" : ""}</b></button><button type="button" className={`source-card ${source === "vocal" ? "selected" : ""}`} onClick={() => setSource("vocal")}><span className="source-mark vocal-mark">G</span><span><strong>人声</strong><small>GAME · 识别完成后直接给出主旋律</small></span><b>{source === "vocal" ? "已选" : ""}</b></button></div></section>
      <div className="desk-grid upload-grid"><section className="control-panel panel"><div className="panel-heading"><div><p className="section-kicker">02 · INPUT</p><h2>放入一段音频</h2></div><span className="step-badge">{source === "instrumental" ? "M" : "G"}</span></div><div className={`dropzone ${file ? "has-file" : ""}`} onClick={() => inputRef.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }} onDragOver={(event) => event.preventDefault()} onDrop={onDrop} role="button" tabIndex={0} aria-label="选择音频文件"><input ref={inputRef} type="file" accept=".mp3,.wav,.flac,.m4a,audio/*" onChange={onFileChange} hidden /><span className="drop-orbit">{file ? "✓" : "↑"}</span>{file ? <><strong>{file.name}</strong><small>{formatBytes(file.size)} · 点击更换</small></> : <><strong>拖入一段音频</strong><small>或点击选择 · MP3 / WAV / FLAC / M4A</small></>}</div><div className="field-stack"><label htmlFor="title">任务标题 <small>可留空，使用文件名</small></label><input id="title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：夜行片段" /></div>{error && <div className="error-message" role="alert"><span>!</span>{error}</div>}{notice && <div className="notice-message" role="status"><span>✦</span>{notice}</div>}<button type="button" className="primary-action" onClick={() => void submit()} disabled={busy || !file}><span>{busy ? "正在送入队列…" : "开始本机识别"}</span><strong>↗</strong></button><p className="privacy-note"><span>⌁</span> 文件只保留在本机任务目录，完成后按服务策略清理。</p></section>
        <section className="status-panel panel" aria-live="polite"><div className="panel-heading"><div><p className="section-kicker">03 · QUEUE</p><h2>{shortStatus(job)}</h2></div>{job && <span className={`status-chip ${job.status}`}><i />{job.status === "completed" ? "完成" : job.status === "failed" ? "失败" : job.status === "selection_ready" ? "可选择" : "处理中"}</span>}</div>{job ? <><div className="job-meta"><span><b>{job.input?.original_name || sourceLabel(source)}</b><small>{sourceLabel(job.v2?.source_kind || source)} · 第 {job.attempt || 1} 次</small></span><span className="job-id">{job.id.slice(0, 8)}</span></div><div className="phase-rail">{PHASES.map(([phase, label], index) => <span key={phase} className={`${index < statusIndex ? "done" : ""} ${index === statusIndex ? "active" : ""}`} title={label}><i /></span>)}</div><div className="phase-labels">{PHASES.map(([phase, label]) => <span key={phase} className={phase === job.status || phase === job.phase ? "current" : ""}>{label}</span>)}</div>{job.progress !== null && job.progress !== undefined && <div className="progress-meter"><span style={{ width: `${Math.max(2, Math.min(100, job.progress * 100))}%` }} /></div>}{(job.status === "failed" || job.status === "interrupted") && <div className="failure-state"><span className="failure-mark">×</span><div><strong>{job.status === "interrupted" ? "任务中断" : "这次识别没有完成"}</strong><p>{job.error?.message || "可以保留当前选择并重试。"}</p><button type="button" className="outline-action" onClick={() => void retry()} disabled={!job.retryable || busy}>重新排队</button></div></div>}</> : <div className="empty-state"><span className="empty-wave">∿</span><p>上传后，这里会显示识别进度与可用产物。</p><small>{capabilities?.hardware?.cuda ? `CUDA · ${capabilities.hardware.gpu || "本机 GPU"}` : "本机任务队列"}</small></div>}</section></div>

      {isInstrumental && isReady && <section className="instrument-workspace panel" aria-labelledby="workspace-heading"><div className="panel-heading workspace-title"><div><p className="section-kicker">04 · SELECTION READY</p><h2 id="workspace-heading">全量识别完成，选出要留下的轨道</h2><p className="heading-note">{trackCountLabel} · 勾选变化不会重新运行 MuScriptor。</p></div><span className="selection-revision">R{job?.v2?.selection_revision || 0}</span></div><div className="listen-strip"><div className="listen-card original-listen"><span className="listen-icon">◉</span><div><strong>原曲试听</strong><small>音频文件的真实播放</small></div>{originalArtifact?.url ? <audio ref={audioRef} controls src={originalArtifact.url} onTimeUpdate={(event) => setOriginalTime(event.currentTarget.currentTime)} onEnded={() => setOriginalTime(0)} /> : <span className="listen-pending">识别后可用</span>}</div><div className="listen-card synth-listen"><span className="listen-icon">⌁</span><div><strong>本机合成试听</strong><small title={soundfontStatus?.sha256 || "官方 SF3 音色库"}>{synthResource}</small></div><button type="button" className={`listen-button ${synthPlaying ? "playing" : ""}`} onClick={() => void playSynth()}>{synthPlaying ? "停止" : "播放选中轨"}</button><span className="synth-time">{formatDuration(synthTime)} / {formatDuration(duration)}</span></div></div><div className="workspace-grid"><div className="roll-column"><div className="subhead"><div><span className="section-kicker">PIANO ROLL</span><strong>时间同步试听轨</strong></div><div className="roll-tools"><label><input type="checkbox" checked={rollVisible} onChange={(event) => setRollVisible(event.target.checked)} />显示</label><label>缩放 <input aria-label="钢琴卷帘缩放" type="range" min="1" max="4" step="0.5" value={rollZoom} onChange={(event) => setRollZoom(Number(event.target.value))} /></label></div></div>{renderPianoRoll()}<p className="roll-caption">播放原曲或本机合成时，米白播放头沿原始秒数移动。颜色只表示轨道，音符时间不经过简谱量化。</p></div><aside className="track-roster"><div className="subhead"><div><span className="section-kicker">TRACK ROSTER</span><strong>乐器清单</strong></div><span className="roster-count">{selectedTrackIds.length}/{tracks.length}</span></div><div className="track-list">{tracks.map((track, index) => { const color = LANE_COLORS[index % LANE_COLORS.length]; const selected = selectedTrackIds.includes(track.track_id); const muted = mutedTrackIds.includes(track.track_id); return <div className={`track-row ${selected ? "selected" : ""}`} key={track.track_id}><span className="track-color" style={{ background: color }} /><label className="track-check"><input type="checkbox" checked={selected} onChange={() => toggleTrack(track.track_id)} /><span><strong>{track.label_zh}</strong><small>{track.instrument_group} · {track.note_count} notes · {track.is_drum ? "不生成简谱" : `program ${track.program}`}</small></span></label><button type="button" className={`mute-button ${muted ? "muted" : ""}`} onClick={() => toggleMute(track.track_id)} aria-label={`${muted ? "恢复" : "静音"}${track.label_zh}`}>{muted ? "静" : "听"}</button></div>; })}</div>{selectedTracks.some((track) => track.is_drum) && <p className="drum-note">鼓组保留在试听与选择 MIDI 中，但不会生成简谱。</p>}{!pitchedSelected && selectedTrackIds.length > 0 && <p className="drum-note">当前只选中鼓组：可以下载 MIDI，简谱按钮会保持禁用。</p>}</aside></div><div className="export-rail"><div className="override-fields"><div><label htmlFor="export-bpm">BPM</label><input id="export-bpm" inputMode="decimal" value={bpm} onChange={(event) => setBpm(event.target.value)} /></div><div><label htmlFor="export-key">调性</label><select id="export-key" value={key} onChange={(event) => setKey(event.target.value)}>{KEYS.map((item) => <option key={item}>{item}</option>)}</select></div><div><label htmlFor="export-meter">拍号</label><select id="export-meter" value={meter} onChange={(event) => setMeter(event.target.value)}>{METERS.map((item) => <option key={item}>{item}</option>)}</select></div></div><label className="merge-option"><input type="checkbox" checked={mergeMainMelody} onChange={(event) => setMergeMainMelody(event.target.checked)} />另外生成单声部主旋律 <small>会丢失和声</small></label><div className="export-actions"><button type="button" className="outline-action" onClick={() => void submitSelection("midi")} disabled={busy || selectedTrackIds.length === 0}>下载所选 MIDI</button><button type="button" className="primary-action compact" onClick={() => void submitSelection("score")} disabled={busy || !pitchedSelected}><span>生成分谱</span><strong>↗</strong></button></div></div>{scoreArtifacts.length > 0 && <div className="score-deck"><div className="subhead"><div><span className="section-kicker">SCORE PAGES</span><strong>已选有音高乐器分谱</strong></div><span className="score-warning">鼓组不在分谱中</span></div><div className="score-grid">{scoreArtifacts.map((artifact) => <a className="score-card" key={artifact.artifact_id} href={artifact.url || "#"} target="_blank" rel="noreferrer"><span className="score-thumb"><img src={artifact.url || ""} alt={artifact.label} /></span><strong>{artifact.label}</strong><small>SVG · 下载或新窗口查看</small></a>)}</div>{selectedMidi && <a className="download-line" href={selectedMidi.url || "#"}>↓ 下载选择版本 MIDI · R{job?.v2?.selection_revision || 0}</a>}</div>}{originalMidi && <a className="download-line muted-line" href={originalMidi.url || "#"}>↓ 完整识别 MIDI（原始时间） · 含全部轨道</a>}</section>}

      {source === "vocal" && job?.status === "completed" && <section className="vocal-result panel" aria-labelledby="vocal-heading"><div className="panel-heading"><div><p className="section-kicker">04 · GAME RESULT</p><h2 id="vocal-heading">人声主旋律已经展开</h2><p className="heading-note">GAME 直接处理人声，结果保留为单一主旋律谱面。</p></div><span className="status-chip completed"><i />完成</span></div><div className="vocal-listen">{originalArtifact?.url && <audio controls src={originalArtifact.url} />}</div><div className="score-grid">{artifacts.filter((item) => item.kind === "score_svg").map((artifact) => <a className="score-card" key={artifact.artifact_id} href={artifact.url || "#"} target="_blank" rel="noreferrer"><span className="score-thumb"><img src={artifact.url || ""} alt={artifact.label} /></span><strong>{artifact.label}</strong><small>人声主旋律 · SVG</small></a>)}</div><div className="download-stack">{vocalMidi?.url && <a className="download-line" href={vocalMidi.url}>↓ 下载人声主旋律 MIDI</a>}{artifacts.filter((item) => ["jianpu_source", "lilypond_source"].includes(item.kind)).map((artifact) => <a className="download-line" key={artifact.artifact_id} href={artifact.url || "#"}>↓ {artifact.label}</a>)}</div></section>}
      {!fixtureMode && job && (job.status === "selection_ready" || job.status === "completed") && <p className="retention-note">任务 {job.id} · 选择版本与旧版本按 revision 隔离，刷新后会从本机任务目录恢复。</p>}
    </main><footer className="footer"><span>谱面工作台 / V2</span><span>{capabilities?.api?.retention_hours ? `本机保留 ${capabilities.api.retention_hours} 小时` : "LOCAL ONLY"}</span></footer>
  </div>;
}

export default App;
