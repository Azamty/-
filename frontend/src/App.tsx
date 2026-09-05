import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Tier = "basic-pitch" | "specialist";
type SourceKind = "vocal" | "instrumental" | "mixed";
type VoiceMode = "monophonic" | "polyphonic";

type Artifact = {
  artifact_id: string;
  kind: string;
  label: string;
  filename: string;
  media_type: string;
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
  phase_index: number | null;
  progress: number | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  attempt: number;
  options: Record<string, unknown>;
  input: { original_name?: string; bytes?: number | null; duration_sec?: number | null };
  error?: { code: string; message: string } | null;
  warnings: string[];
  artifacts: Artifact[];
  summary?: { note_count?: number; voice_count?: number; bpm?: number; key?: string; time_signature?: string } | null;
  retryable: boolean;
  score_available: boolean;
};

type Capabilities = {
  engines?: Record<string, { available?: boolean; reason?: string | null; kind?: string; routes?: Record<string, string> }>;
  api?: { max_upload_bytes?: number; max_duration_sec?: number; retention_hours?: number };
};

const PHASES = [
  ["uploading", "上传中"],
  ["queued", "排队中"],
  ["probing", "检查音频"],
  ["separating", "分离声部"],
  ["recognizing", "识别音符"],
  ["quantizing", "整理节拍"],
  ["rendering", "生成谱面"],
  ["packaging", "整理文件"],
  ["completed", "谱面就绪"],
] as const;

const STEM_LABELS: Record<string, string> = {
  vocals: "人声",
  bass: "低音",
  other: "器乐",
  mixed: "原音",
};

const stemLabel = (stem: string) => STEM_LABELS[stem] || stem;

const formatBytes = (value?: number | null) => {
  if (!value) return "—";
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
};

const formatDuration = (value?: number | null) => {
  if (!value) return "—";
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
};

async function readResponse(response: Response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || "请求没有完成");
  }
  return payload;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [tier, setTier] = useState<Tier>("basic-pitch");
  const [source, setSource] = useState<SourceKind>("instrumental");
  const [voiceMode, setVoiceMode] = useState<VoiceMode>("monophonic");
  const [language, setLanguage] = useState("mixed");
  const [bpm, setBpm] = useState("");
  const [key, setKey] = useState("");
  const [meter, setMeter] = useState("");
  const [title, setTitle] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [previewTab, setPreviewTab] = useState("total");
  const [pageIndex, setPageIndex] = useState(0);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const scorePages = useMemo(
    () => artifacts.filter((item) => item.kind === "score_svg").sort((a, b) => (a.page || 0) - (b.page || 0)),
    [artifacts],
  );
  const stemIds = useMemo(
    () => Array.from(new Set(artifacts.filter((item) => item.kind === "stem_svg" && item.stem_id).map((item) => item.stem_id as string))),
    [artifacts],
  );
  const previewPages = useMemo(
    () => (previewTab === "total" ? scorePages : artifacts.filter((item) => item.kind === "stem_svg" && item.stem_id === previewTab).sort((a, b) => (a.page || 0) - (b.page || 0))),
    [artifacts, previewTab, scorePages],
  );
  const previewPage = previewPages[pageIndex] || previewPages[0];
  const engineReady = capabilities?.engines?.[tier]?.available !== false;
  const hasCompleted = job?.status === "completed";

  const loadJob = async (jobId: string) => {
    try {
      const next = (await readResponse(await fetch(`/api/jobs/${jobId}`))) as Job;
      setJob(next);
      if (next.status === "completed") {
        const result = await readResponse(await fetch(`/api/jobs/${jobId}/artifacts`));
        setArtifacts(result.artifacts || []);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "任务状态读取失败");
    }
  };

  useEffect(() => {
    fetch("/api/capabilities")
      .then(readResponse)
      .then(setCapabilities)
      .catch(() => setCapabilities(null));
    const saved = window.localStorage.getItem("jianpu-job-id");
    if (saved) void loadJob(saved);
  }, []);

  useEffect(() => {
    if (!job || ["completed", "failed", "interrupted"].includes(job.status)) return undefined;
    const timer = window.setInterval(() => void loadJob(job.id), 1400);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (tier === "specialist" && source === "mixed" && voiceMode === "monophonic") {
      setVoiceMode("polyphonic");
    }
  }, [tier, source, voiceMode]);

  useEffect(() => {
    setPageIndex(0);
  }, [previewTab, hasCompleted]);

  const chooseFile = (candidate?: File) => {
    if (!candidate) return;
    setError("");
    setFile(candidate);
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => chooseFile(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    chooseFile(event.dataTransfer.files?.[0]);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError("先放入一段音频，再开始转谱。");
      return;
    }
    setError("");
    setIsSubmitting(true);
    setJob(null);
    setArtifacts([]);
    setPreviewTab("total");
    window.localStorage.removeItem("jianpu-job-id");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("engine", tier);
      form.append("voice_mode", voiceMode);
      form.append("source_kind", source);
      form.append("language", language);
      if (bpm.trim()) form.append("bpm", bpm.trim());
      if (key.trim()) form.append("key", key.trim());
      if (meter.trim()) form.append("time_signature", meter.trim());
      if (title.trim()) form.append("title", title.trim());
      const next = (await readResponse(await fetch("/api/jobs", { method: "POST", body: form }))) as Job;
      setJob(next);
      window.localStorage.setItem("jianpu-job-id", next.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "任务没有创建成功");
    } finally {
      setIsSubmitting(false);
    }
  };

  const retry = async () => {
    if (!job) return;
    setError("");
    try {
      const next = (await readResponse(await fetch(`/api/jobs/${job.id}/retry`, { method: "POST" }))) as Job;
      setJob(next);
      setArtifacts([]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "重试没有开始");
    }
  };

  const activeIndex = job?.phase_index ?? -1;
  const midi = artifacts.find((item) => item.artifact_id === "score-midi");
  const previewMidi = previewTab === "total" ? midi : artifacts.find((item) => item.kind === "stem_midi" && item.stem_id === previewTab);
  const svgZip = artifacts.find((item) => item.artifact_id === "score-svg-zip");

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="谱面工作台首页">
          <span className="brand-glyph">∿</span>
          <span><strong>谱面</strong><small>LOCAL SCORE DESK</small></span>
        </a>
        <div className="topbar-meta">
          <span className="local-dot"><i />仅限本机</span>
          <span className="version-mark">v0.1 / CPU 优先</span>
        </div>
      </header>

      <main className="workspace">
        <section className="intro-block">
          <div className="intro-copy">
            <p className="eyebrow"><span className="eyebrow-rule" /> AUDIO → JIANPU</p>
            <h1>让一首歌留下<br /><em>可读的骨架。</em></h1>
            <p className="intro-text">上传一段音频，等待识别，在这里查看生成的简谱预览。</p>
          </div>
          <div className="notation-stamp" aria-hidden="true">
            <span className="stamp-label">MUSIC<br />ANALYSIS</span>
            <span className="stamp-note">♪</span>
            <span className="stamp-tick" />
            <span className="stamp-caption">127.0.0.1 / NO CLOUD</span>
          </div>
        </section>

        <div className="desk-grid">
          <form className="control-panel panel" onSubmit={submit}>
            <div className="panel-heading">
              <div><p className="section-kicker">START HERE</p><h2>放入音频</h2></div>
              <span className="step-badge">01</span>
            </div>
            <div
              className={`dropzone ${file ? "has-file" : ""}`}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
              aria-label="选择音频文件"
            >
              <input ref={inputRef} type="file" accept=".mp3,.wav,.flac,.m4a,audio/*" onChange={onFileChange} hidden />
              <span className="drop-orbit">{file ? "✓" : "↑"}</span>
              {file ? <><strong>{file.name}</strong><small>{formatBytes(file.size)} · 点击更换文件</small></> : <><strong>拖入一段音频</strong><small>或点击选择 · MP3 / WAV / FLAC / M4A</small></>}
            </div>

            <div className="form-block">
              <div className="field-label-row"><label>识别档位</label><span>建议先用基础档</span></div>
              <div className="segmented tier-segment" role="group" aria-label="识别档位">
                <button type="button" className={tier === "basic-pitch" ? "selected" : ""} onClick={() => setTier("basic-pitch")}><span className="segment-icon">◌</span><span><b>基础</b><small>Basic Pitch · 快速</small></span></button>
                <button type="button" className={tier === "specialist" ? "selected" : ""} onClick={() => setTier("specialist")}><span className="segment-icon">✦</span><span><b>专用</b><small>按声部路由 · 更慢</small></span></button>
              </div>
              {!engineReady && <p className="inline-warning">当前专用环境不可用：{capabilities?.engines?.[tier]?.reason || "请运行工具链检查"}</p>}
            </div>

            <div className="form-block two-column-block">
              <div><label htmlFor="source">来源</label><select id="source" value={source} onChange={(event) => setSource(event.target.value as SourceKind)}><option value="instrumental">纯音乐</option><option value="vocal">人声</option><option value="mixed">混合</option></select></div>
              <div><label htmlFor="language">语言</label><select id="language" value={language} onChange={(event) => setLanguage(event.target.value)}><option value="mixed">中文 / 日文 / 其他混合</option><option value="zh">中文</option><option value="ja">日文</option></select></div>
            </div>
            {tier === "specialist" && source === "mixed" && <p className="inline-warning">混合来源的专用识别会自动分离并保留多个声部；主旋律模式已切换为多声部。</p>}
            <div className="form-block">
              <div className="field-label-row"><label>记谱方式</label><span>{voiceMode === "polyphonic" ? "保留同步与异步声部" : "寻找连续主旋律"}</span></div>
              <div className="segmented compact-segment" role="group" aria-label="记谱方式">
                <button type="button" className={voiceMode === "monophonic" ? "selected" : ""} onClick={() => setVoiceMode("monophonic")}>主旋律</button>
                <button type="button" className={voiceMode === "polyphonic" ? "selected" : ""} onClick={() => setVoiceMode("polyphonic")}>多声部</button>
              </div>
            </div>

            <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced((value) => !value)} aria-expanded={showAdvanced}><span>{showAdvanced ? "收起" : "展开"}生成前覆盖</span><span>{showAdvanced ? "−" : "+"}</span></button>
            {showAdvanced && <div className="advanced-fields">
              <div><label htmlFor="bpm">BPM <small>可留空自动分析</small></label><input id="bpm" inputMode="decimal" placeholder="例如 80" value={bpm} onChange={(event) => setBpm(event.target.value)} /></div>
              <div><label htmlFor="key">调性 <small>例如 C / Am</small></label><input id="key" placeholder="自动" value={key} onChange={(event) => setKey(event.target.value)} /></div>
              <div><label htmlFor="meter">拍号 <small>2/4 · 3/4 · 4/4 · 6/8</small></label><select id="meter" value={meter} onChange={(event) => setMeter(event.target.value)}><option value="">自动（待确认）</option><option value="2/4">2/4</option><option value="3/4">3/4</option><option value="4/4">4/4</option><option value="6/8">6/8</option></select></div>
              <div><label htmlFor="title">谱面标题</label><input id="title" placeholder="使用文件名" value={title} onChange={(event) => setTitle(event.target.value)} /></div>
            </div>}
            {!meter && <p className="meter-note">自动拍号会在 2/4、3/4、4/4、6/8 中先给出候选；当前基础分析回退 4/4，生成后请确认。</p>}
            {error && <div className="error-message" role="alert"><span>!</span>{error}</div>}
            <button className="primary-action" type="submit" disabled={isSubmitting || !file || !engineReady}><span>{isSubmitting ? "正在送入队列…" : "开始转谱"}</span><strong>↗</strong></button>
            <p className="privacy-note"><span>⌁</span> 文件只保留在本机任务目录，完成后 24 小时清理。</p>
          </form>

          <section className="preview-panel panel" aria-live="polite">
            <div className="panel-heading preview-heading">
              <div><p className="section-kicker">SCORE WINDOW</p><h2>{hasCompleted ? "谱页已经展开" : job ? job.phase_label : "等待一页新谱"}</h2></div>
              {job && <span className={`status-chip ${job.status}`}><i />{job.status === "completed" ? "完成" : job.status === "failed" ? "失败" : job.status === "interrupted" ? "待重试" : job.status === "uploading" ? "上传中" : job.status === "queued" ? "排队中" : "处理中"}</span>}
            </div>
            {job && !hasCompleted && job.status !== "failed" && job.status !== "interrupted" && <div className="progress-area">
              <div className="progress-copy"><strong>{job.phase_label}</strong><span>当前阶段 · 完成后即可查看谱页</span></div>
              <div className="phase-rail">{PHASES.map(([phase, label], index) => <span key={phase} className={`${index < activeIndex ? "done" : ""} ${index === activeIndex ? "active" : ""}`} title={label}><i /></span>)}</div>
              <div className="phase-labels">{PHASES.filter(([phase]) => phase !== "completed").map(([phase, label]) => <span key={phase} className={phase === job.phase ? "current" : ""}>{label}</span>)}</div>
              <div className="queue-message"><span className="mini-pulse" />任务 {job.id.slice(0, 8)} · 第 {job.attempt} 次尝试</div>
            </div>}
            {job?.status === "failed" && <div className="failure-state"><span className="failure-mark">×</span><div><strong>{job.error?.code === "no_notes" ? "这一段没有留下音符" : "这次转谱没有完成"}</strong><p>{job.error?.message || "请换一段音频或调整选项后重试。"}</p><button type="button" className="outline-action" onClick={retry}>重新排队</button></div></div>}
            {job?.status === "interrupted" && <div className="failure-state"><span className="failure-mark">↻</span><div><strong>服务重启，中断在这里</strong><p>{job.error?.message || "任务数据仍在本机，可以重新排队。"}</p><button type="button" className="outline-action" onClick={retry}>继续这个任务</button></div></div>}
            {!job && <div className="empty-score">
              <div className="empty-paper"><span className="paper-playhead" /><span className="empty-clef">𝄞</span><div className="fake-staff"><i /><i /><i /><i /><i /></div><span className="fake-notes">1　2　3　—　5　6　7　1′</span></div>
              <p>谱页会在任务完成后出现在这里</p><small>生成后可查看 SVG 谱页预览</small>
            </div>}
            {hasCompleted && <>
              <div className="score-summary"><div><strong>{job.summary?.note_count ?? "—"}</strong><span>音符事件</span></div><div><strong>{job.summary?.voice_count ?? "—"}</strong><span>声部</span></div><div><strong>{job.summary?.key || "—"}</strong><span>调性</span></div><div><strong>{job.summary?.time_signature || "—"}</strong><span>拍号</span></div><div><strong>{job.summary?.bpm ? Math.round(job.summary.bpm) : "—"}</strong><span>BPM</span></div></div>
              <div className="score-tabs" role="tablist" aria-label="谱页选择"><button type="button" className={previewTab === "total" ? "active" : ""} onClick={() => setPreviewTab("total")}>总谱 {scorePages.length ? `· ${scorePages.length}页` : ""}</button>{stemIds.map((stem) => <button type="button" key={stem} className={previewTab === stem ? "active" : ""} onClick={() => setPreviewTab(stem)}>{stemLabel(stem)} 分谱</button>)}</div>
              <div className="score-viewer">{previewPage ? <figure key={previewPage.artifact_id}><img src={previewPage.url || ""} alt={previewPage.label} /><figcaption>{previewPage.label}</figcaption></figure> : <div className="viewer-empty">没有登记到 SVG 页面</div>}</div>
              {previewPages.length > 1 && <div className="page-nav"><button type="button" onClick={() => setPageIndex((index) => Math.max(0, index - 1))} disabled={pageIndex === 0}>← 上一页</button><span>第 {Math.min(pageIndex + 1, previewPages.length)} / {previewPages.length} 页</span><button type="button" onClick={() => setPageIndex((index) => Math.min(previewPages.length - 1, index + 1))} disabled={pageIndex >= previewPages.length - 1}>下一页 →</button></div>}
              <div className="download-row">{previewMidi?.url && <a className="download-action" href={previewMidi.url} download><span>♫</span>{previewTab === "total" ? "下载总谱 MIDI" : `下载 ${stemLabel(previewTab)} MIDI`}</a>}{svgZip?.url && <a className="download-action warm" href={svgZip.url} download><span>⌘</span>下载全部 SVG</a>}<a className="text-link" href={`/api/jobs/${job.id}/score`} target="_blank" rel="noreferrer">查看谱面数据 ↗</a></div>
            </>}
            {job && job.warnings.length > 0 && <div className="warnings"><p>需要确认</p>{Array.from(new Set(job.warnings)).map((warning) => <span key={warning}>↳ {warning}</span>)}</div>}
          </section>
        </div>

        <section className="footer-strip">
          <div><span className="strip-index">A</span><strong>节拍与声部</strong><small>休止、弱起和延音会留在谱面中</small></div>
          <div><span className="strip-index">B</span><strong>下载文件</strong><small>谱页、MIDI 和 ZIP 都可在当前任务下载</small></div>
          <div><span className="strip-index">C</span><strong>可恢复</strong><small>{job ? `任务 ${job.id.slice(0, 8)} 已记住` : "刷新页面可以恢复任务"}</small></div>
        </section>
      </main>
      <footer className="site-footer"><span>谱面工作台 / 本机音频转简谱</span><span>CPU first · localhost only</span></footer>
    </div>
  );
}

export default App;
