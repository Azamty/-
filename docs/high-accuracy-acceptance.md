# 高精度扒谱链路与验收

当前网页入口是 V2：`http://127.0.0.1:8000` 通过 `/api/v2/jobs` 上传、识别、选择轨道并导出。旧的 `/api/jobs` 与 `backend/jianpu_score/pipeline.py` 保留一个版本周期，供 Git 回退和对照使用；它们不是当前网页入口，V2 任务不会调用旧的均匀网格量化器。基准的 baseline 明确调用 `backend.jianpu_score.quantize.quantize_events`，只用于同一 raw 输入的历史对照；它不会改变网页入口，也不能作为新链路结果。

## 工具链

在项目根目录执行：

```powershell
.\scripts\install_high_accuracy.ps1
.\scripts\check_toolchain.ps1
```

BeatNet 依赖放在独立的 `.venv-model-beatnet`，MusicXML 标准化 worker 使用独立的 `.venv-notation`（Python 3.10、`music21==9.9.2`）。MuseScore 4.7.4 的固定版本文件和校验记录放在忽略的本地目录 `tools\musescore-4.7.4` 与 `.cache\packages`；能力检查会报告可执行文件、版本、两个导入 profile、profile SHA-256 和 MusicXML 输出是否可用。器乐默认 profile 的 SHA-256 为 `86742B91922F921F725A1A5810572AB458EB7FB7AAC46FC683C92352B837C9FF`，最短导入单位是 1/32，开启二连音、三连音、四连音及 human performance，关闭 5:4、7:4、9:8 连音。GAME 人声使用固定的 `tools\musescore-4.7.4\midi_import_options_vocal.xml`，SHA-256 为 `B47761C931A649E910E078CAAF57887756D89B529F1379B4E537746DC7653557`；它关闭 human-performance 的重新分段并开启 `SimplifyDurations`，保留稀疏歌声的性能起音，再交由生产 tempo map 播放。两套 profile 都只搜索能由48 TPQ精确表示的二进制时值和3:2三连音；其它连音由标准化器明确失败，不能静默舍入。能力检查会拒绝偏离 profile 的文件。MuseScore CLI 调用使用 `--factory-settings --test-mode -M ... -o ...`，同一服务进程内串行执行以避开 MuseScore crashpad 并发崩溃。

MuseScore 导入前会在只读的临时 MIDI 副本中加入一个独立的 `__JIANPU_SOURCE_ORIGIN_SENTINEL_v1__` track：它在 tick 0 放置一个可追踪的低力度标记音符，作用是让 human-performance importer 观察到原始时间原点；原始性能 MIDI 不会被改写。MusicXML 成功导出后，适配器只按精确 part-name 删除这个 sentinel part，并把 sentinel 名称、channel、导入 track、删除的 part id/name 和剩余 part id 写入服务 manifest；如果 sentinel 没有被保留或无法审计删除，stage 明确失败，不退回无 sentinel 的导入。该方案只恢复 importer 丢失的整体原点，MusicXML 内部音符时值和相对时序仍由 XML 保持。

MusicXML 标准化器保留同声部的合法 3:2 tuplets；若 MuseScore 在 48 TPQ 边界产生紧凑的 4/8 tick 细网格片段，会在每个成员的 nominal duration 为整数且无 tie/voice 歧义时写成显式 `3[` 或受审计的 `3:1[` fine-grid bracket。该表示不移动事件边界，并在 `notation_grid_repairs` 记录 ratio、voice、tick、event id 和 movement=0；不能证明时仍失败。若 tie voice 重分配造成一个连续 bracket 的 start/stop 跨 ScoreVoice，只在无重叠、无 gap、marker ratio 一致且 tick 时值可序列化时把 fragment 归到 start voice，并在 `alignment_report.json` 的 `tuplet_marker_repairs` 中记录原 voice、目标 voice、tick 和原始 marker。孤立 marker 只有在清除后普通 48 TPQ 时值可序列化时才清除；同声部 gap、嵌套或不支持的 ratio 仍会失败。

性能 MIDI 的同音高重叠按区间着色分配到独立 type-1 track；每个 lane 内保证同一 channel/pitch 不重叠，因此 MuseScore 不会因 note-off 配对歧义丢失音符。每个谱表优先使用四条 notation voice，超过四条时增加独立 MIDI track/ScoreVoice；MIDI channel 在可用的 15 条旋律 channel 用尽后允许重复，因为 track identity 仍独立且会写入 performance metadata。lane、track、channel 和 source note accounting 都进入审计；该表示不合并、删除或移动原始 note-on/off。

Production metadata 中的最终拍号（包括用户手动覆盖）优先于 MuseScore 从 performance MIDI 推断的初始拍号。若两者导致 MusicXML 小节边界不一致，标准化器会按最终的 `2/4`、`3/4`、`4/4` 或 `6/8` 以及明确的中途拍号事件重建 timeline；跨新小节线的音符、和弦、休止符会保留总时值，音符/和弦按 pitch 生成 `start/continue/stop` tie。若原始末小节在导入拍号下是完整小节、重划后只差一个尾部休止，标准化器会补足该尾部休止以满足 renderer 的完整小节约束，并在 audit 中记录补足范围。导入的 timeline、最终 meter、partial meter-change boundary 和每次事件拆分都会写入 `alignment_report.json`；tuplets 跨新边界无法安全保持时明确失败，不静默保留冲突拍号。

生产 BeatNet 的速度候选会保留模型 note onset、原曲 onset 和可用 Demucs drums/bass onset 的来源名称与数量；传入 `source_onsets` 的映射会让 `beat_grid.tempo.evidence_sources` 显示实际参与候选排序的多源证据，而不是把所有生产路由压成一个无来源的 `all` 序列。该证据只对 BeatNet 已返回的 half/original/double 候选排序，不替换拍点，也不会自动把鼓/贝斯 onset 当成独立拍号真值；拍号仍保留候选和低置信度告警。

MuseScore 也可以直接启动检查安装（项目解包目录或系统安装目录二选一）：

```powershell
& '.\tools\musescore-4.7.4\MuseScore 4\bin\MuseScore4.exe'
# 或：& 'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
```

若项目目录没有本地副本，安装脚本会从固定版本缓存或下载地址恢复，并先校验 SHA-256；重复执行会复用已校验文件。

## 页面产物和失败语义

选择有音高轨道后，每个乐器单独生成性能 MIDI、MuseScore 导出的 MusicXML、48 TPQ Score JSON、对齐报告、简谱源文本、LilyPond 源文本、分页 SVG、纵向长图 SVG 和最终 `score.mid`。页面先显示长图，分页 SVG 收在可展开区域；最终 MIDI、MusicXML、对齐报告、性能 MIDI、处理清单，以及人声的 GAME 原始/清理后音符都在“高精度产物”区域提供醒目的下载卡片。日志和辅助 JSON 默认收在折叠区。

结果区会显示 `musescore-midi-import`、`beatnet`、BeatNet 版本、MuseScore 版本和 `48 TPQ`。伴奏任务中鼓组只登记 MIDI 和试听产物，不显示为简谱分谱。某条有音高轨道失败时，任务继续保留已成功分谱，并显示轨道、阶段和错误；全部有音高轨道失败时，页面明确显示没有可生成的简谱。BeatNet、MuseScore、MusicXML 标准化或简谱渲染失败不会静默退回旧链路，失败清单和日志仍可下载。

## 重复验收

验收登记文件是 `fixtures/high_accuracy/benchmark_manifest.json`。当前清单登记 30 个选定 production case，另保留 5 个 PJS 孤立歌声诊断 case 和 `luv-letter` 本机完整性候选；脚本不会把音频或语料复制进 Git。30 个选定 case 的构成为：10 个固定 seed 的钢琴/吉他/贝斯/多轨合成样本、10 个 MAESTRO v3 官方钢琴 MIDI 渲染片段、5 个 CCMusic 中文混合曲片段和 5 个弱起/3/4/6/8/三连音/变速/复杂和弦专门 fixture。它们都必须走真实 audio→MuScriptor/GAME→BeatNet raw→baseline/new 链路；`model_output=true`、输入 hash 匹配、存在 pitched events 和独立 beat annotation 是硬条件，reference-isolation raw 永远不能替代 production raw。合成和本地 MIDI 渲染仍保留 `render_domain` 限制，但只要 raw 来自真实模型，就与 CCMusic 一起参加 30-case 主 gate。CCMusic 的五段来自同一首 Yueding 录音，分别覆盖 MusicXML q40、56、72、88、104 的 16 个四分音符（每段 12 秒），用于真实混合人声 production smoke；它们不是五首独立歌曲。

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py --check
```

生成器只写入被 `.gitignore` 忽略的 `.cache\high-accuracy-benchmarks\generated`。第一次只运行两个 smoke case：

```powershell
& .\.venv\Scripts\python.exe scripts\generate_high_accuracy_benchmarks.py `
  --case-id synthetic-piano-01 `
  --case-id special-triplet
```

它会保留本机可用的 PJS `pjs001`–`pjs005`（PJS 数据为 CC BY-SA 4.0）作为孤立歌声 pitch/rhythm 诊断，不把它们算入可靠 production 集或独立 BeatNet gate；同时登记 `E:\edge\first\Luv Letter.mp3`。如果已有服务结果，按 case id 放入结果目录后计算真实的 pitch F1、和弦保留率、节奏误差、拍点 F1、重拍 F1 和崩溃状态：

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py `
  --result-root .\artifacts\review\high-accuracy-benchmark\results
```

没有结果时指标保持 `null`。Luv Letter 的同名 MIDI 只用于首尾和版本核对候选；当前没有可靠音频对齐标注，因此脚本只允许完整性和人工听谱验收，不用于 pitch/rhythm 或“节奏误差下降 20%”结论。仓库不提交受版权保护音频。

批处理编排器是 `scripts/run_high_accuracy_batch.py`。默认的 production route 在独立可终止 worker 中运行：器乐走原曲 MuScriptor + 原曲 BeatNet 一次，人声走原曲 Demucs → GAME → GAME cleanup，并复用原曲 BeatNet 一次。每个 case 的识别只运行一次，原始音符和 beat grid 写入 `raw/` 后按 hash 复用；raw 同时保存 recognizer mode、identity 和 fingerprint，切换 reference/production 时不会静默复用另一种 raw，必须使用新的 mode-isolated result root。baseline 与 new 使用独立结果根并收到同一份 raw 的独立副本。已有成功 pipeline 会在 `--resume` 下跳过，但只在 raw hash、recognizer fingerprint 和 effective scope 都一致时跳过；单 case 失败写入明确的 stage/error manifest；production worker 超时会终止整个子进程树，不能留下后台模型。raw 中的鼓事件保留用于审计，baseline/new 记谱前会过滤鼓事件，鼓不会进入数字谱或最终 score MIDI。没有 adapter 时只记录“未配置”，不会伪造识别或准确率。`--reference-isolation` 是专门的量化器隔离模式：它从可靠参考 MIDI 生成 raw，并在 provenance 中明确 `model_output=false`，不能被当成人声或音频模型的端到端结果。

```powershell
& .\.venv\Scripts\python.exe scripts\run_high_accuracy_batch.py `
  --case-id synthetic-piano-01 `
  --reference-isolation `
  --run-legacy-baseline `
  --run-new-chain `
  --baseline-result-root .\artifacts\review\high-accuracy-benchmark\baseline `
  --new-result-root .\artifacts\review\high-accuracy-benchmark\new
```

MAESTRO 10 条现在已有本地 archive/render cache；selection manifest 记录官方下载地址、CC BY-NC-SA 4.0、实际字节数 `58416533`、校验值 `70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c`、选中文件 hash、源 MIDI 区间以及 clip/output hash。每条 production 输入是源 MIDI 首音符所在小节起的固定八小节片段，裁剪规则在识别前确定，绝不根据模型输出挑片；beat/downbeat grid 由片段源 MIDI 精确生成并独立于模型。该本地音频是确定性的谐波振荡器渲染，保留 MIDI 音高、时值、起音、原始力度、tempo/meter；它不代表真实钢琴音色、踏板噪声、房间声学或原始演奏细节，因此报告必须披露 `maestro_local_midi_render` 域限制。它仍然经过真实 MuScriptor/BeatNet，可以作为计划中的本地渲染 production case。合成 fixture 会按拍号在记谱 downbeat 提高 MIDI velocity，音高和时序保持不变，以便原曲 BeatNet 有可解释的小节重音；guitar 采用较小的 downbeat 增量来保持 MuScriptor 对基音的稳定识别，其他音色使用更明显的增量。这仍是本地合成域，不能冒充真实表演录音。PJS 的拍点文件由同源 MIDI 透明推导，报告会保留这一限制，不把它描述成独立人工 beat 标注。

CCMusic demo 使用官方 Zenodo 记录 `https://zenodo.org/records/5676893`（DOI `10.5281/zenodo.5676893`）。准备脚本只接受本地 archive，要求字节数 `302024881`、MD5 `DBDC4A7E019C6B7A1424D99FDD8A7838` 和 SHA-256 `477B5466936EEC40CEF7DFD43205900E3E4A651B8EC671FDCCAFF48910523053` 全部匹配；记录说明其可用于 computational musicology，但没有 SPDX license identifier。它只解包 cpop/Yueding 的五个成员，使用 pinned music21 worker 读取 MusicXML。新版准备脚本用 MusicXML 音高事件与 guide/vocal 的 chroma 做六个区间的局部仿射拟合，用 guide/accompaniment 的 chroma+onset 局部 DTW 做六个时间锚点的稳健仿射拟合，再以 score→accompaniment 与 score→tuned-vocal 的多锚点差值决定混音位置；所有锚点、残差、斜率、offset 和 feature score 都写入 `selection_manifest.json`。实际结果是 guide t=0 对应 score q≈39.242，guide→accompaniment 为 slope≈1.000004、offset≈28.9996s，score→accompaniment 为 `0.749895*q−0.427607s`，tuned-vocal 的 `vocal_mix_offset_sec≈28.611286s`，完整 vocal/accompaniment 独立校验约 `28.561286s`。旧的 `vocal_to_guide_shift_sec≈-0.44s` 仍保留为相对延迟诊断，明确不参与混音位置。五段按拟合后的绝对 audio 起点约 `29.568/41.566/53.565/65.563/77.561s` 裁剪，score beat grid 仍独立来自 MusicXML；所有生成音频和 MIDI 仍在 `.cache`，不提交到仓库：

CCMusic 的 12 秒片段用于评估窗口诊断，不代表 BeatNet 的完整上下文能力。`scripts/ccmusic_context_beatnet.py` 现在对完整的 `Yueding aligned accompaniment vocal.wav` 只运行一次 BeatNet，并把完整音频 SHA-256 `e38b9fd11dab41fbf603e825462ffc62fdaabf5141958bd27ce60dcb656d2a20`、完整 179 拍网格 SHA-256 `50de27ff6c5a23ae9daeb47d409a81633daea8da0e3d0b9e14231e08d8262f92`、每段绝对/局部时间映射和两侧最近边界拍写入 context provenance。每段 `beats`/`downbeats` 只保留窗口内的局部拍点，边界拍保存在 `context.window.boundary_beats`，GAME 音符逐字节复用原来的 immutable raw，不重跑 Demucs/GAME，也不读取 score beat grid 来调模型。完整上下文切出的 beat F1 为 `0.23/0.35/0.67/0.79/0.73`，downbeat F1 为 `0.00/0.40/0.89/0.89/0.89`；这是五段共享一次整曲识别的诊断，不能增加独立样本数，也不能覆盖原始短片模型失败。context batch 的 new/baseline fixed-total 节奏均值为 `2.030412/1.968370` quarter，pitch F1 均值为 `0.019280/0.046339`；主 gate 仍按真实比较结果失败。BeatNet full-track median tempo 为 `81.08 BPM`、meter `4/4`；独立 librosa beat tracker 在完整混音给出 `161.50 BPM`，显示半拍/双拍本身存在歧义。

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_ccmusic_benchmark.py `
  --archive .\.cache\packages\ccmusic-database-demo.zip `
  --overwrite
```

在已有真实 CCMusic production raw 上建立完整原曲 BeatNet 上下文并重跑两条记谱链：

```powershell
& .\.venv\Scripts\python.exe scripts\ccmusic_context_beatnet.py `
  --raw-root .\.artifacts\review\ccmusic-production-v2 `
  --output-root .\.artifacts\review\ccmusic-production-context-v1
& .\.venv\Scripts\python.exe scripts\run_high_accuracy_batch.py `
  --manifest fixtures\high_accuracy\benchmark_manifest.json `
  --case-id ccmusic-yueding-01 --case-id ccmusic-yueding-02 `
  --case-id ccmusic-yueding-03 --case-id ccmusic-yueding-04 `
  --case-id ccmusic-yueding-05 --production-recognizer `
  --run-legacy-baseline --run-new-chain `
  --result-root .artifacts\review\ccmusic-production-context-v1
```

校验和选取官方 MIDI 的命令是：

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_maestro_benchmark.py `
  --archive .\.cache\packages\maestro-v3.0.0-midi.zip
```

该命令默认不联网；需要下载时必须显式增加 `--download`，脚本仍会先验证固定 SHA-256，再按归档成员名排序选十条、记录成员 hash，并按源 MIDI 规则生成八小节 WAV、裁剪 MIDI 与 beat/downbeat 标注。这里的音频属于明确披露的本地 MIDI 渲染域；它不等于 MAESTRO 原始钢琴表演录音，但会作为真实 production recognizer 的 10 个计划 case，模型失败必须原样进入 gate 诊断。

MIDI 音符指标先把各文件的 tick 精确换算为四分音符位置，因此不同 PPQ 可直接比较；节奏报告同时给出起音和时值误差，单位是 `quarter_note`。当前报告不把 tempo map 推导的秒误差冒充为已计算指标。

要计算新旧链路的准确率门槛，另提供旧链路结果目录：

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py `
  --result-root .\artifacts\review\high-accuracy-benchmark\new `
  --baseline-result-root .\artifacts\review\high-accuracy-benchmark\baseline
```

报告会分别给出 `accuracy_gate_scopes.quantizer_isolation_overall` 与 `accuracy_gate_scopes.production_end_to_end_subset`，但它们只是诊断。`accuracy_claim_ready` 的唯一主门槛是 baseline/new 共享的 30 个可靠 production case，且 production raw 必须来自模型输出；quantizer isolation case、1+1 的 scope 拆分或 reference-derived raw 都不能替代这 30 个 case。production gate 还要求全部 30 个 eligible case 同时有 BeatNet beat/downbeat 指标，不能用少数 case 的高平均分覆盖缺失样本。合成音频和本地 MIDI 渲染的 25 个拍点文件，以及 CCMusic 五段从 MusicXML score timing 生成的拍点文件，都是独立于模型输出的可复现 ground truth；PJS 的 5 个同源 MIDI 派生拍点明确 excluded，不计入独立 BeatNet F1。量化器隔离总体只比较共享 raw 上的节奏、音高和和弦；端到端子集按真实音频模型产物比较同样指标。数值节奏指标采用固定总量 pitch/onset assignment：匹配音符计入 onset+duration 误差，每个未匹配参考音符（FN）或未匹配预测音符（FP）固定罚 `1.0` 个四分音符，并按该 case 的参考音符数归一化；`mean_matched_rhythm_error_quarter` 仅作为覆盖率相同子集的诊断，不能替代 gate 字段。结果必须带 `metric_schema=fixed_total_assignment_v1` 和 `mean_fixed_total_assignment_rhythm_error_quarter`；旧版 baseline report 只有 matched-only 字段时会明确要求重跑，不会 fallback 参与 20% 验收。这样零匹配 case 仍有明确数值，不能在跨 case 平均时被漏掉。主门槛还必须满足无崩溃、固定总量节奏误差相对 baseline 下降至少 20%、pitch F1 下降不超过 0.01 且和弦保留率不下降；缺少任何数据会列出具体原因并保持 `false`。也可以用 `--baseline-report` 读取带有该 fixed-total schema 的 baseline 报告。

运行后端与前端检查：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
Push-Location frontend
npm run test:frontend
npm run build
Pop-Location
```
