# 高精度扒谱链路与验收

当前网页入口是 V2：`http://127.0.0.1:8000` 通过 `/api/v2/jobs` 上传、识别、选择轨道并导出。旧的 `/api/jobs` 与 `backend/jianpu_score/pipeline.py` 保留一个版本周期，供 Git 回退和对照使用；它们不是当前网页入口，V2 任务不会调用旧的均匀网格量化器。基准的 baseline 明确调用 `backend.jianpu_score.quantize.quantize_events`，只用于同一 raw 输入的历史对照；它不会改变网页入口，也不能作为新链路结果。

## 工具链

在项目根目录执行：

```powershell
.\scripts\install_high_accuracy.ps1
.\scripts\check_toolchain.ps1
```

BeatNet 依赖放在独立的 `.venv-model-beatnet`，MusicXML 标准化 worker 使用独立的 `.venv-notation`（Python 3.10、`music21==9.9.2`）。MuseScore 4.7.4 的固定版本文件和校验记录放在忽略的本地目录 `tools\musescore-4.7.4` 与 `.cache\packages`；能力检查会报告可执行文件、版本、两个导入 profile、profile SHA-256 和 MusicXML 输出是否可用。器乐默认 profile 的 SHA-256 为 `86742B91922F921F725A1A5810572AB458EB7FB7AAC46FC683C92352B837C9FF`，最短导入单位是 1/32，开启二连音、三连音、四连音及 human performance，关闭 5:4、7:4、9:8 连音。GAME 人声使用固定的 `tools\musescore-4.7.4\midi_import_options_vocal.xml`，SHA-256 为 `B47761C931A649E910E078CAAF57887756D89B529F1379B4E537746DC7653557`；它关闭 human-performance 的重新分段并开启 `SimplifyDurations`，保留稀疏歌声的性能起音，再交由生产 tempo map 播放。两套 profile 都只搜索能由48 TPQ精确表示的二进制时值和3:2三连音；其它连音由标准化器明确失败，不能静默舍入。能力检查会拒绝偏离 profile 的文件。MuseScore CLI 调用使用 `--factory-settings --test-mode -M ... -o ...`，同一服务进程内串行执行以避开 MuseScore crashpad 并发崩溃。

MusicXML 标准化器保留同声部的合法 3:2 tuplets；若 tie voice 重分配造成一个连续 bracket 的 start/stop 跨 ScoreVoice，只在无重叠、无 gap、marker ratio 一致且 tick 时值可序列化时把 fragment 归到 start voice，并在 `alignment_report.json` 的 `tuplet_marker_repairs` 中记录原 voice、目标 voice、tick 和原始 marker。孤立 marker 只有在清除后普通 48 TPQ 时值可序列化时才清除；同声部 gap、嵌套或不支持的 ratio 仍会失败。

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

验收登记文件是 `fixtures/high_accuracy/benchmark_manifest.json`。当前清单包含 30 个可靠 case，另保留 5 个 PJS 孤立歌声诊断 case 和 `luv-letter` 本机完整性候选；脚本不会把音频或语料复制进 Git。30 个可靠 case 的构成为：10 个固定 seed 的钢琴/吉他/贝斯/多轨合成样本、10 个 MAESTRO v3 官方钢琴 MIDI 渲染候选、5 个 CCMusic 中文混合曲片段和 5 个弱起/3/4/6/8/三连音/变速/复杂和弦专门 fixture。CCMusic 的五段来自同一首 Yueding 录音，分别覆盖 MusicXML q40、56、72、88、104 的 16 个四分音符（每段 12 秒），用于真实混合人声的 production smoke 与主门槛；它们不是五首独立歌曲。

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

MAESTRO 10 条目前只登记官方入口、CC BY-NC-SA 4.0、MIDI archive SHA-256 和选取规则，状态是 `not_downloaded`；它没有被下载或冒充本地结果。待审查后按清单中的官方下载地址取得 archive，校验 `70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c`，再记录实际 archive 字节数、选中文件 hash，并从 MIDI 渲染音频。该本地音频是确定性的谐波振荡器渲染，保留 MIDI 音高、时值、起音、力度和 tempo map；它不代表真实钢琴音色、踏板噪声、房间声学或原始演奏细节，manifest 会把这些限制写入 `render_domain`。合成 fixture 会按拍号在记谱 downbeat 提高 MIDI velocity，音高和时序保持不变，以便原曲 BeatNet 有可解释的小节重音；guitar 采用较小的 downbeat 增量来保持 MuScriptor 对基音的稳定识别，其他音色使用更明显的增量。这仍是本地合成域，不能冒充真实表演录音。PJS 的拍点文件由同源 MIDI 透明推导，报告会保留这一限制，不把它描述成独立人工 beat 标注。

CCMusic demo 使用官方 Zenodo 记录 `https://zenodo.org/records/5676893`（DOI `10.5281/zenodo.5676893`）。准备脚本只接受本地 archive，要求字节数 `302024881`、MD5 `DBDC4A7E019C6B7A1424D99FDD8A7838` 和 SHA-256 `477B5466936EEC40CEF7DFD43205900E3E4A651B8EC671FDCCAFF48910523053` 全部匹配；记录说明其可用于 computational musicology，但没有 SPDX license identifier。它只解包 cpop/Yueding 的五个成员，使用 pinned music21 worker 读取 MusicXML，按 onset correlation 记录 tuned vocal 与 XML guide 的相对延迟，再将 vocal 放入 48 kHz accompaniment，生成五段 12 秒混合音频、裁剪 vocal-score MIDI 和独立于模型输出的 MusicXML beat/downbeat grid。manifest 中的 `vocal_to_guide_shift_sec`（实际约 `-0.44s`）只表示两条源轨的相对延迟诊断，明确不参与混音；`vocal_mix_offset_sec`（实际 `26.345s`）是把 tuned vocal 首个稳定起音锚到 MusicXML 首个音符在完整 accompaniment 时间轴上的绝对放置偏移。所有生成音频和 MIDI 仍在 `.cache`，不提交到仓库：

CCMusic 的 12 秒片段用于评估窗口诊断，不代表 BeatNet 的完整上下文能力：本地复现实验中，整曲 librosa 速度约 `80.75 BPM`，而五个独立短片出现约 `152/95.7/80.75/80.75/161.5 BPM`；BeatNet 短片还分别给出 `3/4、2/4、2/4、2/4、4/4` 候选。后续 context 评估应对整首混音只运行一次 BeatNet，再按每段的绝对窗口切出预测 beats/downbeats 与独立 score grid 比较；五段仍共享一首录音和一次上下文识别，不能增加独立样本数，也不能用上下文结果覆盖短片真实失败。

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_ccmusic_benchmark.py `
  --archive .\.cache\packages\ccmusic-database-demo.zip `
  --overwrite
```

校验和选取官方 MIDI 的命令是：

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_maestro_benchmark.py `
  --archive .\.cache\packages\maestro-v3.0.0-midi.zip
```

该命令默认不联网；需要下载时必须显式增加 `--download`，脚本仍会先验证固定 SHA-256，再按归档成员名排序选十条、记录成员 hash，并在本地渲染 WAV 与 beat/downbeat 标注。这里的音频是 MIDI 渲染音频，属于量化器隔离样本；在拥有独立表演录音前不会被描述为端到端准确率证据。

MIDI 音符指标先把各文件的 tick 精确换算为四分音符位置，因此不同 PPQ 可直接比较；节奏报告同时给出起音和时值误差，单位是 `quarter_note`。当前报告不把 tempo map 推导的秒误差冒充为已计算指标。

要计算新旧链路的准确率门槛，另提供旧链路结果目录：

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py `
  --result-root .\artifacts\review\high-accuracy-benchmark\new `
  --baseline-result-root .\artifacts\review\high-accuracy-benchmark\baseline
```

报告会分别给出 `accuracy_gate_scopes.quantizer_isolation_overall` 与 `accuracy_gate_scopes.production_end_to_end_subset`，但它们只是诊断。`accuracy_claim_ready` 的唯一主门槛是 baseline/new 共享的 30 个可靠 production case，且 production raw 必须来自模型输出；quantizer isolation case、1+1 的 scope 拆分或 reference-derived raw 都不能替代这 30 个 case。production gate 还要求全部 30 个 eligible case 同时有 BeatNet beat/downbeat 指标，不能用少数 case 的高平均分覆盖缺失样本。合成音频和本地 MIDI 渲染的 25 个拍点文件，以及 CCMusic 五段从 MusicXML score timing 生成的拍点文件，都是独立于模型输出的可复现 ground truth；PJS 的 5 个同源 MIDI 派生拍点明确 excluded，不计入独立 BeatNet F1。量化器隔离总体只比较共享 raw 上的节奏、音高和和弦；端到端子集按真实音频模型产物比较同样指标。主门槛还必须满足无崩溃、节奏误差相对 baseline 下降至少 20%、pitch F1 下降不超过 0.01 且和弦保留率不下降；缺少任何数据会列出具体原因并保持 `false`。也可以用 `--baseline-report` 读取已经生成的 baseline 报告。

运行后端与前端检查：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
Push-Location frontend
npm run test:frontend
npm run build
Pop-Location
```
