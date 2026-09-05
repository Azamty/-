# 音频转简谱（jianpu-score）

本项目把本机音频转换为可编辑的简谱数据，并通过 jianpu-ly 与 LilyPond 生成真实 SVG。首版运行在 `127.0.0.1`，不需要登录，不提供歌词编辑或云端 PDF 服务。

## 阶段1基础

- Python 后端环境使用项目内 `.venv`，创建它的解释器优先为本机 `E:\develop\anaconda3\envs\py310\python.exe`（Python 3.10）。
- 模型推理环境与后端环境分开，由 `scripts\install_models.ps1` 创建，后续通过子进程调用。
- `vendor\jianpu-ly\jianpu-ly.py` 固定为项目使用的 jianpu-ly 源码；阶段1使用 LilyPond 2.24.4 项目内工具目录。
- vendor 中保留了必要的本地兼容扩展，允许低于三组逗号或高于三组撇号的合法 MIDI 八度标记；每次扩展都用 LilyPond 编译并回读 MIDI 检查音高。
- 渲染器只接受统一 Score 层之后的记谱文本；人工谱例位于 `fixtures\polyphony.jly`。

## 本地初始化

在 Windows PowerShell 中运行：

```powershell
.\scripts\bootstrap.ps1
.\scripts\install_lilypond.ps1
.\scripts\check_toolchain.ps1
.\scripts\render_fixture.ps1
```

`install_lilypond.ps1` 会从项目缓存或 LilyPond 官方 2.24.4 压缩包重建被忽略的本地工具目录。`render_fixture.ps1` 会在 `artifacts\stage1\` 保存输入、jianpu-ly 生成的 LilyPond 文件、LilyPond 日志、SVG 和 MIDI。脚本不会下载模型权重。

## 可选模型环境

```powershell
.\scripts\install_models.ps1 -Model basic-pitch
.\scripts\install_models.ps1 -Model demucs
.\scripts\install_models.ps1 -Model game
.\scripts\install_models.ps1 -Model tsumugi
```

每个模型依赖安装在自己的项目内 `.venv-model-*`，不修改全局 conda。安装脚本先升级 pip、安装锁定的 CPU 依赖并执行 `pip check` 与 import probe；`game`、`tsumugi` 只使用子进程调用。tsumugi 的 CPU 路径故意不安装 `triton-windows`，避免官方 GPU 默认索引覆盖 CPU 配置。

GAME 源码在 `vendor\GAME-1.0.3`，small 权重、`config.yaml` 和 `lang_map.json` 在 `.cache\models\game\GAME-1.0-small`。GAME 的中文和日文语言 ID 每次从选定权重旁的 `lang_map.json` 读取；混合语言会省略 `--language`，使用官方语言无关默认值，并在结果中记录没有猜测 ID。当前权重的映射是 `zh=4`、`ja=2`，这不是代码中的硬编码。

tsumugi 源码在 `vendor\tsumugi-57b79ac4e1fa30c6f3eb95f14c77271fab637eeb`，各轨道使用独立 checkpoint：人声和声 `vocal_harmony_v1_5`、贝斯 `bass_v2`、其他乐器 `other_v1_5`。人声主旋律使用 GAME；tsumugi 人声模型是和声模型，结果会带有明确提示，不能当作 GAME 主旋律结果。所有适配器都会保留 stem、秒级起止、raw pitch、velocity 和原始模型元数据。

## 阶段2本地闭环

基础环境就绪后，可直接把 WAV、MP3、FLAC 或 M4A 转成 Score JSON、jianpu-ly 输入、LilyPond SVG 与 MIDI：

```powershell
& .\.venv\Scripts\python.exe -m backend.jianpu_score `
  --input .\input.wav `
  --output .\artifacts\job `
  --engine basic-pitch `
  --voice-mode monophonic `
  --bpm 80 --key C --time-signature 4/4
```

`--engine librosa` 是 CPU 基础音高后备；需要按人声或乐器声部识别时，`--separate` 会在独立 Demucs 环境调用 htdemucs。基础引擎未开启分轨时仍直接分析原始混音。所有推理结果先进入 `NoteEvent`、`MusicAnalysis`、`Score` 数据契约，再交给 jianpu-ly/LilyPond 渲染。

## 阶段3专用引擎与能力检测

```powershell
# 查看当前本机真正可用的环境、权重和路由
& .\.venv\Scripts\python.exe -m backend.jianpu_score --print-capabilities

# 人声单旋律：Demucs vocals -> GAME
& .\.venv\Scripts\python.exe -m backend.jianpu_score `
  --input .\input.wav --output .\artifacts\game-job `
  --engine specialist --source vocal --separate --voice-mode monophonic `
  --language zh

# 纯音乐单旋律：Demucs other -> tsumugi other_v1_5
& .\.venv\Scripts\python.exe -m backend.jianpu_score `
  --input .\input.wav --output .\artifacts\tsumugi-job `
  --engine specialist --source instrumental --separate --voice-mode monophonic

# 纯音乐多声部：Demucs bass/other -> 对应 tsumugi checkpoint
& .\.venv\Scripts\python.exe -m backend.jianpu_score `
  --input .\input.wav --output .\artifacts\tsumugi-poly-job `
  --engine specialist --source instrumental --separate --voice-mode polyphonic
```

`specialist` 会按来源和声部显式路由：vocal/monophonic 使用 GAME，instrumental/monophonic 使用 `other_v1_5`，instrumental/polyphonic 使用 `bass_v2` 与 `other_v1_5`，vocal/polyphonic 会对 vocals 使用和声 checkpoint 并同时处理 bass/other。缺少环境、权重或不支持的路由会抛出明确的 unavailable 错误，不会静默改用 Basic Pitch。`librosa` 在能力结果中标为 fallback；chordscope 仍是 Windows 兼容性未验证的 optional unavailable。

## 阶段4本机网页工作台

先在 `frontend\` 安装并构建一次前端，然后以前台方式启动本机服务：

```powershell
Push-Location .\frontend
npm install --no-audit --no-fund
npm run build
Pop-Location
.\scripts\start_server.ps1
```

浏览器打开 `http://127.0.0.1:8000`。网页支持拖入 MP3、WAV、FLAC、M4A，选择基础 Basic Pitch 或专用路由、来源、主旋律/多声部、语言以及生成前的 BPM、调性和拍号覆盖；任务按单 worker 串行执行，刷新后会按本机保存的 job UUID 恢复。完成任务会登记总谱和各声部分谱的全部 SVG 页面、MIDI、Score JSON、LilyPond/jianpu-ly 源文本、SVG ZIP 与任务日志。服务只监听 `127.0.0.1`，原文件最大 100 MB、时长最大 15 分钟，结束任务保留 24 小时。

API 入口为 `GET /api/capabilities`、`POST /api/jobs`、`GET /api/jobs/{job_id}`、`GET /api/jobs/{job_id}/score` 和 `GET /api/jobs/{job_id}/artifacts`；产物下载只接受持久化登记的 artifact ID。服务启动会把上次运行中的任务标记为“中断待重试”，不会删除模型缓存；`-Background` 可启动隐藏后台进程。

## 最终本地验收与使用边界

- 输入格式是 MP3、WAV、FLAC、M4A；上传文件上限 100 MB，解码后的音频时长上限 900 秒。文件、任务状态和生成文件保存在 `artifacts\jobs\<job_id>\`，已结束任务保留 24 小时，可从网页下载谱页、MIDI 和 SVG ZIP。
- 基础档调用独立环境中的 Basic Pitch，适合先快速试谱；专用档按来源调用 GAME 或 tsumugi。所有模型优先 CPU，耗时取决于时长和机器性能。阶段验收中 6 秒样本约 51 秒，180 秒合成曲的完整 API 任务约 214 秒；这些是本机参考值，不是性能保证。
- BPM、调性和拍号可以在生成前覆盖。自动拍号当前只给出 `2/4`、`3/4`、`4/4`、`6/8` 候选并回退到 `4/4`，结果会提示确认；音符先经过共享拍点时间线和 Score，再生成 SVG 与 MIDI。
- 简谱级数、主旋律连续性和分轨结果会受到混音、噪声、复音和模型能力影响。现有真实模型验收使用短参考音阶和 180 秒合成规模样本；没有把中日文或混合语言真实歌曲准确度宣称为已验证，生成结果需要人工复核。
- 180 秒规模验收记录在 `artifacts\review\stage5-final\run-20260904T120124Z\summary.json`：输入 180.0 秒，真实 Demucs + Basic Pitch 用时约 213.8 秒，分析首个事件约 0.012 秒、末个事件约 178.792 秒，Score 为 2 页，总谱和器乐分谱 MIDI 都是 180.0 秒。该目录中的 JSON、SVG、MIDI 和 ZIP 是可复查证据。

后台启动可双击 `scripts\start_server.cmd`，或执行 `.\scripts\start_server.ps1 -Background`；服务 PID 保存在 `artifacts\server.pid`。停止后台服务可双击 `scripts\stop_server.cmd`，或执行 `.\scripts\stop_server.ps1`，脚本会结束已核验的服务进程树，避免模型子进程残留和下一次启动重复 worker。前台运行 `.\scripts\start_server.ps1` 时，在该窗口按 Ctrl+C 退出。

## 模型来源与许可证

GAME 源码随项目放在 `vendor\GAME-1.0.3`，其 `LICENSE` 为 MIT；使用的官方 small 权重及其 `config.yaml`、语言映射位于 `.cache\models\game\GAME-1.0-small`。tsumugi 源码随项目放在 `vendor\tsumugi-57b79ac4e1fa30c6f3eb95f14c77271fab637eeb`，其 `LICENSE` 为 MIT；三个 checkpoint 的来源 revision、SHA256 和文件名记录在 `.cache\models\tsumugi\provenance.json`。权重只保存在本机项目缓存中，不进入后端全局环境。
