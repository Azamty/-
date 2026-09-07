# 高精度扒谱链路与验收

当前网页入口是 V2：`http://127.0.0.1:8000` 通过 `/api/v2/jobs` 上传、识别、选择轨道并导出。旧的 `/api/jobs` 与 `backend/jianpu_score/pipeline.py` 保留一个版本周期，供 Git 回退和对照使用；它们不是当前网页入口，V2 任务不会调用旧的均匀网格量化器。

## 工具链

在项目根目录执行：

```powershell
.\scripts\install_high_accuracy.ps1
.\scripts\check_toolchain.ps1
```

BeatNet 依赖放在独立的 `.venv-model-beatnet`，MusicXML 标准化 worker 使用独立的 `.venv-notation`（Python 3.10、`music21==9.9.2`）。MuseScore 4.7.4 的固定版本文件和校验记录放在忽略的本地目录 `tools\musescore-4.7.4` 与 `.cache\packages`；能力检查会报告可执行文件、版本、两个导入 profile、profile SHA-256 和 MusicXML 输出是否可用。器乐默认 profile 的 SHA-256 为 `86742B91922F921F725A1A5810572AB458EB7FB7AAC46FC683C92352B837C9FF`，最短导入单位是 1/32，开启二连音、三连音、四连音及 human performance，关闭 5:4、7:4、9:8 连音。GAME 人声使用固定的 `tools\musescore-4.7.4\midi_import_options_vocal.xml`，SHA-256 为 `B47761C931A649E910E078CAAF57887756D89B529F1379B4E537746DC7653557`；它关闭 human-performance 的重新分段并开启 `SimplifyDurations`，保留稀疏歌声的性能起音，再交由生产 tempo map 播放。两套 profile 都只搜索能由48 TPQ精确表示的二进制时值和3:2三连音；其它连音由标准化器明确失败，不能静默舍入。能力检查会拒绝偏离 profile 的文件。MuseScore CLI 调用使用 `--factory-settings --test-mode -M ... -o ...`，同一服务进程内串行执行以避开 MuseScore crashpad 并发崩溃。

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

验收登记文件是 `fixtures/high_accuracy/benchmark_manifest.json`，脚本不会把音频或语料复制进 Git：

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py --check
```

它会登记本机可用的 PJS `pjs001`–`pjs005`（PJS 数据为 CC BY-SA 4.0），并登记 `E:\edge\first\Luv Letter.mp3`。如果已有服务结果，按 case id 放入结果目录后计算真实的 pitch F1、和弦保留率、节奏误差、拍点 F1、重拍 F1 和崩溃状态：

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py `
  --result-root .\artifacts\review\high-accuracy-benchmark\results
```

没有结果时指标保持 `null`。Luv Letter 的同名 MIDI 只用于首尾和版本核对候选；当前没有可靠音频对齐标注，因此脚本只允许完整性和人工听谱验收，不用于 pitch/rhythm 或“节奏误差下降 20%”结论。仓库不提交受版权保护音频。

MIDI 音符指标先把各文件的 tick 精确换算为四分音符位置，因此不同 PPQ 可直接比较；节奏报告同时给出起音和时值误差，单位是 `quarter_note`。当前报告不把 tempo map 推导的秒误差冒充为已计算指标。

要计算新旧链路的准确率门槛，另提供旧链路结果目录：

```powershell
& .\.venv\Scripts\python.exe scripts\high_accuracy_benchmark.py `
  --result-root .\artifacts\review\high-accuracy-benchmark\new `
  --baseline-result-root .\artifacts\review\high-accuracy-benchmark\baseline
```

只有至少 30 个可靠结果、拍点 F1 ≥ 0.85、重拍 F1 ≥ 0.75、无崩溃、节奏误差相对 baseline 下降至少 20%、pitch F1 下降不超过 0.01 且和弦保留率不下降时，报告才会将 `accuracy_claim_ready` 设为 `true`；缺少任何数据会列出具体原因并保持 `false`。也可以用 `--baseline-report` 读取已经生成的 baseline 报告。

运行后端与前端检查：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
Push-Location frontend
npm run test:frontend
npm run build
Pop-Location
```
