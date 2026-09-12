# 主旋律 + 和弦组合谱网页

当前网页验收使用 `codex/melody-harmony-web`。本轮从主旋律诊断提交 `cef7b2b`（完整值 `cef7b2b29a5fe2fd34b0ed3c7914e80e280a10c6`）继续加入组合谱、revision 隔离、网页直达和排版修复；旧的 `codex/melody-accuracy-v1` 保留，作为回退和对照点。关键提交包括：组合谱 `6350559`、网页接入 `a5b556b`、API revision 过滤 `7d603dc`、LilyPond 排版 `0404cce` 与 lane 压缩 `2d1339f`。需要回到主旋律诊断基线时使用 `cef7b2b` 或 `codex/melody-accuracy-v1`，不要把验收服务目录中的复制任务文件带回代码分支。

## 启动与直达任务

本轮组合谱验收使用复制任务目录启动的独立 `http://127.0.0.1:8001` 服务；8001 的页面和结果只代表本轮隔离验收，不代表 8000 的后端已经热更新。8000 上的既有任务继续运行。当前验收任务：

- Seaside Steps：`http://127.0.0.1:8001/?job=a7428e72-c181-48cd-9953-f19247955d8e`
- Luv Letter：`http://127.0.0.1:8001/?job=7acc1e55-86ea-445a-bb8f-59f4478a0b2a`

正常使用新代码时，先停止旧的 8000 服务，再在仓库根目录运行 `scripts/start_server.cmd`；服务重新监听 8000 后，打开 `http://127.0.0.1:8000/?job=<UUID>`，或上传新音频。若只需查看本轮复制任务，使用上面的 8001 直达链接，不要重用 8000 的旧任务目录。

## 结果含义

结果页优先显示组合长图：主旋律在上方，伴奏和弦在下方。组合产物使用 `melody_harmony_score_svg_long`、`melody_harmony_score_svg`、`melody_harmony_score_midi` 和 `melody_harmony_score_json`，并按当前 selection revision 展示。分页 SVG 仍保留。

组合谱 MIDI 是与组合长图对应的试听和导出文件；“所选轨道 MIDI”保留选中轨道（可含鼓组）的完整选择版本；“单音主旋律 MIDI”是可选的单声部旋律，生成时会丢失和声；“完整识别 MIDI”保留原始全轨道识别结果。网页中的用户试听用于检查结果是否可用和便于人工对照；本页不宣称音乐准确率已经验证或达到门槛。
