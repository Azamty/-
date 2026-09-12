# 主旋律 + 和弦组合谱网页

当前网页验收使用 `codex/melody-harmony-web`，由主旋律诊断提交 `cef7b2b` 创建；旧的 `codex/melody-accuracy-v1` 保留，便于回退和对照。

## 启动与直达任务

在仓库根目录双击 `scripts/start_server.cmd`，服务监听 `http://127.0.0.1:8000`。网页支持 `?job=<UUID>` 直达已有 V2 任务，URL 中的任务优先于本机上次保存的任务；无效 UUID 不会加载旧任务。8000 上的既有任务继续运行；本轮组合谱验收使用复制到隔离任务目录的 8001 服务，普通启动脚本仍需停止后重启才会加载新的后端代码。当前验收任务：

- Seaside Steps：`http://127.0.0.1:8001/?job=a7428e72-c181-48cd-9953-f19247955d8e`
- Luv Letter：`http://127.0.0.1:8001/?job=7acc1e55-86ea-445a-bb8f-59f4478a0b2a`

## 结果含义

结果页优先显示组合长图：主旋律在上方，伴奏和弦在下方。组合产物使用 `melody_harmony_score_svg_long`、`melody_harmony_score_svg`、`melody_harmony_score_midi` 和 `melody_harmony_score_json`，并按当前 selection revision 展示。分页 SVG 仍保留。

组合谱 MIDI 是与组合长图对应的试听和导出文件；“所选轨道 MIDI”保留选中轨道（可含鼓组）的完整选择版本；“单音主旋律 MIDI”是可选的单声部旋律，生成时会丢失和声；“完整识别 MIDI”保留原始全轨道识别结果。网页中的用户试听用于检查结果是否可用和便于人工对照，不代表准确率门槛已经通过。
