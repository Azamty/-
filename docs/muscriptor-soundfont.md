# MuScriptor 浏览器音色库

V2 的浏览器合成试听使用 [SpessaSynth](https://github.com/spessasus/spessasynth_lib) 的 AudioWorklet 封装和官方 MuScriptor web 使用的 MuseScore General SF3。SpessaSynth 的 npm 包固定为 `4.3.14`，Apache-2.0；`frontend/public/vendor/spessasynth_processor.min.js` 是该包的本地 worklet 处理器，运行时不依赖 CDN。

音色库来自 MuScriptor 官方 assets 仓库：

- 来源：`hf://MuScriptor/assets/MuseScore_General.sf3`
- 镜像：[huggingface.co/MuScriptor/assets](https://huggingface.co/MuScriptor/assets)
- 资产许可：MIT
- SHA-256：`5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3`
- 缓存位置：项目 `.cache/muscriptor/MuseScore_General.sf3`（被 `.gitignore` 忽略）

浏览器第一次请求 `/api/v2/soundfont` 时，后端从已登录的本机 MuScriptor 模型环境取得文件、校验 SHA-256 后再同源提供。页面先读取 `/api/v2/soundfont/status`，会显示首次缓存状态；下载或校验失败时明确显示“音色库不可用”，不会把振荡器冒充音色库。也可以提前运行：

```powershell
.\scripts\prefetch_soundfont.ps1
```

`velocity` 仍保持识别事件的 `None`；SpessaSynth 播放使用固定的 playback default 80。非鼓轨按 MIDI program 切换预置，鼓轨固定 MIDI channel 9 并启用鼓组。
