# MuScriptor 浏览器音色库

V2 的浏览器合成试听使用 [SpessaSynth](https://github.com/spessasus/spessasynth_lib) 的 AudioWorklet 封装和官方 MuScriptor web 使用的 MuseScore General SF3。SpessaSynth 的 npm 包固定为 `4.3.14`，Apache-2.0；`frontend/public/vendor/spessasynth_processor.min.js` 是该包的本地 worklet 处理器，运行时不依赖 CDN。

音色库来自 MuScriptor 官方 assets 仓库：

- 来源：`hf://MuScriptor/assets/MuseScore_General.sf3`
- 镜像：[huggingface.co/MuScriptor/assets](https://huggingface.co/MuScriptor/assets)
- 资产许可：MIT
- SHA-256：`5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3`
- 缓存位置：项目 `.cache/muscriptor/MuseScore_General.sf3`（被 `.gitignore` 忽略）

浏览器第一次请求 `/api/v2/soundfont` 时，后端从已登录的本机 MuScriptor 模型环境取得文件、校验 SHA-256 后再同源提供。这个合法的 MuseScore General SF3 约 40 MB，首次使用每台远端设备仍需完整下载；远程页面会使用 512 KiB Range 分块、单块重试和退避，成功后写入浏览器 Cache Storage 复用，服务端也返回 ETag、Range、Content-Length 和长期缓存头。页面会显示已收到的大小和百分比，而不是在慢隧道上静默等待。只要分块仍在持续收到字节就继续下载；同一分块连续三次网络/无数据超时失败后会自动改用明确标注的“轻量音色”Web Audio 波形试听，下载过程中也可主动点击“立即使用轻量试听”，并可随后“重新加载高质量音色”。轻量方案保持所选有音高轨道的音高和节奏，鼓组使用电子近似音色；点击“立即使用轻量试听”会中止未完成的高质量下载，因此本次不会写入缓存。下载或校验失败时明确显示 HTTP、响应类型、截断、超时或浏览器能力原因，不会把轻量波形冒充 SF3。缓存键包含服务端公布的 SF3 SHA-256 版本；写入后会回读并校验 Content-Length、Content-Type、RIFF/sfbk 和实际字节数，成功显示“高质量音色已缓存”，浏览器不支持或拒绝 Cache Storage 时显示原因并提示下次可能重新下载。远程地址必须使用 HTTPS 才能启用 AudioWorklet；`localhost`/`127.0.0.1` 的本机安全上下文例外。也可以提前运行：

```powershell
.\scripts\prefetch_soundfont.ps1
```

`velocity` 仍保持识别事件的 `None`；SpessaSynth 播放使用固定的 playback default 80。非鼓轨按 MIDI program 切换预置，鼓轨固定 MIDI channel 9 并启用鼓组。
