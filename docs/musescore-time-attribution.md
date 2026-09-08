# MuseScore 时值归因诊断

本次诊断覆盖 12 个 sentinel production 成功 case，输入包括 immutable production raw、performance MIDI、MusicXML alignment audit、new/baseline Score MIDI 和固定 MuseScore profile。profile 候选选择没有读取 benchmark reference MIDI、音频或 beat annotation。逐例 JSON/Markdown 证据保存在 `.artifacts/review/musescore-origin-ablation-v1/musescore-time-attribution.json` 与 `.artifacts/review/musescore-origin-ablation-v1/musescore-time-attribution.md`。

当前边界统计为：MusicXML→Score 阶段共 30 次细网格量化、19 次有界 notation repair、0 次 tuplet marker repair、0 次 tie/voice repair 和 37 次 meter rebar event split。source pitch multiset 和一对一匹配在成功标准化的 case 中保持一致。由此可以把大幅的 source→MusicXML 位移归因到 MuseScore 的 `HumanPerformance=true` 自适应导入；MusicXML→Score 只做报告中记录的 48 TPQ 有界修复和重划小节线。BeatNet tempo/phase 已在 performance MIDI 写入前决定，profile 不能修正上游 beat mapping。

source consistency 的全 12 例比较使用每个 pitch 的顺序配对并拟合一个 source→target affine 坐标，只评估导入对 source 相对时序的保持程度：new 的平均 duration residual 为 `0.02503` quarter，baseline 为 `0.04083` quarter；12 例中 10 例 new 更低。new matched duration error 相对 baseline 略高，是另一件事：baseline 的 legacy uniform grid 偶然把一部分 matched duration 更接近 reference，而 MuseScore 保留了 performance 长度、tuplet 和声部结构；该 matched-only 均值还受两条链路匹配覆盖不同影响。固定总量 FN/FP penalty 后，new `1.395354` 优于 baseline `1.550178`，所以没有证据支持修改生产链去追逐 matched-only duration 均值。

固定 profile 候选的无真值矩阵见 `.artifacts/review/musescore-profile-ablation-v1/matrix-summary.md`，MuseScore 3.6.2 兼容矩阵也给出相同 source-preservation 结论：

- `RecognizePickupBar=false` 在代表例中是 no-op，不能修复 special-6-8 的导入原点。
- `HumanPerformance=false` 能把 special-6-8、piano-03、multitrack-01/02 的大时间位移降到约 ±4 tick，但会改变 voice/staff 分配；multitrack chord retention 降为 `2/3`，并引入 3:2 tuplet 或合并声部语义。
- `SplitStaff=false` 能改善部分和弦集合，但通过合并 imported staff 结构实现，违反多声部保真约束。
- 三项组合虽然时间位移较小，却同时改变 staff/voice 表示，不能作为通用生产 profile。

因此当前没有同时满足 source consistency、和弦保留、连音合法性和多声部结构的严格泛化改动，本阶段不修改 production profile 或标准化算法。
