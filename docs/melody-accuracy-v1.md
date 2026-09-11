# 主旋律选择 v1 分支说明

本分支从 `08bf188` 创建：`codex/melody-accuracy-v1`。它只改主旋律候选选择的后端实验，保留原始 recognition、全和声音符和旧高精度产物；不重跑 MuScriptor、BeatNet 或 MuseScore，也不把诊断结果称为完整简谱验收。

## 阶段 1：selector

`backend/jianpu_score/quantize.py` 增加 `onset-dp-v1`。它先按小的起音容差形成候选组，再用动态规划选择一条单音路径。候选和转移都写入 audit：缺失 confidence 保持中性，`metadata.playback_default` 不参与评分；重叠尾音用软惩罚处理，跳过的起音组也会记录。旧 `select_voice_events(..., mode="polyphonic")` 行为保持不变。

## 阶段 2：V2 合并入口

V2 的 `merge_main_melody=true` 现在从用户选中的全部有音高轨构造候选，即使某个独立乐器分谱转换失败，也会独立尝试主旋律。`main-melody.selection.json` 保存 source index、候选 emission、跳过的 onset 组和转移分数；原始 recognition、全和声 notes、独立分谱失败 manifest 都继续保留。

若独立分谱全部失败但主旋律成功，任务保持 completed、`score_refusal=null`，并在 `track_failures` 留下分谱失败；若主旋律也失败，仍返回显式 `all_pitched_tracks_failed`。此阶段不增加网页引擎开关。

两阶段均可以单独回退到父分支：

```text
git switch codex/high-accuracy-transcription
```

阶段提交分开保留，便于逐段比较旧的 onset 最高音策略、新 selector 和最终产物；回退后旧 V2 合并行为恢复。
