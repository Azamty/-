# 主旋律选择 v1 分支说明

本分支从 `08bf188` 创建：`codex/melody-accuracy-v1`。它只改主旋律候选选择的后端实验，保留原始 recognition、全和声音符和旧高精度产物；不重跑 MuScriptor、BeatNet 或 MuseScore，也不把诊断结果称为完整简谱验收。

## 阶段 1：selector

`backend/jianpu_score/quantize.py` 增加 `onset-dp-v1`。它先按小的起音容差形成候选组，再用动态规划选择一条单音路径。候选和转移都写入 audit：缺失 confidence 保持中性，`metadata.playback_default` 不参与评分；重叠尾音用软惩罚处理，跳过的起音组也会记录。旧 `select_voice_events(..., mode="polyphonic")` 行为保持不变。

这一阶段尚未接入 V2 网站合并入口，因此可以单独回退到父分支：

```text
git switch codex/high-accuracy-transcription
```

阶段提交只包含 selector 和 fixture 回归；后续接入 V2 时会另行提交，便于逐段比较旧的 onset 最高音策略、新 selector 和最终产物。
