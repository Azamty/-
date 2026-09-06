export type ActiveModel = {
  label: string;
  id: string;
};

/** The upload selector remains editable for a restored job; only active submits lock it. */
export function modelChoiceDisabled(busy: boolean): boolean {
  return busy;
}

export function modelChoiceHint(activeModel: ActiveModel | null): string {
  if (activeModel) {
    return `当前恢复任务使用 ${activeModel.label}（${activeModel.id}）；此处选择只用于下一次新任务。`;
  }
  return "默认使用快速模型；质量优先模型由 Demucs 官方说明约慢 4 倍，但可能略好。";
}
