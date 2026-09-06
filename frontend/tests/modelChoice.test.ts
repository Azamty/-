import assert from "node:assert/strict";
import test from "node:test";
import { modelChoiceDisabled, modelChoiceHint } from "../src/modelChoice.ts";

test("restored vocal jobs leave the next-task model selector editable", () => {
  assert.equal(modelChoiceDisabled(false), false);
  assert.match(modelChoiceHint({ label: "快速", id: "htdemucs" }), /下一次新任务/);
  assert.match(modelChoiceHint({ label: "质量优先", id: "htdemucs_ft" }), /htdemucs_ft/);
});

test("model selector locks only while a submission is busy", () => {
  assert.equal(modelChoiceDisabled(true), true);
  assert.match(modelChoiceHint(null), /默认使用快速模型/);
});
