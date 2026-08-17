import test from "node:test";
import assert from "node:assert/strict";

import {
  applyOptimisticLabel,
  buildImageSummaryRows,
  buildLatestLabelRows,
  buildPairSummaryRows,
  createKeyedValueCache,
  datasetProgress,
  deriveProgressSnapshot,
  deriveTaskDisplayLabels,
  mergeQueueItems,
} from "../../dpo_labeler/frontend/app_helpers.mjs";

test("mergeQueueItems updates existing items and appends new queue entries", () => {
  const existing = [
    { dataset_id: "set", session_id: "alpha", is_labeled: false, latest_decision: null },
  ];
  const incoming = [
    { dataset_id: "set", session_id: "alpha", is_labeled: true, latest_decision: "a_good" },
    { dataset_id: "set", session_id: "beta", is_labeled: false, latest_decision: null },
  ];

  const merged = mergeQueueItems(existing, incoming);

  assert.equal(merged.length, 2);
  assert.equal(merged[0].latest_decision, "a_good");
  assert.equal(merged[1].session_id, "beta");
});

test("applyOptimisticLabel advances the known labeled state", () => {
  const queue = [
    { dataset_id: "set", session_id: "alpha", is_labeled: false, latest_decision: null },
    { dataset_id: "set", session_id: "beta", is_labeled: false, latest_decision: null },
  ];

  const updated = applyOptimisticLabel(queue, {
    dataset_id: "set",
    session_id: "alpha",
    decision: "b_good",
  });
  const progress = deriveProgressSnapshot({
    items: updated,
    totalPairs: 2,
    labeledPairs: 0,
    firstUnlabeledIndex: 0,
  });

  assert.equal(updated[0].is_labeled, true);
  assert.equal(updated[0].latest_decision, "b_good");
  assert.equal(progress.labeledPairs, 1);
  assert.equal(progress.firstUnlabeledIndex, 1);
});

test("deriveProgressSnapshot reports queue complete when all pairs are labeled", () => {
  const progress = deriveProgressSnapshot({
    items: [
      { dataset_id: "set", session_id: "alpha", is_labeled: true },
      { dataset_id: "set", session_id: "beta", is_labeled: true },
    ],
    totalPairs: 2,
    labeledPairs: 2,
    firstUnlabeledIndex: 2,
  });

  assert.equal(progress.queueComplete, true);
  assert.equal(progress.firstUnlabeledIndex, 2);
});

test("readable formatting helpers produce human-facing labels", () => {
  const pairRows = buildPairSummaryRows({
    dataset_display_name: "Dataset One",
    session_id: "alpha",
    task_name: "pair_task",
    workflow_name: "WorkflowA",
    primary_ckpt: "model.safetensors",
    global_seed: 42,
  });
  const imageRows = buildImageSummaryRows({
    image_name: "A image",
    seed: 9,
    width: 1024,
    height: 768,
    steps: 30,
    cfg: 7,
    ckpt: "model.safetensors",
    status: "success",
  });
  const labelRows = buildLatestLabelRows({
    decision: "a_good",
    defects_a: ["hand_corruption"],
    defects_b: [],
    note: "Cleaner hands",
    created_at: "2026-03-27T12:34:56+00:00",
  });

  assert.deepEqual(pairRows, []);
  assert.deepEqual(imageRows, [
    { label: "Size", value: "1024 x 768" },
    { label: "Steps / CFG", value: "30 / 7" },
  ]);
  assert.deepEqual(labelRows[0], { label: "Decision", value: "A preferred" });
  assert.equal(labelRows[1].value, "Corrupt hands");
  assert.equal(labelRows[4].value, "2026-03-27 12:34:56 UTC");
});

test("task display labels collapse redundant yaml stem and task name independently", () => {
  const labels = deriveTaskDisplayLabels(
    "batch/tasks/dpo_workflow__all__same_base_prompt__no_upscale__both_no_lora__mix_qwen__different_model_pair__seed_2025/collected",
    {
      task_yaml_name: "all__same_base_prompt__no_upscale__both_no_lora__mix_qwen__different_model_pair.yaml",
      task_name: "all__same_base_prompt__no_upscale__both_no_lora__mix_qwen__different_model_pair",
    },
  );

  assert.deepEqual(labels, { title: "default", subtitle: "default" });
});

test("dataset progress uses reviewed pairs over valid total pairs", () => {
  const progress = datasetProgress({
    tasks: [
      { reviewed_pairs: 3, total_pairs: 46, invalid_pair_count: 4 },
      { reviewed_pairs: 2, total_pairs: 4, invalid_pair_count: 9 },
    ],
  });

  assert.deepEqual(progress, {
    reviewedPairs: 5,
    totalPairs: 50,
    reviewedPercent: 10,
  });
});

test("createKeyedValueCache stores entries by key and evicts oldest by weight", async () => {
  const cache = createKeyedValueCache({
    maxWeight: 4,
    getWeight: (entry) => entry.weight,
  });

  cache.set("alpha", { weight: 2, label: "a" });
  cache.set("beta", { weight: 2, label: "b" });
  cache.touch("alpha");
  cache.set("gamma", { weight: 2, label: "c" });

  const removed = cache.evictToFit({ protectKeys: ["gamma"] });

  assert.deepEqual(removed.map((entry) => entry.key), ["beta"]);
  assert.equal(cache.has("alpha"), true);
  assert.equal(cache.has("beta"), false);
  assert.equal(cache.peek("gamma").label, "c");
});
