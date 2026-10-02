Imported exports / 匯入匯出格式 / 导入导出格式
==========================================

English
-------
`task.json`: task/character names, absolute image directory, image IDs,
prompt snapshots, dimensions, accepted comparison count and per-image exposure.
`import.yaml`: original uploaded YAML.

Each `dimension-N/` contains:

- `label_events.jsonl`: every shared comparison's result for this dimension.
  `dataset_id` identifies this task and dimension; `session_id` and
  `comparison_id` identify the shared image pair presentation.
  Pair-local `display_order` is `[0,1]`; `chosen_image_indices` contains 0 or 1.
  `image_ids` maps those slots to the task's stable images.
  `inferred: true` distinguishes automatic results from human choices.
- `dpo_pairs.jsonl`: human choices only, with `chosen` and `rejected` image
  objects carrying their own `image_path` and `prompt`. Different prompts
  are retained; `strict_dpo` is false. Original strict exports are separate.
- `ranking.json`: rank 1 is best. `groups` contains five disjoint image-ID
  lists. `cutoffs` contains cumulative best 0.2, 0.4, 0.6 and 0.8 lists,
  each with exactly floor(fraction × N) images and its own `certified` flag.
  `rank_intervals` records proven lower/upper rank endpoints for every image.
  `images` provides corresponding source paths and saved prompts.

Tolerance is floor(N/20). A certificate guarantees the requested boundary
against every total order consistent with the saved comparisons. It cannot
guarantee that subjective judgments are correct. Uncertified groups use a
deterministic order of interval midpoints; their cutoffs are provisional.
An empty-label task can also export. Original image files are not bundled.

繁體中文
--------
`task.json` 保存任務、圖片參照、提示詞、比較次數及曝光數。
每維資料夾包含事件、人工選擇的 DPO 配對及排名；推論事件另有標記。
排名 1 為最佳。五組互不重疊；四個累積分界分別含向下取整的指定比例。
排名區間與證書旗標表示已知證據；未驗證分界僅為暫時估計。
容錯為圖片數除以 20 後向下取整。保證以前述比較一致為前提。
不同提示詞分別保留在 chosen/rejected 中；ZIP 不包含原始圖片。

简体中文
--------
`task.json` 保存任务、图片引用、提示词、比较次数及曝光数。
每维文件夹包含事件、人工选择的 DPO 配对及排名；推断事件另有标记。
排名 1 为最佳。五组互不重叠；四个累计分界分别含向下取整的指定比例。
排名区间与证书标记表示已知证据；未验证分界仅为暂时估计。
容错为图片数除以 20 后向下取整。保证以前述比较一致为前提。
不同提示词分别保留在 chosen/rejected 中；ZIP 不包含原始图片。
