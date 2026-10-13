# Imported image exports

See [startup imports and the manifest schema](imported_images.md) for creating a
task. Exports use the signed-in reviewer's progress. The downloaded ZIP keeps the
same `dimension-N/` layout for one selected dimension or all dimensions.

- `task.json`: task and character names, image and prompt directories, stable
  image IDs, saved prompts, dimensions, `reviewer_username`, comparison count,
  and per-image exposure for this reviewer's ranking.
- `import.yaml`: the original server-side manifest.

Each `dimension-N/` contains:

- `label_events.jsonl`: this reviewer's saved events for the dimension, plus any
  inherited shared history from the older implementation. Events retain their
  original reviewer names. `dataset_id` identifies the task and dimension;
  `session_id` and `comparison_id` identify a pair presentation. Pair-local
  `display_order` is `[0,1]`; `chosen_image_indices` contains 0 or 1. `image_ids`
  maps these slots to the task's stable images. `inferred: true` identifies an
  automatic result rather than a human choice.
- `dpo_pairs.jsonl`: human choices only. `chosen` and `rejected` contain their
  respective `image_path` and `prompt`; different prompts are preserved and
  `strict_dpo` is false. Standard labeler exports remain separate.
- `ranking.json`: the reviewer's `reviewer_username`, rank intervals, five
  disjoint image groups, and cumulative best 0.2, 0.4, 0.6 and 0.8 cutoffs. Each
  cutoff contains exactly floor(fraction × N) images and a `certified` flag.
  `images` provides source paths and saved prompts. Rank 1 is best.

Tolerance is floor(N/20). A certificate describes every total order consistent
with the comparisons used by this reviewer's ranking, including inherited shared
history. It does not combine later votes from other reviewers. Uncertified groups
use a deterministic order of interval midpoints; their cutoffs are provisional.
An empty-label task can also export. Original image files are not bundled.

The server saves exports under
`--state-dir/imported/<task-id>/exports/<reviewer-hash>/`. The hash separates
reviewers' files; the ZIP metadata contains the readable username. Comparison
retries do not add duplicate events or increase exposure counts.
