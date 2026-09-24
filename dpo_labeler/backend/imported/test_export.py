import json

from .test_support import ImportCase, rows


class ExportTests(ImportCase):
    def test_zero_and_partial_export_keep_all_dimensions_and_distinct_prompts(self) -> None:
        task = self.create()
        empty = self.archive(task["task_id"])
        self.assertEqual(empty["dimension-0/dpo_pairs.jsonl"], "")
        estimate = json.loads(empty["dimension-0/ranking.json"])
        self.assertFalse(estimate["certified"])
        self.assertEqual([len(g) for g in estimate["groups"]], [2] * 5)
        self.service.submit(task["task_id"], self.vote(task))
        partial = self.archive(task["task_id"])
        dataset_ids, comparisons = set(), set()
        for d in range(5):
            event, = rows(partial[f"dimension-{d}/label_events.jsonl"])
            pair, = rows(partial[f"dimension-{d}/dpo_pairs.jsonl"])
            dataset_ids.add(event["dataset_id"])
            comparisons.add(event["comparison_id"])
            self.assertEqual(event["chosen_image_indices"], [0])
            self.assertFalse(pair["strict_dpo"])
            self.assertEqual(pair["chosen"]["prompt"], "prompt 0")
            self.assertEqual(pair["rejected"]["prompt"], "prompt 1")
        self.assertEqual(len(dataset_ids), 5)
        self.assertEqual(len(comparisons), 1)
        one = self.archive(task["task_id"], 2)
        self.assertIn("dimension-2/dpo_pairs.jsonl", one)
        self.assertNotIn("dimension-0/dpo_pairs.jsonl", one)

    def test_exported_events_remain_independent_in_existing_label_store(self) -> None:
        from ..labels import LabelStore
        task = self.create()
        self.service.submit(task["task_id"], self.vote(task))
        archive = self.archive(task["task_id"])
        labels = self.root / "compatible-labels"
        labels.mkdir()
        (labels / "label_events.jsonl").write_text("".join(
            archive[f"dimension-{d}/label_events.jsonl"] for d in range(5)), encoding="utf-8")
        self.assertEqual(len(LabelStore(labels).get_latest_by_pair_key()), 5)

    def test_partial_cutoffs_include_all_image_and_prompt_references(self) -> None:
        task = self.create(23, 1)
        self.service.submit(task["task_id"], self.vote(task))
        ranking = json.loads(self.archive(task["task_id"])["dimension-0/ranking.json"])
        self.assertEqual(len(ranking["images"]), 23)
        self.assertEqual(ranking["tolerance_ranks"], 1)
        self.assertEqual([len(v["image_ids"]) for v in ranking["cutoffs"].values()], [4, 9, 13, 18])
