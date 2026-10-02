import random

from .test_support import ImportCase


class RankingTests(ImportCase):
    def test_precheck_finishes_monotone_dimensions_in_n_minus_one(self) -> None:
        task = self.create()
        while task["pair"]:
            self.service.submit(task["task_id"], self.vote(task), reviewer_username="reviewer")
            task = self.service.get(task["task_id"])
        self.assertTrue(task["complete"])
        self.assertEqual(task["comparisons"], 9)
        self.assertEqual(max(task["exposure"]), 2)

    def test_shuffled_dimensions_converge_and_satisfy_every_cutoff(self) -> None:
        n = 40
        task = self.create(n)
        orders = [random.Random(d).sample(range(n), n) for d in range(5)]
        saw_locked = False
        while task["pair"]:
            saw_locked |= bool(task["pair"]["locked"])
            self.service.submit(task["task_id"], self.vote(task, orders), reviewer_username="reviewer")
            task = self.service.get(task["task_id"])
            self.assertLessEqual(task["comparisons"], n * (n - 1) // 2)
        self.assertTrue(saw_locked)
        self.assertEqual(sum(task["exposure"]), 2 * task["comparisons"])
        for ranking, order in zip(task["rankings"], orders):
            self.assertTrue(ranking["certified"])
            self.assertEqual(sorted(i for g in ranking["groups"] for i in g), list(range(n)))
            for j in range(1, 5):
                selected = set(ranking["cutoffs"][f"{j / 5:.1f}"]["image_ids"])
                boundary, tolerance = j * n // 5, n // 20
                self.assertEqual(len(selected), boundary)
                self.assertTrue(set(order[:boundary - tolerance]) <= selected)
                self.assertTrue(selected <= set(order[:boundary + tolerance]))

    def test_transitive_dimensions_are_locked_and_recorded_as_inferred(self) -> None:
        task = self.create(10, 2)
        orders = [list(range(10)), [2, 5, 1, 9, 0, 4, 3, 8, 6, 7]]
        while task["pair"] and not task["pair"]["locked"]:
            self.service.submit(task["task_id"], self.vote(task, orders), reviewer_username="reviewer")
            task = self.service.get(task["task_id"])
        self.assertTrue(task["pair"]["locked"])
        payload = self.vote(task, orders)
        for dim in task["pair"]["locked"]:
            del payload["choices"][dim]
        events = self.service.submit(task["task_id"], payload, reviewer_username="reviewer")["events"]
        self.assertTrue(any(e["inferred"] for e in events))
