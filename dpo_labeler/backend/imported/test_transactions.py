from concurrent.futures import ThreadPoolExecutor

from .events import Conflict
from .service import ImportedTasks
from .test_support import ImportCase, rows


class TransactionTests(ImportCase):
    def test_incomplete_vote_is_atomic_and_retry_survives_restart(self) -> None:
        task = self.create()
        payload = self.vote(task)
        incomplete = {**payload, "choices": {"dim-0": "a_good"}}
        with self.assertRaises(ValueError):
            self.service.submit(task["task_id"], incomplete, reviewer_username="reviewer")
        self.assertEqual(self.service.get(task["task_id"])["comparisons"], 0)
        result = self.service.submit(task["task_id"], payload, reviewer_username="reviewer")
        self.service = ImportedTasks(self.root / "state", [self.images])
        replay = self.service.submit(task["task_id"], payload, reviewer_username="reviewer")
        self.assertEqual(result["events"], replay["events"])
        self.assertTrue(replay["replayed"])
        state = self.service.get(task["task_id"])
        self.assertEqual(sum(state["exposure"]), 2)
        self.assertEqual(state["comparisons"], 1)
        self.assertNotEqual(state["pair"]["comparison_id"], task["pair"]["comparison_id"])

    def test_simultaneous_browser_submissions_count_once(self) -> None:
        task = self.create()
        payload = self.vote(task)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.service.submit(
                task["task_id"], payload, reviewer_username="reviewer"), range(8)))
        self.assertEqual(sum(not r["replayed"] for r in results), 1)
        archive = self.archive(task["task_id"])
        for d in range(5):
            self.assertEqual(len(rows(archive[f"dimension-{d}/label_events.jsonl"])), 1)
        different = {**payload, "choices": dict.fromkeys(task["dimensions"], "b_good")}
        with self.assertRaises(Conflict):
            self.service.submit(task["task_id"], different, reviewer_username="reviewer")

    def test_pending_pair_is_shared_across_restarts_and_readers(self) -> None:
        task = self.create()
        reopened = ImportedTasks(self.root / "state", [self.images])
        with ThreadPoolExecutor(max_workers=4) as pool:
            pairs = list(pool.map(lambda _: reopened.get(task["task_id"])["pair"], range(4)))
        self.assertEqual(pairs, [task["pair"]] * 4)
        self.assertEqual(reopened.get(task["task_id"])["comparisons"], 0)
