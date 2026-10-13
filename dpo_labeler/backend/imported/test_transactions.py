from concurrent.futures import ThreadPoolExecutor

from .database import read, transaction, write
from .events import Conflict
from .reviewers import review_state
from .service import ImportedTasks
from .test_support import ImportCase, rows


class TransactionTests(ImportCase):
    def test_reviewers_save_opposite_choices_without_invalidating_each_other(self) -> None:
        first = self.create()
        task_id = first["task_id"]
        second = self.service.get(task_id, reviewer_username="second-reviewer")
        self.assertEqual(first["pair"]["image_ids"], second["pair"]["image_ids"])
        self.assertNotEqual(first["pair"]["comparison_id"], second["pair"]["comparison_id"])
        votes = [self.vote(first), {**self.vote(second), "choices": dict.fromkeys(second["dimensions"], "b_good")}]
        users = [self.reviewer, "second-reviewer"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda i: self.service.submit(
                task_id, votes[i], reviewer_username=users[i]), range(2)))
        self.assertTrue(all(not result["replayed"] for result in results))
        for user, decision, payload in zip(users, ["a_good", "b_good"], votes):
            current = self.service.get(task_id, reviewer_username=user)
            self.assertEqual(current["comparisons"], 1)
            self.assertEqual(sum(current["exposure"]), 2)
            self.assertTrue(self.service.submit(task_id, payload, reviewer_username=user)["replayed"])
            events = rows(self.archive(task_id, reviewer=user)["dimension-0/label_events.jsonl"])
            self.assertEqual([(e["reviewer_username"], e["decision"]) for e in events], [(user, decision)])
        with self.assertRaises(Conflict):
            self.service.submit(task_id, votes[0], reviewer_username=users[1])

    def test_legacy_progress_and_events_are_preserved_when_reviewers_separate(self) -> None:
        task = self.create()
        task_id = task["task_id"]
        payload = self.vote(task)
        saved = self.service.submit(task_id, payload, reviewer_username=self.reviewer)
        database = self.service.root / task_id / "task.sqlite3"
        with transaction(database) as db:
            baseline = review_state(db, self.reviewer)
            write(db, "state", baseline)
            db.execute("DROP TABLE reviewers")
            db.execute("DELETE FROM meta WHERE key='legacy_batch_limit'")
        for user, decision in (("alice", "a_good"), ("bob", "b_good")):
            current = self.service.get(task_id, reviewer_username=user)
            self.assertEqual(current["comparisons"], 1)
            vote = {**self.vote(current), "choices": dict.fromkeys(current["dimensions"], decision)}
            self.service.submit(task_id, vote, reviewer_username=user)
            events = rows(self.archive(task_id, reviewer=user)["dimension-0/label_events.jsonl"])
            self.assertEqual(events[0], saved["events"][0])
            self.assertEqual(events[1]["reviewer_username"], user)
            self.assertEqual(len(events), 2)
        replay = self.service.submit(task_id, payload, reviewer_username=self.reviewer)
        self.assertEqual(replay, {**saved, "replayed": True})
        with transaction(database) as db:
            self.assertEqual(read(db, "state"), baseline)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 3)

    def test_incomplete_vote_is_atomic_and_retry_survives_restart(self) -> None:
        task = self.create()
        payload = self.vote(task)
        incomplete = {**payload, "choices": {"dim-0": "a_good"}}
        with self.assertRaises(ValueError):
            self.service.submit(task["task_id"], incomplete, reviewer_username="reviewer")
        self.assertEqual(self.service.get(task["task_id"], reviewer_username=self.reviewer)["comparisons"], 0)
        result = self.service.submit(task["task_id"], payload, reviewer_username="reviewer")
        self.service = ImportedTasks(self.root / "state", [self.images])
        replay = self.service.submit(task["task_id"], payload, reviewer_username="reviewer")
        self.assertEqual(result["events"], replay["events"])
        self.assertTrue(replay["replayed"])
        state = self.service.get(task["task_id"], reviewer_username=self.reviewer)
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
            pairs = list(pool.map(lambda _: reopened.get(task["task_id"], reviewer_username=self.reviewer)["pair"], range(4)))
        self.assertEqual(pairs, [task["pair"]] * 4)
        self.assertEqual(reopened.get(task["task_id"], reviewer_username=self.reviewer)["comparisons"], 0)
