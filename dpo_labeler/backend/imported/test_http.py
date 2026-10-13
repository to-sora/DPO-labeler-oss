import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
from unittest.mock import patch

import yaml

from ..app import DpoLabelerApp
from .handler import ImportedRequestHandler
from .service import ImportedTasks
from .test_support import ImportCase, rows


class ImportedAuthTests(ImportCase):
    reviewer = "mobile-reviewer"

    def setUp(self) -> None:
        super().setUp()
        self.task = self.create()
        dataset = self.root / "datasets"
        dataset.mkdir()
        app = DpoLabelerApp(dataset_root=dataset, state_dir=self.root / "state",
                            invite_token="change-me", image_roots=[self.images])
        self.addCleanup(app.close)
        self.service = ImportedTasks(self.root / "state", app.catalog_service.image_roots)
        handler = type("TestImportedHandler", (ImportedRequestHandler,), {
            "app": app, "imported": self.service,
            "frontend_dir": Path(__file__).resolve().parents[2] / "frontend",
        })
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path: str, body: dict | None = None, cookie: str = "") -> tuple:
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request("GET" if body is None else "POST", quote(path),
                               body=None if body is None else json.dumps(body),
                               headers={"Content-Type": "application/json", "Cookie": cookie})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def login(self, token: str, reviewer: str = "mobile-reviewer") -> tuple:
        return self.request("/api/v1/session/start", {
            "invite_token": token, "reviewer_username": reviewer,
            "client_instance_id": "mobile-browser-instance",
        })

    def test_wrong_and_missing_invite_tokens_are_rejected(self) -> None:
        for token in ("", "wrong-token"):
            with self.subTest(token=token):
                status, _, body = self.login(token)
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(body)["error"], "Invalid invite token")

    def test_imported_reads_images_and_writes_require_a_session(self) -> None:
        task_id = self.task["task_id"]
        root = f"/api/v1/imported/tasks/{task_id}"
        requests = [
            ("/api/v1/imported/tasks", None), (root, None),
            (f"/media/imported/{task_id}/0", None),
            ("/api/v1/imported/tasks", {"yaml": ""}),
            (root + "/comparisons", self.vote(self.task)), (root + "/export", {}),
        ]
        for path, body in requests:
            with self.subTest(path=path, method="GET" if body is None else "POST"):
                self.assertEqual(self.request(path, body)[0], 401)
        self.assertEqual(self.service.get(task_id, reviewer_username=self.reviewer)["comparisons"], 0)

    def test_correct_token_opens_existing_tasks_and_allows_review(self) -> None:
        status, headers, _ = self.login("change-me")
        self.assertEqual(status, 200)
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertNotIn("Secure", headers["Set-Cookie"])
        status, _, body = self.request("/api/v1/imported/tasks", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"]["tasks"][0]["task_id"], self.task["task_id"])
        task_id = self.task["task_id"]
        status, _, body = self.request(f"/media/imported/{task_id}/0", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(body, (self.images / "0.png").read_bytes())
        status, _, _ = self.request(f"/api/v1/imported/tasks/{task_id}/comparisons",
                                    self.vote(self.task), cookie)
        self.assertEqual(status, 200)
        self.assertEqual(self.service.get(task_id, reviewer_username=self.reviewer)["comparisons"], 1)
        self.assertEqual(self.request("/api/v1/imported/tasks", cookie=cookie + "invalid")[0], 401)

    def test_comparisons_use_session_reviewer_and_ignore_body_identity_on_retry(self) -> None:
        _, headers, _ = self.login("change-me")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        task_id = self.task["task_id"]
        route = f"/api/v1/imported/tasks/{task_id}/comparisons"
        payload = self.vote(self.task)
        status, _, body = self.request(route, {**payload, "reviewer_username": "someone-else"}, cookie)
        self.assertEqual(status, 200)
        result = json.loads(body)["data"]
        self.assertFalse(result["replayed"])
        self.assertEqual({event["reviewer_username"] for event in result["events"]}, {"mobile-reviewer"})
        status, _, body = self.request(route, payload, cookie)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"], {**result, "replayed": True})

        _, headers, _ = self.request("/api/v1/session/start", {
            "invite_token": "change-me", "reviewer_username": "someone-else",
            "client_instance_id": "other-browser",
        })
        other_cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertEqual(self.request(route, {**payload, "reviewer_username": "mobile-reviewer"},
                                      other_cookie)[0], 409)
        next_task = self.service.get(task_id, reviewer_username=self.reviewer)
        self.assertEqual(next_task["comparisons"], 1)
        status, _, body = self.request(route, self.vote(next_task), cookie)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["data"]["replayed"])
        archive = self.archive(task_id)
        for d in range(len(self.task["dimensions"])):
            events = rows(archive[f"dimension-{d}/label_events.jsonl"])
            self.assertEqual(len(events), 2)
            self.assertEqual({event["reviewer_username"] for event in events}, {"mobile-reviewer"})

    def test_import_api_is_disabled_even_for_signed_in_users(self) -> None:
        _, headers, _ = self.login("change-me")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        secret = self.root / "outside-secret.txt"
        secret.write_text("private fixture", encoding="utf-8")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "0.png").write_bytes((self.images / "0.png").read_bytes())
        for override in ({}, {"image_dir": str(outside)}, {"prompt_dir": str(self.root)},
                         {"image": [{"image-path": "0.png", "prompt-path": "../outside-secret.txt"}]},
                         {"image": [{"image-path": "0.png", "prompt-path": str(secret)}]}):
            with self.subTest(override=override):
                source = yaml.safe_dump({**self.manifest(1), **override})
                status, headers, body = self.request("/api/v1/imported/tasks", {"yaml": source}, cookie)
                self.assertEqual(status, 405)
                self.assertEqual(headers["Allow"], "GET")
                self.assertIn("startup", json.loads(body)["error"])
                self.assertNotIn(b"private fixture", body)
        self.assertEqual(len(self.service.catalog()["tasks"]), 1)

    def test_task_routes_reject_wrong_path_segments(self) -> None:
        _, headers, _ = self.login("change-me")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        task_id = self.task["task_id"]
        root = f"/api/v1/imported/not-tasks/{task_id}"
        for path, payload in ((root, None), (root + "/comparisons", self.vote(self.task)),
                              (root + "/export", {})):
            with self.subTest(path=path):
                self.assertEqual(self.request(path, payload, cookie)[0], 404)

    def test_fallback_pages_include_imported_ui(self) -> None:
        for path in ("/", "/index.html", "/review/example", "/imported/"):
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertIn("text/html", headers["Content-Type"])
                self.assertIn(b'/imported/boot.mjs', body)
                self.assertIn(b'/imported/style.css', body)

    def test_locked_database_returns_json_500_and_recovers(self) -> None:
        _, headers, _ = self.login("change-me")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        route = f"/api/v1/imported/tasks/{self.task['task_id']}"
        connect = sqlite3.connect
        lock = connect(Path(self.task["output_dir"]) / "task.sqlite3")
        try:
            lock.execute("BEGIN IMMEDIATE")
            with patch("dpo_labeler.backend.imported.database.sqlite3.connect",
                       side_effect=lambda path, timeout: connect(path, timeout=0.05)):
                for path, payload in ((route, None), (route + "/comparisons", self.vote(self.task))):
                    status, headers, body = self.request(path, payload, cookie)
                    self.assertEqual(status, 500)
                    self.assertIn("application/json", headers["Content-Type"])
                    self.assertEqual(json.loads(body)["error"], "database is locked")
        finally:
            lock.rollback()
            lock.close()
        self.assertEqual(self.request(route, cookie=cookie)[0], 200)

    def test_unexpected_error_returns_json_500(self) -> None:
        _, headers, _ = self.login("change-me")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        with patch.object(self.service, "get", side_effect=RuntimeError("Unexpected failure")):
            status, _, body = self.request(f"/api/v1/imported/tasks/{self.task['task_id']}", cookie=cookie)
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"], "Unexpected failure")

    def test_two_reviewers_can_save_pairs_loaded_before_either_submission(self) -> None:
        route = f"/api/v1/imported/tasks/{self.task['task_id']}"
        cookies, votes = [], []
        for reviewer, decision in (("alice", "a_good"), ("bob", "b_good")):
            _, headers, _ = self.login("change-me", reviewer=reviewer)
            cookie = headers["Set-Cookie"].split(";", 1)[0]
            cookies.append(cookie)
            status, _, body = self.request(route, cookie=cookie)
            self.assertEqual(status, 200)
            task = json.loads(body)["data"]
            votes.append({**self.vote(task), "choices": dict.fromkeys(task["dimensions"], decision)})
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda i: self.request(route + "/comparisons", votes[i], cookies[i]), range(2)))
        for reviewer, decision, cookie, (status, _, body) in zip(
                ("alice", "bob"), ("a_good", "b_good"), cookies, replies):
            self.assertEqual(status, 200)
            events = json.loads(body)["data"]["events"]
            self.assertEqual({(e["reviewer_username"], e["decision"]) for e in events}, {(reviewer, decision)})
            status, _, body = self.request(route, cookie=cookie)
            self.assertEqual(json.loads(body)["data"]["comparisons"], 1)
