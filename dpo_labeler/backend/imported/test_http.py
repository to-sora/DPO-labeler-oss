import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import yaml

from ..app import DpoLabelerApp
from .handler import ImportedRequestHandler
from .service import ImportedTasks
from .test_support import ImportCase, rows


class ImportedAuthTests(ImportCase):
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

    def login(self, token: str) -> tuple:
        return self.request("/api/v1/session/start", {
            "invite_token": token, "reviewer_username": "mobile-reviewer",
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
        self.assertEqual(self.service.get(task_id)["comparisons"], 0)

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
        self.assertEqual(self.service.get(task_id)["comparisons"], 1)
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
        next_task = self.service.get(task_id)
        self.assertEqual(next_task["comparisons"], 1)
        status, _, body = self.request(route, self.vote(next_task), cookie)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["data"]["replayed"])
        archive = self.archive(task_id)
        for d in range(len(self.task["dimensions"])):
            events = rows(archive[f"dimension-{d}/label_events.jsonl"])
            self.assertEqual(len(events), 2)
            self.assertEqual({event["reviewer_username"] for event in events}, {"mobile-reviewer"})

    def test_import_api_rejects_unapproved_directories_and_prompt_escape(self) -> None:
        _, headers, _ = self.login("change-me")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        secret = self.root / "outside-secret.txt"
        secret.write_text("private fixture", encoding="utf-8")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "0.png").write_bytes((self.images / "0.png").read_bytes())
        for override in ({"image_dir": str(outside)}, {"prompt_dir": str(self.root)},
                         {"image": [{"image-path": "0.png", "prompt-path": "../outside-secret.txt"}]},
                         {"image": [{"image-path": "0.png", "prompt-path": str(secret)}]}):
            with self.subTest(override=override):
                source = yaml.safe_dump({**self.manifest(1), **override})
                status, _, body = self.request("/api/v1/imported/tasks", {"yaml": source}, cookie)
                self.assertEqual(status, 400)
                self.assertNotIn(b"private fixture", body)
        self.assertEqual(len(self.service.catalog()["tasks"]), 1)
