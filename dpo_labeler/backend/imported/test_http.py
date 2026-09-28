import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from ..app import DpoLabelerApp
from .handler import ImportedRequestHandler
from .test_support import ImportCase


class ImportedAuthTests(ImportCase):
    def setUp(self) -> None:
        super().setUp()
        self.task = self.create()
        dataset = self.root / "datasets"
        dataset.mkdir()
        app = DpoLabelerApp(dataset_root=dataset, state_dir=self.root / "state",
                            invite_token="change-me")
        self.addCleanup(app.close)
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
