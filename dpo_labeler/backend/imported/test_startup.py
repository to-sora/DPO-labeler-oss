import io
from contextlib import redirect_stderr
from unittest.mock import patch

import yaml

from ..server import main
from .service import ImportedTasks
from .test_support import ImportCase


class StartupTests(ImportCase):
    def test_unchanged_manifest_reuses_progress_and_changes_create_a_new_task(self) -> None:
        path = self.root / "task.yaml"
        manifest = self.manifest()
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        self.service.load_startup([path, path])
        task_id = self.service.catalog()["tasks"][0]["task_id"]
        task = self.service.get(task_id, reviewer_username=self.reviewer)
        self.service.submit(task_id, self.vote(task), reviewer_username="reviewer")
        reopened = ImportedTasks(self.root / "state", [self.images])
        reopened.load_startup([path])
        self.assertEqual(len(reopened.catalog()["tasks"]), 1)
        self.assertEqual(reopened.get(task_id, reviewer_username=self.reviewer)["comparisons"], 1)
        manifest["task-name"] = "Updated task"
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        self.assertEqual(len(reopened.catalog()["tasks"]), 1)
        reopened.load_startup([path])
        self.assertEqual(len(reopened.catalog()["tasks"]), 2)
        self.assertEqual(reopened.get(task_id, reviewer_username=self.reviewer)["comparisons"], 1)

    def arguments(self, path) -> list[str]:
        dataset = self.root / "datasets"
        dataset.mkdir(exist_ok=True)
        return ["--dataset-root", str(dataset), "--state-dir", str(self.root / "state"),
                "--image-root", str(self.images), "--import-yaml", str(path),
                "--host", "127.0.0.1", "--port", "0", "--invite-token", "change-me"]

    def test_server_loads_manifests_before_accepting_requests(self) -> None:
        path = self.root / "task.yaml"
        path.write_text(yaml.safe_dump(self.manifest()), encoding="utf-8")
        second = self.root / "second.yaml"
        second.write_text(yaml.safe_dump(self.manifest(5)), encoding="utf-8")

        def check(server):
            tasks = server.RequestHandlerClass.imported.catalog()["tasks"]
            self.assertEqual(len(tasks), 2)
            self.assertEqual(sorted(task["image_count"] for task in tasks), [5, 10])

        with patch("dpo_labeler.backend.server.ThreadingHTTPServer.serve_forever", check):
            main([*self.arguments(path), "--import-yaml", str(second)])

    def test_invalid_startup_manifest_fails_before_binding(self) -> None:
        path = self.root / "bad.yaml"
        path.write_text("task-name: missing fields", encoding="utf-8")
        with patch("dpo_labeler.backend.server.ThreadingHTTPServer") as server, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(self.arguments(path))
            self.assertEqual(error.exception.code, 2)
            server.assert_not_called()
