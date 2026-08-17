from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

from dpo_labeler.export_viewer.app import ExportViewerApp
from dpo_labeler.export_viewer.server import ExportViewerRequestHandler, _resolve_frontend_asset_path


def _write_hashed_png(root: Path, color: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), color=color).save(buffer, format="PNG")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "deep" / f"{digest}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _pair_export_payload(chosen: Path, rejected: Path) -> str:
    return json.dumps(
        {
            "dataset_id": "alpha__collected",
            "session_id": "session-http",
            "task_key": "alpha__collected::task-sha-http",
            "task_name": "http_pair",
            "task_yaml_name": "http_pair.yaml",
            "workflow_name": "sdxl_ease_lora",
            "primary_ckpt": "sdxl/base.safetensors",
            "strict_dpo": False,
            "decision": "b_good",
            "reviewer_username": "reviewer.http",
            "chosen_image": {
                "image_index": 1,
                "image_name": "image_1",
                "saved_path": f"/foreign/root/{chosen.name}",
                "positive_prompt": "studio portrait",
                "negative_prompt": "low quality",
                "ckpt": "sdxl/base.safetensors",
            },
            "rejected_image": {
                "image_index": 0,
                "image_name": "image_0",
                "saved_path": f"/foreign/root/{rejected.name}",
                "positive_prompt": "studio portrait",
                "negative_prompt": "low quality",
                "ckpt": "sdxl/base.safetensors",
            },
            "label": {
                "created_at": "2026-04-02T08:00:00+00:00",
                "reviewer_username": "reviewer.http",
                "decision": "b_good",
                "defects_by_image_index": {"1": ["bad_crop_framing"], "0": []},
                "note": "better lighting",
            },
        }
    ) + "\n"


class ExportViewerServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.state_dir = self.root / "state"
        self.image_root = self.root / "images"
        self.chosen = _write_hashed_png(self.image_root, "red")
        self.rejected = _write_hashed_png(self.image_root, "blue")

        self.app = ExportViewerApp(self.state_dir, [self.image_root])
        frontend_dir = Path("dpo_labeler/export_viewer/frontend").resolve()
        handler = type(
            "ConfiguredExportViewerRequestHandler",
            (ExportViewerRequestHandler,),
            {"app": self.app, "frontend_dir": frontend_dir},
        )
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except PermissionError as exc:
            self.skipTest(f"socket bind not permitted in this environment: {exc}")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def test_http_import_listing_rows_and_media(self) -> None:
        with urlopen(f"{self.base_url}/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("Export Viewer", html)

        config = self._get_json("/api/v1/config")
        self.assertTrue(config["ok"])
        self.assertIn("supported_import_formats", config["data"])
        self.assertEqual(config["data"]["default_page_size"], 10)
        self.assertEqual(config["data"]["max_page_size"], 50)

        request = Request(
            f"{self.base_url}/api/v1/imports",
            data=json.dumps({"filename": "preference_pairs.jsonl", "text": _pair_export_payload(self.chosen, self.rejected)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            created = json.loads(response.read().decode("utf-8"))
        self.assertTrue(created["ok"])
        import_id = created["data"]["import"]["import_id"]

        listing = self._get_json("/api/v1/imports")
        self.assertEqual(len(listing["data"]["imports"]), 1)

        rows = self._get_json(f"/api/v1/imports/{import_id}/rows?cursor=0&limit=10")
        self.assertEqual(rows["data"]["total"], 1)
        self.assertEqual(rows["data"]["limit"], 10)
        self.assertTrue(rows["data"]["items"][0]["images"][0]["is_good"])
        self.assertTrue(rows["data"]["items"][0]["images"][0]["has_defect"])
        self.assertTrue(rows["data"]["items"][0]["images"][1]["is_bad"])
        media_url = rows["data"]["items"][0]["images"][0]["media_url"]

        analytics = self._get_json(f"/api/v1/imports/{import_id}/analytics")
        self.assertTrue(analytics["ok"])
        self.assertEqual(analytics["data"]["summary"]["format"], "preference-pairs")
        self.assertFalse(analytics["data"]["tables"]["table6"]["available"])
        self.assertTrue(analytics["data"]["tables"]["table10"]["available"])

        with urlopen(f"{self.base_url}{media_url}") as response:
            media_bytes = response.read()
            content_type = response.headers.get("Content-Type")
        self.assertTrue(content_type.startswith("image/"))
        self.assertGreater(len(media_bytes), 0)

        delete_request = Request(f"{self.base_url}/api/v1/imports/{import_id}", method="DELETE")
        with urlopen(delete_request) as response:
            deleted = json.loads(response.read().decode("utf-8"))
        self.assertTrue(deleted["ok"])

    def _get_json(self, path: str) -> dict:
        with urlopen(f"{self.base_url}{path}") as response:
            return json.loads(response.read().decode("utf-8"))


class ExportViewerFrontendPathTests(unittest.TestCase):
    def test_resolve_frontend_asset_path_rejects_sibling_prefix_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontend_dir = root / "frontend"
            sibling_dir = root / "frontend-evil"
            frontend_dir.mkdir(parents=True, exist_ok=True)
            sibling_dir.mkdir(parents=True, exist_ok=True)
            (frontend_dir / "index.html").write_text("INDEX", encoding="utf-8")
            (sibling_dir / "secret.txt").write_text("SECRET", encoding="utf-8")

            resolved = _resolve_frontend_asset_path(frontend_dir, "/../frontend-evil/secret.txt")

            self.assertEqual(resolved, (frontend_dir / "index.html").resolve())
            self.assertEqual(resolved.read_text(encoding="utf-8"), "INDEX")


if __name__ == "__main__":
    unittest.main()
