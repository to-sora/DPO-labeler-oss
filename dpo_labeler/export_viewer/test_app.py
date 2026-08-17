from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dpo_labeler.export_viewer.app import ExportViewerApp, ImportValidationError


def _write_hashed_png(root: Path, name: str, color: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), color=color).save(buffer, format="PNG")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    path = root / name / f"{digest}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _pair_export_row(chosen_path: Path, rejected_path: Path, *, strict_dpo: bool) -> dict[str, object]:
    return {
        "dataset_id": "alpha__collected",
        "session_id": "session-001",
        "task_key": "alpha__collected::task-sha-001",
        "task_name": "portrait_pair",
        "task_yaml_name": "portrait_pair.yaml",
        "workflow_name": "sdxl_ease_lora",
        "primary_ckpt": "sdxl/base.safetensors",
        "strict_dpo": strict_dpo,
        "decision": "a_good",
        "reviewer_username": "reviewer.one",
        "chosen_image": {
            "image_index": 1,
            "image_name": "image_1",
            "saved_path": f"/untrusted/location/{chosen_path.name}",
            "positive_prompt": "blue shirt portrait",
            "negative_prompt": "low quality",
            "ckpt": "sdxl/base.safetensors",
        },
        "rejected_image": {
            "image_index": 0,
            "image_name": "image_0",
            "saved_path": f"/untrusted/location/{rejected_path.name}",
            "positive_prompt": "blue shirt portrait",
            "negative_prompt": "low quality",
            "ckpt": "sdxl/base.safetensors",
        },
        "label": {
            "created_at": "2026-04-02T08:00:00+00:00",
            "reviewer_username": "reviewer.one",
            "decision": "a_good",
            "defects_by_image_index": {"1": ["eyes_off"], "0": []},
            "note": "better anatomy",
        },
    }


def _labels_latest_row(first_path: Path, second_path: Path) -> dict[str, object]:
    return {
        "dataset_id": "beta__collected",
        "dataset_display_name": "beta/collected",
        "task_key": "beta__collected::task-sha-001",
        "session_id": "session-002",
        "task_name": "office_pair",
        "task_yaml_name": "office_pair.yaml",
        "workflow_name": "sdxl_ease_lora",
        "primary_ckpt": "sdxl/base.safetensors",
        "label": {
            "created_at": "2026-04-02T09:00:00+00:00",
            "reviewer_username": "reviewer.two",
            "decision": "b_good",
            "chosen_image_indices": [1],
            "defects_by_image_index": {"0": ["hand_corruption"], "1": []},
            "note": "both acceptable",
        },
        "images": [
            {
                "image_index": 0,
                "image_name": "image_0",
                "saved_path": f"/ignore/me/{first_path.name}",
                "positive_prompt": "woman in office",
                "negative_prompt": "low detail",
                "ckpt": "sdxl/base.safetensors",
            },
            {
                "image_index": 1,
                "image_name": "image_1",
                "saved_path": f"/ignore/me/{second_path.name}",
                "positive_prompt": "woman in office",
                "negative_prompt": "low detail",
                "ckpt": "sdxl/base.safetensors",
            },
        ],
    }


def _analytics_labels_latest_row(
    first_path: Path,
    second_path: Path,
    *,
    session_id: str,
    created_at: str,
    decision: str,
    ckpt_a: str,
    ckpt_b: str,
    reviewer_username: str = "reviewer.analytics",
    defects_by_image_index: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    chosen_by_decision = {
        "a_good": [0],
        "b_good": [1],
        "both_good": [0, 1],
        "both_bad": [],
        "skip": [],
    }
    return {
        "dataset_id": "analytics__merged",
        "dataset_display_name": "analytics/merged",
        "task_key": f"analytics__merged::{session_id}",
        "session_id": session_id,
        "task_name": "analytics_pair",
        "task_yaml_name": "analytics_pair.yaml",
        "workflow_name": "SdxlEaseLoraWorkflow",
        "primary_ckpt": ckpt_a,
        "label": {
            "created_at": created_at,
            "reviewer_username": reviewer_username,
            "decision": decision,
            "chosen_image_indices": chosen_by_decision[decision],
            "defects_by_image_index": defects_by_image_index or {"0": [], "1": []},
            "note": "",
        },
        "images": [
            {
                "image_index": 0,
                "image_name": "image_0",
                "saved_path": f"/ignore/me/{first_path.name}",
                "positive_prompt": "analytics prompt",
                "negative_prompt": "analytics negative",
                "ckpt": ckpt_a,
            },
            {
                "image_index": 1,
                "image_name": "image_1",
                "saved_path": f"/ignore/me/{second_path.name}",
                "positive_prompt": "analytics prompt",
                "negative_prompt": "analytics negative",
                "ckpt": ckpt_b,
            },
        ],
    }


class ExportViewerAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.state_dir = self.root / "state"
        self.image_root = self.root / "images"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pair_export_import_persists_and_reloads(self) -> None:
        chosen = _write_hashed_png(self.image_root, "nested-a", "red")
        rejected = _write_hashed_png(self.image_root, "nested-b", "blue")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        payload = json.dumps(_pair_export_row(chosen, rejected, strict_dpo=False)) + "\n"
        created = app.create_import("preference_pairs.jsonl", payload)

        self.assertEqual(created["import"]["format"], "preference-pairs")
        self.assertEqual(created["import"]["valid_rows"], 1)
        rows = app.get_import_rows(created["import"]["import_id"], 0, 20)
        self.assertEqual(rows["total"], 1)
        self.assertEqual(rows["items"][0]["kind"], "pair_export")
        self.assertTrue(rows["items"][0]["images"][0]["is_good"])
        self.assertFalse(rows["items"][0]["images"][0]["is_bad"])
        self.assertTrue(rows["items"][0]["images"][0]["has_defect"])
        self.assertFalse(rows["items"][0]["images"][1]["is_good"])
        self.assertTrue(rows["items"][0]["images"][1]["is_bad"])

        media_path, mime = app.get_media_path(created["import"]["import_id"], rows["items"][0]["row_id"], "chosen")
        self.assertEqual(media_path, chosen.resolve())
        self.assertTrue(mime.startswith("image/"))

        reloaded = ExportViewerApp(self.state_dir, [self.image_root])
        imports = reloaded.list_imports()["imports"]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["import_id"], created["import"]["import_id"])

    def test_labels_latest_import_supported(self) -> None:
        first = _write_hashed_png(self.image_root, "latest-a", "green")
        second = _write_hashed_png(self.image_root, "latest-b", "yellow")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        payload = json.dumps(_labels_latest_row(first, second)) + "\n"
        created = app.create_import("labels_latest.jsonl", payload)

        self.assertEqual(created["import"]["format"], "labels-latest")
        rows = app.get_import_rows(created["import"]["import_id"], 0, 20)
        self.assertEqual(rows["items"][0]["kind"], "labels_latest")
        self.assertEqual(rows["items"][0]["images"][0]["slot"], "a")
        self.assertEqual(rows["items"][0]["images"][1]["slot"], "b")
        self.assertTrue(rows["items"][0]["images"][0]["has_defect"])
        self.assertTrue(rows["items"][0]["images"][1]["is_good"])
        self.assertTrue(rows["items"][0]["images"][0]["is_bad"])

    def test_import_skips_invalid_rows_but_requires_one_valid(self) -> None:
        chosen = _write_hashed_png(self.image_root, "mixed-a", "purple")
        rejected = _write_hashed_png(self.image_root, "mixed-b", "orange")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        valid = _pair_export_row(chosen, rejected, strict_dpo=True)
        invalid = _pair_export_row(chosen, rejected, strict_dpo=True)
        invalid["chosen_image"]["saved_path"] = "/bad/not-a-sha.png"  # type: ignore[index]

        payload = "\n".join([json.dumps(valid), json.dumps(invalid)]) + "\n"
        created = app.create_import("dpo_pairs.jsonl", payload)

        self.assertEqual(created["import"]["format"], "dpo-pairs")
        self.assertEqual(created["import"]["valid_rows"], 1)
        self.assertEqual(created["import"]["invalid_rows"], 1)
        self.assertEqual(len(created["import"]["warnings"]), 1)

        with self.assertRaises(ImportValidationError):
            app.create_import("bad.jsonl", json.dumps({"not": "supported"}) + "\n")

    def test_page_size_defaults_and_clamps(self) -> None:
        chosen = _write_hashed_png(self.image_root, "page-a", "black")
        rejected = _write_hashed_png(self.image_root, "page-b", "white")
        app = ExportViewerApp(self.state_dir, [self.image_root], default_page_size=3, max_page_size=4)

        lines = []
        for index in range(6):
            row = _pair_export_row(chosen, rejected, strict_dpo=False)
            row["session_id"] = f"session-{index:03d}"
            lines.append(json.dumps(row))
        created = app.create_import("preference_pairs.jsonl", "\n".join(lines) + "\n")

        default_page = app.get_import_rows(created["import"]["import_id"], 0, 0)
        self.assertEqual(default_page["limit"], 3)
        self.assertEqual(len(default_page["items"]), 3)

        capped_page = app.get_import_rows(created["import"]["import_id"], 0, 999)
        self.assertEqual(capped_page["limit"], 4)
        self.assertEqual(len(capped_page["items"]), 4)

        fallback_page = app.get_import_rows(created["import"]["import_id"], "bad-cursor", "bad-limit")
        self.assertEqual(fallback_page["cursor"], 0)
        self.assertEqual(fallback_page["limit"], 3)
        self.assertEqual(len(fallback_page["items"]), 3)

    def test_rejects_non_image_file_even_when_hash_matches_name(self) -> None:
        self.image_root.mkdir(parents=True, exist_ok=True)
        payload = b"definitely not an image"
        digest = hashlib.sha256(payload).hexdigest()
        fake_file = self.image_root / f"{digest}.bin"
        fake_file.write_bytes(payload)

        app = ExportViewerApp(self.state_dir, [self.image_root])
        row = _pair_export_row(fake_file, fake_file, strict_dpo=False)

        with self.assertRaises(ImportValidationError):
            app.create_import("preference_pairs.jsonl", json.dumps(row) + "\n")

    def test_media_request_revalidates_changed_file(self) -> None:
        chosen = _write_hashed_png(self.image_root, "change-a", "pink")
        rejected = _write_hashed_png(self.image_root, "change-b", "cyan")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        created = app.create_import("preference_pairs.jsonl", json.dumps(_pair_export_row(chosen, rejected, strict_dpo=False)) + "\n")
        row = app.get_import_rows(created["import"]["import_id"], 0, 20)["items"][0]

        buffer = io.BytesIO()
        Image.new("RGB", (24, 24), color="black").save(buffer, format="PNG")
        chosen.write_bytes(buffer.getvalue())

        with self.assertRaises(ImportValidationError):
            app.get_media_path(created["import"]["import_id"], row["row_id"], "chosen")

    def test_legacy_persisted_pair_rows_without_state_flags_still_render(self) -> None:
        chosen = _write_hashed_png(self.image_root, "legacy-a", "navy")
        rejected = _write_hashed_png(self.image_root, "legacy-b", "gold")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        created = app.create_import("preference_pairs.jsonl", json.dumps(_pair_export_row(chosen, rejected, strict_dpo=False)) + "\n")
        rows_path = self.state_dir / "imports" / created["import"]["import_id"] / "rows.jsonl"
        legacy_rows = []
        for raw_line in rows_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(raw_line)
            for image in payload["images"]:
                image.pop("is_good", None)
                image.pop("is_bad", None)
                image.pop("has_defect", None)
            legacy_rows.append(payload)
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows), encoding="utf-8")

        reloaded = ExportViewerApp(self.state_dir, [self.image_root])
        rows = reloaded.get_import_rows(created["import"]["import_id"], 0, 20)
        self.assertEqual(len(rows["items"]), 1)
        self.assertTrue(rows["items"][0]["images"][0]["is_good"])
        self.assertFalse(rows["items"][0]["images"][0]["is_bad"])
        self.assertFalse(rows["items"][0]["images"][0]["has_defect"])
        self.assertFalse(rows["items"][0]["images"][1]["is_good"])
        self.assertTrue(rows["items"][0]["images"][1]["is_bad"])

    def test_legacy_persisted_latest_label_rows_without_state_flags_still_render(self) -> None:
        first = _write_hashed_png(self.image_root, "legacy-latest-a", "green")
        second = _write_hashed_png(self.image_root, "legacy-latest-b", "yellow")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        created = app.create_import("labels_latest.jsonl", json.dumps(_labels_latest_row(first, second)) + "\n")
        rows_path = self.state_dir / "imports" / created["import"]["import_id"] / "rows.jsonl"
        legacy_rows = []
        for raw_line in rows_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(raw_line)
            for image in payload["images"]:
                image.pop("is_good", None)
                image.pop("is_bad", None)
                image.pop("has_defect", None)
            legacy_rows.append(payload)
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows), encoding="utf-8")

        reloaded = ExportViewerApp(self.state_dir, [self.image_root])
        rows = reloaded.get_import_rows(created["import"]["import_id"], 0, 20)
        self.assertEqual(len(rows["items"]), 1)
        self.assertTrue(rows["items"][0]["images"][0]["is_bad"])
        self.assertFalse(rows["items"][0]["images"][0]["is_good"])
        self.assertTrue(rows["items"][0]["images"][1]["is_good"])
        self.assertFalse(rows["items"][0]["images"][1]["is_bad"])
        self.assertFalse(rows["items"][0]["images"][0]["has_defect"])

    def test_labels_latest_analytics_cover_timeline_wins_defects_indices_and_heatmaps(self) -> None:
        model_a = _write_hashed_png(self.image_root, "analytics-a", "red")
        model_b = _write_hashed_png(self.image_root, "analytics-b", "blue")
        model_c = _write_hashed_png(self.image_root, "analytics-c", "green")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        rows = [
            _analytics_labels_latest_row(
                model_a,
                model_b,
                session_id="cross-001",
                created_at="2026-04-01T08:00:00+00:00",
                decision="a_good",
                ckpt_a="model_a.safetensors",
                ckpt_b="model_b.safetensors",
                reviewer_username="alice",
            ),
            _analytics_labels_latest_row(
                model_a,
                model_c,
                session_id="cross-002",
                created_at="2026-04-01T12:00:00+00:00",
                decision="both_good",
                ckpt_a="model_a.safetensors",
                ckpt_b="model_c.safetensors",
                reviewer_username="alice",
            ),
            _analytics_labels_latest_row(
                model_b,
                model_c,
                session_id="cross-003",
                created_at="2026-04-02T08:30:00+00:00",
                decision="b_good",
                ckpt_a="model_b.safetensors",
                ckpt_b="model_c.safetensors",
                reviewer_username="bob",
            ),
            _analytics_labels_latest_row(
                model_a,
                model_a,
                session_id="same-001",
                created_at="2026-04-02T10:00:00+00:00",
                decision="both_good",
                ckpt_a="model_a.safetensors",
                ckpt_b="model_a.safetensors",
                reviewer_username="alice",
            ),
            _analytics_labels_latest_row(
                model_b,
                model_b,
                session_id="same-002",
                created_at="2026-04-03T09:00:00+00:00",
                decision="both_bad",
                ckpt_a="model_b.safetensors",
                ckpt_b="model_b.safetensors",
                reviewer_username="bob",
                defects_by_image_index={"0": ["hand_corruption"], "1": []},
            ),
            _analytics_labels_latest_row(
                model_a,
                model_b,
                session_id="cross-004",
                created_at="2026-04-03T18:00:00+00:00",
                decision="a_good",
                ckpt_a="model_a.safetensors",
                ckpt_b="model_b.safetensors",
                reviewer_username="alice",
                defects_by_image_index={"0": [], "1": ["bad_crop_framing"]},
            ),
        ]
        created = app.create_import("labels_latest.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")

        analytics = app.get_import_analytics(created["import"]["import_id"])

        self.assertEqual(analytics["summary"]["format"], "labels-latest")
        self.assertEqual(analytics["summary"]["row_count"], 6)

        reviewer_timeline_rows = {
            (row["date"], row["series_label"]): row["count"]
            for row in analytics["tables"]["table0"]["rows"]
        }
        self.assertEqual(reviewer_timeline_rows[("2026-04-01", "alice")], 2)
        self.assertEqual(reviewer_timeline_rows[("2026-04-02", "alice")], 1)
        self.assertEqual(reviewer_timeline_rows[("2026-04-02", "bob")], 1)
        self.assertEqual(reviewer_timeline_rows[("2026-04-03", "bob")], 1)

        timeline_rows = {
            (row["date"], row["series_key"]): row["count"]
            for row in analytics["tables"]["table1"]["rows"]
        }
        self.assertEqual(timeline_rows[("2026-04-01", "a_good")], 1)
        self.assertEqual(timeline_rows[("2026-04-01", "both_good")], 1)
        self.assertEqual(timeline_rows[("2026-04-03", "both_bad")], 1)

        win_rows = {
            row["label"]: row
            for row in analytics["tables"]["table2"]["rows"]
        }
        self.assertEqual(win_rows["model_a.safetensors"]["value"], 2)
        self.assertEqual(win_rows["model_c.safetensors"]["value"], 1)
        self.assertEqual(win_rows["model_b.safetensors"]["value"], 0)

        win_rate_rows = {
            row["label"]: row
            for row in analytics["tables"]["table3"]["rows"]
        }
        self.assertAlmostEqual(win_rate_rows["model_a.safetensors"]["value"], 2 / 3)
        self.assertAlmostEqual(win_rate_rows["model_c.safetensors"]["value"], 0.5)
        self.assertAlmostEqual(win_rate_rows["model_b.safetensors"]["value"], 0.0)

        defect_rows = {
            row["label"]: row
            for row in analytics["tables"]["table5"]["rows"]
        }
        self.assertAlmostEqual(defect_rows["model_b.safetensors"]["value"], 0.4)
        self.assertEqual(defect_rows["model_a.safetensors"]["count"], 0)

        index1_rows = {
            row["label"]: row
            for row in analytics["tables"]["table6"]["rows"]
        }
        self.assertEqual(index1_rows["model_a.safetensors"]["value"], 1)
        self.assertEqual(index1_rows["model_b.safetensors"]["value"], 0)

        index2_rows = {
            row["label"]: row
            for row in analytics["tables"]["table8"]["rows"]
        }
        self.assertEqual(index2_rows["model_b.safetensors"]["value"], 1)

        table10_pairs = {
            (row["row_label"], row["column_label"]): row["count"]
            for row in analytics["tables"]["table10"]["rows"]
        }
        self.assertEqual(table10_pairs[("model_a.safetensors", "model_b.safetensors")], 1)
        self.assertEqual(table10_pairs[("model_c.safetensors", "model_b.safetensors")], 1)

        table11_pairs = {
            (row["row_label"], row["column_label"]): row["value"]
            for row in analytics["tables"]["table11"]["rows"]
        }
        self.assertAlmostEqual(table11_pairs[("model_a.safetensors", "model_b.safetensors")], 0.5)
        self.assertAlmostEqual(table11_pairs[("model_c.safetensors", "model_b.safetensors")], 1.0)

    def test_pair_export_analytics_mark_same_checkpoint_indices_unavailable(self) -> None:
        chosen = _write_hashed_png(self.image_root, "analytics-pair-a", "purple")
        rejected = _write_hashed_png(self.image_root, "analytics-pair-b", "orange")
        app = ExportViewerApp(self.state_dir, [self.image_root])

        row = _pair_export_row(chosen, rejected, strict_dpo=False)
        row["chosen_image"]["ckpt"] = "model_a.safetensors"  # type: ignore[index]
        row["rejected_image"]["ckpt"] = "model_b.safetensors"  # type: ignore[index]
        created = app.create_import("preference_pairs.jsonl", json.dumps(row) + "\n")

        analytics = app.get_import_analytics(created["import"]["import_id"])

        self.assertFalse(analytics["tables"]["table6"]["available"])
        self.assertFalse(analytics["tables"]["table9"]["available"])
        self.assertTrue(analytics["tables"]["table10"]["available"])
        self.assertEqual(analytics["tables"]["table10"]["matrix_rows"][0]["values"][1]["value"], 1)


if __name__ == "__main__":
    unittest.main()
