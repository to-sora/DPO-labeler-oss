from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dpo_labeler.backend.catalog import PairRecord, SessionImage
from dpo_labeler.backend.common import LabelEventValidationError, resolve_existing_path
from dpo_labeler.backend.labels import LabelStore


def _image(index: int, path: Path) -> SessionImage:
    return SessionImage(
        image_index=index,
        image_name=f"image_{index}",
        saved_path=str(path),
        positive_prompt="prompt",
        negative_prompt="negative",
        ckpt="model.safetensors",
        seed=index,
        status="success",
        workflow_name="test-workflow",
        task_yaml_path="task.yaml",
        prompt_seed=index,
        prompt_seed_control="fixed",
        generation_seed_control="fixed",
        width=512,
        height=512,
        cfg=7.0,
        steps=20,
        runtime_seed_values={},
        lora_stack_config={},
        runner_result={"status": "success"},
        saved_filename=path.name,
        original_filename=path.name,
    )


def _pair(root: Path) -> PairRecord:
    image_a = root / "a.png"
    image_b = root / "b.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    return PairRecord(
        pair_key="dataset::session-1",
        dataset_id="dataset",
        dataset_display_name="dataset",
        task_key="dataset::task",
        task_name="task",
        task_yaml_name="task.yaml",
        task_yaml_path="task.yaml",
        task_yaml_sha256="sha",
        compiler_version="test",
        global_seed=1,
        workflow_name="test-workflow",
        primary_ckpt="model.safetensors",
        session_id="session-1",
        session_index=0,
        images=(_image(0, image_a), _image(1, image_b)),
    )


class SavedPathContainmentTests(unittest.TestCase):
    def test_resolve_existing_path_accepts_separate_allowed_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_root = base / "dataset"
            image_root = base / "images"
            dataset_root.mkdir()
            image_root.mkdir()
            image = image_root / "image.png"
            image.write_bytes(b"image")

            resolved = resolve_existing_path(
                str(image),
                [dataset_root, image_root],
                allowed_roots=[dataset_root, image_root],
            )

            self.assertEqual(resolved, image.resolve())

    def test_resolve_existing_path_accepts_file_inside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            root.mkdir()
            image = root / "image.png"
            image.write_bytes(b"image")

            resolved = resolve_existing_path(
                str(image),
                [root],
                allowed_roots=[root],
            )

            self.assertEqual(resolved, image.resolve())

    def test_resolve_existing_path_rejects_absolute_file_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "dataset"
            root.mkdir()
            outside = base / "secret.txt"
            outside.write_text("secret", encoding="utf-8")

            resolved = resolve_existing_path(
                str(outside),
                [root],
                allowed_roots=[root],
            )

            self.assertIsNone(resolved)

    def test_resolve_existing_path_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            root = base / "dataset"
            root.mkdir()
            outside = base / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            link = root / "image.png"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not available on this platform")

            resolved = resolve_existing_path(
                str(link),
                [root],
                allowed_roots=[root],
            )

            self.assertIsNone(resolved)


class LabelEventIdempotencyTests(unittest.TestCase):
    def test_identical_event_retry_returns_original_without_duplicate_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = LabelStore(root / "state")
            pair = _pair(root)
            payload = {
                "event_id": "event-1",
                "review_id": "review-1",
                "decision": "a_good",
                "display_order": [0, 1],
                "chosen_image_indices": [0],
                "defects_a": [],
                "defects_b": [],
                "defects_by_image_index": {"0": [], "1": []},
                "note": "same event",
            }

            first = store.store_event(
                pair,
                reviewer_username="reviewer",
                client_instance_id="client",
                payload=payload,
            )
            second = store.store_event(
                pair,
                reviewer_username="reviewer",
                client_instance_id="client",
                payload=payload,
            )

            self.assertEqual(first, second)
            self.assertEqual(len(store.iter_events()), 1)
            lines = (root / "state" / "label_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)

    def test_conflicting_event_id_reuse_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = LabelStore(root / "state")
            pair = _pair(root)
            base_payload = {
                "event_id": "event-1",
                "review_id": "review-1",
                "decision": "a_good",
                "display_order": [0, 1],
                "chosen_image_indices": [0],
                "defects_a": [],
                "defects_b": [],
                "defects_by_image_index": {"0": [], "1": []},
                "note": "first",
            }
            store.store_event(
                pair,
                reviewer_username="reviewer",
                client_instance_id="client",
                payload=base_payload,
            )

            conflicting = dict(base_payload)
            conflicting["note"] = "different content"
            with self.assertRaises(LabelEventValidationError):
                store.store_event(
                    pair,
                    reviewer_username="reviewer",
                    client_instance_id="client",
                    payload=conflicting,
                )

            self.assertEqual(len(store.iter_events()), 1)


if __name__ == "__main__":
    unittest.main()
