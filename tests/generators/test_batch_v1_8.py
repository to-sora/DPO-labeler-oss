from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators.generate_batch_v1_8 import DEFAULT_SESSION_COUNT, generate_batch_v1_8


class BatchV18Tests(unittest.TestCase):
    def test_generate_batch_v1_8_writes_v1_through_v8_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = tmp / "batch_v1_8"
            rows = generate_batch_v1_8(out_dir, global_seed=20260408)

            manifest_path = out_dir / "manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 32)
            self.assertEqual(manifest["workflow_count"], 32)
            self.assertEqual(manifest["session_count"], DEFAULT_SESSION_COUNT)
            self.assertEqual({row["version"] for row in rows}, {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"})
            self.assertTrue((out_dir / "b18_v8_sfw_same.yaml").is_file())
            self.assertTrue((out_dir / "b18_v8_nsfw_diff.yaml").is_file())

    def test_representative_v8_batch_task_compiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = tmp / "batch_v1_8"
            generate_batch_v1_8(out_dir, global_seed=20260408)

            task_path = out_dir / "b18_v8_sfw_same.yaml"
            task = load_task_yaml(task_path)
            requests, manifest, _ = compile_requests(task, task_path)

            self.assertEqual(manifest["request_count"], len(requests))
            self.assertTrue(all(row["positive_prompt"] for row in requests))
            self.assertTrue(all("research_v8/" in row["prompt_generator_args"]["template"] for row in requests))


if __name__ == "__main__":
    unittest.main()
