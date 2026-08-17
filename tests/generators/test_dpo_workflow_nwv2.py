from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators.generate_dpo_workflow_matrix_nwv2 import (
    POSITIVE_PREFIX_BY_FAMILY,
    POSITIVE_SUFFIX_BY_FAMILY,
    generate_matrix,
)
from tests._paths import REPO_ROOT


CHARACTER_LIST = REPO_ROOT / "template" / "wildcard" / "custom_character_list.txt"


class DpoWorkflowNwv2Tests(unittest.TestCase):
    def test_generate_nwv2_matrix_writes_new_engine_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows = generate_matrix(
                output_dir=tmp / "dpo_workflow_nwv2",
                character_list=CHARACTER_LIST,
                global_seed=20260408,
            )

            manifest_path = tmp / "dpo_workflow_nwv2" / "manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 2)
            self.assertEqual(manifest["workflow_count"], 2)
            self.assertTrue((tmp / "dpo_workflow_nwv2" / "nwv2_default_same.yaml").is_file())
            self.assertTrue((tmp / "dpo_workflow_nwv2" / "nwv2_default_diff.yaml").is_file())

            task = load_task_yaml(tmp / "dpo_workflow_nwv2" / "nwv2_default_same.yaml")
            args = task["images"][0]["prompt_generator"]["args"]
            self.assertEqual(task["images"][0]["prompt_generator"]["name"], "non_wildcard_v2")
            self.assertEqual(args["positive_prefix_by_ckpt_family"], POSITIVE_PREFIX_BY_FAMILY)
            self.assertEqual(args["positive_suffix_by_ckpt_family"], POSITIVE_SUFFIX_BY_FAMILY)

    def test_generated_nwv2_tasks_compile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            generate_matrix(
                output_dir=tmp / "dpo_workflow_nwv2",
                character_list=CHARACTER_LIST,
                global_seed=20260408,
            )

            task_path = tmp / "dpo_workflow_nwv2" / "nwv2_default_same.yaml"
            task = load_task_yaml(task_path)
            requests, manifest, _ = compile_requests(task, task_path)

            self.assertEqual(manifest["request_count"], len(requests))
            self.assertTrue(all(row["prompt_generator_name"] == "non_wildcard_v2" for row in requests))
            self.assertTrue(all(row["positive_prompt"] for row in requests))
            self.assertTrue(all(row["negative_prompt"] for row in requests))


if __name__ == "__main__":
    unittest.main()
