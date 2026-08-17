from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators._shared_task import (
    SHARED_CFG_VALUES,
    SHARED_PAIR_DROPOUT_CHANCE,
    SHARED_RESOLUTION_VALUES,
    SHARED_RESOLUTION_WEIGHTS,
    SHARED_SEGMENT_DROPOUT_PROB,
    SHARED_SESSION_COUNT,
    SHARED_STEPS_VALUES,
    SHARED_WORKFLOW_NAME,
)
from generators.generate_batch_v1_7 import VERSION_TEMPLATE_BUILDERS, generate_batch_v1_7
from tests._paths import REPO_ROOT


class BatchV17Tests(unittest.TestCase):
    def test_generate_batch_writes_flat_workflow_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "batch_v1_7"
            rows = generate_batch_v1_7(output_dir)
            manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 28)
            self.assertEqual(manifest["workflow_count"], 28)
            self.assertEqual(manifest["session_count"], SHARED_SESSION_COUNT)
            self.assertEqual(manifest["rows"], rows)
            self.assertEqual(
                manifest["random_segment_dropout"],
                {
                    "pair_chance": SHARED_PAIR_DROPOUT_CHANCE,
                    "segment_prob": SHARED_SEGMENT_DROPOUT_PROB,
                },
            )
            self.assertEqual({row["pair_mode"] for row in rows}, {"same", "diff"})

            for row in rows:
                task_path = output_dir / row["file"]
                self.assertTrue(task_path.is_file(), msg=str(task_path))
                task = load_task_yaml(task_path)
                expected_ckpt_control = "session_seed" if row["pair_mode"] == "same" else "image_index_seed"

                self.assertEqual(task["task_name"], row["task_name"])
                self.assertEqual(task["session_count"], SHARED_SESSION_COUNT)
                self.assertEqual(len(task["images"]), 2)
                for image in task["images"]:
                    self.assertEqual(image["workflow_name"], SHARED_WORKFLOW_NAME)
                    self.assertNotIn("workflow_kwargs", image)
                    self.assertFalse(image["lora_stack_config"]["toggle"])
                    self.assertEqual(image["ckpt"]["seed_control"], expected_ckpt_control)

                    sample = image["sample"]
                    self.assertEqual(sample["generation_seed_control"], "image_index_seed")
                    self.assertEqual([item["value"] for item in sample["steps"]["options"]], SHARED_STEPS_VALUES)
                    self.assertEqual([item["value"] for item in sample["cfg"]["options"]], SHARED_CFG_VALUES)
                    self.assertEqual(
                        [item["value"] for item in sample["width"]["options"]],
                        SHARED_RESOLUTION_VALUES,
                    )
                    self.assertEqual(
                        [item["weight"] for item in sample["width"]["options"]],
                        SHARED_RESOLUTION_WEIGHTS,
                    )

                    prompt_args = image["prompt_generator"]["args"]
                    self.assertEqual(prompt_args["seed_control"], "session_seed")
                    self.assertEqual(
                        prompt_args["random_segment_dropout_pair_chance"],
                        SHARED_PAIR_DROPOUT_CHANCE,
                    )
                    self.assertEqual(
                        prompt_args["random_segment_dropout_segment_prob"],
                        SHARED_SEGMENT_DROPOUT_PROB,
                    )
                    self.assertEqual(prompt_args["seed_control_random_segment_dropout"], "image_index_seed")

            expected_prefixes = {
                "v1": "curated/",
                "v2": "research_v2/",
                "v3": "research_v3/",
                "v4": "research_v4/",
                "v5": "research_v5/",
                "v6": "research_v6/",
                "v7": "research_v7/",
            }
            for version_name, _, _ in VERSION_TEMPLATE_BUILDERS:
                representative = next(
                    row for row in rows if row["version"] == version_name and row["pair_mode"] == "same"
                )
                task_path = output_dir / representative["file"]
                task = load_task_yaml(task_path)
                requests, compile_manifest, _ = compile_requests(task, task_path)

                self.assertEqual(compile_manifest["request_count"], SHARED_SESSION_COUNT * 2)
                self.assertEqual(len(requests), SHARED_SESSION_COUNT * 2)
                self.assertTrue(all(request["workflow_name"] == SHARED_WORKFLOW_NAME for request in requests))
                self.assertTrue(
                    all(
                        expected_prefixes[version_name] in request["prompt_generator_args"]["template"]
                        for request in requests
                    )
                )

    def test_root_scheduler_help(self) -> None:
        scheduler_path = REPO_ROOT / "run_multi_task_scheduled_cycle.sh"
        result = subprocess.run(
            ["bash", str(scheduler_path), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--task-yaml", result.stdout)


if __name__ == "__main__":
    unittest.main()
