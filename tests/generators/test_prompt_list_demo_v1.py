from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from prompt_generator import load_prompt_list_lines
from generators.generate_dpo_workflow_matrix_v8 import NEGATIVE_PROMPTS_BY_FAMILY
from generators.generate_prompt_list_demo_v1 import generate_prompt_list_demo_v1
from tests._paths import REPO_ROOT


PROMPT_LIST_ROOT = REPO_ROOT / "template" / "prompt_lists"
PROMPT_LIST_SOURCE_DIR = PROMPT_LIST_ROOT / "prompt_list_demo_v1"


class PromptListDemoV1GeneratorTests(unittest.TestCase):
    def test_generate_prompt_list_demo_v1_writes_same_and_diff_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows = generate_prompt_list_demo_v1(
                output_dir=tmp / "dpo_workflow_prompt_list_demo_v1",
                prompt_list_root=PROMPT_LIST_ROOT,
                prompt_list_source_dir=PROMPT_LIST_SOURCE_DIR,
                global_seed=20260408,
            )

            manifest_path = tmp / "dpo_workflow_prompt_list_demo_v1" / "manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            expected_line_count = len(load_prompt_list_lines(PROMPT_LIST_SOURCE_DIR / "illustration.txt"))

            self.assertEqual(len(rows), 2)
            self.assertEqual(manifest["workflow_count"], 2)
            self.assertEqual(manifest["prompt_line_count"], expected_line_count)
            self.assertEqual(manifest["realistic_prompt_fallback"], "prompt_list_demo_v1/illustration")
            self.assertTrue((tmp / "dpo_workflow_prompt_list_demo_v1" / "prompt_list_demo_v1_same.yaml").is_file())
            self.assertTrue((tmp / "dpo_workflow_prompt_list_demo_v1" / "prompt_list_demo_v1_diff.yaml").is_file())

            same_task = load_task_yaml(tmp / "dpo_workflow_prompt_list_demo_v1" / "prompt_list_demo_v1_same.yaml")
            diff_task = load_task_yaml(tmp / "dpo_workflow_prompt_list_demo_v1" / "prompt_list_demo_v1_diff.yaml")
            self.assertEqual(same_task["session_count"], expected_line_count)
            self.assertEqual(diff_task["session_count"], expected_line_count)
            self.assertEqual(same_task["images"][0]["ckpt"]["seed_control"], "session_seed")
            self.assertEqual(diff_task["images"][0]["ckpt"]["seed_control"], "image_index_seed")

    def test_generated_tasks_compile_to_expected_prompt_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            generate_prompt_list_demo_v1(
                output_dir=tmp / "dpo_workflow_prompt_list_demo_v1",
                prompt_list_root=PROMPT_LIST_ROOT,
                prompt_list_source_dir=PROMPT_LIST_SOURCE_DIR,
                global_seed=20260408,
            )

            expected_lines = {
                "illustration": load_prompt_list_lines(PROMPT_LIST_SOURCE_DIR / "illustration.txt"),
                "anime": load_prompt_list_lines(PROMPT_LIST_SOURCE_DIR / "anime.txt"),
                "pony": load_prompt_list_lines(PROMPT_LIST_SOURCE_DIR / "pony.txt"),
            }

            for task_file in ("prompt_list_demo_v1_same.yaml", "prompt_list_demo_v1_diff.yaml"):
                task_path = tmp / "dpo_workflow_prompt_list_demo_v1" / task_file
                task = load_task_yaml(task_path)
                requests, manifest, _ = compile_requests(task, task_path)

                self.assertEqual(manifest["request_count"], len(requests))
                for row in requests:
                    expected_family_key = {
                        "illustration": "illustration",
                        "sdxl_anime_base": "anime",
                        "pony": "pony",
                        "realistic": "illustration",
                    }[row["ckpt_family"]]
                    self.assertEqual(
                        row["positive_prompt"],
                        expected_lines[expected_family_key][row["session_index"]],
                    )
                    self.assertEqual(
                        row["negative_prompt"],
                        {
                            "illustration": NEGATIVE_PROMPTS_BY_FAMILY["illustration"],
                            "sdxl_anime_base": NEGATIVE_PROMPTS_BY_FAMILY["sdxl_anime_base"],
                            "pony": NEGATIVE_PROMPTS_BY_FAMILY["pony"],
                            "realistic": NEGATIVE_PROMPTS_BY_FAMILY["realistic"],
                        }[row["ckpt_family"]],
                    )

    def test_generate_prompt_list_demo_v1_rejects_mismatched_line_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prompt_list_root = tmp / "prompt_lists"
            source_dir = prompt_list_root / "demo"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "illustration.txt").write_text("one\ntwo\n", encoding="utf-8")
            (source_dir / "anime.txt").write_text("one\n", encoding="utf-8")
            (source_dir / "pony.txt").write_text("one\ntwo\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must have the same usable line count"):
                generate_prompt_list_demo_v1(
                    output_dir=tmp / "dpo_workflow_prompt_list_demo_v1",
                    prompt_list_root=prompt_list_root,
                    prompt_list_source_dir=source_dir,
                    global_seed=20260408,
                )


if __name__ == "__main__":
    unittest.main()
