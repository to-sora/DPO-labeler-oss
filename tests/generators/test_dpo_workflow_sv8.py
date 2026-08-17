from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators.generate_dpo_workflow_matrix_sv8 import generate_sv8_trials
from tests._paths import REPO_ROOT


RAW_V8_DIR = REPO_ROOT / "template" / "wildcard" / "research_v8"
SPECIAL_CHARACTER_FILE = REPO_ROOT / "template" / "wildcard" / "custom_character_list.txt"


class DpoWorkflowSv8Tests(unittest.TestCase):
    def test_generate_sv8_trials_injects_special_character_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result = generate_sv8_trials(
                raw_wildcard_dir=RAW_V8_DIR,
                special_character_file=SPECIAL_CHARACTER_FILE,
                template_output_dir=tmp / "prompt_templates" / "research_v8",
                workflow_output_dir=tmp / "dpo_workflow_sv8",
                global_seed=20260408,
            )

            manifest_path = tmp / "dpo_workflow_sv8" / "manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(result["raw_wildcard_dir"], str(RAW_V8_DIR))
            self.assertEqual(result["special_character_file"], str(SPECIAL_CHARACTER_FILE))
            self.assertEqual(manifest["workflow_count"], 4)
            self.assertEqual(manifest["special_character_relative_path"], "sv8/Character.txt")
            self.assertEqual(manifest["wildcard_mode"], "runtime_first_pointer_to_research_v8")
            self.assertTrue((tmp / "dpo_workflow_sv8" / "sv8_sfw_same.yaml").is_file())
            self.assertTrue((tmp / "dpo_workflow_sv8" / "sv8_nsfw_diff.yaml").is_file())

            template_text = (
                tmp / "prompt_templates" / "research_v8" / "sv8_sfw_hybrid_compact_illustration.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("__research_v8/sv8/Character__", template_text)
            self.assertNotIn("semantic_class", template_text)
            self.assertNotIn("NSFW_Nudity", template_text)

    def test_generated_sv8_tasks_compile_with_sv8_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            generate_sv8_trials(
                raw_wildcard_dir=RAW_V8_DIR,
                special_character_file=SPECIAL_CHARACTER_FILE,
                template_output_dir=tmp / "prompt_templates" / "research_v8",
                workflow_output_dir=tmp / "dpo_workflow_sv8",
                global_seed=20260408,
            )

            task_path = tmp / "dpo_workflow_sv8" / "sv8_sfw_same.yaml"
            task = load_task_yaml(task_path)
            requests, manifest, _ = compile_requests(task, task_path)

            self.assertEqual(manifest["request_count"], len(requests))
            self.assertTrue(all(row["positive_prompt"] for row in requests))
            self.assertTrue(
                all(row["prompt_generator_args"]["template"].startswith("research_v8/sv8_sfw_hybrid_compact") for row in requests)
            )


if __name__ == "__main__":
    unittest.main()
