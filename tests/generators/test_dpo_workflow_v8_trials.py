from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators.generate_dpo_workflow_matrix_v8 import generate_v8_trials
from tests._paths import REPO_ROOT


RAW_V8_DIR = REPO_ROOT / "template" / "wildcard" / "research_v8"


class DpoWorkflowV8TrialTests(unittest.TestCase):
    def test_generate_v8_trials_writes_model_aligned_templates_and_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            result = generate_v8_trials(
                raw_wildcard_dir=RAW_V8_DIR,
                wildcard_output_dir=tmp / "wildcard" / "research_v8",
                template_output_dir=tmp / "prompt_templates" / "research_v8",
                workflow_output_dir=tmp / "dpo_workflow_v8",
                global_seed=20260408,
                session_count=12,
            )

            manifest_path = tmp / "dpo_workflow_v8" / "manifest.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["workflow_count"], 4)
            self.assertEqual(result["active_source_files"], manifest["source_files"])
            self.assertEqual(
                result["template_omitted_source_files"],
                ["Character.txt", "characters/*", "semantic_class.txt"],
            )
            self.assertFalse(result["character_sources_enabled"])
            self.assertEqual(manifest["wildcard_mode"], "runtime_first_with_no_semantic_or_character_fallback")
            self.assertEqual(manifest["top_level_template_fallbacks"], [])
            self.assertNotIn("Character.txt", result["active_source_files"])
            self.assertNotIn("semantic_class.txt", result["active_source_files"])
            self.assertTrue((tmp / "dpo_workflow_v8" / "v8_sfw_same.yaml").is_file())
            self.assertTrue((tmp / "dpo_workflow_v8" / "v8_nsfw_diff.yaml").is_file())

            sfw_illustration = (
                tmp / "prompt_templates" / "research_v8" / "sfw_hybrid_compact_illustration.txt"
            ).read_text(encoding="utf-8")
            nsfw_illustration = (
                tmp / "prompt_templates" / "research_v8" / "nsfw_hybrid_compact_illustration.txt"
            ).read_text(encoding="utf-8")
            sfw_pony = (tmp / "prompt_templates" / "research_v8" / "sfw_hybrid_compact_pony.txt").read_text(
                encoding="utf-8"
            )
            sfw_anime = (
                tmp / "prompt_templates" / "research_v8" / "sfw_hybrid_compact_sdxl_anime_base.txt"
            ).read_text(encoding="utf-8")

            self.assertIn("__research_v8/runtime_sfw/Rating__", sfw_illustration)
            self.assertIn("__research_v8/runtime_sfw/Attire_Accessory__", sfw_illustration)
            self.assertNotIn("NSFW_Nudity", sfw_illustration)
            self.assertNotIn("semantic_class", sfw_illustration)
            self.assertNotIn("__research_v8/Character__", sfw_illustration)
            self.assertIn("__research_v8/runtime_nsfw/NSFW_Nudity__", nsfw_illustration)
            self.assertIn("high score, great score, absurdres", sfw_anime)
            self.assertTrue(sfw_pony.splitlines()[0].startswith("score_9, score_8_up, score_7_up, score_6_up"))

    def test_generated_v8_tasks_compile_with_research_v8_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            generate_v8_trials(
                raw_wildcard_dir=RAW_V8_DIR,
                wildcard_output_dir=tmp / "wildcard" / "research_v8",
                template_output_dir=tmp / "prompt_templates" / "research_v8",
                workflow_output_dir=tmp / "dpo_workflow_v8",
                global_seed=20260408,
            )

            for task_name in ("v8_sfw_same.yaml", "v8_nsfw_same.yaml"):
                task_path = tmp / "dpo_workflow_v8" / task_name
                task = load_task_yaml(task_path)
                requests, manifest, _ = compile_requests(task, task_path)

                self.assertEqual(manifest["request_count"], len(requests))
                self.assertTrue(all(row["positive_prompt"] for row in requests))
                self.assertTrue(all(row["negative_prompt"] for row in requests))
                self.assertTrue(all("research_v8/" in row["prompt_generator_args"]["template"] for row in requests))
                self.assertTrue(all("__research_v8/" not in row["positive_prompt"] for row in requests))


if __name__ == "__main__":
    unittest.main()
