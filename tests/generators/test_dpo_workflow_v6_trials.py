from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators._shared_task import SHARED_SESSION_COUNT, SHARED_STEPS_VALUES, SHARED_WORKFLOW_NAME
from generators.generate_dpo_workflow_matrix_v6 import generate_v6_trials
from tests._paths import REPO_ROOT


RAW_V6_DIR = REPO_ROOT / "template" / "wildcard" / "research_v6"
GREYSCALE_TERMS = (
    "greyscale_with_colored_background",
    "monochrome_background",
    "monochrome",
    "greyscale",
    "multiple_monochrome",
)


def _generate_v6(root: Path) -> dict[str, object]:
    return generate_v6_trials(
        raw_wildcard_dir=RAW_V6_DIR,
        wildcard_output_dir=root / "wildcard" / "research_v6",
        template_output_dir=root / "prompt_templates" / "research_v6",
        workflow_output_dir=root / "dpo_workflow_v6",
    )


class DpoWorkflowV6TrialTests(unittest.TestCase):
    def test_generate_v6_trials_writes_runtime_assets_and_consolidated_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = _generate_v6(root)
            manifest = yaml.safe_load(
                (root / "dpo_workflow_v6" / "manifest.yaml").read_text(encoding="utf-8")
            )

            self.assertEqual(result["active_source_files"], manifest["source_files"])
            self.assertEqual(result["empty_source_files"], manifest["empty_source_files"])
            self.assertEqual(
                result["template_omitted_source_files"],
                ["Body_Appearance.txt", "Attire_Accessory.txt"],
            )
            self.assertTrue(result["character_sources_enabled"])
            self.assertEqual(manifest["workflow_count"], 4)
            self.assertEqual(manifest["wildcard_mode"], "runtime_first_with_top_level_fallback")
            self.assertEqual(manifest["runtime_dirs_by_set"], {"sfw": "runtime_sfw", "nsfw": "runtime_nsfw"})
            self.assertEqual(manifest["top_level_template_fallbacks"], ["characters/*", "semantic_class.txt"])
            self.assertIn("semantic_class.txt", result["active_source_files"])

            expected_tasks = {
                "v6_sfw_same.yaml",
                "v6_sfw_diff.yaml",
                "v6_nsfw_same.yaml",
                "v6_nsfw_diff.yaml",
            }
            self.assertEqual(
                {row["output_yaml"] for row in result["workflows"]},
                expected_tasks,
            )
            self.assertTrue(all((root / "dpo_workflow_v6" / name).is_file() for name in expected_tasks))
            self.assertTrue((root / "wildcard" / "research_v6" / "Action_Pose.txt").is_file())
            self.assertTrue((root / "wildcard" / "research_v6" / "runtime_sfw" / "Action_Pose.txt").is_file())
            self.assertTrue((root / "wildcard" / "research_v6" / "runtime_nsfw" / "Action_Pose.txt").is_file())
            self.assertTrue((root / "wildcard" / "research_v6" / "runtime_manifest.yaml").is_file())

    def test_generated_templates_use_runtime_variants_and_expected_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _generate_v6(root)
            template_dir = root / "prompt_templates" / "research_v6"

            for sfw_path in sorted(template_dir.glob("sfw_*.txt")):
                nsfw_path = template_dir / sfw_path.name.replace("sfw_", "nsfw_", 1)
                sfw_text = sfw_path.read_text(encoding="utf-8")
                nsfw_text = nsfw_path.read_text(encoding="utf-8")

                self.assertIn("__research_v6/runtime_sfw/Rating__", sfw_text)
                self.assertIn("__research_v6/runtime_sfw/Character__", sfw_text)
                self.assertIn("__research_v6/characters/fgo__", sfw_text)
                self.assertIn("__research_v6/semantic_class__", sfw_text)
                self.assertNotIn("__research_v6/Body_Appearance__", sfw_text)
                self.assertNotIn("__research_v6/Attire_Accessory__", sfw_text)

                self.assertIn("__research_v6/runtime_nsfw/Rating__", nsfw_text)
                self.assertIn("__research_v6/runtime_nsfw/Character__", nsfw_text)
                self.assertIn("__research_v6/characters/fgo__", nsfw_text)
                self.assertIn("__research_v6/semantic_class__", nsfw_text)
                self.assertNotIn("__research_v6/Body_Appearance__", nsfw_text)
                self.assertNotIn("__research_v6/Attire_Accessory__", nsfw_text)
                self.assertNotEqual(sfw_text, nsfw_text)

            scene_sfw = (root / "wildcard" / "research_v6" / "runtime_sfw" / "Scene_Background.txt").read_text(
                encoding="utf-8"
            )
            scene_nsfw = (root / "wildcard" / "research_v6" / "runtime_nsfw" / "Scene_Background.txt").read_text(
                encoding="utf-8"
            )
            style = (root / "wildcard" / "research_v6" / "Style_Medium.txt").read_text(encoding="utf-8")
            for term in GREYSCALE_TERMS:
                with self.subTest(term=term):
                    self.assertNotIn(term, scene_sfw)
                    self.assertNotIn(term, scene_nsfw)
                    self.assertNotIn(term, style)

    def test_generated_v6_tasks_compile_with_same_and_diff_checkpoint_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _generate_v6(root)

            for set_name in ("sfw", "nsfw"):
                for pair_mode in ("same", "diff"):
                    task_path = root / "dpo_workflow_v6" / f"v6_{set_name}_{pair_mode}.yaml"
                    task = load_task_yaml(task_path)
                    requests, manifest, _ = compile_requests(task, task_path)
                    expected_ckpt_control = "session_seed" if pair_mode == "same" else "image_index_seed"

                    self.assertEqual(task["session_count"], SHARED_SESSION_COUNT)
                    self.assertEqual(manifest["request_count"], SHARED_SESSION_COUNT * 2)
                    self.assertEqual(len(requests), SHARED_SESSION_COUNT * 2)
                    self.assertEqual(task["images"][0]["ckpt"]["seed_control"], expected_ckpt_control)
                    self.assertEqual(task["images"][1]["ckpt"]["seed_control"], expected_ckpt_control)
                    self.assertNotIn("workflow_kwargs", task["images"][0])
                    self.assertTrue(all(row["workflow_name"] == SHARED_WORKFLOW_NAME for row in requests))
                    self.assertTrue(all(row["prompt_seed_control"] == "session_seed" for row in requests))
                    self.assertTrue(all(row["generation_seed_control"] == "image_index_seed" for row in requests))
                    self.assertTrue(all(row["steps"] in set(SHARED_STEPS_VALUES) for row in requests))
                    self.assertTrue(all("research_v6/" in row["prompt_generator_args"]["template"] for row in requests))
                    self.assertTrue(all("__research_v6/" not in row["positive_prompt"] for row in requests))

                    if pair_mode == "same":
                        checkpoints_by_session: dict[str, list[str]] = {}
                        for row in requests:
                            checkpoints_by_session.setdefault(row["session_id"], []).append(row["ckpt"])
                        self.assertTrue(
                            all(len(set(checkpoints)) == 1 for checkpoints in checkpoints_by_session.values())
                        )


if __name__ == "__main__":
    unittest.main()
