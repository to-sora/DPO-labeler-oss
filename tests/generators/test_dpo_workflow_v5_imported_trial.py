from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators._shared_task import SHARED_SESSION_COUNT, SHARED_STEPS_VALUES, SHARED_WORKFLOW_NAME
from generators.generate_dpo_workflow_matrix_v5 import generate_matrix
from tests._paths import REPO_ROOT


IMPORTED_DIR = REPO_ROOT / "template" / "wildcard" / "research_v5" / "imported"
EXPECTED_IMPORTED_FILES = {
    "subject_count.txt",
    "identity_illustration.txt",
    "identity_sdxl_anime_base.txt",
    "identity_pony.txt",
    "identity_realistic.txt",
    "body_style.txt",
    "expression.txt",
    "face_detail.txt",
    "pose_core.txt",
    "pose_adult.txt",
    "framing.txt",
    "camera.txt",
    "location.txt",
    "lighting.txt",
    "wardrobe_core.txt",
    "wardrobe_adult.txt",
    "surface_detail.txt",
}


class V5ImportedTrialTests(unittest.TestCase):
    def test_imported_manifest_points_to_existing_sources(self) -> None:
        manifest = yaml.safe_load((IMPORTED_DIR / "manifest.yaml").read_text(encoding="utf-8"))
        imported_files = {path.name for path in IMPORTED_DIR.glob("*.txt")}

        self.assertEqual(imported_files, EXPECTED_IMPORTED_FILES)
        self.assertTrue((IMPORTED_DIR / "README.md").is_file())

        for row in manifest["files"]:
            file_name = row["file"]
            with self.subTest(file_name=file_name):
                self.assertIn(file_name, EXPECTED_IMPORTED_FILES)
                imported_path = IMPORTED_DIR / file_name
                self.assertTrue(imported_path.is_file())

                lines = [line.strip() for line in imported_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertTrue(lines)
                self.assertEqual(len(lines), len(set(lines)))

                for source in row["sources"]:
                    self.assertTrue((REPO_ROOT / "template" / "wildcard" / source).exists())

    def test_trial_templates_only_reference_imported_v5_wildcards(self) -> None:
        for path in sorted((REPO_ROOT / "template" / "prompt_templates" / "research_v5").glob("trial_hybrid_compact_*_adult.txt")):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("__research_v5/imported/", text)
                self.assertNotIn("__research_v4/", text)

    def test_generated_v5_compact_same_task_compiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "dpo_workflow_v5"
            generate_matrix(output_dir)
            task_path = output_dir / "v5_compact_same.yaml"
            task = load_task_yaml(task_path)
            requests, manifest, _ = compile_requests(task, task_path)

            self.assertEqual(task["session_count"], SHARED_SESSION_COUNT)
            self.assertEqual(manifest["request_count"], SHARED_SESSION_COUNT * 2)
            self.assertEqual(len(requests), SHARED_SESSION_COUNT * 2)
            self.assertEqual(task["images"][0]["workflow_name"], SHARED_WORKFLOW_NAME)
            self.assertEqual(task["images"][0]["ckpt"]["seed_control"], "session_seed")
            self.assertEqual(task["images"][0]["prompt_generator"]["args"]["seed_control"], "session_seed")
            self.assertEqual(task["images"][0]["sample"]["generation_seed_control"], "image_index_seed")
            self.assertNotIn("workflow_kwargs", task["images"][0])
            self.assertTrue(all(row["workflow_name"] == SHARED_WORKFLOW_NAME for row in requests))
            self.assertTrue(all(row["prompt_seed_control"] == "session_seed" for row in requests))
            self.assertTrue(all(row["generation_seed_control"] == "image_index_seed" for row in requests))
            self.assertTrue(all(row["steps"] in set(SHARED_STEPS_VALUES) for row in requests))
            self.assertTrue(
                all("research_v5/hybrid_compact_" in row["prompt_generator_args"]["template"] for row in requests)
            )

            checkpoints_by_session: dict[str, list[str]] = {}
            for row in requests:
                checkpoints_by_session.setdefault(row["session_id"], []).append(row["ckpt"])
            self.assertTrue(checkpoints_by_session)
            self.assertTrue(all(len(set(checkpoints)) == 1 for checkpoints in checkpoints_by_session.values()))


if __name__ == "__main__":
    unittest.main()
