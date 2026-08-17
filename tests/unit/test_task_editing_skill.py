from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TaskEditingSkillTests(unittest.TestCase):
    def test_packaged_validator_accepts_public_quality_task(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "skills" / "edit-dpo-task" / "scripts" / "validate_task.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "template/tasks/quality_e2e_10x2.yaml",
                "--repo-root",
                str(repo_root),
                "--expected-per-image",
                "10",
                "--publication",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["requests"], 20)
        self.assertEqual(summary["image_counts"], {"image1": 10, "image2": 10})
        self.assertEqual(summary["errors"], [])
        self.assertTrue(all(row["publish"] for row in summary["checkpoints"]))
        self.assertTrue(
            all(row["family_source"] != "default" for row in summary["checkpoints"])
        )


if __name__ == "__main__":
    unittest.main()
