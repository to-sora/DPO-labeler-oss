from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml, write_compile_outputs
from generators.generate_dpo_workflow_matrix import generate_matrix
from split_jsonl_by_shared_config import split_jsonl_by_shared_config
from tests.generators._consolidated_matrix_contract import assert_consolidated_matrix


class DpoWorkflowMatrixTests(unittest.TestCase):
    def test_generate_matrix_writes_consolidated_v1_contract(self) -> None:
        assert_consolidated_matrix(
            self,
            generate_matrix=generate_matrix,
            version_tag="v1",
            template_short_names=("gpt", "qwen"),
            template_prefix="curated/",
        )

    def test_generated_matrix_compiles_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "dpo_workflow"
            generate_matrix(output_dir)

            task_path = output_dir / "v1_gpt_same.yaml"
            task = load_task_yaml(task_path)
            requests, manifest, task_yaml_text = compile_requests(task, task_path)
            compiled_dir = root / "compiled"
            write_compile_outputs(compiled_dir, requests, manifest, task_yaml_text)

            grouped_dir = root / "grouped"
            grouped_manifest = split_jsonl_by_shared_config(compiled_dir / "requests.jsonl", grouped_dir)
            rows = [
                yaml.safe_load(line)
                for line in grouped_manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertTrue(rows)
            for row in rows:
                self.assertLess(len(row["output_jsonl"]), 128)
                self.assertTrue((grouped_dir / row["output_jsonl"]).is_file())


if __name__ == "__main__":
    unittest.main()
