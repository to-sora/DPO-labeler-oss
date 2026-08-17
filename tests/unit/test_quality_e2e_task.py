from __future__ import annotations

import unittest
from collections import Counter, defaultdict
from pathlib import Path

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml


class QualityE2eTaskTests(unittest.TestCase):
    def test_published_task_compiles_to_ten_images_per_configuration(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        task_path = repo_root / "template" / "tasks" / "quality_e2e_10x2.yaml"
        task = load_task_yaml(task_path)

        records, manifest, _ = compile_requests(task, task_path)

        self.assertEqual(manifest["session_count"], 10)
        self.assertEqual(manifest["image_count_per_session"], 2)
        self.assertEqual(manifest["request_count"], 20)
        self.assertEqual(Counter(record["image_name"] for record in records), {"image1": 10, "image2": 10})
        self.assertEqual(
            {record["image_name"]: record["ckpt"] for record in records},
            {
                "image1": "sdxl_novaAnimeXL_ilV150.safetensors",
                "image2": "sdxl_bluePencilXL_v700.safetensors",
            },
        )
        self.assertEqual({record["ckpt_family"] for record in records}, {"sdxl_anime_base"})
        self.assertEqual(
            {record["ckpt_registry_id"] for record in records},
            {"nova-anime-xl-v150", "blue-pencil-xl-v700"},
        )
        self.assertEqual({record["ckpt_family_source"] for record in records}, {"keyword"})
        self.assertEqual({record["ckpt_visibility"] for record in records}, {"public"})
        self.assertEqual({record["ckpt_publish"] for record in records}, {True})
        self.assertTrue(all(record["positive_prompt"] for record in records))
        self.assertTrue(all("__" not in record["positive_prompt"] for record in records))
        self.assertTrue(all(record["lora_stack_config"]["toggle"] is False for record in records))

        records_by_session: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            records_by_session[record["session_id"]].append(record)
        self.assertEqual(len(records_by_session), 10)
        self.assertTrue(
            all({record["image_name"] for record in session} == {"image1", "image2"} for session in records_by_session.values())
        )


if __name__ == "__main__":
    unittest.main()
