from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests
from prompt_generator import PromptListV1PromptGenerator, get_prompt_generator


CKPT_BY_FAMILY = {
    "illustration": "sdxl_hassakuXLIllustrious_v32.safetensors",
    "sdxl_anime_base": "sdxl_animagineXL40_v4Opt.safetensors",
    "pony": "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    "realistic": "sdxl_perfectdeliberate_v60.safetensors",
}


class PromptListV1PromptGeneratorTests(unittest.TestCase):
    def test_registry_returns_prompt_list_generator(self) -> None:
        generator = get_prompt_generator("prompt_list_v1")
        self.assertIsInstance(generator, PromptListV1PromptGenerator)

    def test_generate_uses_exact_session_index_and_preserves_literal_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "prompt_lists"
            root.mkdir(parents=True, exist_ok=True)
            prompt_list_path = root / "demo.txt"
            prompt_list_path.write_text(
                "# comment\nalpha, beta\nliteral, {choice|value}, __token__\nthird line\n",
                encoding="utf-8",
            )

            bundle = PromptListV1PromptGenerator().generate(
                {
                    "prompt_list": "demo",
                    "prompt_list_root": str(root),
                    "_resolved_session_index": 1,
                },
                seed=123,
            )

            self.assertEqual(bundle.positive_prompt, "literal, {choice|value}, __token__")
            self.assertEqual(bundle.prompt_metadata["prompt_line_index"], 1)
            self.assertEqual(bundle.prompt_metadata["prompt_line_count"], 3)

    def test_generate_raises_for_out_of_range_session_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "prompt_lists"
            root.mkdir(parents=True, exist_ok=True)
            (root / "demo.txt").write_text("line one\nline two\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not have line index 2"):
                PromptListV1PromptGenerator().generate(
                    {
                        "prompt_list": "demo",
                        "prompt_list_root": str(root),
                        "_resolved_session_index": 2,
                    },
                    seed=7,
                )

    def test_compile_requests_routes_family_prompt_lists_and_realistic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            prompt_list_root = base_dir / "prompt_lists"
            prompt_list_dir = prompt_list_root / "pack"
            prompt_list_dir.mkdir(parents=True, exist_ok=True)
            (prompt_list_dir / "illustration.txt").write_text("illustration line\n", encoding="utf-8")
            (prompt_list_dir / "anime.txt").write_text("anime line\n", encoding="utf-8")
            (prompt_list_dir / "pony.txt").write_text("pony line\n", encoding="utf-8")

            for family, ckpt in CKPT_BY_FAMILY.items():
                task = {
                    "version": 1,
                    "task_name": f"prompt_list_{family}",
                    "global_seed": 42,
                    "session_count": 1,
                    "images": [
                        {
                            "image_name": "image1",
                            "workflow_name": "SdxlEaseLoraWorkflow",
                            "ckpt": ckpt,
                            "lora_stack_config": {},
                            "prompt_generator": {
                                "name": "prompt_list_v1",
                                "args": {
                                    "prompt_list": "pack/illustration",
                                    "prompt_list_by_ckpt_family": {
                                        "illustration": "pack/illustration",
                                        "anime": "pack/anime",
                                        "pony": "pack/pony",
                                    },
                                    "prompt_list_root": str(prompt_list_root),
                                    "negative_prompt": "base negative",
                                    "negative_prompt_by_ckpt_family": {
                                        "illustration": "ill negative",
                                        "sdxl_anime_base": "anime negative",
                                        "pony": "pony negative",
                                        "realistic": "photo negative",
                                    },
                                },
                            },
                            "sample": {
                                "generation_seed_control": "image_index_seed",
                                "steps": 24,
                                "cfg": 7.0,
                                "width": 768,
                                "height": 768,
                            },
                        }
                    ],
                }
                task_yaml_path = base_dir / f"{family}.yaml"
                task_yaml_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

                records, manifest, _ = compile_requests(task, task_yaml_path)

                self.assertEqual(manifest["request_count"], 1)
                expected_positive = {
                    "illustration": "illustration line",
                    "sdxl_anime_base": "anime line",
                    "pony": "pony line",
                    "realistic": "illustration line",
                }[family]
                expected_negative = {
                    "illustration": "ill negative",
                    "sdxl_anime_base": "anime negative",
                    "pony": "pony negative",
                    "realistic": "photo negative",
                }[family]
                self.assertEqual(records[0]["positive_prompt"], expected_positive)
                self.assertEqual(records[0]["negative_prompt"], expected_negative)


if __name__ == "__main__":
    unittest.main()
