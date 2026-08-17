from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests
from prompt_generator import NonWildcardV2PromptGenerator, get_prompt_generator


CKPT_BY_FAMILY = {
    "illustration": "sdxl_hassakuXLIllustrious_v32.safetensors",
    "sdxl_anime_base": "sdxl_animagineXL40_v4Opt.safetensors",
    "pony": "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    "realistic": "sdxl_perfectdeliberate_v60.safetensors",
}
POSITIVE_PREFIX_BY_FAMILY = {
    "illustration": "illustration prefix",
    "sdxl_anime_base": "anime prefix",
    "pony": "score_9, score_8_up, score_7_up, score_6_up",
    "realistic": "photo prefix",
}
POSITIVE_SUFFIX_BY_FAMILY = {
    "illustration": "",
    "sdxl_anime_base": "high score, great score, absurdres",
    "pony": "",
    "realistic": "realistic finish",
}


class NonWildcardV2PromptGeneratorTests(unittest.TestCase):
    def test_registry_returns_non_wildcard_v2_generator(self) -> None:
        generator = get_prompt_generator("non_wildcard_v2")
        self.assertIsInstance(generator, NonWildcardV2PromptGenerator)

    def test_compile_requests_routes_family_specific_prefix_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            character_list = base_dir / "characters.txt"
            character_list.write_text("altria pendragon\ntohsaka rin\n", encoding="utf-8")

            for family, ckpt in CKPT_BY_FAMILY.items():
                task = {
                    "version": 1,
                    "task_name": f"nwv2_{family}",
                    "global_seed": 42,
                    "session_count": 1,
                    "images": [
                        {
                            "image_name": "image1",
                            "workflow_name": "SdxlEaseLoraWorkflow",
                            "ckpt": ckpt,
                            "lora_stack_config": {},
                            "prompt_generator": {
                                "name": "non_wildcard_v2",
                                "args": {
                                    "character_list": str(character_list.resolve()),
                                    "positive_prefix": "fallback prefix",
                                    "positive_suffix": "fallback suffix",
                                    "positive_prefix_by_ckpt_family": dict(POSITIVE_PREFIX_BY_FAMILY),
                                    "positive_suffix_by_ckpt_family": dict(POSITIVE_SUFFIX_BY_FAMILY),
                                    "negative_prompt": "bad anatomy",
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
                self.assertEqual(records[0]["ckpt_family"], family)
                self.assertEqual(records[0]["prompt_generator_name"], "non_wildcard_v2")
                self.assertEqual(
                    records[0]["prompt_generator_args"]["positive_prefix"],
                    POSITIVE_PREFIX_BY_FAMILY[family],
                )
                self.assertEqual(
                    records[0]["prompt_generator_args"]["positive_suffix"],
                    POSITIVE_SUFFIX_BY_FAMILY[family],
                )
                self.assertTrue(records[0]["positive_prompt"])
                self.assertTrue(records[0]["negative_prompt"])


if __name__ == "__main__":
    unittest.main()
