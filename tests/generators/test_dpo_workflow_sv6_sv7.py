from __future__ import annotations

import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml, resolve_ckpt_family
from generators._shared_task import SHARED_SESSION_COUNT, SHARED_WORKFLOW_NAME
from generators.generate_dpo_workflow_matrix_sv6 import generate_sv6_trials
from generators.generate_dpo_workflow_matrix_sv7 import generate_sv7_trials
from generators.sv_checkpoint_pools import (
    SELF_TRAINED_CHECKPOINTS,
    SHARED_CUSTOM_CHECKPOINTS,
    SHARED_REMAINING_CHECKPOINTS,
    SHARED_TIER1_CHECKPOINTS,
)
from tests._paths import REPO_ROOT


RAW_V6_DIR = REPO_ROOT / "template" / "wildcard" / "research_v6"
RAW_V7_DIR = REPO_ROOT / "template" / "wildcard" / "research_v7"
SPECIAL_CHARACTER_FILE = REPO_ROOT / "template" / "wildcard" / "custom_character_list.txt"
SvGenerator = Callable[..., dict[str, Any]]


def _sum_group_weights(options: list[dict[str, Any]], values: list[str]) -> float:
    value_set = set(values)
    return sum(float(option["weight"]) for option in options if option["value"] in value_set)


def _assert_sv_contract(
    case: unittest.TestCase,
    *,
    generator: SvGenerator,
    version: str,
    research_version: str,
    raw_source_dir: Path,
    expected_template_tokens: tuple[str, ...],
) -> None:
    namespace = f"research_{research_version}"
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_dir = root / "wildcard" / namespace
        template_dir = root / "prompt_templates" / namespace
        workflow_dir = root / f"dpo_workflow_{version}"
        shutil.copytree(raw_source_dir, raw_dir)

        result = generator(
            raw_wildcard_dir=raw_dir,
            special_character_file=SPECIAL_CHARACTER_FILE,
            template_output_dir=template_dir,
            workflow_output_dir=workflow_dir,
        )
        manifest = yaml.safe_load((workflow_dir / "manifest.yaml").read_text(encoding="utf-8"))

        case.assertEqual(result["raw_wildcard_dir"], str(raw_dir))
        case.assertEqual(result["special_character_file"], str(SPECIAL_CHARACTER_FILE))
        case.assertEqual(manifest["workflow_count"], 4)
        case.assertEqual(manifest["wildcard_mode"], f"runtime_first_pointer_to_{namespace}")
        case.assertEqual(manifest["special_character_relative_path"], f"{version}/Character.txt")
        case.assertTrue((raw_dir / version / "Character.txt").is_file())

        expected_tasks = {
            f"{version}_sfw_same.yaml",
            f"{version}_sfw_diff.yaml",
            f"{version}_nsfw_same.yaml",
            f"{version}_nsfw_diff.yaml",
        }
        case.assertEqual({row["output_yaml"] for row in result["workflows"]}, expected_tasks)
        case.assertTrue(all((workflow_dir / name).is_file() for name in expected_tasks))

        template_text = (
            template_dir / f"{version}_sfw_hybrid_compact_illustration.txt"
        ).read_text(encoding="utf-8")
        for token in expected_template_tokens:
            case.assertIn(token, template_text)
        case.assertIn(f"__{namespace}/{version}/Character__", template_text)
        case.assertNotIn(f"__{namespace}/characters/fgo__", template_text)
        case.assertNotIn(f"__{namespace}/runtime_sfw/Character__", template_text)

        for pair_mode in ("same", "diff"):
            task_path = workflow_dir / f"{version}_sfw_{pair_mode}.yaml"
            task = load_task_yaml(task_path)
            requests, compile_manifest, _ = compile_requests(task, task_path)
            expected_ckpt_control = "session_seed" if pair_mode == "same" else "image_index_seed"

            case.assertEqual(task["session_count"], SHARED_SESSION_COUNT)
            case.assertEqual(task["images"][0]["ckpt"]["seed_control"], expected_ckpt_control)
            case.assertEqual(task["images"][1]["ckpt"]["seed_control"], expected_ckpt_control)
            case.assertEqual(compile_manifest["request_count"], SHARED_SESSION_COUNT * 2)
            case.assertEqual(len(requests), SHARED_SESSION_COUNT * 2)
            case.assertTrue(all(row["workflow_name"] == SHARED_WORKFLOW_NAME for row in requests))
            case.assertTrue(all(f"{namespace}/{version}_sfw_" in row["prompt_generator_args"]["template"] for row in requests))
            case.assertTrue(all(f"__{namespace}/" not in row["positive_prompt"] for row in requests))

            options = task["images"][0]["ckpt"]["options"]
            case.assertAlmostEqual(_sum_group_weights(options, SHARED_TIER1_CHECKPOINTS), 1.0 / 3.0)
            case.assertAlmostEqual(_sum_group_weights(options, SHARED_CUSTOM_CHECKPOINTS), 1.0 / 3.0)
            case.assertAlmostEqual(_sum_group_weights(options, SHARED_REMAINING_CHECKPOINTS), 1.0 / 3.0)


class DpoWorkflowSv67TrialTests(unittest.TestCase):
    def test_generate_sv6_trials_uses_special_character_subset(self) -> None:
        _assert_sv_contract(
            self,
            generator=generate_sv6_trials,
            version="sv6",
            research_version="v6",
            raw_source_dir=RAW_V6_DIR,
            expected_template_tokens=("__research_v6/runtime_sfw/Rating__",),
        )

    def test_generate_sv7_trials_uses_special_character_subset(self) -> None:
        _assert_sv_contract(
            self,
            generator=generate_sv7_trials,
            version="sv7",
            research_version="v7",
            raw_source_dir=RAW_V7_DIR,
            expected_template_tokens=(
                "__research_v7/runtime_sfw/Rating__",
                "__research_v7/runtime_sfw/Body_Appearance__",
                "__research_v7/runtime_sfw/Attire_Accessory__",
            ),
        )

    def test_self_trained_checkpoint_family_resolution(self) -> None:
        illustration_families = {
            resolve_ckpt_family(path)
            for path in SELF_TRAINED_CHECKPOINTS
            if "illlustion_base" in path
        }
        animagine_families = {
            resolve_ckpt_family(path)
            for path in SELF_TRAINED_CHECKPOINTS
            if "Animagine_XL_4.0_base" in path
        }
        self.assertEqual(illustration_families, {"illustration"})
        self.assertEqual(animagine_families, {"sdxl_anime_base"})


if __name__ == "__main__":
    unittest.main()
