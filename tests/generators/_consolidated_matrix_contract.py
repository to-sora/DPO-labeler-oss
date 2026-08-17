from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml
from generators._shared_task import (
    SHARED_CFG_VALUES,
    SHARED_GLOBAL_SEED,
    SHARED_PAIR_DROPOUT_CHANCE,
    SHARED_RESOLUTION_VALUES,
    SHARED_RESOLUTION_WEIGHTS,
    SHARED_SEGMENT_DROPOUT_PROB,
    SHARED_SESSION_COUNT,
    SHARED_STEPS_VALUES,
    SHARED_WORKFLOW_NAME,
)
from generators.sv_checkpoint_pools import (
    SHARED_CUSTOM_CHECKPOINTS,
    SHARED_REMAINING_CHECKPOINTS,
    SHARED_TIER1_CHECKPOINTS,
)


MatrixGenerator = Callable[[str | Path], list[dict[str, Any]]]


def _weight_for(options: list[dict[str, Any]], values: Sequence[str]) -> float:
    value_set = set(values)
    return sum(float(option["weight"]) for option in options if option["value"] in value_set)


def assert_consolidated_matrix(
    case: unittest.TestCase,
    *,
    generate_matrix: MatrixGenerator,
    version_tag: str,
    template_short_names: Sequence[str],
    template_prefix: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / version_tag
        rows = generate_matrix(output_dir)
        manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text(encoding="utf-8"))

        expected_names = [
            f"{version_tag}_{short_name}_{pair_mode}"
            for short_name in template_short_names
            for pair_mode in ("same", "diff")
        ]
        case.assertEqual([row["task_name"] for row in rows], expected_names)
        case.assertEqual(manifest["workflow_count"], len(expected_names))
        case.assertEqual(manifest["workflows"], rows)
        case.assertEqual(
            {path.name for path in output_dir.glob("*.yaml")},
            {"manifest.yaml", *(f"{name}.yaml" for name in expected_names)},
        )

        for row in rows:
            task_path = output_dir / row["output_yaml"]
            task = load_task_yaml(task_path)
            expected_seed_control = "session_seed" if row["pair_mode"] == "same" else "image_index_seed"

            case.assertEqual(task["task_name"], row["task_name"])
            case.assertEqual(task["global_seed"], SHARED_GLOBAL_SEED)
            case.assertEqual(task["session_count"], SHARED_SESSION_COUNT)
            case.assertEqual(len(task["images"]), 2)

            for image in task["images"]:
                case.assertEqual(image["workflow_name"], SHARED_WORKFLOW_NAME)
                case.assertNotIn("workflow_kwargs", image)
                case.assertFalse(image["lora_stack_config"]["toggle"])
                case.assertEqual(image["ckpt"]["seed_control"], expected_seed_control)

                prompt_args = image["prompt_generator"]["args"]
                case.assertEqual(prompt_args["seed_control"], "session_seed")
                case.assertEqual(
                    prompt_args["random_segment_dropout_pair_chance"],
                    SHARED_PAIR_DROPOUT_CHANCE,
                )
                case.assertEqual(
                    prompt_args["random_segment_dropout_segment_prob"],
                    SHARED_SEGMENT_DROPOUT_PROB,
                )
                case.assertEqual(
                    prompt_args["seed_control_random_segment_dropout"],
                    "image_index_seed",
                )

                sample = image["sample"]
                case.assertEqual(sample["generation_seed_control"], "image_index_seed")
                case.assertEqual([option["value"] for option in sample["steps"]["options"]], SHARED_STEPS_VALUES)
                case.assertEqual([option["value"] for option in sample["cfg"]["options"]], SHARED_CFG_VALUES)
                case.assertEqual(
                    [option["value"] for option in sample["width"]["options"]],
                    SHARED_RESOLUTION_VALUES,
                )
                case.assertEqual(
                    [option["weight"] for option in sample["width"]["options"]],
                    SHARED_RESOLUTION_WEIGHTS,
                )

            options = task["images"][0]["ckpt"]["options"]
            case.assertAlmostEqual(_weight_for(options, SHARED_TIER1_CHECKPOINTS), 1.0 / 3.0)
            case.assertAlmostEqual(_weight_for(options, SHARED_CUSTOM_CHECKPOINTS), 1.0 / 3.0)
            case.assertAlmostEqual(_weight_for(options, SHARED_REMAINING_CHECKPOINTS), 1.0 / 3.0)

            requests, compile_manifest, _ = compile_requests(task, task_path)
            case.assertEqual(compile_manifest["request_count"], SHARED_SESSION_COUNT * 2)
            case.assertEqual(len(requests), SHARED_SESSION_COUNT * 2)
            case.assertTrue(all(request["workflow_name"] == SHARED_WORKFLOW_NAME for request in requests))
            case.assertTrue(all(request["prompt_seed_control"] == "session_seed" for request in requests))
            case.assertTrue(
                all(template_prefix in request["prompt_generator_args"]["template"] for request in requests)
            )
