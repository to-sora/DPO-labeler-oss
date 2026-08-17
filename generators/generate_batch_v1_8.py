from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_TEMPLATES_DIR, WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from generators._shared_task import (
    SHARED_PAIR_DROPOUT_CHANCE,
    SHARED_SEGMENT_DROPOUT_PROB,
    SHARED_SESSION_COUNT,
    build_shared_pair_task,
    write_yaml,
)
from generators.generate_dpo_workflow_matrix import make_template_args as make_v1_template_args
from generators.generate_dpo_workflow_matrix_v2 import make_template_args as make_v2_template_args
from generators.generate_dpo_workflow_matrix_v3 import make_template_args as make_v3_template_args
from generators.generate_dpo_workflow_matrix_v4 import make_template_args as make_v4_template_args
from generators.generate_dpo_workflow_matrix_v5 import make_template_args as make_v5_template_args
from generators.generate_dpo_workflow_matrix_v6 import make_prompt_args as make_v6_prompt_args
from generators.generate_dpo_workflow_matrix_v7 import make_prompt_args as make_v7_prompt_args
from generators.generate_dpo_workflow_matrix_v8 import make_prompt_args as make_v8_prompt_args
from generators.sv_checkpoint_pools import describe_shared_checkpoint_pools


DEFAULT_OUTPUT_DIR = WORKFLOWS_DIR / "batch_v1_8"
DEFAULT_GLOBAL_SEED = 20260408
DEFAULT_SESSION_COUNT = SHARED_SESSION_COUNT


def _v1_args(template_name: str, *, template_root: str, wildcard_root: str) -> Dict[str, Any]:
    return make_v1_template_args(template_name, template_root=template_root, wildcard_root=wildcard_root)


def _vN_args(builder: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    def _wrapped(template_name: str, *, template_root: str, wildcard_root: str) -> Dict[str, Any]:
        return builder(template_name, template_root=template_root, wildcard_root=wildcard_root)

    return _wrapped


SHORT_NAMES: Dict[str, Dict[str, str]] = {
    "v1": {"mix_gpt": "gpt", "mix_qwen": "qwen"},
    "v2": {"hybrid_compact": "compact", "hybrid_cinematic": "cinematic"},
    "v3": {"hybrid_compact": "compact", "hybrid_cinematic": "cinematic"},
    "v4": {"hybrid_compact": "compact", "hybrid_cinematic": "cinematic"},
    "v5": {"hybrid_compact": "compact", "hybrid_cinematic": "cinematic"},
    "v6": {"sfw": "sfw", "nsfw": "nsfw"},
    "v7": {"sfw": "sfw", "nsfw": "nsfw"},
    "v8": {"sfw": "sfw", "nsfw": "nsfw"},
}

VERSION_TEMPLATE_BUILDERS: tuple[tuple[str, tuple[str, ...], Callable[..., Dict[str, Any]]], ...] = (
    ("v1", ("mix_gpt", "mix_qwen"), _v1_args),
    ("v2", ("hybrid_compact", "hybrid_cinematic"), _vN_args(make_v2_template_args)),
    ("v3", ("hybrid_compact", "hybrid_cinematic"), _vN_args(make_v3_template_args)),
    ("v4", ("hybrid_compact", "hybrid_cinematic"), _vN_args(make_v4_template_args)),
    ("v5", ("hybrid_compact", "hybrid_cinematic"), _vN_args(make_v5_template_args)),
    ("v6", ("sfw", "nsfw"), _vN_args(make_v6_prompt_args)),
    ("v7", ("sfw", "nsfw"), _vN_args(make_v7_prompt_args)),
    ("v8", ("sfw", "nsfw"), _vN_args(make_v8_prompt_args)),
)


def generate_batch_v1_8(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    global_seed: int = DEFAULT_GLOBAL_SEED,
) -> list[Dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    template_root = relative_root_path(output_path, PROMPT_TEMPLATES_DIR)
    wildcard_root = relative_root_path(output_path, WILDCARD_DIR)

    rows: list[Dict[str, Any]] = []
    for version_name, templates, args_builder in VERSION_TEMPLATE_BUILDERS:
        for template_name in templates:
            short = SHORT_NAMES[version_name][template_name]
            prompt_args = args_builder(template_name, template_root=template_root, wildcard_root=wildcard_root)
            for pair_mode in ("same", "diff"):
                task_name = f"b18_{version_name}_{short}_{pair_mode}"
                payload = build_shared_pair_task(
                    task_name=task_name,
                    pair_mode=pair_mode,
                    prompt_generator_name="wildcard_template_generator",
                    prompt_args_image1=prompt_args,
                    session_count=DEFAULT_SESSION_COUNT,
                    global_seed=global_seed,
                )
                file_name = f"{task_name}.yaml"
                write_yaml(output_path / file_name, payload)
                rows.append(
                    {
                        "version": version_name,
                        "template": template_name,
                        "short": short,
                        "pair_mode": pair_mode,
                        "file": file_name,
                        "task_name": task_name,
                    }
                )

    write_yaml(
        output_path / "manifest.yaml",
        {
            "workflow_count": len(rows),
            "global_seed": int(global_seed),
            "session_count": DEFAULT_SESSION_COUNT,
            "random_segment_dropout": {
                "pair_chance": float(SHARED_PAIR_DROPOUT_CHANCE),
                "segment_prob": float(SHARED_SEGMENT_DROPOUT_PROB),
            },
            "checkpoint_pools": describe_shared_checkpoint_pools(),
            "rows": rows,
        },
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate flat batch_v1_8 workflow YAMLs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    rows = generate_batch_v1_8(args.output_dir, global_seed=args.global_seed)
    print(f"Generated {len(rows)} workflow YAMLs in {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
