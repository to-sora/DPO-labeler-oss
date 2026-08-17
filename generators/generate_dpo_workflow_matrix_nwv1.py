from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from generators._shared_task import (
    SHARED_GLOBAL_SEED,
    TemplateEntry,
    emit_version_matrix,
    write_yaml,
)
from generators.sv_checkpoint_pools import describe_shared_checkpoint_pools
from generators.generate_dpo_workflow_matrix_v4 import NEGATIVE_PROMPTS_BY_FAMILY


DEFAULT_OUTPUT_DIR = WORKFLOWS_DIR / "dpo_workflow_nwv1"
DEFAULT_CHARACTER_LIST = WILDCARD_DIR / "custom_character_list.txt"

# Positive prefixes — verbatim from V6/V7 build_template_lines (first variant of each family)
# so nwv1 output is directly comparable to V6/V7 across checkpoint families.
POSITIVE_PREFIX_BY_FAMILY: Dict[str, str] = {
    "illustration":    "masterpiece, best quality, very aesthetic, absurdres",
    "sdxl_anime_base": "masterpiece, best quality, very aesthetic, absurdres",
    "pony":            "score_9, score_8_up, score_7_up, score_6_up",
    "realistic":       "photo, cinematic portrait",
}

# Positive suffix — empty by default; only sdxl_anime_base needs the trailing
# booster that V6/V7 places at the end of its sdxl_anime_base body.
POSITIVE_SUFFIX_BY_FAMILY: Dict[str, str] = {
    "illustration":    "",
    "sdxl_anime_base": "masterpiece, best quality, very aesthetic, absurdres",
    "pony":            "",
    "realistic":       "",
}


def make_prompt_args(*, character_list: str) -> Dict[str, Any]:
    return {
        "character_list": character_list,
        "positive_prefix": POSITIVE_PREFIX_BY_FAMILY["illustration"],
        "positive_suffix": POSITIVE_SUFFIX_BY_FAMILY["illustration"],
        "positive_prefix_by_ckpt_family": dict(POSITIVE_PREFIX_BY_FAMILY),
        "positive_suffix_by_ckpt_family": dict(POSITIVE_SUFFIX_BY_FAMILY),
        "negative_prompt": NEGATIVE_PROMPTS_BY_FAMILY["illustration"],
        "negative_prompt_by_ckpt_family": dict(NEGATIVE_PROMPTS_BY_FAMILY),
    }


def build_nwv1_templates(*, character_list: str) -> List[TemplateEntry]:
    return [
        TemplateEntry(
            short_name="default",
            prompt_generator_name="non_wildcard_v1",
            prompt_args=make_prompt_args(character_list=character_list),
        ),
    ]


def generate_matrix(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    character_list: str | Path = DEFAULT_CHARACTER_LIST,
    global_seed: int = SHARED_GLOBAL_SEED,
) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    char_path = Path(character_list)
    if not char_path.is_file():
        raise ValueError(f"Character list not found: {char_path}")

    # Pass the character list as a workflow-relative path so the runtime
    # resolves it correctly regardless of where the task YAML is loaded from.
    char_arg = relative_root_path(out_dir, char_path)

    templates = build_nwv1_templates(character_list=char_arg)
    manifest = emit_version_matrix(
        version_tag="nwv1",
        templates=templates,
        output_dir=out_dir,
        global_seed=global_seed,
    )
    write_yaml(
        out_dir / "manifest.yaml",
        {
            "workflow_count": len(manifest),
            "global_seed": int(global_seed),
            "character_list": str(char_path),
            "checkpoint_pools": describe_shared_checkpoint_pools(),
            "workflows": manifest,
        },
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate non_wildcard_v1 DPO workflow YAMLs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--character-list", default=str(DEFAULT_CHARACTER_LIST))
    parser.add_argument("--global-seed", type=int, default=SHARED_GLOBAL_SEED)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = generate_matrix(
        args.output_dir,
        character_list=args.character_list,
        global_seed=args.global_seed,
    )
    print(f"Wrote {len(rows)} workflow YAML files to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
