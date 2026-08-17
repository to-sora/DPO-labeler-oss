from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from generators._shared_task import TemplateEntry, emit_version_matrix, write_yaml
from generators.generate_dpo_workflow_matrix_v8 import NEGATIVE_PROMPTS_BY_FAMILY
from generators.sv_checkpoint_pools import describe_shared_checkpoint_pools


DEFAULT_OUTPUT_DIR = WORKFLOWS_DIR / "dpo_workflow_nwv2"
DEFAULT_CHARACTER_LIST = WILDCARD_DIR / "custom_character_list.txt"
DEFAULT_GLOBAL_SEED_NWV2 = 20260408

POSITIVE_PREFIX_BY_FAMILY: Dict[str, str] = {
    "illustration": "refined illustration, polished character art",
    "sdxl_anime_base": "anime key visual, high score",
    "pony": "score_9, score_8_up, score_7_up, score_6_up",
    "realistic": "cinematic photo, editorial lighting",
}

POSITIVE_SUFFIX_BY_FAMILY: Dict[str, str] = {
    "illustration": "",
    "sdxl_anime_base": "high score, great score, absurdres",
    "pony": "",
    "realistic": "sharp focus, realistic skin texture",
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


def build_nwv2_templates(*, character_list: str) -> List[TemplateEntry]:
    return [
        TemplateEntry(
            short_name="default",
            prompt_generator_name="non_wildcard_v2",
            prompt_args=make_prompt_args(character_list=character_list),
        ),
    ]


def generate_matrix(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    character_list: str | Path = DEFAULT_CHARACTER_LIST,
    global_seed: int = DEFAULT_GLOBAL_SEED_NWV2,
) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    char_path = Path(character_list)
    if not char_path.is_file():
        raise ValueError(f"Character list not found: {char_path}")

    char_arg = relative_root_path(out_dir, char_path)
    templates = build_nwv2_templates(character_list=char_arg)
    manifest = emit_version_matrix(
        version_tag="nwv2",
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
    parser = argparse.ArgumentParser(description="Generate non_wildcard_v2 DPO workflow YAMLs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--character-list", default=str(DEFAULT_CHARACTER_LIST))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED_NWV2)
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
