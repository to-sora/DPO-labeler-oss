from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_TEMPLATES_DIR, WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from generators._shared_task import SHARED_SESSION_COUNT, TemplateEntry, emit_version_matrix, write_yaml
from generators.generate_dpo_workflow_matrix_v8 import (
    DEFAULT_RAW_WILDCARD_DIR,
    FAMILIES,
    NEGATIVE_PROMPTS_BY_FAMILY,
    RUNTIME_SUBDIR_BY_SET,
    SET_VARIANTS,
    TEMPLATE_TOKEN_ORDER_BY_SET,
    _build_family_lines,
    _ordered_tokens,
    discover_source_files,
    load_lines,
    write_text,
)
from generators.sv_checkpoint_pools import describe_shared_checkpoint_pools


DEFAULT_SPECIAL_CHARACTER_FILE = WILDCARD_DIR / "custom_character_list.txt"
DEFAULT_TEMPLATE_DIR = PROMPT_TEMPLATES_DIR / "research_v8"
DEFAULT_WORKFLOW_DIR = WORKFLOWS_DIR / "dpo_workflow_sv8"
DEFAULT_GLOBAL_SEED_SV8 = 20260408
DEFAULT_SESSION_COUNT = SHARED_SESSION_COUNT
SPECIAL_CHARACTER_RELATIVE_PATH = Path("sv8") / "Character.txt"
WILDCARD_NAMESPACE = "research_v8"
TEMPLATE_FILENAME_PREFIX = "sv8"


def _token_for(relative_path: str | Path) -> str:
    path = Path(relative_path)
    return f"__{WILDCARD_NAMESPACE}/{path.with_suffix('').as_posix()}__"


def _special_character_choice(special_character_file: Path) -> str:
    if not special_character_file.is_file():
        raise ValueError(f"Missing required Sv8 character subset file: {special_character_file}")
    if not load_lines(special_character_file):
        raise ValueError(f"Sv8 character subset file is empty: {special_character_file}")
    return _token_for(SPECIAL_CHARACTER_RELATIVE_PATH)


def _inject_character(tokens: list[str], *, character_choice: str) -> list[str]:
    if len(tokens) < 2:
        return [*tokens, character_choice]
    return [tokens[0], tokens[1], character_choice, *tokens[2:]]


def build_template_lines(
    set_name: str,
    family: str,
    source_files: list[str],
    *,
    raw_dir: Path,
    special_character_file: Path,
) -> list[str]:
    common_order = list(TEMPLATE_TOKEN_ORDER_BY_SET[set_name])
    omitted_files = {"Character.txt", "semantic_class.txt"}
    if set_name == "sfw":
        omitted_files.add("NSFW_Nudity.txt")
    extras = [
        file_name
        for file_name in source_files
        if file_name not in common_order
        and file_name not in omitted_files
    ]
    body_tokens = _inject_character(
        _ordered_tokens(raw_dir, set_name, [*common_order, *extras]),
        character_choice=_special_character_choice(special_character_file),
    )

    if family == "illustration":
        return _build_family_lines(
            (
                "refined illustration",
                "lush character art",
                "polished digital painting",
                "high-detail visual novel art",
            ),
            tokens=body_tokens,
            suffixes=(
                "coherent composition",
                "clean rendering",
                "rich color separation",
                "focused storytelling",
            ),
        )

    if family == "sdxl_anime_base":
        return _build_family_lines(
            (
                "anime key visual",
                "stylized character illustration",
                "dramatic portrait",
                "clean anime scene",
            ),
            tokens=body_tokens,
            suffixes=(
                "high score, great score, absurdres",
                "high score, great score, absurdres",
                "high score, great score, absurdres",
                "high score, great score, absurdres",
            ),
        )

    if family == "pony":
        return _build_family_lines(
            (
                "score_9, score_8_up, score_7_up, score_6_up",
                "score_9, score_8_up, score_7_up, score_6_up",
                "score_9, score_8_up, score_7_up, score_6_up",
                "score_9, score_8_up, score_7_up, score_6_up",
            ),
            tokens=body_tokens,
            suffixes=(
                "sharp focus",
                "dynamic composition",
                "expressive pose",
                "detailed background",
            ),
        )

    if family == "realistic":
        return _build_family_lines(
            (
                "cinematic photo",
                "editorial portrait",
                "high-end fashion photo",
                "moody studio photo",
            ),
            tokens=body_tokens,
            suffixes=(
                "realistic lighting",
                "natural skin texture",
                "shallow depth of field",
                "grounded color grading",
            ),
        )

    raise ValueError(f"Unsupported family {family!r}")


def build_templates(
    template_output_dir: Path,
    *,
    raw_dir: Path,
    source_files: list[str],
    empty_files: list[str],
    special_character_file: Path,
) -> None:
    template_output_dir.mkdir(parents=True, exist_ok=True)
    for set_name in SET_VARIANTS:
        for family in FAMILIES:
            lines = build_template_lines(
                set_name,
                family,
                source_files,
                raw_dir=raw_dir,
                special_character_file=special_character_file,
            )
            write_text(
                template_output_dir / f"{TEMPLATE_FILENAME_PREFIX}_{set_name}_hybrid_compact_{family}.txt",
                lines,
            )

    readme_lines = [
        "# Sv8 Prompt Templates",
        "",
        "Sv8 reuses the V8 wildcard schema and inserts a curated character token after rating/count.",
        f"Special character source: {special_character_file}",
    ]
    if source_files:
        readme_lines.extend(["", "Active source files:"])
        readme_lines.extend(f"- {file_name}" for file_name in source_files)
    if empty_files:
        readme_lines.extend(["", "Empty source files skipped by the wildcard loader:"])
        readme_lines.extend(f"- {file_name}" for file_name in empty_files)
    write_text(template_output_dir / "README_sv8.md", readme_lines)


def make_prompt_args(*, set_name: str, template_root: str, wildcard_root: str) -> Dict[str, Any]:
    base = f"research_v8/{TEMPLATE_FILENAME_PREFIX}_{set_name}_hybrid_compact"
    return {
        "seed_control": "session_seed",
        "template": f"{base}_illustration",
        "template_by_ckpt_family": {
            "illustration": f"{base}_illustration",
            "sdxl_anime_base": f"{base}_sdxl_anime_base",
            "pony": f"{base}_pony",
            "realistic": f"{base}_realistic",
        },
        "template_root": template_root,
        "wildcard_root": wildcard_root,
        "negative_prompt": NEGATIVE_PROMPTS_BY_FAMILY["illustration"],
        "negative_prompt_by_ckpt_family": dict(NEGATIVE_PROMPTS_BY_FAMILY),
    }


def _build_entries(template_root: str, wildcard_root: str) -> list[TemplateEntry]:
    return [
        TemplateEntry(
            short_name=set_name,
            prompt_generator_name="wildcard_template_generator",
            prompt_args=make_prompt_args(set_name=set_name, template_root=template_root, wildcard_root=wildcard_root),
        )
        for set_name in SET_VARIANTS
    ]


def _stage_special_character_file(raw_dir: Path, character_file: Path) -> Path:
    if not character_file.is_file():
        raise ValueError(f"Missing shared character list: {character_file}")
    if not load_lines(character_file):
        raise ValueError(f"Shared character list is empty: {character_file}")
    staged_path = raw_dir / SPECIAL_CHARACTER_RELATIVE_PATH
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_text(character_file.read_text(encoding="utf-8"), encoding="utf-8")
    return staged_path


def generate_sv8_trials(
    *,
    raw_wildcard_dir: str | Path = DEFAULT_RAW_WILDCARD_DIR,
    special_character_file: str | Path = DEFAULT_SPECIAL_CHARACTER_FILE,
    template_output_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    workflow_output_dir: str | Path = DEFAULT_WORKFLOW_DIR,
    global_seed: int = DEFAULT_GLOBAL_SEED_SV8,
    session_count: int = DEFAULT_SESSION_COUNT,
) -> dict[str, Any]:
    raw_dir = Path(raw_wildcard_dir)
    character_file = Path(special_character_file)
    template_dir = Path(template_output_dir)
    workflow_dir = Path(workflow_output_dir)
    template_root = relative_root_path(workflow_dir, template_dir.parent)
    wildcard_root = relative_root_path(workflow_dir, raw_dir.parent)

    if not raw_dir.is_dir():
        raise ValueError(f"Missing required Sv8 wildcard source directory: {raw_dir}")

    active_source_files, empty_source_files = discover_source_files(raw_dir)
    if not active_source_files:
        raise ValueError(f"No non-empty wildcard txt files found in {raw_dir}")

    _stage_special_character_file(raw_dir, character_file)
    build_templates(
        template_dir,
        raw_dir=raw_dir,
        source_files=active_source_files,
        empty_files=empty_source_files,
        special_character_file=character_file,
    )

    entries = _build_entries(template_root, wildcard_root)
    workflow_rows = emit_version_matrix(
        version_tag="sv8",
        templates=entries,
        output_dir=workflow_dir,
        global_seed=global_seed,
    )

    write_yaml(
        workflow_dir / "manifest.yaml",
        {
            "workflow_count": len(workflow_rows),
            "global_seed": int(global_seed),
            "requested_session_count": int(session_count),
            "workflows": workflow_rows,
            "wildcard_source_dir": str(raw_dir),
            "special_character_file": str(character_file),
            "source_files": active_source_files,
            "empty_source_files": empty_source_files,
            "character_sources_enabled": True,
            "wildcard_mode": "runtime_first_pointer_to_research_v8",
            "runtime_dirs_by_set": dict(RUNTIME_SUBDIR_BY_SET),
            "special_character_relative_path": SPECIAL_CHARACTER_RELATIVE_PATH.as_posix(),
            "checkpoint_pools": describe_shared_checkpoint_pools(),
        },
    )

    return {
        "raw_wildcard_dir": str(raw_dir),
        "special_character_file": str(character_file),
        "template_output_dir": str(template_dir),
        "workflow_output_dir": str(workflow_dir),
        "active_source_files": active_source_files,
        "empty_source_files": empty_source_files,
        "character_sources_enabled": True,
        "workflows": workflow_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Sv8 workflow YAMLs with V8 family templates plus curated character injection."
    )
    parser.add_argument("--raw-wildcard-dir", default=str(DEFAULT_RAW_WILDCARD_DIR))
    parser.add_argument("--special-character-file", default=str(DEFAULT_SPECIAL_CHARACTER_FILE))
    parser.add_argument("--template-output-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--workflow-output-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED_SV8)
    parser.add_argument("--session-count", type=int, default=DEFAULT_SESSION_COUNT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = generate_sv8_trials(
        raw_wildcard_dir=args.raw_wildcard_dir,
        special_character_file=args.special_character_file,
        template_output_dir=args.template_output_dir,
        workflow_output_dir=args.workflow_output_dir,
        global_seed=args.global_seed,
        session_count=args.session_count,
    )
    print(f"Using pointer wildcard tree for Sv8 from: {result['raw_wildcard_dir']}")
    print(f"Using Sv8 character subset file: {result['special_character_file']}")
    print(f"Wrote Sv8 templates to: {result['template_output_dir']}")
    print(f"Wrote Sv8 workflows to: {result['workflow_output_dir']}")


if __name__ == "__main__":
    main()
