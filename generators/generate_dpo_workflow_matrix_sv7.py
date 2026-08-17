from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_TEMPLATES_DIR, WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from generators._shared_task import (
    SHARED_SESSION_COUNT,
    TemplateEntry,
    emit_version_matrix,
    write_yaml as _shared_write_yaml,
)
from generators.generate_dpo_workflow_matrix_v4 import NEGATIVE_PROMPTS_BY_FAMILY
from generators.generate_dpo_workflow_matrix_v7 import discover_source_files, load_lines
from generators.sv_checkpoint_pools import describe_shared_checkpoint_pools


DEFAULT_RAW_WILDCARD_DIR = WILDCARD_DIR / "research_v7"
# Shared character list — single source of truth for SV6 and SV7.
DEFAULT_SPECIAL_CHARACTER_FILE = WILDCARD_DIR / "custom_character_list.txt"
# SV7 emits its templates alongside V7 so ``template_root`` / ``wildcard_root``
# are identical to V7's.
DEFAULT_TEMPLATE_DIR = PROMPT_TEMPLATES_DIR / "research_v7"
DEFAULT_WORKFLOW_DIR = WORKFLOWS_DIR / "dpo_workflow_sv7"
DEFAULT_GLOBAL_SEED_SV7 = 20260404
DEFAULT_SESSION_COUNT = SHARED_SESSION_COUNT
SET_VARIANTS = ("sfw", "nsfw")
RUNTIME_SUBDIR_BY_SET = {
    "sfw": "runtime_sfw",
    "nsfw": "runtime_nsfw",
}
FAMILIES = ("illustration", "sdxl_anime_base", "pony", "realistic")
SPECIAL_CHARACTER_RELATIVE_PATH = Path("sv7") / "Character.txt"
WILDCARD_NAMESPACE = "research_v7"
TEMPLATE_FILENAME_PREFIX = "sv7"


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _token_for(relative_path: str | Path) -> str:
    path = Path(relative_path)
    return f"__{WILDCARD_NAMESPACE}/{path.with_suffix('').as_posix()}__"


def _resolve_runtime_first_relative_path(raw_dir: Path, set_name: str, file_name: str) -> Path | None:
    runtime_path = raw_dir / RUNTIME_SUBDIR_BY_SET[set_name] / file_name
    if runtime_path.is_file() and load_lines(runtime_path):
        return Path(RUNTIME_SUBDIR_BY_SET[set_name]) / file_name

    top_level_path = raw_dir / file_name
    if top_level_path.is_file() and load_lines(top_level_path):
        return Path(file_name)
    return None


def _ordered_tokens(raw_dir: Path, set_name: str, names: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for file_name in names:
        relative_path = _resolve_runtime_first_relative_path(raw_dir, set_name, file_name)
        if relative_path is not None:
            tokens.append(_token_for(relative_path))
    return tokens


def _special_character_choice(special_character_file: Path) -> str:
    if not special_character_file.is_file():
        raise ValueError(f"Missing required Sv7 character subset file: {special_character_file}")
    if not load_lines(special_character_file):
        raise ValueError(f"Sv7 character subset file is empty: {special_character_file}")
    return _token_for(SPECIAL_CHARACTER_RELATIVE_PATH)


def build_template_lines(
    set_name: str,
    family: str,
    source_files: list[str],
    *,
    raw_dir: Path,
    special_character_file: Path,
) -> list[str]:
    character_choice = _special_character_choice(special_character_file)
    common_order = [
        "Rating.txt",
        "Person_Count_Identity.txt",
        "Species_Fantasy_Trait.txt",
        "Body_Appearance.txt",
        "Attire_Accessory.txt",
        "NSFW_Nudity.txt",
        "Expression_Gaze.txt",
        "Action_Pose.txt",
        "Composition_Camera.txt",
        "Scene_Background.txt",
        "Object_Prop.txt",
        "Relationship_Interaction.txt",
        "Other_General.txt",
        "Style_Medium.txt",
    ]
    extras = [
        file_name
        for file_name in source_files
        if file_name not in common_order
        and file_name != "Character.txt"
    ]

    if family == "illustration":
        prefixes = [
            "masterpiece, best quality, very aesthetic, absurdres",
            "masterpiece, amazing quality, very aesthetic, absurdres",
            "masterpiece, best quality, newest, absurdres",
            "masterpiece, best quality, vivid illustration, absurdres",
        ]
        left_names = common_order[:10]
        right_names = common_order[10:] + extras
        return [
            ", ".join(
                [prefix]
                + _ordered_tokens(raw_dir, set_name, left_names[:2])
                + [character_choice]
                + _ordered_tokens(raw_dir, set_name, left_names[2:])
                + ["BREAK"]
                + _ordered_tokens(raw_dir, set_name, right_names)
            )
            for prefix in prefixes
        ]

    if family == "sdxl_anime_base":
        mid_labels = [
            "original character portrait",
            "adult glamour portrait",
            "late-night illustration",
            "fashion-focused portrait",
        ]
        body_names = common_order + extras
        return [
            ", ".join(
                _ordered_tokens(raw_dir, set_name, body_names[:2])
                + [character_choice]
                + _ordered_tokens(raw_dir, set_name, body_names[2:])
                + [label, "masterpiece", "best quality", "very aesthetic", "absurdres"]
            )
            for label in mid_labels
        ]

    if family == "pony":
        prefixes = [
            "score_9, score_8_up, score_7_up, score_6_up",
            "score_9, score_8_up, score_7_up, score_6_up, masterpiece",
            "score_9, score_8_up, score_7_up, score_6_up, best quality",
            "score_9, score_8_up, score_7_up, score_6_up, raw",
        ]
        body_names = common_order + extras
        return [
            ", ".join(
                [prefix]
                + _ordered_tokens(raw_dir, set_name, body_names[:2])
                + [character_choice]
                + _ordered_tokens(raw_dir, set_name, body_names[2:])
            )
            for prefix in prefixes
        ]

    if family == "realistic":
        prefixes = [
            "photo, cinematic portrait",
            "photo, moody studio portrait",
            "photo, soft light portrait",
            "photo, late-night glamour portrait",
        ]
        body_names = common_order + extras
        return [
            ", ".join(
                [prefix]
                + _ordered_tokens(raw_dir, set_name, body_names[:2])
                + [character_choice]
                + _ordered_tokens(raw_dir, set_name, body_names[2:])
            )
            for prefix in prefixes
        ]

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
        "# Sv7 Prompt Templates",
        "",
        "These templates point at the existing `research_v7` wildcard tree.",
        "Runtime-filtered category files still come from `runtime_sfw/` and `runtime_nsfw/`.",
        "Body and attire stay active, and character selection is narrowed to:",
        f"- {special_character_file}",
    ]
    if source_files:
        readme_lines.extend(["", "Active top-level source files:"])
        readme_lines.extend(f"- {file_name}" for file_name in source_files)
    if empty_files:
        readme_lines.extend(["", "Empty top-level source files skipped by the wildcard loader:"])
        readme_lines.extend(f"- {file_name}" for file_name in empty_files)
    write_text(template_output_dir / "README.md", readme_lines)


def make_prompt_args(*, set_name: str, template_root: str, wildcard_root: str) -> Dict[str, Any]:
    base = f"research_v7/{TEMPLATE_FILENAME_PREFIX}_{set_name}_hybrid_compact"
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
        "negative_prompt_by_ckpt_family": {
            "illustration": NEGATIVE_PROMPTS_BY_FAMILY["illustration"],
            "sdxl_anime_base": NEGATIVE_PROMPTS_BY_FAMILY["sdxl_anime_base"],
            "pony": "score_4, score_5, score_6, lowres, blurry, bad anatomy, distorted hands, distorted fingers, extra fingers, watermark, text, logo, duplicate, poorly drawn face, 3d",
            "realistic": NEGATIVE_PROMPTS_BY_FAMILY["realistic"],
        },
    }


def _build_entries(template_root: str, wildcard_root: str) -> list[TemplateEntry]:
    return [
        TemplateEntry(
            short_name=set_name,
            prompt_generator_name="wildcard_template_generator",
            prompt_args=make_prompt_args(
                set_name=set_name, template_root=template_root, wildcard_root=wildcard_root
            ),
        )
        for set_name in SET_VARIANTS
    ]


def _stage_special_character_file(raw_dir: Path, character_file: Path) -> Path:
    """Materialise the shared character list into the research_v7 wildcard tree."""
    if not character_file.is_file():
        raise ValueError(f"Missing shared character list: {character_file}")
    if not load_lines(character_file):
        raise ValueError(f"Shared character list is empty: {character_file}")
    staged_path = raw_dir / SPECIAL_CHARACTER_RELATIVE_PATH
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_text(character_file.read_text(encoding="utf-8"), encoding="utf-8")
    return staged_path


def generate_sv7_trials(
    *,
    raw_wildcard_dir: str | Path = DEFAULT_RAW_WILDCARD_DIR,
    special_character_file: str | Path = DEFAULT_SPECIAL_CHARACTER_FILE,
    template_output_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    workflow_output_dir: str | Path = DEFAULT_WORKFLOW_DIR,
    global_seed: int = DEFAULT_GLOBAL_SEED_SV7,
    session_count: int = DEFAULT_SESSION_COUNT,
) -> dict[str, Any]:
    raw_dir = Path(raw_wildcard_dir)
    character_file = Path(special_character_file)
    template_dir = Path(template_output_dir)
    workflow_dir = Path(workflow_output_dir)
    template_root = relative_root_path(workflow_dir, template_dir.parent)
    wildcard_root = relative_root_path(workflow_dir, raw_dir.parent)

    if not raw_dir.is_dir():
        raise ValueError(f"Missing required Sv7 wildcard source directory: {raw_dir}")

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
        version_tag="sv7", templates=entries, output_dir=workflow_dir
    )

    _shared_write_yaml(
        workflow_dir / "manifest.yaml",
        {
            "workflow_count": len(workflow_rows),
            "workflows": workflow_rows,
            "wildcard_source_dir": str(raw_dir),
            "special_character_file": str(character_file),
            "source_files": active_source_files,
            "empty_source_files": empty_source_files,
            "character_sources_enabled": True,
            "wildcard_mode": "runtime_first_pointer_to_research_v7",
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
        description="Generate Sv7 trial assets that point at the existing runtime-first V7 wildcard tree."
    )
    parser.add_argument("--raw-wildcard-dir", default=str(DEFAULT_RAW_WILDCARD_DIR))
    parser.add_argument("--special-character-file", default=str(DEFAULT_SPECIAL_CHARACTER_FILE))
    parser.add_argument("--template-output-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--workflow-output-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED_SV7)
    parser.add_argument("--session-count", type=int, default=DEFAULT_SESSION_COUNT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = generate_sv7_trials(
        raw_wildcard_dir=args.raw_wildcard_dir,
        special_character_file=args.special_character_file,
        template_output_dir=args.template_output_dir,
        workflow_output_dir=args.workflow_output_dir,
        global_seed=args.global_seed,
        session_count=args.session_count,
    )
    print(f"Using pointer wildcard tree for Sv7 from: {result['raw_wildcard_dir']}")
    print(f"Using Sv7 character subset file: {result['special_character_file']}")
    print(f"Wrote Sv7 templates to: {result['template_output_dir']}")
    print(f"Wrote Sv7 workflows to: {result['workflow_output_dir']}")


if __name__ == "__main__":
    main()
