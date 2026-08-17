from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

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


DEFAULT_RAW_WILDCARD_DIR = WILDCARD_DIR / "research_v6"
DEFAULT_WILDCARD_DIR = WILDCARD_DIR / "research_v6"
DEFAULT_TEMPLATE_DIR = PROMPT_TEMPLATES_DIR / "research_v6"
DEFAULT_WORKFLOW_DIR = WORKFLOWS_DIR / "dpo_workflow_v6"
DEFAULT_GLOBAL_SEED_V6 = 20260328
DEFAULT_SESSION_COUNT = SHARED_SESSION_COUNT
DEFAULT_STEPS = 28
TEMPLATE_OMITTED_SOURCE_FILES = ("Body_Appearance.txt", "Attire_Accessory.txt")
CHARACTER_SOURCES_ENABLED = True

SET_VARIANTS = ("sfw", "nsfw")
RUNTIME_SUBDIR_BY_SET = {
    "sfw": "runtime_sfw",
    "nsfw": "runtime_nsfw",
}
CROSS_MODEL_SET_NAME = SET_VARIANTS[0]
FAMILIES = ("illustration", "sdxl_anime_base", "pony", "realistic")
CHARACTER_PACK_FILES = (
    "fgo.txt",
    "sao.txt",
    "ggo.txt",
    "zzz.txt",
    "genshin_impact.txt",
    "blue_archive.txt",
    "nikke.txt",
    "arknights.txt",
    "honkai_star_rail.txt",
    "azur_lane.txt",
    "punishing_gray_raven.txt",
)
TOP_LEVEL_TXT_ORDER = (
    "Action_Pose.txt",
    "Attire_Accessory.txt",
    "Body_Appearance.txt",
    "Character.txt",
    "Composition_Camera.txt",
    "Expression_Gaze.txt",
    "NSFW_Nudity.txt",
    "Object_Prop.txt",
    "Other_General.txt",
    "Person_Count_Identity.txt",
    "Rating.txt",
    "Relationship_Interaction.txt",
    "Scene_Background.txt",
    "Species_Fantasy_Trait.txt",
    "Style_Medium.txt",
    "semantic_class.txt",
)


def option_spec(seed_control: str, values: Iterable[Any], *, weight: float = 1.0) -> Dict[str, Any]:
    return {
        "seed_control": seed_control,
        "options": [{"value": value, "weight": weight} for value in values],
    }


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
    return deduped


def _sort_source_paths(paths: Iterable[Path]) -> list[Path]:
    order_map = {name: index for index, name in enumerate(TOP_LEVEL_TXT_ORDER)}
    return sorted(paths, key=lambda path: (order_map.get(path.name, len(order_map)), path.name.casefold()))


def discover_source_files(raw_dir: Path) -> tuple[list[str], list[str]]:
    txt_paths = _sort_source_paths(raw_dir.glob("*.txt"))
    active_files: list[str] = []
    empty_files: list[str] = []
    for path in txt_paths:
        if load_lines(path):
            active_files.append(path.name)
        else:
            empty_files.append(path.name)
    return active_files, empty_files


def mirror_raw_sources(*, raw_dir: Path, wildcard_output_dir: Path) -> list[str]:
    if raw_dir.resolve() == wildcard_output_dir.resolve():
        return []

    if wildcard_output_dir.exists():
        shutil.rmtree(wildcard_output_dir)
    shutil.copytree(raw_dir, wildcard_output_dir)
    return [
        str(path.relative_to(raw_dir))
        for path in sorted(raw_dir.rglob("*"))
        if path.is_file()
    ]


def _token_for(relative_path: str | Path) -> str:
    path = Path(relative_path)
    return f"__research_v6/{path.with_suffix('').as_posix()}__"


def _runtime_source_path(raw_dir: Path, set_name: str, file_name: str) -> Path:
    return raw_dir / RUNTIME_SUBDIR_BY_SET[set_name] / file_name


def _resolve_runtime_first_relative_path(raw_dir: Path, set_name: str, file_name: str) -> Path | None:
    runtime_path = _runtime_source_path(raw_dir, set_name, file_name)
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


def _character_source_choice(raw_dir: Path, set_name: str) -> str:
    sources: list[str] = []
    character_relative_path = _resolve_runtime_first_relative_path(raw_dir, set_name, "Character.txt")
    if character_relative_path is not None:
        sources.append(_token_for(character_relative_path))

    characters_dir = raw_dir / "characters"
    for file_name in CHARACTER_PACK_FILES:
        path = characters_dir / file_name
        if not path.is_file():
            raise ValueError(f"Missing required V6 character pack: {path}")
        if not load_lines(path):
            raise ValueError(f"Character pack is empty: {path}")
        sources.append(_token_for(Path("characters") / file_name))

    if not sources:
        raise ValueError("No character sources available for V6 templates")
    if len(sources) == 1:
        return sources[0]
    return "{" + "|".join(sources) + "}"


def build_template_lines(set_name: str, family: str, source_files: list[str], *, raw_dir: Path) -> list[str]:
    character_choice = _character_source_choice(raw_dir, set_name)
    common_order = [
        "Rating.txt",
        "Person_Count_Identity.txt",
        "Species_Fantasy_Trait.txt",
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
        and file_name not in TEMPLATE_OMITTED_SOURCE_FILES
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


def build_templates(template_output_dir: Path, *, raw_dir: Path, source_files: list[str], empty_files: list[str]) -> None:
    template_output_dir.mkdir(parents=True, exist_ok=True)
    for set_name in SET_VARIANTS:
        for family in FAMILIES:
            lines = build_template_lines(set_name, family, source_files, raw_dir=raw_dir)
            write_text(template_output_dir / f"{set_name}_hybrid_compact_{family}.txt", lines)

    readme_lines = [
        "# V6 Prompt Templates",
        "",
        "These templates use `runtime_sfw/` and `runtime_nsfw/` first, with top-level fallback only",
        "for files not present in the runtime pack.",
        "Template assembly keeps runtime-filtered character sources active and falls back to:",
        "- characters/*",
        "- semantic_class.txt",
        "",
        "Templates still omit:",
        "- Body_Appearance.txt",
        "- Attire_Accessory.txt",
    ]
    if source_files:
        readme_lines.extend(["", "Active source files:"])
        readme_lines.extend(f"- {file_name}" for file_name in source_files)
    if empty_files:
        readme_lines.extend(["", "Empty source files skipped by the wildcard loader:"])
        readme_lines.extend(f"- {file_name}" for file_name in empty_files)
    write_text(template_output_dir / "README.md", readme_lines)


def make_prompt_args(set_name: str, *, template_root: str, wildcard_root: str) -> Dict[str, Any]:
    return {
        "seed_control": "session_seed",
        "template": f"research_v6/{set_name}_hybrid_compact_illustration",
        "template_by_ckpt_family": {
            "illustration": f"research_v6/{set_name}_hybrid_compact_illustration",
            "sdxl_anime_base": f"research_v6/{set_name}_hybrid_compact_sdxl_anime_base",
            "pony": f"research_v6/{set_name}_hybrid_compact_pony",
            "realistic": f"research_v6/{set_name}_hybrid_compact_realistic",
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
                set_name, template_root=template_root, wildcard_root=wildcard_root
            ),
        )
        for set_name in SET_VARIANTS
    ]


def generate_v6_trials(
    *,
    raw_wildcard_dir: str | Path = DEFAULT_RAW_WILDCARD_DIR,
    wildcard_output_dir: str | Path = DEFAULT_WILDCARD_DIR,
    template_output_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    workflow_output_dir: str | Path = DEFAULT_WORKFLOW_DIR,
    global_seed: int = DEFAULT_GLOBAL_SEED_V6,
    session_count: int = DEFAULT_SESSION_COUNT,
) -> dict[str, Any]:
    raw_dir = Path(raw_wildcard_dir)
    wildcard_dir = Path(wildcard_output_dir)
    template_dir = Path(template_output_dir)
    workflow_dir = Path(workflow_output_dir)
    template_root = relative_root_path(workflow_dir, template_dir.parent)
    wildcard_root = relative_root_path(workflow_dir, wildcard_dir.parent)

    if not raw_dir.is_dir():
        raise ValueError(f"Missing required V6 source directory: {raw_dir}")

    active_source_files, empty_source_files = discover_source_files(raw_dir)
    if not active_source_files:
        raise ValueError(f"No non-empty wildcard txt files found in {raw_dir}")

    mirrored_files = mirror_raw_sources(raw_dir=raw_dir, wildcard_output_dir=wildcard_dir)
    build_templates(template_dir, raw_dir=raw_dir, source_files=active_source_files, empty_files=empty_source_files)

    entries = _build_entries(template_root, wildcard_root)
    workflow_rows = emit_version_matrix(
        version_tag="v6", templates=entries, output_dir=workflow_dir
    )

    _shared_write_yaml(
        workflow_dir / "manifest.yaml",
        {
            "workflow_count": len(workflow_rows),
            "workflows": workflow_rows,
            "source_files": active_source_files,
            "empty_source_files": empty_source_files,
            "mirrored_files": mirrored_files,
            "template_omitted_source_files": list(TEMPLATE_OMITTED_SOURCE_FILES),
            "character_sources_enabled": CHARACTER_SOURCES_ENABLED,
            "wildcard_mode": "runtime_first_with_top_level_fallback",
            "runtime_dirs_by_set": dict(RUNTIME_SUBDIR_BY_SET),
            "top_level_template_fallbacks": [
                "characters/*",
                "semantic_class.txt",
            ],
        },
    )

    return {
        "raw_wildcard_dir": str(raw_dir),
        "wildcard_output_dir": str(wildcard_dir),
        "template_output_dir": str(template_dir),
        "workflow_output_dir": str(workflow_dir),
        "active_source_files": active_source_files,
        "empty_source_files": empty_source_files,
        "mirrored_files": mirrored_files,
        "template_omitted_source_files": list(TEMPLATE_OMITTED_SOURCE_FILES),
        "character_sources_enabled": CHARACTER_SOURCES_ENABLED,
        "workflows": workflow_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate V6 trial assets from runtime-first V6 wildcard packs with top-level fallback."
    )
    parser.add_argument("--raw-wildcard-dir", default=str(DEFAULT_RAW_WILDCARD_DIR))
    parser.add_argument("--wildcard-output-dir", default=str(DEFAULT_WILDCARD_DIR))
    parser.add_argument("--template-output-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--workflow-output-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED_V6)
    parser.add_argument("--session-count", type=int, default=DEFAULT_SESSION_COUNT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = generate_v6_trials(
        raw_wildcard_dir=args.raw_wildcard_dir,
        wildcard_output_dir=args.wildcard_output_dir,
        template_output_dir=args.template_output_dir,
        workflow_output_dir=args.workflow_output_dir,
        global_seed=args.global_seed,
        session_count=args.session_count,
    )
    print(f"Using runtime-first V6 wildcards from: {result['raw_wildcard_dir']}")
    print(f"Wrote V6 templates to: {result['template_output_dir']}")
    print(f"Wrote V6 workflows to: {result['workflow_output_dir']}")


if __name__ == "__main__":
    main()
