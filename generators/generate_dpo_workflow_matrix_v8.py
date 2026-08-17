from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_TEMPLATES_DIR, WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from generators._shared_task import SHARED_SESSION_COUNT, TemplateEntry, emit_version_matrix, write_yaml


DEFAULT_RAW_WILDCARD_DIR = WILDCARD_DIR / "research_v8"
DEFAULT_WILDCARD_DIR = WILDCARD_DIR / "research_v8"
DEFAULT_TEMPLATE_DIR = PROMPT_TEMPLATES_DIR / "research_v8"
DEFAULT_WORKFLOW_DIR = WORKFLOWS_DIR / "dpo_workflow_v8"
DEFAULT_GLOBAL_SEED_V8 = 20260408
DEFAULT_SESSION_COUNT = SHARED_SESSION_COUNT
TEMPLATE_OMITTED_SOURCE_FILES = ("Character.txt", "characters/*", "semantic_class.txt")
CHARACTER_SOURCES_ENABLED = False

SET_VARIANTS = ("sfw", "nsfw")
RUNTIME_SUBDIR_BY_SET = {
    "sfw": "runtime_sfw",
    "nsfw": "runtime_nsfw",
}
FAMILIES = ("illustration", "sdxl_anime_base", "pony", "realistic")
TOP_LEVEL_TXT_ORDER = (
    "Action_Pose.txt",
    "Attire_Accessory.txt",
    "Body_Appearance.txt",
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
)
NEGATIVE_PROMPTS_BY_FAMILY = {
    "illustration": (
        "lowres, blurry, bad anatomy, bad hands, extra fingers, deformed face, "
        "text, watermark, logo, jpeg artifacts"
    ),
    "sdxl_anime_base": (
        "lowres, blurry, bad anatomy, bad hands, extra fingers, low score, average score, "
        "text, watermark, logo"
    ),
    "pony": "score_4, score_5, score_6, lowres, blurry, bad anatomy, bad hands, extra fingers, text, watermark, logo",
    "realistic": (
        "anime, illustration, cartoon, painting, cgi, 3d, lowres, blurry, bad anatomy, "
        "bad hands, extra fingers, text, watermark"
    ),
}
TEMPLATE_TOKEN_ORDER_BY_SET = {
    "sfw": (
        "Rating.txt",
        "Person_Count_Identity.txt",
        "Species_Fantasy_Trait.txt",
        "Body_Appearance.txt",
        "Attire_Accessory.txt",
        "Expression_Gaze.txt",
        "Action_Pose.txt",
        "Composition_Camera.txt",
        "Scene_Background.txt",
        "Object_Prop.txt",
        "Relationship_Interaction.txt",
        "Style_Medium.txt",
        "Other_General.txt",
    ),
    "nsfw": (
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
        "Style_Medium.txt",
        "Other_General.txt",
    ),
}


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    return f"__research_v8/{path.with_suffix('').as_posix()}__"


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


def _build_family_lines(prefixes: Iterable[str], *, tokens: list[str], suffixes: Iterable[str] | None = None) -> list[str]:
    resolved_prefixes = list(prefixes)
    resolved_suffixes = list(suffixes) if suffixes is not None else [""] * len(resolved_prefixes)
    if len(resolved_prefixes) != len(resolved_suffixes):
        raise ValueError("prefixes and suffixes must have the same length")

    lines: list[str] = []
    for prefix, suffix in zip(resolved_prefixes, resolved_suffixes):
        segments = [prefix, *tokens]
        if suffix:
            segments.append(suffix)
        lines.append(", ".join(segment for segment in segments if segment))
    return lines


def build_template_lines(set_name: str, family: str, source_files: list[str], *, raw_dir: Path) -> list[str]:
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
    body_tokens = _ordered_tokens(raw_dir, set_name, [*common_order, *extras])

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


def build_templates(template_output_dir: Path, *, raw_dir: Path, source_files: list[str], empty_files: list[str]) -> None:
    template_output_dir.mkdir(parents=True, exist_ok=True)
    for set_name in SET_VARIANTS:
        for family in FAMILIES:
            lines = build_template_lines(set_name, family, source_files, raw_dir=raw_dir)
            write_text(template_output_dir / f"{set_name}_hybrid_compact_{family}.txt", lines)

    readme_lines = [
        "# V8 Prompt Templates",
        "",
        "These templates target current model-family guidance more directly than V6/V7.",
        "SFW uses runtime-first files but omits NSFW_Nudity entirely.",
        "General V8 omits Character, characters/*, and semantic_class.",
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
        "template": f"research_v8/{set_name}_hybrid_compact_illustration",
        "template_by_ckpt_family": {
            "illustration": f"research_v8/{set_name}_hybrid_compact_illustration",
            "sdxl_anime_base": f"research_v8/{set_name}_hybrid_compact_sdxl_anime_base",
            "pony": f"research_v8/{set_name}_hybrid_compact_pony",
            "realistic": f"research_v8/{set_name}_hybrid_compact_realistic",
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
            prompt_args=make_prompt_args(set_name, template_root=template_root, wildcard_root=wildcard_root),
        )
        for set_name in SET_VARIANTS
    ]


def generate_v8_trials(
    *,
    raw_wildcard_dir: str | Path = DEFAULT_RAW_WILDCARD_DIR,
    wildcard_output_dir: str | Path = DEFAULT_WILDCARD_DIR,
    template_output_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    workflow_output_dir: str | Path = DEFAULT_WORKFLOW_DIR,
    global_seed: int = DEFAULT_GLOBAL_SEED_V8,
    session_count: int = DEFAULT_SESSION_COUNT,
) -> dict[str, Any]:
    raw_dir = Path(raw_wildcard_dir)
    wildcard_dir = Path(wildcard_output_dir)
    template_dir = Path(template_output_dir)
    workflow_dir = Path(workflow_output_dir)
    template_root = relative_root_path(workflow_dir, template_dir.parent)
    wildcard_root = relative_root_path(workflow_dir, wildcard_dir.parent)

    if not raw_dir.is_dir():
        raise ValueError(f"Missing required V8 source directory: {raw_dir}")

    active_source_files, empty_source_files = discover_source_files(raw_dir)
    if not active_source_files:
        raise ValueError(f"No non-empty wildcard txt files found in {raw_dir}")

    mirrored_files = mirror_raw_sources(raw_dir=raw_dir, wildcard_output_dir=wildcard_dir)
    build_templates(template_dir, raw_dir=raw_dir, source_files=active_source_files, empty_files=empty_source_files)

    entries = _build_entries(template_root, wildcard_root)
    workflow_rows = emit_version_matrix(
        version_tag="v8",
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
            "source_files": active_source_files,
            "empty_source_files": empty_source_files,
            "mirrored_files": mirrored_files,
            "template_omitted_source_files": list(TEMPLATE_OMITTED_SOURCE_FILES),
            "character_sources_enabled": CHARACTER_SOURCES_ENABLED,
            "wildcard_mode": "runtime_first_with_no_semantic_or_character_fallback",
            "runtime_dirs_by_set": dict(RUNTIME_SUBDIR_BY_SET),
            "top_level_template_fallbacks": [],
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
        description="Generate V8 model-aligned wildcard templates and workflow YAMLs."
    )
    parser.add_argument("--raw-wildcard-dir", default=str(DEFAULT_RAW_WILDCARD_DIR))
    parser.add_argument("--wildcard-output-dir", default=str(DEFAULT_WILDCARD_DIR))
    parser.add_argument("--template-output-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--workflow-output-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED_V8)
    parser.add_argument("--session-count", type=int, default=DEFAULT_SESSION_COUNT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = generate_v8_trials(
        raw_wildcard_dir=args.raw_wildcard_dir,
        wildcard_output_dir=args.wildcard_output_dir,
        template_output_dir=args.template_output_dir,
        workflow_output_dir=args.workflow_output_dir,
        global_seed=args.global_seed,
        session_count=args.session_count,
    )
    print(f"Using V8 wildcard sources from: {result['raw_wildcard_dir']}")
    print(f"Wrote V8 templates to: {result['template_output_dir']}")
    print(f"Wrote V8 workflows to: {result['workflow_output_dir']}")


if __name__ == "__main__":
    main()
