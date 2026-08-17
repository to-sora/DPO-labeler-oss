from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_LISTS_DIR, WORKFLOWS_DIR, relative_path as relative_root_path
from prompt_generator import load_prompt_list_lines
from generators._shared_task import build_shared_pair_task, write_yaml
from generators.generate_dpo_workflow_matrix_v8 import NEGATIVE_PROMPTS_BY_FAMILY
from generators.sv_checkpoint_pools import describe_shared_checkpoint_pools


DEFAULT_PROMPT_LIST_ROOT = PROMPT_LISTS_DIR
DEFAULT_PROMPT_LIST_SOURCE_DIR = PROMPT_LISTS_DIR / "prompt_list_demo_v1"
DEFAULT_OUTPUT_DIR = WORKFLOWS_DIR / "dpo_workflow_prompt_list_demo_v1"
DEFAULT_GLOBAL_SEED = 20260408
PROMPT_FILE_NAMES = {
    "illustration": "illustration.txt",
    "anime": "anime.txt",
    "pony": "pony.txt",
}


def _load_demo_lines(source_dir: Path) -> dict[str, list[str]]:
    prompt_lines: dict[str, list[str]] = {}
    for family_key, file_name in PROMPT_FILE_NAMES.items():
        path = source_dir / file_name
        if not path.is_file():
            raise ValueError(f"Missing required prompt list file: {path}")
        lines = load_prompt_list_lines(path)
        if not lines:
            raise ValueError(f"Prompt list file is empty: {path}")
        prompt_lines[family_key] = lines
    return prompt_lines


def _validate_shared_line_count(prompt_lines: dict[str, list[str]], *, source_dir: Path) -> int:
    counts = {family_key: len(lines) for family_key, lines in prompt_lines.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            f"Prompt list files in {source_dir} must have the same usable line count, got {counts}"
        )
    return next(iter(unique_counts))


def _relative_prompt_list_identifier(prompt_list_root: Path, prompt_list_path: Path) -> str:
    try:
        relative_path = prompt_list_path.relative_to(prompt_list_root)
    except ValueError as exc:
        raise ValueError(
            f"Prompt list file {prompt_list_path} must live under prompt list root {prompt_list_root}"
        ) from exc
    return relative_path.with_suffix("").as_posix()


def make_prompt_args(*, prompt_list_root: str, prompt_identifiers: dict[str, str]) -> Dict[str, Any]:
    return {
        "prompt_list": prompt_identifiers["illustration"],
        "prompt_list_by_ckpt_family": {
            "illustration": prompt_identifiers["illustration"],
            "anime": prompt_identifiers["anime"],
            "pony": prompt_identifiers["pony"],
        },
        "prompt_list_root": prompt_list_root,
        "negative_prompt": NEGATIVE_PROMPTS_BY_FAMILY["illustration"],
        "negative_prompt_by_ckpt_family": {
            "illustration": NEGATIVE_PROMPTS_BY_FAMILY["illustration"],
            "sdxl_anime_base": NEGATIVE_PROMPTS_BY_FAMILY["sdxl_anime_base"],
            "pony": NEGATIVE_PROMPTS_BY_FAMILY["pony"],
            "realistic": NEGATIVE_PROMPTS_BY_FAMILY["realistic"],
        },
    }


def generate_prompt_list_demo_v1(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    prompt_list_root: str | Path = DEFAULT_PROMPT_LIST_ROOT,
    prompt_list_source_dir: str | Path = DEFAULT_PROMPT_LIST_SOURCE_DIR,
    global_seed: int = DEFAULT_GLOBAL_SEED,
) -> list[Dict[str, Any]]:
    output_path = Path(output_dir)
    prompt_list_root_path = Path(prompt_list_root)
    source_dir = Path(prompt_list_source_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not prompt_list_root_path.is_dir():
        raise ValueError(f"Prompt list root not found: {prompt_list_root_path}")
    if not source_dir.is_dir():
        raise ValueError(f"Prompt list source directory not found: {source_dir}")

    prompt_lines = _load_demo_lines(source_dir)
    prompt_line_count = _validate_shared_line_count(prompt_lines, source_dir=source_dir)
    prompt_identifiers = {
        family_key: _relative_prompt_list_identifier(prompt_list_root_path, source_dir / file_name)
        for family_key, file_name in PROMPT_FILE_NAMES.items()
    }
    prompt_args = make_prompt_args(
        prompt_list_root=relative_root_path(output_path, prompt_list_root_path),
        prompt_identifiers=prompt_identifiers,
    )

    rows: list[Dict[str, Any]] = []
    for pair_mode in ("same", "diff"):
        task_name = f"prompt_list_demo_v1_{pair_mode}"
        payload = build_shared_pair_task(
            task_name=task_name,
            pair_mode=pair_mode,
            prompt_generator_name="prompt_list_v1",
            prompt_args_image1=prompt_args,
            session_count=prompt_line_count,
            global_seed=global_seed,
        )
        file_name = f"{task_name}.yaml"
        write_yaml(output_path / file_name, payload)
        rows.append(
            {
                "task_name": task_name,
                "output_yaml": file_name,
                "pair_mode": pair_mode,
                "prompt_generator": "prompt_list_v1",
                "session_count": prompt_line_count,
            }
        )

    write_yaml(
        output_path / "manifest.yaml",
        {
            "workflow_count": len(rows),
            "global_seed": int(global_seed),
            "prompt_list_root": str(prompt_list_root_path),
            "prompt_list_source_dir": str(source_dir),
            "prompt_files": {
                family_key: str(source_dir / file_name)
                for family_key, file_name in PROMPT_FILE_NAMES.items()
            },
            "prompt_identifiers": dict(prompt_identifiers),
            "prompt_line_count": prompt_line_count,
            "realistic_prompt_fallback": prompt_identifiers["illustration"],
            "checkpoint_pools": describe_shared_checkpoint_pools(),
            "workflows": rows,
        },
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate same/diff demo workflows from 3 literal prompt-list files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-list-root", default=str(DEFAULT_PROMPT_LIST_ROOT))
    parser.add_argument("--prompt-list-source-dir", default=str(DEFAULT_PROMPT_LIST_SOURCE_DIR))
    parser.add_argument("--global-seed", type=int, default=DEFAULT_GLOBAL_SEED)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = generate_prompt_list_demo_v1(
        output_dir=args.output_dir,
        prompt_list_root=args.prompt_list_root,
        prompt_list_source_dir=args.prompt_list_source_dir,
        global_seed=args.global_seed,
    )
    print(f"Generated {len(rows)} prompt-list demo workflows in {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
