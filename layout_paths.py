from __future__ import annotations

import os
from pathlib import Path


TEMPLATE_DIR = Path("template")
TASKS_DIR = TEMPLATE_DIR / "tasks"
WORKFLOWS_DIR = TEMPLATE_DIR / "workflows"
PROMPT_TEMPLATES_DIR = TEMPLATE_DIR / "prompt_templates"
PROMPT_LISTS_DIR = TEMPLATE_DIR / "prompt_lists"
WILDCARD_DIR = TEMPLATE_DIR / "wildcard"


def relative_path(from_dir: str | Path, to_dir: str | Path) -> str:
    return Path(os.path.relpath(Path(to_dir), start=Path(from_dir))).as_posix()
