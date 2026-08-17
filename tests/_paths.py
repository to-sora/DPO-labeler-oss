from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "template"
TASKS_DIR = TEMPLATE_DIR / "tasks"
PROMPT_TEMPLATES_DIR = TEMPLATE_DIR / "prompt_templates"
WILDCARD_DIR = TEMPLATE_DIR / "wildcard"
WORKFLOWS_DIR = TEMPLATE_DIR / "workflows"
