from pathlib import Path
from typing import Sequence

import yaml
from PIL import Image

from ..common import path_is_within_roots


def text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def import_directory(value: object, field: str, image_roots: Sequence[Path]) -> Path:
    root = Path(text(value, field))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError(f"{field} must be an existing absolute backend path")
    root = root.resolve()
    if not path_is_within_roots(root, image_roots):
        raise ValueError(f"{field} must stay inside a configured image root")
    return root


def parse_manifest(source: str, image_roots: Sequence[Path]) -> dict:
    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("YAML must contain a task mapping")
    task = text(data.get("task-name"), "task-name")
    character = text(data.get("character-name"), "character-name")
    root = import_directory(data.get("image_dir"), "image_dir", image_roots)
    prompt_root = import_directory(data.get("prompt_dir", str(root)), "prompt_dir", image_roots)
    dims = data.get("dim-name")
    if not isinstance(dims, list) or not dims:
        raise ValueError("dim-name must be a nonempty list")
    dims = [text(d, "dim-name entry") for d in dims]
    if len(set(dims)) != len(dims):
        raise ValueError("Duplicate dimension names")
    entries = data.get("image")
    if not isinstance(entries, list) or not entries:
        raise ValueError("image must be a nonempty list")
    images, seen = [], set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"image[{index}] must be a mapping")
        relative = Path(text(item.get("image-path"), "image-path"))
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("image-path must stay inside image_dir")
        if path in seen:
            raise ValueError(f"Duplicate image: {relative}")
        seen.add(path)
        try:
            with Image.open(path) as image:
                image.verify()
            prompt_path = item.get("prompt-path")
            prompt_file = (prompt_root / text(prompt_path, "prompt-path")).resolve() if prompt_path else None
            if prompt_file and not prompt_file.is_relative_to(prompt_root):
                raise ValueError("prompt-path must stay inside prompt_dir")
            prompt = item.get("prompt", "")
            if prompt_file and prompt_file.exists():
                prompt = prompt_file.read_text(encoding="utf-8")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read image/prompt {relative}: {exc}") from exc
        images.append({"image_id": index, "image_path": str(path),
                       "prompt_path": str(prompt_file) if prompt_file else None,
                       "prompt": prompt})
    return {"task_name": task, "character_name": character,
            "image_dir": str(root), "prompt_dir": str(prompt_root),
            "dimensions": dims, "images": images}
