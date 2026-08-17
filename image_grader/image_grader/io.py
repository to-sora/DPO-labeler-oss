from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TextIO


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class ImageJob:
    image_path: Path
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return self.request_id or str(self.image_path)


def iter_image_dir(input_dir: str | Path, extensions: Iterable[str], recursive: bool = True) -> Iterator[ImageJob]:
    root = Path(input_dir)
    if not root.is_dir():
        raise InputError(f"input dir does not exist or is not a directory: {root}")
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in allowed:
            yield ImageJob(image_path=path.resolve())


def iter_image_json(input_path: str | Path) -> Iterator[ImageJob]:
    path = Path(input_path)
    if not path.is_file():
        raise InputError(f"input JSON/JSONL file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            yield from _iter_jsonl(handle, path)
        return

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise InputError(f"input file is empty: {path}")
    if text[0] in "[{":
        payload = json.loads(text)
        yield from _jobs_from_json_payload(payload, path)
        return

    with path.open("r", encoding="utf-8") as handle:
        yield from _iter_jsonl(handle, path)


def append_jsonl(output_path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _iter_jsonl(handle: TextIO, path: Path) -> Iterator[ImageJob]:
    for line_number, raw_line in enumerate(handle, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"invalid JSON on line {line_number} in {path}: {exc}") from exc
        yield _job_from_item(payload, f"{path}:{line_number}")


def _jobs_from_json_payload(payload: Any, path: Path) -> Iterator[ImageJob]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise InputError(f"{path} must contain a JSON object, an array, or an object with items")
    for index, item in enumerate(items, start=1):
        yield _job_from_item(item, f"{path}:items[{index}]")


def _job_from_item(item: Any, source: str) -> ImageJob:
    if isinstance(item, str):
        return ImageJob(image_path=Path(item).expanduser().resolve())
    if not isinstance(item, Mapping):
        raise InputError(f"{source} must be a string path or object")
    raw_path = item.get("image_path", item.get("path", item.get("file")))
    if raw_path is None:
        raise InputError(f"{source} is missing image_path")
    metadata = item.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise InputError(f"{source}.metadata must be an object")
    request_id = item.get("request_id")
    return ImageJob(
        image_path=Path(str(raw_path)).expanduser().resolve(),
        request_id=str(request_id) if request_id not in (None, "") else None,
        metadata=dict(metadata),
    )
