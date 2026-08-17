from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

APP_VERSION = "0.3.0"
DECISIONS = ["a_good", "b_good", "both_good", "both_bad", "skip"]
DEFECT_TAGS = [
    "face_off",
    "eyes_off",
    "hand_corruption",
    "anatomy_other",
    "bad_color_lighting",
    "bad_composition",
    "bad_crop_framing",
    "background_artifacts",
    "low_detail_blur",
    "text_or_watermark_artifact",
]
EXPORT_FILENAMES = {
    "label-events": "label_events.jsonl",
    "labels-latest": "labels_latest.jsonl",
    "preference-pairs": "preference_pairs.jsonl",
    "dpo-pairs": "dpo_pairs.jsonl",
}


class LabelEventValidationError(ValueError):
    pass


class AuthenticationError(ValueError):
    pass


class CatalogError(ValueError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object JSONL line in {path} at line {line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def stable_hex(*parts: object) -> str:
    payload = "::".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_existing_path(value: str | None, candidates: Sequence[Path]) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    if raw.is_absolute() and raw.exists():
        return raw
    for base in candidates:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
    if raw.exists():
        return raw.resolve()
    return None


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
