from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import mimetypes
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..backend.common import APP_VERSION, stable_hex, utc_timestamp, write_jsonl

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_IMPORT_FORMATS = ("preference-pairs", "dpo-pairs", "labels-latest")
DECISION_ORDER = ("a_good", "b_good", "both_good", "both_bad", "skip", "unknown")
DECISION_LABELS = {
    "a_good": "A Good",
    "b_good": "B Good",
    "both_good": "Both Good",
    "both_bad": "Both Bad",
    "skip": "Skip",
    "unknown": "Unknown",
}
DECISION_COLORS = {
    "a_good": "#2f855a",
    "b_good": "#2465a5",
    "both_good": "#b27321",
    "both_bad": "#b64949",
    "skip": "#5f6672",
    "unknown": "#8b7f71",
}
UNKNOWN_CHECKPOINT = "(Unknown checkpoint)"


class ImportValidationError(ValueError):
    pass


class ViewerNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class CachedImagePath:
    path: Path
    size: int
    mtime_ns: int


class ImageLocator:
    def __init__(self, image_roots: Sequence[str | Path]) -> None:
        if not image_roots:
            raise ValueError("At least one image root is required")
        self.image_roots = tuple(self._validate_root(root) for root in image_roots)
        self._lock = threading.Lock()
        self._cache: dict[str, CachedImagePath] = {}

    @staticmethod
    def _validate_root(root: str | Path) -> Path:
        resolved = Path(root).resolve()
        if not resolved.exists():
            raise ValueError(f"image_root does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"image_root must be a directory: {resolved}")
        return resolved

    def resolve(self, expected_sha256: str) -> Path:
        digest = self._normalize_sha(expected_sha256)
        with self._lock:
            cached = self._cache.get(digest)
        if cached and self._cached_entry_matches(cached, digest):
            return cached.path

        candidates: list[Path] = []
        for root in self.image_roots:
            for candidate in root.rglob(f"{digest}.*"):
                if candidate.is_file():
                    candidates.append(candidate.resolve())
        for candidate in sorted({path for path in candidates}, key=lambda item: str(item)):
            if self._verify_image_file(candidate, digest):
                cached = self._build_cache_entry(candidate)
                with self._lock:
                    self._cache[digest] = cached
                return candidate
        raise ImportValidationError(f"No valid server-side image found for sha256 {digest}")

    def validate_path(self, path: str | Path, expected_sha256: str) -> Path:
        candidate = Path(path).resolve()
        digest = self._normalize_sha(expected_sha256)
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(candidate)
        if not self._verify_image_file(candidate, digest):
            raise ImportValidationError(f"Server-side image no longer matches sha256 {digest}")
        cached = self._build_cache_entry(candidate)
        with self._lock:
            self._cache[digest] = cached
        return candidate

    @staticmethod
    def _normalize_sha(value: str) -> str:
        digest = str(value or "").strip().lower()
        if not SHA256_RE.fullmatch(digest):
            raise ImportValidationError(f"Expected a 64-character sha256 value, got {value!r}")
        return digest

    def _cached_entry_matches(self, cached: CachedImagePath, expected_sha256: str) -> bool:
        if not cached.path.exists() or not cached.path.is_file():
            return False
        stat = cached.path.stat()
        if stat.st_size == cached.size and stat.st_mtime_ns == cached.mtime_ns:
            return True
        if not self._verify_image_file(cached.path, expected_sha256):
            return False
        refreshed = self._build_cache_entry(cached.path)
        with self._lock:
            self._cache[expected_sha256] = refreshed
        return True

    @staticmethod
    def _build_cache_entry(path: Path) -> CachedImagePath:
        stat = path.stat()
        return CachedImagePath(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    @staticmethod
    def _verify_image_file(path: Path, expected_sha256: str) -> bool:
        if _sha256_file(path) != expected_sha256:
            return False
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            return False
        return True


class ExportViewerApp:
    def __init__(
        self,
        state_dir: str | Path,
        image_roots: Sequence[str | Path],
        default_page_size: int = 10,
        max_page_size: int = 50,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.imports_dir = self.state_dir / "imports"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        self.image_locator = ImageLocator(image_roots)
        self.default_page_size = max(int(default_page_size), 1)
        self.max_page_size = max(int(max_page_size), self.default_page_size)
        self._lock = threading.Lock()
        self._imports: dict[str, dict[str, Any]] = {}
        self._load_persisted_imports()

    def get_config(self) -> dict[str, Any]:
        return {
            "app_version": APP_VERSION,
            "supported_import_formats": list(SUPPORTED_IMPORT_FORMATS),
            "image_roots": [str(path) for path in self.image_locator.image_roots],
            "import_count": len(self._imports),
            "default_page_size": self.default_page_size,
            "max_page_size": self.max_page_size,
        }

    def list_imports(self) -> dict[str, Any]:
        with self._lock:
            imports = [self._public_import_summary(record) for record in self._sorted_imports()]
        return {"imports": imports}

    def get_import(self, import_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require_import(import_id)
            return {"import": self._public_import_detail(record)}

    def get_import_analytics(self, import_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require_import(import_id)
            rows_path = Path(record["rows_path"])
            import_summary = self._public_import_detail(record)
            import_format = str(record["format"])
        rows = [_hydrate_legacy_row(row) for row in _load_jsonl_file(rows_path)]
        return {
            "import": import_summary,
            "summary": {
                "format": import_format,
                "row_count": len(rows),
                "note": "Internal dataset_id values are merged for analytics. Table 1 uses imported row timestamps only.",
            },
            "tables": _build_import_analytics(import_format, rows),
        }

    def get_import_rows(self, import_id: str, cursor: int, limit: int) -> dict[str, Any]:
        try:
            start = max(int(cursor), 0)
        except (TypeError, ValueError):
            start = 0
        page_size = self._coerce_page_size(limit)
        with self._lock:
            record = self._require_import(import_id)
            rows_path = Path(record["rows_path"])
            total = int(record["valid_rows"])
        end = min(start + page_size, total)
        page = [self._public_row(import_id, row) for row in _read_row_page(rows_path, start, page_size)]
        return {
            "import": self._public_import_summary(record),
            "items": page,
            "cursor": start,
            "next_cursor": end if end < total else None,
            "total": total,
            "limit": page_size,
        }

    def create_import(self, filename: str, text: str) -> dict[str, Any]:
        cleaned_filename = Path((filename or "upload.jsonl").strip() or "upload.jsonl").name or "upload.jsonl"
        rows, format_name, warnings = self._normalize_upload_text(text)
        if not rows:
            raise ImportValidationError("Upload contains no valid rows")

        created_at = utc_timestamp()
        import_id = self._allocate_import_id(cleaned_filename, text, created_at)
        import_dir = self.imports_dir / import_id
        import_dir.mkdir(parents=True, exist_ok=False)

        upload_path = import_dir / cleaned_filename
        rows_path = import_dir / "rows.jsonl"
        manifest_path = import_dir / "manifest.json"
        upload_path.write_text(text, encoding="utf-8")
        write_jsonl(rows_path, rows)
        manifest = {
            "import_id": import_id,
            "filename": cleaned_filename,
            "format": format_name,
            "created_at": created_at,
            "total_rows": len(rows) + len(warnings),
            "valid_rows": len(rows),
            "invalid_rows": len(warnings),
            "warnings": warnings,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        record = {
            **manifest,
            "dir": str(import_dir),
            "upload_path": str(upload_path),
            "rows_path": str(rows_path),
        }
        with self._lock:
            self._imports[import_id] = record
        return {"import": self._public_import_detail(record)}

    def delete_import(self, import_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require_import(import_id)
            self._imports.pop(import_id, None)
        shutil.rmtree(record["dir"], ignore_errors=True)
        return {"deleted": True, "import_id": import_id}

    def get_media_path(self, import_id: str, row_id: str, slot: str) -> tuple[Path, str]:
        with self._lock:
            record = self._require_import(import_id)
            rows_path = Path(record["rows_path"])
        row = _find_row_by_id(rows_path, row_id)
        image = self._require_image(row, slot)
        path = image["resolved_path"]
        expected_sha256 = image["expected_sha256"]
        validated = self.image_locator.validate_path(path, expected_sha256)
        mime = mimetypes.guess_type(str(validated))[0] or "application/octet-stream"
        return validated, mime

    def _load_persisted_imports(self) -> None:
        loaded: dict[str, dict[str, Any]] = {}
        for import_dir in sorted(self.imports_dir.iterdir()) if self.imports_dir.exists() else []:
            if not import_dir.is_dir():
                continue
            manifest_path = import_dir / "manifest.json"
            rows_path = import_dir / "rows.jsonl"
            upload_candidates = [path for path in import_dir.iterdir() if path.is_file() and path.name not in {"manifest.json", "rows.jsonl"}]
            if not manifest_path.exists() or not rows_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                _validate_rows_file(rows_path)
            except Exception as exc:
                print(f"Warning: failed to load persisted import from {import_dir}: {exc}")
                continue
            if not isinstance(manifest, dict):
                continue
            record = {
                **manifest,
                "dir": str(import_dir),
                "upload_path": str(upload_candidates[0]) if upload_candidates else "",
                "rows_path": str(rows_path),
            }
            import_id = str(manifest.get("import_id", import_dir.name))
            loaded[import_id] = record
        with self._lock:
            self._imports = loaded

    def _coerce_page_size(self, limit: int) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError):
            value = self.default_page_size
        if value < 1:
            value = self.default_page_size
        return min(value, self.max_page_size)

    def _allocate_import_id(self, filename: str, text: str, created_at: str) -> str:
        base = stable_hex(filename, text, created_at)[:16]
        candidate = base
        suffix = 1
        while (self.imports_dir / candidate).exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _normalize_upload_text(self, text: str) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
        detected_kind: str | None = None
        rows: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImportValidationError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ImportValidationError(f"Line {line_number} must be a JSON object")
            kind = self._detect_row_kind(payload)
            if kind is None:
                warnings.append({"line_number": line_number, "message": "Unsupported export row shape"})
                continue
            if detected_kind and kind != detected_kind:
                raise ImportValidationError("Mixed export formats are not supported in one upload")
            detected_kind = kind
            try:
                rows.append(self._normalize_row(kind, payload, len(rows) + 1, line_number))
            except ImportValidationError as exc:
                warnings.append({"line_number": line_number, "message": str(exc)})

        if detected_kind is None:
            raise ImportValidationError("No supported JSONL rows were found in the upload")
        if not rows:
            raise ImportValidationError("Upload contains no valid rows")
        return rows, self._format_name(detected_kind, rows), warnings

    @staticmethod
    def _detect_row_kind(row: Mapping[str, Any]) -> str | None:
        if isinstance(row.get("chosen_image"), Mapping) and isinstance(row.get("rejected_image"), Mapping):
            return "pair_export"
        images = row.get("images")
        label = row.get("label")
        if isinstance(label, Mapping) and isinstance(images, list):
            return "labels_latest"
        return None

    @staticmethod
    def _format_name(kind: str, rows: Sequence[Mapping[str, Any]]) -> str:
        if kind == "labels_latest":
            return "labels-latest"
        strict_values = {bool(row.get("strict_dpo")) for row in rows}
        if strict_values == {True}:
            return "dpo-pairs"
        return "preference-pairs"

    def _normalize_row(
        self,
        kind: str,
        row: Mapping[str, Any],
        row_number: int,
        line_number: int,
    ) -> dict[str, Any]:
        if kind == "pair_export":
            return self._normalize_pair_export_row(row, row_number, line_number)
        if kind == "labels_latest":
            return self._normalize_labels_latest_row(row, row_number, line_number)
        raise ImportValidationError(f"Unsupported row kind {kind!r}")

    def _normalize_pair_export_row(self, row: Mapping[str, Any], row_number: int, line_number: int) -> dict[str, Any]:
        chosen = row.get("chosen_image")
        rejected = row.get("rejected_image")
        if not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            raise ImportValidationError(f"Line {line_number} must contain chosen_image and rejected_image objects")
        label = row.get("label")
        label_data = label if isinstance(label, Mapping) else {}
        chosen_index = _safe_int(chosen.get("image_index"))
        rejected_index = _safe_int(rejected.get("image_index"))
        defect_indices = _extract_defect_indices(label_data, fallback_order=[chosen_index, rejected_index])
        return {
            "row_id": f"row-{row_number:06d}",
            "line_number": line_number,
            "kind": "pair_export",
            "dataset_id": str(row.get("dataset_id", "")).strip(),
            "session_id": str(row.get("session_id", "")).strip(),
            "task_name": str(row.get("task_name", "")).strip(),
            "task_yaml_name": str(row.get("task_yaml_name", "")).strip(),
            "workflow_name": str(row.get("workflow_name", "")).strip(),
            "primary_ckpt": str(row.get("primary_ckpt", "")).strip(),
            "decision": str(row.get("decision", label_data.get("decision", ""))).strip(),
            "reviewer_username": str(row.get("reviewer_username", label_data.get("reviewer_username", ""))).strip(),
            "created_at": str(label_data.get("created_at", "")).strip(),
            "note": str(label_data.get("note", "")).strip(),
            "strict_dpo": bool(row.get("strict_dpo")),
            "images": [
                self._normalize_image(
                    chosen,
                    slot="chosen",
                    slot_label="Chosen",
                    is_good=True,
                    is_bad=False,
                    has_defect=chosen_index in defect_indices,
                ),
                self._normalize_image(
                    rejected,
                    slot="rejected",
                    slot_label="Rejected",
                    is_good=False,
                    is_bad=True,
                    has_defect=rejected_index in defect_indices,
                ),
            ],
        }

    def _normalize_labels_latest_row(self, row: Mapping[str, Any], row_number: int, line_number: int) -> dict[str, Any]:
        label = row.get("label")
        images = row.get("images")
        if not isinstance(label, Mapping):
            raise ImportValidationError(f"Line {line_number} must contain a label object")
        if not isinstance(images, list) or len(images) != 2:
            raise ImportValidationError(f"Line {line_number} must contain exactly 2 images")
        sorted_images = sorted(images, key=lambda item: (_safe_int(item.get("image_index")), str(item.get("image_name", ""))))
        image_indices = [_safe_int(image.get("image_index")) for image in sorted_images if isinstance(image, Mapping)]
        good_indices, bad_indices = _resolve_labels_latest_states(str(label.get("decision", "")).strip(), image_indices, label)
        defect_indices = _extract_defect_indices(label, fallback_order=image_indices)
        normalized_images = []
        for index, image in enumerate(sorted_images):
            if not isinstance(image, Mapping):
                raise ImportValidationError(f"Line {line_number} contains a non-object image entry")
            image_index = _safe_int(image.get("image_index"))
            normalized_images.append(
                self._normalize_image(
                    image,
                    slot="a" if index == 0 else "b",
                    slot_label="A" if index == 0 else "B",
                    is_good=image_index in good_indices,
                    is_bad=image_index in bad_indices,
                    has_defect=image_index in defect_indices,
                )
            )
        return {
            "row_id": f"row-{row_number:06d}",
            "line_number": line_number,
            "kind": "labels_latest",
            "dataset_id": str(row.get("dataset_id", "")).strip(),
            "session_id": str(row.get("session_id", "")).strip(),
            "task_name": str(row.get("task_name", "")).strip(),
            "task_yaml_name": str(row.get("task_yaml_name", "")).strip(),
            "workflow_name": str(row.get("workflow_name", "")).strip(),
            "primary_ckpt": str(row.get("primary_ckpt", "")).strip(),
            "decision": str(label.get("decision", "")).strip(),
            "reviewer_username": str(label.get("reviewer_username", "")).strip(),
            "created_at": str(label.get("created_at", "")).strip(),
            "note": str(label.get("note", "")).strip(),
            "strict_dpo": None,
            "images": normalized_images,
        }

    def _normalize_image(
        self,
        image: Mapping[str, Any],
        *,
        slot: str,
        slot_label: str,
        is_good: bool,
        is_bad: bool,
        has_defect: bool,
    ) -> dict[str, Any]:
        expected_sha256 = _extract_expected_sha256(image)
        resolved_path = self.image_locator.resolve(expected_sha256)
        return {
            "slot": slot,
            "slot_label": slot_label,
            "image_index": image.get("image_index"),
            "image_name": str(image.get("image_name", "")).strip(),
            "expected_sha256": expected_sha256,
            "positive_prompt": str(image.get("positive_prompt", "")).strip(),
            "negative_prompt": str(image.get("negative_prompt", "")).strip(),
            "ckpt": str(image.get("ckpt", "")).strip(),
            "resolved_path": str(resolved_path),
            "is_good": bool(is_good),
            "is_bad": bool(is_bad),
            "has_defect": bool(has_defect),
        }

    def _public_import_summary(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "import_id": record["import_id"],
            "filename": record["filename"],
            "format": record["format"],
            "created_at": record["created_at"],
            "total_rows": record["total_rows"],
            "valid_rows": record["valid_rows"],
            "invalid_rows": record["invalid_rows"],
        }

    def _public_import_detail(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **self._public_import_summary(record),
            "warnings": list(record.get("warnings", [])),
        }

    def _public_row(self, import_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        hydrated = _hydrate_legacy_row(row)
        return {
            "row_id": hydrated["row_id"],
            "line_number": hydrated["line_number"],
            "kind": hydrated["kind"],
            "dataset_id": hydrated["dataset_id"],
            "session_id": hydrated["session_id"],
            "task_name": hydrated["task_name"],
            "task_yaml_name": hydrated["task_yaml_name"],
            "workflow_name": hydrated["workflow_name"],
            "primary_ckpt": hydrated["primary_ckpt"],
            "decision": hydrated["decision"],
            "reviewer_username": hydrated["reviewer_username"],
            "created_at": hydrated["created_at"],
            "note": hydrated["note"],
            "strict_dpo": hydrated["strict_dpo"],
            "images": [self._public_image(import_id, hydrated["row_id"], image) for image in hydrated["images"]],
        }

    @staticmethod
    def _public_image(import_id: str, row_id: str, image: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "slot": image["slot"],
            "slot_label": image["slot_label"],
            "image_index": image["image_index"],
            "image_name": image["image_name"],
            "expected_sha256": image["expected_sha256"],
            "positive_prompt": image["positive_prompt"],
            "negative_prompt": image["negative_prompt"],
            "ckpt": image["ckpt"],
            "is_good": image["is_good"],
            "is_bad": image["is_bad"],
            "has_defect": image["has_defect"],
            "media_url": f"/media/imports/{import_id}/{row_id}/{image['slot']}",
        }

    @staticmethod
    def _require_row(record: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
        for row in record["rows"]:
            if row["row_id"] == row_id:
                return row
        raise ViewerNotFoundError(f"Unknown row_id: {row_id}")

    @staticmethod
    def _require_image(row: Mapping[str, Any], slot: str) -> Mapping[str, Any]:
        for image in row["images"]:
            if image["slot"] == slot:
                return image
        raise ViewerNotFoundError(f"Unknown image slot: {slot}")

    def _require_import(self, import_id: str) -> dict[str, Any]:
        record = self._imports.get(import_id)
        if record is None:
            raise ViewerNotFoundError(f"Unknown import_id: {import_id}")
        return record

    def _sorted_imports(self) -> list[dict[str, Any]]:
        return sorted(self._imports.values(), key=lambda record: (record["created_at"], record["import_id"]), reverse=True)


def _build_import_analytics(import_format: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reviewer_timeline_counts: dict[tuple[str, str], int] = defaultdict(int)
    timeline_counts: dict[tuple[str, str], int] = defaultdict(int)
    cross_appearances: dict[str, int] = defaultdict(int)
    same_appearances: dict[str, int] = defaultdict(int)
    all_appearances: dict[str, int] = defaultdict(int)
    win_counts: dict[str, int] = defaultdict(int)
    defect_counts: dict[str, int] = defaultdict(int)
    index1_counts: dict[str, int] = defaultdict(int)
    index2_counts: dict[str, int] = defaultdict(int)
    clean_advantage: dict[tuple[str, str], int] = defaultdict(int)
    matchup_totals: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        decision = _decision_key(row.get("decision"))
        date_bucket = _timestamp_bucket(row.get("created_at"))
        reviewer = _reviewer_key(row.get("reviewer_username"))
        reviewer_timeline_counts[(date_bucket, reviewer)] += 1
        timeline_counts[(date_bucket, decision)] += 1

        images = [image for image in row.get("images", []) if isinstance(image, Mapping)]
        if len(images) != 2:
            continue

        checkpoints = [_checkpoint_name(image) for image in images]
        for checkpoint, image in zip(checkpoints, images):
            all_appearances[checkpoint] += 1
            if bool(image.get("has_defect")):
                defect_counts[checkpoint] += 1

        if checkpoints[0] == checkpoints[1]:
            same_appearances[checkpoints[0]] += 1
            if decision == "both_good":
                index1_counts[checkpoints[0]] += 1
            elif decision == "both_bad":
                index2_counts[checkpoints[0]] += 1
            continue

        for checkpoint in checkpoints:
            cross_appearances[checkpoint] += 1

        winner_and_loser = _winner_and_loser_images(images)
        if winner_and_loser is None:
            continue
        winner, loser = winner_and_loser
        winner_checkpoint = _checkpoint_name(winner)
        loser_checkpoint = _checkpoint_name(loser)
        if winner_checkpoint == loser_checkpoint:
            continue
        win_counts[winner_checkpoint] += 1
        matchup_totals[_unordered_pair(winner_checkpoint, loser_checkpoint)] += 1
        if not bool(loser.get("has_defect")):
            clean_advantage[(winner_checkpoint, loser_checkpoint)] += 1

    tables = {
        "table0": _build_timeline_table(
            table_id="table0",
            title="Reviewer Session Count Over Time",
            description="Daily labeled-session count grouped by reviewer, regardless of decision type.",
            timeline_counts=reviewer_timeline_counts,
            x_label="Date",
            y_label="Session Count",
            series_label_fn=lambda reviewer: reviewer,
            series_color_fn=_reviewer_color,
            value_field_name="session_count",
            empty_message="No reviewer timestamps were found in this import.",
        ),
        "table1": _build_timeline_table(
            table_id="table1",
            title="Label Pattern Over Time",
            description="Daily label count grouped by the imported row timestamp and colored by decision.",
            timeline_counts=timeline_counts,
            x_label="Date",
            y_label="Label Count",
            series_label_fn=_decision_label,
            series_color_fn=lambda decision: DECISION_COLORS.get(decision, DECISION_COLORS["unknown"]),
            value_field_name="count",
            empty_message="No timestamped labels were found in this import.",
        ),
        "table2": _build_count_bar_table(
            table_id="table2",
            title="Cross-Checkpoint Wins",
            description="Absolute win count when a checkpoint wins a cross-checkpoint comparison.",
            counts=win_counts,
            exposures=cross_appearances,
            y_label="Absolute Win Count",
            empty_message="No cross-checkpoint checkpoint appearances were found.",
        ),
        "table3": _build_percentage_bar_table(
            table_id="table3",
            title="Cross-Checkpoint Win Rate",
            description="Win count divided by cross-checkpoint appearance count for each checkpoint.",
            numerators=win_counts,
            denominators=cross_appearances,
            y_label="Win Rate",
            empty_message="No cross-checkpoint checkpoint appearances were found.",
        ),
        "table4": _build_count_bar_table(
            table_id="table4",
            title="Checkpoint Defect Count",
            description="Absolute defect count when a checkpoint image carries any defect tag.",
            counts=defect_counts,
            exposures=all_appearances,
            y_label="Absolute Defect Count",
            empty_message="No checkpoint image appearances were found.",
        ),
        "table5": _build_percentage_bar_table(
            table_id="table5",
            title="Checkpoint Defect Rate",
            description="Defect count divided by all image appearances for each checkpoint.",
            numerators=defect_counts,
            denominators=all_appearances,
            y_label="Defect Rate",
            empty_message="No checkpoint image appearances were found.",
        ),
        "table10": _build_heatmap_table(
            table_id="table10",
            title="Winner Over Clean Loser",
            description="Directed raw counts where checkpoint A wins and checkpoint B loses without any defect tags.",
            values=clean_advantage,
            matchup_totals=matchup_totals,
            cross_appearances=cross_appearances,
            normalized=False,
            empty_message="No cross-checkpoint single-winner matchups were found.",
        ),
        "table11": _build_heatmap_table(
            table_id="table11",
            title="Winner Over Clean Loser Rate",
            description="Table 10 normalized by total cross-checkpoint single-winner matchups for each checkpoint pair.",
            values=clean_advantage,
            matchup_totals=matchup_totals,
            cross_appearances=cross_appearances,
            normalized=True,
            empty_message="No cross-checkpoint single-winner matchups were found.",
        ),
    }

    if import_format == "labels-latest":
        tables["table6"] = _build_count_bar_table(
            table_id="table6",
            title="Index1 Absolute Count",
            description="Absolute count of same-checkpoint pairs labeled both_good.",
            counts=index1_counts,
            exposures=same_appearances,
            y_label="Absolute Index1 Count",
            empty_message="No same-checkpoint labels were found.",
        )
        tables["table7"] = _build_percentage_bar_table(
            table_id="table7",
            title="Index1 Rate",
            description="Index1 count divided by same-checkpoint appearance count for each checkpoint.",
            numerators=index1_counts,
            denominators=same_appearances,
            y_label="Index1 Rate",
            empty_message="No same-checkpoint labels were found.",
        )
        tables["table8"] = _build_count_bar_table(
            table_id="table8",
            title="Index2 Absolute Count",
            description="Absolute count of same-checkpoint pairs labeled both_bad.",
            counts=index2_counts,
            exposures=same_appearances,
            y_label="Absolute Index2 Count",
            empty_message="No same-checkpoint labels were found.",
        )
        tables["table9"] = _build_percentage_bar_table(
            table_id="table9",
            title="Index2 Rate",
            description="Index2 count divided by same-checkpoint appearance count for each checkpoint.",
            numerators=index2_counts,
            denominators=same_appearances,
            y_label="Index2 Rate",
            empty_message="No same-checkpoint labels were found.",
        )
    else:
        unsupported_message = "This import format only contains single-winner pair exports, so same-checkpoint both_good/both_bad metrics are unavailable."
        tables["table6"] = _unsupported_table(
            table_id="table6",
            title="Index1 Absolute Count",
            description="Absolute count of same-checkpoint pairs labeled both_good.",
            kind="bar",
            y_label="Absolute Index1 Count",
            message=unsupported_message,
        )
        tables["table7"] = _unsupported_table(
            table_id="table7",
            title="Index1 Rate",
            description="Index1 count divided by same-checkpoint appearance count for each checkpoint.",
            kind="bar",
            y_label="Index1 Rate",
            message=unsupported_message,
        )
        tables["table8"] = _unsupported_table(
            table_id="table8",
            title="Index2 Absolute Count",
            description="Absolute count of same-checkpoint pairs labeled both_bad.",
            kind="bar",
            y_label="Absolute Index2 Count",
            message=unsupported_message,
        )
        tables["table9"] = _unsupported_table(
            table_id="table9",
            title="Index2 Rate",
            description="Index2 count divided by same-checkpoint appearance count for each checkpoint.",
            kind="bar",
            y_label="Index2 Rate",
            message=unsupported_message,
        )

    return tables


def _build_timeline_table(
    *,
    table_id: str,
    title: str,
    description: str,
    timeline_counts: Mapping[tuple[str, str], int],
    x_label: str,
    y_label: str,
    series_label_fn: Any,
    series_color_fn: Any,
    value_field_name: str,
    empty_message: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    points_by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals_by_series: dict[str, int] = defaultdict(int)

    for (date_bucket, decision), count in sorted(
        timeline_counts.items(),
        key=lambda item: (_date_sort_key(item[0][0]), str(series_label_fn(item[0][1])).lower(), str(item[0][1]).lower()),
    ):
        label = str(series_label_fn(decision))
        rows.append(
            {
                "date": date_bucket,
                "series_key": decision,
                "series_label": label,
                value_field_name: int(count),
                "count": int(count),
            }
        )
        points_by_decision[decision].append({"x": date_bucket, "y": int(count)})
        totals_by_series[decision] += int(count)

    series = [
        {
            "key": decision,
            "label": str(series_label_fn(decision)),
            "color": str(series_color_fn(decision)),
            "points": points_by_decision[decision],
        }
        for decision in sorted(points_by_decision, key=lambda key: (-totals_by_series[key], str(series_label_fn(key)).lower()))
    ]

    return {
        "id": table_id,
        "title": title,
        "description": description,
        "kind": "scatter",
        "available": True,
        "x_label": x_label,
        "y_label": y_label,
        "value_format": "integer",
        "series": series,
        "rows": rows,
        "empty_message": empty_message,
    }


def _build_count_bar_table(
    *,
    table_id: str,
    title: str,
    description: str,
    counts: Mapping[str, int],
    exposures: Mapping[str, int],
    y_label: str,
    empty_message: str,
) -> dict[str, Any]:
    checkpoints = sorted(exposures, key=lambda checkpoint: (-int(counts.get(checkpoint, 0)), -int(exposures.get(checkpoint, 0)), checkpoint.lower()))
    rows = [
        {
            "key": checkpoint,
            "label": checkpoint,
            "value": int(counts.get(checkpoint, 0)),
            "count": int(counts.get(checkpoint, 0)),
            "denominator": int(exposures.get(checkpoint, 0)),
        }
        for checkpoint in checkpoints
    ]
    return {
        "id": table_id,
        "title": title,
        "description": description,
        "kind": "bar",
        "available": True,
        "x_label": "Checkpoint",
        "y_label": y_label,
        "value_format": "integer",
        "rows": rows,
        "empty_message": empty_message,
    }


def _build_percentage_bar_table(
    *,
    table_id: str,
    title: str,
    description: str,
    numerators: Mapping[str, int],
    denominators: Mapping[str, int],
    y_label: str,
    empty_message: str,
) -> dict[str, Any]:
    rows = []
    for checkpoint, denominator in denominators.items():
        if int(denominator) <= 0:
            continue
        numerator = int(numerators.get(checkpoint, 0))
        value = numerator / int(denominator)
        rows.append(
            {
                "key": checkpoint,
                "label": checkpoint,
                "value": value,
                "count": numerator,
                "denominator": int(denominator),
            }
        )
    rows.sort(key=lambda row: (-float(row["value"]), -int(row["count"]), -int(row["denominator"]), str(row["label"]).lower()))
    return {
        "id": table_id,
        "title": title,
        "description": description,
        "kind": "bar",
        "available": True,
        "x_label": "Checkpoint",
        "y_label": y_label,
        "value_format": "percentage",
        "rows": rows,
        "empty_message": empty_message,
    }


def _build_heatmap_table(
    *,
    table_id: str,
    title: str,
    description: str,
    values: Mapping[tuple[str, str], int],
    matchup_totals: Mapping[tuple[str, str], int],
    cross_appearances: Mapping[str, int],
    normalized: bool,
    empty_message: str,
) -> dict[str, Any]:
    checkpoints = {
        checkpoint
        for pair in matchup_totals
        for checkpoint in pair
    } | {
        checkpoint
        for pair in values
        for checkpoint in pair
    }
    ordered_checkpoints = sorted(checkpoints, key=lambda checkpoint: (-int(cross_appearances.get(checkpoint, 0)), checkpoint.lower()))
    if not ordered_checkpoints:
        return {
            "id": table_id,
            "title": title,
            "description": description,
            "kind": "heatmap",
            "available": True,
            "x_label": "Opponent checkpoint",
            "y_label": "Winner checkpoint",
            "value_format": "percentage" if normalized else "integer",
            "columns": [],
            "matrix_rows": [],
            "rows": [],
            "empty_message": empty_message,
        }

    matrix_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for winner in ordered_checkpoints:
        values_row = []
        row_total = 0.0
        for loser in ordered_checkpoints:
            if winner == loser:
                values_row.append({"column": loser, "value": None, "display": "—"})
                continue
            numerator = int(values.get((winner, loser), 0))
            denominator = int(matchup_totals.get(_unordered_pair(winner, loser), 0))
            if normalized:
                value = (numerator / denominator) if denominator else None
                display = "—" if value is None else f"{value * 100:.1f}%"
                if value is not None:
                    row_total += value
                if denominator:
                    cell_rows.append(
                        {
                            "row_label": winner,
                            "column_label": loser,
                            "value": value,
                            "count": numerator,
                            "denominator": denominator,
                        }
                    )
            else:
                value = numerator
                display = str(numerator)
                row_total += float(numerator)
                if numerator:
                    cell_rows.append(
                        {
                            "row_label": winner,
                            "column_label": loser,
                            "value": numerator,
                            "count": numerator,
                            "denominator": denominator,
                        }
                    )
            values_row.append({"column": loser, "value": value, "display": display, "count": numerator, "denominator": denominator})
        matrix_rows.append({"key": winner, "label": winner, "values": values_row, "total": row_total})

    cell_rows.sort(
        key=lambda row: (
            -float(row["value"]) if row["value"] is not None else 1.0,
            -int(row["count"]),
            -int(row["denominator"]),
            str(row["row_label"]).lower(),
            str(row["column_label"]).lower(),
        )
    )
    return {
        "id": table_id,
        "title": title,
        "description": description,
        "kind": "heatmap",
        "available": True,
        "x_label": "Losing checkpoint",
        "y_label": "Winning checkpoint",
        "value_format": "percentage" if normalized else "integer",
        "columns": [{"key": checkpoint, "label": checkpoint} for checkpoint in ordered_checkpoints],
        "matrix_rows": matrix_rows,
        "rows": cell_rows,
        "empty_message": empty_message,
    }


def _unsupported_table(
    *,
    table_id: str,
    title: str,
    description: str,
    kind: str,
    y_label: str,
    message: str,
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "description": description,
        "kind": kind,
        "available": False,
        "x_label": "Checkpoint",
        "y_label": y_label,
        "value_format": "integer",
        "rows": [],
        "empty_message": message,
    }


def _winner_and_loser_images(images: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    winners = [image for image in images if bool(image.get("is_good"))]
    losers = [image for image in images if bool(image.get("is_bad"))]
    if len(winners) != 1 or len(losers) != 1:
        return None
    return winners[0], losers[0]


def _checkpoint_name(image: Mapping[str, Any]) -> str:
    checkpoint = str(image.get("ckpt", "")).strip()
    return checkpoint or UNKNOWN_CHECKPOINT


def _decision_key(value: Any) -> str:
    decision = str(value or "").strip()
    return decision if decision in DECISION_LABELS else "unknown"


def _decision_label(decision: str) -> str:
    return DECISION_LABELS.get(decision, DECISION_LABELS["unknown"])


def _reviewer_key(value: Any) -> str:
    reviewer = str(value or "").strip()
    return reviewer or "Unknown reviewer"


def _reviewer_color(reviewer: str) -> str:
    digest = stable_hex(reviewer)
    hue = int(digest[:6], 16) % 360
    return f"hsl({hue} 58% 42%)"


def _decision_sort_key(decision: str) -> int:
    try:
        return DECISION_ORDER.index(decision)
    except ValueError:
        return len(DECISION_ORDER)


def _timestamp_bucket(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        candidate = text[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            return candidate
        return "Unknown"


def _date_sort_key(value: str) -> tuple[int, str]:
    return (1, value) if value == "Unknown" else (0, value)


def _unordered_pair(left: str, right: str) -> tuple[str, str]:
    first, second = sorted((left, right))
    return first, second


def _extract_expected_sha256(image: Mapping[str, Any]) -> str:
    explicit = str(image.get("image_sha256", "")).strip().lower()
    if explicit:
        if not SHA256_RE.fullmatch(explicit):
            raise ImportValidationError(f"image_sha256 must be a 64-character sha256 value, got {explicit!r}")
        return explicit
    saved_path = str(image.get("saved_path", "")).strip()
    candidate = Path(saved_path).stem.lower()
    if not SHA256_RE.fullmatch(candidate):
        raise ImportValidationError(
            f"saved_path basename must use a 64-character sha256 filename stem, got {Path(saved_path).name!r}"
        )
    return candidate


def _hydrate_legacy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    images = row.get("images")
    if not isinstance(images, list):
        return dict(row)
    if all(
        isinstance(image, Mapping)
        and {"is_good", "is_bad", "has_defect"}.issubset(image.keys())
        for image in images
    ):
        return dict(row)

    hydrated = dict(row)
    kind = str(row.get("kind", "")).strip()
    decision = str(row.get("decision", "")).strip()
    hydrated_images: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, Mapping):
            hydrated_images.append(dict())
            continue
        normalized = dict(image)
        if "is_good" not in normalized or "is_bad" not in normalized:
            derived_good, derived_bad = _derive_legacy_image_state(kind, decision, normalized)
            normalized.setdefault("is_good", derived_good)
            normalized.setdefault("is_bad", derived_bad)
        normalized.setdefault("has_defect", False)
        hydrated_images.append(normalized)
    hydrated["images"] = hydrated_images
    return hydrated


def _derive_legacy_image_state(kind: str, decision: str, image: Mapping[str, Any]) -> tuple[bool, bool]:
    slot = str(image.get("slot", "")).strip().lower()
    if kind == "pair_export":
        if slot == "chosen":
            return True, False
        if slot == "rejected":
            return False, True
        return False, False
    if kind != "labels_latest":
        return False, False
    if decision == "both_good":
        return True, False
    if decision == "both_bad":
        return False, True
    if decision == "a_good":
        return slot == "a", slot == "b"
    if decision == "b_good":
        return slot == "b", slot == "a"
    return False, False


def _resolve_labels_latest_states(
    decision: str,
    image_indices: Sequence[int],
    label: Mapping[str, Any],
) -> tuple[set[int], set[int]]:
    normalized_indices = set(_unique_ints(image_indices))
    if decision == "both_good":
        return normalized_indices, set()
    if decision == "both_bad":
        return set(), normalized_indices
    if decision == "skip":
        return set(), set()
    if decision not in {"a_good", "b_good"}:
        return set(), set()

    chosen_indices = set(_extract_chosen_indices(label))
    chosen_indices = {index for index in chosen_indices if index in normalized_indices}
    if not chosen_indices:
        display_order = _extract_display_order(label, fallback_order=image_indices)
        if decision == "a_good" and len(display_order) >= 1:
            chosen_indices = {display_order[0]}
        elif decision == "b_good" and len(display_order) >= 2:
            chosen_indices = {display_order[1]}
        chosen_indices = {index for index in chosen_indices if index in normalized_indices}
    bad_indices = normalized_indices - chosen_indices if chosen_indices else set()
    return chosen_indices, bad_indices


def _extract_chosen_indices(label: Mapping[str, Any]) -> list[int]:
    chosen = label.get("chosen_image_indices")
    if not isinstance(chosen, list):
        return []
    return _unique_ints(chosen)


def _extract_defect_indices(label: Mapping[str, Any], fallback_order: Sequence[Any]) -> set[int]:
    defect_indices: set[int] = set()
    by_index = label.get("defects_by_image_index")
    if isinstance(by_index, Mapping):
        for key, defects in by_index.items():
            if _has_defect_items(defects):
                defect_indices.add(_safe_int(key))
        if defect_indices:
            return defect_indices

    display_order = _extract_display_order(label, fallback_order=fallback_order)
    if len(display_order) >= 1 and _has_defect_items(label.get("defects_a")):
        defect_indices.add(display_order[0])
    if len(display_order) >= 2 and _has_defect_items(label.get("defects_b")):
        defect_indices.add(display_order[1])
    return defect_indices


def _extract_display_order(label: Mapping[str, Any], fallback_order: Sequence[Any]) -> list[int]:
    display_order = label.get("display_order")
    if isinstance(display_order, list) and display_order:
        normalized = _unique_ints(display_order)
        if normalized:
            return normalized
    return _unique_ints(fallback_order)


def _has_defect_items(value: Any) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return False


def _unique_ints(values: Sequence[Any]) -> list[int]:
    output: list[int] = []
    for value in values:
        normalized = _safe_int(value)
        if normalized not in output:
            output.append(normalized)
    return output


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at line {line_number} of {path}")
            rows.append(payload)
    return rows


def _validate_rows_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at line {line_number} of {path}")


def _read_row_page(path: Path, start: int, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    end = start + limit
    current = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if current >= end:
                break
            if current >= start:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(payload)
            current += 1
    return rows


def _find_row_by_id(path: Path, row_id: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}")
            if payload.get("row_id") == row_id:
                return payload
    raise ViewerNotFoundError(f"Unknown row_id: {row_id}")
