from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import CatalogError, load_jsonl, normalize_text, resolve_existing_path, stable_hex, to_jsonable


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    sessions_jsonl: Path


@dataclass(frozen=True)
class SessionImage:
    image_index: int
    image_name: str
    saved_path: str
    positive_prompt: str
    negative_prompt: str
    ckpt: str
    seed: Any
    status: str
    workflow_name: str
    task_yaml_path: str
    prompt_seed: Any
    prompt_seed_control: Any
    generation_seed_control: Any
    width: Any
    height: Any
    cfg: Any
    steps: Any
    runtime_seed_values: dict[str, Any]
    lora_stack_config: Any
    runner_result: Any
    saved_filename: Any
    original_filename: Any

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class PairRecord:
    pair_key: str
    dataset_id: str
    dataset_display_name: str
    task_key: str
    task_name: str
    task_yaml_name: str
    task_yaml_path: str
    task_yaml_sha256: str
    compiler_version: str
    global_seed: Any
    workflow_name: str
    primary_ckpt: str
    session_id: str
    session_index: int
    images: tuple[SessionImage, SessionImage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_key": self.pair_key,
            "dataset_id": self.dataset_id,
            "dataset_display_name": self.dataset_display_name,
            "task_key": self.task_key,
            "task_name": self.task_name,
            "task_yaml_name": self.task_yaml_name,
            "task_yaml_path": self.task_yaml_path,
            "task_yaml_sha256": self.task_yaml_sha256,
            "compiler_version": self.compiler_version,
            "global_seed": self.global_seed,
            "workflow_name": self.workflow_name,
            "primary_ckpt": self.primary_ckpt,
            "session_id": self.session_id,
            "session_index": self.session_index,
            "images": [image.to_dict() for image in self.images],
        }


@dataclass(frozen=True)
class TaskRecord:
    task_key: str
    dataset_id: str
    dataset_display_name: str
    task_name: str
    task_yaml_name: str
    task_yaml_path: str
    task_yaml_sha256: str
    pair_keys: tuple[str, ...]
    invalid_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "dataset_id": self.dataset_id,
            "dataset_display_name": self.dataset_display_name,
            "task_name": self.task_name,
            "task_yaml_name": self.task_yaml_name,
            "task_yaml_path": self.task_yaml_path,
            "task_yaml_sha256": self.task_yaml_sha256,
            "pair_keys": list(self.pair_keys),
            "invalid_pair_count": self.invalid_pair_count,
        }


@dataclass(frozen=True)
class CatalogSnapshot:
    review_round_seed: str
    catalog_version: str
    datasets: tuple[DatasetConfig, ...]
    tasks_by_key: dict[str, TaskRecord]
    pairs_by_key: dict[str, PairRecord]
    warnings: tuple[dict[str, Any], ...]


class CatalogService:
    def __init__(
        self,
        dataset_root: str | Path,
        repo_root: str | Path,
        rescan_seconds: int = 30,
        review_round_seed: str = "default-round-v1",
        exclude_dirs: "list[str] | tuple[str, ...] | None" = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.rescan_seconds = max(int(rescan_seconds), 5)
        self.review_round_seed = normalize_text(review_round_seed) or "default-round-v1"
        self.exclude_dirs: tuple[str, ...] = tuple(exclude_dirs or ())
        self._lock = threading.Lock()
        self._scan_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._snapshot = self._scan_root()
        self._last_scan_monotonic = time.monotonic()
        self._last_scan_error: Exception | None = None
        self._rescan_thread = threading.Thread(target=self._rescan_loop, name="dpo-labeler-catalog-rescan", daemon=True)
        self._rescan_thread.start()

    def get_snapshot(self) -> CatalogSnapshot:
        with self._lock:
            return self._snapshot

    def force_rescan(self) -> CatalogSnapshot:
        return self._scan_and_store_snapshot()

    def rescan_if_due(self) -> CatalogSnapshot:
        return self.get_snapshot()

    def close(self) -> None:
        self._stop_event.set()
        if self._rescan_thread.is_alive() and threading.current_thread() is not self._rescan_thread:
            self._rescan_thread.join(timeout=1)

    def _rescan_loop(self) -> None:
        while not self._stop_event.wait(self.rescan_seconds):
            self._background_rescan()

    def _background_rescan(self) -> None:
        try:
            self._scan_and_store_snapshot()
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_scan_monotonic = time.monotonic()
                self._last_scan_error = exc

    def _scan_and_store_snapshot(self) -> CatalogSnapshot:
        with self._scan_lock:
            snapshot = self._scan_root()
            with self._lock:
                self._snapshot = snapshot
                self._last_scan_monotonic = time.monotonic()
                self._last_scan_error = None
                return self._snapshot

    def _scan_root(self) -> CatalogSnapshot:
        if not self.dataset_root.exists():
            raise CatalogError(f"dataset_root does not exist: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            raise CatalogError(f"dataset_root must be a directory: {self.dataset_root}")

        datasets = self._discover_datasets()
        tasks: dict[str, dict[str, Any]] = {}
        pairs_by_key: dict[str, PairRecord] = {}
        warnings: list[dict[str, Any]] = []

        for dataset in datasets:
            rows = load_jsonl(dataset.sessions_jsonl)
            for row in rows:
                normalized = self._normalize_session(dataset, row)
                if isinstance(normalized, PairRecord):
                    pairs_by_key[normalized.pair_key] = normalized
                    task = tasks.setdefault(
                        normalized.task_key,
                        {
                            "task_key": normalized.task_key,
                            "dataset_id": normalized.dataset_id,
                            "dataset_display_name": normalized.dataset_display_name,
                            "task_name": normalized.task_name,
                            "task_yaml_name": normalized.task_yaml_name,
                            "task_yaml_path": normalized.task_yaml_path,
                            "task_yaml_sha256": normalized.task_yaml_sha256,
                            "pair_keys": [],
                            "invalid_pair_count": 0,
                        },
                    )
                    task["pair_keys"].append(normalized.pair_key)
                    continue

                warning = normalized
                warnings.append(warning)
                task_key = warning.get("task_key")
                if task_key:
                    task = tasks.setdefault(
                        task_key,
                        {
                            "task_key": task_key,
                            "dataset_id": warning["dataset_id"],
                            "dataset_display_name": warning["dataset_display_name"],
                            "task_name": warning["task_name"],
                            "task_yaml_name": warning["task_yaml_name"],
                            "task_yaml_path": warning["task_yaml_path"],
                            "task_yaml_sha256": warning["task_yaml_sha256"],
                            "pair_keys": [],
                            "invalid_pair_count": 0,
                        },
                    )
                    task["invalid_pair_count"] += 1

        tasks_by_key = {
            task_key: TaskRecord(
                task_key=task["task_key"],
                dataset_id=task["dataset_id"],
                dataset_display_name=task["dataset_display_name"],
                task_name=task["task_name"],
                task_yaml_name=task["task_yaml_name"],
                task_yaml_path=task["task_yaml_path"],
                task_yaml_sha256=task["task_yaml_sha256"],
                pair_keys=tuple(sorted(task["pair_keys"])),
                invalid_pair_count=int(task["invalid_pair_count"]),
            )
            for task_key, task in sorted(tasks.items())
        }

        version_basis = {
            "dataset_root": str(self.dataset_root),
            "review_round_seed": self.review_round_seed,
            "datasets": [
                (dataset.dataset_id, str(dataset.sessions_jsonl), dataset.sessions_jsonl.stat().st_mtime_ns)
                for dataset in datasets
            ],
            "task_keys": sorted(tasks_by_key),
            "pair_keys": sorted(pairs_by_key),
            "warnings": warnings,
        }
        catalog_version = stable_hex(json.dumps(version_basis, ensure_ascii=False, sort_keys=True))
        return CatalogSnapshot(
            review_round_seed=self.review_round_seed,
            catalog_version=catalog_version,
            datasets=tuple(datasets),
            tasks_by_key=tasks_by_key,
            pairs_by_key=pairs_by_key,
            warnings=tuple(warnings),
        )

    def _is_excluded(self, sessions_path: Path) -> bool:
        if not self.exclude_dirs:
            return False
        import fnmatch
        try:
            rel = sessions_path.parent.relative_to(self.dataset_root)
        except ValueError:
            return False
        segments = [seg for seg in rel.as_posix().split("/") if seg and seg != "."]
        for pattern in self.exclude_dirs:
            for seg in segments:
                if fnmatch.fnmatch(seg, pattern):
                    return True
        return False

    def _discover_datasets(self) -> list[DatasetConfig]:
        sessions_paths = sorted(path.resolve() for path in self.dataset_root.rglob("sessions.jsonl") if path.is_file())
        if self.exclude_dirs:
            sessions_paths = [p for p in sessions_paths if not self._is_excluded(p)]
        datasets: list[DatasetConfig] = []
        seen_dataset_ids: set[str] = set()
        for sessions_path in sessions_paths:
            relative_parent = sessions_path.parent.relative_to(self.dataset_root)
            relative_text = relative_parent.as_posix()
            dataset_id = self._dataset_id(relative_parent)
            if dataset_id in seen_dataset_ids:
                dataset_id = f"{dataset_id}-{stable_hex(relative_text, sessions_path)[:8]}"
            seen_dataset_ids.add(dataset_id)
            display_name = relative_text if relative_text and relative_text != "." else (self.dataset_root.name or "root")
            datasets.append(
                DatasetConfig(
                    dataset_id=dataset_id,
                    display_name=display_name,
                    sessions_jsonl=sessions_path,
                )
            )
        return datasets

    def _normalize_session(self, dataset: DatasetConfig, row: dict[str, Any]) -> PairRecord | dict[str, Any]:
        session_id = normalize_text(row.get("session_id"))
        task_yaml_path = normalize_text(row.get("task_yaml_path"))
        task_yaml_sha256 = normalize_text(row.get("task_yaml_sha256"))
        task_name = normalize_text(row.get("task_name")) or Path(task_yaml_path or "unknown.yaml").stem
        task_yaml_name = Path(task_yaml_path or "unknown.yaml").name
        task_key = self._task_key(dataset.dataset_id, task_yaml_sha256, task_yaml_path, task_name)

        def warning(reason: str, *, detail: str | None = None) -> dict[str, Any]:
            return {
                "dataset_id": dataset.dataset_id,
                "dataset_display_name": dataset.display_name,
                "task_key": task_key,
                "task_name": task_name,
                "task_yaml_name": task_yaml_name,
                "task_yaml_path": task_yaml_path,
                "task_yaml_sha256": task_yaml_sha256,
                "session_id": session_id,
                "reason": reason,
                "detail": detail or "",
            }

        if not session_id:
            return warning("missing_session_id")

        images = row.get("images")
        if not isinstance(images, list) or len(images) != 2:
            return warning("invalid_pair_size", detail="session must contain exactly 2 images")

        sorted_images = sorted(images, key=lambda image: (image.get("image_index", 0), image.get("image_name", "")))
        path_bases = [dataset.sessions_jsonl.parent, self.dataset_root, self.repo_root]
        normalized_images: list[SessionImage] = []
        for image_index, image in enumerate(sorted_images):
            saved_path_value = normalize_text(image.get("saved_path"))
            saved_path = resolve_existing_path(
                saved_path_value,
                path_bases,
                allowed_roots=(self.dataset_root,),
            )
            if saved_path is None:
                return warning(
                    "missing_or_disallowed_image_path",
                    detail=f"image_index={image_index} saved_path={saved_path_value!r}",
                )
            normalized_images.append(
                SessionImage(
                    image_index=int(image.get("image_index", image_index)),
                    image_name=normalize_text(image.get("image_name")) or f"image_{image_index}",
                    saved_path=str(saved_path),
                    positive_prompt=str(image.get("positive_prompt", "")),
                    negative_prompt=str(image.get("negative_prompt", "")),
                    ckpt=str(image.get("ckpt", "")),
                    seed=image.get("seed"),
                    status=str(image.get("status", "")),
                    workflow_name=str(image.get("workflow_name", "")),
                    task_yaml_path=str(image.get("task_yaml_path", task_yaml_path)),
                    prompt_seed=image.get("prompt_seed"),
                    prompt_seed_control=image.get("prompt_seed_control"),
                    generation_seed_control=image.get("generation_seed_control"),
                    width=image.get("width"),
                    height=image.get("height"),
                    cfg=image.get("cfg"),
                    steps=image.get("steps"),
                    runtime_seed_values=dict(image.get("runtime_seed_values", {})),
                    lora_stack_config=image.get("lora_stack_config"),
                    runner_result=image.get("runner_result"),
                    saved_filename=image.get("saved_filename"),
                    original_filename=image.get("original_filename"),
                )
            )

        pair_key = f"{dataset.dataset_id}::{session_id}"
        first = normalized_images[0]
        return PairRecord(
            pair_key=pair_key,
            dataset_id=dataset.dataset_id,
            dataset_display_name=dataset.display_name,
            task_key=task_key,
            task_name=task_name,
            task_yaml_name=task_yaml_name,
            task_yaml_path=task_yaml_path,
            task_yaml_sha256=task_yaml_sha256,
            compiler_version=str(row.get("compiler_version", "")),
            global_seed=row.get("global_seed"),
            workflow_name=first.workflow_name,
            primary_ckpt=first.ckpt,
            session_id=session_id,
            session_index=int(row.get("session_index", 0)),
            images=(normalized_images[0], normalized_images[1]),
        )

    @staticmethod
    def _task_key(dataset_id: str, task_yaml_sha256: str, task_yaml_path: str, task_name: str) -> str:
        basis = normalize_text(task_yaml_sha256)
        if not basis:
            basis = stable_hex(dataset_id, task_yaml_path, task_name)[:16]
        return f"{dataset_id}::{basis}"

    @staticmethod
    def _dataset_id(relative_parent: Path) -> str:
        if relative_parent == Path("."):
            return "root"
        parts: list[str] = []
        for part in relative_parent.parts:
            cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("._-").lower()
            if cleaned:
                parts.append(cleaned)
        if parts:
            return "__".join(parts)
        return f"dataset-{stable_hex(relative_parent.as_posix())[:8]}"
