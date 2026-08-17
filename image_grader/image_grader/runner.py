from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

from .config import GraderConfig
from .fingerprint import fingerprint_file
from .io import ImageJob, append_jsonl
from .models import ModelBackend, create_registry
from .state import ScoreState

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


@dataclass
class PreparedImage:
    request_id: str | None
    image_path: Path
    image_id: str
    size_bytes: int
    mtime_ns: int
    width: int
    height: int
    preprocess_policy: str = "native"
    metadata: dict[str, Any] = field(default_factory=dict)
    image: Any = None

    def image_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.image_path),
            "image_id": self.image_id,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class RunStats:
    seen: int = 0
    emitted: int = 0
    cached_scores: int = 0
    computed_scores: int = 0
    failed_images: int = 0


class BatchRunner:
    def __init__(
        self,
        config: GraderConfig,
        *,
        state_dir: str | Path,
        registry: Any | None = None,
        state: ScoreState | None = None,
    ) -> None:
        self.config = config
        self.state = state or ScoreState(state_dir)
        self.registry = registry or create_registry(config)

    def close(self) -> None:
        self.registry.close()
        self.state.close()

    def score_jobs(
        self,
        jobs: Iterable[ImageJob],
        *,
        model_ids: tuple[str, ...] | list[str] | None = None,
        output_path: str | Path | None = None,
        emit_cached: bool = False,
        chunk_size: int | None = None,
        show_progress: bool = True,
        preprocess_policy: str = "native",
    ) -> RunStats:
        preprocess_policy = _normalize_preprocess_policy(preprocess_policy)
        selected_models = self.config.selected_model_ids(tuple(model_ids) if model_ids else None)
        chunk_size = int(chunk_size or _default_chunk_size(self.config, selected_models))
        stats = _MutableRunStats()
        progress = _progress(desc="Scoring images", disable=not show_progress)
        try:
            for chunk in _chunks(jobs, chunk_size):
                rows = self.score_job_chunk(
                    chunk,
                    selected_models=selected_models,
                    preprocess_policy=preprocess_policy,
                )
                for row in rows:
                    stats.seen += 1
                    if not row["ok"] and not row.get("scores"):
                        stats.failed_images += 1
                    for score in row.get("scores", {}).values():
                        if score.get("cached"):
                            stats.cached_scores += 1
                        else:
                            stats.computed_scores += 1
                    if output_path is not None and (emit_cached or _row_has_new_score(row) or not row["ok"]):
                        append_jsonl(output_path, row)
                        stats.emitted += 1
                    progress.update(1)
        finally:
            progress.close()
        return stats.freeze()

    def score_job_chunk(
        self,
        jobs: list[ImageJob],
        *,
        selected_models: tuple[str, ...],
        preprocess_policy: str = "native",
    ) -> list[dict[str, Any]]:
        preprocess_policy = _normalize_preprocess_policy(preprocess_policy)
        prepared_rows: list[PreparedImage] = []
        output_rows: list[dict[str, Any]] = []
        for job in jobs:
            try:
                prepared = prepare_image(job, self.config, preprocess_policy=preprocess_policy)
            except Exception as exc:
                output_rows.append(_image_error_row(job, exc, preprocess_policy=preprocess_policy))
                continue
            prepared_rows.append(prepared)
            output_rows.append(_base_output_row(prepared))

        rows_by_request = {
            row["request_id"] or row["image"]["path"]: row
            for row in output_rows
            if row.get("image") and row["ok"]
        }
        for model_id in selected_models:
            config_hash = self.config.model_config_hash(model_id, preprocess_policy=preprocess_policy)
            pending: list[PreparedImage] = []
            for prepared in prepared_rows:
                cached = self.state.get_score(prepared.image_id, model_id, config_hash)
                output_row = rows_by_request[prepared.request_id or str(prepared.image_path)]
                if cached is not None:
                    output_row["scores"][model_id] = cached.to_output(cached=True)
                else:
                    pending.append(prepared)
            if not pending:
                continue
            backend = self.registry.get(model_id)
            for sub_batch in _chunks(pending, self.config.models[model_id].batch_size):
                scores = _score_with_error_capture(backend, sub_batch)
                for prepared, score in zip(sub_batch, scores, strict=True):
                    score = {**score, "cached": False}
                    output_row = rows_by_request[prepared.request_id or str(prepared.image_path)]
                    output_row["scores"][model_id] = score
                    self.state.put_score(
                        image_id=prepared.image_id,
                        model_id=model_id,
                        config_hash=config_hash,
                        image_path=str(prepared.image_path),
                        size_bytes=prepared.size_bytes,
                        width=prepared.width,
                        height=prepared.height,
                        score=score,
                    )

        for row in output_rows:
            if row["ok"]:
                row["ok"] = all(score.get("ok") for score in row["scores"].values()) and bool(row["scores"])
                if not row["ok"]:
                    row["error"] = "one or more model scores failed"
            row["elapsed_ms"] = round((time.time() - row["_started_at"]) * 1000.0, 3)
            del row["_started_at"]
        return output_rows


def prepare_image(job: ImageJob, config: GraderConfig, preprocess_policy: str = "native") -> PreparedImage:
    from PIL import Image, ImageOps

    preprocess_policy = _normalize_preprocess_policy(preprocess_policy)
    fingerprint = fingerprint_file(job.image_path, config.cache)
    with Image.open(job.image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        loaded = preprocess_pil_image(image, preprocess_policy)
    return PreparedImage(
        request_id=job.request_id,
        image_path=job.image_path,
        image_id=fingerprint.image_id,
        size_bytes=fingerprint.size_bytes,
        mtime_ns=fingerprint.mtime_ns,
        width=width,
        height=height,
        preprocess_policy=preprocess_policy,
        metadata=dict(job.metadata),
        image=loaded,
    )


def preprocess_pil_image(image: Any, policy: str) -> Any:
    policy = _normalize_preprocess_policy(policy)
    if policy == "native":
        return image.copy()
    if policy == "center_crop_square":
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))
    if policy == "fit_pad_square":
        from PIL import Image

        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas
    raise ValueError(f"unsupported preprocess policy: {policy}")


def _normalize_preprocess_policy(policy: str) -> str:
    normalized = str(policy or "native").strip()
    if normalized not in {"native", "fit_pad_square", "center_crop_square"}:
        raise ValueError("preprocess_policy must be one of: native, fit_pad_square, center_crop_square")
    return normalized


def _score_with_error_capture(backend: ModelBackend, images: list[PreparedImage]) -> list[dict[str, Any]]:
    try:
        return backend.score_batch(images)
    except Exception as exc:
        return [
            {
                "ok": False,
                "score": None,
                "scale": "0_10",
                "raw": {},
                "error": str(exc),
            }
            for _ in images
        ]


def _base_output_row(prepared: PreparedImage) -> dict[str, Any]:
    return {
        "request_id": prepared.request_id,
        "ok": True,
        "image": prepared.image_payload(),
        "metadata": prepared.metadata,
        "preprocess_policy": prepared.preprocess_policy,
        "scores": {},
        "error": None,
        "_started_at": time.time(),
    }


def _image_error_row(job: ImageJob, exc: Exception, *, preprocess_policy: str = "native") -> dict[str, Any]:
    return {
        "request_id": job.request_id,
        "ok": False,
        "image": {"path": str(job.image_path)},
        "metadata": job.metadata,
        "preprocess_policy": preprocess_policy,
        "scores": {},
        "error": str(exc),
        "elapsed_ms": 0.0,
    }


def _row_has_new_score(row: dict[str, Any]) -> bool:
    return any(not score.get("cached") for score in row.get("scores", {}).values())


def _default_chunk_size(config: GraderConfig, model_ids: tuple[str, ...]) -> int:
    if not model_ids:
        return 16
    return max(config.models[model_id].batch_size for model_id in model_ids)


T = TypeVar("T")


def _chunks(items: Iterable[T], size: int) -> Iterator[list[T]]:
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _progress(*, desc: str, disable: bool) -> Any:
    if disable or _tqdm is None:
        return _NoOpProgress()
    return _tqdm(desc=desc, unit="img")


class _NoOpProgress:
    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


class _MutableRunStats:
    def __init__(self) -> None:
        self.seen = 0
        self.emitted = 0
        self.cached_scores = 0
        self.computed_scores = 0
        self.failed_images = 0

    def freeze(self) -> RunStats:
        return RunStats(
            seen=self.seen,
            emitted=self.emitted,
            cached_scores=self.cached_scores,
            computed_scores=self.computed_scores,
            failed_images=self.failed_images,
        )
