from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

from PIL import Image, ImageOps

from .auth import AuthService
from .catalog import CatalogService, PairRecord
from .common import APP_VERSION, DECISIONS, DEFECT_TAGS, EXPORT_FILENAMES, LabelEventValidationError
from .exports import ExportService
from .filters import FilterEngine, FilterValidationError
from .labels import LabelStore
from .prompt_bolding import PromptBoldingService
from .review import ReviewService
from .x_poster import XPoster, XPosterError


class DpoLabelerApp:
    def __init__(
        self,
        dataset_root: str | Path,
        state_dir: str | Path,
        invite_token: str,
        session_secret: str | None = None,
        preview_max_width: int = 960,
        rescan_seconds: int = 30,
        session_max_age_seconds: int = 60 * 60 * 24 * 7,
        cookie_secure: bool = False,
        review_round_seed: str = "default-round-v1",
        prompt_bold_path: str | Path | None = None,
        x_consumer_key: str | None = None,
        x_consumer_secret: str | None = None,
        x_access_token: str | None = None,
        x_access_token_secret: str | None = None,
        exclude_dirs: "list[str] | tuple[str, ...] | None" = None,
        image_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.preview_max_width = preview_max_width
        self.previews_dir = self.state_dir / "previews"
        self.prompt_bold_path = Path(prompt_bold_path).resolve() if prompt_bold_path else self._default_prompt_bold_path()
        self.prompt_bolding_service = self._load_prompt_bolding_service(self.prompt_bold_path)

        self.catalog_service = CatalogService(
            dataset_root=dataset_root,
            repo_root=Path.cwd().resolve(),
            image_roots=image_roots,
            rescan_seconds=rescan_seconds,
            review_round_seed=review_round_seed,
            exclude_dirs=exclude_dirs,
        )
        self.auth_service = AuthService(
            invite_token=invite_token,
            session_secret=session_secret,
            session_max_age_seconds=session_max_age_seconds,
            cookie_secure=cookie_secure,
        )
        self.label_store = LabelStore(self.state_dir)
        self.filter_engine = FilterEngine()
        self.export_service = ExportService(self.filter_engine)
        self.review_service = ReviewService()
        self.x_poster = XPoster(
            consumer_key=x_consumer_key,
            consumer_secret=x_consumer_secret,
            access_token=x_access_token,
            access_token_secret=x_access_token_secret,
        )
        self._cache_lock = threading.Lock()
        self._catalog_payload_cache: tuple[str, int, dict[str, Any]] | None = None
        self._review_queue_cache: dict[str, tuple[str, int, list[dict[str, Any]]]] = {}
        self._preview_lock = threading.Lock()
        self._preview_build_locks: dict[Path, threading.Lock] = {}
        self._preview_build_semaphore = threading.Semaphore(2)

    @staticmethod
    def _default_prompt_bold_path() -> Path:
        return Path(__file__).resolve().parents[1] / "bold.txt"

    @staticmethod
    def _load_prompt_bolding_service(config_path: Path) -> PromptBoldingService:
        if not config_path.exists():
            print(f"Warning: prompt bold config not found at {config_path}; continuing with bolding disabled.")
            return PromptBoldingService([])
        try:
            service = PromptBoldingService.from_path(config_path)
        except OSError as exc:
            print(f"Warning: could not read prompt bold config {config_path}: {exc}; continuing with bolding disabled.")
            return PromptBoldingService([])
        print(f"Loaded prompt bold config from {config_path} ({service.pattern_count} patterns).")
        return service

    def get_config(self) -> dict[str, Any]:
        return {
            "app_version": APP_VERSION,
            "decisions": DECISIONS,
            "defect_tags": DEFECT_TAGS,
            "filter_schema": self.filter_engine.metadata(),
            "catalog_version": self.catalog_service.get_snapshot().catalog_version,
            "x_post_enabled": self.x_poster.is_configured,
        }

    def close(self) -> None:
        self.catalog_service.close()

    def start_session(self, invite_token: str, reviewer_username: str, client_instance_id: str) -> tuple[dict[str, Any], str]:
        session, cookie_header = self.auth_service.start_session(invite_token, reviewer_username, client_instance_id)
        return {"session": session.to_dict(), "app_version": APP_VERSION}, cookie_header

    def end_session(self) -> tuple[dict[str, Any], str]:
        return {"ok": True}, self.auth_service.end_session_header()

    def get_session(self, cookie_header: str | None) -> dict[str, Any]:
        session = self.auth_service.require_session(cookie_header)
        return {"session": session.to_dict(), "app_version": APP_VERSION}

    def get_catalog(self) -> dict[str, Any]:
        snapshot = self.catalog_service.get_snapshot()
        label_version, latest_labels = self.label_store.get_latest_snapshot()
        cache_key = (snapshot.catalog_version, label_version)
        with self._cache_lock:
            cached = self._catalog_payload_cache
            if cached and cached[:2] == cache_key:
                return cached[2]
        payload = self._build_catalog_payload(snapshot, latest_labels)
        with self._cache_lock:
            self._catalog_payload_cache = (snapshot.catalog_version, label_version, payload)
        return payload

    def _build_catalog_payload(
        self,
        snapshot: Any,
        latest_labels: Mapping[str, Any],
    ) -> dict[str, Any]:
        datasets: dict[str, dict[str, Any]] = {}

        for dataset in snapshot.datasets:
            datasets[dataset.dataset_id] = {
                "dataset_id": dataset.dataset_id,
                "display_name": dataset.display_name,
                "sessions_jsonl": str(dataset.sessions_jsonl),
                "tasks": [],
            }

        for task in snapshot.tasks_by_key.values():
            labeled_pairs = 0
            reviewer_counts: dict[str, int] = {}
            for pair_key in task.pair_keys:
                latest = latest_labels.get(pair_key)
                if latest is None:
                    continue
                labeled_pairs += 1
                reviewer_counts[latest.reviewer_username] = reviewer_counts.get(latest.reviewer_username, 0) + 1
            total_pairs = len(task.pair_keys)
            unlabeled_pairs = total_pairs - labeled_pairs
            reviewed_percent = int(round((labeled_pairs / total_pairs) * 100)) if total_pairs else 0
            datasets[task.dataset_id]["tasks"].append(
                {
                    "task_key": task.task_key,
                    "task_name": task.task_name,
                    "task_yaml_name": task.task_yaml_name,
                    "task_yaml_path": task.task_yaml_path,
                    "task_yaml_sha256": task.task_yaml_sha256,
                    "total_pairs": total_pairs,
                    "labeled_pairs": labeled_pairs,
                    "unlabeled_pairs": unlabeled_pairs,
                    "reviewed_pairs": labeled_pairs,
                    "remaining_pairs": unlabeled_pairs,
                    "reviewed_percent": reviewed_percent,
                    "invalid_pair_count": task.invalid_pair_count,
                    "reviewers": reviewer_counts,
                }
            )

        return {
            "catalog_version": snapshot.catalog_version,
            "review_round_seed": snapshot.review_round_seed,
            "datasets": [
                {
                    **dataset,
                    "tasks": sorted(dataset["tasks"], key=lambda item: (item["task_yaml_name"], item["task_name"], item["task_key"])),
                }
                for dataset in sorted(datasets.values(), key=lambda item: item["display_name"])
            ],
            "warnings": list(snapshot.warnings),
        }

    def create_review_session(
        self,
        cookie_header: str | None,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        session = self.auth_service.require_session(cookie_header)
        snapshot = self.catalog_service.get_snapshot()
        task_keys = payload.get("task_keys")
        if not isinstance(task_keys, list):
            raise LabelEventValidationError("task_keys must be a list")
        selection = self.review_service.create_review_selection(
            snapshot=snapshot,
            reviewer_username=session.reviewer_username,
            task_keys=[str(item) for item in task_keys],
            mode=str(payload.get("mode", "sequence")),
        )
        queue = self._get_review_queue_items(snapshot, selection)
        return {
            "review": selection.to_dict(),
            "queue_total": len(queue),
            "first_unlabeled_index": self._first_unlabeled_index(queue),
            "catalog_version": snapshot.catalog_version,
        }

    def get_review_queue(self, cookie_header: str | None, review_id: str, cursor: int, limit: int) -> dict[str, Any]:
        self.auth_service.require_session(cookie_header)
        snapshot = self.catalog_service.get_snapshot()
        selection = self.review_service.get_review_selection(review_id)
        queue = self._get_review_queue_items(snapshot, selection)
        start = max(int(cursor), 0)
        end = min(start + max(int(limit), 1), len(queue))
        return {
            "review": selection.to_dict(),
            "items": [dict(item) for item in queue[start:end]],
            "cursor": start,
            "next_cursor": end if end < len(queue) else None,
            "total": len(queue),
            "first_unlabeled_index": self._first_unlabeled_index(queue),
            "catalog_version": snapshot.catalog_version,
        }

    def get_review_pair(self, cookie_header: str | None, review_id: str, dataset_id: str, session_id: str) -> dict[str, Any]:
        self.auth_service.require_session(cookie_header)
        snapshot = self.catalog_service.get_snapshot()
        selection = self.review_service.get_review_selection(review_id)
        pair = self.review_service.validate_pair_in_selection(snapshot, selection, dataset_id, session_id)
        if self.label_store.get_latest_for_pair(pair.pair_key) is not None:
            raise KeyError("Pair not available in this review session")
        return self._build_pair_payload(pair)

    def submit_label_event(self, cookie_header: str | None, payload: Mapping[str, Any]) -> dict[str, Any]:
        session = self.auth_service.require_session(cookie_header)
        dataset_id = str(payload.get("dataset_id", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        if not dataset_id or not session_id:
            raise LabelEventValidationError("dataset_id and session_id are required")
        pair_key = f"{dataset_id}::{session_id}"
        snapshot = self.catalog_service.get_snapshot()
        pair = snapshot.pairs_by_key.get(pair_key)
        if pair is None:
            raise LabelEventValidationError("Unknown dataset_id/session_id")
        event = self.label_store.store_event(
            pair,
            reviewer_username=session.reviewer_username,
            client_instance_id=session.client_instance_id,
            payload=payload,
        )
        return event.to_dict()

    def preview_export(self, cookie_header: str | None, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.auth_service.require_session(cookie_header)
        export_type = self._validate_export_type(payload.get("export_type"))
        filter_ast = self.filter_engine.validate(payload.get("filter"))
        snapshot = self.catalog_service.get_snapshot()
        return {
            "export_type": export_type,
            "count": self.export_service.preview_count(export_type, filter_ast, snapshot, self.label_store),
            "catalog_version": snapshot.catalog_version,
        }

    def export_text(self, export_type: str, filter_ast: Any) -> str:
        export_type = self._validate_export_type(export_type)
        validated_filter = self.filter_engine.validate(filter_ast)
        snapshot = self.catalog_service.get_snapshot()
        return self.export_service.render_export(export_type, validated_filter, snapshot, self.label_store)

    def export_text_from_request(self, cookie_header: str | None, payload: Mapping[str, Any]) -> tuple[str, str]:
        self.auth_service.require_session(cookie_header)
        export_type = self._validate_export_type(payload.get("export_type"))
        text = self.export_text(export_type, payload.get("filter"))
        return text, self.export_service.default_filename(export_type)

    def create_twitter_share(self, cookie_header: str | None, payload: Mapping[str, Any], base_url: str) -> dict[str, Any]:
        self.auth_service.require_session(cookie_header)
        dataset_id = str(payload.get("dataset_id", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        try:
            image_index = int(payload.get("image_index"))
        except (TypeError, ValueError) as exc:
            raise LabelEventValidationError("image_index must be an integer") from exc
        _, image = self._require_pair_image(dataset_id, session_id, image_index)
        share_token = self._encode_public_share_token(dataset_id, session_id, image_index)
        normalized_base_url = base_url.rstrip("/")
        share_url = f"{normalized_base_url}/public-share/{share_token}"
        text = self._build_twitter_share_text(image.positive_prompt)
        twitter_intent_url = f"https://twitter.com/intent/tweet?{urlencode({'text': text, 'url': share_url})}"
        return {
            "share_url": share_url,
            "twitter_intent_url": twitter_intent_url,
            "text": text,
        }

    def post_twitter_image(self, cookie_header: str | None, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.auth_service.require_session(cookie_header)
        dataset_id = str(payload.get("dataset_id", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        try:
            image_index = int(payload.get("image_index"))
        except (TypeError, ValueError) as exc:
            raise LabelEventValidationError("image_index must be an integer") from exc
        _, image = self._require_pair_image(dataset_id, session_id, image_index)
        preview_path = self._get_preview_path(dataset_id, session_id, image_index)
        text = self._build_twitter_share_text(image.positive_prompt)
        try:
            result = self.x_poster.post_image_tweet(
                preview_path.read_bytes(),
                "image/jpeg",
                text,
            )
        except XPosterError as exc:
            raise LabelEventValidationError(str(exc)) from exc
        return {
            "tweet_id": result.get("tweet_id"),
            "tweet_url": result.get("tweet_url"),
            "screen_name": result.get("screen_name"),
            "text": text,
        }

    def get_image_path(self, cookie_header: str | None, dataset_id: str, session_id: str, image_index: int) -> Path:
        self.auth_service.require_session(cookie_header)
        return self._get_image_path(dataset_id, session_id, image_index)

    def get_preview_path(self, cookie_header: str | None, dataset_id: str, session_id: str, image_index: int) -> Path:
        self.auth_service.require_session(cookie_header)
        return self._get_preview_path(dataset_id, session_id, image_index)

    def get_public_share_preview_path(self, share_token: str) -> Path:
        dataset_id, session_id, image_index = self._decode_public_share_token(share_token)
        return self._get_preview_path(dataset_id, session_id, image_index)

    def render_public_share_page(self, share_token: str, base_url: str) -> str:
        dataset_id, session_id, image_index = self._decode_public_share_token(share_token)
        pair, image = self._require_pair_image(dataset_id, session_id, image_index)
        normalized_base_url = base_url.rstrip("/")
        image_url = f"{normalized_base_url}/public-share/media/{share_token}.jpg"
        title = "AI Generated Image"
        description = self._build_public_share_description(image.positive_prompt)
        page_title = escape(title)
        page_description = escape(description)
        page_prompt = escape(image.positive_prompt)
        page_image_url = escape(image_url)
        page_task = escape(pair.task_name)
        return (
            "<!doctype html>"
            "<html lang='en'>"
            "<head>"
            "<meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1, viewport-fit=cover'>"
            f"<title>{page_title}</title>"
            "<meta property='og:type' content='website'>"
            f"<meta property='og:title' content='{page_title}'>"
            f"<meta property='og:description' content='{page_description}'>"
            f"<meta property='og:image' content='{page_image_url}'>"
            "<meta name='twitter:card' content='summary_large_image'>"
            f"<meta name='twitter:title' content='{page_title}'>"
            f"<meta name='twitter:description' content='{page_description}'>"
            f"<meta name='twitter:image' content='{page_image_url}'>"
            "<style>"
            "body{margin:0;padding:24px;font-family:IBM Plex Sans,Segoe UI,sans-serif;background:#f6f1e8;color:#20262f;}"
            ".shell{max-width:880px;margin:0 auto;display:grid;gap:18px;}"
            ".card{background:#fff;border:1px solid #d8cfbf;border-radius:16px;padding:18px;box-shadow:0 10px 24px rgba(80,62,33,.08);}"
            "img{display:block;width:100%;border-radius:12px;background:#e9e1d5;}"
            ".eyebrow{margin:0 0 6px;color:#1f6aa5;text-transform:uppercase;letter-spacing:.08em;font-size:.74rem;font-weight:700;}"
            "h1,p{margin:0;}"
            ".prompt{white-space:pre-wrap;line-height:1.6;overflow-wrap:anywhere;}"
            "</style>"
            "</head>"
            "<body>"
            "<main class='shell'>"
            "<section class='card'>"
            "<p class='eyebrow'>AI Generated</p>"
            f"<h1>{page_task}</h1>"
            f"<p>{page_description}</p>"
            "</section>"
            "<section class='card'>"
            f"<img src='{page_image_url}' alt='AI generated image preview'>"
            "</section>"
            "<section class='card'>"
            "<p class='eyebrow'>Prompt</p>"
            f"<p class='prompt'>{page_prompt}</p>"
            "</section>"
            "</main>"
            "</body>"
            "</html>"
        )

    def _get_image_path(self, dataset_id: str, session_id: str, image_index: int) -> Path:
        _, image = self._require_pair_image(dataset_id, session_id, image_index)
        return Path(image.saved_path)

    def _get_preview_path(self, dataset_id: str, session_id: str, image_index: int) -> Path:
        original = self._get_image_path(dataset_id, session_id, image_index)
        preview_path = self.previews_dir / dataset_id / session_id / f"{image_index}.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if self._preview_is_fresh(preview_path, original):
            return preview_path
        build_lock = self._get_preview_build_lock(preview_path)
        with build_lock:
            if self._preview_is_fresh(preview_path, original):
                return preview_path
            with self._preview_build_semaphore:
                if self._preview_is_fresh(preview_path, original):
                    return preview_path
                with Image.open(original) as image:
                    image = ImageOps.exif_transpose(image)
                    if image.mode not in {"RGB", "L"}:
                        image = image.convert("RGB")
                    image.thumbnail((self.preview_max_width, self.preview_max_width))
                    image.save(preview_path, format="JPEG", quality=84, optimize=True)
        return preview_path

    def _get_review_queue_items(self, snapshot: Any, selection: Any) -> list[dict[str, Any]]:
        label_version, latest_labels = self.label_store.get_latest_snapshot()
        cache_key = (snapshot.catalog_version, label_version)
        with self._cache_lock:
            cached = self._review_queue_cache.get(selection.review_id)
            if cached and cached[:2] == cache_key:
                return cached[2]
        queue = self.review_service.build_queue(snapshot, selection, latest_labels)
        with self._cache_lock:
            self._review_queue_cache[selection.review_id] = (snapshot.catalog_version, label_version, queue)
        return queue

    @staticmethod
    def _preview_is_fresh(preview_path: Path, original_path: Path) -> bool:
        return preview_path.exists() and preview_path.stat().st_mtime >= original_path.stat().st_mtime

    def _get_preview_build_lock(self, preview_path: Path) -> threading.Lock:
        with self._preview_lock:
            lock = self._preview_build_locks.get(preview_path)
            if lock is None:
                lock = threading.Lock()
                self._preview_build_locks[preview_path] = lock
            return lock

    def _build_pair_payload(self, pair: PairRecord) -> dict[str, Any]:
        latest = self.label_store.get_latest_for_pair(pair.pair_key)
        images_payload = []
        for image in pair.images:
            images_payload.append(
                {
                    **image.to_dict(),
                    "positive_prompt_segments": self.prompt_bolding_service.build_segments(image.positive_prompt),
                    "preview_url": f"/media/preview/{pair.dataset_id}/{pair.session_id}/{image.image_index}.jpg",
                    "original_url": f"/media/original/{pair.dataset_id}/{pair.session_id}/{image.image_index}",
                }
            )
        return {
            "pair_key": pair.pair_key,
            "dataset_id": pair.dataset_id,
            "dataset_display_name": pair.dataset_display_name,
            "task_key": pair.task_key,
            "task_name": pair.task_name,
            "task_yaml_name": pair.task_yaml_name,
            "task_yaml_path": pair.task_yaml_path,
            "task_yaml_sha256": pair.task_yaml_sha256,
            "compiler_version": pair.compiler_version,
            "global_seed": pair.global_seed,
            "workflow_name": pair.workflow_name,
            "primary_ckpt": pair.primary_ckpt,
            "session_id": pair.session_id,
            "session_index": pair.session_index,
            "images": images_payload,
            "latest_label": latest.to_dict() if latest else None,
        }

    def _require_pair_image(self, dataset_id: str, session_id: str, image_index: int) -> tuple[PairRecord, Any]:
        if not dataset_id or not session_id:
            raise LabelEventValidationError("dataset_id and session_id are required")
        snapshot = self.catalog_service.get_snapshot()
        try:
            pair = snapshot.pairs_by_key[f"{dataset_id}::{session_id}"]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset/session pair: {dataset_id}/{session_id}") from exc
        for image in pair.images:
            if int(image.image_index) == int(image_index):
                return pair, image
        raise KeyError(f"No image_index={image_index} for {dataset_id}/{session_id}")

    def _encode_public_share_token(self, dataset_id: str, session_id: str, image_index: int) -> str:
        payload = json.dumps(
            {
                "dataset_id": dataset_id,
                "session_id": session_id,
                "image_index": int(image_index),
                "kind": "public-share-v1",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        signature = hmac.new(
            self.auth_service.session_secret.encode("utf-8"),
            encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return f"{encoded}.{signature}"

    def _decode_public_share_token(self, token: str) -> tuple[str, str, int]:
        try:
            encoded, signature = token.split(".", 1)
        except ValueError as exc:
            raise LabelEventValidationError("Invalid share token") from exc
        expected = hmac.new(
            self.auth_service.session_secret.encode("utf-8"),
            encoded.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise LabelEventValidationError("Invalid share token")
        try:
            padded = encoded + ("=" * (-len(encoded) % 4))
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        except Exception as exc:
            raise LabelEventValidationError("Invalid share token") from exc
        if not isinstance(payload, Mapping) or payload.get("kind") != "public-share-v1":
            raise LabelEventValidationError("Invalid share token")
        try:
            image_index = int(payload.get("image_index"))
        except (TypeError, ValueError) as exc:
            raise LabelEventValidationError("Invalid share token") from exc
        return str(payload.get("dataset_id", "")).strip(), str(payload.get("session_id", "")).strip(), image_index

    @staticmethod
    def _build_twitter_share_text(prompt: str) -> str:
        return f"#AIGenerated\n\n{prompt}".strip()

    @staticmethod
    def _build_public_share_description(prompt: str) -> str:
        text = DpoLabelerApp._build_twitter_share_text(prompt)
        if len(text) <= 280:
            return text
        return f"{text[:277]}..."

    @staticmethod
    def _first_unlabeled_index(queue: list[dict[str, Any]]) -> int:
        for index, item in enumerate(queue):
            if not item["is_labeled"]:
                return index
        return len(queue)

    @staticmethod
    def _validate_export_type(value: Any) -> str:
        export_type = str(value or "").strip()
        if export_type not in EXPORT_FILENAMES:
            raise LabelEventValidationError(
                f"Unsupported export_type: {export_type or '<empty>'}"
            )
        return export_type


__all__ = [
    "APP_VERSION",
    "DECISIONS",
    "DEFECT_TAGS",
    "EXPORT_FILENAMES",
    "DpoLabelerApp",
    "LabelEventValidationError",
    "FilterValidationError",
]
