from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from checkpoint_registry import resolve_checkpoint
from image_grader.config import GraderConfig
from image_grader.io import ImageJob
from image_grader.runner import BatchRunner, preprocess_pil_image

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None


PREPROCESS_POLICIES = ("native", "fit_pad_square", "center_crop_square")
APP_LABEL_VERSION = "image-grader-adapter-ui-v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_RANK_EXPRESSION = "avg(scores())"
MAX_RANK_EXPRESSION_LENGTH = 1000
MAX_RANK_EXPRESSION_NODES = 160


class AdapterError(ValueError):
    pass


def checkpoint_display_name(checkpoint: str) -> str:
    raw = str(checkpoint or "").strip()
    if not raw:
        return "unknown"
    try:
        model_id = resolve_checkpoint(raw).model_id
    except (OSError, ValueError):
        model_id = None
    if model_id:
        return model_id
    return Path(raw.replace("\\", "/")).name or raw


@dataclass(frozen=True)
class WorkDirLayout:
    root: Path
    templates_dir: Path
    runs_dir: Path
    reports_dir: Path
    exports_dir: Path
    previews_dir: Path
    ai_label_events_path: Path
    scores_dir: Path

    @classmethod
    def create(cls, work_dir: str | Path) -> "WorkDirLayout":
        root = Path(work_dir).expanduser().resolve()
        layout = cls(
            root=root,
            templates_dir=root / "templates",
            runs_dir=root / "runs",
            reports_dir=root / "reports",
            exports_dir=root / "exports",
            previews_dir=root / "previews",
            ai_label_events_path=root / "ai_label_events.jsonl",
            scores_dir=root / "image_grader_state",
        )
        for directory in (
            layout.root,
            layout.templates_dir,
            layout.runs_dir,
            layout.reports_dir,
            layout.exports_dir,
            layout.previews_dir,
            layout.scores_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not layout.ai_label_events_path.exists():
            layout.ai_label_events_path.touch()
        return layout


@dataclass(frozen=True)
class GeneratedImage:
    image_index: int
    image_name: str
    saved_path: Path
    workflow_name: str
    ckpt: str
    ckpt_family: str
    width: int | None
    height: int | None
    positive_prompt: str
    negative_prompt: str
    prompt_generator_name: str
    prompt_generator_args: dict[str, Any]
    raw: dict[str, Any]

    @property
    def aspect_ratio(self) -> str:
        return aspect_ratio_key(self.width, self.height)

    @property
    def orientation(self) -> str:
        return orientation_key(self.width, self.height)

    @property
    def prompt_template_key(self) -> str:
        return prompt_template_key(self.raw)

    def public_summary(self) -> dict[str, Any]:
        return {
            "image_index": self.image_index,
            "image_name": self.image_name,
            "saved_path": str(self.saved_path),
            "workflow_name": self.workflow_name,
            "ckpt": self.ckpt,
            "ckpt_alias": checkpoint_display_name(self.ckpt),
            "ckpt_family": self.ckpt_family,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "orientation": self.orientation,
            "prompt_template_key": self.prompt_template_key,
            "prompt_generator_name": self.prompt_generator_name,
        }


@dataclass(frozen=True)
class GeneratedSession:
    session_key: str
    dataset_id: str
    dataset_display_name: str
    sessions_jsonl: Path
    session_id: str
    session_index: int
    task_name: str
    task_yaml_name: str
    task_yaml_path: str
    task_yaml_sha256: str
    workflow_name: str
    primary_ckpt: str
    images: tuple[GeneratedImage, ...]
    raw: dict[str, Any]

    def public_summary(self) -> dict[str, Any]:
        prompt_keys = sorted({image.prompt_template_key for image in self.images})
        aspects = sorted({image.aspect_ratio for image in self.images})
        ckpts = sorted({image.ckpt for image in self.images})
        return {
            "session_key": self.session_key,
            "dataset_id": self.dataset_id,
            "dataset_display_name": self.dataset_display_name,
            "session_id": self.session_id,
            "session_index": self.session_index,
            "task_name": self.task_name,
            "task_yaml_name": self.task_yaml_name,
            "workflow_name": self.workflow_name,
            "primary_ckpt": self.primary_ckpt,
            "prompt_template_keys": prompt_keys,
            "aspect_ratios": aspects,
            "ckpts": ckpts,
            "checkpoints": [
                {"name": checkpoint, "alias": checkpoint_display_name(checkpoint)}
                for checkpoint in ckpts
            ],
            "image_count": len(self.images),
            "first_image_index": self.images[0].image_index if self.images else 0,
        }


@dataclass(frozen=True)
class ScoreRecord:
    run_id: str
    session_key: str
    dataset_id: str
    session_id: str
    image_index: int
    image_path: str
    task_name: str
    task_yaml_name: str
    workflow_name: str
    ckpt: str
    ckpt_family: str
    prompt_template_key: str
    aspect_ratio: str
    orientation: str
    eval_model: str
    preprocess_policy: str
    ok: bool
    score: float | None
    error: str | None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_key": self.session_key,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "image_index": self.image_index,
            "image_path": self.image_path,
            "task_name": self.task_name,
            "task_yaml_name": self.task_yaml_name,
            "workflow_name": self.workflow_name,
            "ckpt": self.ckpt,
            "ckpt_family": self.ckpt_family,
            "prompt_template_key": self.prompt_template_key,
            "aspect_ratio": self.aspect_ratio,
            "orientation": self.orientation,
            "eval_model": self.eval_model,
            "preprocess_policy": self.preprocess_policy,
            "ok": self.ok,
            "score": self.score,
            "error": self.error,
            "raw": self.raw,
        }


RunnerFactory = Callable[[GraderConfig, Path], BatchRunner]


class AdapterApp:
    def __init__(
        self,
        *,
        work_dir: str | Path,
        dataset_root: str | Path,
        grader_config: GraderConfig,
        repo_root: str | Path | None = None,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self.layout = WorkDirLayout.create(work_dir)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.exists() or not self.dataset_root.is_dir():
            raise AdapterError(f"dataset_root must be an existing directory: {self.dataset_root}")
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
        self.grader_config = grader_config
        self.runner_factory = runner_factory or _default_runner_factory
        self._lock = threading.Lock()
        self._sessions_cache: tuple[int, list[GeneratedSession]] | None = None

    def close(self) -> None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "app_version": APP_LABEL_VERSION,
            "work_dir": str(self.layout.root),
            "dataset_root": str(self.dataset_root),
            "templates_dir": str(self.layout.templates_dir),
            "reports_dir": str(self.layout.reports_dir),
            "ai_label_events_path": str(self.layout.ai_label_events_path),
            "enabled_models": list(self.grader_config.enabled_models),
            "models": {model_id: model.to_dict() for model_id, model in self.grader_config.models.items()},
            "preprocess_policies": list(PREPROCESS_POLICIES),
        }

    def list_sessions(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        cursor: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        sessions = self._filtered_sessions(filters or {})
        start = max(int(cursor), 0)
        page_size = max(1, min(int(limit), 500))
        end = min(start + page_size, len(sessions))
        return {
            "items": [session.public_summary() for session in sessions[start:end]],
            "total": len(sessions),
            "cursor": start,
            "next_cursor": end if end < len(sessions) else None,
            "limit": page_size,
        }

    def get_facets(self) -> dict[str, Any]:
        sessions = self._discover_sessions()
        checkpoint_aliases: dict[str, str] = {}
        facets: dict[str, set[str]] = {
            "dataset_ids": set(),
            "task_yaml_names": set(),
            "task_names": set(),
            "workflow_names": set(),
            "ckpts": set(),
            "ckpt_families": set(),
            "prompt_template_keys": set(),
            "aspect_ratios": set(),
        }
        for session in sessions:
            facets["dataset_ids"].add(session.dataset_id)
            facets["task_yaml_names"].add(session.task_yaml_name)
            facets["task_names"].add(session.task_name)
            for image in session.images:
                facets["workflow_names"].add(image.workflow_name or session.workflow_name)
                facets["ckpts"].add(image.ckpt)
                checkpoint_aliases[image.ckpt] = checkpoint_display_name(image.ckpt)
                facets["ckpt_families"].add(image.ckpt_family)
                facets["prompt_template_keys"].add(image.prompt_template_key)
                facets["aspect_ratios"].add(image.aspect_ratio)
        return {
            "total_sessions": len(sessions),
            "facets": {key: sorted(value) for key, value in facets.items()},
            "checkpoint_aliases": dict(sorted(checkpoint_aliases.items())),
        }

    def list_templates(self) -> dict[str, Any]:
        templates = []
        for path in sorted(self.layout.templates_dir.glob("*.yaml")):
            try:
                template = self.load_template(path.stem)
            except Exception as exc:
                templates.append({"name": path.stem, "filename": path.name, "error": str(exc)})
                continue
            templates.append({"name": template["name"], "filename": path.name, "template": template})
        return {"templates": templates}

    def load_template(self, name: str) -> dict[str, Any]:
        path = self._template_path(name)
        if not path.is_file():
            raise AdapterError(f"template not found: {name}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AdapterError(f"template must be a mapping: {path.name}")
        return validate_template(payload, self.grader_config)

    def save_template(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        template = validate_template(payload, self.grader_config)
        path = self._template_path(template["name"])
        path.write_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=False), encoding="utf-8")
        return {"template": template, "path": str(path)}

    def run_report(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        template = self._template_from_request(payload)
        sessions = self._selected_sessions_from_payload(payload)
        if not sessions:
            raise AdapterError("no sessions matched the selected filters")

        run_id = stable_hex("run", template["name"], utc_timestamp(), len(sessions))[:16]
        scores = self._score_sessions(run_id, template, sessions)
        report = build_score_report(
            run_id=run_id,
            template=template,
            sessions=sessions,
            scores=scores,
        )
        run_dir = self.layout.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(run_dir / "scores.jsonl", [record.to_dict() for record in scores])
        (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path = self.layout.reports_dir / f"{run_id}.json"
        shutil.copyfile(run_dir / "report.json", report_path)
        response = {
            "run_id": run_id,
            "report": report,
            "score_count": len(scores),
            "run_dir": str(run_dir),
            "report_path": str(report_path),
        }
        if bool(payload.get("include_scores")):
            response["scores"] = [record.to_dict() for record in scores]
        return response

    def run_playground(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request["include_scores"] = True
        if not request.get("session_keys"):
            request.setdefault("limit", 4)
        result = self.run_report(request)
        scores = [ScoreRecord(**row) for row in result.get("scores", [])]
        result["playground_ranking"] = build_playground_ranking(
            scores=scores,
            expression=payload.get("rank_expression"),
            percent=payload.get("rank_percent") or 10.0,
        )
        return result

    def dry_run_labels(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = self.run_report(payload)
        run_id = str(result["run_id"])
        template = result["report"]["template"]
        sessions = self._selected_sessions_from_payload(payload)
        scores = [ScoreRecord(**row) for row in load_jsonl(self.layout.runs_dir / run_id / "scores.jsonl")]
        human_latest = self._latest_external_labels()
        decisions = build_ai_label_events(
            template=template,
            sessions=sessions,
            scores=scores,
            existing_latest=human_latest,
            write=False,
        )
        return {**result, "labels": decisions}

    def write_ai_labels(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = self.run_report(payload)
        run_id = str(result["run_id"])
        template = result["report"]["template"]
        sessions = self._selected_sessions_from_payload(payload)
        scores = [ScoreRecord(**row) for row in load_jsonl(self.layout.runs_dir / run_id / "scores.jsonl")]
        human_latest = self._latest_external_labels()
        decisions = build_ai_label_events(
            template=template,
            sessions=sessions,
            scores=scores,
            existing_latest=human_latest,
            write=True,
        )
        events = [item["event"] for item in decisions["items"] if item.get("event")]
        if events:
            append_jsonl(self.layout.ai_label_events_path, events)
        aligned = self.export_aligned_labels()
        return {
            **result,
            "labels": decisions,
            "written_events": len(events),
            "ai_label_events_path": str(self.layout.ai_label_events_path),
            "aligned_export": aligned,
        }

    def export_aligned_labels(self) -> dict[str, Any]:
        external_events = self._external_label_events()
        ai_events = load_jsonl(self.layout.ai_label_events_path)
        by_event_id: dict[str, dict[str, Any]] = {}
        for event in [*external_events, *ai_events]:
            event_id = str(event.get("event_id", "")).strip()
            if event_id:
                by_event_id[event_id] = event
        events = sorted(by_event_id.values(), key=lambda item: str(item.get("created_at", "")))
        label_events_path = self.layout.exports_dir / "label_events_aligned.jsonl"
        labels_latest_path = self.layout.exports_dir / "labels_latest_aligned.jsonl"
        write_jsonl(label_events_path, events)
        write_jsonl(labels_latest_path, self._build_labels_latest(events))
        return {
            "label_events_path": str(label_events_path),
            "labels_latest_path": str(labels_latest_path),
            "event_count": len(events),
        }

    def get_media_path(self, session_key: str, image_index: int) -> tuple[Path, str]:
        session = self._session_by_key(session_key)
        for image in session.images:
            if int(image.image_index) == int(image_index):
                return image.saved_path, _mime_for_path(image.saved_path)
        raise AdapterError("image not found")

    def get_preprocessed_preview_path(self, session_key: str, image_index: int, policy: str) -> tuple[Path, str]:
        policy = normalize_preprocess_policy(policy)
        session = self._session_by_key(session_key)
        image = next((item for item in session.images if int(item.image_index) == int(image_index)), None)
        if image is None:
            raise AdapterError("image not found")
        source = image.saved_path
        digest = stable_hex(str(source), source.stat().st_mtime_ns, policy)[:24]
        preview_path = self.layout.previews_dir / f"{digest}.jpg"
        if preview_path.exists():
            return preview_path, "image/jpeg"
        if Image is None or ImageOps is None:
            raise AdapterError("Pillow is required for previews")
        with Image.open(source) as opened:
            prepared = ImageOps.exif_transpose(opened).convert("RGB")
            prepared = preprocess_pil_image(prepared, policy)
            prepared.thumbnail((960, 960))
            prepared.save(preview_path, format="JPEG", quality=86, optimize=True)
        return preview_path, "image/jpeg"

    def _template_from_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(payload.get("template"), Mapping):
            return validate_template(payload["template"], self.grader_config)
        template_name = str(payload.get("template_name", "")).strip()
        if not template_name:
            raise AdapterError("template_name or template object is required")
        return self.load_template(template_name)

    def _score_sessions(
        self,
        run_id: str,
        template: Mapping[str, Any],
        sessions: Sequence[GeneratedSession],
    ) -> list[ScoreRecord]:
        model_ids = tuple(template["models"])
        policies = tuple(template["preprocess_policies"])
        records: list[ScoreRecord] = []
        runner = self.runner_factory(self.grader_config, self.layout.scores_dir)
        try:
            for policy in policies:
                jobs: list[ImageJob] = []
                image_lookup: dict[str, tuple[GeneratedSession, GeneratedImage]] = {}
                for session in sessions:
                    for image in session.images:
                        request_id = f"{session.session_key}:{image.image_index}"
                        jobs.append(ImageJob(image_path=image.saved_path, request_id=request_id))
                        image_lookup[request_id] = (session, image)
                rows = runner.score_job_chunk(jobs, selected_models=model_ids, preprocess_policy=policy)
                for row in rows:
                    request_id = str(row.get("request_id", ""))
                    if request_id not in image_lookup:
                        continue
                    session, image = image_lookup[request_id]
                    scores = row.get("scores", {})
                    if not isinstance(scores, Mapping):
                        scores = {}
                    for model_id in model_ids:
                        score_payload = scores.get(model_id)
                        if not isinstance(score_payload, Mapping):
                            records.append(
                                _score_record_from_error(
                                    run_id,
                                    session,
                                    image,
                                    model_id,
                                    policy,
                                    "missing score payload",
                                )
                            )
                            continue
                        records.append(
                            ScoreRecord(
                                run_id=run_id,
                                session_key=session.session_key,
                                dataset_id=session.dataset_id,
                                session_id=session.session_id,
                                image_index=image.image_index,
                                image_path=str(image.saved_path),
                                task_name=session.task_name,
                                task_yaml_name=session.task_yaml_name,
                                workflow_name=image.workflow_name or session.workflow_name,
                                ckpt=image.ckpt,
                                ckpt_family=image.ckpt_family,
                                prompt_template_key=image.prompt_template_key,
                                aspect_ratio=image.aspect_ratio,
                                orientation=image.orientation,
                                eval_model=model_id,
                                preprocess_policy=policy,
                                ok=bool(score_payload.get("ok")),
                                score=_optional_float(score_payload.get("score")),
                                error=str(score_payload["error"]) if score_payload.get("error") is not None else None,
                                raw=dict(score_payload.get("raw", {})),
                            )
                        )
        finally:
            runner.close()
        return records

    def _filtered_sessions(self, filters: Mapping[str, Any]) -> list[GeneratedSession]:
        sessions = self._discover_sessions()
        return [session for session in sessions if session_matches_filters(session, filters)]

    def _selected_sessions_from_payload(self, payload: Mapping[str, Any]) -> list[GeneratedSession]:
        filters = payload.get("filters", {})
        if not isinstance(filters, Mapping):
            raise AdapterError("filters must be an object")
        session_keys = payload.get("session_keys")
        if session_keys is not None:
            if not isinstance(session_keys, list):
                raise AdapterError("session_keys must be a list")
            selected = {str(item) for item in session_keys if str(item)}
            if selected:
                filters = {"session_keys": sorted(selected)}
        limit = _optional_int(payload.get("limit"))
        sessions = self._filtered_sessions(filters)
        if limit is not None:
            sessions = sessions[:limit]
        return sessions

    def _discover_sessions(self) -> list[GeneratedSession]:
        signature = self._dataset_signature()
        with self._lock:
            if self._sessions_cache and self._sessions_cache[0] == signature:
                return list(self._sessions_cache[1])
        sessions: list[GeneratedSession] = []
        for sessions_jsonl in sorted(self.dataset_root.rglob("sessions.jsonl")):
            if self._is_inside_work_dir(sessions_jsonl):
                continue
            dataset_id = dataset_id_for_path(self.dataset_root, sessions_jsonl.parent)
            display_name = display_name_for_path(self.dataset_root, sessions_jsonl.parent)
            for row in load_jsonl(sessions_jsonl):
                try:
                    session = normalize_session(
                        dataset_root=self.dataset_root,
                        repo_root=self.repo_root,
                        sessions_jsonl=sessions_jsonl,
                        dataset_id=dataset_id,
                        dataset_display_name=display_name,
                        row=row,
                    )
                except Exception:
                    continue
                if session is not None:
                    sessions.append(session)
        sessions.sort(key=lambda item: (item.dataset_display_name, item.task_yaml_name, item.session_index, item.session_id))
        with self._lock:
            self._sessions_cache = (signature, sessions)
        return list(sessions)

    def _dataset_signature(self) -> int:
        value = 0
        for path in self.dataset_root.rglob("sessions.jsonl"):
            if self._is_inside_work_dir(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            value ^= hash((str(path), stat.st_mtime_ns, stat.st_size))
        return value

    def _is_inside_work_dir(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.layout.root)
            return True
        except ValueError:
            return False

    def _template_path(self, name: str) -> Path:
        safe = safe_name(name)
        if not safe:
            raise AdapterError("template name is required")
        return self.layout.templates_dir / f"{safe}.yaml"

    def _session_by_key(self, session_key: str) -> GeneratedSession:
        for session in self._discover_sessions():
            if session.session_key == session_key:
                return session
        raise AdapterError("session not found")

    def _external_label_event_paths(self) -> list[Path]:
        paths: list[Path] = []
        for path in self.dataset_root.rglob("label_events.jsonl"):
            resolved = path.resolve()
            if resolved == self.layout.ai_label_events_path:
                continue
            if self._is_inside_work_dir(resolved):
                continue
            paths.append(resolved)
        return sorted(set(paths))

    def _external_label_events(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self._external_label_event_paths():
            try:
                rows.extend(load_jsonl(path))
            except Exception:
                continue
        return rows

    def _latest_external_labels(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._external_label_events():
            key = pair_key_from_event(event)
            if not key:
                continue
            latest[key] = event
        return latest

    def _build_labels_latest(self, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        latest: dict[str, Mapping[str, Any]] = {}
        for event in events:
            key = pair_key_from_event(event)
            if key:
                latest[key] = event
        sessions_by_pair_key = {f"{session.dataset_id}::{session.session_id}": session for session in self._discover_sessions()}
        rows: list[dict[str, Any]] = []
        for pair_key, label in latest.items():
            session = sessions_by_pair_key.get(pair_key)
            if session is None:
                continue
            rows.append(
                {
                    "dataset_id": session.dataset_id,
                    "dataset_display_name": session.dataset_display_name,
                    "task_key": label.get("task_key", ""),
                    "session_id": session.session_id,
                    "session_index": session.session_index,
                    "task_name": session.task_name,
                    "task_yaml_name": session.task_yaml_name,
                    "workflow_name": session.workflow_name,
                    "primary_ckpt": session.primary_ckpt,
                    "label": dict(label),
                    "images": [image.raw for image in sorted(session.images, key=lambda item: item.image_index)],
                }
            )
        return sorted(rows, key=lambda item: (item["dataset_id"], item["session_index"], item["session_id"]))


def _default_runner_factory(config: GraderConfig, state_dir: Path) -> BatchRunner:
    return BatchRunner(config, state_dir=state_dir)


def normalize_session(
    *,
    dataset_root: Path,
    repo_root: Path,
    sessions_jsonl: Path,
    dataset_id: str,
    dataset_display_name: str,
    row: Mapping[str, Any],
) -> GeneratedSession | None:
    session_id = str(row.get("session_id", "")).strip()
    if not session_id:
        return None
    raw_images = row.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        return None
    images: list[GeneratedImage] = []
    for fallback_index, raw_image in enumerate(raw_images):
        if not isinstance(raw_image, Mapping):
            continue
        saved_path = resolve_image_path(raw_image.get("saved_path"), sessions_jsonl.parent, dataset_root, repo_root)
        if saved_path is None:
            continue
        images.append(
            GeneratedImage(
                image_index=int(raw_image.get("image_index", fallback_index)),
                image_name=str(raw_image.get("image_name", f"image_{fallback_index}")),
                saved_path=saved_path,
                workflow_name=str(raw_image.get("workflow_name", "")),
                ckpt=str(raw_image.get("ckpt", "")),
                ckpt_family=str(raw_image.get("ckpt_family", "")),
                width=_optional_int(raw_image.get("width") or raw_image.get("image_width")),
                height=_optional_int(raw_image.get("height") or raw_image.get("image_height")),
                positive_prompt=str(raw_image.get("positive_prompt", "")),
                negative_prompt=str(raw_image.get("negative_prompt", "")),
                prompt_generator_name=str(raw_image.get("prompt_generator_name", "")),
                prompt_generator_args=dict(raw_image.get("prompt_generator_args", {}))
                if isinstance(raw_image.get("prompt_generator_args"), Mapping)
                else {},
                raw=dict(raw_image),
            )
        )
    if not images:
        return None
    images.sort(key=lambda item: (item.image_index, item.image_name))
    task_yaml_path = str(row.get("task_yaml_path", "")).strip()
    task_yaml_name = Path(task_yaml_path or "unknown.yaml").name
    task_name = str(row.get("task_name", "") or Path(task_yaml_name).stem)
    session_key = stable_hex(str(sessions_jsonl), dataset_id, session_id)[:24]
    return GeneratedSession(
        session_key=session_key,
        dataset_id=dataset_id,
        dataset_display_name=dataset_display_name,
        sessions_jsonl=sessions_jsonl,
        session_id=session_id,
        session_index=int(row.get("session_index", 0)),
        task_name=task_name,
        task_yaml_name=task_yaml_name,
        task_yaml_path=task_yaml_path,
        task_yaml_sha256=str(row.get("task_yaml_sha256", "")),
        workflow_name=images[0].workflow_name,
        primary_ckpt=images[0].ckpt,
        images=tuple(images),
        raw=dict(row),
    )


def resolve_image_path(value: Any, *bases: Path) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [base / path for base in bases]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            return resolved
    return None


def prompt_template_key(image: Mapping[str, Any]) -> str:
    args = image.get("prompt_generator_args")
    if isinstance(args, Mapping):
        for key in ("template", "template_name", "template_id", "prompt_list"):
            value = args.get(key)
            if value not in (None, ""):
                return str(value)
    generator_name = str(image.get("prompt_generator_name", "")).strip()
    if generator_name:
        return generator_name
    return "unknown_prompt_template"


def aspect_ratio_key(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    divisor = math.gcd(int(width), int(height))
    return f"{int(width) // divisor}:{int(height) // divisor}"


def orientation_key(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    if int(width) == int(height):
        return "square"
    return "landscape" if int(width) > int(height) else "portrait"


def aspect_ratio_float(aspect: str) -> float | None:
    try:
        ratio = Fraction(aspect)
    except Exception:
        return None
    return float(ratio)


def session_matches_filters(session: GeneratedSession, filters: Mapping[str, Any]) -> bool:
    def selected(key: str) -> set[str]:
        value = filters.get(key)
        if value in (None, ""):
            return set()
        if isinstance(value, list):
            return {str(item) for item in value if str(item)}
        return {str(value)}

    if selected("session_keys") and session.session_key not in selected("session_keys"):
        return False
    if selected("dataset_ids") and session.dataset_id not in selected("dataset_ids"):
        return False
    if selected("task_yaml_names") and session.task_yaml_name not in selected("task_yaml_names"):
        return False
    if selected("task_names") and session.task_name not in selected("task_names"):
        return False
    if selected("workflow_names") and not {image.workflow_name for image in session.images}.intersection(selected("workflow_names")):
        return False
    if selected("ckpts") and not {image.ckpt for image in session.images}.intersection(selected("ckpts")):
        return False
    if selected("ckpt_families") and not {image.ckpt_family for image in session.images}.intersection(selected("ckpt_families")):
        return False
    if selected("prompt_template_keys") and not {image.prompt_template_key for image in session.images}.intersection(
        selected("prompt_template_keys")
    ):
        return False
    if selected("aspect_ratios") and not {image.aspect_ratio for image in session.images}.intersection(selected("aspect_ratios")):
        return False
    min_index = _optional_int(filters.get("min_session_index"))
    max_index = _optional_int(filters.get("max_session_index"))
    if min_index is not None and session.session_index < min_index:
        return False
    if max_index is not None and session.session_index > max_index:
        return False
    return True


def validate_template(payload: Mapping[str, Any], config: GraderConfig) -> dict[str, Any]:
    name = safe_name(str(payload.get("name", "")).strip())
    if not name:
        raise AdapterError("template.name is required")
    models = payload.get("models") or list(config.enabled_models)
    if not isinstance(models, list) or not models:
        raise AdapterError("template.models must be a non-empty list")
    model_ids = [str(item) for item in models]
    config.selected_model_ids(model_ids)
    policies = payload.get("preprocess_policies") or ["native"]
    if not isinstance(policies, list) or not policies:
        raise AdapterError("template.preprocess_policies must be a non-empty list")
    normalized_policies = [normalize_preprocess_policy(str(item)) for item in policies]
    formula = validate_formula(payload.get("score_formula"), model_ids, normalized_policies)
    report = dict(payload.get("report", {})) if isinstance(payload.get("report", {}), Mapping) else {}
    decision = validate_decision(payload.get("decision"))
    return {
        "name": name,
        "models": model_ids,
        "preprocess_policies": normalized_policies,
        "score_formula": formula,
        "report": {
            "bad_absolute_threshold": float(report.get("bad_absolute_threshold", 5.0)),
            "bad_relative_delta": float(report.get("bad_relative_delta", 1.0)),
            "error_rate_threshold": float(report.get("error_rate_threshold", 0.2)),
            "below_score_threshold": float(report.get("below_score_threshold", 5.0)),
        },
        "decision": decision,
    }


def validate_formula(value: Any, model_ids: Sequence[str], policies: Sequence[str]) -> dict[str, Any]:
    if value is None:
        model_id = model_ids[0]
        policy = policies[0]
        return {
            "type": "weighted_sum",
            "missing": "fail",
            "terms": [{"model": model_id, "preprocess_policy": policy, "weight": 1.0}],
        }
    if not isinstance(value, Mapping):
        raise AdapterError("score_formula must be an object")
    formula_type = str(value.get("type", "weighted_sum"))
    if formula_type != "weighted_sum":
        raise AdapterError("score_formula.type must be weighted_sum")
    missing = str(value.get("missing", "fail"))
    if missing not in {"fail", "skip"}:
        raise AdapterError("score_formula.missing must be fail or skip")
    terms = value.get("terms")
    if not isinstance(terms, list) or not terms:
        raise AdapterError("score_formula.terms must be a non-empty list")
    normalized_terms = []
    for term in terms:
        if not isinstance(term, Mapping):
            raise AdapterError("score_formula terms must be objects")
        model_id = str(term.get("model", ""))
        if model_id not in model_ids:
            raise AdapterError(f"formula references model not selected: {model_id}")
        policy = normalize_preprocess_policy(str(term.get("preprocess_policy", policies[0])))
        if policy not in policies:
            raise AdapterError(f"formula references preprocess policy not selected: {policy}")
        normalized_terms.append({"model": model_id, "preprocess_policy": policy, "weight": float(term.get("weight", 1.0))})
    return {"type": "weighted_sum", "missing": missing, "terms": normalized_terms}


def validate_decision(value: Any) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, Mapping) else {}
    return {
        "winner_min_score": float(payload.get("winner_min_score", 6.0)),
        "winner_min_delta": float(payload.get("winner_min_delta", 0.75)),
        "tie_delta": float(payload.get("tie_delta", 0.25)),
        "both_good_min": float(payload.get("both_good_min", 7.5)),
        "both_bad_max": float(payload.get("both_bad_max", 3.5)),
        "on_error": str(payload.get("on_error", "skip")),
        "on_ambiguous": str(payload.get("on_ambiguous", "skip")),
    }


def normalize_preprocess_policy(value: str) -> str:
    normalized = str(value or "native").strip()
    if normalized not in PREPROCESS_POLICIES:
        raise AdapterError(f"unknown preprocess policy: {normalized}")
    return normalized


def build_score_report(
    *,
    run_id: str,
    template: Mapping[str, Any],
    sessions: Sequence[GeneratedSession],
    scores: Sequence[ScoreRecord],
) -> dict[str, Any]:
    report_cfg = template["report"]
    tables = {
        "prompt_template_model_aspect": aggregate_scores(
            scores,
            fields=("prompt_template_key", "ckpt", "aspect_ratio", "eval_model", "preprocess_policy"),
            below_threshold=float(report_cfg["below_score_threshold"]),
        ),
        "workflow_model_aspect": aggregate_scores(
            scores,
            fields=("workflow_name", "ckpt", "aspect_ratio", "eval_model", "preprocess_policy"),
            below_threshold=float(report_cfg["below_score_threshold"]),
        ),
        "checkpoint_family_prompt_template": aggregate_scores(
            scores,
            fields=("ckpt_family", "prompt_template_key", "eval_model", "preprocess_policy"),
            below_threshold=float(report_cfg["below_score_threshold"]),
        ),
    }
    flags = flag_bad_fit(
        scores,
        absolute_threshold=float(report_cfg["bad_absolute_threshold"]),
        relative_delta=float(report_cfg["bad_relative_delta"]),
        error_rate_threshold=float(report_cfg["error_rate_threshold"]),
    )
    return {
        "run_id": run_id,
        "created_at": utc_timestamp(),
        "template": dict(template),
        "summary": {
            "session_count": len(sessions),
            "image_count": sum(len(session.images) for session in sessions),
            "score_count": len(scores),
            "ok_score_count": sum(1 for record in scores if record.ok and record.score is not None),
            "error_score_count": sum(1 for record in scores if not record.ok or record.score is None),
        },
        "tables": tables,
        "bad_fit_flags": flags,
    }


def aggregate_scores(
    scores: Sequence[ScoreRecord],
    *,
    fields: Sequence[str],
    below_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[ScoreRecord]] = defaultdict(list)
    for record in scores:
        grouped[tuple(getattr(record, field) for field in fields)].append(record)
    rows: list[dict[str, Any]] = []
    for key, records in grouped.items():
        values = [float(record.score) for record in records if record.ok and record.score is not None]
        errors = [record for record in records if not record.ok or record.score is None]
        row = {field: value for field, value in zip(fields, key, strict=True)}
        row.update(score_stats(values, total_count=len(records), error_count=len(errors), below_threshold=below_threshold))
        rows.append(row)
    rows.sort(key=lambda item: tuple(str(item.get(field, "")) for field in fields))
    return rows


def score_stats(values: Sequence[float], *, total_count: int, error_count: int, below_threshold: float) -> dict[str, Any]:
    sorted_values = sorted(values)
    ok_count = len(sorted_values)
    payload: dict[str, Any] = {
        "count": total_count,
        "ok_count": ok_count,
        "error_count": error_count,
        "error_rate": error_count / total_count if total_count else 0.0,
        "below_threshold_count": sum(1 for value in sorted_values if value < below_threshold),
        "below_threshold_rate": sum(1 for value in sorted_values if value < below_threshold) / ok_count if ok_count else 0.0,
    }
    if not sorted_values:
        payload.update({"mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None, "stdev": None})
        return payload
    payload.update(
        {
            "mean": statistics.fmean(sorted_values),
            "median": statistics.median(sorted_values),
            "p10": percentile(sorted_values, 10),
            "p90": percentile(sorted_values, 90),
            "min": min(sorted_values),
            "max": max(sorted_values),
            "stdev": statistics.pstdev(sorted_values) if len(sorted_values) > 1 else 0.0,
        }
    )
    return payload


def percentile(sorted_values: Sequence[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def flag_bad_fit(
    scores: Sequence[ScoreRecord],
    *,
    absolute_threshold: float,
    relative_delta: float,
    error_rate_threshold: float,
) -> list[dict[str, Any]]:
    rows = aggregate_scores(
        scores,
        fields=("prompt_template_key", "aspect_ratio", "ckpt", "eval_model", "preprocess_policy"),
        below_threshold=absolute_threshold,
    )
    median_by_context: dict[tuple[str, str, str, str], float] = {}
    by_context: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        context = (
            str(row["prompt_template_key"]),
            str(row["aspect_ratio"]),
            str(row["eval_model"]),
            str(row["preprocess_policy"]),
        )
        by_context[context].append(row)
    for context, context_rows in by_context.items():
        means = [float(row["mean"]) for row in context_rows if row.get("mean") is not None]
        if means:
            median_by_context[context] = statistics.median(means)

    flags: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        mean = row.get("mean")
        context = (
            str(row["prompt_template_key"]),
            str(row["aspect_ratio"]),
            str(row["eval_model"]),
            str(row["preprocess_policy"]),
        )
        context_median = median_by_context.get(context)
        if row["error_rate"] >= error_rate_threshold:
            reasons.append("high_eval_error_rate")
        if mean is not None and float(mean) < absolute_threshold:
            reasons.append("below_absolute_threshold")
        if mean is not None and context_median is not None and float(mean) < context_median - relative_delta:
            reasons.append("below_template_aspect_peer_median")
        if reasons:
            flags.append({**row, "context_median": context_median, "reasons": reasons})
    flags.sort(key=lambda item: (len(item["reasons"]), item.get("mean") if item.get("mean") is not None else -1), reverse=True)
    return flags


def build_ai_label_events(
    *,
    template: Mapping[str, Any],
    sessions: Sequence[GeneratedSession],
    scores: Sequence[ScoreRecord],
    existing_latest: Mapping[str, Mapping[str, Any]],
    write: bool,
) -> dict[str, Any]:
    reviewer = f"AI_{safe_name(str(template['name']))}"
    score_index: dict[tuple[str, int, str, str], ScoreRecord] = {}
    for record in scores:
        score_index[(record.session_key, record.image_index, record.eval_model, record.preprocess_policy)] = record
    items: list[dict[str, Any]] = []
    for session in sessions:
        if len(session.images) != 2:
            continue
        pair_key = f"{session.dataset_id}::{session.session_id}"
        latest = existing_latest.get(pair_key)
        latest_reviewer = str(latest.get("reviewer_username", "")) if latest else ""
        if latest and not latest_reviewer.startswith("AI_"):
            items.append({"session_key": session.session_key, "session_id": session.session_id, "skipped": True, "reason": "human_latest_label"})
            continue
        decision_payload = decide_pair(template, session, score_index)
        event = build_label_event(template, session, reviewer, decision_payload) if write else None
        items.append(
            {
                "session_key": session.session_key,
                "session_id": session.session_id,
                "skipped": False,
                "decision": decision_payload,
                "event": event,
            }
        )
    return {
        "reviewer_username": reviewer,
        "write": write,
        "items": items,
        "event_count": sum(1 for item in items if item.get("event")),
        "skipped_count": sum(1 for item in items if item.get("skipped")),
    }


def decide_pair(
    template: Mapping[str, Any],
    session: GeneratedSession,
    score_index: Mapping[tuple[str, int, str, str], ScoreRecord],
) -> dict[str, Any]:
    images = sorted(session.images, key=lambda item: item.image_index)
    scored: list[tuple[GeneratedImage, float | None, list[str]]] = []
    for image in images:
        score, errors = formula_score(template["score_formula"], session.session_key, image.image_index, score_index)
        scored.append((image, score, errors))
    all_errors = [error for _, _, errors in scored for error in errors]
    decision_cfg = template["decision"]
    if all_errors:
        return {
            "decision": decision_cfg["on_error"],
            "chosen_image_indices": [],
            "scores": {str(image.image_index): score for image, score, _ in scored},
            "errors": all_errors,
        }
    first_image, first_score, _ = scored[0]
    second_image, second_score, _ = scored[1]
    assert first_score is not None and second_score is not None
    high = max(first_score, second_score)
    low = min(first_score, second_score)
    delta = abs(first_score - second_score)
    if high < float(decision_cfg["both_bad_max"]):
        decision = "both_bad"
        chosen: list[int] = []
    elif delta <= float(decision_cfg["tie_delta"]) and low >= float(decision_cfg["both_good_min"]):
        decision = "both_good"
        chosen = [first_image.image_index, second_image.image_index]
    elif delta >= float(decision_cfg["winner_min_delta"]) and high >= float(decision_cfg["winner_min_score"]):
        if first_score > second_score:
            decision = "a_good"
            chosen = [first_image.image_index]
        else:
            decision = "b_good"
            chosen = [second_image.image_index]
    else:
        decision = decision_cfg["on_ambiguous"]
        chosen = []
    return {
        "decision": decision,
        "chosen_image_indices": chosen,
        "scores": {str(image.image_index): score for image, score, _ in scored},
        "errors": [],
    }


def formula_score(
    formula: Mapping[str, Any],
    session_key: str,
    image_index: int,
    score_index: Mapping[tuple[str, int, str, str], ScoreRecord],
) -> tuple[float | None, list[str]]:
    weighted_sum = 0.0
    weight_sum = 0.0
    errors: list[str] = []
    for term in formula["terms"]:
        key = (session_key, image_index, str(term["model"]), str(term["preprocess_policy"]))
        record = score_index.get(key)
        if record is None:
            errors.append(f"missing score for {term['model']}:{term['preprocess_policy']}")
            continue
        if not record.ok or record.score is None:
            errors.append(record.error or f"score failed for {term['model']}:{term['preprocess_policy']}")
            continue
        weight = float(term["weight"])
        weighted_sum += float(record.score) * weight
        weight_sum += weight
    if errors and formula.get("missing") == "fail":
        return None, errors
    if weight_sum == 0.0:
        return None, errors or ["formula has zero usable weight"]
    return weighted_sum / weight_sum, errors


def build_playground_ranking(
    *,
    scores: Sequence[ScoreRecord],
    expression: Any,
    percent: Any,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[ScoreRecord]] = defaultdict(list)
    for record in scores:
        grouped[(record.session_key, int(record.image_index))].append(record)
    expression = normalize_rank_expression(expression or default_rank_expression_for_scores(scores))
    percent = normalize_rank_percent(percent)

    ranked: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for key in sorted(grouped):
        records = grouped[key]
        first = records[0]
        value, errors = rank_expression_score(expression, records)
        base = {
            "key": f"{first.session_key}:{first.image_index}",
            "session_key": first.session_key,
            "session_id": first.session_id,
            "image_index": first.image_index,
            "task_name": first.task_name,
            "prompt_template_key": first.prompt_template_key,
            "ckpt": first.ckpt,
            "aspect_ratio": first.aspect_ratio,
            "errors": errors,
        }
        if value is None:
            invalid.append(base)
            continue
        ranked.append({**base, "score": value})

    ranked.sort(key=lambda item: (-float(item["score"]), item["key"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    bottom_ranked = sorted(ranked, key=lambda item: (float(item["score"]), item["key"]))
    for reverse_rank, item in enumerate(bottom_ranked, start=1):
        item["reverse_rank"] = reverse_rank

    bucket_size = math.ceil(len(ranked) * percent / 100.0) if ranked else 0
    bucket_size = max(1, bucket_size) if ranked else 0
    return {
        "expression": expression,
        "percent": percent,
        "total_images": len(grouped),
        "ranked_count": len(ranked),
        "invalid_count": len(invalid),
        "bucket_size": bucket_size,
        "top": [_rank_bucket_item(item, "top") for item in ranked[:bucket_size]],
        "bottom": [_rank_bucket_item(item, "bottom") for item in bottom_ranked[:bucket_size]],
        "errors": invalid[:50],
    }


def normalize_rank_expression(expression: str) -> str:
    normalized = str(expression or DEFAULT_RANK_EXPRESSION).replace(";", ",").strip()
    if not normalized:
        normalized = DEFAULT_RANK_EXPRESSION
    if len(normalized) > MAX_RANK_EXPRESSION_LENGTH:
        raise AdapterError(f"rank_expression is limited to {MAX_RANK_EXPRESSION_LENGTH} characters")
    return normalized


def default_rank_expression_for_scores(scores: Sequence[ScoreRecord]) -> str:
    policies = sorted({record.preprocess_policy for record in scores})
    model_ids = sorted({record.eval_model for record in scores})
    terms = [score_lookup_expression(policy, model_id) for policy in policies for model_id in model_ids]
    if not terms:
        return DEFAULT_RANK_EXPRESSION
    return f"({' + '.join(terms)}) / {len(terms)}"


def score_lookup_expression(policy: str, model_id: str) -> str:
    return f"score[{json.dumps(policy)}][{json.dumps(model_id)}]"


def normalize_rank_percent(value: float) -> float:
    try:
        percent = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError("rank_percent must be a number") from exc
    if not math.isfinite(percent) or percent <= 0.0 or percent > 50.0:
        raise AdapterError("rank_percent must be greater than 0 and no more than 50")
    return percent


def rank_expression_score(expression: str, records: Sequence[ScoreRecord]) -> tuple[float | None, list[str]]:
    try:
        expression = normalize_rank_expression(expression)
        value = RankExpressionEvaluator(records).evaluate(expression)
        number = coerce_rank_number(value)
    except AdapterError as exc:
        return None, [str(exc)]
    if not math.isfinite(number):
        return None, ["rank expression returned a non-finite value"]
    return number, []


def coerce_rank_number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise AdapterError("rank expression did not return a numeric value")
    if isinstance(value, (int, float)):
        return float(value)
    raise AdapterError("rank expression did not return a numeric value")


def _rank_bucket_item(item: Mapping[str, Any], bucket: str) -> dict[str, Any]:
    copied = dict(item)
    copied["bucket"] = bucket
    return copied


class RankExpressionEvaluator:
    def __init__(self, records: Sequence[ScoreRecord]) -> None:
        self.records = list(records)
        self.model_ids = sorted({record.eval_model for record in self.records})
        self.policies = sorted({record.preprocess_policy for record in self.records})
        self.identifier_tokens: dict[str, str] = {}
        for token in [*self.model_ids, *self.policies]:
            self.identifier_tokens.setdefault(rank_identifier(token), token)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                self.identifier_tokens.setdefault(token, token)

    def evaluate(self, expression: str) -> Any:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise AdapterError(f"invalid rank expression: {exc.msg}") from exc
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > MAX_RANK_EXPRESSION_NODES:
            raise AdapterError("rank expression is too complex")
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str)) or node.value is None:
                return node.value
            raise AdapterError("rank expression constants must be numbers or strings")
        if isinstance(node, ast.Name):
            return self._eval_name(node.id)
        if isinstance(node, ast.Subscript):
            return self._eval_subscript(node)
        if isinstance(node, ast.List):
            return [self._eval(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(item) for item in node.elts)
        if isinstance(node, ast.UnaryOp):
            value = coerce_rank_number(self._eval(node.operand))
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise AdapterError("unsupported unary operator in rank expression")
        if isinstance(node, ast.BinOp):
            left = coerce_rank_number(self._eval(node.left))
            right = coerce_rank_number(self._eval(node.right))
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0.0:
                    raise AdapterError("rank expression divided by zero")
                return left / right
            raise AdapterError("rank expression supports +, -, *, and /")
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise AdapterError("unsupported syntax in rank expression")

    def _eval_name(self, name: str) -> Any:
        if name in {"None", "none", "null"}:
            return None
        if name == "all_scores":
            return self._score_values()
        if name == "score":
            return self._score_table()
        token = self.identifier_tokens.get(name)
        if token is None:
            raise AdapterError(f"unknown rank expression name: {name}")
        if token in self.policies:
            return self._mean_or_none(self._score_values(policy=token))
        return self._mean_or_none(self._score_values(model=token))

    def _eval_call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise AdapterError("rank expression only supports direct function calls")
        if node.keywords:
            raise AdapterError("rank expression does not support keyword arguments")
        name = node.func.id
        if name in {"score", "scores", "model", "method", "policy"}:
            return self._eval_lookup_call(name, node.args)
        args = [self._eval(arg) for arg in node.args]
        if name in {"avg", "mean"}:
            return self._mean_or_none(flatten_rank_numbers(args))
        if name == "sum":
            return math.fsum(flatten_rank_numbers(args))
        if name == "min":
            values = flatten_rank_numbers(args)
            return min(values) if values else None
        if name == "max":
            values = flatten_rank_numbers(args)
            return max(values) if values else None
        if name in {"percentile", "quantile"}:
            if len(args) < 2:
                raise AdapterError(f"{name} expects one or more values plus a percentile from 0 to 100")
            return rank_percentile(flatten_rank_numbers(args[:-1]), coerce_rank_number(args[-1]))
        if name in {"p10", "p25", "p50", "p75", "p90"}:
            if not args:
                raise AdapterError(f"{name} expects one or more values")
            return rank_percentile(flatten_rank_numbers(args), float(name[1:]))
        if name == "lower_quartile":
            if not args:
                raise AdapterError("lower_quartile expects one or more values")
            return rank_percentile(flatten_rank_numbers(args), 25.0)
        if name == "upper_quartile":
            if not args:
                raise AdapterError("upper_quartile expects one or more values")
            return rank_percentile(flatten_rank_numbers(args), 75.0)
        if name == "count":
            return float(len(flatten_rank_numbers(args)))
        if name == "abs":
            if len(args) != 1:
                raise AdapterError("abs expects one value")
            return abs(coerce_rank_number(args[0]))
        if name == "round":
            if len(args) not in {1, 2}:
                raise AdapterError("round expects one value and optional digits")
            digits = int(coerce_rank_number(args[1])) if len(args) == 2 else None
            return round(coerce_rank_number(args[0]), digits) if digits is not None else round(coerce_rank_number(args[0]))
        raise AdapterError(f"unknown rank expression function: {name}")

    def _eval_subscript(self, node: ast.Subscript) -> Any:
        container = self._eval(node.value)
        key = self._eval_subscript_key(node.slice)
        if not isinstance(container, Mapping):
            raise AdapterError("rank expression can only index score mappings")
        try:
            return container[key]
        except KeyError as exc:
            raise AdapterError(f"rank expression score key not found: {key}") from exc

    def _eval_subscript_key(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return str(node.value)
        if isinstance(node, ast.Name):
            token = self.identifier_tokens.get(node.id)
            if token is not None:
                return token
        raise AdapterError("score indexes must be selected names or string keys")

    def _eval_lookup_call(self, name: str, args: Sequence[ast.AST]) -> Any:
        if name == "scores":
            if len(args) > 2:
                raise AdapterError("scores expects zero, one, or two lookup values")
            model, policy = self._lookup_args(args)
            return self._score_values(model=model, policy=policy)
        if name == "score":
            if len(args) not in {1, 2}:
                raise AdapterError("score expects one lookup value or model plus policy")
            model, policy = self._lookup_args(args)
            if model is not None and policy is not None:
                values = self._score_values(model=model, policy=policy)
                return values[0] if values else None
            return self._mean_or_none(self._score_values(model=model, policy=policy))
        if name == "model":
            if len(args) != 1:
                raise AdapterError("model expects one model id")
            model, policy = self._lookup_args(args)
            if model is None or policy is not None:
                raise AdapterError("model expects a selected eval model id")
            return self._mean_or_none(self._score_values(model=model))
        if name in {"method", "policy"}:
            if len(args) != 1:
                raise AdapterError(f"{name} expects one preprocess policy")
            model, policy = self._lookup_args(args)
            if policy is None or model is not None:
                raise AdapterError(f"{name} expects a selected preprocess policy")
            return self._mean_or_none(self._score_values(policy=policy))
        raise AdapterError(f"unknown rank expression function: {name}")

    def _score_table(self) -> dict[str, dict[str, float | None]]:
        table: dict[str, dict[str, float | None]] = {}
        for policy in self.policies:
            table[policy] = {
                model_id: self._single_score_or_none(model=model_id, policy=policy)
                for model_id in self.model_ids
            }
        return table

    def _single_score_or_none(self, *, model: str, policy: str) -> float | None:
        values = self._score_values(model=model, policy=policy)
        return values[0] if values else None

    def _lookup_args(self, args: Sequence[ast.AST]) -> tuple[str | None, str | None]:
        model: str | None = None
        policy: str | None = None
        for arg in args:
            token = self._lookup_token(arg)
            if token in self.model_ids:
                if model is not None:
                    raise AdapterError("rank expression includes more than one model lookup")
                model = token
                continue
            if token in self.policies:
                if policy is not None:
                    raise AdapterError("rank expression includes more than one preprocess lookup")
                policy = token
                continue
            raise AdapterError(f"rank expression lookup is not selected: {token}")
        return model, policy

    def _lookup_token(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return str(node.value)
        if isinstance(node, ast.Name):
            token = self.identifier_tokens.get(node.id)
            if token is not None:
                return token
        raise AdapterError("score/model/method lookups must use a selected name or string")

    def _score_values(self, *, model: str | None = None, policy: str | None = None) -> list[float]:
        values: list[float] = []
        for record in self.records:
            if model is not None and record.eval_model != model:
                continue
            if policy is not None and record.preprocess_policy != policy:
                continue
            if record.ok and record.score is not None:
                score = float(record.score)
                if math.isfinite(score):
                    values.append(score)
        return values

    @staticmethod
    def _mean_or_none(values: Sequence[float]) -> float | None:
        if not values:
            return None
        return math.fsum(values) / len(values)


def flatten_rank_numbers(values: Iterable[Any]) -> list[float]:
    flattened: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            flattened.extend(flatten_rank_numbers(value))
            continue
        flattened.append(coerce_rank_number(value))
    return flattened


def rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not math.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
        raise AdapterError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def rank_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not identifier:
        identifier = "value"
    if not re.match(r"[A-Za-z_]", identifier):
        identifier = f"_{identifier}"
    return identifier


def build_label_event(
    template: Mapping[str, Any],
    session: GeneratedSession,
    reviewer: str,
    decision_payload: Mapping[str, Any],
) -> dict[str, Any]:
    images = sorted(session.images, key=lambda item: item.image_index)
    display_order = [int(image.image_index) for image in images[:2]]
    now = utc_timestamp()
    event_id = stable_hex(
        "ai-label",
        template["name"],
        session.dataset_id,
        session.session_id,
        decision_payload.get("decision"),
        json.dumps(decision_payload.get("scores", {}), sort_keys=True),
    )
    return {
        "event_id": event_id,
        "created_at": now,
        "client_ts": now,
        "app_version": APP_LABEL_VERSION,
        "dataset_id": session.dataset_id,
        "session_id": session.session_id,
        "task_key": f"{session.dataset_id}::{session.task_yaml_sha256 or stable_hex(session.task_yaml_path, session.task_name)[:16]}",
        "task_name": session.task_name,
        "task_yaml_name": session.task_yaml_name,
        "workflow_name": session.workflow_name,
        "primary_ckpt": session.primary_ckpt,
        "reviewer_username": reviewer,
        "client_instance_id": "image_grader_adapter_ui",
        "review_id": f"AI_{template['name']}",
        "decision": str(decision_payload.get("decision", "skip")),
        "defects_a": [],
        "defects_b": [],
        "display_order": display_order,
        "chosen_image_indices": [int(item) for item in decision_payload.get("chosen_image_indices", [])],
        "defects_by_image_index": {str(index): [] for index in display_order},
        "note": json.dumps({"scores": decision_payload.get("scores", {}), "errors": decision_payload.get("errors", [])}, sort_keys=True),
    }


def _score_record_from_error(
    run_id: str,
    session: GeneratedSession,
    image: GeneratedImage,
    model_id: str,
    policy: str,
    error: str,
) -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        session_key=session.session_key,
        dataset_id=session.dataset_id,
        session_id=session.session_id,
        image_index=image.image_index,
        image_path=str(image.saved_path),
        task_name=session.task_name,
        task_yaml_name=session.task_yaml_name,
        workflow_name=image.workflow_name or session.workflow_name,
        ckpt=image.ckpt,
        ckpt_family=image.ckpt_family,
        prompt_template_key=image.prompt_template_key,
        aspect_ratio=image.aspect_ratio,
        orientation=image.orientation,
        eval_model=model_id,
        preprocess_policy=policy,
        ok=False,
        score=None,
        error=error,
        raw={},
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise AdapterError(f"expected object JSONL line {line_number} of {path}")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        os.fsync(handle.fileno())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")


def stable_hex(*parts: Any) -> str:
    text = "||".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def dataset_id_for_path(dataset_root: Path, dataset_dir: Path) -> str:
    try:
        relative = dataset_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        relative = Path(dataset_dir.name)
    if relative == Path("."):
        return "root"
    parts = [safe_name(part).lower() for part in relative.parts if safe_name(part)]
    return "__".join(parts) if parts else "root"


def display_name_for_path(dataset_root: Path, dataset_dir: Path) -> str:
    try:
        relative = dataset_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return dataset_dir.name
    text = relative.as_posix()
    return text if text and text != "." else dataset_root.name


def pair_key_from_event(event: Mapping[str, Any]) -> str:
    dataset_id = str(event.get("dataset_id", "")).strip()
    session_id = str(event.get("session_id", "")).strip()
    return f"{dataset_id}::{session_id}" if dataset_id and session_id else ""


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"
