from __future__ import annotations

import json
import socket
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import __version__
from .config import GraderConfig
from .io import ImageJob
from .runtime import validate_runtime
from .runner import BatchRunner


class GraderServerError(ValueError):
    pass


class ImageGraderRequestHandler(BaseHTTPRequestHandler):
    app: "ImageGraderServerApp"
    request_timeout_seconds = 60.0

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.request_timeout_seconds)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.health(), "error": None})
                return
            if parsed.path == "/v1/models":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.models(), "error": None})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "data": None, "error": "not found"})
        except Exception as exc:  # pragma: no cover
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "data": None, "error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/v1/score":
                payload = self._read_json_body()
                data = self.app.score(payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "data": None, "error": "not found"})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "data": None, "error": "invalid JSON body"})
        except GraderServerError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "data": None, "error": str(exc)})
        except socket.timeout:
            self._send_json(HTTPStatus.REQUEST_TIMEOUT, {"ok": False, "data": None, "error": "request timed out"})
        except Exception as exc:  # pragma: no cover
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "data": None, "error": str(exc)})

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_headers("application/json; charset=utf-8", 0)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise GraderServerError("JSON body must be an object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(
            {**payload, "request_id": payload.get("request_id") or f"req-{uuid.uuid4().hex}"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        self.send_response(status)
        self._send_headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_headers(self, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")


class ImageGraderServerApp:
    def __init__(self, config: GraderConfig, runner: BatchRunner) -> None:
        self.config = config
        self.runner = runner
        self.allowed_roots = tuple(Path(root).expanduser().resolve() for root in config.server.allowed_roots)

    def health(self) -> dict[str, Any]:
        return {"ok": True, "version": __version__, "enabled_models": list(self.config.enabled_models)}

    def models(self) -> dict[str, Any]:
        return {
            "enabled_models": list(self.config.enabled_models),
            "models": {model_id: model.to_dict() for model_id, model in self.config.models.items()},
        }

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise GraderServerError("items must be a non-empty list")
        if len(items) > self.config.server.max_items_per_request:
            raise GraderServerError(f"items exceeds max_items_per_request={self.config.server.max_items_per_request}")
        raw_models = payload.get("models")
        if raw_models is not None and not isinstance(raw_models, list):
            raise GraderServerError("models must be a list of model ids")
        model_ids = self.config.selected_model_ids(raw_models)
        preprocess_policy = str(payload.get("preprocess_policy", "native") or "native")
        jobs = [self._job_from_item(item, index) for index, item in enumerate(items, start=1)]
        try:
            rows = self.runner.score_job_chunk(
                jobs,
                selected_models=model_ids,
                preprocess_policy=preprocess_policy,
            )
        except ValueError as exc:
            raise GraderServerError(str(exc)) from exc
        return {"results": rows}

    def _job_from_item(self, item: Any, index: int) -> ImageJob:
        if isinstance(item, str):
            path = Path(item).expanduser().resolve()
            request_id = None
            metadata = {}
        elif isinstance(item, dict):
            raw_path = item.get("image_path", item.get("path", item.get("file")))
            if raw_path is None:
                raise GraderServerError(f"items[{index}] is missing image_path")
            path = Path(str(raw_path)).expanduser().resolve()
            request_id = str(item["request_id"]) if item.get("request_id") not in (None, "") else None
            metadata = item.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                raise GraderServerError(f"items[{index}].metadata must be an object")
        else:
            raise GraderServerError(f"items[{index}] must be a path string or object")
        self._validate_path(path)
        return ImageJob(image_path=path, request_id=request_id, metadata=dict(metadata))

    def _validate_path(self, path: Path) -> None:
        if not self.allowed_roots:
            return
        for root in self.allowed_roots:
            try:
                path.relative_to(root)
                return
            except ValueError:
                continue
        raise GraderServerError(f"path is outside allowed roots: {path}")


def serve(config: GraderConfig, *, state_dir: str | Path, host: str, port: int) -> None:
    validate_runtime(config)
    runner = BatchRunner(config, state_dir=state_dir)
    app = ImageGraderServerApp(config, runner)
    handler = type("ConfiguredImageGraderRequestHandler", (ImageGraderRequestHandler,), {"app": app})
    server = ThreadingHTTPServer((host, int(port)), handler)
    print(f"Image grader {__version__} listening on http://{host}:{port}/")
    try:
        server.serve_forever()
    finally:
        runner.close()
