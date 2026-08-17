from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from image_grader.config import ConfigError, load_config
from image_grader.runtime import RuntimeValidationError, validate_runtime

from .app import AdapterApp, AdapterError


def detect_tailscale_host() -> str:
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "127.0.0.1"
    if result.returncode != 0:
        return "127.0.0.1"
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return "127.0.0.1"


def resolve_bind_host(value: str | None) -> str:
    normalized = str(value or "tailscale-auto").strip()
    if normalized in {"tailscale-auto", "auto", ""}:
        return detect_tailscale_host()
    return normalized


def _default_grader_config_path() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    candidate = package_root / "config.example.json"
    if candidate.is_file():
        return candidate
    return Path("config.example.json")


def _resolve_frontend_asset_path(frontend_dir: Path, request_path: str) -> Path:
    frontend_root = frontend_dir.resolve()
    relative = request_path.lstrip("/") or "index.html"
    candidate = (frontend_root / relative).resolve()
    try:
        candidate.relative_to(frontend_root)
    except ValueError:
        return frontend_root / "index.html"
    if not candidate.exists() or candidate.is_dir():
        return frontend_root / "index.html"
    return candidate


class AdapterRequestHandler(BaseHTTPRequestHandler):
    app: AdapterApp
    frontend_dir: Path
    request_timeout_seconds = 120.0

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.request_timeout_seconds)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/v1/config":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.get_config(), "error": None})
                return
            if parsed.path == "/api/v1/sessions":
                params = parse_qs(parsed.query)
                cursor = int(params.get("cursor", ["0"])[0])
                limit = int(params.get("limit", ["100"])[0])
                data = self.app.list_sessions(_filters_from_query(params), cursor=cursor, limit=limit)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            if parsed.path == "/api/v1/facets":
                data = self.app.get_facets()
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            if parsed.path == "/api/v1/templates":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.list_templates(), "error": None})
                return
            if parsed.path.startswith("/media/original/"):
                session_key, image_index = self._parse_original_media_path(parsed.path)
                path, mime = self.app.get_media_path(session_key, image_index)
                self._send_file(path, force_mime=mime)
                return
            if parsed.path.startswith("/media/preprocessed/"):
                session_key, image_index, policy = self._parse_preprocessed_media_path(parsed.path)
                path, mime = self.app.get_preprocessed_preview_path(session_key, image_index, policy)
                self._send_file(path, force_mime=mime)
                return
            self._serve_frontend(parsed.path)
        except AdapterError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "file not found")
        except socket.timeout:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "request timed out")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/v1/templates":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.save_template(payload), "error": None})
                return
            if parsed.path == "/api/v1/reports":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.run_report(payload), "error": None})
                return
            if parsed.path == "/api/v1/playground":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.run_playground(payload), "error": None})
                return
            if parsed.path == "/api/v1/labels/dry-run":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.dry_run_labels(payload), "error": None})
                return
            if parsed.path == "/api/v1/labels/write":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.write_ai_labels(payload), "error": None})
                return
            if parsed.path == "/api/v1/exports/aligned":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.export_aligned_labels(), "error": None})
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid JSON body")
        except AdapterError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except socket.timeout:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "request timed out")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_security_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _serve_frontend(self, path: str) -> None:
        self._send_file(_resolve_frontend_asset_path(self.frontend_dir, path))

    def _send_file(self, path: Path, force_mime: str | None = None) -> None:
        body = path.read_bytes()
        mime = force_mime or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, body, mime)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "data": None, "error": message})

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._send_security_headers()
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application") else "public, max-age=300")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self';",
        )

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise AdapterError("JSON body must be an object")
        return payload

    @staticmethod
    def _parse_original_media_path(path: str) -> tuple[str, int]:
        parts = path.removeprefix("/media/original/").split("/")
        if len(parts) != 2:
            raise AdapterError("expected /media/original/{session_key}/{image_index}")
        return parts[0], int(parts[1].split(".")[0])

    @staticmethod
    def _parse_preprocessed_media_path(path: str) -> tuple[str, int, str]:
        parts = path.removeprefix("/media/preprocessed/").split("/")
        if len(parts) != 3:
            raise AdapterError("expected /media/preprocessed/{session_key}/{image_index}/{policy}")
        policy = parts[2].split(".")[0]
        return parts[0], int(parts[1]), policy


def _filters_from_query(params: dict[str, list[str]]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    mapping = {
        "dataset_id": "dataset_ids",
        "task_yaml_name": "task_yaml_names",
        "task_name": "task_names",
        "workflow_name": "workflow_names",
        "ckpt": "ckpts",
        "ckpt_family": "ckpt_families",
        "prompt_template_key": "prompt_template_keys",
        "aspect_ratio": "aspect_ratios",
        "session_key": "session_keys",
    }
    for query_key, filter_key in mapping.items():
        values = [value for value in params.get(query_key, []) if value]
        if values:
            filters[filter_key] = values
    for key in ("min_session_index", "max_session_index"):
        if key in params and params[key]:
            filters[key] = params[key][0]
    return filters


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the image grader adapter admin console.")
    parser.add_argument("--work-dir", required=True, help="Adapter work directory for templates, runs, reports, cache, and AI labels")
    parser.add_argument("--dataset-root", required=True, help="Root output directory to scan recursively for sessions.jsonl")
    parser.add_argument("--port", default=8087, type=int, help="Port to bind, defaults to 8087")
    parser.add_argument("--host", default="tailscale-auto", help="Host to bind. Default detects Tailscale IPv4, then 127.0.0.1")
    parser.add_argument("--grader-config", default=str(_default_grader_config_path()), help="Path to image_grader JSON config")
    parser.add_argument("--frontend-dir", default=None, help="Override frontend static asset directory")
    parser.add_argument("--validate-runtime", action="store_true", help="Validate configured model files before serving")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.grader_config)
        if args.validate_runtime:
            validate_runtime(config)
        app = AdapterApp(work_dir=args.work_dir, dataset_root=args.dataset_root, grader_config=config)
    except (ConfigError, RuntimeValidationError, AdapterError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    frontend_dir = Path(args.frontend_dir).resolve() if args.frontend_dir else (Path(__file__).resolve().parent / "frontend")
    handler = type("ConfiguredAdapterRequestHandler", (AdapterRequestHandler,), {"app": app, "frontend_dir": frontend_dir})
    host = resolve_bind_host(args.host)

    class AdapterThreadingHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

    server = AdapterThreadingHTTPServer((host, int(args.port)), handler)
    print(f"Image grader adapter listening on http://{host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
