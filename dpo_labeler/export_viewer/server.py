from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .app import ExportViewerApp, ImportValidationError, ViewerNotFoundError


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


class ExportViewerRequestHandler(BaseHTTPRequestHandler):
    app: ExportViewerApp
    frontend_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/v1/config":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.get_config(), "error": None})
                return
            if parsed.path == "/api/v1/imports":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.list_imports(), "error": None})
                return
            if parsed.path.startswith("/api/v1/imports/") and parsed.path.endswith("/analytics"):
                import_id = parsed.path.split("/")[4]
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "data": self.app.get_import_analytics(import_id), "error": None},
                )
                return
            if parsed.path.startswith("/api/v1/imports/") and parsed.path.endswith("/rows"):
                import_id = parsed.path.split("/")[4]
                params = parse_qs(parsed.query)
                cursor = params.get("cursor", ["0"])[0]
                limit = params.get("limit", ["0"])[0]
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "data": self.app.get_import_rows(import_id, cursor, limit), "error": None},
                )
                return
            if parsed.path.startswith("/api/v1/imports/"):
                import_id = parsed.path.split("/")[4]
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.get_import(import_id), "error": None})
                return
            if parsed.path.startswith("/media/imports/"):
                import_id, row_id, slot = self._parse_media_path(parsed.path)
                path, mime = self.app.get_media_path(import_id, row_id, slot)
                self._send_file(path, force_mime=mime)
                return
            self._serve_frontend(parsed.path)
        except ImportValidationError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except ViewerNotFoundError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        except FileNotFoundError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "File not found")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/v1/imports":
                data = self.app.create_import(str(payload.get("filename", "")), str(payload.get("text", "")))
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except ImportValidationError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/v1/imports/"):
                import_id = parsed.path.split("/")[4]
                data = self.app.delete_import(import_id)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except ViewerNotFoundError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
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
        candidate = _resolve_frontend_asset_path(self.frontend_dir, path)
        self._send_file(candidate)

    def _send_file(self, path: Path, force_mime: str | None = None) -> None:
        payload = path.read_bytes()
        mime = force_mime or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send_bytes(HTTPStatus.OK, payload, mime)

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        extra_headers: Dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._send_security_headers()
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application") else "public, max-age=300")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        status: HTTPStatus,
        payload: Dict[str, Any],
    ) -> None:
        body = json.dumps(
            {
                **payload,
                "request_id": payload.get("request_id") or self._request_id(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "data": None, "error": message})

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self';",
        )

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ImportValidationError("JSON body must be an object")
        return payload

    @staticmethod
    def _parse_media_path(path: str) -> tuple[str, str, str]:
        parts = path.removeprefix("/media/imports/").split("/")
        if len(parts) != 3:
            raise ImportValidationError("Expected /media/imports/{import_id}/{row_id}/{slot}")
        return parts[0], parts[1], parts[2]

    def _request_id(self) -> str:
        return f"req-{id(self):x}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the export viewer web app.")
    parser.add_argument("--state-dir", required=True, help="Directory for persisted import state")
    parser.add_argument(
        "--image-root",
        action="append",
        dest="image_roots",
        required=True,
        help="Directory to recursively search for server-side hashed image files. Repeat for multiple roots.",
    )
    parser.add_argument("--frontend-dir", default=None, help="Override frontend static asset directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8084, type=int, help="Port to bind")
    parser.add_argument("--default-page-size", default=10, type=int, help="Default rows per page")
    parser.add_argument("--max-page-size", default=50, type=int, help="Maximum rows per page")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    app = ExportViewerApp(
        state_dir=args.state_dir,
        image_roots=args.image_roots,
        default_page_size=args.default_page_size,
        max_page_size=args.max_page_size,
    )
    frontend_dir = Path(args.frontend_dir).resolve() if args.frontend_dir else (Path(__file__).resolve().parent / "frontend")
    handler = type("ConfiguredExportViewerRequestHandler", (ExportViewerRequestHandler,), {"app": app, "frontend_dir": frontend_dir})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Export viewer {app.get_config()['app_version']} listening on http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
