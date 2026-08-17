from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .app import DpoLabelerApp, FilterValidationError, LabelEventValidationError
from .common import AuthenticationError, EXPORT_FILENAMES


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


class LabelerRequestHandler(BaseHTTPRequestHandler):
    app: DpoLabelerApp
    frontend_dir: Path
    request_timeout_seconds = 30.0

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.request_timeout_seconds)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/v1/config":
                self._send_json(HTTPStatus.OK, {"ok": True, "data": self.app.get_config(), "error": None})
                return
            if parsed.path == "/api/v1/session/me":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "data": self.app.get_session(self.headers.get("Cookie")), "error": None},
                )
                return
            if parsed.path == "/api/v1/catalog":
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "data": self.app.get_catalog(), "error": None},
                )
                return
            if parsed.path.startswith("/api/v1/review-sessions/") and parsed.path.endswith("/queue"):
                review_id = parsed.path.split("/")[4]
                params = parse_qs(parsed.query)
                cursor = int(params.get("cursor", ["0"])[0])
                limit = int(params.get("limit", ["24"])[0])
                payload = self.app.get_review_queue(self.headers.get("Cookie"), review_id, cursor, limit)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": payload, "error": None})
                return
            if parsed.path.startswith("/api/v1/review-sessions/") and "/pairs/" in parsed.path:
                parts = parsed.path.split("/")
                review_id = parts[4]
                dataset_id = parts[6]
                session_id = parts[7]
                payload = self.app.get_review_pair(self.headers.get("Cookie"), review_id, dataset_id, session_id)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": payload, "error": None})
                return
            if parsed.path.startswith("/media/original/"):
                dataset_id, session_id, image_index = self._parse_media_path(parsed.path, "/media/original/")
                self._send_file(
                    self.app.get_image_path(self.headers.get("Cookie"), dataset_id, session_id, image_index)
                )
                return
            if parsed.path.startswith("/media/preview/"):
                dataset_id, session_id, image_index = self._parse_media_path(parsed.path, "/media/preview/")
                self._send_file(
                    self.app.get_preview_path(self.headers.get("Cookie"), dataset_id, session_id, image_index),
                    force_mime="image/jpeg",
                )
                return
            if parsed.path.startswith("/public-share/media/"):
                share_token = parsed.path.removeprefix("/public-share/media/").removesuffix(".jpg")
                self._send_file(
                    self.app.get_public_share_preview_path(share_token),
                    force_mime="image/jpeg",
                )
                return
            if parsed.path.startswith("/public-share/"):
                share_token = parsed.path.removeprefix("/public-share/")
                payload = self.app.render_public_share_page(share_token, self._request_origin())
                self._send_bytes(HTTPStatus.OK, payload.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._serve_frontend(parsed.path)
        except (LabelEventValidationError, FilterValidationError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except AuthenticationError as exc:
            self._send_error_json(HTTPStatus.UNAUTHORIZED, str(exc))
        except KeyError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except FileNotFoundError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "File not found")
        except socket.timeout:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "Request timed out")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/v1/session/start":
                data, cookie_header = self.app.start_session(
                    invite_token=str(payload.get("invite_token", "")),
                    reviewer_username=str(payload.get("reviewer_username", "")),
                    client_instance_id=str(payload.get("client_instance_id", "")),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "data": data, "error": None},
                    extra_headers={"Set-Cookie": cookie_header},
                )
                return
            if parsed.path == "/api/v1/session/end":
                data, cookie_header = self.app.end_session()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "data": data, "error": None},
                    extra_headers={"Set-Cookie": cookie_header},
                )
                return
            if parsed.path == "/api/v1/review-sessions":
                data = self.app.create_review_session(self.headers.get("Cookie"), payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            if parsed.path == "/api/v1/label-events":
                event = self.app.submit_label_event(self.headers.get("Cookie"), payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": event, "error": None})
                return
            if parsed.path == "/api/v1/exports/preview":
                data = self.app.preview_export(self.headers.get("Cookie"), payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            if parsed.path == "/api/v1/exports/download":
                payload_text, filename = self.app.export_text_from_request(self.headers.get("Cookie"), payload)
                self._send_bytes(
                    HTTPStatus.OK,
                    payload_text.encode("utf-8"),
                    "application/x-ndjson; charset=utf-8",
                    extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
                return
            if parsed.path == "/api/v1/shares/twitter":
                data = self.app.create_twitter_share(self.headers.get("Cookie"), payload, self._request_origin())
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            if parsed.path == "/api/v1/shares/twitter/post":
                data = self.app.post_twitter_image(self.headers.get("Cookie"), payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "data": data, "error": None})
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except (LabelEventValidationError, FilterValidationError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except AuthenticationError as exc:
            self._send_error_json(HTTPStatus.UNAUTHORIZED, str(exc))
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
        except socket.timeout:
            self._send_error_json(HTTPStatus.REQUEST_TIMEOUT, "Request timed out")
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
        *,
        extra_headers: Dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            {
                **payload,
                "request_id": payload.get("request_id") or self._request_id(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", extra_headers=extra_headers)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "data": None, "error": message})

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; manifest-src 'self'; worker-src 'self';",
        )

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise LabelEventValidationError("JSON body must be an object")
        return payload

    @staticmethod
    def _parse_media_path(path: str, prefix: str) -> tuple[str, str, int]:
        parts = path.removeprefix(prefix).split("/")
        if len(parts) != 3:
            raise LabelEventValidationError("Expected dataset_id/session_id/image_index path")
        image_index = int(parts[2].split(".")[0])
        return parts[0], parts[1], image_index

    def _request_id(self) -> str:
        return f"req-{uuid.uuid4().hex}"

    def _request_origin(self) -> str:
        forwarded_proto = (self.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip()
        forwarded_host = (self.headers.get("X-Forwarded-Host") or "").split(",")[0].strip()
        host = forwarded_host or (self.headers.get("Host") or f"{self.server.server_name}:{self.server.server_port}")
        proto = forwarded_proto or "http"
        return f"{proto}://{host}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the mobile-first DPO labeling web app.")
    parser.add_argument("--dataset-root", required=True, help="Root directory to recursively scan for sessions.jsonl files")
    parser.add_argument("--state-dir", required=True, help="Directory for label events, preview cache, and exports")
    parser.add_argument("--frontend-dir", default=None, help="Override frontend static asset directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8787, type=int, help="Port to bind")
    parser.add_argument("--invite-token", required=True, help="Shared invite token required to start an app session")
    parser.add_argument("--session-secret", default=None, help="Secret used to sign app session cookies")
    parser.add_argument("--review-round-seed", default="default-round-v1", help="Stable seed for deterministic random review order")
    parser.add_argument("--rescan-seconds", default=30, type=int, help="Minimum seconds between dataset rescans")
    parser.add_argument("--preview-max-width", default=960, type=int, help="Maximum preview width")
    parser.add_argument("--session-max-age-seconds", default=60 * 60 * 24 * 7, type=int, help="Session cookie lifetime")
    parser.add_argument("--cookie-secure", action="store_true", help="Set the Secure attribute on app session cookies")
    parser.add_argument("--x-consumer-key", default=None, help="X API consumer key for direct image posting")
    parser.add_argument("--x-consumer-secret", default=None, help="X API consumer secret for direct image posting")
    parser.add_argument("--x-access-token", default=None, help="X API access token for direct image posting")
    parser.add_argument("--x-access-token-secret", default=None, help="X API access token secret for direct image posting")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        dest="exclude_dirs",
        metavar="PATTERN",
        help="Exclude any dataset path segment matching this rsync-style glob (repeatable)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    app = DpoLabelerApp(
        dataset_root=args.dataset_root,
        state_dir=args.state_dir,
        invite_token=args.invite_token,
        session_secret=args.session_secret,
        preview_max_width=args.preview_max_width,
        rescan_seconds=args.rescan_seconds,
        session_max_age_seconds=args.session_max_age_seconds,
        cookie_secure=args.cookie_secure,
        review_round_seed=args.review_round_seed,
        x_consumer_key=args.x_consumer_key,
        x_consumer_secret=args.x_consumer_secret,
        x_access_token=args.x_access_token,
        x_access_token_secret=args.x_access_token_secret,
        exclude_dirs=args.exclude_dirs,
    )
    frontend_dir = Path(args.frontend_dir).resolve() if args.frontend_dir else (Path(__file__).resolve().parents[1] / "frontend")
    handler = type("ConfiguredLabelerRequestHandler", (LabelerRequestHandler,), {"app": app, "frontend_dir": frontend_dir})
    class LabelerThreadingHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

    server = LabelerThreadingHTTPServer((args.host, args.port), handler)
    print(f"DPO labeler {app.get_config()['app_version']} listening on http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
