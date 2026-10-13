import socket
from urllib.parse import unquote, urlparse

from ..common import AuthenticationError
from .events import Conflict


def dispatch(handler, method: str) -> None:
    path = urlparse(handler.path).path
    parts = [unquote(p) for p in path.strip("/").split("/")]
    try:
        session = handler.app.auth_service.require_session(handler.headers.get("Cookie"))
        service = handler.imported
        if method == "GET" and parts[:2] == ["media", "imported"] and len(parts) == 4:
            handler._send_file(service.image(parts[2], int(parts[3])))
            return
        if parts[:4] != ["api", "v1", "imported", "tasks"]:
            raise KeyError(path)
        if len(parts) == 4 and method == "POST":
            handler._send_json(405, {"ok": False, "data": None,
                                    "error": "Tasks are imported on the server at startup"},
                               extra_headers={"Allow": "GET"})
            return
        payload = handler._read_json_body() if method == "POST" else {}
        if len(parts) == 4 and method == "GET":
            data = service.catalog(reviewer_username=session.reviewer_username)
        elif len(parts) == 5 and method == "GET":
            data = service.get(parts[4], reviewer_username=session.reviewer_username)
        elif len(parts) == 6 and method == "POST" and parts[5] == "comparisons":
            data = service.submit(parts[4], payload, reviewer_username=session.reviewer_username)
        elif len(parts) == 6 and method == "POST" and parts[5] == "export":
            dimension = payload.get("dimension")
            if dimension is not None and type(dimension) is not int:
                raise ValueError("dimension must be an integer")
            archive = service.export(parts[4], dimension, reviewer_username=session.reviewer_username)
            handler._send_bytes(200, archive, "application/zip",
                                extra_headers={"Content-Disposition": 'attachment; filename="imported-preferences.zip"'})
            return
        else:
            raise KeyError(path)
        handler._send_json(200, {"ok": True, "data": data, "error": None})
    except AuthenticationError as exc:
        handler._send_error_json(401, str(exc))
    except Conflict as exc:
        handler._send_error_json(409, str(exc))
    except (ValueError, TypeError) as exc:
        handler._send_error_json(400, str(exc))
    except (KeyError, FileNotFoundError):
        handler._send_error_json(404, "Task or image not found")
    except socket.timeout:
        handler._send_error_json(408, "Request timed out")
    except OSError as exc:
        handler._send_error_json(400, str(exc))
    except Exception as exc:
        handler._send_error_json(500, str(exc))
