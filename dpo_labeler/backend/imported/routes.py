from urllib.parse import unquote, urlparse

from .events import Conflict


def dispatch(handler, method: str) -> None:
    path = urlparse(handler.path).path
    parts = [unquote(p) for p in path.strip("/").split("/")]
    try:
        service = handler.imported
        if parts[:2] == ["media", "imported"] and len(parts) == 4:
            handler._send_file(service.image(parts[2], int(parts[3])))
            return
        payload = handler._read_json_body() if method == "POST" else {}
        if len(parts) == 4 and parts[-1] == "tasks":
            data = service.catalog() if method == "GET" else service.import_yaml(payload.get("yaml", ""))
        elif len(parts) == 5 and method == "GET":
            data = service.get(parts[4])
        elif len(parts) == 6 and method == "POST" and parts[5] == "comparisons":
            data = service.submit(parts[4], payload)
        elif len(parts) == 6 and method == "POST" and parts[5] == "export":
            dimension = payload.get("dimension")
            if dimension is not None and type(dimension) is not int:
                raise ValueError("dimension must be an integer")
            handler._send_bytes(200, service.export(parts[4], dimension), "application/zip",
                                extra_headers={"Content-Disposition": 'attachment; filename="imported-preferences.zip"'})
            return
        else:
            raise KeyError(path)
        handler._send_json(200, {"ok": True, "data": data, "error": None})
    except Conflict as exc:
        handler._send_error_json(409, str(exc))
    except (ValueError, TypeError) as exc:
        handler._send_error_json(400, str(exc))
    except (KeyError, FileNotFoundError):
        handler._send_error_json(404, "Task or image not found")
    except OSError as exc:
        handler._send_error_json(400, str(exc))
