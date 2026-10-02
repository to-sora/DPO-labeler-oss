from urllib.parse import urlparse

from ..server import LabelerRequestHandler
from .routes import dispatch


class ImportedRequestHandler(LabelerRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith(("/api/v1/imported/", "/media/imported/")):
            dispatch(self, "GET")
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path.startswith("/api/v1/imported/"):
            dispatch(self, "POST")
        else:
            super().do_POST()

    def _serve_frontend(self, path: str) -> None:
        if path in ("/", "/index.html"):
            html = (self.frontend_dir / "index.html").read_text(encoding="utf-8")
            html = html.replace("</head>", '<link rel="stylesheet" href="/imported/style.css">'
                                '<link rel="stylesheet" href="/imported/layout.css"></head>')
            html = html.replace("</body>", '<script type="module" src="/imported/boot.mjs"></script></body>')
            self._send_bytes(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        super()._serve_frontend(path)
