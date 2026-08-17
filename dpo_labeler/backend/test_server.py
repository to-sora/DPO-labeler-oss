from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dpo_labeler.backend.server import _resolve_frontend_asset_path


class LabelerFrontendPathTests(unittest.TestCase):
    def test_resolve_frontend_asset_path_rejects_sibling_prefix_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frontend_dir = root / "frontend"
            sibling_dir = root / "frontend-evil"
            frontend_dir.mkdir(parents=True, exist_ok=True)
            sibling_dir.mkdir(parents=True, exist_ok=True)
            (frontend_dir / "index.html").write_text("INDEX", encoding="utf-8")
            (sibling_dir / "secret.txt").write_text("SECRET", encoding="utf-8")

            resolved = _resolve_frontend_asset_path(frontend_dir, "/../frontend-evil/secret.txt")

            self.assertEqual(resolved, (frontend_dir / "index.html").resolve())
            self.assertEqual(resolved.read_text(encoding="utf-8"), "INDEX")


if __name__ == "__main__":
    unittest.main()
