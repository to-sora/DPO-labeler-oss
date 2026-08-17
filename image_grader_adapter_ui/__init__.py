from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STANDALONE_ROOT = _REPO_ROOT / "image_grader"
_STANDALONE_PACKAGE = _STANDALONE_ROOT / "image_grader_adapter_ui"

if _STANDALONE_ROOT.is_dir() and str(_STANDALONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_STANDALONE_ROOT))

if _STANDALONE_PACKAGE.is_dir():
    __path__.append(str(_STANDALONE_PACKAGE))
