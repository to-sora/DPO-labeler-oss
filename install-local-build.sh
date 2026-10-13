#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .local-tool
export PIP_CACHE_DIR="$PWD/.local-tool/pip-cache"
export TMPDIR="$PWD/.local-tool/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
if [ ! -x .local-tool/uat/bin/python ]; then
  "${LABELER_PYTHON:-python3}" -m venv .local-tool/uat
fi
.local-tool/uat/bin/python -m pip install -r requirements.txt selenium==4.35.0
echo 'Browser checks also require Firefox and geckodriver on PATH (or FIREFOX_BINARY and GECKODRIVER).'
