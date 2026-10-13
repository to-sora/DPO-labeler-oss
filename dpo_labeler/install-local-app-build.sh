#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
labeler_env=dpo_labeler/.local-tool-app/venv
export PIP_CACHE_DIR="$PWD/dpo_labeler/.local-tool-app/pip-cache"
export TMPDIR="$PWD/dpo_labeler/.local-tool-app/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
if [ ! -x "$labeler_env/bin/python" ]; then "${LABELER_PYTHON:-python3}" -m venv "$labeler_env"; fi
"$labeler_env/bin/python" -m pip install -r requirements.txt
