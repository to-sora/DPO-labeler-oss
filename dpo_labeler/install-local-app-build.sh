#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
labeler_env=dpo_labeler/.local-tool-app/venv
export PIP_CACHE_DIR="$PWD/dpo_labeler/.local-tool-app/pip-cache"
export TMPDIR="$PWD/dpo_labeler/.local-tool-app/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
if [ ! -x "$labeler_env/bin/python" ]; then py13 python -m venv "$labeler_env"; fi
"$labeler_env/bin/python" scripts/pip_ipv4.py install Pillow==11.3.0 PyYAML==6.0.2 requests==2.32.5 tqdm==4.67.1
