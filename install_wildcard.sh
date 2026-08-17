#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_PYTHON="${OSS_SETUP_PYTHON:-${PYTHON_BIN:-python3}}"
exec "$SETUP_PYTHON" "$SCRIPT_DIR/scripts/install_wildcards.py" "$@"
