#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
labeler_python=dpo_labeler/.local-tool-app/venv/bin/python
if [ -x .venv/bin/python ]; then labeler_python=.venv/bin/python; fi
if [ ! -x "$labeler_python" ]; then
  echo 'Run dpo_labeler/install-local-app-build.sh / 請先建立應用環境' >&2
  exit 1
fi
exec "$labeler_python" -m dpo_labeler.launch "$@"
