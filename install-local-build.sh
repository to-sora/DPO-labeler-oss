#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .local-tool
export PIP_CACHE_DIR="$PWD/.local-tool/pip-cache"
export TMPDIR="$PWD/.local-tool/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
if [ ! -x .local-tool/uat/bin/python ]; then
  py13 python -m venv .local-tool/uat
fi
.local-tool/uat/bin/python scripts/pip_ipv4.py install selenium==4.35.0 Pillow==11.3.0 PyYAML==6.0.2
if [ ! -x "$HOME/UAT-firefox/firefox" ]; then
  curl -4fL --connect-timeout 15 --max-time 300 \
    https://archive.mozilla.org/pub/firefox/releases/144.0.2/linux-x86_64/en-US/firefox-144.0.2.tar.xz \
    -o .local-tool/firefox.tar.xz
  mkdir -p "$HOME/UAT-firefox"
  tar -xJf .local-tool/firefox.tar.xz -C "$HOME/UAT-firefox" --strip-components=1
  rm .local-tool/firefox.tar.xz
fi
if [ ! -x .local-tool/geckodriver ]; then
  curl -4fL --connect-timeout 15 --max-time 120 \
    https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz \
    -o .local-tool/geckodriver.tar.gz
  tar -xzf .local-tool/geckodriver.tar.gz -C .local-tool
  rm .local-tool/geckodriver.tar.gz
fi
