#!/usr/bin/env bash
set -euo pipefail
# Administrator-run prerequisites / 由管理員檢閱後執行
sudo apt-get update
sudo apt-get install -y make cmake curl xz-utils openssl iproute2 libgtk-3-0 libdbus-glib-1-2
