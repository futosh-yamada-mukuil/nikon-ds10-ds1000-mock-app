#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
SETUP_PYTHON="${NIKON_SETUP_PYTHON:-python3.12}"

"$SETUP_PYTHON" "$PROJECT_DIR/packaging/check_environment.py" macos
"$SETUP_PYTHON" -m venv "$PROJECT_DIR/.venv"
APP_PYTHON="$PROJECT_DIR/.venv/bin/python"
"$APP_PYTHON" -m pip install --upgrade pip
"$APP_PYTHON" -m pip install 'torch==2.8.0' 'torchvision==0.23.0'
"$APP_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$APP_PYTHON" -m pip check
echo "Environment prepared. Configure config/models.local.json, then run bash run_mac.sh."
