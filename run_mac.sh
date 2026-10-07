#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
APP_PYTHON="${NIKON_PYTHON:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$APP_PYTHON" ]]; then
  echo "Python environment not found. Run bash setup_mac.sh first, or set NIKON_PYTHON." >&2
  exit 1
fi
APP_PYTHON="$(cd -- "$(dirname -- "$APP_PYTHON")" && pwd)/$(basename -- "$APP_PYTHON")"

cd -- "$PROJECT_DIR"
exec "$APP_PYTHON" -B -m app "$@"
