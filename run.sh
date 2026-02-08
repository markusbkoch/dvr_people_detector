#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it first (e.g. python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt)." >&2
  exit 1
fi

PYTHON_BIN=".venv/bin/python"

GALLERY_HOST="${GALLERY_HOST:-127.0.0.1}"
GALLERY_PORT="${GALLERY_PORT:-8765}"
GALLERY_DB_PATH="${GALLERY_DB_PATH:-data/faces.db}"
GALLERY_SNAPSHOT_DIR="${GALLERY_SNAPSHOT_DIR:-data/snapshots}"

echo "Face gallery: http://${GALLERY_HOST}:${GALLERY_PORT}"
echo "Starting surveillance pipeline..."
GALLERY_HOST="$GALLERY_HOST" \
GALLERY_PORT="$GALLERY_PORT" \
GALLERY_DB_PATH="$GALLERY_DB_PATH" \
GALLERY_SNAPSHOT_DIR="$GALLERY_SNAPSHOT_DIR" \
"$PYTHON_BIN" main.py
