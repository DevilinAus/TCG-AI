#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting TCG AI demo server..."
echo "Open http://127.0.0.1:8000 once the server is ready."

exec python3 -m backend.tcg_ai.server
