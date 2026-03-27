#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for a self-play worker machine.
#
# Fastest form:
#   bash scripts/start_standard_self_play_worker.sh
#
# Optional:
#   export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://192.168.0.175:8787
#   export TCG_AI_STANDARD_SELF_PLAY_WORKER_ID_PREFIX=macbook-m1
#   export TCG_AI_STANDARD_SELF_PLAY_WORKER_COUNT=8
#   export TCG_AI_STANDARD_SELF_PLAY_POLL_SECONDS=2
#   export TCG_AI_STANDARD_SELF_PLAY_REQUEST_TIMEOUT_SECONDS=30
#   export TCG_AI_STANDARD_SELF_PLAY_HEARTBEAT_INTERVAL_SECONDS=15
#   export TCG_AI_STANDARD_SELF_PLAY_PROGRESS_DIR=standard_ml_data/progress/workers
#
# This script launches one worker process per detected CPU core by default.
# The main coordinator host for this repo currently defaults to:
#   http://192.168.0.175:8787

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/start_standard_self_play_worker.sh [coordinator-url] [worker-prefix]

Examples:
  bash scripts/start_standard_self_play_worker.sh
  bash scripts/start_standard_self_play_worker.sh http://192.168.0.175:8787 macbook-m1
  bash scripts/start_standard_self_play_worker.sh http://192.168.0.175:8787 macbook-m1 --workers 4

Defaults:
  coordinator-url: http://192.168.0.175:8787
  worker-prefix: hostname
  workers: one per detected CPU core

Or use env vars:
  export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://192.168.0.175:8787
  export TCG_AI_STANDARD_SELF_PLAY_WORKER_ID_PREFIX=macbook-m1
  export TCG_AI_STANDARD_SELF_PLAY_WORKER_COUNT=8
  bash scripts/start_standard_self_play_worker.sh
EOF
  exit 0
fi

POSITIONAL_COORDINATOR_URL=""
POSITIONAL_WORKER_PREFIX=""

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  POSITIONAL_COORDINATOR_URL="$1"
  shift
fi

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  POSITIONAL_WORKER_PREFIX="$1"
  shift
fi

COORDINATOR_URL="${TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL:-${POSITIONAL_COORDINATOR_URL:-http://192.168.0.175:8787}}"
WORKER_PREFIX="${TCG_AI_STANDARD_SELF_PLAY_WORKER_ID_PREFIX:-$POSITIONAL_WORKER_PREFIX}"

ARGS=(
  "$COORDINATOR_URL"
)

if [[ -n "$WORKER_PREFIX" ]]; then
  ARGS+=("$WORKER_PREFIX")
fi

ARGS+=(
  --poll-seconds "${TCG_AI_STANDARD_SELF_PLAY_POLL_SECONDS:-5}"
  --request-timeout-seconds "${TCG_AI_STANDARD_SELF_PLAY_REQUEST_TIMEOUT_SECONDS:-30}"
  --heartbeat-interval-seconds "${TCG_AI_STANDARD_SELF_PLAY_HEARTBEAT_INTERVAL_SECONDS:-15}"
)

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_WORKER_COUNT:-}" ]]; then
  ARGS+=(--workers "${TCG_AI_STANDARD_SELF_PLAY_WORKER_COUNT}")
fi

echo "[worker-launch] coordinator=${COORDINATOR_URL}"
if [[ -n "$WORKER_PREFIX" ]]; then
  echo "[worker-launch] worker_prefix=${WORKER_PREFIX}"
else
  echo "[worker-launch] worker_prefix=hostname"
fi

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_PROGRESS_DIR:-}" ]]; then
  ARGS+=(--progress-log-dir "${TCG_AI_STANDARD_SELF_PLAY_PROGRESS_DIR}")
fi

exec python3 scripts/start_standard_self_play_workers.py "${ARGS[@]}" "$@"
