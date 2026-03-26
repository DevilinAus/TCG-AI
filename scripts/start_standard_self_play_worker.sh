#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for a self-play worker machine.
#
# Required:
#   export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://<coordinator-host>:8787
#
# Fastest form:
#   bash scripts/start_standard_self_play_worker.sh http://<coordinator-host>:8787 my-worker-1
#
# Optional:
#   export TCG_AI_STANDARD_SELF_PLAY_WORKER_ID=macbook-m1
#   export TCG_AI_STANDARD_SELF_PLAY_POLL_SECONDS=2
#   export TCG_AI_STANDARD_SELF_PLAY_REQUEST_TIMEOUT_SECONDS=30
#   export TCG_AI_STANDARD_SELF_PLAY_HEARTBEAT_INTERVAL_SECONDS=15
#   export TCG_AI_STANDARD_SELF_PLAY_PROGRESS_LOG=standard_ml_data/progress/worker.log
#
# This script only launches a worker. It does not start training or evaluation.
# The main Linux machine should run the coordinator and later the training/eval steps.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/start_standard_self_play_worker.sh <coordinator-url> [worker-id]

Examples:
  bash scripts/start_standard_self_play_worker.sh http://192.168.1.50:8787 macbook-m1

Or use env vars:
  export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://192.168.1.50:8787
  export TCG_AI_STANDARD_SELF_PLAY_WORKER_ID=macbook-m1
  bash scripts/start_standard_self_play_worker.sh
EOF
  exit 0
fi

POSITIONAL_COORDINATOR_URL=""
POSITIONAL_WORKER_ID=""

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  POSITIONAL_COORDINATOR_URL="$1"
  shift
fi

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  POSITIONAL_WORKER_ID="$1"
  shift
fi

COORDINATOR_URL="${TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL:-$POSITIONAL_COORDINATOR_URL}"
if [[ -z "$COORDINATOR_URL" ]]; then
  echo "Set TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL or pass the coordinator URL as the first argument." >&2
  echo "Example: bash scripts/start_standard_self_play_worker.sh http://192.168.1.20:8787 macbook-m1" >&2
  exit 1
fi

ARGS=(
  --coordinator-url "$COORDINATOR_URL"
  --poll-seconds "${TCG_AI_STANDARD_SELF_PLAY_POLL_SECONDS:-5}"
  --request-timeout-seconds "${TCG_AI_STANDARD_SELF_PLAY_REQUEST_TIMEOUT_SECONDS:-30}"
  --heartbeat-interval-seconds "${TCG_AI_STANDARD_SELF_PLAY_HEARTBEAT_INTERVAL_SECONDS:-15}"
)

WORKER_ID="${TCG_AI_STANDARD_SELF_PLAY_WORKER_ID:-$POSITIONAL_WORKER_ID}"
if [[ -n "$WORKER_ID" ]]; then
  ARGS+=(--worker-id "$WORKER_ID")
fi

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_PROGRESS_LOG:-}" ]]; then
  ARGS+=(--progress-log "$TCG_AI_STANDARD_SELF_PLAY_PROGRESS_LOG")
fi

echo "[worker-launch] coordinator=${COORDINATOR_URL}"
if [[ -n "$WORKER_ID" ]]; then
  echo "[worker-launch] worker_id=${WORKER_ID}"
fi

exec python3 scripts/run_standard_self_play_worker.py "${ARGS[@]}" "$@"
