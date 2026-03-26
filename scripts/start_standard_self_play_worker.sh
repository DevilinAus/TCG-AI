#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for a self-play worker machine.
#
# Required:
#   export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://<linux-box>:8787
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

COORDINATOR_URL="${TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL:-}"
if [[ -z "$COORDINATOR_URL" ]]; then
  echo "Set TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL to the coordinator base URL, for example http://192.168.1.20:8787" >&2
  exit 1
fi

ARGS=(
  --coordinator-url "$COORDINATOR_URL"
  --poll-seconds "${TCG_AI_STANDARD_SELF_PLAY_POLL_SECONDS:-5}"
  --request-timeout-seconds "${TCG_AI_STANDARD_SELF_PLAY_REQUEST_TIMEOUT_SECONDS:-30}"
  --heartbeat-interval-seconds "${TCG_AI_STANDARD_SELF_PLAY_HEARTBEAT_INTERVAL_SECONDS:-15}"
)

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_WORKER_ID:-}" ]]; then
  ARGS+=(--worker-id "$TCG_AI_STANDARD_SELF_PLAY_WORKER_ID")
fi

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_PROGRESS_LOG:-}" ]]; then
  ARGS+=(--progress-log "$TCG_AI_STANDARD_SELF_PLAY_PROGRESS_LOG")
fi

exec python3 scripts/run_standard_self_play_worker.py "${ARGS[@]}" "$@"
