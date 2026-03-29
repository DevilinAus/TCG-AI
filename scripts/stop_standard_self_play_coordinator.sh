#!/usr/bin/env bash
set -euo pipefail

# Gracefully stop the distributed self-play coordinator launched by
# scripts/start_standard_self_play_coordinator.sh.
#
# Usage:
#   bash scripts/stop_standard_self_play_coordinator.sh
#   bash scripts/stop_standard_self_play_coordinator.sh /path/to/coordinator.pid
#
# Optional:
#   TCG_AI_STANDARD_SELF_PLAY_PID_FILE=standard_ml_data/distributed_self_play/run_20260326T120000Z/coordinator.pid

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PID_FILE="${1:-${TCG_AI_STANDARD_SELF_PLAY_PID_FILE:-}}"

if [[ -z "$PID_FILE" ]]; then
  echo "[coordinator-stop] provide a pid file path or set TCG_AI_STANDARD_SELF_PLAY_PID_FILE" >&2
  exit 1
fi

if [[ ! -f "$PID_FILE" ]]; then
  echo "[coordinator-stop] pid file not found: $PID_FILE" >&2
  exit 1
fi

PID="$(tr -d '[:space:]' < "$PID_FILE")"

if [[ -z "$PID" ]]; then
  echo "[coordinator-stop] pid file was empty: $PID_FILE" >&2
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "[coordinator-stop] process $PID is not running; removing stale pid file"
  rm -f "$PID_FILE"
  exit 0
fi

echo "[coordinator-stop] sending SIGTERM to pid=$PID"
kill -TERM "$PID"

for _ in {1..30}; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[coordinator-stop] stopped cleanly"
    exit 0
  fi
  sleep 1
done

echo "[coordinator-stop] process still running after 30s: $PID" >&2
exit 1
