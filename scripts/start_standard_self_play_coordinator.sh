#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for the main coordinator machine.
#
# Fastest form:
#   bash scripts/start_standard_self_play_coordinator.sh
#
# Adjust these with env vars if needed:
#   TCG_AI_STANDARD_SELF_PLAY_HOST=0.0.0.0
#   TCG_AI_STANDARD_SELF_PLAY_PORT=8787
#   TCG_AI_STANDARD_SELF_PLAY_RUN_ID=run_20260326T120000Z
#   TCG_AI_STANDARD_SELF_PLAY_GAMES=10000
#   TCG_AI_STANDARD_SELF_PLAY_CHUNK_SIZE=10
#   TCG_AI_STANDARD_SELF_PLAY_OUTPUT_ROOT=standard_ml_data/distributed_self_play
#   TCG_AI_STANDARD_SELF_PLAY_OUTPUT_DIR=standard_ml_data/distributed_self_play/run_20260326T120000Z
#   TCG_AI_STANDARD_SELF_PLAY_LEASE_TIMEOUT_SECONDS=1800
#   TCG_AI_STANDARD_SELF_PLAY_PID_FILE=/path/to/custom/coordinator.pid
#
# Search knobs:
#   TCG_AI_STANDARD_SELF_PLAY_MAX_DEPTH=2
#   TCG_AI_STANDARD_SELF_PLAY_BEAM_WIDTH=4
#   TCG_AI_STANDARD_SELF_PLAY_OPPONENT_BRANCH_WIDTH=2
#
# Optional:
#   TCG_AI_STANDARD_SELF_PLAY_SEED=12345
#   TCG_AI_STANDARD_SELF_PLAY_MAX_ACTIONS_PER_GAME=200
#   TCG_AI_STANDARD_SELF_PLAY_ORACLE=heuristic
#   TCG_AI_STANDARD_SELF_PLAY_CHECKPOINT=/path/to/champion.pt
#
# Workers on other machines should point at:
#   http://<linux-box>:<port>

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HOST="${TCG_AI_STANDARD_SELF_PLAY_HOST:-0.0.0.0}"
PORT="${TCG_AI_STANDARD_SELF_PLAY_PORT:-8787}"
RUN_ID="${TCG_AI_STANDARD_SELF_PLAY_RUN_ID:-}"
GAMES="${TCG_AI_STANDARD_SELF_PLAY_GAMES:-10000}"
CHUNK_SIZE="${TCG_AI_STANDARD_SELF_PLAY_CHUNK_SIZE:-10}"
DEFAULT_OUTPUT_ROOT="standard_ml_data/distributed_self_play"
OUTPUT_ROOT="${TCG_AI_STANDARD_SELF_PLAY_OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
OUTPUT_DIR="${TCG_AI_STANDARD_SELF_PLAY_OUTPUT_DIR:-}"
LEASE_TIMEOUT_SECONDS="${TCG_AI_STANDARD_SELF_PLAY_LEASE_TIMEOUT_SECONDS:-1800}"
PID_FILE="${TCG_AI_STANDARD_SELF_PLAY_PID_FILE:-}"
MAX_DEPTH="${TCG_AI_STANDARD_SELF_PLAY_MAX_DEPTH:-2}"
BEAM_WIDTH="${TCG_AI_STANDARD_SELF_PLAY_BEAM_WIDTH:-4}"
OPPONENT_BRANCH_WIDTH="${TCG_AI_STANDARD_SELF_PLAY_OPPONENT_BRANCH_WIDTH:-2}"
MAX_ACTIONS_PER_GAME="${TCG_AI_STANDARD_SELF_PLAY_MAX_ACTIONS_PER_GAME:-200}"
ORACLE="${TCG_AI_STANDARD_SELF_PLAY_ORACLE:-heuristic}"

if [[ -z "${TCG_AI_STANDARD_SELF_PLAY_OUTPUT_DIR:-}" && -t 0 ]]; then
  echo "[coordinator-launch] storage directory"
  echo "  Press Enter to keep the default storage root."
  echo "  Default: ${DEFAULT_OUTPUT_ROOT}"
  echo "  You can also paste an existing run directory to resume it directly."
  read -r -p "  Storage dir: " PROMPTED_OUTPUT_PATH
  if [[ -n "${PROMPTED_OUTPUT_PATH}" ]]; then
    BASENAME="$(basename "${PROMPTED_OUTPUT_PATH}")"
    if [[ "${BASENAME}" == run_* ]] || [[ -f "${PROMPTED_OUTPUT_PATH}/coordinator_state.json" ]] || [[ -f "${PROMPTED_OUTPUT_PATH}/summary.json" ]]; then
      OUTPUT_DIR="${PROMPTED_OUTPUT_PATH}"
    else
      OUTPUT_ROOT="${PROMPTED_OUTPUT_PATH}"
    fi
  fi
fi

ARGS=(
  --host "$HOST"
  --port "$PORT"
  --games "$GAMES"
  --chunk-size "$CHUNK_SIZE"
  --lease-timeout-seconds "$LEASE_TIMEOUT_SECONDS"
  --max-depth "$MAX_DEPTH"
  --beam-width "$BEAM_WIDTH"
  --opponent-branch-width "$OPPONENT_BRANCH_WIDTH"
  --max-actions-per-game "$MAX_ACTIONS_PER_GAME"
  --oracle "$ORACLE"
)

if [[ -n "$RUN_ID" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
else
  ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [[ -n "$PID_FILE" ]]; then
  ARGS+=(--pid-file "$PID_FILE")
fi

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_SEED:-}" ]]; then
  ARGS+=(--seed "$TCG_AI_STANDARD_SELF_PLAY_SEED")
fi

if [[ -n "${TCG_AI_STANDARD_SELF_PLAY_CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$TCG_AI_STANDARD_SELF_PLAY_CHECKPOINT")
fi

echo "[coordinator-launch] dashboard=http://127.0.0.1:${PORT}/dashboard"
if [[ -n "$RUN_ID" ]]; then
  echo "[coordinator-launch] requested_run_id=${RUN_ID}"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  echo "[coordinator-launch] requested_output_dir=${OUTPUT_DIR}"
else
  echo "[coordinator-launch] storage_root=${OUTPUT_ROOT}"
fi
if [[ -n "$PID_FILE" ]]; then
  echo "[coordinator-launch] requested_pid_file=${PID_FILE}"
fi
echo "[coordinator-launch] coordinator will resume the newest incomplete run before creating a new one"
echo "[coordinator-launch] worker example:"
echo "  bash scripts/start_standard_self_play_worker.sh http://192.168.0.175:${PORT}"
echo "  scripts\\start_standard_self_play_worker.cmd http://192.168.0.175:${PORT}"

exec python3 scripts/run_standard_self_play_coordinator.py "${ARGS[@]}" "$@"
