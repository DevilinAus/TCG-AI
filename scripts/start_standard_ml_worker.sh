#!/usr/bin/env bash
set -euo pipefail

# Linux worker quick notes:
# - Change MODEL_CHECKPOINT below, or pass --checkpoint /path/to/champion.pt, if your model lives elsewhere.
# - Set TCG_AI_STANDARD_REMOTE_API_TOKEN (or pass --token ...) so only your Mac can call this worker.
# - If you want the faster FastAPI worker, install the optional deps first:
#     pip install -e '.[standard-ml]'
# - Your Mac backend should point at:
#     http://<linux-ip>:8100/api/standard-ml/decision
#     http://<linux-ip>:8100/api/standard-ml/batch-eval
# - If the checkpoint file does not exist yet, the worker will still boot, but NN mode in the UI will stay unavailable.
#
# Example:
#   bash scripts/start_standard_ml_worker.sh \
#     --checkpoint /home/andrew/models/champion.pt \
#     --token change-me
#
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CHECKPOINT="$ROOT_DIR/standard_ml_data/champion.pt"

HOST="${TCG_AI_STANDARD_ML_HOST:-0.0.0.0}"
PORT="${TCG_AI_STANDARD_ML_PORT:-8100}"
MODEL_CHECKPOINT="${TCG_AI_STANDARD_MODEL_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
PREFERRED_SERVER="${TCG_AI_STANDARD_ML_SERVER:-auto}"

usage() {
  cat <<EOF
Usage: bash scripts/start_standard_ml_worker.sh [options]

Starts the remote Standard ML worker for NN mode.

Options:
  --host <host>              Bind host. Default: ${HOST}
  --port <port>              Bind port. Default: ${PORT}
  --checkpoint <path>        Model checkpoint path. Default: ${MODEL_CHECKPOINT}
  --token <token>            Shared API token for the worker
  --fastapi                  Force the FastAPI worker
  --stdlib                   Force the stdlib HTTP worker
  -h, --help                 Show this help message

Environment overrides:
  PYTHON_BIN
  TCG_AI_STANDARD_ML_HOST
  TCG_AI_STANDARD_ML_PORT
  TCG_AI_STANDARD_MODEL_CHECKPOINT
  TCG_AI_STANDARD_REMOTE_API_TOKEN
  TCG_AI_STANDARD_ML_SERVER=auto|fastapi|stdlib

Examples:
  bash scripts/start_standard_ml_worker.sh
  bash scripts/start_standard_ml_worker.sh --checkpoint /home/andrew/models/champion.pt
  bash scripts/start_standard_ml_worker.sh --port 8100 --token my-shared-token
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --checkpoint)
      MODEL_CHECKPOINT="${2:-}"
      shift 2
      ;;
    --token)
      export TCG_AI_STANDARD_REMOTE_API_TOKEN="${2:-}"
      shift 2
      ;;
    --fastapi)
      PREFERRED_SERVER="fastapi"
      shift
      ;;
    --stdlib)
      PREFERRED_SERVER="stdlib"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Could not find a Python interpreter. Set PYTHON_BIN or create .venv." >&2
  exit 1
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TCG_AI_STANDARD_ML_HOST="$HOST"
export TCG_AI_STANDARD_ML_PORT="$PORT"
export TCG_AI_STANDARD_MODEL_CHECKPOINT="$MODEL_CHECKPOINT"

echo "Starting Standard ML worker from $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Bind: http://${HOST}:${PORT}"
echo "Checkpoint: $MODEL_CHECKPOINT"

if [[ -n "${TCG_AI_STANDARD_REMOTE_API_TOKEN:-}" ]]; then
  echo "API token: configured"
else
  echo "API token: not set"
  echo "Warning: the worker will accept unauthenticated LAN requests."
fi

if [[ ! -f "$MODEL_CHECKPOINT" ]]; then
  echo "Warning: checkpoint not found at $MODEL_CHECKPOINT"
  echo "The worker will start, but NN mode will stay unavailable until a model checkpoint exists."
fi

supports_fastapi() {
  "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('fastapi') and importlib.util.find_spec('uvicorn') else 1)"
}

run_fastapi() {
  echo "Server: FastAPI"
  exec "$PYTHON_BIN" -c 'import os; from backend.tcg_ai.game_modes.standard.ml_fastapi import run; run(host=os.environ["TCG_AI_STANDARD_ML_HOST"], port=int(os.environ["TCG_AI_STANDARD_ML_PORT"]))'
}

run_stdlib() {
  echo "Server: stdlib"
  exec "$PYTHON_BIN" -c 'import os; from backend.tcg_ai.game_modes.standard.ml_server import run; run(host=os.environ["TCG_AI_STANDARD_ML_HOST"], port=int(os.environ["TCG_AI_STANDARD_ML_PORT"]))'
}

case "$PREFERRED_SERVER" in
  fastapi)
    if ! supports_fastapi; then
      echo "FastAPI worker requested, but fastapi/uvicorn are not installed." >&2
      echo "Install the 'standard-ml' extras or rerun with --stdlib." >&2
      exit 1
    fi
    run_fastapi
    ;;
  stdlib)
    run_stdlib
    ;;
  auto)
    if supports_fastapi; then
      run_fastapi
    fi
    run_stdlib
    ;;
  *)
    echo "Unknown TCG_AI_STANDARD_ML_SERVER value: $PREFERRED_SERVER" >&2
    exit 1
    ;;
esac
