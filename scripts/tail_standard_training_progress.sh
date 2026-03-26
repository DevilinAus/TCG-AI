#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PROGRESS_DIR="standard_ml_data/progress"
LATEST_PROGRESS_LINK="${PROGRESS_DIR}/latest.log"

print_help() {
  cat <<'EOF'
Usage: bash scripts/tail_standard_training_progress.sh [--log-file <path>]

Tails the latest Standard training pipeline progress log.

Examples:
  bash scripts/tail_standard_training_progress.sh
  bash scripts/tail_standard_training_progress.sh --log-file standard_ml_data/progress/run_20260326T053217Z.log
EOF
}

LOG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LOG_FILE}" ]]; then
  if [[ ! -e "${LATEST_PROGRESS_LINK}" ]]; then
    echo "No latest progress log found at ${LATEST_PROGRESS_LINK}" >&2
    exit 1
  fi
  LOG_FILE="${LATEST_PROGRESS_LINK}"
fi

echo "[progress] tailing ${LOG_FILE}"
tail -f "${LOG_FILE}"
