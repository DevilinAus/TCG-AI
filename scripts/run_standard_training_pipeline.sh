#!/usr/bin/env bash
set -euo pipefail

# Standard self-play -> train -> evaluate/promote pipeline for the current
# Ampharos ex Battle Deck vs Lucario ex Battle Deck MVP.
#
# Adjust these flags when launching if you want to change scale or hardware use:
#   --games 100000
#   --workers 8
#   --chunk-size 250
#   --train-device cuda
#   --train-batch-size 128
#   --epochs 1
#   --promote-path standard_ml_data/champion.pt
#
# Notes:
# - This script expects the repo checkout to exist on the Linux box.
# - PyTorch/CUDA must be installed for training on `--train-device cuda`.
# - The promoted checkpoint is written in the same format the remote Standard
#   NN worker already loads, so after promotion you can point the worker at it
#   or leave it at the default `standard_ml_data/champion.pt`.
# - If you already have a current champion checkpoint and want to compare
#   against something else, pass `--baseline /path/to/baseline.pt`.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ID="$(date -u +run_%Y%m%dT%H%M%SZ)"
SELF_PLAY_DIR="standard_ml_data/self_play/${RUN_ID}"
CHECKPOINT_DIR="standard_ml_data/checkpoints/${RUN_ID}"
EVAL_DIR="standard_ml_data/evaluations/${RUN_ID}"

GAMES=100000
WORKERS=8
CHUNK_SIZE=250
MAX_ACTIONS_PER_GAME=200
MAX_DEPTH=2
BEAM_WIDTH=4
OPPONENT_BRANCH_WIDTH=2
SELF_PLAY_SEED=""

TRAIN_DEVICE="auto"
TRAIN_BATCH_SIZE=128
EPOCHS=1
LEARNING_RATE=3e-4
WEIGHT_DECAY=1e-4
VALUE_LOSS_WEIGHT=0.5
LOG_EVERY=50
SAVE_EVERY=500
EVAL_BATCHES=20
VALIDATION_MOD=20
VALIDATION_BUCKET=0
MAX_TRAIN_RECORDS=""
MAX_EVAL_RECORDS=""

EVAL_GAMES=400
EVAL_WORKERS=1
EVAL_CHUNK_SIZE=50
EVAL_SEED=""
BASELINE=""
PROMOTE_PATH="standard_ml_data/champion.pt"
PROMOTION_THRESHOLD=0.55

print_help() {
  cat <<'EOF'
Usage: bash scripts/run_standard_training_pipeline.sh [options]

Runs the full Standard MVP training loop:
1. self-play data generation
2. model training
3. checkpoint evaluation and optional champion promotion

Options:
  --run-id <id>                     Override the generated UTC run id.
  --games <n>                       Self-play games. Default: 100000
  --workers <n>                     Self-play workers. Default: 8
  --chunk-size <n>                  Self-play chunk size. Default: 250
  --max-actions-per-game <n>        Max actions per self-play game. Default: 200
  --max-depth <n>                   Search max depth. Default: 2
  --beam-width <n>                  Search beam width. Default: 4
  --opponent-branch-width <n>       Opponent branch width. Default: 2
  --self-play-seed <n>              Optional fixed self-play seed.
  --train-device <auto|cuda|cpu>    Training device. Default: auto
  --train-batch-size <n>            Training batch size. Default: 128
  --epochs <n>                      Training epochs. Default: 1
  --learning-rate <float>           Training learning rate. Default: 3e-4
  --weight-decay <float>            Training weight decay. Default: 1e-4
  --value-loss-weight <float>       Value loss weight. Default: 0.5
  --log-every <n>                   Training log frequency. Default: 50
  --save-every <n>                  Training checkpoint frequency. Default: 500
  --eval-batches <n>                Validation batches per epoch. Default: 20
  --validation-mod <n>              Validation split modulus. Default: 20
  --validation-bucket <n>           Validation split bucket. Default: 0
  --max-train-records <n>           Optional cap for train records.
  --max-eval-records <n>            Optional cap for eval records.
  --eval-games <n>                  Head-to-head evaluation games. Default: 400
  --eval-workers <n>                Evaluation workers. Default: 1
  --eval-chunk-size <n>             Evaluation chunk size. Default: 50
  --eval-seed <n>                   Optional fixed evaluation seed.
  --baseline <path>                 Optional baseline checkpoint.
  --promote-path <path>             Champion output path. Default: standard_ml_data/champion.pt
  --promotion-threshold <float>     Promotion win rate threshold. Default: 0.55
  --help                            Show this message.

Example:
  bash scripts/run_standard_training_pipeline.sh --games 100000 --workers 8 --train-device cuda --epochs 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      SELF_PLAY_DIR="standard_ml_data/self_play/${RUN_ID}"
      CHECKPOINT_DIR="standard_ml_data/checkpoints/${RUN_ID}"
      EVAL_DIR="standard_ml_data/evaluations/${RUN_ID}"
      shift 2
      ;;
    --games) GAMES="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --max-actions-per-game) MAX_ACTIONS_PER_GAME="$2"; shift 2 ;;
    --max-depth) MAX_DEPTH="$2"; shift 2 ;;
    --beam-width) BEAM_WIDTH="$2"; shift 2 ;;
    --opponent-branch-width) OPPONENT_BRANCH_WIDTH="$2"; shift 2 ;;
    --self-play-seed) SELF_PLAY_SEED="$2"; shift 2 ;;
    --train-device) TRAIN_DEVICE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --value-loss-weight) VALUE_LOSS_WEIGHT="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --eval-batches) EVAL_BATCHES="$2"; shift 2 ;;
    --validation-mod) VALIDATION_MOD="$2"; shift 2 ;;
    --validation-bucket) VALIDATION_BUCKET="$2"; shift 2 ;;
    --max-train-records) MAX_TRAIN_RECORDS="$2"; shift 2 ;;
    --max-eval-records) MAX_EVAL_RECORDS="$2"; shift 2 ;;
    --eval-games) EVAL_GAMES="$2"; shift 2 ;;
    --eval-workers) EVAL_WORKERS="$2"; shift 2 ;;
    --eval-chunk-size) EVAL_CHUNK_SIZE="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --promote-path) PROMOTE_PATH="$2"; shift 2 ;;
    --promotion-threshold) PROMOTION_THRESHOLD="$2"; shift 2 ;;
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

mkdir -p "${SELF_PLAY_DIR}" "${CHECKPOINT_DIR}" "${EVAL_DIR}"

SELF_PLAY_CMD=(
  python3 scripts/run_standard_self_play.py
  --games "${GAMES}"
  --workers "${WORKERS}"
  --chunk-size "${CHUNK_SIZE}"
  --max-actions-per-game "${MAX_ACTIONS_PER_GAME}"
  --max-depth "${MAX_DEPTH}"
  --beam-width "${BEAM_WIDTH}"
  --opponent-branch-width "${OPPONENT_BRANCH_WIDTH}"
  --output-dir "${SELF_PLAY_DIR}"
)
if [[ -n "${SELF_PLAY_SEED}" ]]; then
  SELF_PLAY_CMD+=(--seed "${SELF_PLAY_SEED}")
fi

TRAIN_CMD=(
  python3 scripts/train_standard_model.py
  --input-dir "${SELF_PLAY_DIR}"
  --output-dir "${CHECKPOINT_DIR}"
  --device "${TRAIN_DEVICE}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --value-loss-weight "${VALUE_LOSS_WEIGHT}"
  --log-every "${LOG_EVERY}"
  --save-every "${SAVE_EVERY}"
  --eval-batches "${EVAL_BATCHES}"
  --validation-mod "${VALIDATION_MOD}"
  --validation-bucket "${VALIDATION_BUCKET}"
)
if [[ -n "${MAX_TRAIN_RECORDS}" ]]; then
  TRAIN_CMD+=(--max-train-records "${MAX_TRAIN_RECORDS}")
fi
if [[ -n "${MAX_EVAL_RECORDS}" ]]; then
  TRAIN_CMD+=(--max-eval-records "${MAX_EVAL_RECORDS}")
fi

CANDIDATE_CHECKPOINT="${CHECKPOINT_DIR}/final.pt"
EVAL_CMD=(
  python3 scripts/evaluate_standard_checkpoints.py
  --candidate "${CANDIDATE_CHECKPOINT}"
  --games "${EVAL_GAMES}"
  --workers "${EVAL_WORKERS}"
  --chunk-size "${EVAL_CHUNK_SIZE}"
  --max-actions-per-game "${MAX_ACTIONS_PER_GAME}"
  --max-depth "${MAX_DEPTH}"
  --beam-width "${BEAM_WIDTH}"
  --opponent-branch-width "${OPPONENT_BRANCH_WIDTH}"
  --output-dir "${EVAL_DIR}"
  --promote-path "${PROMOTE_PATH}"
  --promotion-threshold "${PROMOTION_THRESHOLD}"
)
if [[ -n "${BASELINE}" ]]; then
  EVAL_CMD+=(--baseline "${BASELINE}")
fi
if [[ -n "${EVAL_SEED}" ]]; then
  EVAL_CMD+=(--seed "${EVAL_SEED}")
fi

echo "[pipeline] project_root=${PROJECT_ROOT}"
echo "[pipeline] run_id=${RUN_ID}"
echo "[pipeline] self_play_dir=${SELF_PLAY_DIR}"
echo "[pipeline] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[pipeline] evaluation_dir=${EVAL_DIR}"
echo "[pipeline] promote_path=${PROMOTE_PATH}"

echo "[pipeline] starting self-play"
"${SELF_PLAY_CMD[@]}"

echo "[pipeline] starting training"
"${TRAIN_CMD[@]}"

echo "[pipeline] starting evaluation"
"${EVAL_CMD[@]}"

echo "[pipeline] complete"
echo "[pipeline] candidate_checkpoint=${CANDIDATE_CHECKPOINT}"
echo "[pipeline] promoted_checkpoint=${PROMOTE_PATH}"
