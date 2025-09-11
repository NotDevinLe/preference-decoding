#!/usr/bin/env bash
# Start Learner Server (FastAPI/uvicorn)

# Safety: avoid -u because some conda hooks choke on nounset
set -eo pipefail

# ===== Defaults (override with flags) =====
D=100
K=10
LR=1e-3
SPARSITY_WEIGHT=0.1
TAU_INIT=1.0
HOST="0.0.0.0"         # important for cross-node access / tunnels
PORT=8002
DEVICE="cuda:0"
CHECKPOINT_DIR="./checkpoints"
LOG_LEVEL="INFO"
USE_WANDB=false

# ===== Parse flags =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --d) D="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --sparsity-weight) SPARSITY_WEIGHT="$2"; shift 2 ;;
    --tau-init) TAU_INIT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --use-wandb) USE_WANDB=true; shift ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]
  --d N                  (default: $D)
  --k N                  (default: $K)
  --lr F                 (default: $LR)
  --sparsity-weight F    (default: $SPARSITY_WEIGHT)
  --tau-init F           (default: $TAU_INIT)
  --host HOST            (default: $HOST)
  --port PORT            (default: $PORT)
  --device DEV           (default: $DEVICE)
  --checkpoint-dir PATH  (default: $CHECKPOINT_DIR)
  --log-level LEVEL      (default: $LOG_LEVEL)
  --use-wandb            Enable wandb logging
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "== $(date) =="
echo "Host: $(hostname)  SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Python: $(command -v python)  ($(python -V 2>&1))"
nvidia-smi || true
echo "Params -> d=$D, k=$K, lr=$LR, sparsity=$SPARSITY_WEIGHT, tau=$TAU_INIT"
echo "Bind   -> $HOST:$PORT  device=$DEVICE  log=$LOG_LEVEL  ckpt=$CHECKPOINT_DIR  wandb=$USE_WANDB"
echo

mkdir -p "$CHECKPOINT_DIR"

# Ensure port is free
if command -v lsof >/dev/null 2>&1; then
  if lsof -i:"$PORT" >/dev/null 2>&1; then
    echo "Error: Port $PORT is already in use on $(hostname)." >&2
    exit 1
  fi
fi

# Build command (unbuffered -u so logs stream)
CMD=( python -u learner_server.py
  --d "$D"
  --k "$K"
  --lr "$LR"
  --sparsity-weight "$SPARSITY_WEIGHT"
  --tau-init "$TAU_INIT"
  --host "$HOST"
  --port "$PORT"
  --device "$DEVICE"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --log-level "$LOG_LEVEL"
)
[[ "$USE_WANDB" == "true" ]] && CMD+=( --use-wandb )

exec "${CMD[@]}"
