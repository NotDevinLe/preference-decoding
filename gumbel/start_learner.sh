#!/bin/bash

# Start Learner Server
# Usage: ./start_learner.sh [options]

# Default parameters
D=400
K=50
LR=0.001
SPARSITY_WEIGHT=0.0
TAU_INIT=1.0
HOST="localhost"
PORT=8002
DEVICE="cuda:1"
CHECKPOINT_DIR="./checkpoints"
LOG_LEVEL="INFO"
USE_WANDB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --d)
            D="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --sparsity-weight)
            SPARSITY_WEIGHT="$2"
            shift 2
            ;;
        --tau-init)
            TAU_INIT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --d NUM                      Number of attributes (default: $D)"
            echo "  --k NUM                      Number of components (default: $K)"
            echo "  --lr FLOAT                   Learning rate (default: $LR)"
            echo "  --sparsity-weight FLOAT      Sparsity weight (default: $SPARSITY_WEIGHT)"
            echo "  --tau-init FLOAT             Initial temperature (default: $TAU_INIT)"
            echo "  --port NUM                   Server port (default: $PORT)"
            echo "  --device DEVICE              CUDA device (default: $DEVICE)"
            echo "  --checkpoint-dir PATH        Checkpoint directory (default: $CHECKPOINT_DIR)"
            echo "  --log-level LEVEL            Log level (default: $LOG_LEVEL)"
            echo "  --use-wandb                  Enable wandb logging"
            echo "  --help, -h                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting Learner Server..."
echo "Parameters:"
echo "  Attributes (d): $D"
echo "  Components (k): $K"
echo "  Learning Rate: $LR"
echo "  Sparsity Weight: $SPARSITY_WEIGHT"
echo "  Initial Temperature: $TAU_INIT"
echo "  Port: $PORT"
echo "  Device: $DEVICE"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "  Log Level: $LOG_LEVEL"
echo "  Use Wandb: $USE_WANDB"
echo ""

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Build command
CMD="python learner_server.py \
    --d $D \
    --k $K \
    --lr $LR \
    --sparsity-weight $SPARSITY_WEIGHT \
    --tau-init $TAU_INIT \
    --host $HOST \
    --port $PORT \
    --device $DEVICE \
    --checkpoint-dir $CHECKPOINT_DIR \
    --log-level $LOG_LEVEL"

# Add wandb flag if enabled
if [[ "$USE_WANDB" == "true" ]]; then
    CMD="$CMD --use-wandb"
fi

# Start the learner server
exec $CMD