#!/bin/bash

# Start Learner Server using config.json
# Usage: ./start_learner_config.sh [--config CONFIG_FILE] [overrides...]

CONFIG_FILE="config.json"

# Parse config file argument first
if [[ "$1" == "--config" ]]; then
    CONFIG_FILE="$2"
    shift 2
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Load configuration using Python helper
echo "Loading configuration from: $CONFIG_FILE"
if ! python load_config.py --config "$CONFIG_FILE" --component learner >/dev/null 2>&1; then
    echo "Error: Failed to load config file. Please check the format."
    exit 1
fi

# Extract values from config using Python
eval "$(python -c "
import sys
sys.path.append('.')
from load_config import load_config, get_learner_args

try:
    config = load_config('$CONFIG_FILE')
    args = get_learner_args(config)
    
    for key, value in args.items():
        if isinstance(value, str):
            print(f'{key.upper()}=\"{value}\"')
        elif isinstance(value, bool):
            print(f'{key.upper()}={\"true\" if value else \"false\"}')
        else:
            print(f'{key.upper()}=\"{value}\"')
except Exception as e:
    print(f'echo \"Error loading config: {e}\"', file=sys.stderr)
    print('exit 1', file=sys.stderr)
")"

# Build command with config values
CMD=(python -u learner_server.py
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

# Add wandb flag if enabled
if [[ "$USE_WANDB" == "true" ]]; then
    CMD+=(--use-wandb)
fi

# Override with command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --d) CMD[2]="$2"; shift 2 ;;
        --k) CMD[4]="$2"; shift 2 ;;
        --lr) CMD[6]="$2"; shift 2 ;;
        --sparsity-weight) CMD[8]="$2"; shift 2 ;;
        --tau-init) CMD[10]="$2"; shift 2 ;;
        --host) CMD[12]="$2"; shift 2 ;;
        --port) CMD[14]="$2"; shift 2 ;;
        --device) CMD[16]="$2"; shift 2 ;;
        --checkpoint-dir) CMD[18]="$2"; shift 2 ;;
        --log-level) CMD[20]="$2"; shift 2 ;;
        --use-wandb) 
            # Add if not already present
            if [[ ! " ${CMD[*]} " =~ " --use-wandb " ]]; then
                CMD+=(--use-wandb)
            fi
            shift ;;
        --help|-h)
            echo "Usage: $0 [--config CONFIG_FILE] [overrides...]"
            echo ""
            echo "Config file: $CONFIG_FILE"
            echo "Available overrides:"
            echo "  --d NUM                      Number of attributes"
            echo "  --k NUM                      Number of components"
            echo "  --lr FLOAT                   Learning rate"
            echo "  --sparsity-weight FLOAT      Sparsity weight"
            echo "  --tau-init FLOAT             Initial temperature"
            echo "  --host HOST                  Server host"
            echo "  --port NUM                   Server port"
            echo "  --device DEVICE              CUDA device"
            echo "  --checkpoint-dir PATH        Checkpoint directory"
            echo "  --log-level LEVEL            Log level"
            echo "  --use-wandb                  Enable wandb logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "== $(date) =="
echo "Host: $(hostname)  SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "Config: $CONFIG_FILE"
echo "Command: ${CMD[*]}"
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Execute
exec "${CMD[@]}"