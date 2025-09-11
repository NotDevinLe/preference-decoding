#!/bin/bash

# Start Collector Server using config.json
# Usage: ./start_collector_config.sh [--config CONFIG_FILE] [overrides...]

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
if ! python load_config.py --config "$CONFIG_FILE" --component collector >/dev/null 2>&1; then
    echo "Error: Failed to load config file. Please check the format."
    exit 1
fi

# Extract values from config using Python
eval "$(python -c "
import sys
sys.path.append('.')
from load_config import load_config, get_collector_args

try:
    config = load_config('$CONFIG_FILE')
    args = get_collector_args(config)
    
    for key, value in args.items():
        if isinstance(value, str):
            print(f'{key.upper()}=\"{value}\"')
        else:
            print(f'{key.upper()}=\"{value}\"')
except Exception as e:
    print(f'echo \"Error loading config: {e}\"', file=sys.stderr)
    print('exit 1', file=sys.stderr)
")"

# Build command with config values
CMD=(python -u collector_server.py
    --d "$D"
    --dataset-path "$DATASET_PATH"
    --attribute-prompts-path "$ATTRIBUTE_PROMPTS_PATH"
    --vllm-server-url "$VLLM_SERVER_URL"
    --model-name "$MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --device "$DEVICE"
    --log-level "$LOG_LEVEL"
)

# Override with command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --d) CMD[2]="$2"; shift 2 ;;
        --dataset-path) CMD[4]="$2"; shift 2 ;;
        --attribute-prompts-path) CMD[6]="$2"; shift 2 ;;
        --vllm-server-url) CMD[8]="$2"; shift 2 ;;
        --model-name) CMD[10]="$2"; shift 2 ;;
        --host) CMD[12]="$2"; shift 2 ;;
        --port) CMD[14]="$2"; shift 2 ;;
        --device) CMD[16]="$2"; shift 2 ;;
        --log-level) CMD[18]="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--config CONFIG_FILE] [overrides...]"
            echo ""
            echo "Config file: $CONFIG_FILE"
            echo "Available overrides:"
            echo "  --d NUM                      Number of attributes"
            echo "  --dataset-path PATH          Dataset path"
            echo "  --attribute-prompts-path PATH Attribute prompts file"
            echo "  --vllm-server-url URL        VLLM server URL"
            echo "  --model-name MODEL           Model name for API requests"
            echo "  --host HOST                  Server host"
            echo "  --port NUM                   Server port"
            echo "  --device DEVICE              CUDA device"
            echo "  --log-level LEVEL            Log level"
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

# Execute
exec "${CMD[@]}"