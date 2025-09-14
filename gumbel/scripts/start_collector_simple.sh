#!/bin/bash

# Simple collector server start script
# Usage: ./start_collector_simple.sh

# Set default values
D=${D:-100}
DATASET_PATH=${DATASET_PATH:-"gumbel/data/persona_train_dataset.pkl"}
ATTRIBUTE_PROMPTS_PATH=${ATTRIBUTE_PROMPTS_PATH:-"gumbel/configs/attribute_prompts.json"}
VLLM_SERVER_URL=${VLLM_SERVER_URL:-"http://localhost:8000"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}
DEVICE=${DEVICE:-"cuda:0"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo "== $(date) =="
echo "Starting Collector Server..."
echo "Host: $(hostname)  SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo ""
echo "Configuration:"
echo "  d: $D"
echo "  dataset-path: $DATASET_PATH"
echo "  attribute-prompts-path: $ATTRIBUTE_PROMPTS_PATH"
echo "  vllm-server-url: $VLLM_SERVER_URL"
echo "  model-name: $MODEL_NAME"
echo "  host: $HOST"
echo "  port: $PORT"
echo "  device: $DEVICE"
echo "  log-level: $LOG_LEVEL"
echo ""

# Build and execute command
exec python -u -m gumbel.core.collector_server \
    --d "$D" \
    --dataset-path "$DATASET_PATH" \
    --attribute-prompts-path "$ATTRIBUTE_PROMPTS_PATH" \
    --vllm-server-url "$VLLM_SERVER_URL" \
    --model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --log-level "$LOG_LEVEL"