#!/bin/bash

# Start Collector Server
# Usage: ./start_collector.sh [options]

# Default parameters
D=400
DATASET_PATH="../data/reward_matrix_flexible.npz"
ATTRIBUTE_PROMPTS_PATH="test_attribute_prompts.json"
VLLM_MODEL="meta-llama/Llama-3.2-1B-Instruct"
GPU_MEMORY_UTIL=0.4
HOST="localhost"
PORT=8001
DEVICE="cuda:0"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --d)
            D="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --attribute-prompts-path)
            ATTRIBUTE_PROMPTS_PATH="$2"
            shift 2
            ;;
        --vllm-model)
            VLLM_MODEL="$2"
            shift 2
            ;;
        --gpu-memory-util)
            GPU_MEMORY_UTIL="$2"
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
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --d NUM                      Number of attributes (default: $D)"
            echo "  --dataset-path PATH          Dataset path (default: $DATASET_PATH)"
            echo "  --attribute-prompts-path PATH Attribute prompts file (default: $ATTRIBUTE_PROMPTS_PATH)"
            echo "  --vllm-model MODEL           VLLM model name (default: $VLLM_MODEL)"
            echo "  --gpu-memory-util FLOAT      GPU memory utilization (default: $GPU_MEMORY_UTIL)"
            echo "  --port NUM                   Server port (default: $PORT)"
            echo "  --device DEVICE              CUDA device (default: $DEVICE)"
            echo "  --log-level LEVEL            Log level (default: $LOG_LEVEL)"
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

echo "Starting Collector Server..."
echo "Parameters:"
echo "  Attributes (d): $D"
echo "  Dataset: $DATASET_PATH"
echo "  Attribute Prompts: $ATTRIBUTE_PROMPTS_PATH"
echo "  VLLM Model: $VLLM_MODEL"
echo "  GPU Memory Util: $GPU_MEMORY_UTIL"
echo "  Port: $PORT"
echo "  Device: $DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo ""

# Check if dataset exists
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Check if attribute prompts exist
if [[ ! -f "$ATTRIBUTE_PROMPTS_PATH" ]]; then
    echo "Error: Attribute prompts file not found: $ATTRIBUTE_PROMPTS_PATH"
    exit 1
fi

# Start the collector server
exec python collector_server.py \
    --d "$D" \
    --dataset-path "$DATASET_PATH" \
    --attribute-prompts-path "$ATTRIBUTE_PROMPTS_PATH" \
    --vllm-model "$VLLM_MODEL" \
    --gpu-memory-util "$GPU_MEMORY_UTIL" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --log-level "$LOG_LEVEL"