# Collector Server Usage Guide

## Prerequisites

1. **Install the package:**
```bash
pip install -e .
```

2. **Start a VLLM server:**
```bash
# In a separate terminal
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000
```

## Running the Collector Server

### Method 1: Using Console Scripts (Recommended)
```bash
# After pip install -e .
gumbel-collector \
  --d 100 \
  --dataset-path gumbel/data/persona_train_dataset.pkl \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --vllm-server-url http://localhost:8000 \
  --attribute-prompts-path gumbel/configs/attribute_prompts.json \
  --port 8001
```

### Method 2: Using Python Module
```bash
python -m gumbel.core.collector_server \
  --d 100 \
  --dataset-path gumbel/data/persona_train_dataset.pkl \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --vllm-server-url http://localhost:8000 \
  --attribute-prompts-path gumbel/configs/attribute_prompts.json \
  --port 8001
```

### Method 3: Using Configuration File
```bash
# Uses gumbel/configs/config.json by default
bash gumbel/scripts/start_collector_config.sh

# Or with custom config
bash gumbel/scripts/start_collector_config.sh --config my_config.json

# Override specific settings
bash gumbel/scripts/start_collector_config.sh --port 8002 --device cuda:1
```

### Method 4: Using Simple Shell Script
```bash
# Set environment variables
export VLLM_SERVER_URL="http://localhost:8000"
export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
export PORT=8001

# Run with environment defaults
bash gumbel/scripts/start_collector_simple.sh
```

## Configuration

### Config File Format (`gumbel/configs/config.json`)
```json
{
  "model": {
    "d": "auto",           // "auto" detects from attribute_prompts.json
    "k": 10,
    "lr": 1e-3,
    "sparsity_weight": 0.1,
    "tau_init": 1.0
  },
  "data": {
    "dataset_path": "gumbel/data/persona_train_dataset.pkl",
    "attribute_prompts_path": "gumbel/configs/attribute_prompts.json"
  },
  "vllm": {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "server_url": "http://localhost:8000",
    "gpu_memory_util": 0.6
  },
  "servers": {
    "collector": {
      "host": "0.0.0.0",
      "port": 8001,
      "device": "cuda:0"
    }
  },
  "monitoring": {
    "log_level": "INFO"
  }
}
```

### Environment Variables (for simple script)
```bash
export D=100                                    # Number of attributes
export DATASET_PATH="gumbel/data/persona_train_dataset.pkl"
export ATTRIBUTE_PROMPTS_PATH="gumbel/configs/attribute_prompts.json"
export VLLM_SERVER_URL="http://localhost:8000"
export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
export HOST="0.0.0.0"
export PORT=8001
export DEVICE="cuda:0"
export LOG_LEVEL="INFO"
```

## Testing the Collector

### Basic Health Check
```bash
curl http://localhost:8001/health
# Should return: {"status": "healthy"}
```

### Status Check
```bash
curl http://localhost:8001/status
# Should return: {"status": "running", "collections_served": 0}
```

### Performance Testing
```bash
# Method 1: Using console script
gumbel-test-collector --collector-url http://localhost:8001

# Method 2: Using Python module
python -m gumbel.tests.test_collector --collector-url http://localhost:8001

# Method 3: Custom parameters
python -m gumbel.tests.test_collector \
  --collector-url http://localhost:8001 \
  --users-per-batch 16 \
  --samples-per-user 8 \
  --max-batches 50
```

### Manual Test Request
```bash
curl -X POST http://localhost:8001/generate_batch \
  -H "Content-Type: application/json" \
  -d '{"users_per_batch": 4, "samples_per_user": 2}'
```

## Recent Fixes

### v2 Updates (2025-09-14)
- ✅ **Fixed FastAPI deprecation warning**: Updated from `@app.on_event("shutdown")` to lifespan handlers
- ✅ **Fixed async event loop error**: Moved aiohttp session creation to lifespan startup
- ✅ **Updated package imports**: All imports now use proper package structure

## Troubleshooting

### Common Issues

1. **Config format error:**
   ```bash
   # Test config loading
   python -m gumbel.utils.load_config --config gumbel/configs/config.json --component collector
   ```

2. **Import errors:**
   ```bash
   # Test package installation
   python test_imports.py
   ```

3. **VLLM connection issues:**
   - Check if VLLM server is running on the specified URL
   - Verify the model name matches what's loaded in VLLM
   - Check firewall/network connectivity

4. **Dataset not found:**
   - Verify the dataset path exists
   - Use absolute paths if relative paths don't work

5. **Attribute prompts not found:**
   - Check if the attribute prompts file exists
   - Verify the JSON format is correct (array or object with "prompts" key)

### Debugging

1. **Increase log level:**
   ```bash
   gumbel-collector --log-level DEBUG ...
   ```

2. **Check server logs:**
   - Look for error messages in the console output
   - Check for timeout issues with VLLM

3. **Test individual components:**
   ```python
   # Test async utilities directly
   python -c "
   import asyncio
   from gumbel.utils import get_log_probs_async
   # ... test code
   "
   ```

## Performance Tuning

### Concurrency Settings
```bash
# Set environment variable for collector concurrency
export COLLECTOR_CONCURRENCY=512

# Then start collector
gumbel-collector ...
```

### Batch Size Optimization
- Start with smaller batches (4-8 users, 2-4 samples)
- Gradually increase based on GPU memory and performance
- Monitor for timeout errors

### Network Optimization
- Use local VLLM server when possible
- Consider using faster network connections for remote VLLM
- Monitor network latency with the test script