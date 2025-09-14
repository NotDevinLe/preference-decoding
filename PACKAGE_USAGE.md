# Gumbel Preference Decoding Package Usage

## Installation

### Development Installation
```bash
# From the project root directory
pip install -e .
```

### Regular Installation  
```bash
pip install .
```

## Package Structure

```
gumbel/
├── __init__.py          # Main package
├── core/                # Core components
│   ├── __init__.py
│   ├── collector_server.py  # Collector server
│   ├── learner_server.py    # Learner server
│   ├── coordinator.py       # Coordinator
│   └── sampler.py           # Data sampler
├── utils/               # Utility functions
│   ├── __init__.py
│   └── async_utils.py       # Async utilities
├── scripts/             # Scripts and models
│   ├── __init__.py
│   └── skeleton.py          # SparseMaskModel
└── tests/               # Test modules
    ├── __init__.py
    └── test_collector.py    # Collector tests
```

## Usage Examples

### Import Core Components
```python
# Import the main package
import gumbel

# Import specific components
from gumbel.core import DataSampler
from gumbel.utils import async_utils, get_log_probs_async, build_full_prompt
```

### Running Servers

#### Collector Server
```python
# Method 1: Run directly
python -m gumbel.core.collector_server --help

# Method 2: Use console script (after installation)
gumbel-collector --help
```

#### Learner Server  
```python
# Method 1: Run directly
python -m gumbel.core.learner_server --help

# Method 2: Use console script (after installation)
gumbel-learner --help
```

#### Test Collector Performance
```python
# Method 1: Run directly
python -m gumbel.tests.test_collector --help

# Method 2: Use console script (after installation) 
gumbel-test-collector --help
```

### Using Async Utilities
```python
import asyncio
import aiohttp
from transformers import AutoTokenizer
from gumbel.utils import get_log_probs_async, build_full_prompt, compute_drift_rewards

async def example():
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample data
    system_prompts = ["You are a helpful assistant."]
    user_prompts = ["What is AI?"]
    completions = ["AI stands for Artificial Intelligence."]
    
    # Get log probabilities
    async with aiohttp.ClientSession() as session:
        log_probs, token_counts = await get_log_probs_async(
            session=session,
            tokenizer=tokenizer,
            system_prompts=system_prompts,
            user_prompts=user_prompts,
            completion_texts=completions,
            vllm_url="http://localhost:8000/v1/completions",
            model_id="meta-llama/Llama-3.2-1B-Instruct"
        )
    
    print(f"Log probabilities: {log_probs}")
    print(f"Token counts: {token_counts}")

# Run the example
asyncio.run(example())
```

### Using Data Sampler
```python
from gumbel.core import DataSampler
import torch

# Initialize sampler
sampler = DataSampler(dataset_path="path/to/your/dataset.pkl")

# Sample data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
user_data = sampler(
    users_per_batch=8,
    samples_per_user=4,
    device=device
)

print(f"Sampled {len(user_data['prompts'])} prompts")
print(f"Sampled {len(user_data['outputs'])} outputs")
```

## Console Scripts

After installation with `pip install -e .`, these console scripts are available:

- `gumbel-collector`: Start the collector server
- `gumbel-learner`: Start the learner server  
- `gumbel-coordinator`: Start the coordinator
- `gumbel-test-collector`: Test collector performance

## Configuration

The package uses environment variables and command-line arguments for configuration:

- `COLLECTOR_CONCURRENCY`: Set concurrency limit for collector (default: 256)
- Standard FastAPI/uvicorn configurations for servers
- Custom VLLM server URLs and model names via parameters

## Example Complete Setup

```bash
# 1. Install the package
pip install -e .

# 2. Start VLLM server (separate terminal)
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000

# 3. Start collector server (separate terminal)
gumbel-collector \
  --d 100 \
  --dataset-path data/your_dataset.pkl \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --vllm-server-url http://localhost:8000 \
  --attribute-prompts-path configs/attribute_prompts.json \
  --port 8001

# 4. Test collector performance (separate terminal)
gumbel-test-collector \
  --collector-url http://localhost:8001 \
  --users-per-batch 16 \
  --samples-per-user 8 \
  --max-batches 50
```

## Development

For development, the package structure allows easy imports and testing:

```python
# Test import functionality
python test_imports.py

# Run individual servers
python -c "from gumbel.core.collector_server import main; main()"

# Use utils in scripts
from gumbel.utils import async_utils
# ... use async_utils functions
```