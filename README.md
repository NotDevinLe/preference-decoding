# Preference Decoding

A framework for aligning language models with user preferences through drift-based decoding and persona evaluation.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/preference-decoding.git
cd preference-decoding

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Setting up VLLM Server

This project uses a VLLM server for LLM-based evaluations. Start your server:

```bash
# Example: Start VLLM with Llama-3.3-70B
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2

# Set environment variables
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"
```

## 📊 Running Evaluations

### 1. Generate Persona Evaluations

Evaluate responses using the persona rubric:

```bash
python scripts/evaluate/run_persona_evaluation.py \
    --data_path data/bon.json \
    --output_path results/evaluations/persona_scores.jsonl \
    --max_outputs 20 \
    --workers 4
```

### 2. Run BON (Best-of-N) Comparison

Compare different selection methods:

```bash
python scripts/evaluate/run_bon_comparison.py \
    --data_path data/bon.json \
    --persona_eval_path results/evaluations/persona_scores.jsonl \
    --n_values 5,10,20,50,100 \
    --plot
```

### 3. Analyze Results

Generate analysis reports:

```bash
python src/evaluation/metrics/persona_analysis.py \
    --input results/evaluations/persona_scores.jsonl \
    --report results/reports/analysis.txt \
    --plots
```

## 🏗️ Project Structure

```
preference-decoding/
├── src/                      # Main source code
│   ├── core/                 # Core algorithms (drift, MLE, etc.)
│   ├── evaluation/           # Evaluation methods
│   │   ├── bon/             # Best-of-N evaluations
│   │   ├── judges/          # LLM judges and scoring
│   │   └── metrics/         # Analysis metrics
│   ├── generation/          # Data generation
│   └── models/              # Model interfaces
├── scripts/                  # Executable scripts
│   ├── evaluate/            # Evaluation scripts
│   ├── generate/            # Generation scripts
│   └── analysis/            # Analysis scripts
├── data/                    # Data files
├── results/                 # Output results
└── configs/                 # Configuration files
```

## 🔑 Key Features

### Persona Evaluation Rubric

The system uses a 5-dimensional rubric for evaluating persona adherence:

1. **Speaking Style & Voice** (1-5): Distinctive vocabulary and tone
2. **Personality Traits** (1-5): Core personality integration
3. **Knowledge & Interests** (1-5): Persona-specific knowledge
4. **Behavioral Consistency** (1-5): Character-consistent actions
5. **Emotional Authenticity** (1-5): Genuine emotional responses

### Evaluation Methods

- **Drift-based Selection**: Uses preference drift to select outputs
- **Persona-based Selection**: Selects based on persona rubric scores
- **Random Baseline**: Random selection for comparison
- **Oracle Selection**: Best possible selection (upper bound)

## 📈 Example Results

After running evaluations, you'll get:

1. **Detailed Scores**: Individual scores for each dimension
2. **Comparisons**: Performance across different N values
3. **Analysis Reports**: Statistical analysis and correlations
4. **Visualizations**: Plots comparing methods

Example output:
```
BON Performance (N=20):
- Persona-based: 4.23 ± 0.45
- Drift-based: 3.89 ± 0.52  
- Random: 3.41 ± 0.61
- Oracle: 4.67 ± 0.31
```

## 🛠️ Advanced Usage

### Custom Personas

Override persona detection with custom personas:

```bash
python scripts/evaluate/run_persona_evaluation.py \
    --persona "A helpful but sarcastic AI assistant" \
    --data_path data/custom.json
```

### Async Mode for Speed

Use async evaluation for faster processing:

```bash
python scripts/evaluate/run_persona_evaluation.py \
    --async_mode \
    --workers 10
```

### Specific Dimension Focus

Evaluate based on specific dimensions:

```python
from src.evaluation.bon.persona_bon import evaluate_with_persona_scores

results = evaluate_with_persona_scores(
    bon_data,
    evaluations,
    n_values=[10, 20, 50],
    score_dimension='personality'  # Focus on personality dimension
)
```

## 📚 Documentation

- [API Documentation](docs/API.md) - Detailed API reference
- [Experiment Guide](docs/EXPERIMENTS.md) - Running experiments
- [Project Structure](STRUCTURE.md) - Detailed structure explanation

## 🧪 Testing

Run tests:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_llm_judge.py

# With coverage
pytest --cov=src tests/
```

## 📦 Dependencies

Main dependencies:
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers
- `vllm` - Fast LLM inference
- `numpy`, `pandas` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `aiohttp` - Async HTTP client

See `requirements.txt` for full list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- VLLM team for the fast inference server
- Hugging Face for transformers library
- Anthropic for Claude

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.