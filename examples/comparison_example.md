# Response Comparison Script Usage

This script (`scripts/evaluate/compare_generated_responses.py`) compares pre-generated responses from multiple methods using an LLM judge.

## Features

- **No model loading**: Only loads pre-generated JSON response files
- **Dynamic method support**: Can compare any number of methods
- **Comprehensive judging**: Uses single LLM prompt to evaluate all methods simultaneously
- **Online VLLM serving**: Uses HTTP API calls to VLLM server
- **Caching**: Avoids re-evaluating identical comparisons
- **Detailed analysis**: Provides rankings, scores, win rates, and reasoning

## Expected Input Format

Response files should be JSON with this structure:
```json
{
  "method": "Method-Name",
  "prompts": ["prompt1", "prompt2", "prompt3"],
  "responses": ["response1", "response2", "response3"],
  "parameters": { /* method-specific params */ }
}
```

## Usage Examples

### Basic Comparison
```bash
python scripts/evaluate/compare_generated_responses.py \
    --response_files results/bon_drift_responses.json results/qalign_drift_responses.json \
    --method_names "BON-Drift" "QAlign-Drift" \
    --judge_model meta-llama/Llama-3.3-70B-Instruct \
    --judge_base_url http://localhost:8000/v1 \
    --output_path results/comparison_results.json
```

### Multiple Methods with Custom Persona
```bash
python scripts/evaluate/compare_generated_responses.py \
    --response_files \
        results/bon_drift_responses.json \
        results/bon_mle_responses.json \
        results/qalign_drift_responses.json \
        results/qalign_mle_responses.json \
    --method_names "BON-Drift" "BON-MLE" "QAlign-Drift" "QAlign-MLE" \
    --judge_model meta-llama/Llama-3.3-70B-Instruct \
    --judge_base_url http://localhost:8000/v1 \
    --persona "A knowledgeable and concise assistant that provides accurate information" \
    --output_path results/four_method_comparison.json
```

### Test Run (Limited Prompts)
```bash
python scripts/evaluate/compare_generated_responses.py \
    --response_files examples/sample_responses_method1.json examples/sample_responses_method2.json \
    --method_names "BON-Drift" "QAlign-Drift" \
    --judge_base_url http://localhost:8000/v1 \
    --max_prompts 3 \
    --output_path results/test_comparison.json
```

## Output

The script produces:
1. **Console output**: Formatted comparison table with rankings and statistics
2. **JSON results**: Detailed results including individual judgments and reasoning
3. **Cache**: Stored judgments to avoid re-evaluation

### Sample Console Output
```
================================================================================
COMPARISON RESULTS
================================================================================
Method               Mean Score   Mean Rank    Win Rate     Wins    
--------------------------------------------------------------------------------
QAlign-Drift         7.85±1.23    1.20         65.0%        13      
BON-Drift           7.12±1.45     1.80         35.0%        7       
================================================================================
Total comparisons: 20
```

## Judge Configuration

The script uses an LLM judge that:
- Evaluates responses on relevance, accuracy, helpfulness, clarity, completeness, and persona alignment
- Provides rankings (1=best) and scores (1-10 scale) for each method
- Identifies a winner and provides detailed reasoning
- Caches results to avoid duplicate API calls

## Prerequisites

1. **VLLM Server**: Must be running and accessible at the specified URL
2. **Response Files**: Pre-generated using `scripts/generate/run_generation.py` or similar
3. **Matching Prompts**: All response files must contain responses to the same prompts in the same order

## Advanced Features

- **Caching**: Judgments are cached by prompt+responses hash
- **Error Handling**: Graceful fallback for API failures
- **Progress Tracking**: Shows comparison progress and cache hit rates
- **Flexible Input**: Handles different JSON formats automatically
- **Statistics**: Tracks API usage and cache efficiency