#!/bin/bash
# Run the complete sparse coding pipeline for persona selection

# Set default values
DATA_FILE="data/preference/user1_train.json"
OUTPUT_DIR="results/sparse_coding"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_QUESTIONS=100
NUM_PERSONAS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-questions)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --num-personas)
            NUM_PERSONAS="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=1
            shift
            ;;
        --skip-reward)
            SKIP_REWARD=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "SPARSE CODING PIPELINE FOR PERSONA SELECTION"
echo "=========================================="
echo "Data file: $DATA_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Questions: $NUM_QUESTIONS"
echo "Personas: $NUM_PERSONAS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate persona responses
if [ -z "$SKIP_GENERATION" ]; then
    echo "Step 1: Generating persona responses..."
    echo "----------------------------------------"
    
    python scripts/generate/generate_persona_data.py \
        --data-file "$DATA_FILE" \
        --output-dir "data/persona_responses" \
        --model-name "$MODEL" \
        --num-questions "$NUM_QUESTIONS" \
        --num-personas "$NUM_PERSONAS" \
        --batch-size 8 \
        --temperature 0.7 \
        --resume
    
    if [ $? -ne 0 ]; then
        echo "Error: Persona generation failed"
        exit 1
    fi
    echo "✓ Persona responses generated"
else
    echo "Step 1: Skipping persona generation (--skip-generation flag)"
fi

echo ""

# Step 2: Run sparse coding experiments
echo "Step 2: Running sparse coding experiments..."
echo "--------------------------------------------"

python scripts/analysis/run_sparse_coding.py \
    --persona-dir "data/persona_responses" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL" \
    --base-prompt "You are a helpful assistant." \
    --sweep

if [ $? -ne 0 ]; then
    echo "Error: Sparse coding experiments failed"
    exit 1
fi

echo "✓ Sparse coding complete"
echo ""

# Step 3: Generate report
echo "Step 3: Generating analysis report..."
echo "-------------------------------------"

# Check if results exist
if [ -f "$OUTPUT_DIR/all_experiments.json" ]; then
    echo "Results summary:"
    echo ""
    
    # Parse and display key results using Python
    python -c "
import json
with open('$OUTPUT_DIR/all_experiments.json', 'r') as f:
    experiments = json.load(f)
    
# Find best experiment
best = min(experiments, key=lambda x: x['final_error'])
print(f'Best configuration:')
print(f'  k={best[\"parameters\"][\"k\"]}')
print(f'  λ₁={best[\"parameters\"][\"lambda1\"]}')
print(f'  λ₂₁={best[\"parameters\"][\"lambda21\"]}')
print(f'  Error: {best[\"final_error\"]:.4f}')
print(f'  Sparsity: {best[\"final_sparsity\"]:.3f}')
print(f'  Selected: {best[\"num_selected\"]} personas')
    "
    
    echo ""
    echo "Visualizations saved to:"
    echo "  - $OUTPUT_DIR/sparse_coding_results.png"
    echo "  - $OUTPUT_DIR/persona_weights.png"
    echo ""
    echo "Analysis saved to:"
    echo "  - $OUTPUT_DIR/persona_selection_analysis.json"
    echo "  - $OUTPUT_DIR/all_experiments.json"
else
    echo "Warning: No results found"
fi

echo ""
echo "=========================================="
echo "✓ PIPELINE COMPLETE"
echo "=========================================="
echo "All results saved to: $OUTPUT_DIR"