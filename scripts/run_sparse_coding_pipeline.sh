#!/bin/bash
# Run the complete sparse coding pipeline for persona selection

# Set default values
DATA_FILE="data/preference/user1_train.json"
QUESTIONS_FILE="data/questions.json"
OUTPUT_DIR="results/sparse_coding"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_QUESTIONS=100
NUM_PERSONAS=100
QUESTION_SOURCE="dolly"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --questions-file)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --question-source)
            QUESTION_SOURCE="$2"
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
        --skip-questions)
            SKIP_QUESTIONS=1
            shift
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
echo "Questions file: $QUESTIONS_FILE"
echo "Question source: $QUESTION_SOURCE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Questions: $NUM_QUESTIONS"
echo "Personas: $NUM_PERSONAS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 0: Prepare questions
if [ -z "$SKIP_QUESTIONS" ]; then
    echo "Step 0: Preparing questions..."
    echo "------------------------------"
    
    if [ "$QUESTION_SOURCE" = "dolly" ]; then
        python scripts/generate/prepare_questions.py \
            --source dolly \
            --output-file "$QUESTIONS_FILE" \
            --num-questions "$NUM_QUESTIONS" \
            --seed 42
    else
        python scripts/generate/prepare_questions.py \
            --source existing \
            --input-file "$DATA_FILE" \
            --output-file "$QUESTIONS_FILE" \
            --num-questions "$NUM_QUESTIONS"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Question preparation failed"
        exit 1
    fi
    echo "✓ Questions prepared"
else
    echo "Step 0: Skipping question preparation (--skip-questions flag)"
fi

echo ""

# Step 1: Generate persona responses
if [ -z "$SKIP_GENERATION" ]; then
    echo "Step 1: Generating persona responses..."
    echo "----------------------------------------"
    
    python scripts/generate/generate_persona_data.py \
        --data-file "$QUESTIONS_FILE" \
        --output-file "data/persona_responses.json" \
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
    --persona-file "data/persona_responses.json" \
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