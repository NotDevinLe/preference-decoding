# Preference Decoding - Project Structure

## Proposed Organization

```
preference-decoding/
│
├── src/                           # Main source code
│   ├── __init__.py
│   ├── core/                      # Core algorithms and models
│   │   ├── __init__.py
│   │   ├── drift.py              # Drift algorithm implementation
│   │   ├── mle.py                # MLE optimization
│   │   └── l1_regularization.py  # L1 regularization utilities
│   │
│   ├── evaluation/               # Evaluation methods
│   │   ├── __init__.py
│   │   ├── bon/                  # Best-of-N evaluations
│   │   │   ├── __init__.py
│   │   │   ├── drift_bon.py
│   │   │   ├── persona_bon.py
│   │   │   └── random_bon.py
│   │   ├── judges/               # LLM judges and scoring
│   │   │   ├── __init__.py
│   │   │   ├── llm_judge.py
│   │   │   ├── persona_rubric.py
│   │   │   └── golden_reward.py
│   │   └── metrics/              # Evaluation metrics
│   │       ├── __init__.py
│   │       └── analysis.py
│   │
│   ├── generation/               # Data generation
│   │   ├── __init__.py
│   │   ├── bon_generate.py
│   │   └── synthetic.py
│   │
│   └── models/                   # Model interfaces
│       ├── __init__.py
│       ├── vllm_client.py       # VLLM server interface
│       └── reward_model.py
│
├── scripts/                      # Executable scripts
│   ├── evaluate/                 # Evaluation scripts
│   │   ├── run_bon_evaluation.py
│   │   ├── run_persona_evaluation.py
│   │   └── compare_methods.py
│   ├── generate/                 # Generation scripts
│   │   ├── generate_bon_data.py
│   │   └── generate_evaluations.py
│   └── analysis/                 # Analysis scripts
│       ├── analyze_results.py
│       └── plot_results.py
│
├── configs/                      # Configuration files
│   ├── model_configs.yaml
│   ├── evaluation_configs.yaml
│   └── personas.yaml
│
├── data/                        # Data files
│   ├── raw/                     # Raw input data
│   ├── processed/               # Processed data
│   └── cache/                   # Cached evaluations
│
├── results/                     # Output results
│   ├── evaluations/             # Evaluation results
│   ├── plots/                   # Generated plots
│   └── reports/                 # Analysis reports
│
├── tests/                       # Unit tests
│   ├── test_drift.py
│   ├── test_llm_judge.py
│   └── test_evaluations.py
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── exploration.ipynb
│   └── visualization.ipynb
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── API.md
│   └── EXPERIMENTS.md
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .gitignore
└── README.md                    # Main documentation
```

## Current Issues to Fix

1. **Flat utils directory**: Everything is dumped in utils/ with unclear organization
2. **Mixed concerns**: Evaluation, generation, and analysis code are mixed together
3. **Duplicate/similar files**: Multiple BON files with slight variations (drift_bon.py, drift_bon_with_gold.py, etc.)
4. **Unclear naming**: Files like "temp.py", "test_rm.py" in production directories
5. **Scattered scripts**: Important scripts buried in subdirectories
6. **No clear entry points**: Hard to know where to start or how to run evaluations
7. **Missing dependencies list**: No requirements.txt file

## Migration Plan

### Phase 1: Core Reorganization
1. Create new directory structure under `src/`
2. Move core algorithms to `src/core/`
3. Consolidate evaluation methods in `src/evaluation/`

### Phase 2: Script Organization
1. Create clear entry point scripts in `scripts/`
2. Each script should have clear documentation and CLI interface
3. Remove duplicate/temporary files

### Phase 3: Data and Results
1. Organize data into raw/processed/cache subdirectories
2. Separate results by type (evaluations/plots/reports)
3. Add .gitignore entries for cache and generated files

### Phase 4: Documentation
1. Create comprehensive README with quick start guide
2. Document API for key modules
3. Add docstrings to all functions

### Phase 5: Testing and CI
1. Add unit tests for core functionality
2. Create integration tests for evaluation pipelines
3. Set up basic CI/CD if needed

## Benefits of New Structure

1. **Clear separation of concerns**: Core logic, evaluation, generation are separate
2. **Easy navigation**: Clear where to find specific functionality
3. **Reusability**: Core modules can be imported cleanly
4. **Maintainability**: Easier to update and extend
5. **Onboarding**: New developers can understand the project quickly
6. **Testing**: Clear structure for unit and integration tests