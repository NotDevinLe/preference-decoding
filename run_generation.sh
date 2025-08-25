#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python scripts/generate/run_generation.py --method "qalign-drift" --data_path "results/bon_outputs.json" --output_path "results/qalign_drift_responses.json" --base_model_path "meta-llama/Meta-Llama-3.1-8B-Instruct" --drift_model_path "meta-llama/Llama-3.2-1B-Instruct" --p_vector_path "results/drift_vector.json" --temperature 0.7 --qalign_steps 16 --max_length 512 --qalign_beta 1