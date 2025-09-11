#!/bin/bash
#SBATCH --job-name=collector
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- setup ---
set -eo pipefail
mkdir -p logs

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# --- run your server ---
./start_collector_config.sh
