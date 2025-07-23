#!/bin/bash
#SBATCH --job-name=approximation_job
#SBATCH --account=ark
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=6

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python find_user_p.py --name user${SLURM_ARRAY_TASK_ID} --save_path="../results/user_p_stable.jsonl"
