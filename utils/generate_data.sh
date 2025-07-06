#!/bin/bash
#SBATCH --job-name=generate_dataset_job
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=0-3   # 4 jobs: for user0, user1, user2, user3

# Load conda and activate env
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH

# Optional HF cache
export HF_HOME="/gscratch/ark/devinl6/hf_cache"

# Run your script with the array index as user ID
python generate.py --name user${SLURM_ARRAY_TASK_ID}

