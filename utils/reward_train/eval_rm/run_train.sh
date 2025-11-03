#!/bin/bash
#SBATCH --job-name=run_llamafactory_users
#SBATCH --account=ark
#SBATCH --partition=gpu-a40
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=8,9,10,11,12,13,14

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python utils/reward_train/eval_rm/run_llamafactory_users.py --user ${SLURM_ARRAY_TASK_ID}
