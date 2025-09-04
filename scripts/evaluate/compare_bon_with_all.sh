#!/bin/bash#SBATCH --job-name=compare_bon_with_all
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=1

# Load conda and activate env
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache
export WANDB_PROJECT=Alignment

# Run your script with the array index as user ID
python scripts/evaluate/compare_bon_with_all.py --name_idx ${SLURM_ARRAY_TASK_ID}
