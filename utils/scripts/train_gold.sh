#!/bin/bash
#SBATCH --job-name=train_reward_model
#SBATCH --account=ark
#SBATCH --partition=gpu-a40
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=12-17

# Load conda and activate env
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH


# Run your script with the array index as user ID
python train_and_eval.py --name user${SLURM_ARRAY_TASK_ID} --model_path reward_model_user${SLURM_ARRAY_TASK_ID}
