#!/bin/bash
#SBATCH --job-name=run_ipcae
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
#SBATCH --array=40

# Load conda and activate env
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate ipcae
export PATH=/gscratch/ark/devinl6/envs/ipcae/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run your script with the array index as user ID
python run_ipcae_rewards.py --k ${SLURM_ARRAY_TASK_ID} --run_name "k${SLURM_ARRAY_TASK_ID}"
