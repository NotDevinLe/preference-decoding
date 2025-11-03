#!/bin/bash
#SBATCH --job-name=train_ipcae
#SBATCH --account=ark
#SBATCH --partition=gpu-a40
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=5,10,20,30

# Load conda and activate env
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate ipcae

# Run your script with the array index as user ID
python main_pl.py --config ../configs/rewards_k${SLURM_ARRAY_TASK_ID}_o400.yaml