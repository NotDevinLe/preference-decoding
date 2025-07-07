#!/bin/bash
#SBATCH --job-name=generate_dataset_job
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=5-8

# Load conda and activate env
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH

# Run your script with the array index as user ID
python vllm_generate.py --name user${SLURM_ARRAY_TASK_ID} --sample_size 7000

