#!/bin/bash
#SBATCH --job-name=generate_dataset_job
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=14-17

# Load conda and activate env
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH

# Run your script with the array index as user ID
python generate.py --name user${SLURM_ARRAY_TASK_ID} --sample_size 7000

