#!/bin/bash
#SBATCH --job-name=generate_toy_data
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=2,3,4,5,6,7

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python toy_generate.py --name user${SLURM_ARRAY_TASK_ID} --sample_size=10000 --split train
