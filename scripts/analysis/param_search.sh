#!/bin/bash
#SBATCH --job-name=param_search_gumbel
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt

# Load conda and activate env
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache
export WANDB_PROJECT=gumbel

# Run your script with the array index as user ID
python parameter_search_gumbel.py
