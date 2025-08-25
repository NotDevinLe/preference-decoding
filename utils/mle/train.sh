#!/bin/bash
#SBATCH --job-name=train_mle
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=1

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

python train_mle.py --name user3 --num_expectation_samples 8 --num_mc_samples 8 --use_wandb --sample_size 200 --max_epochs 10000 --attribute_indices '0,1,2,31,33,37,43' --global_lambda 0.01