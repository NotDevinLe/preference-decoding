#!/bin/bash
#SBATCH --job-name=train_mle
#SBATCH --account=ark
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/output_%A_%a.txt
#SBATCH --error=../logs/error_%A_%a.txt
#SBATCH --array=16

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

python train_mle.py --name user1 --num_expectation_samples 200 --num_epochs 40000 --learning_rate 0.01 --beta 1.0 --num_mc_samples ${SLURM_ARRAY_TASK_ID} --use_wandb --wandb_project mle-preference --sample_size 200 --load_chosen_rewards