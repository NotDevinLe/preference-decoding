#!/bin/bash
#SBATCH --job-name=rm_acc_n
#SBATCH --account=ark
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=70,80,90,100,120,140,160,180,200

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python rm_acc_n.py --name user1 --n ${SLURM_ARRAY_TASK_ID}
