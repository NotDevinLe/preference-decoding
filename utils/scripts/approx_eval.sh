#!/bin/bash
#SBATCH --job-name=eval_approx
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --array=2

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python eval_approx.py --name user${SLURM_ARRAY_TASK_ID} --sample_size 100 --p_path "../results/l1_reg/p/${SLURM_ARRAY_TASK_ID}.jsonl" --k 7 --save_path "../results/l1_reg/small_results.jsonl"