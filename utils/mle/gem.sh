#!/bin/bash
#SBATCH --job-name=generate_expectation_matrix
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
#SBATCH --array=200

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

python generate_expectation_matrix.py --prompts_file "../../data/preference/user1_train.json" --num_expectation_samples 200 --output_path "../../data/expectation_matrices/user1_expectation_n200_size200.pt" --sample_size 200