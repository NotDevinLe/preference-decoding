#!/bin/bash
#SBATCH --job-name=serve
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt

# Properly load conda
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Run Python script with unbuffered output
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --host 0.0.0.0 --port 8000 --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.95 --max-num-seqs 16
