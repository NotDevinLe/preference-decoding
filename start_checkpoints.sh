#!/bin/bash
#SBATCH --job-name=start_checkpoints
#SBATCH --account=ark
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --gres=gpu:a40:4
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache
export TORCHDYNAMO_VERBOSE=1
export VLLM_USE_MODELSCOPE=False

# Get the task ID for port assignment
NODE_ID=${SLURM_PROCID}

echo "Starting vLLM server on task $NODE_ID"

literegistry vllm \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1024 \
  --registry "/gscratch/ark/devinl6/registry" \