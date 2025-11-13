#!/bin/bash
#SBATCH --job-name=start_checkpoints
#SBATCH --account=ark
#SBATCH --partition=gpu-h200
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache
export TORCHDYNAMO_VERBOSE=1
export VLLM_USE_MODELSCOPE=False
export VLLM_ATTENTION_BACKEND=XFORMERS

literegistry vllm \
  --model "meta-llama/Llama-3.2-1B-Instruct" \
  --registry gscratch/ark/devinl6/registry \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --dtype bfloat16 \