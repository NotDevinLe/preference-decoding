#!/bin/bash
#SBATCH --job-name=start_vllm_server
#SBATCH --account=ark
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%j_%t.txt
#SBATCH --error=logs/error_%j_%t.txt

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache
export vllm_cache_dir=/gscratch/ark/devinl6/vllm_cache

# Get the task ID for port assignment
NODE_ID=${SLURM_PROCID}
PORT=$((8000 + NODE_ID))

echo "Starting vLLM server on task $NODE_ID, port $PORT"

srun python /gscratch/ark/devinl6/preference/preference-decoding/vllm_server.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --registry_dir "/gscratch/ark/devinl6/registry" \
  --max_new_tokens 1024 \
  --max_prompt_length 2048 \
  --dtype bfloat16
