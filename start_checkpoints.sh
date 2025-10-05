#!/bin/bash
#SBATCH --job-name=start_checkpoints
#SBATCH --account=ark
#SBATCH --partition=ckpt-g2
#SBATCH --gpus-per-task=4
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%j_%t.txt
#SBATCH --error=logs/error_%j_%t.txt

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

# Get the task ID for port assignment
NODE_ID=${SLURM_PROCID}

echo "Starting vLLM server on task $NODE_ID"

srun python /gscratch/ark/devinl6/preference/preference-decoding/vllm_server.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.70 \
  --registry_dir "/gscratch/ark/devinl6/registry" \
  --max_new_tokens 2048 \
  --max_prompt_length 2048 \
  --dtype bfloat16
