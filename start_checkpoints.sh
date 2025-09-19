#!/bin/bash
#SBATCH --job-name=start_checkpoints
#SBATCH --account=ark
#SBATCH --partition=ckpt-g2
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

python /gscratch/ark/devinl6/preference/preference-decoding/vllm_server.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --registry_dir "/gscratch/ark/devinl6/registry" \
  --max_new_tokens 800 \
  --max_prompt_length 1200 \
  --dtype bfloat16
