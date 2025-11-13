#!/bin/bash
#SBATCH --job-name=precompute
#SBATCH --account=ark
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/output_%A.txt
#SBATCH --error=logs/error_%A.txt

source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export PATH=/gscratch/ark/devinl6/envs/align/bin:$PATH
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache

PORT=8080

# Start gateway on this node, in the background
srun --ntasks=1 literegistry gateway \
    --port ${PORT} \
    --host 0.0.0.0 \
    --registry "/gscratch/ark/devinl6/registry" &
GATEWAY_PID=$!

echo "Gateway started on 0.0.0.0:${PORT}"

sleep 10

echo "Starting reward matrix computation"

python scripts/precompute/compute_reward_matrix.py

kill ${GATEWAY_PID} 2>/dev/null || true
wait ${GATEWAY_PID} 2>/dev/null || true