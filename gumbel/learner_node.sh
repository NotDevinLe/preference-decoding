#!/usr/bin/env bash
#SBATCH --job-name=learner
#SBATCH --account=ark
#SBATCH --partition=gpu-l40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Safety: avoid -u due to conda deactivate hooks
set -eo pipefail
mkdir -p logs

# === Env ===
# (Adjust these to your setup)
source /gscratch/ark/devinl6/miniconda3/etc/profile.d/conda.sh
conda activate align
export HF_HOME=/mmfs1/gscratch/ark/devinl6/hf_cache
export CUDA_VISIBLE_DEVICES=0   # maps to your script's --device if you want

echo "== $(date) =="
echo "Node: $(hostname)  SLURM_JOB_ID: ${SLURM_JOB_ID}"
python -V
nvidia-smi || true

# === Run directly (no nested sbatch) ===
# You can use srun for cleaner Slurm I/O:
srun --unbuffered ./start_learner.sh \
  --d 400 \
  --k 50 \
  --lr 1e-3 \
  --sparsity-weight 1e-4 \
  --tau-init 1.0 \
  --host 0.0.0.0 \
  --port 8002 \
  --device cuda:0 \
  --checkpoint-dir ./checkpoints \
  --log-level INFO
