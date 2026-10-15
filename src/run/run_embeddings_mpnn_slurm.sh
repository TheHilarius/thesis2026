#!/bin/bash
#SBATCH --job-name=mpnn_windows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mpnn_windows_%A_%a.out
#SBATCH --error=logs/mpnn_windows_%A_%a.err

# ProteinMPNN encoder-only 29-mer window embeddings, one array task per checkpoint.
#   array 0 -> v_48_020 (default; moderate 0.20A training noise, robust to AF2 error)
#   array 1 -> v_48_002 (0.02A noise; sensitivity arm)
#
# Submit from repo root (server: healthtech hub1):
#   mkdir -p logs
#   sbatch src/run/run_embeddings_mpnn_slurm.sh
#
# Writes (does NOT overwrite; script aborts if files exist):
#   data/processed/embeddings/mpnn_<ckpt>_context_embeddings_{zero,boundary}.h5

set -euo pipefail

# projects1 / projects2 are symlinks to projects on the server (same as ESM-IF script).
PROJECT_DIR="/home/projects1/thesis_s204692_s204581/thesis2026"
cd "$PROJECT_DIR" || exit 1

if [ -f "env_mpnn_server/bin/activate" ]; then
    source env_mpnn_server/bin/activate
    echo "Activated env_mpnn_server"
else
    echo "ERROR: No ProteinMPNN venv found at env_mpnn_server/bin/activate"
    exit 1
fi

# checkpoint per array task
CKPTS=(v_48_020 v_48_002)
CKPT="${CKPTS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: checkpoint=${CKPT}"

mkdir -p logs data/processed/embeddings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Starting ProteinMPNN window embedding (ckpt=${CKPT}, modes=zero,boundary) at $(date)"

python src/tools/proteinmpnn/embed_peptides_mpnn.py \
    --checkpoint "${CKPT}" \
    --modes zero,boundary \
    --out-dir data/processed/embeddings

echo "Done at $(date)"
