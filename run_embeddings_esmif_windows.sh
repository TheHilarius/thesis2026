#!/bin/bash
#SBATCH --job-name=esmif_windows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/esmif_windows_%j.out
#SBATCH --error=logs/esmif_windows_%j.err

# ── Edit these two lines to match your HPC setup ──────────────────────────────
PROJECT_DIR="/home/projects1/thesis_s204692_s204581/thesis2026"
CONDA_ENV="esm_gpu"          # fair-esm env (has esm.inverse_folding)

cd "$PROJECT_DIR" || exit 1

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Reduces GPU memory fragmentation (same flag as the NetSurfP run)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f "env_esmif_server/bin/activate" ]; then
    source env_esmif_server/bin/activate
    echo "Activated env_esmif_server"
else
    echo "ERROR: No ESM-IF venv found"
    exit 1
fi

echo "Starting ESM-IF fixed-29 window embedding (3 padding modes) at $(date)"

python src/tools/esm/embed_windows_esmif.py

echo "Done at $(date)"
