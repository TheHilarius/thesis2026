#!/bin/bash
#SBATCH --job-name=esmif_windows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/esmif_windows_%j.out
#SBATCH --error=logs/esmif_windows_%j.err

# projects1 / projects2 are symlinks to projects on the server.
PROJECT_DIR="/home/projects1/thesis_s204692_s204581/thesis2026"

cd "$PROJECT_DIR" || exit 1

# Activate the ESM-IF venv (server uses venvs, not conda)
if [ -f "env_esmif_server/bin/activate" ]; then
    source env_esmif_server/bin/activate
    echo "Activated env_esmif_server"
else
    echo "ERROR: No ESM-IF venv found"
    exit 1
fi

# Ensure output directories exist
mkdir -p logs
mkdir -p data/processed/embeddings

# Reduces GPU memory fragmentation (same flag as the NetSurfP run)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Starting ESM-IF fixed-29 window embedding (4 padding modes: zero, pad, boundary, eos_bos_repeat) at $(date)"

python src/tools/esm/embed_windows_esmif.py

echo "Done at $(date)"
