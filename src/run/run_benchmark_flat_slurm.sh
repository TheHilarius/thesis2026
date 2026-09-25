#!/bin/bash
#SBATCH --job-name=bench_flat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_flat_%j.out
#SBATCH --error=logs/bench_flat_%j.err

set -euo pipefail

cd /home/projects/thesis_s204692_s204581/thesis2026

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No ESM-C venv found at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs
export PYTHONUNBUFFERED=1

# Single-combo benchmark for flat mode (rf x handcrafted_sparse_esmc_win_zeropad, pca 26).
# Run this first to time one flat combo before launching the full batch.
python src/pipeline/python/04_modelling.py \
  --model rf \
  --features handcrafted_sparse_esmc_win_zeropad \
  --pca 26 \
  --pca-mode flat

echo "Benchmark done."
