#!/bin/bash
#SBATCH --job-name=pcavar_flat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcavar_flat_%j.out
#SBATCH --error=logs/pcavar_flat_%j.err

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

# Flat variance curve -> results/figures/models/pca_optimization/
#   pca_variance_windows_{esmc_flat,esmif_flat}.csv (+ .png)
# Use these to pick the flat sweep k's (1st/3rd/5th threshold).
python src/investigation/08_pca_variance_analysis.py \
  --pca-mode flat \
  --flat-n-components "${FLAT_N_COMPONENTS:-1024}"

echo "08 flat variance analysis completed."
