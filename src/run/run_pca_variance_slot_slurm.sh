#!/bin/bash
#SBATCH --job-name=pcavar_slot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcavar_slot_%j.out
#SBATCH --error=logs/pcavar_slot_%j.err

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

# Slot variance curves -> results/figures/models/pca_optimization/
#   pca_variance_windows_{esmc_zeropad,esmc_impute_boundary,esmc_impute_bos_eos,esmc_padtoken,
#                         esmif_zero,esmif_pad,esmif_boundary,esmif_eos_bos_repeat}.csv (+ .png)
# Use these to re-derive the slot sweep k's (1st/3rd/5th threshold) instead of
# reusing the legacy-context reference values.
python src/investigation/08_pca_variance_analysis.py \
  --pca-mode slot

echo "08 slot variance analysis completed."
