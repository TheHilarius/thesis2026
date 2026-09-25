#!/bin/bash
#SBATCH --job-name=winmat_slot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/winmat_slot_%A_%a.out
#SBATCH --error=logs/winmat_slot_%A_%a.err

set -euo pipefail

cd /home/projects/thesis_s204692_s204581/thesis2026

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No ESM-C venv found at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs/winmat_slot results/tables
export PYTHONUNBUFFERED=1

# SLOT pass: per-position shared PCA.
# Sweep values from 08_pca_variance_analysis.py --pca-mode slot
# (50/85/95% EV of the window-slot curve):
#   esmc slot: 9,215,334,501,742,1040 -> 9,334,742
#   esmif slot: 47,171,214,274,359,466 -> 47,214,359
ESMC_PCA="9,334,742"
ESMIF_PCA="47,214,359"

# ── 8 embeddings: feature_set|pca1,pca2,pca3 ────────────────────────────────
CONFIGS=(
  "handcrafted_sparse_esmc_win_zeropad|${ESMC_PCA}"
  "handcrafted_sparse_esmc_win_padtoken|${ESMC_PCA}"
  "handcrafted_sparse_esmc_win_impute_bos_eos|${ESMC_PCA}"
  "handcrafted_sparse_esmc_win_impute_boundary|${ESMC_PCA}"
  "handcrafted_sparse_esmif_win|${ESMIF_PCA}"
  "handcrafted_sparse_esmif_pad|${ESMIF_PCA}"
  "handcrafted_sparse_esmif_boundary|${ESMIF_PCA}"
  "handcrafted_sparse_esmif_eos_bos_repeat|${ESMIF_PCA}"
)

IDX="${SLURM_ARRAY_TASK_ID}"
CFG="${CONFIGS[$IDX]}"
FEAT="${CFG%%|*}"
PCA="${CFG##*|}"

echo "Matrix run (slot): features=${FEAT}  pca=${PCA}  (array task ${IDX})"

python src/pipeline/python/04b_run_matrix.py \
  --pca-mode slot \
  --pca-sweep \
    "rf:${FEAT}:${PCA}" \
    "xgb:${FEAT}:${PCA}" \
    "lr_l2:${FEAT}:${PCA}"

echo "Matrix run for ${FEAT} completed."
