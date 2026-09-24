#!/bin/bash
#SBATCH --job-name=winmat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/winmat_%A_%a.out
#SBATCH --error=logs/winmat_%A_%a.err

set -euo pipefail

cd /home/projects/thesis_s204692_s204581/thesis2026

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No ESM-C venv found at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs results/tables
export PYTHONUNBUFFERED=1

# ── 8 embeddings: feature_set|pca1,pca2,pca3 ────────────────────────────────
# PCA values are the 1st, 3rd and 5th entries of the explained-variance
# reference (esmc: 1,13,26,66,218,718 -> 1,26,218 ;
#           esmif: 9,64,95,148,248,420 -> 9,95,248).
CONFIGS=(
  "handcrafted_sparse_esmc_win_zeropad|1,26,218"
  "handcrafted_sparse_esmc_win_padtoken|1,26,218"
  "handcrafted_sparse_esmc_win_impute_bos_eos|1,26,218"
  "handcrafted_sparse_esmc_win_impute_boundary|1,26,218"
  "handcrafted_sparse_esmif_win|9,95,248"
  "handcrafted_sparse_esmif_pad|9,95,248"
  "handcrafted_sparse_esmif_boundary|9,95,248"
  "handcrafted_sparse_esmif_eos_bos_repeat|9,95,248"
)

IDX="${SLURM_ARRAY_TASK_ID}"
CFG="${CONFIGS[$IDX]}"
FEAT="${CFG%%|*}"
PCA="${CFG##*|}"

echo "Matrix run: features=${FEAT}  pca=${PCA}  (array task ${IDX})"

python src/pipeline/python/04b_run_matrix.py \
  --pca-sweep \
    "rf:${FEAT}:${PCA}" \
    "xgb:${FEAT}:${PCA}" \
    "lr_l2:${FEAT}:${PCA}"

echo "Matrix run for ${FEAT} completed."
