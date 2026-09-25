#!/bin/bash
#SBATCH --job-name=winmat_flat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/winmat_flat_%A_%a.out
#SBATCH --error=logs/winmat_flat_%A_%a.err

set -euo pipefail

cd /home/projects/thesis_s204692_s204581/thesis2026

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No ESM-C venv found at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs/winmat_flat results/tables
export PYTHONUNBUFFERED=1

# FLAT pass: flattened-window PCA.
# Sweep values from 08_pca_variance_analysis.py --pca-mode flat
# (30/50/65% EV of the flat curve, which caps at 72%/68%):
#   esmc flat: 30%=10, 50%=97, 65%=469
#   esmif flat: 30%=76, 50%=345, 65%=851
# Override via ESMC_PCA / ESMIF_PCA env vars if the sweep changes.
ESMC_PCA="${ESMC_PCA:-10,97,469}"
ESMIF_PCA="${ESMIF_PCA:-76,345,851}"

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

echo "Matrix run (flat): features=${FEAT}  pca=${PCA}  (array task ${IDX})"

python src/pipeline/python/04b_run_matrix.py \
  --pca-mode flat \
  --pca-sweep \
    "rf:${FEAT}:${PCA}" \
    "xgb:${FEAT}:${PCA}" \
    "lr_l2:${FEAT}:${PCA}"

echo "Matrix run for ${FEAT} completed."
