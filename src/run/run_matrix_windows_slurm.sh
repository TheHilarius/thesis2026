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

# PCA scheme: 'slot' (default) or 'flat'. Override via PCA_MODE=flat.
PCA_MODE="${PCA_MODE:-slot}"

# PCA sweep values per toolkit. Slot values are the 1st/3rd/5th entries of the
# explained-variance reference (esmc: 1,13,26,66,218,718 -> 1,26,218 ;
# esmif: 9,64,95,148,248,420 -> 9,95,248). Flat values must be re-derived from
# 08_pca_variance_analysis.py --pca-mode flat before running the flat pass.
if [ "$PCA_MODE" = "flat" ]; then
    ESMC_PCA="${ESMC_PCA:-1,26,218}"
    ESMIF_PCA="${ESMIF_PCA:-9,95,248}"
else
    ESMC_PCA="1,26,218"
    ESMIF_PCA="9,95,248"
fi

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

echo "Matrix run: features=${FEAT}  pca=${PCA}  pca_mode=${PCA_MODE}  (array task ${IDX})"

python src/pipeline/python/04b_run_matrix.py \
  --pca-mode "${PCA_MODE}" \
  --pca-sweep \
    "rf:${FEAT}:${PCA}" \
    "xgb:${FEAT}:${PCA}" \
    "lr_l2:${FEAT}:${PCA}"

echo "Matrix run for ${FEAT} completed."
