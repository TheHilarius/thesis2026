#!/bin/bash
#SBATCH --job-name=padfull_flat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --array=0-24%4
#SBATCH --output=logs/padfull_flat_%A_%a.out
#SBATCH --error=logs/padfull_flat_%A_%a.err

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

# ════════════════════════════════════════════════════════════════════════════
# Ordered padmode × PC grid — XGBOOST ONLY.
#
# PC dims = flat-full variance thresholds (08_pca_variance_analysis.py
# --pca-mode flat-full; zero-mode representatives — spectra are mode-
# invariant for zero/boundary/eos_bos; padtoken shifts ~1-2 PCs).
#   ESM-C : 50%=97  80%=2346 85%=3888 90%=6389 95%=10895 99%=20142  100%=33408
#   ESM-IF: 50%=345 80%=2125 85%=2974 90%=4308 95%=6700  99%=11243  100%=14848
#
# Queue order (Slurm starts array tasks in ascending index order under %4):
#   0-3   ESM-C main  (50-95%)   ~44h
#   4-7   ESM-IF main (50-95%)   ~33h
#   8     baseline (handcrafted_sparse, no embeddings)   minutes
#   9-12  ESM-C 99%   ~36h
#   13-16 ESM-IF 99%  ~23h
#   17-20 ESM-C 100%  (flat_raw, 33408 dims, no PCA)  ~60-80h
#   21-24 ESM-IF 100% (flat_raw, 14848 dims, no PCA)  ~35h
# ════════════════════════════════════════════════════════════════════════════

ESMC_MAIN="97,2346,3888,6389,10895"
ESMC_99="20142"
ESMIF_MAIN="345,2125,2974,4308,6700"
ESMIF_99="11243"

# Entry format: MODEL|FEATURES|PCA_LIST|PCA_MODE
# (PCA_LIST "none" is ignored for flat_raw — all raw dims are used)
CONFIGS=(
  # 0-3 ESM-C main (50-95%)
  "xgb|handcrafted_sparse_esmc_win_zeropad|${ESMC_MAIN}|flat"
  "xgb|handcrafted_sparse_esmc_win_padtoken|${ESMC_MAIN}|flat"
  "xgb|handcrafted_sparse_esmc_win_impute_bos_eos|${ESMC_MAIN}|flat"
  "xgb|handcrafted_sparse_esmc_win_impute_boundary|${ESMC_MAIN}|flat"
  # 4-7 ESM-IF main (50-95%)
  "xgb|handcrafted_sparse_esmif_win|${ESMIF_MAIN}|flat"
  "xgb|handcrafted_sparse_esmif_pad|${ESMIF_MAIN}|flat"
  "xgb|handcrafted_sparse_esmif_boundary|${ESMIF_MAIN}|flat"
  "xgb|handcrafted_sparse_esmif_eos_bos_repeat|${ESMIF_MAIN}|flat"
  # 8 baseline (structural + one-hot, NO embeddings)
  "xgb|handcrafted_sparse|none|slot"
  # 9-12 ESM-C 99%
  "xgb|handcrafted_sparse_esmc_win_zeropad|${ESMC_99}|flat"
  "xgb|handcrafted_sparse_esmc_win_padtoken|${ESMC_99}|flat"
  "xgb|handcrafted_sparse_esmc_win_impute_bos_eos|${ESMC_99}|flat"
  "xgb|handcrafted_sparse_esmc_win_impute_boundary|${ESMC_99}|flat"
  # 13-16 ESM-IF 99%
  "xgb|handcrafted_sparse_esmif_win|${ESMIF_99}|flat"
  "xgb|handcrafted_sparse_esmif_pad|${ESMIF_99}|flat"
  "xgb|handcrafted_sparse_esmif_boundary|${ESMIF_99}|flat"
  "xgb|handcrafted_sparse_esmif_eos_bos_repeat|${ESMIF_99}|flat"
  # 17-20 ESM-C 100% (flat_raw — all 33408 dims, no PCA)
  "xgb|handcrafted_sparse_esmc_win_zeropad|none|flat_raw"
  "xgb|handcrafted_sparse_esmc_win_padtoken|none|flat_raw"
  "xgb|handcrafted_sparse_esmc_win_impute_bos_eos|none|flat_raw"
  "xgb|handcrafted_sparse_esmc_win_impute_boundary|none|flat_raw"
  # 21-24 ESM-IF 100% (flat_raw — all 14848 dims, no PCA)
  "xgb|handcrafted_sparse_esmif_win|none|flat_raw"
  "xgb|handcrafted_sparse_esmif_pad|none|flat_raw"
  "xgb|handcrafted_sparse_esmif_boundary|none|flat_raw"
  "xgb|handcrafted_sparse_esmif_eos_bos_repeat|none|flat_raw"
)

IDX="${SLURM_ARRAY_TASK_ID}"
CFG="${CONFIGS[$IDX]}"
MODEL="${CFG%%|*}";  REST="${CFG#*|}"
FEAT="${REST%%|*}";  REST="${REST#*|}"
PCA="${REST%%|*}"
PCAMODE="${REST##*|}"

# flat_raw ignores PCA (runs all raw dims); "none" placeholder for slot/flat
if [ "$PCAMODE" = "flat_raw" ]; then
  PCA_ARG="0"
else
  PCA_ARG="$PCA"
fi

echo "Array task ${IDX}: model=${MODEL} features=${FEAT} pca=${PCA_ARG} mode=${PCAMODE}"

python src/pipeline/python/04b_run_matrix.py \
  --pca-mode "$PCAMODE" \
  --no-aggregate \
  --pca-sweep "${MODEL}:${FEAT}:${PCA_ARG}"

echo "Array task ${IDX} completed."
