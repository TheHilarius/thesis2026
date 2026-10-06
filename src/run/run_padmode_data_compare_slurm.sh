#!/bin/bash
#SBATCH --job-name=padmode_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --array=0-10%4
#SBATCH --output=logs/padmode_data_%A_%a.out
#SBATCH --error=logs/padmode_data_%A_%a.err

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

# Run-folder root, exported by the submit one-liner below. Unset = each run
# auto-creates runs/<run_tag>_<timestamp>/; set = shared campaign folder.
OUT="${RUN:-.}"

# Link campaign inputs (run-folder submissions only; skip existing).
if [ -n "${RUN:-}" ]; then
  [ -e "$OUT/data/df_all_with_folds.csv" ] || ln -s ../../../data/processed/df_all_with_folds.csv "$OUT/data/"
  for h in esmc_win_zeropad esmc_win_padtoken esmc_win_impute_bos_eos esmc_win_impute_boundary \
           esmif_zero_windows esmif_pad_windows esmif_boundary_windows esmif_eos_bos_repeat_windows; do
    [ -e "$OUT/data/${h}_prepared.h5" ] || ln -s "../../../data/processed/embeddings_prepared/${h}_prepared.h5" "$OUT/data/"
  done
fi

# Submit with a timestamped run folder (paste, no new files needed):
#   RUN="runs/padmode_data_$(date +%Y%m%d_%H%M)"; mkdir -p "$RUN"/{models,logs/slurm,results,plots,data} && sbatch --job-name=padmode_data --output="$RUN/logs/slurm/%A_%a.out" --error="$RUN/logs/slurm/%A_%a.err" --export=RUN="$RUN",ALL src/run/run_padmode_data_compare_slurm.sh

# ════════════════════════════════════════════════════════════════════════════
# DATA-COMPARISON GRID — XGBOOST ONLY, --pca-mode flat everywhere.
#
# Bars (11 total):
#   0-2  CSV-only baselines (no embeddings; shared across both toolkits):
#        sparse (one-hot AA), blosum, blosum + engineered (handcrafted_blosum).
#   3-6  ESM-C (600M) embeddings only, 4 pad modes — PCA @ 90% EV (k=6389).
#   7-10 ESM-IF1 embeddings only, 4 pad modes — PCA @ 90% EV (k=4308).
#
# PC dims from 08_pca_variance_analysis.py --pca-mode flat-full (spectra are
# mode-invariant by construction: fit uses fully-real rows only).
#   ESM-C : 90%=6389
#   ESM-IF: 90%=4308
# ════════════════════════════════════════════════════════════════════════════

ESMC_PCA="6389"
ESMIF_PCA="4308"

# Entry format: MODEL|FEATURES|PCA|PCA_MODE
# PCA "none" = no PCA (CSV-only sets; flat mode is ignored for them).
CONFIGS=(
  # 0-2 CSV-only baselines
  "xgb|sparse|none|flat"
  "xgb|blosum|none|flat"
  "xgb|handcrafted_blosum|none|flat"
  # 3-6 ESM-C embeddings only (90% EV)
  "xgb|emb_esmc_win_zeropad|${ESMC_PCA}|flat"
  "xgb|emb_esmc_win_padtoken|${ESMC_PCA}|flat"
  "xgb|emb_esmc_win_impute_bos_eos|${ESMC_PCA}|flat"
  "xgb|emb_esmc_win_impute_boundary|${ESMC_PCA}|flat"
  # 7-10 ESM-IF embeddings only (90% EV)
  "xgb|emb_esmif_win|${ESMIF_PCA}|flat"
  "xgb|emb_esmif_pad|${ESMIF_PCA}|flat"
  "xgb|emb_esmif_boundary|${ESMIF_PCA}|flat"
  "xgb|emb_esmif_eos_bos_repeat|${ESMIF_PCA}|flat"
)

IDX="${SLURM_ARRAY_TASK_ID}"
CFG="${CONFIGS[$IDX]}"
MODEL="${CFG%%|*}";  REST="${CFG#*|}"
FEAT="${REST%%|*}";  REST="${REST#*|}"
PCA="${REST%%|*}"
PCAMODE="${REST##*|}"

# "none" means: don't pass --pca to 04b at all.
if [ "$PCA" = "none" ]; then
  PCA_ARG="none"
else
  PCA_ARG="$PCA"
fi

echo "Array task ${IDX}: model=${MODEL} features=${FEAT} pca=${PCA_ARG} mode=${PCAMODE}"

python src/pipeline/python/04b_run_matrix.py \
  --pca-mode "$PCAMODE" \
  --no-aggregate \
  --out-root "$OUT" \
  --pca-sweep "${MODEL}:${FEAT}:${PCA_ARG}"

echo "Array task ${IDX} completed."