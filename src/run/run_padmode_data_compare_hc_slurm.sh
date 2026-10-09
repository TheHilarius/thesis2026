#!/bin/bash
#SBATCH --job-name=padmode_hc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --output=logs/padmode_hc_%A_%a.out
#SBATCH --error=logs/padmode_hc_%A_%a.err

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

# Run-folder root, exported by the submit one-liner (same convention as
# run_padmode_data_compare_slurm.sh). Unset = auto runs/<tag>_<timestamp>/.
OUT="${RUN:-.}"

# ══════════════════════════════════════════════════════════════════
# HANDCRAFTED-ONLY BAR — XGBOOST, --pca-mode flat.
# Engineered/structural features ONLY: no embeddings, no AA encoding.
# Companion to run_padmode_data_compare_slurm.sh (bars 0-2 there).
# With the component-aware fix: structural cols only (~74 on server).
# ══════════════════════════════════════════════════════════════════

# Entry format: MODEL|FEATURES|PCA|PCA_MODE
CONFIGS=(
  "xgb|handcrafted_sparse|none|flat"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
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

echo "Task ${IDX}: model=${MODEL} features=${FEAT} pca=${PCA_ARG} mode=${PCAMODE}"

python src/pipeline/python/04b_run_matrix.py \
  --pca-mode "$PCAMODE" \
  --no-aggregate \
  --out-root "$OUT" \
  --pca-sweep "${MODEL}:${FEAT}:${PCA_ARG}"

echo "Task ${IDX} completed."
