#!/bin/bash
#SBATCH --job-name=pool_xgb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-27%4
#SBATCH --output=runs/pool_xgb/logs/slurm/%A_%a.out
#SBATCH --error=runs/pool_xgb/logs/slurm/%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No venv at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs
export PYTHONUNBUFFERED=1

# Campaign folder shared by the whole pooling grid (pooled features are
# tiny — no PCA, straight to XGB). Override per submit, e.g.
#   RUN="runs/pool_xgb" (default).
OUT="${RUN:-runs/pool_xgb}"
mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/results" "$OUT/plots"

# Grid: 2 embeddings x 7 pools x 2 masks = 28 array tasks.
# Embeddings are embedding-only feature sets, so the pooled vector is the
# whole signal (cleanest pooling comparison). To add structural/one-hot
# context instead, swap in handcrafted_sparse_esmc300m_win_padtoken /
# handcrafted_sparse_esmif_eos_bos_repeat below.
FEATS=(
  "emb_esmc300m_win_padtoken"
  "emb_esmif_eos_bos_repeat"
)
POOLS=(
  "mean" "median" "max" "min" "mean_max" "mean_median" "mean_std"
)
MASKS=(
  "masked" "naive"
)

N_FEATS=${#FEATS[@]}
N_POOLS=${#POOLS[@]}
N_MASKS=${#MASKS[@]}

IDX="${SLURM_ARRAY_TASK_ID}"
FEAT="${FEATS[$(( IDX / (N_POOLS * N_MASKS) ))]}"
REST=$(( IDX % (N_POOLS * N_MASKS) ))
POOL="${POOLS[$(( REST / N_MASKS ))]}"
MASK="${MASKS[$(( REST % N_MASKS ))]}"

echo "task=${IDX} features=${FEAT} pool=${POOL} mask=${MASK} out=${OUT}"

python src/pipeline/python/04_modelling.py \
  --model xgb \
  --features "$FEAT" \
  --pool "$POOL" \
  --pool-mask "$MASK" \
  --out-root "$OUT"

echo "Pool task ${IDX} completed."

# Submit into the fixed campaign folder (paste on hub1).
# NOTE: create the slurm log dir first -- sbatch will not make it for you:
#   mkdir -p runs/pool_xgb/logs/slurm && sbatch --export=RUN="runs/pool_xgb",ALL src/run/run_pool_xgb_slurm.sh
# Single-method test (task 0 = esmc300m/mean/masked):
#   mkdir -p runs/pool_xgb_test/logs/slurm && sbatch --array=0 --export=RUN="runs/pool_xgb_test",ALL src/run/run_pool_xgb_slurm.sh
# Aggregate after the grid finishes (one manual call, array jobs skip it):
#   python src/pipeline/python/extract_model_metrics.py --results "runs/pool_xgb/models/flat/cv_results_*.json" --since "<grid start>" --out runs/pool_xgb/results/model_matrix_pool_xgb.tsv
