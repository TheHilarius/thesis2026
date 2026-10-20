#!/bin/bash
#SBATCH --job-name=mpnn_matrix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mpnn_matrix_%j.out
#SBATCH --error=logs/mpnn_matrix_%j.err

# ProteinMPNN XGBoost matrix runner (embedding-only, flat PCA).
#
# Parameterized via --export so the same script covers every MPNN grid:
#   OUT    run-folder root shared by all combos of one submission
#   COMBOS space-separated PIPE-delimited "model|features|pca" specs.
#
# ⚠️ COMBOS must use '|' between fields, never ','. Slurm splits --export on
#    commas, so a comma-delimited COMBOS is silently truncated to its first
#    fragment (COMBOS="xgb") and 04b dies with IndexError. This script
#    translates '|' -> ',' below.
#
# Submit from the repo root on the server. Do NOT use `sbatch --wrap` for this
# work: --wrap runs under dash, where `source` does not exist (exit 127).
#
#   Checkpoint benchmark — 4 checkpoints, zero-pad, each at its own N90:
#     sbatch --export=ALL,OUT=runs/mpnn_ckpt_zeropad,COMBOS="xgb|emb_mpnn_v48002_zeropad|1478 xgb|emb_mpnn_v48010_zeropad|1482 xgb|emb_mpnn_v48020_zeropad|1325 xgb|emb_mpnn_v48030_zeropad|1301" src/run/run_mpnn_matrix_slurm.sh
#
#   Pad-mode check on the chosen checkpoint:
#     sbatch --export=ALL,OUT=runs/mpnn_padmode_final,COMBOS="xgb|emb_mpnn_v48XXX_zeropad|<N90> xgb|emb_mpnn_v48XXX_boundary|<N90>" src/run/run_mpnn_matrix_slurm.sh
#
# Results: $OUT/models/flat/cv_results_*.json (one per combo).
# Aggregation is deliberately OFF; run extract_model_metrics.py once at the end.

set -euo pipefail

if [ -z "${COMBOS:-}" ]; then
    echo "ERROR: set COMBOS (space-separated 'model|features|pca' specs)"
    exit 1
fi
if [ -z "${OUT:-}" ]; then
    echo "ERROR: set OUT (run-folder root, e.g. runs/mpnn_ckpt_zeropad)"
    exit 1
fi

# Slurm splits --export on commas, so the command line must use '|' between
# fields. Translate to the comma form 04b expects (model,features,pca).
COMBOS="${COMBOS//|/,}"
case "$COMBOS" in
    *,*) ;;
    *) echo "ERROR: COMBOS must be pipe-delimited:" \
             "'model|features|pca [model|features|pca ...]'"
       exit 1 ;;
esac

# Same project path as run_matrix_windows_flat_slurm.sh.
cd /home/projects/thesis_s204692_s204581/thesis2026 || exit 1

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No modelling venv found at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs
export PYTHONUNBUFFERED=1

echo "OUT=${OUT}"
echo "COMBOS=${COMBOS}"
echo "Starting ProteinMPNN XGB matrix at $(date)"

# ${COMBOS} is intentionally unquoted: word-splitting into one spec per combo.
python src/pipeline/python/04b_run_matrix.py \
    --pca-mode flat \
    --no-aggregate \
    --out-root "${OUT}" \
    --combos ${COMBOS}

echo "Done at $(date)"
