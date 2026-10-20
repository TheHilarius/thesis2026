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
#   COMBOS space-separated "model,features,pca" specs (04b --combos)
#
# Submit from the repo root on the server. Do NOT use `sbatch --wrap` for this
# work: --wrap runs under dash, where `source` does not exist (exit 127).
#
#   Phase 2 — pad-mode decision (v_48_020, 1325 PCs = 90% variance):
#     sbatch --export=ALL,OUT=runs/mpnn_padmode_v48020,COMBOS="xgb,emb_mpnn_v48020_zeropad,1325 xgb,emb_mpnn_v48020_boundary,1325" src/run/run_mpnn_matrix_slurm.sh
#
#   Phase 3 — train-noise comparison (4 checkpoints, each at its own N90):
#     sbatch --export=ALL,OUT=runs/mpnn_noise_<mode>,COMBOS="xgb,emb_mpnn_v48002_<mode>,1478 xgb,emb_mpnn_v48010_<mode>,<N90> xgb,emb_mpnn_v48020_<mode>,1325 xgb,emb_mpnn_v48030_<mode>,<N90>" src/run/run_mpnn_matrix_slurm.sh
#
# Results: $OUT/models/flat/cv_results_*.json (one per combo).
# Aggregation is deliberately OFF; run extract_model_metrics.py once at the end.

set -euo pipefail

if [ -z "${COMBOS:-}" ]; then
    echo "ERROR: set COMBOS (space-separated 'model,features,pca' specs)"
    exit 1
fi
if [ -z "${OUT:-}" ]; then
    echo "ERROR: set OUT (run-folder root, e.g. runs/mpnn_padmode_v48020)"
    exit 1
fi

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
