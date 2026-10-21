#!/bin/bash
# run_xgb_lr_sweep.sh — XGBoost learning-rate sweep via --xgb-params.
#
# One sbatch per (learning_rate:n_estimators) variant. Baseline
# lr0.1:n1000 already exists (encodings_sweep blosum50, AUC 0.7633) and is
# re-usable: same seed/folds/defaults -> identical result, do not rerun.
#
# Submit (interactive shell, repo root on hub1):
#   bash src/run/run_xgb_lr_sweep.sh --submit --out runs/xgb_lr_blosum50 \
#     --features blosum50 \
#     --variant 0.1:2500 --variant 0.05:1000 --variant 0.05:2500 \
#     --variant 0.025:1000 --variant 0.025:2500
#
# Solo-embedding arms (later, one feature at a time):
#   bash src/run/run_xgb_lr_sweep.sh --submit --out runs/xgb_lr_esmc_padtoken \
#     --features emb_esmc_win_padtoken --pca 2346 --variant 0.05:2500 ...
#
# Aggregate afterwards:
#   python src/pipeline/python/extract_model_metrics.py \
#     --results 'runs/xgb_lr_blosum50/models/flat/cv_results_*.json' \
#     --out runs/xgb_lr_blosum50/results/lr_sweep.tsv
#SBATCH --job-name=xgb_lr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO=/home/projects/thesis_s204692_s204581/thesis2026

# ── submitter mode ──
if [[ "${1:-}" == "--submit" ]]; then
    shift
    OUT_ROOT=""
    FEATURES="blosum50"
    PCA=""
    HOURS="2"
    VARIANTS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out)      OUT_ROOT="$2"; shift 2 ;;
            --features) FEATURES="$2"; shift 2 ;;
            --pca)      PCA="$2"; shift 2 ;;
            --hours)    HOURS="$2"; shift 2 ;;
            --variant)  shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                            VARIANTS+=("$1"); shift; done ;;
            *) echo "unknown arg $1" >&2; exit 1 ;;
        esac
    done
    [[ -n "$OUT_ROOT" && ${#VARIANTS[@]} -gt 0 ]] || {
        echo "usage: --submit --out DIR [--features F] [--pca N] [--hours H] --variant lr:n_est ..." >&2
        exit 1; }
    mkdir -p "$OUT_ROOT"/{models/flat,logs/slurm,results}
    for spec in "${VARIANTS[@]}"; do
        IFS=: read -r lr nest <<< "$spec"
        tag="lr${lr//./p}n${nest}"
        echo "submit $FEATURES pca=${PCA:-None} $tag (${HOURS}h) -> $OUT_ROOT"
        sbatch --job-name="xgb_${tag}" --time="${HOURS}:00:00" \
            --output="$OUT_ROOT/logs/slurm/${tag}_%j.out" \
            --error="$OUT_ROOT/logs/slurm/${tag}_%j.err" \
            --export=ALL,LR="$lr",NEST="$nest",FEATURES="$FEATURES",PCA="$PCA",OUT_ROOT="$OUT_ROOT" \
            "$0"
    done
    echo "submitted ${#VARIANTS[@]} jobs; squeue -u \$USER -n xgb_ to monitor"
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${LR:?LR env missing}"; : "${NEST:?NEST env missing}"
: "${FEATURES:?FEATURES env missing}"; : "${OUT_ROOT:?}"
PCA="${PCA:-}"
cd "$REPO"
. env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"/{models/flat,logs/slurm,results}

echo "xgb lr=$LR n_estimators=$NEST features=$FEATURES pca=${PCA:-None} out=$OUT_ROOT  ($(date))"
CMD=(python src/pipeline/python/04_modelling.py
     --model xgb --features "$FEATURES"
     --xgb-params "{\"learning_rate\": $LR, \"n_estimators\": $NEST, \"early_stopping_rounds\": 50}"
     --out-root "$OUT_ROOT")
[[ -n "$PCA" ]] && CMD+=(--pca "$PCA")
"${CMD[@]}"
echo "DONE $(date)"
