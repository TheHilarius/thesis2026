#!/bin/bash
# run_pca_k.sh — one XGB PCA-k job per file; per-k resources via submitter mode.
#
# Submit (interactive shell, repo root):
#   bash src/run/run_pca_k.sh --submit \
#     --feature emb_esmif_eos_bos_repeat --out runs/esmif_eos_sweep \
#     --k 345:8:16:12  2125:12:32:24  2974:12:32:24  6700:12:32:48  11243:12:56:96
#   (--k spec = k:cpus:memGB:hours, one sbatch job each)
#
# Worker mode is what SLURM executes (env: FEATURE, OUT_ROOT, K).
#SBATCH --job-name=pca_k
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO=/home/projects/thesis_s204692_s204581/thesis2026

# ── submitter mode ──
if [[ "${1:-}" == "--submit" ]]; then
    shift
    FEATURE="" OUT_ROOT=""
    K_SPECS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --feature) FEATURE="$2"; shift 2 ;;
            --out)     OUT_ROOT="$2"; shift 2 ;;
            --k)       shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                           K_SPECS+=("$1"); shift; done ;;
            *) echo "unknown arg: $1" >&2; exit 1 ;;
        esac
    done
    [[ -n "$FEATURE" && -n "$OUT_ROOT" && ${#K_SPECS[@]} -gt 0 ]] || {
        echo "usage: --submit --feature F --out DIR --k k:cpus:mem:hrs ..." >&2; exit 1; }
    mkdir -p "$OUT_ROOT"/{models,logs/slurm,results,plots,data}
    ln -sfn ../../../data/processed/df_all_with_folds.csv \
        "$OUT_ROOT/data/df_all_with_folds.csv" 2>/dev/null || true
    for spec in "${K_SPECS[@]}"; do
        IFS=: read -r k cpus mem hrs <<< "$spec"
        echo "submit k=$k  cpus=$cpus  mem=${mem}G  time=${hrs}h  -> $OUT_ROOT"
        sbatch --job-name="${FEATURE}_k${k}" \
            --cpus-per-task="$cpus" --mem="${mem}G" --time="${hrs}:00:00" \
            --output="$OUT_ROOT/logs/slurm/k${k}_%j.out" \
            --error="$OUT_ROOT/logs/slurm/k${k}_%j.err" \
            --export=ALL,FEATURE="$FEATURE",OUT_ROOT="$OUT_ROOT",K="$k" \
            "$0"
    done
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${FEATURE:?FEATURE env missing}"; : "${OUT_ROOT:?}"; : "${K:?}"
cd "$REPO"
. env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
echo "k=$K feature=$FEATURE out=$OUT_ROOT  ($(date))"
python src/pipeline/python/04b_run_matrix.py --pca-mode flat --no-aggregate \
    --out-root "$OUT_ROOT" --pca-sweep "xgb:${FEATURE}:${K}"
