#!/bin/bash
# run_ffnn_optuna_slurm.sh — nested Optuna tuning campaign for the thesis FFNN.
#
# Plan (locked 2026-11-26, runs/<arm>/ conventions from 46): one study per
# OUTER bucket, TPE multivariate + anchors (ffnn_simple/2layer twins),
# MedianPruner on inner-fold step, budget-atomic trials (no ES), 25 trials
# per outer study by default. Array = outer folds; refit job (multi-seed
# winner refits + nested summary) chains after the array with
# --dependency=afterok, then the shared 05f plot script runs.
#
# Submit (interactive shell, repo root on the server):
#   bash src/run/run_ffnn_optuna_slurm.sh --submit
#   bash src/run/run_ffnn_optuna_slurm.sh --submit --trials-per-job 30
#   bash src/run/run_ffnn_optuna_slurm.sh --submit --smoke     # wiring check
# Options: --features F (default emb_esmc300m_win_padtoken) --trials-per-job
#          25  --hours 24  --refit-hours 4  --variant V
#          (default flank5_rmdup_updated_h6_pepcut1 = go-forward folds)
# Monitor: squeue -u $USER -n ffnn_opt
#
# Results land in runs/ffnn_optuna_<features>/ (study.db,
# trials_outer<k>_tuned.csv, best_params_outer<k>.json, then after refit:
# winner_stability.csv, nested_summary.json, best_params.json, plots/).
#SBATCH --job-name=ffnn_opt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/projects/thesis_s204692_s204581/thesis2026
TUNER="src/pipeline/python/06e_ffnn_optuna_tune.py"
PLOTTER="src/pipeline/python/05f_cnn_optuna_plot.py"   # shared CNN+FFNN

# ── submitter mode ──
if [[ "${1:-}" == "--submit" ]]; then
    shift
    TRIALS_PER_JOB="25"
    HOURS="24"
    REFIT_HOURS="4"
    FEATURES="emb_esmc300m_win_padtoken"
    VARIANT="flank5_rmdup_updated_h6_pepcut1"
    SMOKE="0"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --trials-per-job) TRIALS_PER_JOB="$2"; shift 2 ;;
            --hours)          HOURS="$2"; shift 2 ;;
            --refit-hours)    REFIT_HOURS="$2"; shift 2 ;;
            --features)       FEATURES="$2"; shift 2 ;;
            --variant)        VARIANT="$2"; shift 2 ;;
            --smoke)          SMOKE="1"; shift ;;
            *) echo "unknown arg $1" >&2; exit 1 ;;
        esac
    done
    OUT_ROOT="runs/ffnn_optuna_${FEATURES}"
    mkdir -p "$OUT_ROOT/logs/slurm"

    if [[ "$SMOKE" == "1" ]]; then
        echo "smoke: outer 0 only, 2 trials, 3 epochs -> $OUT_ROOT (study_smoke)"
        sbatch --job-name="ffnn_opt_smoke" --time="01:00:00" \
            --array=0 --output="$OUT_ROOT/logs/slurm/smoke_%A.out" \
            --error="$OUT_ROOT/logs/slurm/smoke_%A.err" \
            --export=ALL,MODE=smoke,TRIALS=2,FEATURES="$FEATURES",VARIANT="$VARIANT" \
            "$0"
        echo "submitted smoke; monitor: squeue -u \$USER -n ffnn_opt"
        exit 0
    fi

    echo "arm: ${FEATURES} variant=${VARIANT} -> $OUT_ROOT"
    echo "array job: 6 tasks (outer 0-5), ${TRIALS_PER_JOB} trials each, seeds 42+id"
    JOB_ARRAY=$(sbatch --job-name="ffnn_opt_tune" --time="${HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/tune_%A_%a.out" \
        --error="$OUT_ROOT/logs/slurm/tune_%A_%a.err" \
        --export=ALL,MODE=tune,TRIALS="$TRIALS_PER_JOB",FEATURES="$FEATURES",VARIANT="$VARIANT" \
        "$0" | awk '{print $4}')
    echo "refit job (waits for array ${JOB_ARRAY}): multi-seed winner refits + nested summary + plots"
    sbatch --job-name="ffnn_opt_refit" --time="${REFIT_HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/refit_%j.out" \
        --error="$OUT_ROOT/logs/slurm/refit_%j.err" \
        --export=ALL,MODE=refit,FEATURES="$FEATURES",VARIANT="$VARIANT" \
        --dependency="afterok:${JOB_ARRAY}" "$0"
    echo "submitted array=${JOB_ARRAY} + refit; monitor: squeue -u \$USER -n ffnn_opt"
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${MODE:?MODE env missing}"   # tune | refit | smoke
FEATURES="${FEATURES:-emb_esmc300m_win_padtoken}"
VARIANT="${VARIANT:-flank5_rmdup_updated_h6_pepcut1}"
OUTER="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((42 + OUTER))            # per-worker TPE seed: startup trials differ

cd "$REPO"
if [ -f "env_esmc_server/bin/activate" ]; then
    . env_esmc_server/bin/activate
else
    echo "ERROR: no venv at env_esmc_server/bin/activate"; exit 1
fi
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export THESIS_VARIANT="$VARIANT"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda ok')"
# optuna ships in the venv? if not, install into it (plain pip; --user is
# ILLEGAL inside a virtualenv and killed the XGB campaign's first jobs)
python -c "import optuna" 2>/dev/null || pip install optuna

if [[ "$MODE" == "smoke" ]]; then
    echo "ffnn optuna SMOKE: outer ${OUTER}, variant '${VARIANT}'  ($(date))"
    python "$TUNER" --outer "$OUTER" --smoke
    python "$PLOTTER" --model ffnn --features "$FEATURES" || \
        echo "smoke plots skipped (shared db may lack studies yet)"
elif [[ "$MODE" == "tune" ]]; then
    : "${TRIALS:?TRIALS env missing}"
    echo "ffnn optuna TUNE: outer ${OUTER}, ${TRIALS} trials, seed ${SEED}, variant '${VARIANT}'  ($(date))"
    python "$TUNER" --outer "$OUTER" --trials "$TRIALS" --seed "$SEED" --features "$FEATURES"
elif [[ "$MODE" == "refit" ]]; then
    echo "ffnn optuna REFIT: multi-seed winner refits over 6 outer studies, variant '${VARIANT}'  ($(date))"
    python "$TUNER" --refit --features "$FEATURES"
    echo "plots ($(date))"
    python "$PLOTTER" --model ffnn --features "$FEATURES"
fi
echo "DONE $(date)"
