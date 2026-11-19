#!/bin/bash
# run_xgb_optuna_slurm.sh — Optuna tuning campaign for the thesis XGB model.
#
# Locked plan (engram #420): blosum50 flat on flank5_rmdup, objective =
# 6-bucket mean AUC-ROC, fixed lr=0.1/n_estimators=1000, no early stopping,
# TPE seed 42, 100 trials split across TWO parallel 8-CPU jobs sharing one
# SQLite study, then a low-LR polish job (4 candidates) chained with
# --dependency=afterok.
#
# Submit (interactive shell, repo root on the server):
#   bash src/run/run_xgb_optuna_slurm.sh --submit
# Options: --trials-per-job 50  --hours 8  --polish-hours 3
# Monitor: squeue -u \$USER -n xgb_opt
#
# Results land in runs/xgb_optuna/ (study.db, trials_tuned.csv,
# best_params_tuned.json, polish_results.json, best_params.json).
#SBATCH --job-name=xgb_opt
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO=/home/projects/thesis_s204692_s204581/thesis2026

# ── submitter mode ──
if [[ "${1:-}" == "--submit" ]]; then
    shift
    TRIALS_PER_JOB="50"
    HOURS="8"
    POLISH_HOURS="3"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --trials-per-job) TRIALS_PER_JOB="$2"; shift 2 ;;
            --hours)          HOURS="$2"; shift 2 ;;
            --polish-hours)   POLISH_HOURS="$2"; shift 2 ;;
            *) echo "unknown arg $1" >&2; exit 1 ;;
        esac
    done
    OUT_ROOT="runs/xgb_optuna"
    mkdir -p "$OUT_ROOT"/{logs/slurm,plots}

    echo "tune job A: ${TRIALS_PER_JOB} trials, seed 42 -> $OUT_ROOT"
    JOB_A=$(sbatch --job-name="xgb_opt_a" --time="${HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/tune_a_%j.out" \
        --error="$OUT_ROOT/logs/slurm/tune_a_%j.err" \
        --export=ALL,MODE=tune,TRIALS="$TRIALS_PER_JOB",SEED=42 \
        "$0" | awk '{print $4}')
    echo "tune job B: ${TRIALS_PER_JOB} trials, seed 143 -> $OUT_ROOT"
    JOB_B=$(sbatch --job-name="xgb_opt_b" --time="${HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/tune_b_%j.out" \
        --error="$OUT_ROOT/logs/slurm/tune_b_%j.err" \
        --export=ALL,MODE=tune,TRIALS="$TRIALS_PER_JOB",SEED=143 \
        "$0" | awk '{print $4}')
    echo "polish job (waits for $JOB_A and $JOB_B)"
    sbatch --job-name="xgb_opt_p" --time="${POLISH_HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/polish_%j.out" \
        --error="$OUT_ROOT/logs/slurm/polish_%j.err" \
        --export=ALL,MODE=polish \
        --dependency="afterok:${JOB_A}:${JOB_B}" "$0"
    echo "submitted A=$JOB_A B=$JOB_B + polish; monitor: squeue -u \$USER -n xgb_opt"
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${MODE:?MODE env missing}"   # tune | polish
cd "$REPO"
. env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p runs/xgb_optuna/{logs,plots}

# optuna ships in the venv? if not, user-install once per node
python -c "import optuna" 2>/dev/null || pip install --user optuna

TUNER="src/investigation/46_xgb_optuna_tune.py"
if [[ "$MODE" == "tune" ]]; then
    : "${TRIALS:?TRIALS env missing}"; : "${SEED:?SEED env missing}"
    echo "xgb_optuna tune: ${TRIALS} trials, seed ${SEED}  ($(date))"
    python "$TUNER" --trials "$TRIALS" --seed "$SEED"
elif [[ "$MODE" == "polish" ]]; then
    echo "xgb_optuna polish  ($(date))"
    python "$TUNER" --polish
fi
echo "DONE $(date)"
