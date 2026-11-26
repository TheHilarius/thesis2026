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
#   bash src/run/run_xgb_optuna_slurm.sh --submit --features handcrafted_blosum50
# Options: --features F (FEATURE_SETS key; default blosum50; each arm gets
#          its own runs/xgb_optuna_<features>/ dir + study, so trials never
#          mix)  --trials-per-job 50  --hours 8  --polish-hours 3
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
    FEATURES="blosum50"
    FULL="0"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --trials-per-job) TRIALS_PER_JOB="$2"; shift 2 ;;
            --hours)          HOURS="$2"; shift 2 ;;
            --polish-hours)   POLISH_HOURS="$2"; shift 2 ;;
            --features)       FEATURES="$2"; shift 2 ;;
            --full)           FULL="1"; shift ;;
            *) echo "unknown arg $1" >&2; exit 1 ;;
        esac
    done
    # blosum50 keeps the original campaign dir; other arms get their own
    if [[ "$FEATURES" == "blosum50" ]]; then
        OUT_ROOT="runs/xgb_optuna"; SUFFIX=""
    else
        OUT_ROOT="runs/xgb_optuna_${FEATURES}"; SUFFIX="_${FEATURES}"
    fi
    FSUF=""
    if [[ "$FULL" == "1" ]]; then FSUF="_full"; fi
    mkdir -p "$OUT_ROOT"/{logs/slurm,plots}

    echo "arm: ${FEATURES}${FSUF:+ (full 9-param space)} -> $OUT_ROOT (50 trials x 2 jobs + polish)"
    echo "tune job A: ${TRIALS_PER_JOB} trials, seed 42"
    JOB_A=$(sbatch --job-name="xgb_opt_a${SUFFIX}${FSUF}" --time="${HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/tune_a${FSUF}_%j.out" \
        --error="$OUT_ROOT/logs/slurm/tune_a${FSUF}_%j.err" \
        --export=ALL,MODE=tune,TRIALS="$TRIALS_PER_JOB",SEED=42,FEATURES="$FEATURES",FULL="$FULL" \
        "$0" | awk '{print $4}')
    echo "tune job B: ${TRIALS_PER_JOB} trials, seed 143"
    JOB_B=$(sbatch --job-name="xgb_opt_b${SUFFIX}${FSUF}" --time="${HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/tune_b${FSUF}_%j.out" \
        --error="$OUT_ROOT/logs/slurm/tune_b${FSUF}_%j.err" \
        --export=ALL,MODE=tune,TRIALS="$TRIALS_PER_JOB",SEED=143,FEATURES="$FEATURES",FULL="$FULL" \
        "$0" | awk '{print $4}')
    echo "polish job (waits for $JOB_A and $JOB_B)"
    sbatch --job-name="xgb_opt_p${SUFFIX}${FSUF}" --time="${POLISH_HOURS}:00:00" \
        --output="$OUT_ROOT/logs/slurm/polish${FSUF}_%j.out" \
        --error="$OUT_ROOT/logs/slurm/polish${FSUF}_%j.err" \
        --export=ALL,MODE=polish,FEATURES="$FEATURES",FULL="$FULL" \
        --dependency="afterok:${JOB_A}:${JOB_B}" "$0"
    echo "submitted A=$JOB_A B=$JOB_B + polish; monitor: squeue -u \$USER -n xgb_opt"
    exit 0
fi

# ── worker mode (under SLURM) ──
: "${MODE:?MODE env missing}"   # tune | polish
FEATURES="${FEATURES:-blosum50}"
FULL="${FULL:-0}"
EXTRA=""
if [[ "$FULL" == "1" ]]; then EXTRA="--full-space"; fi
cd "$REPO"
. env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
# (the tuner mkdirs its own arm dir; slurm log dirs exist from the submitter)

# optuna ships in the venv? if not, install into it (plain pip; --user is
# ILLEGAL inside a virtualenv and killed the first campaign's jobs)
python -c "import optuna" 2>/dev/null || pip install optuna

TUNER="src/investigation/46_xgb_optuna_tune.py"
if [[ "$MODE" == "tune" ]]; then
    : "${TRIALS:?TRIALS env missing}"; : "${SEED:?SEED env missing}"
    echo "xgb_optuna tune: arm=${FEATURES}${EXTRA:+ (full space)} ${TRIALS} trials, seed ${SEED}  ($(date))"
    python "$TUNER" --trials "$TRIALS" --seed "$SEED" --features "$FEATURES" $EXTRA
elif [[ "$MODE" == "polish" ]]; then
    echo "xgb_optuna polish: arm=${FEATURES}${EXTRA:+ (full space)}  ($(date))"
    python "$TUNER" --polish --features "$FEATURES" $EXTRA
fi
echo "DONE $(date)"
