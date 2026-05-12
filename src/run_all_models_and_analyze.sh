#!/bin/bash
# src/run_all_models_and_analyze.sh
# Runs all model x feature-set combinations, then analyzes all results.
# Strategy: env_esmc for everything, env_esmif only for ESM-IF runs.

set -u

############################################################
# Setup: cd into this script's directory
############################################################
cd "$(dirname "$0")" || { echo "FATAL: cannot cd to script dir"; exit 1; }
echo ">>> Working dir: $(pwd)"

############################################################
# Config (paths relative to src/)
############################################################
SCRIPT="04_modelling.py"
ANALYSIS_SCRIPT="05_model_analysis.py"
ENV_ESMC="../env_esmc"
ENV_ESMIF="../env_esmif"
LOG_DIR="../logs"
MODEL_DIR="../models"

ESMC_DIM=1152
ESMIF_DIM=512

# Timestamp marking the start of this batch — used to filter
# only the cv_results_*.json files produced by THIS run for analysis.
BATCH_START=$(date +%s)

MODELS=("rf" "lr")

NO_EMB_FEATURES=(
    "handcrafted_sparse"
    "handcrafted_blosum"
)

ESMC_FEATURES=(
    "esmc"
    "handcrafted_sparse_esmc"
    "handcrafted_blosum_esmc"
)

ESMIF_FEATURES=(
    "esmif"
    "handcrafted_sparse_esmif"
    "handcrafted_blosum_esmif"
)

BOTH_FEATURES=(
    "all_sparse"
    "all_blosum"
)

mkdir -p "$LOG_DIR"

############################################################
# Helpers
############################################################
activate_env () {
    env_path="$1"
    if command -v deactivate >/dev/null 2>&1; then
        deactivate 2>/dev/null || true
    fi
    if [ ! -f "${env_path}/bin/activate" ]; then
        echo "FATAL: virtualenv not found at ${env_path}/bin/activate" >&2
        exit 1
    fi
    source "${env_path}/bin/activate"
    echo ">>> Activated: ${env_path}  (python: $(which python))"
}

run_cmd () {
    desc="\$1"; shift
    echo ""
    echo "############################################################"
    echo ">>> ${desc}"
    echo ">>> CMD: $*"
    echo "############################################################"
    "$@"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "!!! FAILED (exit $rc): $*" >&2
    fi
    return $rc
}

run_feature_importance () {
    mode_label="$1"
    feat="$2"

    run_cmd "Feature importance [$mode_label] $feat" \
        bash -c "cd .. && python src/06_feature_importance.py --features '$feat' --models lr rf --top 30 --top_compare 30 --top_big 70 --out_root 'results/figures/models/feature_importance_${mode_label}'"
}

############################################################
# Block 1: env_esmc — no-embedding + ESM-C runs
############################################################
activate_env "$ENV_ESMC"

for model in "${MODELS[@]}"; do
    for feat in "${NO_EMB_FEATURES[@]}"; do
        run_cmd "[$model] $feat" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

for feat in "${NO_EMB_FEATURES[@]}"; do
    run_feature_importance "no_embedding" "$feat"
done

for model in "${MODELS[@]}"; do
    for feat in "${ESMC_FEATURES[@]}"; do
        run_cmd "[$model] $feat (PCA)" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

for feat in "${ESMC_FEATURES[@]}"; do
    run_feature_importance "esmc_pca" "$feat"
done

for model in "${MODELS[@]}"; do
    for feat in "${ESMC_FEATURES[@]}"; do
        run_cmd "[$model] $feat (no-PCA, full ${ESMC_DIM} dims/region)" \
            python "$SCRIPT" --model "$model" --features "$feat" --pca "$ESMC_DIM"
    done
done

for feat in "${ESMC_FEATURES[@]}"; do
    run_feature_importance "esmc_full" "$feat"
done

############################################################
# Block 2: env_esmif — ESM-IF + combined runs
############################################################
activate_env "$ENV_ESMIF"

for model in "${MODELS[@]}"; do
    for feat in "${ESMIF_FEATURES[@]}"; do
        run_cmd "[$model] $feat (PCA)" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

for feat in "${ESMIF_FEATURES[@]}"; do
    run_feature_importance "esmif_pca" "$feat"
done

for model in "${MODELS[@]}"; do
    for feat in "${ESMIF_FEATURES[@]}"; do
        run_cmd "[$model] $feat (no-PCA, full ${ESMIF_DIM} dims/region)" \
            python "$SCRIPT" --model "$model" --features "$feat" --pca "$ESMIF_DIM"
    done
done

for feat in "${ESMIF_FEATURES[@]}"; do
    run_feature_importance "esmif_full" "$feat"
done

for model in "${MODELS[@]}"; do
    for feat in "${BOTH_FEATURES[@]}"; do
        run_cmd "[$model] $feat (PCA)" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

for feat in "${BOTH_FEATURES[@]}"; do
    run_feature_importance "all_pca" "$feat"
done

for model in "${MODELS[@]}"; do
    for feat in "${BOTH_FEATURES[@]}"; do
        run_cmd "[$model] $feat (no-PCA, full dims)" \
            python "$SCRIPT" --model "$model" --features "$feat" --pca "$ESMC_DIM"
    done
done

for feat in "${BOTH_FEATURES[@]}"; do
    run_feature_importance "all_full" "$feat"
done

echo ""
echo "############################################################"
echo ">>> All training runs complete. Starting analysis."
echo "############################################################"

############################################################
# Block 3: Run 05_model_analysis.py on all freshly-generated results
############################################################
# Stay in env_esmif (it has matplotlib/sklearn — same deps as env_esmc).
# If env_esmif is missing matplotlib, switch to env_esmc with: activate_env "$ENV_ESMC"

# Collect cv_results JSON files newer than BATCH_START
activate_env "$ENV_ESMC"
NEW_RESULTS=()
for f in "$MODEL_DIR"/cv_results_*.json; do
    [ -f "$f" ] || continue
    file_mtime=$(stat -c %Y "$f")
    if [ "$file_mtime" -ge "$BATCH_START" ]; then
        NEW_RESULTS+=("$f")
    fi
done

echo ""
echo ">>> Found ${#NEW_RESULTS[@]} new result file(s) from this batch."

if [ ${#NEW_RESULTS[@]} -eq 0 ]; then
    echo "!!! No new cv_results_*.json files found. Skipping analysis."
else
    run_cmd "Running 05_model_analysis.py on all new results" \
        python "$ANALYSIS_SCRIPT" --results "${NEW_RESULTS[@]}"
fi

echo ""
echo "############################################################"
echo ">>> All done: training + analysis."
echo "############################################################"
