#!/bin/bash
# src/run/run_all_models.sh
# Runs all model x feature-set combinations for the MHC-I pipeline.
# Run from inside the src/ directory:
#   cd src
#   ./run_all_models.sh

set -u

############################################################
# Setup: cd into this script's directory so relative paths work
############################################################
cd "$(dirname "$0")" || { echo "FATAL: cannot cd to script dir"; exit 1; }
echo ">>> Working dir: $(pwd)"

############################################################
# Config (all paths relative to src/)
############################################################
SCRIPT="../pipeline/python/04_modelling.py"
ENV_ESMC="../../env_esmc"
ENV_ESMIF="../../env_esmif"
LOG_DIR="../../logs"

ESMC_DIM=1152
ESMIF_DIM=512

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
    local env_path="$1"
    if command -v deactivate >/dev/null 2>&1; then
        deactivate 2>/dev/null || true
    fi
    if [ ! -f "${env_path}/bin/activate" ]; then
        echo "FATAL: virtualenv not found at ${env_path}/bin/activate" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${env_path}/bin/activate"
    echo ">>> Activated: ${env_path}  (python: $(which python))"
}

run_cmd () {
    local desc="\$1"; shift
    echo ""
    echo "############################################################"
    echo ">>> ${desc}"
    echo ">>> CMD: $*"
    echo "############################################################"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "!!! FAILED (exit $rc): $*" >&2
    fi
    return $rc
}

############################################################
# Run blocks
############################################################

# --- 1. No-embedding feature sets ---
activate_env "$ENV_ESMC"
for model in "${MODELS[@]}"; do
    for feat in "${NO_EMB_FEATURES[@]}"; do
        run_cmd "[$model] $feat" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

# --- 2. ESM-C feature sets, with PCA (default) ---
activate_env "$ENV_ESMC"
for model in "${MODELS[@]}"; do
    for feat in "${ESMC_FEATURES[@]}"; do
        run_cmd "[$model] $feat (PCA)" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

# --- 3. ESM-C feature sets, NO PCA (full dims) ---
for model in "${MODELS[@]}"; do
    for feat in "${ESMC_FEATURES[@]}"; do
        run_cmd "[$model] $feat (no-PCA, full ${ESMC_DIM} dims/region)" \
            python "$SCRIPT" --model "$model" --features "$feat" --pca "$ESMC_DIM"
    done
done

# --- 4. ESM-IF feature sets, with PCA (default) ---
activate_env "$ENV_ESMIF"
for model in "${MODELS[@]}"; do
    for feat in "${ESMIF_FEATURES[@]}"; do
        run_cmd "[$model] $feat (PCA)" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

# --- 5. ESM-IF feature sets, NO PCA (full dims) ---
for model in "${MODELS[@]}"; do
    for feat in "${ESMIF_FEATURES[@]}"; do
        run_cmd "[$model] $feat (no-PCA, full ${ESMIF_DIM} dims/region)" \
            python "$SCRIPT" --model "$model" --features "$feat" --pca "$ESMIF_DIM"
    done
done

# --- 6. Combined ESM-C + ESM-IF feature sets ---
activate_env "$ENV_ESMIF"
for model in "${MODELS[@]}"; do
    for feat in "${BOTH_FEATURES[@]}"; do
        run_cmd "[$model] $feat (PCA)" \
            python "$SCRIPT" --model "$model" --features "$feat"
    done
done

for model in "${MODELS[@]}"; do
    for feat in "${BOTH_FEATURES[@]}"; do
        run_cmd "[$model] $feat (no-PCA, full dims)" \
            python "$SCRIPT" --model "$model" --features "$feat" --pca "$ESMC_DIM"
    done
done

echo ""
echo "############################################################"
echo ">>> All runs complete."
echo "############################################################"
