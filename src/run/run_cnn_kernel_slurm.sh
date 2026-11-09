#!/bin/bash
#SBATCH --job-name=cnn_kernel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=runs/cnn_kernel/logs/slurm/%A_%a.out
#SBATCH --error=runs/cnn_kernel/logs/slurm/%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No venv at env_esmc_server/bin/activate"
    exit 1
fi

# Hard fail without GPU (login node has no CUDA; GPU node must)
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda ok')"
nvidia-smi  # memory baseline in log for right-sizing --mem later

mkdir -p logs
export PYTHONUNBUFFERED=1
# Reduce CUDA fragmentation across sequential runs; each python proc frees on exit
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
SEED_MODE="${SEED_MODE:-perfold}"
VARIANT="${VARIANT:-flank5}"
FEAT="${FEAT:-emb_esmc300m_win_padtoken}"
OUT="${RUN:-runs/cnn_kernel}"
mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/plots" "$OUT/results"

echo "out=${OUT} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}'"
if [ -n "$VARIANT" ]; then
    export THESIS_VARIANT="$VARIANT"
fi
# Log resolved data paths (variant redirect visible here)
python -c "
import sys; sys.path.insert(0,'src/pipeline/python')
from config import get_feature_set_config, get_feature_component, get_embedding_source
feat = get_feature_set_config('$FEAT')
emb = [c for c in feat['components'] if get_feature_component(c)['type'] == 'embedding_windows'][0]
src = get_embedding_source(get_feature_component(emb)['embedding_key'])
print('h5   :', src['prepared_path'])
"

# Parallel array: one MODEL per task, one GPU each. Safe to share OUT:
# run_tag splits .pt / cv_results / logs / plots per model, no overwrite.
# Default = canonical k3-k19 curve on flank5 (9 models, array 0-8).
# shellcheck disable=SC2206
MODELS_LIST=(${MODELS:-cnn_simple cnn_simple_k5 cnn_simple_k7 cnn_simple_k9 cnn_simple_k11 cnn_simple_k13 cnn_simple_k15 cnn_simple_k17 cnn_simple_k19})

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Local / non-array fallback: run all sequentially in one proc chain
    for MODEL in "${MODELS_LIST[@]}"; do
        echo "=== ${MODEL} x ${FEAT} (epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}') ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
        python src/pipeline/python/05_cnn_modelling.py \
            --model "$MODEL" \
            --features "$FEAT" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --seed-mode "$SEED_MODE" \
            --out-root "$OUT"
        echo "--- ${MODEL} done ---"
    done
else
    IDX="${SLURM_ARRAY_TASK_ID}"
    if [ "$IDX" -ge "${#MODELS_LIST[@]}" ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range for ${#MODELS_LIST[@]} models (${MODELS_LIST[*]})"
        exit 1
    fi
    MODEL="${MODELS_LIST[$IDX]}"
    echo "Array task ${IDX}: model=${MODEL} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}' out=${OUT}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    python src/pipeline/python/05_cnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --seed-mode "$SEED_MODE" \
        --out-root "$OUT"
    echo "--- ${MODEL} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
fi

echo "CNN kernel sweep task completed."

# Canonical k3-k19 on FLANK5-19mer (defaults: VARIANT=flank5, 30/5/perfold).
# 9 tasks, ~1-2h each (19mer faster than full windows):
#   OUT="runs/cnn_kernel3_19_flank5_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0-8 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_cnn_kernel_slurm.sh
# Full-29mer override (variant empty):
#   OUT="runs/cnn_kernel3_19_full_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0-8 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",VARIANT=,MODELS="cnn_simple cnn_simple_k5 cnn_simple_k7 cnn_simple_k9 cnn_simple_k11 cnn_simple_k13 cnn_simple_k15 cnn_simple_k17 cnn_simple_k19",ALL src/run/run_cnn_kernel_slurm.sh
# Legacy large-k sweep (k11-k29, old 20/3 fixed-seed protocol already recorded):
#   OUT="runs/cnn_kernel_20261106"; sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=20,PATIENCE=3,SEED_MODE=fixed,ALL src/run/run_cnn_kernel_slurm.sh
# Custom subset (array size must match count; e.g. 3 models -> --array=0-2):
#   OUT="runs/cnn_kernel_20261106"; sbatch --array=0-2 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODELS="cnn_simple_k5 cnn_simple_k7 cnn_simple_k9",ALL src/run/run_cnn_kernel_slurm.sh

# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   OUT="runs/cnn_kernel_20261106"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_cnn_kernel_slurm.sh
# Override epochs:
#   OUT="runs/cnn_kernel_20261106"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=30,ALL src/run/run_cnn_kernel_slurm.sh
# Smoke test (2 epochs, single task):
#   OUT="runs/cnn_kernel_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=2,ALL src/run/run_cnn_kernel_slurm.sh
