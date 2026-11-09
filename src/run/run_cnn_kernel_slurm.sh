#!/bin/bash
#SBATCH --job-name=cnn_kernel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
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

EPOCHS="${EPOCHS:-20}"
FEAT="${FEAT:-emb_esmc300m_win_padtoken}"
OUT="${RUN:-runs/cnn_kernel}"
mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/plots" "$OUT/results"

echo "out=${OUT} features=${FEAT} epochs=${EPOCHS}"

# Parallel array: one MODEL per task, one GPU each. Safe to share OUT:
# run_tag splits .pt / cv_results / logs / plots per model, no overwrite.
# shellcheck disable=SC2206
MODELS_LIST=(${MODELS:-cnn_simple_k11 cnn_simple_k13 cnn_simple_k15 cnn_simple_k17 cnn_simple_k19 cnn_simple_k21 cnn_simple_k23 cnn_simple_k25 cnn_simple_k27 cnn_simple_k29})

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Local / non-array fallback: run all sequentially in one proc chain
    for MODEL in "${MODELS_LIST[@]}"; do
        echo "=== ${MODEL} x ${FEAT} (epochs=${EPOCHS}) ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
        python src/pipeline/python/05_cnn_modelling.py \
            --model "$MODEL" \
            --features "$FEAT" \
            --epochs "$EPOCHS" \
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
    echo "Array task ${IDX}: model=${MODEL} features=${FEAT} epochs=${EPOCHS} out=${OUT}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    python src/pipeline/python/05_cnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --epochs "$EPOCHS" \
        --out-root "$OUT"
    echo "--- ${MODEL} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
fi

echo "CNN kernel sweep task completed."

# Continue in SAME experiment folder (appends; run_tag splits outputs, no overwrite):
#   OUT="runs/cnn_kernel_20261106"; sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_cnn_kernel_slurm.sh
# Custom subset (array size must match count; e.g. 3 models -> --array=0-2):
#   OUT="runs/cnn_kernel_20261106"; sbatch --array=0-2 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODELS="cnn_simple_k5 cnn_simple_k7 cnn_simple_k9",ALL src/run/run_cnn_kernel_slurm.sh

# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   OUT="runs/cnn_kernel_20261106"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_cnn_kernel_slurm.sh
# Override epochs:
#   OUT="runs/cnn_kernel_20261106"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=30,ALL src/run/run_cnn_kernel_slurm.sh
# Smoke test (2 epochs, single task):
#   OUT="runs/cnn_kernel_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=2,ALL src/run/run_cnn_kernel_slurm.sh
