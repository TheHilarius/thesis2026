#!/bin/bash
#SBATCH --job-name=cnn_chan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/cnn_channels/logs/slurm/%A_%a.out
#SBATCH --error=runs/cnn_channels/logs/slurm/%A_%a.err

# Channel sweep for ONE kernel: the AUC winner of the k3-k19 flank5 curve.
# Separate experiment -> own script + own campaign folder
# (run_cnn_kernel_slurm.sh stays kernel-only, records untouched).
#
# One channel count per array task. MODEL has no default on purpose: set it
# at submit time to the winning registry key (k3 baseline = cnn_simple,
# k5-k19 = cnn_simple_kK). VARIANT defaults to flank5 (channel question is
# asked on the winning 19mer data).
#
# Unified protocol: 30 epochs / patience 5 / per-fold seeds.
# C=32 is included by default so the folder is self-contained (it duplicates
# the kernel-curve point under the same protocol; drop it via
# CHANNELS="8 16 64 128" --array=0-3 if you prefer to reuse that point).
#
# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   OUT="runs/cnn_channels_k<K>_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODEL="cnn_simple_k<K>",ALL src/run/run_cnn_channel_slurm.sh
# Example with k7 as winner:
#   OUT="runs/cnn_channels_k7_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODEL="cnn_simple_k7",ALL src/run/run_cnn_channel_slurm.sh
# Custom channel grid (array size must match count; e.g. 3 channels -> --array=0-2):
#   OUT="runs/cnn_channels_k7_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0-2 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODEL="cnn_simple_k7",CHANNELS="16 64 128",ALL src/run/run_cnn_channel_slurm.sh
# Smoke test (2 epochs, single task):
#   OUT="runs/cnn_channels_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODEL="cnn_simple_k5",EPOCHS=2,ALL src/run/run_cnn_channel_slurm.sh

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
OUT="${RUN:-runs/cnn_channels}"
MODEL="${MODEL:-}"

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL unset. Pass the winning kernel key at submit time, e.g."
    echo "  --export=RUN=...,MODEL=cnn_simple_k7,ALL src/run/run_cnn_channel_slurm.sh"
    echo "Valid keys: cnn_simple (k3), cnn_simple_k5 .. cnn_simple_k29 (odd k)."
    exit 1
fi

mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/plots" "$OUT/results"

echo "out=${OUT} model=${MODEL} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}'"
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

# Parallel array: one CHANNELS value per task, one GPU each. Safe to share OUT:
# run_tag embeds _cC so .pt / cv_results / logs / plots split per width.
# shellcheck disable=SC2206
CHANNELS_LIST=(${CHANNELS:-8 16 32 64 128})

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Local / non-array fallback: run all sequentially in one proc chain
    for CH in "${CHANNELS_LIST[@]}"; do
        echo "=== ${MODEL} C=${CH} x ${FEAT} (epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE}) ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
        python src/pipeline/python/05_cnn_modelling.py \
            --model "$MODEL" \
            --features "$FEAT" \
            --channels "$CH" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --seed-mode "$SEED_MODE" \
            --out-root "$OUT"
        echo "--- ${MODEL} C=${CH} done ---"
    done
else
    IDX="${SLURM_ARRAY_TASK_ID}"
    if [ "$IDX" -ge "${#CHANNELS_LIST[@]}" ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range for ${#CHANNELS_LIST[@]} channels (${CHANNELS_LIST[*]})"
        exit 1
    fi
    CH="${CHANNELS_LIST[$IDX]}"
    echo "Array task ${IDX}: model=${MODEL} channels=${CH} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}' out=${OUT}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    python src/pipeline/python/05_cnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --channels "$CH" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --seed-mode "$SEED_MODE" \
        --out-root "$OUT"
    echo "--- ${MODEL} C=${CH} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
fi

echo "CNN channel sweep task completed."
