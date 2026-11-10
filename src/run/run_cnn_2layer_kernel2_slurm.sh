#!/bin/bash
#SBATCH --job-name=cnn_k2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=runs/cnn_2layer_k17_k2/logs/slurm/%A_%a.out
#SBATCH --error=runs/cnn_2layer_k17_k2/logs/slurm/%A_%a.err

# Layer-2 kernel sweep at fixed layer-1 k1=17, C=32 on flank5-19mer.
# One k2 per array task (9 tasks: 3 5 7 9 11 13 15 17 19 odd).
# (17,17) doubles as the Stage-0 2-layer baseline. Stage 2 (channels at
# winner) reuses this script via CHANNELS override; Stage 3 (dropout/3-layer)
# is out of scope.
#
# Grid discipline: one parameter at a time — k1=17 and C=32 fixed.
# Unified protocol: 30 epochs / patience 5 / per-fold seeds.
#
# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   OUT="runs/cnn_2layer_k17_k2_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0-8 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_cnn_2layer_kernel2_slurm.sh
# Custom k2 subset (array size must match count; e.g. 3 values -> --array=0-2):
#   OUT="runs/cnn_2layer_k17_k2_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0-2 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",K2S="7 9 11",ALL src/run/run_cnn_2layer_kernel2_slurm.sh
# Stage 2 reuse (channels at winning k2, e.g. k7):
#   OUT="runs/cnn_2layer_k17_k7_chan_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",K2S="7",CHANNELS="64",ALL src/run/run_cnn_2layer_kernel2_slurm.sh
# Smoke test (2 epochs, single task):
#   OUT="runs/cnn_2layer_k2_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=2,ALL src/run/run_cnn_2layer_kernel2_slurm.sh

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
OUT="${RUN:-runs/cnn_2layer_k17_k2_$(date +%Y%m%d)}"
CHANNELS="${CHANNELS:-32}"

mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/plots" "$OUT/results"

echo "out=${OUT} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}' channels=${CHANNELS}"
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

# Parallel array: one K2 value per task, one GPU each. Safe to share OUT:
# run_tag embeds the model key (cnn_2layer_k17_k<K2>) so .pt / cv_results /
# logs / plots split per k2, no overwrite.
# shellcheck disable=SC2206
K2_LIST=(${K2S:-3 5 7 9 11 13 15 17 19})

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Local / non-array fallback: run all sequentially in one proc chain
    for K2 in "${K2_LIST[@]}"; do
        MODEL="cnn_2layer_k17_k${K2}"
        echo "=== ${MODEL} C=${CHANNELS} x ${FEAT} (epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE}) ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
        python src/pipeline/python/05_cnn_modelling.py \
            --model "$MODEL" \
            --features "$FEAT" \
            --channels "$CHANNELS" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --seed-mode "$SEED_MODE" \
            --out-root "$OUT"
        echo "--- ${MODEL} C=${CHANNELS} done ---"
    done
else
    IDX="${SLURM_ARRAY_TASK_ID}"
    if [ "$IDX" -ge "${#K2_LIST[@]}" ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range for ${#K2_LIST[@]} k2 values (${K2_LIST[*]})"
        exit 1
    fi
    K2="${K2_LIST[$IDX]}"
    MODEL="cnn_2layer_k17_k${K2}"
    echo "Array task ${IDX}: model=${MODEL} channels=${CHANNELS} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}' out=${OUT}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    python src/pipeline/python/05_cnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --channels "$CHANNELS" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --seed-mode "$SEED_MODE" \
        --out-root "$OUT"
    echo "--- ${MODEL} C=${CHANNELS} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
fi

echo "CNN 2-layer k2 sweep task completed."
