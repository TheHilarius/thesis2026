#!/bin/bash
#SBATCH --job-name=ffnn_width
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=runs/ffnn_width/logs/slurm/%A_%a.out
#SBATCH --error=runs/ffnn_width/logs/slurm/%A_%a.err

# FFNN width sweep: analog of run_cnn_channel_slurm.sh. One hidden width
# per array task on ffnn_simple (Linear(in->w) -> ReLU -> Linear(w->1)).
# Same data as the CNN campaigns: emb_esmc300m_win_padtoken windows,
# flattened (L,D)->(L*D) per row, 30 epochs / patience 5 / per-fold seeds.
#
# Default VARIANT=flank5_rmdup (config default, N=45,278). Set VARIANT=flank5
# for byte-comparison against the legacy CNN flank5 campaign rows.
#
# W=32 is included by default so the folder is self-contained (baseline
# point under the same protocol); drop it via
# WIDTHS="16 64 128 256" --array=0-3 if you prefer to reuse that point.
#
# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   OUT="runs/ffnn_width_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_ffnn_width_slurm.sh
# Custom width grid (array size must match count; e.g. 3 widths -> --array=0-2):
#   OUT="runs/ffnn_width_$(date +%Y%m%d)"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0-2 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",WIDTHS="32 64 128",ALL src/run/run_ffnn_width_slurm.sh
# Smoke test (2 epochs, single task):
#   OUT="runs/ffnn_width_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=2,ALL src/run/run_ffnn_width_slurm.sh

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
VARIANT="${VARIANT:-flank5_rmdup}"
FEAT="${FEAT:-emb_esmc300m_win_padtoken}"
OUT="${RUN:-runs/ffnn_width}"
MODEL="${MODEL:-ffnn_simple}"

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

# Parallel array: one WIDTH per task, one GPU each. Safe to share OUT:
# run_tag embeds _wW so .pt / cv_results / logs / plots split per width.
# shellcheck disable=SC2206
WIDTHS_LIST=(${WIDTHS:-32 64 128 256 512})

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Local / non-array fallback: run all sequentially in one proc chain
    for W in "${WIDTHS_LIST[@]}"; do
        echo "=== ${MODEL} w=${W} x ${FEAT} (epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE}) ==="
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
        python src/pipeline/python/06a_ffnn_modelling.py \
            --model "$MODEL" \
            --features "$FEAT" \
            --width "$W" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --seed-mode "$SEED_MODE" \
            --out-root "$OUT"
        echo "--- ${MODEL} w=${W} done ---"
    done
else
    IDX="${SLURM_ARRAY_TASK_ID}"
    if [ "$IDX" -ge "${#WIDTHS_LIST[@]}" ]; then
        echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range for ${#WIDTHS_LIST[@]} widths (${WIDTHS_LIST[*]})"
        exit 1
    fi
    W="${WIDTHS_LIST[$IDX]}"
    echo "Array task ${IDX}: model=${MODEL} width=${W} features=${FEAT} epochs=${EPOCHS} patience=${PATIENCE} seed_mode=${SEED_MODE} variant='${VARIANT}' out=${OUT}"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    python src/pipeline/python/06a_ffnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --width "$W" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --seed-mode "$SEED_MODE" \
        --out-root "$OUT"
    echo "--- ${MODEL} w=${W} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
fi

echo "FFNN width sweep task completed."
