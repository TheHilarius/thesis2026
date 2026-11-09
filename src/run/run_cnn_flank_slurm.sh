#!/bin/bash
#SBATCH --job-name=cnn_flank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=runs/cnn_flank/logs/slurm/%A_%a.out
#SBATCH --error=runs/cnn_flank/logs/slurm/%A_%a.err

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
# Separate proc per task; each frees GPU on exit
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-cnn_simple_k5}"
FEAT="${FEAT:-emb_esmc300m_win_padtoken}"
EPOCHS="${EPOCHS:-20}"
OUT="${RUN:-runs/cnn_flank}"

run_one() {
    local SUB="$1" VARIANT="$2"
    local TASK_OUT="$OUT/$SUB"
    mkdir -p "$TASK_OUT/models" "$TASK_OUT/logs/slurm" "$TASK_OUT/plots" "$TASK_OUT/results"
    echo "=== task SUB=${SUB} VARIANT='${VARIANT}' model=${MODEL} features=${FEAT} epochs=${EPOCHS} out=${TASK_OUT} ==="
    if [ -n "$VARIANT" ]; then
        export THESIS_VARIANT="$VARIANT"
    else
        unset THESIS_VARIANT || true
    fi
    # Log resolved paths (variant redirect visible here)
    python -c "
import sys; sys.path.insert(0,'src/pipeline/python');
from config import SPLIT_DATA_PATH, get_embedding_source;
src = get_embedding_source('esmc300m_win_padtoken');
print('split:', SPLIT_DATA_PATH);
print('h5   :', src['prepared_path']);
import pathlib; assert pathlib.Path(src['prepared_path']).exists(), 'MISSING H5';
"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
    python src/pipeline/python/05_cnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --epochs "$EPOCHS" \
        --out-root "$TASK_OUT"
    echo "--- ${SUB} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Local / non-array fallback: both back-to-back
    run_one full ""
    run_one flank5 "flank5"
else
    case "${SLURM_ARRAY_TASK_ID}" in
        0) run_one full "" ;;
        1) run_one flank5 "flank5" ;;
        *) echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (0-1)"; exit 1 ;;
    esac
fi

echo "CNN flank task completed (model=${MODEL}, k=5)."

# New experiment folder, both tasks in parallel (appends under one parent):
#   OUT="runs/cnn_flank_20261109"; mkdir -p "$OUT/logs/slurm" && sbatch --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",ALL src/run/run_cnn_flank_slurm.sh
# Override model/epochs:
#   OUT="runs/cnn_flank_20261109"; sbatch --array=0-1 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",MODEL=cnn_simple_k5,EPOCHS=20,ALL src/run/run_cnn_flank_slurm.sh
# Smoke test (2 epochs, full only):
#   OUT="runs/cnn_flank_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --array=0 --output="$OUT/logs/slurm/%A_%a.out" --error="$OUT/logs/slurm/%A_%a.err" --export=RUN="$OUT",EPOCHS=2,ALL src/run/run_cnn_flank_slurm.sh
