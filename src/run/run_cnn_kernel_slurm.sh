#!/bin/bash
#SBATCH --job-name=cnn_kernel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/cnn_kernel/logs/slurm/%j.out
#SBATCH --error=runs/cnn_kernel/logs/slurm/%j.err

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

for MODEL in cnn_simple_k5 cnn_simple_k7 cnn_simple_k9; do
    echo "=== ${MODEL} x ${FEAT} (epochs=${EPOCHS}) ==="
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    python src/pipeline/python/05_cnn_modelling.py \
        --model "$MODEL" \
        --features "$FEAT" \
        --epochs "$EPOCHS" \
        --out-root "$OUT"
    echo "--- ${MODEL} done ---"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
done

echo "CNN kernel sweep (k5/k7/k9 sequential, one GPU, separate procs) completed."

# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   OUT="runs/cnn_kernel_20261106"; mkdir -p "$OUT/logs/slurm" && sbatch --export=RUN="$OUT",ALL src/run/run_cnn_kernel_slurm.sh
# Override epochs:
#   OUT="runs/cnn_kernel_20261106"; mkdir -p "$OUT/logs/slurm" && sbatch --export=RUN="$OUT",EPOCHS=30,ALL src/run/run_cnn_kernel_slurm.sh
# Smoke test (2 epochs):
#   OUT="runs/cnn_kernel_smoke"; mkdir -p "$OUT/logs/slurm" && sbatch --export=RUN="$OUT",EPOCHS=2,ALL src/run/run_cnn_kernel_slurm.sh
