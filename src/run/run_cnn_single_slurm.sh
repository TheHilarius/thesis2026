#!/bin/bash
#SBATCH --job-name=cnn_simple
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cnn_simple_%j.out
#SBATCH --error=logs/cnn_simple_%j.err

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

mkdir -p logs
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-cnn_simple}"
FEAT="${FEAT:-emb_esmc300m_win_padtoken}"
EPOCHS="${EPOCHS:-20}"
OUT="${RUN:-.}"

echo "model=${MODEL} features=${FEAT} epochs=${EPOCHS} out=${OUT}"

python src/pipeline/python/05_cnn_modelling.py \
  --model "$MODEL" \
  --features "$FEAT" \
  --epochs "$EPOCHS" \
  --out-root "$OUT"

echo "CNN single run completed."

# Submit with a timestamped run folder (paste on hub1):
#   RUN="runs/cnn_simple_$(date +%Y%m%d_%H%M)"; mkdir -p "$RUN"/{models,logs/slurm} && sbatch --job-name=cnn_simple --output="$RUN/logs/slurm/%j.out" --error="$RUN/logs/slurm/%j.err" --export=RUN="$RUN",ALL src/run/run_cnn_single_slurm.sh
# Smoke first (2 epochs, then check cv_results JSON):
#   RUN="runs/cnn_smoke_$(date +%Y%m%d_%H%M)"; mkdir -p "$RUN"/{models,logs/slurm} && sbatch --export=RUN="$RUN",EPOCHS=2,ALL src/run/run_cnn_single_slurm.sh
