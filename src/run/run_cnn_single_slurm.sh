#!/bin/bash
#SBATCH --job-name=cnn_simple
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/cnn_cv/logs/slurm/%j.out
#SBATCH --error=runs/cnn_cv/logs/slurm/%j.err

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

MODEL="${MODEL:-cnn_simple}"
FEAT="${FEAT:-emb_esmc300m_win_padtoken}"
EPOCHS="${EPOCHS:-20}"
OUT="${RUN:-runs/cnn_cv}"
mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/plots"

echo "model=${MODEL} features=${FEAT} epochs=${EPOCHS} out=${OUT}"

python src/pipeline/python/05_cnn_modelling.py \
  --model "$MODEL" \
  --features "$FEAT" \
  --epochs "$EPOCHS" \
  --out-root "$OUT"

echo "CNN single run completed."

# Submit into the fixed campaign folder (paste on hub1).
# NOTE: create the slurm log dir first -- sbatch will not make it for you:
#   mkdir -p runs/cnn_cv/logs/slurm && sbatch --export=RUN="runs/cnn_cv",EPOCHS=2,ALL src/run/run_cnn_single_slurm.sh
# Full run (defaults: cnn_simple, emb_esmc300m_win_padtoken, 20 epochs):
#   mkdir -p runs/cnn_cv/logs/slurm && sbatch --export=RUN="runs/cnn_cv",ALL src/run/run_cnn_single_slurm.sh
