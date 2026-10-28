#!/bin/bash
#SBATCH --job-name=cnn_2layer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/cnn_2layer/logs/slurm/%j.out
#SBATCH --error=runs/cnn_2layer/logs/slurm/%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No venv at env_esmc_server/bin/activate"
    exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda ok')"
nvidia-smi

mkdir -p logs
export PYTHONUNBUFFERED=1

EPOCHS="${EPOCHS:-20}"
OUT="runs/cnn_2layer"
mkdir -p "$OUT/models" "$OUT/logs/slurm" "$OUT/plots" "$OUT/results"

echo "model=cnn_2layer epochs=${EPOCHS} out=${OUT}"

python src/pipeline/python/05_cnn_modelling.py \
  --model cnn_2layer \
  --features emb_esmc300m_win_padtoken \
  --epochs "$EPOCHS" \
  --out-root "$OUT"

python src/pipeline/python/05_cnn_modelling.py \
  --model cnn_2layer \
  --features emb_esmif_eos_bos_repeat \
  --epochs "$EPOCHS" \
  --out-root "$OUT"

echo "CNN 2layer both-embeddings run completed."

# Submit (make the slurm log dir first -- sbatch will not make it for you):
#   mkdir -p runs/cnn_2layer/logs/slurm && sbatch src/run/run_cnn_2layer_slurm.sh
# Override epochs:
#   mkdir -p runs/cnn_2layer/logs/slurm && sbatch --export=EPOCHS=30,ALL src/run/run_cnn_2layer_slurm.sh
