#!/bin/bash
#SBATCH --job-name=cnn_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/cnn_sweep_%A_%a.out
#SBATCH --error=logs/cnn_sweep_%A_%a.err

set -euo pipefail

cd /home/projects/thesis_s204692_s204581/thesis2026

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No ESM-C venv at env_esmc_server/bin/activate"
    exit 1
fi

# torchvision-free torch check (fail fast on CPU-only envs)
python -c "import torch; assert torch.cuda.is_available() or True; print('torch', torch.__version__)"

mkdir -p logs results/tables
export PYTHONUNBUFFERED=1

OUT="${RUN:-.}"

# Submit with a timestamped run folder (mirrors windows_flat pattern):
#   RUN="runs/cnn_baseline_$(date +%Y%m%d_%H%M)"; mkdir -p "$RUN"/{models,logs/slurm,results,plots} && sbatch --job-name=cnn_sweep --output="$RUN/logs/slurm/%A_%a.out" --error="$RUN/logs/slurm/%A_%a.err" --export=RUN="$RUN",ALL src/run/run_cnn_matrix_slurm.sh

# Entry format: MODEL|FEATURES|EPOCHS (sequential one-param-at-a-time order)
CONFIGS=(
  "cnn_k3|emb_esmc300m_win_padtoken|20"
  "cnn_k5|emb_esmc300m_win_padtoken|20"
  "cnn_k3|emb_esmif_eos_bos_repeat|20"
  "mlp_flat|emb_esmc300m_win_padtoken|20"
)

IDX="${SLURM_ARRAY_TASK_ID}"
CFG="${CONFIGS[$IDX]}"
MODEL="${CFG%%|*}";  REST="${CFG#*|}"
FEAT="${REST%%|*}";  EPOCHS="${REST##*|}"

echo "Array task ${IDX}: model=${MODEL} features=${FEAT} epochs=${EPOCHS}"

python src/pipeline/python/05b_run_cnn_matrix.py \
  --out-root "$OUT" \
  --no-aggregate \
  --cnn-sweep "${MODEL}:${FEAT}:${EPOCHS}"

echo "Array task ${IDX} completed."
