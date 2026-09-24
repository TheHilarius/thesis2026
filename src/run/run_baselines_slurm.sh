#!/bin/bash
#SBATCH --job-name=winbase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/winbase_%j.out
#SBATCH --error=logs/winbase_%j.err

set -euo pipefail

cd /home/projects/thesis_s204692_s204581/thesis2026

if [ -f "env_esmc_server/bin/activate" ]; then
    source env_esmc_server/bin/activate
    echo "Activated env_esmc_server"
else
    echo "ERROR: No ESM-C venv found at env_esmc_server/bin/activate"
    exit 1
fi

mkdir -p logs results/tables
export PYTHONUNBUFFERED=1

# Mode-independent baselines (handcrafted + one-hot), run once.
python src/pipeline/python/04b_run_matrix.py \
  --combos \
    rf,handcrafted,None rf,handcrafted_sparse,None \
    xgb,handcrafted,None xgb,handcrafted_sparse,None \
    lr_l2,handcrafted,None lr_l2,handcrafted_sparse,None

echo "Baseline matrix run completed."
