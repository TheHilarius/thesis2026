#!/bin/bash
#SBATCH --job-name=winmatrix_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/winmatrix_all_%j.out
#SBATCH --error=logs/winmatrix_all_%j.err

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

python src/pipeline/python/04b_run_matrix.py \
  --pca-sweep \
    rf:handcrafted_sparse_esmif_win:9,64,148,420 \
    xgb:handcrafted_sparse_esmif_win:9,64,148,420 \
    lr_elasticnet:handcrafted_sparse_esmif_win:9,64,148,420 \
  --combos \
    rf,handcrafted,None rf,handcrafted_sparse,None \
    xgb,handcrafted,None xgb,handcrafted_sparse,None \
    lr_elasticnet,handcrafted,None lr_elasticnet,handcrafted_sparse,None \
  "$@"

echo "Window matrix run (all models) completed."
