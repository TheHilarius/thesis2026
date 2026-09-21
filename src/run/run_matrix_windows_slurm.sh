#!/bin/bash
#SBATCH --job-name=winmatrix_${MODEL}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd /home/projects/thesis_s204692_s204581/thesis2026
source /home/projects/thesis_s204692_s204581/thesis2026/env_esmc_server/bin/activate
mkdir -p logs results/tables
export PYTHONUNBUFFERED=1

python src/pipeline/python/04b_run_matrix.py \
  --pca-sweep ${MODEL}:handcrafted_sparse_esmif_win:9,64,148,420 \
  --combos ${MODEL},handcrafted,None ${MODEL},handcrafted_sparse,None \
  "$@"

echo "Window matrix run for ${MODEL} completed."
