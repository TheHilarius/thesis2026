#!/bin/bash
#SBATCH --job-name=matrix_${MODEL}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

cd /home/projects/thesis_s204692_s204581/thesis2026
source /home/projects/thesis_s204692_s204581/thesis2026/env_esmc_server/bin/activate
mkdir -p logs results/tables
export PYTHONUNBUFFERED=1

python src/pipeline/python/04b_run_matrix.py --pca-sweep \
  ${MODEL}:handcrafted_sparse_esmc:1,13,26,66,218,718 \
  ${MODEL}:handcrafted_sparse_esmif:9,64,95,148,248,420 \
  "$@"

echo "Matrix run for ${MODEL} completed."
