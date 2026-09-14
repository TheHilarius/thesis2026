#!/bin/bash
#SBATCH --job-name=matrix_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matrix_run_%j.out
#SBATCH --error=logs/matrix_run_%j.err

# Navigate to the project root on the compute node
cd /home/projects/thesis_s204692_s204581/thesis2026

# Activate the correct environment
eval "$(conda shell.bash hook)"
conda activate esm_gpu

# Ensure output directories exist
mkdir -p logs results/tables

# Run the model matrix runner (unbuffered output)
export PYTHONUNBUFFERED=1
python src/pipeline/python/04b_run_matrix.py

echo "Matrix run completed."
