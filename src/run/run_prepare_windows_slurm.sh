#!/bin/bash
#SBATCH --job-name=prepwin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prepwin_%j.out
#SBATCH --error=logs/prepwin_%j.err

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

# Re-prepare all 8 window embeddings (raw -> embeddings_prepared) so the
# modelling step reads the current raw files. This overwrites the stale
# ESM-IF prepared files and the ESM-C pad_token prepared file.
python src/pipeline/python/03b_run_matrix.py

echo "Prepare embeddings completed."
