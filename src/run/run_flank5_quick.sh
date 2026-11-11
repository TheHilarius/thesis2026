#!/bin/bash
#SBATCH --job-name=flank5_quick
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/flank5_quick_%j.out
#SBATCH --error=logs/flank5_quick_%j.err
set -euo pipefail
cd /home/projects/thesis_s204692_s204581/thesis2026
source env_esmc_server/bin/activate
mkdir -p logs results/tables_flank5
export PYTHONUNBUFFERED=1
export THESIS_VARIANT=flank5

python src/pipeline/python/04_modelling.py \
  --model xgb --features blosum50 \
  --out-root runs/flank5_compare

python src/pipeline/python/04_modelling.py \
  --model xgb --features handcrafted_blosum50 \
  --out-root runs/flank5_compare

echo "FLANK5 QUICK DONE"
