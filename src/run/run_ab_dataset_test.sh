#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
cd /home/projects/thesis_s204692_s204581/thesis2026
source env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
M=src/pipeline/python/04_modelling.py
python $M --model xgb --features blosum50 --out-root runs/ab_dataset_test
THESIS_VARIANT=remove715negonly python $M --model xgb --features blosum50 --out-root runs/ab_dataset_test
python $M --model xgb --features emb_esmc300m_win_padtoken --pca-mode dimselect --top-n default --out-root runs/ab_dataset_test
THESIS_VARIANT=remove715negonly python $M --model xgb --features emb_esmc300m_win_padtoken --pca-mode dimselect --top-n default --out-root runs/ab_dataset_test
echo "ALL 4 A/B RUNS DONE"
