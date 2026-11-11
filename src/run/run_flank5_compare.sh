#!/bin/bash
#SBATCH --job-name=flank5_cmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/flank5_cmp_%j.out
#SBATCH --error=logs/flank5_cmp_%j.err

set -euo pipefail
cd /home/projects/thesis_s204692_s204581/thesis2026
source env_esmc_server/bin/activate
mkdir -p logs results/tables_flank5
export PYTHONUNBUFFERED=1
export THESIS_VARIANT=flank5

# 1) flank5 per-dim variance ranking -> separate dir (never overwrite 29mer JSON)
python src/investigation/10_dim_variance.py \
  --keys esmc300m_win_padtoken --ev 80 \
  --out-dir results/tables_flank5 \
  --out-root runs/flank5_compare

# 2) frontier: XGB handcrafted + BLOSUM50 (~15 min)
python src/pipeline/python/04_modelling.py \
  --model xgb --features handcrafted_blosum50 \
  --out-root runs/flank5_compare

# 3) ESM-C 300M EV80 vardims (~24 h)
python src/pipeline/python/04_modelling.py \
  --model xgb --features handcrafted_esmc300m_win_padtoken \
  --vardims results/tables_flank5/dim_variance_esmc300m_win_padtoken_ev80.json \
  --out-root runs/flank5_compare

echo "FLANK5 COMPARE DONE"
