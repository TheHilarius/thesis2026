#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
cd /home/projects/thesis_s204692_s204581/thesis2026
source env_esmc_server/bin/activate
export PYTHONUNBUFFERED=1
# COMBOS fields use '|', combos space-separated — sbatch --export cannot carry ','
COMBOS_ARG=${COMBOS//|/,}
python src/pipeline/python/04b_run_matrix.py \
  --out-root "runs/${CAMPAIGN}" --no-aggregate \
  ${VARDIMS:+--vardims "$VARDIMS"} \
  --combos ${COMBOS_ARG}
echo "DONE campaign=${CAMPAIGN} vardims=${VARDIMS:-none} combos=${COMBOS}"
