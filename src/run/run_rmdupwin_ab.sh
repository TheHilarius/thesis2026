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
# remove_dup_windows variant: one row per identical context window,
# positive preferred (make_variant_dataset.py), folds re-split,
# prepared H5s subset by prepare_variant_h5.py.
# Blosum50 arm first (fast readout); embedding arm appended later.
THESIS_VARIANT=remove_dup_windows python $M --model xgb --features blosum50 --out-root runs/ab_dataset_test
echo "RMDUPWIN BLOSUM50 RUN DONE"
