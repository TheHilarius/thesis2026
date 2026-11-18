#!/bin/bash
# Fold-audit sweep across window cutoffs c=1..8 (robust 3-rule split,
# pepcut1 fixed) on the THESIS dataset (flank5_rmdup).
#   chain per c: 29_build_cluster_map (context mode) -> 02_datasplit_analysis
#                -> 30_plot_fold_audit (derived outdir)
# Usage (repo root):  bash src/run/run_fold_audit_c_sweep.sh
#   override the dataset with THESIS_VARIANT (default flank5_rmdup).
set -u
cd "$(dirname "$0")/../.." || exit 1
VARIANT="${THESIS_VARIANT:-flank5_rmdup}"
for c in 1 2 3 4 5 6 7 8; do
  split="data/processed/df_all_with_folds_${VARIANT}_updated_h${c}_pepcut1.csv"
  cmap="data/processed/cluster_map_context_h${c}_pepcut1.csv"
  # run tree: flank5_rmdup -> runs/r19_rmdup, flank5 -> runs/r19_flank5
  run_root="runs/r19_${VARIANT#flank5_}"
  audit="${run_root}/results/fold_audit_context_h${c}_pepcut1.csv"
  echo "=== c=$c map $(date +%H:%M:%S) ==="
  python3 src/investigation/29_build_cluster_map.py --group-by context \
    --window-cutoff "$c" --pep-cutoff 1 --split "$split" --out "$cmap" \
    > "/tmp/audit29_h${c}.log" 2>&1 || { echo "29 FAILED c=$c"; tail -5 "/tmp/audit29_h${c}.log"; continue; }
  echo "=== c=$c audit $(date +%H:%M:%S) ==="
  python3 src/investigation/02_datasplit_analysis.py --split "$split" \
    --cluster-map "$cmap" --audit-out "$audit" --tag "context_h${c}" \
    > "/tmp/audit02_h${c}.log" 2>&1 || { echo "02 FAILED c=$c"; tail -5 "/tmp/audit02_h${c}.log"; continue; }
  echo "=== c=$c plots $(date +%H:%M:%S) ==="
  python3 src/investigation/30_plot_fold_audit.py --audit "$audit" \
    --cmap "$cmap" --run-root "$run_root" \
    > "/tmp/audit30_h${c}.log" 2>&1 || { echo "30 FAILED c=$c"; }
done
echo ALL_PLOTS_DONE
