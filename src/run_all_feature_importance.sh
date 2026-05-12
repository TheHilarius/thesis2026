#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
source env_esmc/bin/activate

FEATURE_SETS=( # uncomment to only run selected
  handcrafted_sparse
  handcrafted_blosum
  handcrafted_sparse_esmc
  handcrafted_blosum_esmc
  esmc
  handcrafted_sparse_esmif
  handcrafted_blosum_esmif
  esmif
  all_sparse
  all_blosum
)

for feat in "${FEATURE_SETS[@]}"; do
  echo "============================================================"
  echo "Feature importance for: $feat"
  echo "============================================================"

  python src/06_feature_importance.py \
    --features "$feat" \
    --models lr rf \
    --top 30 \
    --top_compare 30 \
    --top_big 70 \
    --out_root "results/figures/models/feature_importance" 
done
# out_root when dist correction: 	"results/figures/models/feature_importance_dist_correction"
