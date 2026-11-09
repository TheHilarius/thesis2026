#!/usr/bin/env python3
"""
29_build_cluster_map.py
Reconstruct peptide-homology clusters with 01_datasplit.py logic and save a
reusable peptide -> cluster_id map.

The map lets fold-difficulty audits (02_datasplit_analysis.py) run
cluster-preserving permutation nulls without re-clustering every time.

Clusters (01_datasplit.cluster_by_hamming): unique peptides linked by
DIRECT Hamming distance <= --hamming-cutoff via Union-Find (transitive
closure). Folds are groups of whole clusters assigned by greedy
bin-packing -- many clusters per fold. The cutoff MUST match the one
used to build the split file (01_datasplit.py --hamming-cutoff).

Usage:
    python src/investigation/29_build_cluster_map.py
    python src/investigation/29_build_cluster_map.py --out data/processed/cluster_map_peptide.csv
    python src/investigation/29_build_cluster_map.py \
        --split data/processed/df_all_with_folds_h2.csv --hamming-cutoff 2 \
        --out data/processed/cluster_map_peptide_h2.csv

Output (default DATA_DIR/cluster_map_peptide.csv):
    peptide, cluster_id, cluster_size, cluster_rows, fold
  cluster_id   stable int, 0 = largest cluster
  cluster_size n unique peptides in the cluster
  cluster_rows n data rows (incl. duplicate peptide rows from other proteins)
  fold         fold assignment from df_all_with_folds.csv (must be constant
               within a cluster; violations = homology leakage in the split)
"""

import argparse
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, SPLIT_DATA_PATH, LOG_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    PEPTIDE_COL, LABEL_COL, FOLD_COL, HAMMING_CUTOFF,
    validate_config,
)

DEFAULT_OUT = DATA_DIR / "cluster_map_peptide.csv"


def main():
    ap = argparse.ArgumentParser(description="Build reusable peptide-homology cluster map.")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output CSV path")
    ap.add_argument("--split", default=None,
                    help="split CSV path (default: config SPLIT_DATA_PATH), "
                         "e.g. data/processed/df_all_with_folds_h2.csv")
    ap.add_argument("--hamming-cutoff", type=int, default=HAMMING_CUTOFF,
                    help="DIRECT Hamming cutoff for clustering (1-3); must "
                         "match the cutoff used to build --split. "
                         "Default: config HAMMING_CUTOFF (%(default)s)")
    ap.add_argument("--n-show", type=int, default=10,
                    help="how many largest clusters to print")
    a = ap.parse_args()

    if a.hamming_cutoff not in (1, 2, 3):
        ap.error("--hamming-cutoff must be 1, 2 or 3")

    split_path = Path(a.split) if a.split else Path(SPLIT_DATA_PATH)

    validate_config()

    print("=" * 80)
    print("  29_BUILD_CLUSTER_MAP -- PEPTIDE HOMOLOGY CLUSTER EXPORT")
    print("=" * 80)
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data path:    {split_path}")
    print(f"  Output path:  {a.out}")
    print(f"  Hamming cutoff: <= {a.hamming_cutoff}"
          f"{' (CLI override)' if a.hamming_cutoff != HAMMING_CUTOFF else ''}")

    # -- Load --
    print(f"\nLoading {split_path} ...")
    df = pd.read_csv(split_path, usecols=[PEPTIDE_COL, LABEL_COL, FOLD_COL])
    print(f"  {len(df)} rows, {df[PEPTIDE_COL].nunique()} unique peptides")

    # -- Sanity 1: no peptide split across folds --
    pep_nfolds = df.groupby(PEPTIDE_COL)[FOLD_COL].nunique()
    split_peps = pep_nfolds[pep_nfolds > 1]
    if len(split_peps):
        print(f"\nFATAL: {len(split_peps)} peptides appear in >1 fold "
              f"(e.g. {list(split_peps.index[:5])}). Refusing to build map.")
        sys.exit(1)
    print("  [OK] every peptide maps to exactly one fold")

    # -- Cluster (same code as the split) --
    datasplit = importlib.import_module("01_datasplit")
    unique_peptides = df[PEPTIDE_COL].unique().tolist()
    t0 = time.time()
    print(f"\nClustering {len(unique_peptides)} peptides by Hamming <= {a.hamming_cutoff} ...")
    clusters = datasplit.cluster_by_hamming(unique_peptides, max_dist=a.hamming_cutoff)
    print(f"  done in {time.time() - t0:.1f}s: {len(clusters)} clusters")

    # -- Stable IDs: largest first, ties by alphabetically-first peptide --
    items = sorted(clusters.values(), key=lambda s: (-len(s), min(s)))

    pep_fold = df.drop_duplicates(PEPTIDE_COL).set_index(PEPTIDE_COL)[FOLD_COL]
    row_counts = df[PEPTIDE_COL].value_counts()

    out_rows = []
    violations = []
    for cid, peps in enumerate(items):
        peps = sorted(peps)
        folds = {pep_fold[p] for p in peps}
        if len(folds) > 1:
            violations.append((cid, sorted(folds), peps))
        fold = pep_fold[peps[0]]
        n_rows = int(sum(row_counts[p] for p in peps))
        for p in peps:
            out_rows.append((p, cid, len(peps), n_rows, fold))

    cmap = pd.DataFrame(out_rows, columns=["peptide", "cluster_id",
                                           "cluster_size", "cluster_rows", "fold"])

    # -- Sanity 2: no homology cluster split across folds --
    if violations:
        print(f"\nWARNING: {len(violations)} homology clusters span >1 fold "
              f"(split leaks homology):")
        for cid, folds, peps in violations[:5]:
            shown = ", ".join(peps[:6]) + (f" ... +{len(peps) - 6} more" if len(peps) > 6 else "")
            print(f"  cluster {cid}: folds {folds}  peptides: {shown}")
    else:
        print("  [OK] no homology cluster spans more than one fold")

    # -- Summary --
    sizes = np.array([len(p) for p in items])
    print(f"\nCluster summary:")
    print(f"  n clusters:        {len(items)}")
    print(f"  singletons:        {int((sizes == 1).sum())} "
          f"({(sizes == 1).mean() * 100:.1f}%)")
    print(f"  largest cluster:   {sizes.max()} peptides (id 0)")
    print(f"  mean size:         {sizes.mean():.2f}  median: {np.median(sizes):.0f}")

    print(f"\nTop {a.n_show} largest clusters:")
    print(f"  {'id':>6} {'peps':>6} {'rows':>6} {'fold':>5}  peptides")
    for cid in range(min(a.n_show, len(items))):
        peps = sorted(items[cid])
        fold = pep_fold[peps[0]]
        n_rows = int(sum(row_counts[p] for p in peps))
        shown = ", ".join(peps[:4]) + (f" ... +{len(peps) - 4} more" if len(peps) > 4 else "")
        print(f"  {cid:>6} {len(peps):>6} {n_rows:>6} {fold:>5}  {shown}")

    # -- Save --
    out_path = a.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmap.to_csv(out_path, index=False)
    print(f"\nSaved {len(cmap)} peptide rows -> {out_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
