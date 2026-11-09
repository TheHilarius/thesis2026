#!/usr/bin/env python3
"""
32_fold_scatter.py
Per-peptide UMAP scatter colored by fold (visual companion to 31).

Each peptide is encoded as concatenated BLOSUM50 rows (same matrix as
31_fold_similarity, verbatim from src/pipeline/r/functions.R):
    --context peptide : 9-mer            -> 180-dim vector
    --context full    : 29-mer from positional cols N10..C10 (NaN -> 'X')
                        -> 609-dim vector (21-letter alphabet incl. X)

UMAP on the vectors (euclidean, seed 42), scatter colored by fold with
alpha blending + faceted highlight panels (one fold in color per panel
over a grey background).

Usage:
    python src/investigation/32_fold_scatter.py
    python src/investigation/32_fold_scatter.py --context full --n-sample 3000
    python src/investigation/32_fold_scatter.py \
        --split data/processed/df_all_with_folds_h2.csv --tag h2

Outputs (per-context subfolder, default results/figures/fold_similarity/<context>[_<tag>]/):
    umap_<context>[_<tag>]_all.png     all peptides, one color per fold
    umap_<context>[_<tag>]_facets.png  6 panels, one highlighted fold each
Plus results/tables/umap_<context>[_<tag>].csv  (x, y, fold, seq) for reuse.
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
for _p in (PIPELINE_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    SPLIT_DATA_PATH,
    PEPTIDE_COL, FOLD_COL, HELD_OUT_INDEX,
    validate_config,
)

sim31 = importlib.import_module("31_fold_similarity")
BLOSUM50, AA_INDEX = sim31.BLOSUM50, sim31.AA_INDEX

# Full-context alphabet: canonical 20 + X (terminus padding / missing flank
# positions). X row/col = -1 everywhere, verbatim from functions.R.
AA_ORDER_X = "ACDEFGHIKLMNPQRSTVWY X".replace(" ", "")
AA_INDEX_X = {a: i for i, a in enumerate(AA_ORDER_X)}
BLOSUM50_X = np.pad(BLOSUM50, ((0, 1), (0, 1)), constant_values=-1.0)

FOLD_COLORS = {
    0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
    3: "#d62728", 4: "#9467bd", 5: "#8c564b",
}


def encode_profile(seqs, lut=BLOSUM50, index=AA_INDEX):
    """seqs: array of equal-length strings -> (n, len*lut.shape[0]) float32."""
    n = len(seqs)
    L = len(seqs[0])
    D = lut.shape[0]
    out = np.empty((n, L * D), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i] = lut[[index[a] for a in s]].ravel()
    return out


def load_vectors(context, n_sample, seed, split_path):
    from config import POSITION_AA_COLS
    if context == "peptide":
        df = pd.read_csv(split_path, usecols=[PEPTIDE_COL, FOLD_COL])
        df = df.drop_duplicates(PEPTIDE_COL).reset_index(drop=True)
        seqs = df[PEPTIDE_COL].to_numpy()
        lut, index = BLOSUM50, AA_INDEX
    elif context == "full":
        # Aligned 29-mers from positional columns (NaN = missing flank
        # at protein termini -> 'X', matching the R pipeline convention).
        df = pd.read_csv(split_path,
                         usecols=[PEPTIDE_COL, FOLD_COL] + POSITION_AA_COLS)
        seqs = (df[POSITION_AA_COLS].fillna("X").agg("".join, axis=1)
                .to_numpy())
        lut, index = BLOSUM50_X, AA_INDEX_X
    else:
        raise ValueError(f"Unknown context '{context}'")
    if n_sample and len(df) > n_sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=n_sample, replace=False))
        df, seqs = df.iloc[idx].reset_index(drop=True), seqs[idx]
    X = encode_profile(seqs, lut, index)
    return df, seqs, X


def plot_all(xy, folds, out_path, context):
    fig, ax = plt.subplots(figsize=(9, 7))
    for f in sorted(np.unique(folds)):
        m = folds == f
        role = "held-out" if f == HELD_OUT_INDEX else f"fold {f}"
        ax.scatter(xy[m, 0], xy[m, 1], s=1, alpha=0.25,
                   c=FOLD_COLORS.get(int(f), "grey"),
                   label=f"{role} (n={m.sum()})", rasterized=True)
    ax.set_xlabel("UMAP 1 (BLOSUM50 profile, euclidean)")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Peptide space by fold ({context} context, "
                 f"n={len(folds)}) — overlap expected if folds mix")
    ax.legend(markerscale=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def plot_facets(xy, folds, out_path, context):
    uniq = sorted(np.unique(folds))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, f in zip(axes.flat, uniq):
        ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.06, c="grey",
                   rasterized=True)
        m = folds == f
        role = "held-out" if f == HELD_OUT_INDEX else f"fold {f}"
        ax.scatter(xy[m, 0], xy[m, 1], s=1, alpha=0.3,
                   c=FOLD_COLORS.get(int(f), "grey"), rasterized=True)
        ax.set_title(f"{role} (n={m.sum()})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Peptide space per fold ({context} context) — "
                 f"color vs grey background")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="Per-peptide UMAP scatter by fold.")
    ap.add_argument("--context", default="peptide", choices=["peptide", "full"])
    ap.add_argument("--split", default=None,
                    help="split CSV path (default: config SPLIT_DATA_PATH), "
                         "e.g. data/processed/df_all_with_folds_h2.csv")
    ap.add_argument("--tag", default="",
                    help="suffix for output names, e.g. 'h2' -> "
                         "umap_peptide_h2.csv and figures under peptide_h2/ "
                         "(keeps canonical outputs untouched)")
    ap.add_argument("--n-sample", type=int, default=0,
                    help="stratified cap on peptides (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--outdir", default=None,
                    help="default: results/figures/fold_similarity/<context>[_<tag>]")
    ap.add_argument("--tables", default="results/tables")
    a = ap.parse_args()

    validate_config()
    split_path = Path(a.split) if a.split else Path(SPLIT_DATA_PATH)
    tag = ("_" + a.tag) if a.tag else ""
    print("=" * 80)
    print(f"  32_FOLD_SCATTER -- context: {a.context}")
    print(f"  Split: {split_path}")
    print("=" * 80)

    df, seqs, X = load_vectors(a.context, a.n_sample or 0, a.seed, split_path)
    print(f"  {len(df)} sequences, {X.shape[1]}-dim BLOSUM profiles")

    import umap
    t0 = time.time()
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=a.min_dist,
                   random_state=a.seed).fit_transform(X)
    print(f"  UMAP fit in {time.time() - t0:.1f}s")

    folds = df[FOLD_COL].to_numpy()
    tables = Path(a.tables)
    tables.mkdir(parents=True, exist_ok=True)
    emb = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                        "fold": folds, "seq": seqs})
    csv_path = tables / f"umap_{a.context}{tag}.csv"
    emb.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    outdir = Path(a.outdir or f"results/figures/fold_similarity/{a.context}{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    p1 = plot_all(xy, folds, outdir / f"umap_{a.context}{tag}_all.png", a.context)
    print(f"  wrote {p1}")
    p2 = plot_facets(xy, folds, outdir / f"umap_{a.context}{tag}_facets.png",
                     a.context)
    print(f"  wrote {p2}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
