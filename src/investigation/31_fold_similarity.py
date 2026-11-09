#!/usr/bin/env python3
"""
31_fold_similarity.py
Fold-to-fold sequence similarity (Analysis B/C companion).

Measure registry (one script, extensible):
    blosum-nn  -- BLOSUM50 nearest-neighbor similarity (THIS RUN).
                 For peptide x in fold i: best(x,j) = max sim(x,y) over y
                 in fold j; S(i,j) = mean(best). Asymmetric 6x6 matrix.
                 Diagonal excludes self-matches.
    (future) jsd      -- positional PPM Jensen-Shannon divergence.
    (future) protein  -- source-protein / family overlap.

BLOSUM50 matrix below is verbatim the project's R matrix
(src/pipeline/r/functions.R, order ACDEFGHIKLMNPQRSTVWY, X column
dropped -- split data contains canonical residues only).

sim(p,q) = sum_i M(p_i,q_i) / max(self(p), self(q)); 1 = identical.

Usage:
    python src/investigation/31_fold_similarity.py
    python src/investigation/31_fold_similarity.py --measure blosum-nn \
        --outdir results/figures/fold_similarity
    python src/investigation/31_fold_similarity.py \
        --split data/processed/df_all_with_folds_h2.csv --tag h2

Outputs:
    results/tables/fold_similarity_<measure>[_<tag>].csv
        (fold_i, fold_j, similarity; i==j rows = within-fold)
    <outdir>/similarity_heatmap.png   (A: clustered heatmap, asymmetric values,
                                       order from symmetrized distance)
    <outdir>/similarity_mds.png       (B: 2D MDS map of folds, annotated)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

from config import (
    DATA_DIR, SPLIT_DATA_PATH,
    N_CV_FOLDS, HELD_OUT_INDEX,
    PEPTIDE_COL, LABEL_COL, FOLD_COL,
    validate_config,
)

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}

# Project BLOSUM50 (functions.R), X row/col dropped.
BLOSUM50 = np.array([
    [5, -1, -2, -1, -3, 0, -2, -1, -1, -2, -1, -1, -1, -1, -2, 1, 0, 0, -3, -2],
    [-1, 13, -4, -3, -2, -3, -3, -2, -3, -2, -2, -2, -4, -3, -4, -1, -1, -1, -5, -3],
    [-2, -4, 8, 2, -5, -1, -1, -4, -1, -4, -4, 2, -1, 0, -2, 0, -1, -4, -5, -3],
    [-1, -3, 2, 6, -3, -3, 0, -4, 1, -3, -2, 0, -1, 2, 0, -1, -1, -3, -3, -2],
    [-3, -2, -5, -3, 8, -4, -1, 0, -4, 1, 0, -4, -4, -4, -3, -3, -2, -1, 1, 4],
    [0, -3, -1, -3, -4, 8, -2, -4, -2, -4, -3, 0, -2, -2, -3, 0, -2, -4, -3, -3],
    [-2, -3, -1, 0, -1, -2, 10, -4, 0, -3, -1, 1, -2, 1, 0, -1, -2, -4, -3, 2],
    [-1, -2, -4, -4, 0, -4, -4, 5, -3, 2, 2, -3, -3, -3, -4, -3, -1, 4, -3, -1],
    [-1, -3, -1, 1, -4, -2, 0, -3, 6, -3, -2, 0, -1, 2, 3, 0, -1, -3, -3, -2],
    [-2, -2, -4, -3, 1, -4, -3, 2, -3, 5, 3, -4, -4, -2, -3, -3, -1, 1, -2, -1],
    [-1, -2, -4, -2, 0, -3, -1, 2, -2, 3, 7, -2, -3, 0, -2, -2, -1, 1, -1, 0],
    [-1, -2, 2, 0, -4, 0, 1, -3, 0, -4, -2, 7, -2, 0, -1, 1, 0, -3, -4, -2],
    [-1, -4, -1, -1, -4, -2, -2, -3, -1, -4, -3, -2, 10, -1, -3, -1, -1, -3, -4, -3],
    [-1, -3, 0, 2, -4, -2, 1, -3, 2, -2, 0, 0, -1, 7, 1, 0, -1, -3, -1, -1],
    [-2, -4, -2, 0, -3, -3, 0, -4, 3, -3, -2, -1, -3, 1, 7, -1, -1, -3, -3, -1],
    [1, -1, 0, -1, -3, 0, -1, -3, 0, -3, -2, 1, -1, 0, -1, 5, 2, -2, -4, -2],
    [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 2, 5, 0, -3, -2],
    [0, -1, -4, -3, -1, -4, -4, 4, -3, 1, 1, -3, -3, -3, -3, -2, 0, 5, -3, -1],
    [-3, -5, -5, -3, 1, -3, -3, -3, -3, -2, -1, -4, -4, -1, -3, -4, -3, -3, 15, 2],
    [-2, -3, -3, -2, 4, -3, 2, -1, -2, -1, 0, -2, -3, -1, -1, -2, -2, -1, 2, 8],
], dtype=np.float64)

CHUNK = 1000  # rows of A per chunk (keeps temps ~60 MB)


def encode(peptides):
    return np.array([[AA_INDEX[a] for a in p] for p in peptides], dtype=np.int64)


def self_scores(enc):
    return np.array([BLOSUM50[enc[r], enc[r]].sum() for r in range(len(enc))])


def block_maxima(A_enc, A_self, B_enc, B_self, exclude_self=False):
    """
    Row-maxima of sim(A rows, B rows) and column-maxima, chunked over A.
    exclude_self: A and B are the same list -- ignore the self match.
    Returns (row_max (nA,), col_max (nB,)).
    """
    nA, nB = len(A_enc), len(B_enc)
    row_max = np.full(nA, -np.inf)
    col_max = np.full(nB, -np.inf)
    for a0 in range(0, nA, CHUNK):
        a1 = min(a0 + CHUNK, nA)
        raw = np.zeros((a1 - a0, nB))
        for p in range(A_enc.shape[1]):
            raw += BLOSUM50[A_enc[a0:a1, p][:, None], B_enc[None, :, p]]
        denom = np.maximum(A_self[a0:a1, None], B_self[None, :])
        sim = raw / denom
        if exclude_self:
            # global column index of the diagonal within this row-chunk
            g = np.arange(a0, a1)
            valid = g < nB
            sim[np.arange(a1 - a0)[valid], g[valid]] = -np.inf
        row_max[a0:a1] = sim.max(axis=1)
        col_max = np.maximum(col_max, sim.max(axis=0))
    return row_max, col_max


def compute_blosum_nn(peps_by_fold, folds):
    """Asymmetric S[i,j]: mean over x in fold i of max_y sim(x,y), y in fold j."""
    enc = {f: encode(peps_by_fold[f]) for f in folds}
    ss = {f: self_scores(enc[f]) for f in folds}
    S = pd.DataFrame(np.nan, index=folds, columns=folds)
    for a, f in enumerate(folds):
        # diagonal (within-fold, self excluded)
        rm, _ = block_maxima(enc[f], ss[f], enc[f], ss[f], exclude_self=True)
        S.loc[f, f] = float(rm.mean())
        for g in folds[a + 1:]:
            rm_fg, cm_fg = block_maxima(enc[f], ss[f], enc[g], ss[g])
            S.loc[f, g] = float(rm_fg.mean())
            S.loc[g, f] = float(cm_fg.mean())
    return S


MEASURES = {"blosum-nn": compute_blosum_nn}


def plot_heatmap(S, folds, out_path):
    """Two panels sharing the dendrogram order: absolute S(i,j) (A) and
    relative drop S(i,j) - S(i,i) per query row (B). B's diagonal is 0
    by construction; near-zero off-diagonal cells flag leaky fold pairs."""
    M = S.loc[folds, folds].to_numpy(dtype=float)
    sym = (M + M.T) / 2.0
    dist = np.clip(1.0 - sym, 0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    Mo = M[np.ix_(order, order)]
    Do = Mo - np.diag(Mo)[:, None]
    fo = [folds[i] for i in order]

    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.25, 1],
                          hspace=0.05, wspace=0.25)
    for col in range(2):
        ax_d = fig.add_subplot(gs[0, col])
        dendrogram(Z, ax=ax_d, no_labels=True, color_threshold=0)
        ax_d.set_xticks([]); ax_d.set_yticks([])
        for t in list(ax_d.texts):
            t.set_visible(False)
        for t in list(ax_d.get_xticklabels()) + list(ax_d.get_yticklabels()):
            t.set_visible(False)
    ax_abs = fig.add_subplot(gs[1, 0])
    ax_drop = fig.add_subplot(gs[1, 1])

    im0 = ax_abs.imshow(Mo, vmin=np.nanmin(Mo), vmax=np.nanmax(Mo))
    ax_abs.set_xticks(range(len(fo)))
    ax_abs.set_xticklabels([f"fold {f}" for f in fo])
    ax_abs.set_yticks(range(len(fo)))
    ax_abs.set_yticklabels([f"fold {f}" for f in fo])
    ax_abs.set_xlabel("reference fold j")
    ax_abs.set_ylabel("query fold i")
    ax_abs.set_title("A: absolute similarity S(i,j)")
    for i in range(len(fo)):
        for j in range(len(fo)):
            ax_abs.text(j, i, f"{Mo[i, j]:.3f}", ha="center", va="center",
                        fontsize=9)
    fig.colorbar(im0, ax=ax_abs, label="mean max BLOSUM50 similarity")

    vmax = float(np.nanmax(np.abs(Do)))
    im1 = ax_drop.imshow(Do, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax_drop.set_xticks(range(len(fo)))
    ax_drop.set_xticklabels([f"fold {f}" for f in fo])
    ax_drop.set_yticks(range(len(fo)))
    ax_drop.set_yticklabels([])
    ax_drop.set_xlabel("reference fold j")
    ax_drop.set_title("B: drop vs within-fold S(i,j) - S(i,i)")
    for i in range(len(fo)):
        for j in range(len(fo)):
            ax_drop.text(j, i, f"{Do[i, j]:+.3f}", ha="center", va="center",
                         fontsize=9)
    fig.colorbar(im1, ax=ax_drop, label="cross-fold drop (0 = diagonal)")

    fig.suptitle("Fold similarity: BLOSUM50 nearest-neighbor "
                 "(order from symmetrized distance)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_mds(S, folds, labels, out_path):
    M = S.loc[folds, folds].to_numpy(dtype=float)
    sym = (M + M.T) / 2.0
    dist = np.clip(1.0 - sym, 0, None)
    xy = MDS(n_components=2, dissimilarity="precomputed",
             random_state=42).fit_transform(dist)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=120)
    for k, f in enumerate(folds):
        ax.annotate(f"fold {f}\n{labels[f]}", xy[k], fontsize=9,
                    ha="center", va="bottom", xytext=(0, 7),
                    textcoords="offset points")
    ax.margins(0.30)
    ax.set_xlabel("MDS 1 (from 1 - similarity)")
    ax.set_ylabel("MDS 2")
    ax.set_title("Fold map: BLOSUM50 nearest-neighbor similarity "
                 "(schematic, 6 points)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="Fold-to-fold sequence similarity.")
    ap.add_argument("--measure", default="blosum-nn", choices=list(MEASURES),
                    help="similarity measure")
    ap.add_argument("--split", default=None,
                    help="split CSV path (default: config SPLIT_DATA_PATH), "
                         "e.g. data/processed/df_all_with_folds_h2.csv")
    ap.add_argument("--tag", default="",
                    help="suffix for output names, e.g. 'h2' -> "
                         "fold_similarity_blosum_nn_h2.csv and figures under "
                         "peptide_h2/ (keeps canonical outputs untouched)")
    ap.add_argument("--outdir", default=None,
                    help="per-run figure folder (default: "
                         "results/figures/fold_similarity/peptide[_<tag>])")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--reuse-table", action="store_true",
                    help="load existing similarity CSV instead of recomputing")
    a = ap.parse_args()

    if a.measure not in MEASURES:
        raise ValueError(f"Unknown measure '{a.measure}'. "
                         f"Implemented: {sorted(MEASURES)}")

    validate_config()
    split_path = Path(a.split) if a.split else Path(SPLIT_DATA_PATH)
    tag = ("_" + a.tag) if a.tag else ""
    print("=" * 80)
    print(f"  31_FOLD_SIMILARITY -- measure: {a.measure}")
    print("=" * 80)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Split: {split_path}")

    df = pd.read_csv(split_path, usecols=[PEPTIDE_COL, LABEL_COL, FOLD_COL])
    folds = sorted(df[FOLD_COL].unique())
    print(f"  Folds in file: {folds} (held-out {HELD_OUT_INDEX} included)")

    peps_by_fold = {f: sorted(df.loc[df[FOLD_COL] == f, PEPTIDE_COL].unique())
                    for f in folds}
    for f in folds:
        print(f"    fold {f}: {len(peps_by_fold[f])} unique peptides")

    t0 = time.time()
    csv_name = f"fold_similarity_{a.measure.replace('-', '_')}{tag}.csv"
    tables = Path(a.tables)
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / csv_name
    if a.reuse_table and csv_path.exists():
        print(f"  reusing {csv_path} (no recompute)")
        long_df = pd.read_csv(csv_path)
        S = long_df.pivot(index="fold_i", columns="fold_j",
                          values="similarity")
        S = S.loc[folds, folds]
    else:
        S = MEASURES[a.measure](peps_by_fold, folds)
        print(f"  similarity matrix in {time.time() - t0:.1f}s")
        print(S.round(4).to_string())
        long_df = S.reset_index().melt(id_vars="index", var_name="fold_j",
                                       value_name="similarity")
        long_df = long_df.rename(columns={"index": "fold_i"})
        long_df.to_csv(csv_path, index=False)
        print(f"  saved {csv_path}")

    outdir = Path(a.outdir or f"results/figures/fold_similarity/peptide{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    labels = {f: (f"n={len(df[df[FOLD_COL] == f])}, "
                  f"pos={(df[df[FOLD_COL] == f][LABEL_COL] == 1).mean() * 100:.1f}%")
              for f in folds}
    p1 = plot_heatmap(S, folds, outdir / "similarity_heatmap.png")
    print(f"  wrote {p1}")
    p2 = plot_mds(S, folds, labels, outdir / "similarity_mds.png")
    print(f"  wrote {p2}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
