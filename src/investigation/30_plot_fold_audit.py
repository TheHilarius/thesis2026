#!/usr/bin/env python3
"""
30_plot_fold_audit.py
Visualize fold-difficulty audit results (Analysis A).

Plot 1 -- point-range panels: per CV fold, observed stat (point) vs
  cluster-preserving permutation null 95% interval (whisker).
  Red = outside null (OUTLIER), blue = inside. Held-out has no null
  and is excluded.
Plot 2 -- cluster-size distribution per fold: counts of homology clusters
  by size (1..max), log-y, one curve per fold. Shows whether any fold
  hoards the large homology blocks.

Usage:
    python src/investigation/30_plot_fold_audit.py
    python src/investigation/30_plot_fold_audit.py --outdir results/figures/fold_audit

Inputs:
    results/tables/fold_audit_cluster.csv  (from 02_datasplit_analysis.py)
    data/processed/cluster_map_peptide.csv  (from 29_build_cluster_map.py)
"""

import argparse
import os
import sys
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

from config import DATA_DIR, N_CV_FOLDS, HELD_OUT_INDEX

DEFAULT_AUDIT = Path("results/tables/fold_audit_cluster.csv")
DEFAULT_CMAP = DATA_DIR / "cluster_map_peptide.csv"
DEFAULT_OUTDIR = Path("results/figures/fold_audit")

STAT_ORDER = ["rows", "pos_pct", "dup_frac", "n_clusters",
              "mean_cl_size", "largest_share"]
STAT_LABELS = {
    "rows": "Rows",
    "pos_pct": "Positive %",
    "dup_frac": "Duplicate-row fraction",
    "n_clusters": "N clusters",
    "mean_cl_size": "Mean cluster size",
    "largest_share": "Largest-cluster share (%)",
}

OK_COLOR = "#2a6fb0"
BAD_COLOR = "#d62728"


def plot_intervals(audit_df, out_path):
    cv = audit_df[audit_df["fold"] != HELD_OUT_INDEX].copy()
    folds = sorted(cv["fold"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, stat in zip(axes.flat, STAT_ORDER):
        sub = cv[cv["stat"] == stat].set_index("fold").loc[folds]
        obs = sub["observed"].to_numpy(dtype=float)
        lo = sub["null_lo"].to_numpy(dtype=float)
        hi = sub["null_hi"].to_numpy(dtype=float)
        is_out = (sub["flag"].to_numpy() != "ok")
        colors = [BAD_COLOR if o else OK_COLOR for o in is_out]

        x = np.arange(len(folds))
        # Outlier cells have observed outside [lo, hi]; clip so the
        # whisker collapses on that side (red point outside whisker).
        ax.errorbar(x, obs,
                    yerr=[np.clip(obs - lo, 0, None),
                          np.clip(hi - obs, 0, None)],
                    fmt="o", ecolor="grey", elinewidth=1.5, capsize=4,
                    zorder=1)
        ax.scatter(x, obs, c=colors, s=60, zorder=2)
        # Label outlier points with their observed value
        for xi, o, out in zip(x, obs, is_out):
            if out:
                ax.annotate(f"{o:g}", (xi, o), fontsize=8, color=BAD_COLOR,
                            ha="center", va="bottom", xytext=(0, 6),
                            textcoords="offset points")
        ax.set_xticks(x)
        ax.set_xticklabels([f"fold {f}" for f in folds])
        ax.set_title(STAT_LABELS[stat])
        # Y-range must include the observed points, not just the null
        # whiskers -- outlier points outside the null otherwise fall
        # off-axis (e.g. largest_share fold 0 far above null hi).
        v_lo = min(np.nanmin(lo), np.nanmin(obs))
        v_hi = max(np.nanmax(hi), np.nanmax(obs))
        span = v_hi - v_lo
        if span > 0 and np.isfinite(span):
            ax.set_ylim(v_lo - 0.15 * span, v_hi + 0.15 * span)

    n_out = int((cv["flag"] != "ok").sum())
    fig.suptitle(f"Fold-difficulty audit: observed vs cluster-preserving null "
                 f"(95% whiskers, {n_out} OUTLIER cells)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def plot_cluster_sizes(cmap_df, out_path):
    one = cmap_df.drop_duplicates("cluster_id")
    folds = sorted(one["fold"].unique())
    max_size = int(one["cluster_size"].max())

    fig, ax = plt.subplots(figsize=(9, 5))
    for f in folds:
        sizes = one.loc[one["fold"] == f, "cluster_size"].to_numpy()
        counts = np.bincount(sizes, minlength=max_size + 1)[1:]
        xs = np.arange(1, max_size + 1)
        mask = counts > 0
        role = "held-out" if f == HELD_OUT_INDEX else f"fold {f}"
        ax.plot(xs[mask], counts[mask], marker="o", ms=4, label=role)

    ax.set_yscale("log")
    ax.set_xlabel("Homology-cluster size (unique peptides)")
    ax.set_ylabel("N clusters (log)")
    ax.set_title("Homology-cluster size distribution per fold")
    ax.legend(fontsize=8, title="Bucket")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="Plot fold-difficulty audit results.")
    ap.add_argument("--audit", default=str(DEFAULT_AUDIT))
    ap.add_argument("--cmap", default=str(DEFAULT_CMAP))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    a = ap.parse_args()

    audit_df = pd.read_csv(a.audit)
    cmap_df = pd.read_csv(a.cmap)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p1 = plot_intervals(audit_df, outdir / "fold_audit_intervals.png")
    print(f"  wrote {p1}")
    p2 = plot_cluster_sizes(cmap_df, outdir / "cluster_size_dist.png")
    print(f"  wrote {p2}")


if __name__ == "__main__":
    main()
