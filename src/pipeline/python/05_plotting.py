#!/usr/bin/env python3
"""
05_plotting.py
Learning-curve plots for 05 CNN runs. Used two ways:

  1. Automatic: 05_cnn_modelling.py calls plot_learning_curves() via
     subprocess after every run (CSV + meta JSON in, PNG out).
  2. Standalone: re-plot any old run from its curves CSV:
       python 05_plotting.py --csv runs/cnn_cv/plots/learning_curves_<tag>.csv \\
           --png /tmp/curves.png [--meta ..._meta.json] [--title "..."]

Plot spec (thesis decision, Oct 2026):
  - One combined PNG, two panels (val AUC | train loss).
  - Colour = outer validation bucket = pre-analysis data split (6 colours).
    Solid line = outer mean, faint lines = its 5 inner models.
  - Legend: outer bucket + split size/pos-rate, so split-level trends
    (e.g. "split 2 always easy") are readable directly.
  - Y: AUC fixed 0-1, loss 0-max. No std bands.

Only dependency is matplotlib (guarded: missing mpl -> warning, no PNG).
Reads the long-form CSV written by 05_cnn_modelling.py, never the H5.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

OUTER_COLORS = [
    "#1f77b4",  # outer 0 - blue
    "#ff7f0e",  # outer 1 - orange
    "#2ca02c",  # outer 2 - green
    "#d62728",  # outer 3 - red
    "#9467bd",  # outer 4 - purple
    "#8c564b",  # outer 5 - brown
]


def load_curves(csv_path):
    """{outer: {inner: [(train_loss, val_auc), ...]}} in epoch order."""
    curves = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            o, i = str(row["outer"]), str(row["inner"])
            curves.setdefault(o, {}).setdefault(i, []).append(
                (float(row["train_loss"]), float(row["val_auc"])))
    return curves


def outer_mean(inners, key_idx, max_len):
    idx = {"train_loss": 0, "val_auc": 1}[key_idx]
    mean = []
    for e in range(max_len):
        vals = [h[e][idx] for h in inners.values() if len(h) > e]
        mean.append(sum(vals) / len(vals) if vals else float("nan"))
    return mean


def plot_learning_curves(csv_path, png_path, title=None, meta=None):
    """Write combined PNG. Returns path str, or None if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing -- PNG skipped, CSV kept)")
        return None

    curves = load_curves(csv_path)
    meta = meta or {}
    buckets = meta.get("buckets", {})
    max_ep = max(len(h) for inners in curves.values() for h in inners.values())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for oi, o in enumerate(sorted(curves, key=int)):
        color = OUTER_COLORS[int(o) % len(OUTER_COLORS)]
        inners = curves[o]
        mlen = max(len(h) for h in inners.values())
        ep = list(range(1, mlen + 1))
        for h in inners.values():
            e = list(range(1, len(h) + 1))
            ax[0].plot(e, [v[1] for v in h], color=color, alpha=0.22, linewidth=1)
            ax[1].plot(e, [v[0] for v in h], color=color, alpha=0.22, linewidth=1)
        b = buckets.get(str(o), {})
        lab = f"outer {o}" + (f" (n={b['n']}, pos={b['pos_pct']}%)" if b else "")
        ax[0].plot(ep, outer_mean(inners, "val_auc", mlen), color=color,
                    linewidth=2.5, label=lab)
        ax[1].plot(ep, outer_mean(inners, "train_loss", mlen), color=color,
                    linewidth=2.5, label=lab)

    ax[0].set_ylim(0, 1.0)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("val AUC")
    ax[0].legend(fontsize=8, title="validation split")
    loss_max = max(v[0] for inners in curves.values() for h in inners.values() for v in h)
    ax[1].set_ylim(0, loss_max * 1.05)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("train loss")
    ax[1].legend(fontsize=8, title="validation split")
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"Learning curves: {Path(csv_path).stem}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return str(png_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot 05 CNN learning curves from CSV.")
    p.add_argument("--csv", required=True, help="learning_curves_*.csv from a 05 run")
    p.add_argument("--png", required=True, help="output PNG path")
    p.add_argument("--title", default=None, help="figure suptitle")
    p.add_argument("--meta", default=None, help="optional *_meta.json with bucket stats")
    a = p.parse_args()
    meta = json.loads(Path(a.meta).read_text()) if a.meta else None
    out = plot_learning_curves(a.csv, a.png, a.title, meta)
    print(f"  wrote {out}" if out else "  no PNG (matplotlib missing)")
