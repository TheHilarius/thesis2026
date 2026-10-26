#!/usr/bin/env python3
"""
24_xgb_fixed_grid_plot.py
Fixed-budget surface around the XGB baseline (no early stopping anywhere),
padmode-style bar charts, y-axis 0-1 like plot_padmode_data_compare.

Figures (written to runs/xgb_fixed_grid/plots/):
  fixed_grid_lr_bars.png   AUC-ROC bars: lr sweep at n=1000 (runs/xgb_fixed_grid)
                           + baseline bar; DeLong+Holm stars from
                           delong_vs_baseline.tsv; dashed baseline line.
  fixed_grid_n_bars.png    AUC-ROC bars: n sweep at lr=0.1
                           (runs/xgb_n_estimators_sweep) + baseline bar;
                           placeholder until >=2 arms exist there.

Usage:
  python src/investigation/24_xgb_fixed_grid_plot.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROJECT_ROOT

PLOTS = PROJECT_ROOT / "runs" / "xgb_fixed_grid" / "plots"
BASELINE = (PROJECT_ROOT / "runs" / "encodings_sweep" / "models" / "flat" /
            "cv_results_xgb_blosum50_flat_20261014_154653.json")
GRID_DIR = PROJECT_ROOT / "runs" / "xgb_fixed_grid" / "models" / "flat"
NBUD_DIR = PROJECT_ROOT / "runs" / "xgb_n_estimators_sweep" / "models" / "flat"
DELONG_TSV = (PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "results" /
              "delong_vs_baseline.tsv")

# Paul Tol bright (colour-blind safe), same scheme as the other figures
SWEEP_COLOR = "#32CD32"   # lime — fixed-budget sweep arms
BASE_COLOR = "#BBBBBB"    # grey — baseline


def load(path: Path):
    res = json.load(open(path))
    m = re.search(r"xgblr([\dp]+)n(\d+)", path.name)
    if not m:
        return None
    return {
        "lr": float(m.group(1).replace("p", ".")),
        "n": int(m.group(2)),
        "auc": float(res["cv_summary"]["auc_roc"]["mean"]),
        "auc_sd": float(res["cv_summary"]["auc_roc"]["std"]),
    }


def load_baseline():
    s = json.load(open(BASELINE))["cv_summary"]["auc_roc"]
    return {"lr": 0.1, "n": 1000, "auc": float(s["mean"]),
            "auc_sd": float(s["std"])}


def load_stars():
    if not DELONG_TSV.exists():
        return {}
    df = pd.read_csv(DELONG_TSV, sep="\t")
    return {str(r["arm"]): str(r["sig"]) for _, r in df.iterrows()}


def bar_figure(rows, stars, star_prefix, fname, xlabel):
    """Padmode-style bars: 0-1 y-axis, dashed baseline, value+star labels."""
    n = len(rows)
    x = np.arange(n)
    colors = [BASE_COLOR if r.get("is_base") else SWEEP_COLOR for r in rows]
    vals = np.array([r["auc"] for r in rows])
    errs = np.array([r["auc_sd"] for r in rows])
    base_val = rows[0]["auc"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, vals, yerr=errs, color=colors, width=0.62,
           edgecolor="black", linewidth=0.5,
           error_kw=dict(lw=0.8, capsize=3))
    for xi, v, e, r in zip(x, vals, errs, rows):
        lab = f"{v:.4f}"
        if not r.get("is_base"):
            star = stars.get(f"{star_prefix}{r['lr'] if 'lr' in star_prefix else r['n']}", "")
            if star:
                lab += f"\n{star}"
        ax.text(xi, v + e + 0.015, lab, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold" if r.get("is_base") else "normal")

    ax.axhline(base_val, color="black", linestyle="--", lw=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=0, fontsize=9.5)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1)   # padmode convention
    ax.set_title(f"XGB blosum50 — {xlabel} (mean ± std, 6 outer buckets; "
                 f"dashed: baseline {base_val:.4f})", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BASE_COLOR,
                             edgecolor="black", linewidth=0.5),
               plt.Rectangle((0, 0), 1, 1, facecolor=SWEEP_COLOR,
                             edgecolor="black", linewidth=0.5)]
    ax.legend(handles, ["Fixed budget (baseline)", "Fixed-budget sweep"],
              fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def lr_figure(stars):
    rows = [load(p) for p in sorted(GRID_DIR.glob("cv_results_*.json"))]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("  [skip] lr figure: no arms in xgb_fixed_grid")
        return
    base = load_baseline()
    base.update(is_base=True, label="baseline\n0.1/1000")
    rows.sort(key=lambda r: r["lr"])
    for r in rows:
        r["label"] = f"{r['lr']:g}\n/1000"
    rows.insert(0, base)
    bar_figure(rows, stars, "fixed_grid_lr", "fixed_grid_lr_bars.png",
               "learning rate, n_estimators = 1000")


def nbud_figure(stars):
    rows = [load(p) for p in sorted(NBUD_DIR.glob("cv_results_*.json"))]
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        out = PLOTS / "fixed_grid_n_bars.png"
        PLOTS.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.text(0.5, 0.5, "n_estimators sweep running —\n"
                "bars appear when results land",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC-ROC")
        ax.set_title("XGB blosum50 — n_estimators, lr = 0.1 "
                     "(mean ± std, 6 outer buckets)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved: {out} (placeholder)")
        return
    base = load_baseline()
    base.update(is_base=True, label="baseline\n0.1/1000")
    rows.sort(key=lambda r: r["n"])
    for r in rows:
        r["label"] = f"0.1\n/{r['n']}"
    rows.insert(0, base)
    bar_figure(rows, stars, "n_budget_lr0.1", "fixed_grid_n_bars.png",
               "n_estimators, learning rate = 0.1")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    stars = load_stars()
    lr_figure(stars)
    nbud_figure(stars)


if __name__ == "__main__":
    main()
