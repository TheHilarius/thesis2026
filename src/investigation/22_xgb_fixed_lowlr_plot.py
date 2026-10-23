#!/usr/bin/env python3
"""
22_xgb_fixed_lowlr_plot.py
Comparison figure for the fixed-budget low-LR runs (runs/xgb_fixed_lowlr):
baseline lr0.1/n1000 vs lr0.05/n3000 vs lr0.025/n4000 — the classic
"lower LR + larger budget" configurations, no early stopping anywhere.

Figures (written to runs/xgb_fixed_lowlr/plots/):
  fixed_lowlr_auc_bars.png   AUC-ROC mean +/- std, 6 outer buckets,
                             DeLong+Holm stars from delong_vs_baseline.tsv
  fixed_lowlr_mcc_bars.png   same layout, MCC

Usage:
  python src/investigation/22_xgb_fixed_lowlr_plot.py
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

PLOTS = PROJECT_ROOT / "runs" / "xgb_fixed_lowlr" / "plots"
BASELINE = (PROJECT_ROOT / "runs" / "encodings_sweep" / "models" / "flat" /
            "cv_results_xgb_blosum50_flat_20261014_154653.json")
ARMS_DIR = PROJECT_ROOT / "runs" / "xgb_fixed_lowlr" / "models" / "flat"
DELONG_TSV = (PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "results" /
              "delong_vs_baseline.tsv")

# same Tol/lime palette as the ES diagnostics figures
COLORS = {"baseline": "#BBBBBB", "fixed05": "#9ACD32", "fixed025": "#32CD32"}


def load(path: Path):
    res = json.load(open(path))
    tag = path.name
    m = re.search(r"xgblr([\dp]+)n(\d+)", tag)
    lr = float(m.group(1).replace("p", ".")) if m else None
    n = int(m.group(2)) if m else None
    return {
        "tag": tag,
        "label": f"fixed\n{lr:g}/{n}" if m else tag,
        "auc": float(res["cv_summary"]["auc_roc"]["mean"]),
        "auc_sd": float(res["cv_summary"]["auc_roc"]["std"]),
        "mcc": float(res["cv_summary"]["mcc"]["mean"]),
        "mcc_sd": float(res["cv_summary"]["mcc"]["std"]),
    }


def load_stars():
    if not DELONG_TSV.exists():
        return {}
    df = pd.read_csv(DELONG_TSV, sep="\t")
    return {str(r["arm"]): str(r["sig"]) for _, r in df.iterrows()
            if str(r["arm"]).startswith("fixed_low")}


def bar_figure(rows, stars, value, sd, fname, ylabel):
    n = len(rows)
    x = np.arange(n)
    vals = np.array([r[value] for r in rows])
    errs = np.array([r[sd] for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(x, vals, yerr=errs, color=[COLORS[r["color"]] for r in rows],
           width=0.55, edgecolor="black", linewidth=0.5,
           error_kw=dict(lw=0.8, capsize=3))

    base_val = rows[0][value]
    for xi, v, e, r in zip(x, vals, errs, rows):
        key = r.get("star_key")
        star = stars.get(key, "")
        ax.text(xi, v + e + 0.0025, f"{v:.4f}" + (f"\n{star}" if star else ""),
                ha="center", va="bottom", fontsize=9)

    ax.axhline(base_val, color="black", linestyle="--", lw=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(f"XGB blosum50 — {ylabel}: fixed-budget low-LR vs baseline "
                 "(mean ± std, 6 outer buckets; dashed: baseline "
                 f"{base_val:.4f})", fontsize=10.5)
    ax.set_ylim(min(vals - errs) - 0.005, max(vals + errs) + 0.010)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    stars = load_stars()

    base = load(BASELINE)
    base["label"] = "baseline\n0.1/1000"
    base["color"] = "baseline"
    rows = [base]
    for p in sorted(ARMS_DIR.glob("cv_results_*.json")):
        a = load(p)
        # lr0.05 -> fixed05, lr0.025 -> fixed025
        a["color"] = "fixed05" if a["label"].find("0.05") > -1 else "fixed025"
        m = re.search(r"xgblr([\dp]+)n(\d+)", a["tag"])
        a["star_key"] = f"fixed_low_lr{float(m.group(1).replace('p', '.'))}"
        rows.append(a)
    print(f"  {len(rows)} bars: baseline + {len(rows) - 1} fixed arms")

    bar_figure(rows, stars, "auc", "auc_sd", "fixed_lowlr_auc_bars.png",
               "AUC-ROC")
    bar_figure(rows, stars, "mcc", "mcc_sd", "fixed_lowlr_mcc_bars.png",
               "MCC")


if __name__ == "__main__":
    main()
