#!/usr/bin/env python3
"""
plot_flank5_compare.py — 19mer (5-aa flank) vs 29mer (10-aa flank)
grouped barplot for the runs/flank5_compare campaign.

One bar group per feature set, two bars each (29mer grey / flank5 blue):
    BLOSUM50 · BLOSUM50 + engineered · ESM-C dimselect N=643

Values = nested-CV AUC-ROC (mean ± std over 6 outer iterations) from
cv_summary.auc_roc of the newest matching cv_results JSONs (same globs
as 32_flank5_significance.py — keep them in sync).

Significance brackets (stars + ΔAUC) come from
32_flank5_significance.py's paired_tests.csv and are drawn automatically
when present.

Style mirrors 22_compositional_barplot.py / padmode_choice figures:
grey baseline, capsize error bars, bold mean labels, chance line at 0.5,
0-1 axis.

Usage (project root):
    python src/investigation/plot_flank5_compare.py
    python src/investigation/plot_flank5_compare.py --stats-csv PATH
"""

import argparse
import csv
import json
from pathlib import Path

import sys
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

R = PROJECT_ROOT / "runs"

# group label -> (29mer glob, flank5 glob) — newest match wins
GROUPS = [
    ("BLOSUM50",
     "encodings_sweep/models/flat/cv_results_xgb_blosum50_flat_*.json",
     "flank5_compare/models/flat/cv_results_xgb_blosum50_flat_*.json"),
    ("BLOSUM50 +\nengineered",
     "handcrafted_sweep/models/flat/"
     "cv_results_xgb_handcrafted_blosum50_flat_*.json",
     "flank5_compare/models/flat/"
     "cv_results_xgb_handcrafted_blosum50_flat_*.json"),
    ("ESM-C dimselect\nN=643",
     "dimselect_matched/models/dimselect/"
     "cv_results_xgb_emb_esmc300m_win_padtoken_dimselect_top643_xgb_*.json",
     "dimselect_flank5/models/dimselect/"
     "cv_results_xgb_emb_esmc300m_win_padtoken_dimselect_top643_xgb_*.json"),
    ("CNN k5\n(ESM-C windows)",
     "flank5_compare/cnn/full/models/cnn/"
     "cv_results_cnn_simple_k5_*.json",
     "flank5_compare/cnn/flank5/models/cnn/"
     "cv_results_cnn_simple_k5_*.json"),
]

C_29 = "#BBBBBB"   # grey — 29mer (padmode AA-baseline grey)
C_19 = "#4477AA"   # Paul Tol blue — flank5


def load_cv(glob_rel: str):
    """(auc_mean, auc_std) from cv_summary.auc_roc of the newest JSON."""
    hits = sorted(R.glob(glob_rel))
    if not hits:
        return None
    d = json.loads(hits[-1].read_text())
    s = d["cv_summary"]["auc_roc"]
    return (s["mean"], s["std"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stats-csv", default=str(
        R / "flank5_compare/results/paired_tests.csv"),
        help="32_flank5_significance.py output; brackets drawn when it "
             "exists (pass a missing path to disable)")
    ap.add_argument("--out", default=str(
        R / "flank5_compare/plots/flank5_vs_29mer_auc_roc"),
        help="output prefix (PNG + CSV)")
    args = ap.parse_args()
    prefix = Path(args.out)

    # ── values ──
    bars = []  # (group_idx, variant, mean, std)
    missing = []
    for gi, (label, g29, g19) in enumerate(GROUPS):
        for variant, g in (("29", g29), ("19", g19)):
            v = load_cv(g)
            if v is None:
                missing.append((label.replace("\n", " "), variant, g))
            else:
                bars.append((gi, variant, *v))
    if missing:
        for label, variant, g in missing:
            print(f"  WARN: no results for {label} [{variant}] ({g})")
    if not bars:
        raise SystemExit("no flank5/29mer cv_results JSONs found")

    n_groups = len(GROUPS)
    # x layout: two slots per group, gap between groups
    slot = {(gi, v): gi * 2 + (0 if v == "29" else 1)
            for gi, v, _, _ in bars}
    xs = [slot[(gi, v)] for gi, v, _, _ in bars]
    means = [m for _, _, m, _ in bars]
    stds = [s for _, _, _, s in bars]
    colors = [C_29 if v == "29" else C_19 for _, v, _, _ in bars]

    fig, ax = plt.subplots(figsize=(1.15 * (2 * n_groups) + 4.5, 5.5))
    ax.bar(xs, means, yerr=stds, capsize=4, width=0.62,
           color=colors, edgecolor="black", linewidth=0.6)
    for x, m in zip(xs, means):
        ax.text(x, m + 0.012, f"{m:.4f}", ha="center",
                fontsize=9, fontweight="bold")

    # ── x ticks: group labels centered between the two slots ──
    ax.set_xticks([gi * 2 + 0.5 for gi in range(n_groups)])
    ax.set_xticklabels([lbl for lbl, _, _ in GROUPS], fontsize=9.5)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")

    # ── significance brackets (32_flank5_significance.py) ──
    stats_path = Path(args.stats_csv)
    if stats_path.exists():
        with open(stats_path) as fh:
            stats_rows = list(csv.DictReader(fh))
        # map 32_'s arm names (blosum50_29/blosum50_19, hc_blosum50_*,
        # esmc_dimselect_*) to this plot's group slots by prefix — robust
        # to skipped pairs in the CSV
        prefix_to_group = {"blosum50": 0, "hc_blosum50": 1,
                           "esmc_dimselect": 2, "cnn_k5": 3}
        data_top = max(m + s for m, s in zip(means, stds))
        y = data_top + 0.045
        for r in stats_rows:
            ga = prefix_to_group.get(r["arm_a"].rsplit("_", 1)[0])
            gb = prefix_to_group.get(r["arm_b"].rsplit("_", 1)[0])
            if ga is None or ga != gb:
                continue
            xa, xb = ga * 2, ga * 2 + 1
            ax.plot([xa, xa, xb, xb], [y - 0.004, y, y, y - 0.004],
                    lw=0.8, color="black")
            delta = float(r["delta"])
            txt = f"{r['star']}  Δ={delta:+.3f}"
            ax.text((xa + xb) / 2, y + 0.004, txt,
                    ha="center", va="bottom", fontsize=9)
            y += 0.055
    else:
        print(f"  stats CSV not found ({stats_path}) — no brackets")

    ax.legend(handles=[
        Patch(facecolor=C_29, edgecolor="black",
              label="29mer (10-aa flank)"),
        Patch(facecolor=C_19, edgecolor="black",
              label="19mer (5-aa flank)")],
        loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
        framealpha=0.9)
    ax.set_title("Flank trim: 19mer (5-aa) vs 29mer (10-aa) — "
                 "XGBoost nested-CV AUC-ROC", fontsize=12)
    fig.tight_layout()

    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(prefix) + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")

    csvp = Path(str(prefix) + ".csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "variant", "auc_roc", "std"])
        for gi, v, m, s in bars:
            w.writerow([GROUPS[gi][0].replace("\n", " "),
                        "29mer" if v == "29" else "19mer",
                        f"{m:.6f}", f"{s:.6f}"])
    print(f"  wrote {csvp}")


if __name__ == "__main__":
    main()
