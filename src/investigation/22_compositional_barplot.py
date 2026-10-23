#!/usr/bin/env python3
"""
22_compositional_barplot.py — AUC-ROC cross-validation barplots for the
compositional encoding (frac_* features) vs BLOSUM50.

Two figures (both written per invocation):
  1. compositional alone vs BLOSUM50
     runs/<runtag>/plots/compositional_vs_blosum50_auc.{png,csv}
  2. compositional + handcrafted_compositional vs BLOSUM50
     runs/<runtag>/plots/compositional_handcrafted_vs_blosum50_auc.{png,csv}

Values = nested-CV AUC-ROC (mean +/- std over 6 outer iterations) from
cv_summary.auc_roc in the flat cv_results JSONs:
  runs/try_out_compositional/models/flat/cv_results_xgb_compositional_*.json
  runs/try_out_compositional/models/flat/cv_results_xgb_handcrafted_compositional_*.json
  runs/encodings_sweep/models/flat/cv_results_xgb_blosum50_*.json

Style mirrors 20_dim21_barplot.py / padmode_choice figures: grey = AA
encoding baseline, capsize error bars, bold mean labels, chance line at 0.5.

Usage (project root):
    python src/investigation/22_compositional_barplot.py
    python src/investigation/22_compositional_barplot.py --runtag try_out_compositional
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

# fallback values if the cv_results JSONs are absent (verified 2026-10-22)
FALLBACK = {
    "compositional": (0.5794, 0.0061),
    "handcrafted_compositional": (0.6047, 0.0054),
    "blosum50": (0.7633, 0.0044),
}

N_FEATS = {
    "compositional": 21,
    "handcrafted_compositional": 77,
    "blosum50": 609,
}

COLORS = {
    "compositional": "#FDB462",               # light orange
    "handcrafted_compositional": "#D95F0E",   # deep orange
    "blosum50": "#BBBBBB",                    # AA-encoding grey (pca-sweep refs)
}


def load_cv_json(path_glob: str):
    """(auc_mean, auc_std) from cv_summary.auc_roc of the newest matching JSON."""
    hits = sorted(PROJECT_ROOT.glob(path_glob))
    if not hits:
        return None
    d = json.loads(hits[-1].read_text())
    s = d["cv_summary"]["auc_roc"]
    return (s["mean"], s["std"])


def draw(prefix: Path, bars, title: str, stats_csv=None):
    """bars: list of (label, (mean, std), color_key). Writes PNG + CSV."""
    fig, ax = plt.subplots(figsize=(1.15 * len(bars) + 4.5, 5.5))
    rows = []
    for i, (label, val, key) in enumerate(bars):
        ax.bar(i, val[0], yerr=val[1], capsize=4, width=0.62,
               color=COLORS[key], edgecolor="black", linewidth=0.6)
        ax.text(i, val[0] + val[1] + 0.012, f"{val[0]:.4f}",
                ha="center", fontsize=9, fontweight="bold")
        rows.append({"bar": label.replace("\n", " "),
                     "n_features": N_FEATS[key],
                     "auc_roc": f"{val[0]:.6f}", "std": f"{val[1]:.6f}"})

    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(
        [f"{k}\n({N_FEATS[k]} feat)" for _, _, k in bars], fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")

    # ── significance brackets (23_compositional_delong.py output) ──
    # pairs whose bars are absent from this plot are skipped silently
    if stats_csv:
        with open(stats_csv) as fh:
            stats_rows = list(csv.DictReader(fh, delimiter="\t"))
        arm_x = {k: i for i, (_, _, k) in enumerate(bars)}
        data_top = max(v[0] + v[1] for _, v, _ in bars)
        base, step, fam_lvl = data_top + 0.04, 0.034, {}
        for r in stats_rows:
            a, b = r["arm_a"], r["arm_b"]
            if a not in arm_x or b not in arm_x:
                continue  # pair not in this plot's bars
            lvl = fam_lvl.get(r["family"], 0)
            fam_lvl[r["family"]] = lvl + 1
            y = base + lvl * step
            xa, xb = sorted((arm_x[a], arm_x[b]))
            ax.plot([xa, xa, xb, xb], [y - 0.004, y, y, y - 0.004],
                    lw=0.8, color="black")
            ax.text((xa + xb) / 2, y + 0.002, r["star"],
                    ha="center", va="bottom", fontsize=9.5)
    ax.legend(handles=[Patch(facecolor=COLORS[k], edgecolor="black",
                             label=lab) for lab, _, k in bars],
              loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              framealpha=0.9)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(prefix) + "_auc.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    csvp = Path(str(prefix) + "_auc.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bar", "n_features", "auc_roc", "std"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {png}")
    print(f"  wrote {csvp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runtag", default="try_out_compositional",
                    help="run folder under runs/ that owns the outputs "
                         "(default: try_out_compositional)")
    ap.add_argument("--stats-csv", default=None,
                    help="compositional_delong_pairs.tsv from "
                         "23_compositional_delong.py — draw significance "
                         "brackets with stars. Default: runs/<runtag>/results/"
                         "compositional_delong_pairs.tsv if it exists "
                         "(brackets drawn automatically)")
    args = ap.parse_args()
    stats_csv = args.stats_csv
    if stats_csv is None:
        default_stats = (PROJECT_ROOT / "runs" / args.runtag / "results" /
                         "compositional_delong_pairs.tsv")
        stats_csv = str(default_stats) if default_stats.exists() else None
    print(f"  stats: {stats_csv or 'none (run 23_compositional_delong.py first)'}")

    print("Loading cv_results ...")
    vals = {}
    for key, glob_pat in {
        "compositional":
            f"runs/{args.runtag}/models/flat/cv_results_xgb_compositional_flat_*.json",
        "handcrafted_compositional":
            f"runs/{args.runtag}/models/flat/cv_results_xgb_handcrafted_compositional_flat_*.json",
        "blosum50":
            "runs/encodings_sweep/models/flat/cv_results_xgb_blosum50_flat_*.json",
    }.items():
        v = load_cv_json(glob_pat)
        vals[key] = v or FALLBACK[key]
        src = "cv_results" if v else "fallback"
        print(f"  {key:<28} {vals[key][0]:.4f} +/- {vals[key][1]:.4f}  ({src})")

    plots_dir = PROJECT_ROOT / "runs" / args.runtag / "plots"

    print("Plot 1: compositional vs BLOSUM50 ...")
    draw(
        plots_dir / "compositional_vs_blosum50",
        [
            ("compositional", vals["compositional"], "compositional"),
            ("BLOSUM50", vals["blosum50"], "blosum50"),
        ],
        "crossvalidation performance: AA composition",
        stats_csv=stats_csv,
    )

    print("Plot 2: compositional + handcrafted vs BLOSUM50 ...")
    draw(
        plots_dir / "compositional_handcrafted_vs_blosum50",
        [
            ("compositional", vals["compositional"], "compositional"),
            ("handcrafted_compositional", vals["handcrafted_compositional"],
             "handcrafted_compositional"),
            ("BLOSUM50", vals["blosum50"], "blosum50"),
        ],
        "crossvalidation performance: AA composition + handcrafted",
        stats_csv=stats_csv,
    )


if __name__ == "__main__":
    main()
