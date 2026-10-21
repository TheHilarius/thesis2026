#!/usr/bin/env python3
"""
16_mpnn_padmode_plot.py
Pad-mode (zero-pad vs boundary-repeat) comparison figures for ProteinMPNN.

Mirrors plot_padmode_data_compare.py (padmode_choice convention): AUC-ROC
bars with a DeLong significance bracket spanning the two bars, one figure
for all peptides and a _subset duplicate for the padded peptides only.

Single data source: 15_padmode_significance.py's paired_tests.csv, which
carries auc_a/auc_b + delta + DeLong p/Holm per test group — bars, bracket
and CSV sidecar all stay mutually consistent (no mixing with cv_summary).

Outputs (default <run>/plots/):
    mpnn_padmode_auc_roc.png/.csv           all peptides
    mpnn_padmode_auc_roc_subset.png/.csv    padded peptides only

Usage:
    python src/investigation/16_mpnn_padmode_plot.py
    python src/investigation/16_mpnn_padmode_plot.py \
        --run runs/mpnn_padmode_v48020 --smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# padmode_choice reference: reuse its star/p-text formatting so bracket
# wording is identical across campaigns
import plot_padmode_data_compare as pdp

from config import PROJECT_ROOT  # noqa: E402

DEFAULT_RUN = PROJECT_ROOT / "runs" / "mpnn_padmode_v48020"

# Paul Tol bright (colour-blind safe); green = the MPNN group
MPNN_COLOR = "#44AA99"

# bar order: chosen pad mode first (zero-pad), challenger second.
# CSV schema: model_a=boundary / model_b=zero -> auc_a / auc_b columns.
BARS = [  # (auc column in paired_tests.csv, label, feature-set key)
    ("auc_b", "zero-pad", "emb_mpnn_v48020_zeropad"),
    ("auc_a", "boundary-repeat", "emb_mpnn_v48020_boundary"),
]

# (suffix, test_group, title) — full data first, padded subset second
FIGURES = [
    ("", "padmode_proteinmpnn_full", "all peptides"),
    ("_subset", "padmode_proteinmpnn", "padded peptides only"),
]


def load_group(paired_csv: Path, test_group: str) -> dict:
    """One paired_tests.csv row -> bar values + bracket stats.

    CSV stores model_a=boundary / model_b=zero, so auc_a is the boundary
    bar and auc_b the zero-pad bar.
    """
    with open(paired_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["test_group"].strip() == test_group:
                vals = {col: float(r[col]) for col, _, _ in BARS
                        if r[col].strip()}
                return {
                    "vals": vals,
                    "n_rows": int(float(r["n_rows"])),
                    "delta": float(r["delta_auc"]),
                    "p": float(r["p_delong"]),
                    "p_adj": float(r["p_delong_holm"]),
                }
    raise SystemExit(f"no row test_group={test_group!r} in {paired_csv}")


def plot_bars(g: dict, title: str, outfile: Path) -> None:
    """Two AUC bars + DeLong bracket — padmode_choice figure style."""
    labels = [lbl for _, lbl, _ in BARS]
    values = [g["vals"][key] for key, _, _ in BARS]

    x = np.arange(len(BARS))
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.bar(x, values, color=MPNN_COLOR, width=0.62,
           edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(0, 1.0)

    # significance bracket over the two bars (same wording as
    # plot_padmode_data_compare: raw p + Holm p_adj, stars from p_adj)
    y0 = max(values) * 1.08
    ax.plot([x[0], x[0], x[1], x[1]],
            [y0 * 0.985, y0, y0, y0 * 0.985],
            color="black", lw=0.9)
    star_src = g["p_adj"] if g["p_adj"] is not None else g["p"]
    ax.text(x.mean(), y0 + max(values) * 0.02,
            f"{pdp.fmt_p_pair(g['p'], g['p_adj'])} ({pdp.stars(star_src)})",
            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {outfile}")

    csv_path = outfile.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "feature_set", "label", "metric", "value"])
        for (key, lbl, feat) in BARS:
            w.writerow(["mpnn", feat, lbl, "auc_roc", g["vals"][key]])
    print(f"  saved: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ProteinMPNN pad-mode comparison figures.")
    ap.add_argument("--run", default=str(DEFAULT_RUN),
                    help="campaign folder (default: %(default)s)")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run>/plots")
    ap.add_argument("--smoke", action="store_true",
                    help="print the loaded rows and exit before plotting")
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_csv = run_dir / "results" / "paired_tests.csv"

    groups = [(suffix, tg, title, load_group(paired_csv, tg))
              for suffix, tg, title in FIGURES]
    for _, tg, _, g in groups:
        print(f"  {tg}: n={g['n_rows']:,}  zero={g['vals']['auc_b']:.5f}  "
              f"boundary={g['vals']['auc_a']:.5f}  "
              f"delta={g['delta']:+.4f}  p={g['p']:.4g}  p_adj={g['p_adj']:.4g}")
    if args.smoke:
        return

    for suffix, _, scope, g in groups:
        plot_bars(
            g,
            f"ProteinMPNN — held-out AUC-ROC, {scope}\n"
            f"nested CV (OOF), n={g['n_rows']:,}; DeLong bracket: "
            "raw p + Holm p_adj",
            out_dir / f"mpnn_padmode_auc_roc{suffix}.png",
        )
    print(f"\n  plots in: {out_dir}")


if __name__ == "__main__":
    main()
