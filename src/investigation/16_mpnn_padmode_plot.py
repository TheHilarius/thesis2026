#!/usr/bin/env python3
"""
16_mpnn_padmode_plot.py
ProteinMPNN comparison figures — padmode_choice figure convention:
AUC-ROC bars + DeLong significance bracket (raw p + Holm p_adj), one
figure for all peptides and a _subset duplicate for padded peptides only.

Campaigns (--campaign):
    padmode  zero-pad vs boundary-repeat @ v_48_020
             -> runs/mpnn_padmode_v48020/plots/mpnn_padmode_auc_roc[_subset]
    ckpt     4 train-noise checkpoints, all zeropad
             -> runs/mpnn_ckpt_zeropad/plots/mpnn_ckpt_auc_roc[_subset]
             bracket spans the top-2 checkpoints by that figure's AUC.

Single data source: 15_padmode_significance.py's paired_tests.csv (per-arm
auc + delta + DeLong p/Holm) — bars, bracket and CSV sidecar stay mutually
consistent (no mixing with cv_summary).

Usage:
    python src/investigation/16_mpnn_padmode_plot.py --smoke
    python src/investigation/16_mpnn_padmode_plot.py
    python src/investigation/16_mpnn_padmode_plot.py --campaign ckpt
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

# Paul Tol bright (colour-blind safe); green = the MPNN group
MPNN_COLOR = "#44AA99"

# CSV schema: 15_ stores model_a/model_b per pair row -> auc_a/auc_b.
# Bars: (model key in paired_tests.csv, label, feature-set key, auc column
# of the single pair row — used by padmode's 2-bar layout).
CAMPAIGNS = {
    "padmode": {
        "run": PROJECT_ROOT / "runs" / "mpnn_padmode_v48020",
        "stem": "mpnn_padmode_auc_roc",
        "groups": [("", "padmode_proteinmpnn_full", "all peptides"),
                   ("_subset", "padmode_proteinmpnn", "padded peptides only")],
        "bars": [("zero", "zero-pad", "emb_mpnn_v48020_zeropad"),
                 ("boundary", "boundary-repeat", "emb_mpnn_v48020_boundary")],
    },
    "ckpt": {
        "run": PROJECT_ROOT / "runs" / "mpnn_ckpt_zeropad",
        "stem": "mpnn_ckpt_auc_roc",
        "groups": [("", "ckpt_proteinmpnn_full", "all peptides"),
                   ("_subset", "ckpt_proteinmpnn", "padded peptides only")],
        "bars": [("v48002", "v_48_002\n(0.02 Å)", "emb_mpnn_v48002_zeropad"),
                 ("v48010", "v_48_010\n(0.10 Å)", "emb_mpnn_v48010_zeropad"),
                 ("v48020", "v_48_020\n(0.20 Å)", "emb_mpnn_v48020_zeropad"),
                 ("v48030", "v_48_030\n(0.30 Å)", "emb_mpnn_v48030_zeropad")],
    },
}


def load_group(paired_csv: Path, test_group: str, arms: list[str]) -> dict:
    """One test_group's rows -> per-arm AUCs + per-pair DeLong stats.

    AUC per arm is identical in every row it appears in (same pooled OOF
    vector); taken from the first row, consistency-checked.
    """
    arm_auc, pairs = {}, {}
    with open(paired_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["test_group"].strip() != test_group:
                continue
            for side in ("a", "b"):
                arm, auc = r[f"model_{side}"].strip(), float(r[f"auc_{side}"])
                if arm not in arms:
                    continue
                if arm in arm_auc:
                    assert abs(arm_auc[arm] - auc) < 1e-12, (arm, arm_auc[arm], auc)
                arm_auc[arm] = auc
            pairs[frozenset((r["model_a"].strip(), r["model_b"].strip()))] = {
                "delta": float(r["delta_auc"]),
                "p": float(r["p_delong"]),
                "p_adj": float(r["p_delong_holm"]),
                "n_rows": int(float(r["n_rows"])),
            }
    missing = [a for a in arms if a not in arm_auc]
    if missing:
        raise SystemExit(f"{test_group}: no rows for arms {missing} in {paired_csv}")
    return {"auc": arm_auc, "pairs": pairs}


def plot_bars(g: dict, bars: list[tuple], title: str, outfile: Path) -> None:
    """AUC bars + DeLong bracket over the top-2 arms — padmode_choice style."""
    labels = [lbl for _, lbl, _ in bars]
    values = [g["auc"][key] for key, _, _ in bars]

    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.bar(x, values, color=MPNN_COLOR, width=0.62,
           edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(0, 1.0)

    # bracket spans the top-2 arms by this figure's AUC (headline pair)
    order = sorted(range(len(bars)), key=lambda i: -values[i])
    i_best, i_run = order[:2]
    pair = g["pairs"][frozenset((bars[i_best][0], bars[i_run][0]))]
    y0 = max(values) * 1.08
    ax.plot([x[i_best], x[i_best], x[i_run], x[i_run]],
            [y0 * 0.985, y0, y0, y0 * 0.985],
            color="black", lw=0.9)
    star_src = pair["p_adj"] if pair["p_adj"] is not None else pair["p"]
    ax.text((x[i_best] + x[i_run]) / 2, y0 + max(values) * 0.02,
            f"{pdp.fmt_p_pair(pair['p'], pair['p_adj'])} ({pdp.stars(star_src)})",
            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
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
        for key, lbl, feat in bars:
            w.writerow(["mpnn", feat, lbl.replace("\n", " "), "auc_roc",
                        g["auc"][key]])
    print(f"  saved: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ProteinMPNN comparison figures (padmode_choice style).")
    ap.add_argument("--campaign", default="padmode", choices=sorted(CAMPAIGNS),
                    help="padmode = zero vs boundary @ v_48_020; "
                         "ckpt = 4 checkpoints, zeropad")
    ap.add_argument("--run", default=None,
                    help="override campaign run folder")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run>/plots")
    ap.add_argument("--smoke", action="store_true",
                    help="print the loaded rows and exit before plotting")
    args = ap.parse_args()

    camp = CAMPAIGNS[args.campaign]
    run_dir = Path(args.run) if args.run else Path(camp["run"])
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_csv = run_dir / "results" / "paired_tests.csv"
    bars = camp["bars"]
    arms = [key for key, _, _ in bars]

    groups = [(suffix, scope, load_group(paired_csv, tg, arms))
              for suffix, tg, scope in camp["groups"]]
    for (suffix, scope, g), (_, tg, _) in zip(groups, camp["groups"]):
        auc_str = "  ".join(f"{a}={g['auc'][a]:.5f}" for a in arms)
        top = sorted(arms, key=lambda a: -g["auc"][a])[:2]
        pair = g["pairs"][frozenset(top)]
        print(f"  {tg} (n={pair['n_rows']:,}): {auc_str}  "
              f"top2={top[0]}/{top[1]} delta={pair['delta']:+.4f} "
              f"p={pair['p']:.4g} p_adj={pair['p_adj']:.4g}")
    if args.smoke:
        return

    for suffix, scope, g in groups:
        # n_rows is identical across a group's pair rows (same test rows)
        n_rows = next(iter(g["pairs"].values()))["n_rows"]
        plot_bars(
            g, bars,
            f"ProteinMPNN — held-out AUC-ROC, {scope}\n"
            f"nested CV (OOF), n={n_rows:,}; "
            "DeLong bracket: top-2 arms, raw p + Holm p_adj",
            out_dir / f"{camp['stem']}{suffix}.png",
        )
    print(f"\n  plots in: {out_dir}")


if __name__ == "__main__":
    main()
