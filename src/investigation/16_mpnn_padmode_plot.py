#!/usr/bin/env python3
"""
16_mpnn_padmode_plot.py
Pad-mode (zero-pad vs boundary-repeat) comparison figures for ProteinMPNN.

Reproduces the bar-plot convention of 14_plot_grid_auc.plot_metric (same
PADTYPE_COLOR / PADMODE_ORDER / label style) and adds a paired-DeLong CI
panel from the 15_padmode_significance output.

Bar data comes from the cv_results_*.json CV metrics; the CI panel comes
from paired_tests.csv (delta_auc + ci95_lo/hi + p_delong_holm).

Inputs (defaults target the v_48_020 pad-mode campaign):
    <run>/models/flat/cv_results_xgb_emb_mpnn_*_flat_*.json
    <run>/results/paired_tests.csv

Outputs (default <run>/plots/):
    mpnn_padmode_auc.png      CV AUC-ROC mean +- std, per pad mode
    mpnn_padmode_mcc.png      CV MCC mean +- std, per pad mode
    mpnn_padmode_delong.png   delta-AUC with DeLong 95% CI, both families

Usage:
    python src/investigation/16_mpnn_padmode_plot.py
    python src/investigation/16_mpnn_padmode_plot.py \
        --run runs/mpnn_padmode_v48020 --smoke
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 14_/15_ have a leading digit — import via importlib. 14_ owns the bar-plot
# convention (plot_metric) and the pad-mode registry (FEATURE_MAP, colors).
_g = importlib.import_module("14_plot_grid_auc")
PADTYPE_MAP = _g.FEATURE_MAP
plot_metric = _g.plot_metric

from config import PROJECT_ROOT  # noqa: E402

DEFAULT_RUN = PROJECT_ROOT / "runs" / "mpnn_padmode_v48020"


# MPNN arms in a campaign may sit at different PC counts (90% variance is
# checkpoint-specific), so parse k from the filename like 15_ does.
_PCA_RE = re.compile(r"_pca(\d+)_flat_")


def load_bar_rows(run_dir: Path) -> pd.DataFrame:
    """One row per pad mode: CV mean +- std for AUC and MCC, plus runtime."""
    rows = []
    for p in sorted((run_dir / "models" / "flat").glob(
            "cv_results_xgb_emb_mpnn_*_flat_*.json")):
        r = json.load(open(p))
        feats = r["features_key"]
        toolkit, padtype = PADTYPE_MAP[feats]
        m = _PCA_RE.search(p.name)
        rows.append({
            "toolkit": toolkit,
            "padtype": padtype,
            # single PC level for every MPNN arm (90% variance, ckpt-specific)
            "k_label": "90%",
            "pca": int(m.group(1)) if m else np.nan,
            "runtime_h": r["runtime_seconds"] / 3600.0,
            "auc": r["cv_mean_auc_roc"],
            "auc_std": r["cv_std_auc_roc"],
            "mcc": r["cv_mean_mcc"],
            "mcc_std": r["cv_std_mcc"],
        })
    if not rows:
        raise SystemExit(f"No MPNN cv_results found under {run_dir}/models/flat")
    return pd.DataFrame(rows)


def plot_delong_ci(paired_csv: Path, outfile: Path, title: str) -> None:
    """delta-AUC with DeLong 95% CI, one row per test family.

    A CI touching/crossing zero = no significant difference at alpha 0.05.
    """
    df = pd.read_csv(paired_csv)
    df = df[df["test_group"].str.startswith("padmode_")].copy()
    if df.empty:
        raise SystemExit(f"no padmode_* rows in {paired_csv}")

    # friendlier family labels; padmode_proteinmpnn -> "padded subset (n=...)"
    def label(r):
        fam = "full data" if r["test_group"].endswith("_full") else "padded subset"
        return f"{fam} (n={int(r['n_rows']):,})"

    df["_label"] = df.apply(label, axis=1)
    df = df.sort_values("n_rows")           # subset first, full data last

    y = np.arange(len(df))
    d = df["delta_auc"].to_numpy(float)
    lo = df["ci95_lo"].to_numpy(float)
    hi = df["ci95_hi"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10.5, 0.9 * len(df) + 2.0))
    ax.barh(y, d, xerr=[d - lo, hi - d], color="#4C72B0", edgecolor="black",
            linewidth=0.4, height=0.55,
            error_kw=dict(lw=0.9, capsize=3, ecolor="black"))
    ax.axvline(0.0, color="black", lw=0.8, ls="--")

    span = max(float(hi.max() - lo.min()), 1e-9)
    ax.set_xlim(float(lo.min()) - 0.25 * span, float(hi.max()) + 0.25 * span)

    for yi, di, l, h, (_, r) in zip(y, d, lo, hi, df.iterrows()):
        # p_holm is the multiplicity-corrected DeLong p
        p = r.get("p_delong_holm", np.nan)
        verdict = "n/a" if pd.isna(p) else ("*" if p < 0.05 else "ns")
        ax.annotate(
            f"{di:+.4f}   [{l:+.4f}, {h:+.4f}]   p={p:.3f} {verdict}",
            xy=(1.01, yi), xycoords=("data", "y"), va="center", ha="left",
            fontsize=8, family="monospace", annotation_clip=False)

    ax.set_yticks(y)
    ax.set_yticklabels(df["_label"], fontsize=9)
    ax.set_xlabel("ΔAUC-ROC  (boundary − zero-pad)")
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    # room on the right for the annotation block
    fig.subplots_adjust(left=0.22, right=0.62, top=0.86, bottom=0.18)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  saved: {outfile}")


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

    df = load_bar_rows(run_dir)
    print(f"  {len(df)} pad-mode arms from {run_dir}")
    print(df[["toolkit", "padtype", "auc", "auc_std", "mcc", "mcc_std",
              "runtime_h"]].to_string(index=False,
                                      float_format=lambda x: f"{x:.4f}"))
    if args.smoke:
        return

    tag = "ProteinMPNN"
    plot_metric(df, "auc", "auc_std",
                f"{tag} — CV AUC-ROC (mean ± std, 5-fold), zero vs boundary",
                out_dir / "mpnn_padmode_auc.png", "AUC-ROC")
    plot_metric(df, "mcc", "mcc_std",
                f"{tag} — CV MCC (mean ± std, 5-fold), zero vs boundary",
                out_dir / "mpnn_padmode_mcc.png", "MCC")

    plot_delong_ci(
        run_dir / "results" / "paired_tests.csv",
        out_dir / "mpnn_padmode_delong.png",
        f"{tag} — ΔAUC (boundary − zero) with DeLong 95% CI\n"
        "dashed = no difference; CI crossing it ⇒ not significant",
    )
    print(f"\n  plots in: {out_dir}")


if __name__ == "__main__":
    main()
