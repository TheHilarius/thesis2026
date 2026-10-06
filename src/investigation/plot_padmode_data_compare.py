#!/usr/bin/env python3
"""
plot_padmode_data_compare.py — held-out AUC-ROC barplot for the
data-comparison grid (11 bars).

Bars (fixed order):
  sparse (one-hot AA) | BLOSUM50 | BLOSUM50 + engineered
  then 4 ESM-C (600M) pad modes and 4 ESM-IF1 pad modes, embeddings only.

Input: the aggregate TSV from extract_model_metrics.py (tab-delimited, one
row per run). Rows are filtered on model_key == the model used for the grid
(default xgb) and matched by feature_set.

Usage:
    python src/investigation/plot_padmode_data_compare.py \
        --tsv runs/padmode_data_<ts>/results/model_matrix_padmode.tsv \
        --out runs/padmode_data_<ts>/plots/

Outputs:
    padmode_data_compare_auc_roc.png   — barplot
    padmode_data_compare_auc_roc.csv   — the plotted values (ordered)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR_ORDER = [
    ("sparse", "Sparse one-hot AA", "csv"),
    ("blosum", "BLOSUM50 AA", "csv"),
    ("handcrafted_blosum", "BLOSUM50 + engineered", "csv"),
    ("emb_esmc_win_zeropad", "ESM-C zero-pad", "esmc"),
    ("emb_esmc_win_padtoken", "ESM-C pad-token", "esmc"),
    ("emb_esmc_win_impute_bos_eos", "ESM-C impute BOS/EOS", "esmc"),
    ("emb_esmc_win_impute_boundary", "ESM-C impute boundary", "esmc"),
    ("emb_esmif_win", "ESM-IF zero", "esmif"),
    ("emb_esmif_pad", "ESM-IF pad", "esmif"),
    ("emb_esmif_boundary", "ESM-IF boundary", "esmif"),
    ("emb_esmif_eos_bos_repeat", "ESM-IF eos/bos-repeat", "esmif"),
]

GROUP_COLORS = {
    "csv":   "#7f8c8d",
    "esmc":  "#c0392b",
    "esmif": "#2980b9",
}


def load_tsv(path: Path):
    """Return list of dict rows from the extract_model_metrics TSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_rows(rows, model, metric):
    """Index rows by feature_set for the requested model."""
    by_feat = {}
    for r in rows:
        if r.get("model_key", "").strip().lower() != model.lower():
            continue
        val = r.get(metric, "").strip()
        if val == "":
            continue
        by_feat[r["feature_set"].strip()] = float(val)
    return by_feat


def main():
    ap = argparse.ArgumentParser(
        description="Barplot of held-out AUC-ROC for the pad-mode/data grid."
    )
    ap.add_argument("--tsv", required=True, help="Aggregate TSV (extract_model_metrics.py output)")
    ap.add_argument("--out", default="results/figures/investigation",
                    help="Output directory for PNG + CSV")
    ap.add_argument("--model", default="xgb", help="model_key to plot (default: xgb)")
    ap.add_argument("--metric", default="heldout_auc_roc",
                    help="Metric column (default: heldout_auc_roc)")
    args = ap.parse_args()

    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        raise SystemExit(f"TSV not found: {tsv_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_tsv(tsv_path)
    by_feat = pick_rows(rows, args.model, args.metric)
    if not by_feat:
        raise SystemExit(
            f"No rows for model='{args.model}' / metric='{args.metric}' in {tsv_path}"
        )

    labels, groups, values = [], [], []
    missing = []
    for key, label, group in BAR_ORDER:
        if key in by_feat:
            labels.append(label)
            groups.append(group)
            values.append(by_feat[key])
        else:
            missing.append(key)

    if missing:
        print(f"WARNING: missing feature_set(s) in TSV: {missing}")
    if not labels:
        raise SystemExit("No matching bars in TSV")

    n = len(labels)
    x = np.arange(n)
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="black", linewidth=0.5)

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Held-out AUC-ROC")
    ax.set_ylim(0, min(1.0, max(values) * 1.18 + 0.02))
    ax.set_title(f"Data comparison grid — {args.model.upper()} "
                 f"(embeddings @ 90% EV, held-out AUC-ROC)")
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(6.5, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor="black")
               for c in GROUP_COLORS.values()]
    ax.legend(handles, [g.upper() for g in GROUP_COLORS], fontsize=8, loc="upper right")

    fig.tight_layout()
    png_path = out_dir / "padmode_data_compare_auc_roc.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = out_dir / "padmode_data_compare_auc_roc.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "feature_set", "label", "metric", "value"])
        for (key, label, group), v in zip(
            [(k, l, g) for k, l, g in BAR_ORDER if k in by_feat],
            values,
        ):
            w.writerow([group, key, label, args.metric, v])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()