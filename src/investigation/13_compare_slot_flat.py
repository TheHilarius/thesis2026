#!/usr/bin/env python3
"""
13_compare_slot_flat.py
Slot-PCA vs flat-PCA comparison across the window-embedding modelling grid.

Scope: the base PCA sweeps only (the three jobs 50540/50541/50542 plus the
individual slot-ESM-IF re-runs), i.e. the 8 window feature sets x 3 models
(rf, xgb, lr_l2) x 2 PCA modes (slot, flat) x 3 sweep values.

PCA values were chosen to hit explained-variance (EV) anchors, so slot and
flat are NOT comparable at equal component counts. This script maps every
result to its true cumulative EV (read from the saved variance curves) so the
head-to-head can be anchored on EV.

Only ~50% EV is shared between the modes:
    slot  esmc 9=50.1%  334=85.0%  742=95.0%   | esmif 47=50.4%  214=85.0%  359=95.0%
    flat  esmc 10=30.3% 97=50.1%   469=65.0%   | esmif 76=30.1%  345=50.0%  851=65.0%
slot 85/95% exceed the flat EV ceiling (~72%), so the EV-curve figure carries
the non-matched points and the delta table only reports the 50% anchor.

Inputs (local after rsync into server_results/):
    server_results/**/cv_results_*.json
    server_results/results/figures/models/pca_optimization/slot/
        pca_variance_windows_{key}.csv            (slot per-position curves)
    server_results/runs/padfull_flat_20261001_1645/plots/pca_variance/
        pca_variance_windows_{key}_full.csv       (flat full-spectrum curves)

Outputs:
    results/tables/slot_vs_flat_results.tsv        long form, one row/result
    results/tables/slot_vs_flat_leaderboard.tsv    ranked by cv AUC-ROC
    results/tables/slot_vs_flat_deltas_50ev.tsv    flat - slot at ~50% EV
    results/figures/slot_vs_flat/slot_vs_flat_auc.png
    results/figures/slot_vs_flat/slot_vs_flat_curve.png
    results/figures/slot_vs_flat/slot_vs_flat_padding.png

Usage:
    python src/investigation/13_compare_slot_flat.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent

# ── Scope ─────────────────────────────────────────────────────────────────────
# feature_key -> (embedding, pad label, variance-curve key)
CORE_FEATURES = {
    "handcrafted_sparse_esmc_win_zeropad": ("esmc", "zero-pad", "esmc_zeropad"),
    "handcrafted_sparse_esmc_win_padtoken": ("esmc", "pad-token", "esmc_padtoken"),
    "handcrafted_sparse_esmc_win_impute_bos_eos": (
        "esmc", "impute BOS/EOS", "esmc_impute_bos_eos"),
    "handcrafted_sparse_esmc_win_impute_boundary": (
        "esmc", "impute boundary", "esmc_impute_boundary"),
    "handcrafted_sparse_esmif_win": ("esmif", "zero (win)", "esmif_zero"),
    "handcrafted_sparse_esmif_pad": ("esmif", "pad", "esmif_pad"),
    "handcrafted_sparse_esmif_boundary": ("esmif", "boundary", "esmif_boundary"),
    "handcrafted_sparse_esmif_eos_bos_repeat": (
        "esmif", "EOS/BOS repeat", "esmif_eos_bos_repeat"),
}

SLOT_PCA = {"esmc": {9, 334, 742}, "esmif": {47, 214, 359}}
FLAT_PCA = {"esmc": {10, 97, 469}, "esmif": {76, 345, 851}}

EMB_ORDER = {"esmc": "ESM-C", "esmif": "ESM-IF"}
MODEL_ORDER = ["rf", "xgb", "lr_l2"]
PAD_ORDER = {
    "esmc": ["zero-pad", "pad-token", "impute BOS/EOS", "impute boundary"],
    "esmif": ["zero (win)", "pad", "boundary", "EOS/BOS repeat"],
}
MODE_COLORS = {"slot": "#1f77b4", "flat": "#d62728"}
EV_ANCHOR = 50.0

METRICS = ["auc_roc", "auc_pr", "mcc", "f1", "accuracy"]


def load_variance_curves(root: Path) -> dict:
    """Return {(mode, curve_key): {component_idx: cumulative_variance}}."""
    curves = {}
    slot_dir = root / "results" / "figures" / "models" / "pca_optimization" / "slot"
    flat_dir = (root / "runs" / "padfull_flat_20261001_1645" / "plots" / "pca_variance")
    key2file = {v[2]: k for k, v in CORE_FEATURES.items()}
    for curve_key in key2file:
        p = slot_dir / f"pca_variance_windows_{curve_key}.csv"
        if p.exists():
            curves[("slot", curve_key)] = _read_curve(p)
        p = flat_dir / f"pca_variance_windows_{curve_key}_full.csv"
        if p.exists():
            curves[("flat", curve_key)] = _read_curve(p)
    return curves


def _read_curve(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["component_idx"])] = float(row["cumulative_variance"])
    return out


def _run_timestamp(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else ""


def collect(root: Path, curves: dict) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("**/cv_results_*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        feat = d.get("features_key", "")
        if feat not in CORE_FEATURES:
            continue
        mode = d.get("config", {}).get("pca_mode")
        if mode not in ("slot", "flat"):
            continue
        m = re.search(r"_pca(\d+)", path.name)
        pca = int(m.group(1)) if m else None
        emb, pad, curve_key = CORE_FEATURES[feat]
        allowed = SLOT_PCA if mode == "slot" else FLAT_PCA
        if pca not in allowed[emb]:
            continue  # partner's expanded sweeps / legacy values

        cfg = d.get("config", {})
        cv = d.get("cv_summary", {})
        comp = cfg.get("component_info", {}).get("emb_components", [])
        emb_pca_total = comp[0].get("pca_total") if comp else None
        curve = curves.get((mode, curve_key), {})
        ev = curve.get(pca)

        row = {
            "model": d.get("model_key"),
            "embedding": emb,
            "pad_method": pad,
            "pca_mode": mode,
            "pca": pca,
            "ev_pct": round(ev * 100, 2) if ev is not None else np.nan,
            "n_features": cfg.get("n_features"),
            "emb_pca_total": emb_pca_total,
            "timestamp": d.get("timestamp", ""),
            "result_file": str(path.relative_to(root)),
        }
        for metric in METRICS:
            stats = cv.get(metric, {}) or {}
            row[f"cv_{metric}_mean"] = stats.get("mean", np.nan)
            row[f"cv_{metric}_std"] = stats.get("std", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # de-duplicate (model, feature, mode, pca): keep latest timestamp
    df["__key"] = (
        df["model"] + "|" + df["embedding"] + "|" + df["pad_method"]
        + "|" + df["pca_mode"] + "|" + df["pca"].astype(str)
    )
    df = (df.sort_values("timestamp")
            .drop_duplicates("__key", keep="last")
            .drop(columns="__key")
            .reset_index(drop=True))
    return df


def build_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """flat - slot at the shared ~50% EV anchor, per (model, embedding, pad)."""
    rows = []
    for (model, emb, pad), g in df.groupby(["model", "embedding", "pad_method"]):
        slot = g[g.pca_mode == "slot"]
        flat = g[g.pca_mode == "flat"]
        if slot.empty or flat.empty:
            continue
        s = slot.iloc[(slot.ev_pct - EV_ANCHOR).abs().argmin()]
        f = flat.iloc[(flat.ev_pct - EV_ANCHOR).abs().argmin()]
        row = {
            "model": model, "embedding": emb, "pad_method": pad,
            "slot_pca": int(s.pca), "slot_ev": s.ev_pct,
            "flat_pca": int(f.pca), "flat_ev": f.ev_pct,
        }
        for metric in ["auc_roc", "auc_pr", "mcc"]:
            row[f"slot_{metric}"] = s[f"cv_{metric}_mean"]
            row[f"flat_{metric}"] = f[f"cv_{metric}_mean"]
            row[f"delta_{metric}"] = f[f"cv_{metric}_mean"] - s[f"cv_{metric}_mean"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["embedding", "model", "pad_method"]).reset_index(drop=True)
    return out


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_auc(df: pd.DataFrame, out_dir: Path):
    """Mean AUC (over pad methods) at 50% EV: slot vs flat, per model."""
    d = df[(df.ev_pct - EV_ANCHOR).abs() < 5].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, emb in zip(axes, ["esmc", "esmif"]):
        sub = d[d.embedding == emb]
        models = [m for m in MODEL_ORDER if m in set(sub.model)]
        x = np.arange(len(models))
        w = 0.36
        for i, mode in enumerate(["slot", "flat"]):
            means, errs = [], []
            for m in models:
                vals = sub[(sub.model == m) & (sub.pca_mode == mode)][
                    "cv_auc_roc_mean"]
                means.append(vals.mean())
                errs.append(vals.std())
            ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3,
                   label=mode, color=MODE_COLORS[mode], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(f"{EMB_ORDER[emb]}  (~50% EV)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("CV AUC-ROC (mean over pad methods)")
    axes[1].legend(title="PCA mode")
    fig.suptitle("Slot vs flat PCA — matched at ~50% explained variance")
    fig.tight_layout()
    fig.savefig(out_dir / "slot_vs_flat_auc.png", dpi=150)
    plt.close(fig)


def fig_curve(df: pd.DataFrame, out_dir: Path):
    """AUC vs explained variance; slot vs flat curves per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    styles = {"rf": "o", "xgb": "s", "lr_l2": "^"}
    for ax, emb in zip(axes, ["esmc", "esmif"]):
        sub = df[df.embedding == emb]
        for model in MODEL_ORDER:
            for mode in ["slot", "flat"]:
                g = sub[(sub.model == model) & (sub.pca_mode == mode)]
                if g.empty:
                    continue
                agg = g.groupby("ev_pct")["cv_auc_roc_mean"].agg(["mean", "std"]).reset_index()
                agg = agg.sort_values("ev_pct")
                ax.errorbar(agg.ev_pct, agg["mean"], yerr=agg["std"],
                            fmt=styles[model], color=MODE_COLORS[mode],
                            ls="-" if mode == "slot" else "--",
                            mfc="none" if mode == "flat" else None,
                            capsize=2, label=f"{model} {mode}")
        ax.axvline(EV_ANCHOR, color="grey", ls=":", lw=1)
        ax.set_xlabel("cumulative explained variance (%)")
        ax.set_title(EMB_ORDER[emb])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("CV AUC-ROC (mean over pad methods)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Slot vs flat PCA — CV AUC-ROC vs explained variance")
    fig.tight_layout()
    fig.savefig(out_dir / "slot_vs_flat_curve.png", dpi=150)
    plt.close(fig)


def fig_padding(df: pd.DataFrame, out_dir: Path):
    """Padding method comparison at ~50% EV, slot vs flat, per model."""
    d = df[(df.ev_pct - EV_ANCHOR).abs() < 5].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for r, emb in enumerate(["esmc", "esmif"]):
        pads = PAD_ORDER[emb]
        x = np.arange(len(pads))
        w = 0.36
        for c, model in enumerate(MODEL_ORDER):
            ax = axes[r, c]
            for i, mode in enumerate(["slot", "flat"]):
                vals = []
                for p in pads:
                    v = d[(d.embedding == emb) & (d.model == model)
                          & (d.pad_method == p) & (d.pca_mode == mode)][
                        "cv_auc_roc_mean"]
                    vals.append(v.iloc[0] if len(v) else np.nan)
                ax.bar(x + (i - 0.5) * w, vals, w, label=mode,
                       color=MODE_COLORS[mode], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(pads, rotation=30, ha="right", fontsize=8)
            if r == 0:
                ax.set_title(model)
            if c == 0:
                ax.set_ylabel(f"{EMB_ORDER[emb]}\nCV AUC-ROC")
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(title="PCA mode")
    fig.suptitle("Padding-method comparison at ~50% EV (horizontal grid)")
    fig.tight_layout()
    fig.savefig(out_dir / "slot_vs_flat_padding.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=PROJECT_ROOT / "server_results",
                    help="Rsynced results root (default: server_results/)")
    ap.add_argument("--tables-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "figures" / "slot_vs_flat")
    args = ap.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    curves = load_variance_curves(args.root)
    print(f"loaded {len(curves)} variance curves from {args.root}")
    df = collect(args.root, curves)
    if df.empty:
        raise SystemExit("No core results found — check --root / rsync.")
    print(f"collected {len(df)} core results")

    # long table
    df_out = df.sort_values(
        ["embedding", "model", "pad_method", "pca_mode", "pca"]
    ).reset_index(drop=True)
    df_out.to_csv(args.tables_dir / "slot_vs_flat_results.tsv",
                  sep="\t", index=False)

    # leaderboard
    (df.sort_values("cv_auc_roc_mean", ascending=False)
       .reset_index(drop=True)
       .to_csv(args.tables_dir / "slot_vs_flat_leaderboard.tsv",
               sep="\t", index=False))

    # deltas at 50% EV
    deltas = build_deltas(df)
    deltas.to_csv(args.tables_dir / "slot_vs_flat_deltas_50ev.tsv",
                  sep="\t", index=False)

    # figures
    fig_auc(df, args.figures_dir)
    fig_curve(df, args.figures_dir)
    fig_padding(df, args.figures_dir)

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  SLOT vs FLAT — top 15 by CV AUC-ROC")
    print("=" * 78)
    cols = ["model", "embedding", "pad_method", "pca_mode", "pca", "ev_pct",
            "n_features", "cv_auc_roc_mean", "cv_auc_pr_mean", "cv_mcc_mean"]
    with pd.option_context("display.width", 160, "display.max_columns", 50):
        print(df.sort_values("cv_auc_roc_mean", ascending=False)
                .head(15)[cols].to_string(index=False))

    print("\n" + "=" * 78)
    print("  flat - slot delta at ~50% EV (AUC-ROC)")
    print("=" * 78)
    if not deltas.empty:
        with pd.option_context("display.width", 160):
            print(deltas[["model", "embedding", "pad_method",
                          "slot_pca", "slot_ev", "flat_pca", "flat_ev",
                          "slot_auc_roc", "flat_auc_roc", "delta_auc_roc",
                          "delta_mcc"]].to_string(index=False))
        print(f"\nmean delta AUC-ROC (flat - slot): "
              f"{deltas['delta_auc_roc'].mean():+.4f}")
        print(f"mean delta MCC     (flat - slot): "
              f"{deltas['delta_mcc'].mean():+.4f}")

    # gaps
    print("\n" + "=" * 78)
    print("  COVERAGE GAPS (expected 4 pads x 3 pca per model x mode x embedding)")
    print("=" * 78)
    for emb in ["esmc", "esmif"]:
        for mode in ["slot", "flat"]:
            for model in MODEL_ORDER:
                n = len(df[(df.embedding == emb) & (df.pca_mode == mode)
                           & (df.model == model)])
                if n != 12:
                    print(f"  {EMB_ORDER[emb]:6s} {mode:4s} {model:6s}: "
                          f"{n}/12 cells")

    print(f"\nWrote tables to {args.tables_dir}")
    print(f"Wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
