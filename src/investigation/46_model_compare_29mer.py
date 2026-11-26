#!/usr/bin/env python3
"""
46_model_compare_29mer.py — XGBoost vs simpler models (rf, lr_l2,
lr_elasticnet) on BOTH aa encodings (one-hot sparse, BLOSUM50), 29mer split.

Dataset: runs/modelcompare_29mer_46617 (THESIS_VARIANT=none +
THESIS_SPLIT=df_all_with_folds.csv, n=46,617, duplicate windows kept).
The xgb arms are symlinked from runs/encodings_sweep (20261014_154653).

Part A  padmode-style bar figure (2 groups = encodings, 4 bars = models,
        pooled OOF AUC +/- outer-fold SD, stars-only DeLong brackets xgb vs
        each simpler model within each encoding, Holm family = 3 tests/enc).
Part B  one ROC figure per encoding: 4 pooled-OOF curves, AUC in legend.
Part C  results/*.tsv: metrics + full DeLong stats (raw p + Holm p_adj).

Env pinning: THESIS_VARIANT / THESIS_SPLIT are set BEFORE d18 (and hence
config) is imported, so a shell default of flank5_rmdup can never load the
wrong folds; a 46,617-row assert backs it up.

Run from repo root:
    python src/investigation/46_model_compare_29mer.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── pin the split BEFORE config is imported (d18 imports config) ───────────
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["THESIS_VARIANT"] = "none"
os.environ["THESIS_SPLIT"] = str(
    PROJECT_ROOT / "data" / "processed" / "df_all_with_folds.csv")

spec = importlib.util.spec_from_file_location(
    "d18", PROJECT_ROOT / "src" / "investigation" / "18_encodings_sweep_delong.py")
d18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d18)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

CAMPAIGN = PROJECT_ROOT / "runs" / "modelcompare_29mer_46617"
PLOTS = CAMPAIGN / "plots"
RESULTS = CAMPAIGN / "results"

# model key, display label, colour (padmode palette: grey / red / blue / tint)
MODELS = [
    ("xgb",            "XGBoost",              "#BBBBBB"),
    ("rf",             "Random forest",        "#EE6677"),
    ("lr_l2",          "LR (L2)",              "#4477AA"),
    ("lr_elasticnet",  "LR (elastic net)",     "#88CCEE"),
]
ENCODINGS = ["sparse", "blosum50"]
ENC_LABEL = {"sparse": "One-hot (sparse)", "blosum50": "BLOSUM50"}
N_ROWS_EXPECTED = 46617
STEP_G = 4.0          # bar centres per group: 4 models


def load_arms(fold_vec, y):
    """(model, features) -> d18.load_model dict. Newest timestamp wins;
    symlinked xgb JSONs resolve like any other file."""
    arms = {}
    flat = CAMPAIGN / "models" / "flat"
    for model, _lbl, _c in MODELS:
        for enc in ENCODINGS:
            hits = sorted(glob.glob(str(
                flat / f"cv_results_{model}_{enc}_flat_*.json")))
            if not hits:
                sys.exit(f"FATAL: missing cv_results for {model} x {enc} "
                         f"under {flat} (symlink the xgb arms first)")
            if not os.path.exists(hits[-1]):
                sys.exit(f"FATAL: dangling symlink: {hits[-1]} -- fetch the "
                         f"encodings_sweep xgb JSON to the laptop first")
            m = d18.load_model(hits[-1], fold_vec, y, f"{model}/{enc}")
            m["model"] = model
            m["enc"] = enc
            m["file"] = os.path.basename(hits[-1])
            arms[(model, enc)] = m
            print(f"  {model:<14} {enc:<9} pooled AUC {m['auc']:.4f} "
                  f"+/- {m['outer_std']:.4f}  {m['file']}")
    return arms


def delong_table(arms, y):
    """xgb vs each simpler model, per encoding; Holm family = 3/encoding."""
    rows = []
    for enc in ENCODINGS:
        fam = []
        px = arms[("xgb", enc)]["probs"]
        for model, lbl, _c in MODELS[1:]:
            pa, pb = arms[("xgb", enc)]["auc"], arms[(model, enc)]["auc"]
            _a, _b, delta, se, z, p = d18.delong_full(y, px,
                                                      arms[(model, enc)]["probs"])
            fam.append(dict(encoding=enc, model_a="xgb", model_b=model,
                            label_a="XGBoost", label_b=lbl,
                            auc_a=pa, auc_b=pb, delta=delta, se=se, z=z, p=p))
        adj = d18.holm_adjust(np.array([r["p"] for r in fam]))
        for r, a in zip(fam, adj):
            r["p_adj"] = float(a)
            r["stars"] = d18.stars(float(a))
            rows.append(r)
            print(f"  {enc:<9} xgb vs {r['model_b']:<14} delta={r['delta']:+.4f} "
                  f"z={r['z']:+.2f} p={r['p']:.3g} p_adj={r['p_adj']:.3g} "
                  f"{r['stars']}")
    return rows


def plot_bars(arms, delong_rows):
    """padmode-style: 2 encoding groups x 4 model bars + stars-only brackets."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    xs, vals, errs, colors = [], [], [], []
    for gi, enc in enumerate(ENCODINGS):
        for mi, (model, _lbl, c) in enumerate(MODELS):
            m = arms[(model, enc)]
            xs.append(gi * STEP_G + mi)
            vals.append(float(m["auc"]))
            errs.append(float(m["outer_std"]))
            colors.append(c)
    d18._padmode_bars(ax, xs, vals, errs, colors)
    ax.axvline(3.5, color="grey", linestyle="--", linewidth=0.8)  # divider

    fam = {(r["encoding"], r["model_b"]): r for r in delong_rows}
    pairs = []
    for gi, enc in enumerate(ENCODINGS):
        for mi in (1, 2, 3):                       # xgb (mi=0) vs each other
            r = fam[(enc, MODELS[mi][0])]
            pairs.append((gi * STEP_G + 0, gi * STEP_G + mi,
                          r["p"], r["p_adj"]))
    d18._padmode_brackets(ax, pairs, vals, stars_only=True)

    # padmode style: group labels as x ticks at group centres
    ax.set_xticks([gi * STEP_G + (len(MODELS) - 1) / 2
                   for gi in range(len(ENCODINGS))])
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODINGS],
                       rotation=0, ha="center", fontsize=9)

    legend = [(lbl, c) for _m, lbl, c in MODELS]
    d18._finish_padmode(
        fig, ax,
        "Which model do we take forward?",
        PLOTS / "model_compare_29mer_bars.png",
        legend)


def plot_roc(arms, fold_vec, y):
    for enc in ENCODINGS:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        for model, lbl, c in MODELS:
            m = arms[(model, enc)]
            fpr, tpr, _ = roc_curve(y, m["probs"])
            ax.plot(fpr, tpr, color=c, lw=1.6,
                    label=f"{lbl} (AUC={m['auc']:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=0.9)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {ENC_LABEL[enc]}, 29mer (n=46,617)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9.5, loc="lower right", frameon=False)
        fig.tight_layout()
        out = PLOTS / f"roc_{enc}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out}")


def write_tsvs(arms, delong_rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "model_compare_29mer_metrics.tsv", "w") as f:
        f.write("encoding\tmodel\tpooled_auc\touter_mean\touter_std\t"
                "n_features\truntime_s\tjson\n")
        for enc in ENCODINGS:
            for model, _lbl, _c in MODELS:
                m = arms[(model, enc)]
                f.write(f"{enc}\t{model}\t{m['auc']:.6f}\t"
                        f"{np.mean(m['outer']):.6f}\t{m['outer_std']:.6f}\t"
                        f"{m['k']}\t{m['runtime_s']:.1f}\t{m['file']}\n")
    with open(RESULTS / "model_compare_29mer_delong.tsv", "w") as f:
        f.write("encoding\tmodel_a\tmodel_b\tauc_a\tauc_b\tdelta\tse\tz\t"
                "p\tp_adj\tstars\n")
        for r in delong_rows:
            f.write(f"{r['encoding']}\t{r['model_a']}\t{r['model_b']}\t"
                    f"{r['auc_a']:.6f}\t{r['auc_b']:.6f}\t{r['delta']:.6f}\t"
                    f"{r['se']:.6f}\t{r['z']:.3f}\t{r['p']:.6g}\t"
                    f"{r['p_adj']:.6g}\t{r['stars']}\n")
    print(f"  Saved: {RESULTS / 'model_compare_29mer_metrics.tsv'}")
    print(f"  Saved: {RESULTS / 'model_compare_29mer_delong.tsv'}")


def main():
    from config import SPLIT_DATA_PATH
    split = pd.read_csv(SPLIT_DATA_PATH)
    if len(split) != N_ROWS_EXPECTED:
        sys.exit(f"FATAL: split {SPLIT_DATA_PATH} has {len(split)} rows, "
                 f"expected {N_ROWS_EXPECTED} -- wrong split file")
    fold_vec = split["fold"].to_numpy(dtype=int)
    y = split["label"].to_numpy(dtype=int)
    print(f"Split: {SPLIT_DATA_PATH.name}  n={len(y)}  pos={int(y.sum())}  "
          f"folds={len(set(fold_vec))}")

    PLOTS.mkdir(parents=True, exist_ok=True)
    print("Loading arms:")
    arms = load_arms(fold_vec, y)
    print("DeLong (Holm within encoding):")
    delong_rows = delong_table(arms, y)
    print("Plots:")
    plot_bars(arms, delong_rows)
    plot_roc(arms, fold_vec, y)
    write_tsvs(arms, delong_rows)


if __name__ == "__main__":
    main()
