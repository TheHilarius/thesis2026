#!/usr/bin/env python3
"""48_xgb_optuna_delong.py
Before/after significance + figure for the XGB Optuna tuning campaign.

Four canon runs (flank5_rmdup, h6+pepcut1, n=45,278), all official nested CV:
  blosum50 defaults   runs/r19_rmdup/h6/.../cv_results_xgb_blosum50_flat_20261114_004521.json
  blosum50 tuned      runs/xgb_optuna_validation/..._xgblr0p02n5000_20261122_005810.json
  hc_blosum50 defaults runs/xgb_hc_baseline/..._20261126_102808.json
  hc_blosum50 tuned   runs/xgb_optuna_validation_hc/..._xgblr0p02n5000_20261126_104038.json

Statistical protocol (verbatim from 18_encodings_sweep_delong /
15_padmode_significance, imported — not copied):
  * bar heights = pooled OOF AUC (roc_auc_score on concatenated outer-fold
    predictions) — the same vector DeLong tests
  * error bars  = std of the 6 outer-fold AUCs
  * paired DeLong two-sided z on pooled OOF, Holm over ALL 3 comparisons
    (one family), brackets drawn only for Holm-significant pairs

Emits (runs/xgb_tuning_final/):
  plots/tuning_delong_comparison.png   4-bar padmode-style figure
  results/tuning_delong_pairs.tsv      the 3 DeLong tests
  results/best_params_table.tsv        defaults vs tuned params, both arms

Usage:
  python src/investigation/48_xgb_optuna_delong.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
for p in (str(SRC_DIR), str(PIPELINE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import DEFAULT_SPLIT_PATH, FOLD_COL, LABEL_COL, PROJECT_ROOT  # noqa: E402

# ── proven stats: reuse 18_encodings_sweep_delong (reconstruct/delong/holm) ──
_spec = importlib.util.spec_from_file_location(
    "d18", SRC_DIR / "18_encodings_sweep_delong.py")
d18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d18)

# ── house figure style: reuse 19_handcrafted_sweep_delong padmode helpers ──
_spec19 = importlib.util.spec_from_file_location(
    "d19", SRC_DIR / "19_handcrafted_sweep_delong.py")
d19 = importlib.util.module_from_spec(_spec19)
_spec19.loader.exec_module(d19)

RUNS = {
    "blosum_def": PROJECT_ROOT / "runs" / "r19_rmdup" / "h6" / "models" / "flat"
                  / "cv_results_xgb_blosum50_flat_20261114_004521.json",
    "blosum_tuned": PROJECT_ROOT / "runs" / "xgb_optuna_validation"
                    / "models" / "flat"
                    / "cv_results_xgb_blosum50_flat_xgblr0p02n5000_20261122_005810.json",
    "hc_def": PROJECT_ROOT / "runs" / "xgb_hc_baseline" / "models" / "flat"
              / "cv_results_xgb_handcrafted_blosum50_flat_20261126_102808.json",
    "hc_tuned": PROJECT_ROOT / "runs" / "xgb_optuna_validation_hc"
                / "models" / "flat"
                / "cv_results_xgb_handcrafted_blosum50_flat_xgblr0p02n5000_20261126_104038.json",
}
BEST_PARAMS = {
    "blosum": PROJECT_ROOT / "runs" / "xgb_optuna" / "best_params.json",
    "hc": PROJECT_ROOT / "runs" / "xgb_optuna_handcrafted_blosum50"
          / "best_params.json",
}
# Holm family: tuned vs defaults per arm + cross-arm tuned pair
# delta = auc_a - auc_b, so a is always the "first-mentioned" arm
PAIRS = [
    ("blosum50: tuned vs defaults", "blosum_tuned", "blosum_def"),
    ("hc_blosum50: tuned vs defaults", "hc_tuned", "hc_def"),
    ("tuned: hc vs blosum50", "hc_tuned", "blosum_tuned"),
]
# param display order: the nine article parameters + n_estimators
PARAM_ROWS = ["max_depth", "min_child_weight", "gamma", "subsample",
              "colsample_bytree", "reg_alpha", "reg_lambda",
              "learning_rate", "n_estimators", "scale_pos_weight"]


def load_all():
    df_split = pd.read_csv(DEFAULT_SPLIT_PATH)
    fold_vec = df_split[FOLD_COL].to_numpy()
    y = df_split[LABEL_COL].to_numpy(dtype=int)
    out = {}
    for tag, path in RUNS.items():
        if not path.exists():
            sys.exit(f"FATAL: missing {path}")
        res = json.loads(path.read_text())
        if "outer_val_predictions" not in res:
            sys.exit(f"FATAL: {path.name} has no outer_val_predictions")
        probs = d18.reconstruct(res, fold_vec, y, tag)
        outer = [float(m["auc_roc"]) for m in res["outer_val_metrics"]]
        assert len(outer) == 6, f"{tag}: expected 6 outer folds"
        pooled = float(roc_auc_score(y, probs))
        reported = float(res["cv_summary"]["auc_roc"]["mean"])
        assert abs(pooled - reported) < 0.002, (
            f"{tag}: pooled OOF {pooled:.4f} vs cv_summary {reported:.4f}")
        out[tag] = {"probs": probs, "pooled": pooled,
                    "outer_std": float(np.std(outer, ddof=1)),
                    "file": path.name, "reported": reported}
        print(f"  [OK] {tag:<13} pooled AUC={pooled:.4f} "
              f"(fold-std {out[tag]['outer_std']:.4f})  {path.name}")
    return out, y


def run_delong(data, y, out_dir: Path):
    rows = []
    for name, a, b in PAIRS:
        auc_a, auc_b, delta, se, z, p = d18.delong_full(
            y, data[a]["probs"], data[b]["probs"])
        rows.append({"pair": name, "arm_a": a, "arm_b": b,
                     "auc_a": auc_a, "auc_b": auc_b, "delta": delta,
                     "se": se, "z": z, "p": p})
    p_holm = d18.holm_adjust([r["p"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph
        r["star"] = d18.stars(ph)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cols = ["pair", "arm_a", "arm_b", "auc_a", "auc_b", "delta", "se",
            "z", "p", "p_holm", "star"]
    with open(results_dir / "tuning_delong_pairs.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(
                str(r[c]) if isinstance(r[c], str) else f"{r[c]:.6g}"
                for c in cols) + "\n")
    for r in rows:
        print(f"  {r['pair']:<34} delta={r['delta']:+.4f}  "
              f"p={r['p']:.3e}  Holm p={r['p_holm']:.3e}  {r['star']}")
    return rows


def fig_bars(data, rows, out_dir: Path):
    """4-bar before/after figure, house style: 19_handcrafted_sweep_delong
    padmode helpers verbatim (0-1 axis, 'CV-AUC', value labels, stars-only
    brackets under 0.97, dpi 200). BLOSUM50 pair: defaults Tol indigo #332288
    (h6 dataset color, same run as the split-rule figure's right bar), tuned
    Tol wine #882255 (bordeaux). Handcrafted pair: defaults Tol sand, tuned
    darker sand. All four bars distinct; all colors Tol (colour-blind safe)."""
    order = ["blosum_def", "blosum_tuned", "hc_def", "hc_tuned"]
    colors = ["#332288",                       # Tol indigo — h6 dataset (same run as split fig's right bar)
              "#882255",                       # Tol wine (bordeaux) — blosum50 Optuna tuned
              d19.C_EXP,                       # sand — handcrafted defaults (explicit-family)
              d19._tint(d19.C_EXP, mix=-0.5)]  # darker sand — hc Optuna tuned (original tone)
    heights = [data[k]["pooled"] for k in order]
    yerrs = [data[k]["outer_std"] for k in order]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    xs = [0, 1, 2, 3]   # contiguous bars; groups follow the reference
                        # step_g=2 layout, separated by an axvline
    d19._padmode_bars(ax, xs, heights, yerrs, colors)
    ax.axvline(1.5, color="grey", linestyle="--", linewidth=0.8)
    # two-level horizontal x labels: condition per bar, feature set below
    ax.set_xticks(xs)
    ax.set_xticklabels(["defaults", "Optuna tuned", "defaults", "Optuna tuned"],
                       rotation=0, fontsize=9)
    trans = ax.get_xaxis_transform()          # data x, axes-fraction y
    ax.text(0.5, -0.08, "BLOSUM50", ha="center", va="top",
            fontsize=10, fontweight="bold", transform=trans)
    ax.text(2.5, -0.08, "handcrafted+BLOSUM50", ha="center", va="top",
            fontsize=10, fontweight="bold", transform=trans)

    # brackets: Holm-significant pairs only, stars-only, widest span on top
    idx = dict(zip(order, xs))
    pairs = [(idx[r["arm_a"]], idx[r["arm_b"]], r["p"], r["p_holm"])
             for r in rows if r["p_holm"] < 0.05]
    d19._padmode_brackets(ax, pairs, heights, stars_only=True)

    # caption lives OUTSIDE the figure: printed + caption.txt (thesis-ready)
    caption = (
        "flank5_rmdup (19aa dedup, n=45,278); h6+pepcut1 split; XGB nested "
        "CV, 6 outer buckets. Bars = pooled OOF AUC, error = SD of the 6 "
        "outer-fold AUCs; paired DeLong on pooled OOF, Holm over 3 "
        "comparisons; official nested-CV runs only (tuner-objective scores "
        "excluded).")
    for r in rows:
        if r["p_holm"] >= 0.05:
            caption += (f" NS: {r['pair']} (delta={r['delta']:+.4f}, "
                        f"Holm p={r['p_holm']:.3f}).")
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "caption.txt").write_text(caption + "\n")
    print(f"  Caption -> {plots / 'caption.txt'}")
    print(f"  CAPTION: {caption}")
    d19._finish_padmode(
        fig, ax,
        "Does hyperparameter tuning improve XGBoost?",
        plots / "tuning_delong_comparison.png",
        legend_entries=[])     # fully self-labeled: no legend needed
    print(f"  Figure -> {plots / 'tuning_delong_comparison.png'}")


def params_table(out_dir: Path, data):
    from config import get_model_config
    defaults = dict(get_model_config("xgb")["params"])
    defaults["scale_pos_weight"] = "auto"
    tuned = {}
    for arm, path in BEST_PARAMS.items():
        if not path.exists():
            sys.exit(f"FATAL: missing {path}")
        tuned[arm] = json.loads(path.read_text())["params"]
    tuned["blosum"]["scale_pos_weight"] = "auto"
    tuned["hc"]["scale_pos_weight"] = "auto"

    def fmt(v):
        if isinstance(v, float) and v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    header = ["param", "blosum50_defaults", "blosum50_tuned",
              "hc_blosum50_defaults", "hc_blosum50_tuned"]
    lines = [header]
    for p in PARAM_ROWS:
        lines.append([p, fmt(defaults.get(p, "-")),
                      fmt(tuned["blosum"].get(p, "-")),
                      fmt(defaults.get(p, "-")),
                      fmt(tuned["hc"].get(p, "-"))])
    lines.append(["auc_roc (official)",
                  f"{data['blosum_def']['reported']:.4f}",
                  f"{data['blosum_tuned']['reported']:.4f}",
                  f"{data['hc_def']['reported']:.4f}",
                  f"{data['hc_tuned']['reported']:.4f}"])
    lines.append(["cv_results source", data["blosum_def"]["file"],
                  data["blosum_tuned"]["file"], data["hc_def"]["file"],
                  data["hc_tuned"]["file"]])

    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "best_params_table.tsv", "w") as fh:
        for row in lines:
            fh.write("\t".join(row) + "\n")

    widths = [max(len(r[i]) for r in lines[:-1]) for i in range(5)]
    print("\n  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  " + "-" * (sum(widths) + 8))
    for row in lines[1:-1]:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print(f"\n  Table -> {results_dir / 'best_params_table.tsv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", default=None,
                    help="output root (default runs/xgb_tuning_final)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else \
        PROJECT_ROOT / "runs" / "xgb_tuning_final"

    print("Loading runs + reconstructing pooled OOF ...")
    data, y = load_all()
    print("\nDeLong (paired on pooled OOF, Holm over 3):")
    rows = run_delong(data, y, out_dir)
    print("\nFigure:")
    fig_bars(data, rows, out_dir)
    print("\nBest-parameters table:")
    params_table(out_dir, data)
    print("\nDONE")


if __name__ == "__main__":
    main()
