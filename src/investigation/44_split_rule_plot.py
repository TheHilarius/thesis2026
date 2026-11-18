#!/usr/bin/env python3
"""
44_split_rule_plot.py
Split-rule comparison figure: canonical pep-ham<=1 split vs the CHOSEN
robust rule (pepcut1 fixed = pep-ham<=1 ∪ identical-flank ∪ window-ham<=
WINDOW_CUTOFF, i.e. c=6), XGB+BLOSUM50 nested CV on flank5_rmdup.

Regenerates runs/r19_rmdup/plots/split_rule_auc.{png,csv} (the earlier
version compared against the c=1 arm; the chosen rule is c=6).

Usage (repo root):
    python src/investigation/44_split_rule_plot.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "runs" / "r19_rmdup"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"

spec = importlib.util.spec_from_file_location(
    "d18", PROJECT_ROOT / "src" / "investigation" / "18_encodings_sweep_delong.py")
d18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d18)

ARMS = [
    ("canonical pep-ham<=1 only", "canonical",
     "df_all_with_folds_flank5_rmdup.csv", "#8c8c8c"),
    ("robust pepcut1 (window c=6)", "h6",
     "df_all_with_folds_flank5_rmdup_updated_h6_pepcut1.csv", "#4477AA"),
]


def newest_json(run: str) -> Path:
    hits = sorted(glob.glob(str(RUN_ROOT / run / "models" / "flat" /
                                "cv_results_*.json")), key=os.path.getmtime)
    if not hits:
        sys.exit(f"FATAL: no cv_results under {RUN_ROOT / run}")
    return Path(hits[-1])


def oof_aligned(json_path: Path, split_path: Path):
    fold = pd.read_csv(split_path, usecols=["fold"])["fold"].values
    lab = pd.read_csv(split_path, usecols=["label"])["label"].values
    ev = json.load(open(json_path))["outer_val_predictions"]
    y = np.empty(len(fold), dtype=int)
    p = np.empty(len(fold))
    for fk, e in ev.items():
        pos = np.where(fold == int(fk))[0]
        yt = np.asarray(e["y_true"])
        if len(pos) != len(yt) or not (yt == lab[pos]).all():
            sys.exit(f"FATAL: fold {fk} in {json_path.name} not in split order")
        y[pos] = yt
        p[pos] = np.asarray(e["y_prob"])
    return y, p


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, vals, errs, runs, splits = [], [], [], {}, {}
    for x, (label, run, split_name, _) in enumerate(ARMS):
        j = newest_json(run)
        runs[x] = j
        splits[x] = SPLIT_DIR / split_name
        s = json.load(open(j))["cv_summary"]["auc_roc"]
        m, sd = float(s["mean"]), float(s["std"])
        xs.append(x)
        vals.append(m)
        errs.append(sd)
        print(f"{label}: AUC {m:.4f} ± {sd:.4f}  ({j.name})")

    y0, p0 = oof_aligned(runs[0], splits[0])
    y1, p1 = oof_aligned(runs[1], splits[1])
    if not (y0 == y1).all():
        sys.exit("FATAL: labels differ between arms (row order?)")
    _, _, delta, _, z, p = d18.delong_full(y0, p0, p1)
    print(f"DeLong canonical vs c=6: delta={delta:+.4f} z={z:+.2f} p={p:.4f}")

    color0, color1 = ARMS[0][3], ARMS[1][3]
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    d18._padmode_bars(ax, xs, vals, errs, [color0, color1])
    p_adj = d18.holm_adjust(np.array([p]))[0]
    print(f"  Holm (1 pair): p_adj={p_adj:.4f} {d18.stars(p_adj)}")
    d18._padmode_brackets(
        ax, [(0, 1, p, p_adj)], vals)
    ax.set_xticks(xs)
    ax.set_xticklabels(["canonical\npep-ham≤1 only", "robust (pepcut1)\n"
                        "pep≤1 ∪ flank-id ∪ win-6"], fontsize=9.5)
    d18._finish_padmode(
        fig, ax,
        "Does the chosen robust homology split change XGB+BLOSUM50 performance?",
        RUN_ROOT / "plots" / "split_rule_auc.png",
        [("canonical (no flank rule)", color0),
         ("robust pepcut1 (window c=6)", color1)],
        footnote=("flank5_rmdup (19aa dedup, n=45,278); XGB+BLOSUM50 nested CV, 6 folds; "
                  "paired DeLong on identical rows, Holm over displayed comparisons. "
                  f"Δ={delta:+.4f} (z={z:+.2f})."))

    out_csv = RUN_ROOT / "plots" / "split_rule_auc.csv"
    with open(out_csv, "w") as f:
        f.write("rule,auc_mean,auc_std,split_file,json\n")
        for x, (label, run, split_name, _) in enumerate(ARMS):
            f.write(f'"{label}",{vals[x]:.6f},{errs[x]:.6f},{split_name},'
                    f"{runs[x].name}\n")
        f.write(f"de_long_canonical_vs_c6,,,{delta:.6f},{p:.6f}\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
