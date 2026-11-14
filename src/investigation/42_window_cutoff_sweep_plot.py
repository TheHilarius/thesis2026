#!/usr/bin/env python3
"""
42_window_cutoff_sweep_plot.py
Window-Hamming cutoff sweep figure (padmode_choice style) for the
flank5_rmdup homology investigation.

Sweep (pepcut1 fixed = pep-ham<=1 ∪ identical-flank ∪ window-c, 19aa):
    window c = 1, 2, 3  -> byte-identical folds -> ONE model run (pc1)
    window c = 4..8     -> own splits/runs; folds repack while structure
                           grows slowly: -58 / -179 / -477 / -847 / -1,436
                           components of 40,144 (largest 138 -> 436 at c=8)
                           — measured, shown, caveated.

Inputs (newest cv_results JSON wins):
    runs/r19_rmdup/pc1/models/flat/cv_results_*.json   (c=1,2,3)
    runs/r19_rmdup/h{4,5,6,7,8}/models/flat/cv_results_*.json
    splits: data/processed/df_all_with_folds_flank5_rmdup_updated_h{c}_pepcut1.csv

Outputs:
    runs/r19_rmdup/plots/sweep_window_cutoff_auc.png / .csv
    (stdout: paired DeLong c1-vs-c4 / c1-vs-c5, Holm over displayed pairs)

Usage (repo root):
    python src/investigation/42_window_cutoff_sweep_plot.py
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
OUT = RUN_ROOT / "plots" / "sweep_window_cutoff_auc"

spec = importlib.util.spec_from_file_location(
    "d18", PROJECT_ROOT / "src" / "investigation" / "18_encodings_sweep_delong.py")
d18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d18)

# c=1,2,3 share one run (folds byte-identical); c=4/c=5 have their own
C_SOURCES = {1: "pc1", 2: "pc1", 3: "pc1",
             4: "h4", 5: "h5", 6: "h6", 7: "h7", 8: "h8"}


def newest_json(run: str) -> Path:
    hits = sorted(glob.glob(str(RUN_ROOT / run / "models" / "flat" /
                                "cv_results_*.json")), key=os.path.getmtime)
    if not hits:
        sys.exit(f"FATAL: no cv_results under {RUN_ROOT / run}")
    return Path(hits[-1])


def mean_std(path: Path) -> tuple[float, float]:
    s = json.load(open(path))["cv_summary"]["auc_roc"]
    return float(s["mean"]), float(s["std"])


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

    xs, vals, errs, runs = [], [], [], {}
    for c in (1, 2, 3, 4, 5, 6, 7, 8):
        j = newest_json(C_SOURCES[c])
        runs[c] = j
        m, sd = mean_std(j)
        xs.append(c - 1)
        vals.append(m)
        errs.append(sd)
        print(f"c={c}: AUC {m:.4f} ± {sd:.4f}  ({C_SOURCES[c]}: {j.name})")

    # paired DeLong on identical rows (same dataset, different folds)
    y1, p1 = oof_aligned(runs[1],
        SPLIT_DIR / "df_all_with_folds_flank5_rmdup_updated_h1_pepcut1.csv")
    pairs = []
    for c in (4, 5, 6, 7, 8):
        yc, pc = oof_aligned(runs[c],
            SPLIT_DIR / f"df_all_with_folds_flank5_rmdup_updated_h{c}_pepcut1.csv")
        if not (y1 == yc).all():
            sys.exit(f"FATAL: c={c} labels differ from c=1")
        _, _, d, se, z, p = d18.delong_full(y1, p1, pc)
        pairs.append((0, c - 1, p))
        print(f"DeLong c=1 vs c={c}: delta={d:+.4f} z={z:+.2f} p={p:.4f}")
    # Holm over the displayed pairs
    p_adj = d18.holm_adjust(np.array([p for _, _, p in pairs]))
    disp = [(a, b, float(pa), float(pb))
            for (a, b, pa), pb in zip(pairs, p_adj)]
    for (a, b, pa), pb in zip(pairs, p_adj):
        print(f"  c={a+1} vs c={b+1}: p={pa:.4f} p_adj={pb:.4f} {d18.stars(pb)}")

    color = "#4477AA"
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    d18._padmode_bars(ax, xs, vals, errs, [color] * 5)
    d18._padmode_brackets(ax, disp, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"window c={c}" for c in range(1, 9)], fontsize=9)
    ax.annotate("c=1–3: fold columns byte-identical -> one model\n"
                "c=4–8: folds repack, structure grows slowly\n"
                "(-58 → -1,436 of 40,144 components; largest 138 → 436)",
                xy=(3.5, 0.42), ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f0f0",
                          edgecolor="grey", lw=0.6))
    footnote = ("flank5_rmdup (19aa dedup, n=45,278); XGB+BLOSUM50 nested CV, 6 folds; "
                "pepcut1 fixed (pep-ham<=1 ∪ identical-flank ∪ window-c); paired DeLong "
                "c1-vs-c4..c8 on identical rows, Holm over 5 displayed pairs; "
                "no significant AUC change up to c=8 (p_adj printed per bracket).")
    d18._finish_padmode(
        fig, ax,
        "Window-Hamming sweep at 19aa: does the full-context cutoff matter?",
        OUT.with_suffix(".png"),
        [("pepcut1 (all c)", color)],
        footnote=footnote)

    with open(OUT.with_suffix(".csv"), "w") as f:
        f.write("window_cutoff,auc_mean,auc_std,run,json\n")
        for c in range(1, 9):
            f.write(f"{c},{vals[c-1]:.6f},{errs[c-1]:.6f},"
                    f"{C_SOURCES[c]},{runs[c].name}\n")
        for (a, b, pa), pb in zip(pairs, p_adj):
            f.write(f"de_long_c{a+1}_vs_c{b+1},,,p={pa:.6f},p_adj={pb:.6f}\n")
    print(f"Saved: {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
