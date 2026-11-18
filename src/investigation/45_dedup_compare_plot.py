#!/usr/bin/env python3
"""
45_dedup_compare_plot.py
Dedup A/B figure: flank5 WITH duplicate windows vs flank5_rmdup (deduped),
BOTH on the same canonical pep-ham<=1 split rule -- only the dedup differs.

The right bar (flank5_rmdup, canonical split, n=45,278) is the SAME run as
the canonical bar in 44_split_rule_plot.py (shared JSON, enforced here), so
the two figures chain: dedup first, homology rule second.

Inputs:
    runs/flank5_compare/models/flat/cv_results_xgb_blosum50_flat_20261107_215706.json
        (flank5, canonical split df_all_with_folds_flank5.csv, n=46,617)
    runs/r19_rmdup/canonical/models/flat/cv_results_xgb_blosum50_flat_20261113_224901.json
        (flank5_rmdup, canonical split df_all_with_folds_flank5_rmdup.csv, n=45,278)

Outputs:
    runs/r19_rmdup/plots/dedup_flank5_vs_rmdup_auc.png / .csv

Usage (repo root):
    python src/investigation/45_dedup_compare_plot.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"

spec = importlib.util.spec_from_file_location(
    "d18", PROJECT_ROOT / "src" / "investigation" / "18_encodings_sweep_delong.py")
d18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d18)

ARMS = [
    ("flank5 (canonical split)", "runs/flank5_compare",
     "df_all_with_folds_flank5.csv", "#8c8c8c"),
    ("flank5_rmdup (canonical split)", "runs/r19_rmdup/canonical",
     "df_all_with_folds_flank5_rmdup.csv", "#4477AA"),
]
RUN_ROOT = PROJECT_ROOT / "runs" / "r19_rmdup"


def newest_json(run_dir: str) -> Path:
    hits = sorted(glob.glob(str(PROJECT_ROOT / run_dir / "models" / "flat" /
                                "cv_results_xgb_blosum50_flat_*.json")),
                  key=os.path.getmtime)
    if not hits:
        sys.exit(f"FATAL: no cv_results under {PROJECT_ROOT / run_dir}")
    return Path(hits[-1])


def row_predictions(json_path: Path, split_path: Path):
    """Ordered list of (context, label, prob) for a run's OOF predictions.

    The JSON stores per-fold prediction arrays in split row order; the split
    CSV rows carry full_context in the same order, so assign sequentially
    within each fold."""
    sp = pd.read_csv(split_path, usecols=["full_context", "fold", "label"])
    ev = json.load(open(json_path))["outer_val_predictions"]
    out = []
    for fk, e in ev.items():
        pos = np.where(sp["fold"].values == int(fk))[0]
        y_true = np.asarray(e["y_true"])
        if len(pos) != len(y_true) or not (y_true == sp["label"].values[pos]).all():
            sys.exit(f"FATAL: fold {fk} order mismatch in {json_path.name}")
        for p_idx, prob in zip(pos, np.asarray(e["y_prob"])):
            out.append((sp["full_context"].iloc[p_idx],
                        int(sp["label"].iloc[p_idx]), float(prob)))
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds, vals, errs, runs = [], [], [], {}
    for x, (label, run_dir, split_name, _) in enumerate(ARMS):
        j = newest_json(run_dir)
        runs[x] = j
        s = json.load(open(j))["cv_summary"]["auc_roc"]
        vals.append(float(s["mean"]))
        errs.append(float(s["std"]))
        preds.append(row_predictions(j, SPLIT_DIR / split_name))
        print(f"{label}: AUC {vals[x]:.4f} ± {errs[x]:.4f}  ({j.name})")

    # shared-bar guarantee with the split_rule figure: arm 1 JSON must be the
    # canonical rmdup run that 44_ uses
    canon44 = newest_json("runs/r19_rmdup/canonical")
    if runs[1] != canon44:
        sys.exit(f"FATAL: arm-1 JSON {runs[1].name} != split_rule canonical "
                 f"{canon44.name} -- figures would not share the bar")

    # Pair each rmdup row (one per context) with its KEPT flank5 row:
    # the dedup rule keeps the first positive row per window, else the
    # first row -- i.e. the first flank5 row with the same (context, label).
    # rmdup preserves flank5 row order, so a per-(context,label) counter
    # over flank5 rows pairs the kept rows exactly.
    seen = defaultdict(int)
    idx_by = defaultdict(list)
    for i, (ctx, lab, _p) in enumerate(preds[0]):
        idx_by[(ctx, lab)].append(i)
    pairs = []
    for ctx, lab, prob1 in preds[1]:
        k = seen[(ctx, lab)]
        seen[(ctx, lab)] += 1
        cand = idx_by[(ctx, lab)]
        if k >= len(cand):
            sys.exit(f"FATAL: rmdup row ({ctx},{lab}) has no flank5 counterpart")
        i0 = cand[k]
        pairs.append((i0, prob1))
    if len(pairs) != len(preds[1]):
        sys.exit("FATAL: pairing count mismatch")
    y0 = np.array([preds[0][i][1] for i, _ in pairs])
    p0 = np.array([preds[0][i][2] for i, _ in pairs])
    p1 = np.array([prob1 for _, prob1 in pairs])
    y1 = y0.copy()
    _, _, delta, _, z, p = d18.delong_full(y0, p0, p1)
    p_adj = d18.holm_adjust(np.array([p]))[0]
    print(f"paired rows: {len(pairs)}  DeLong dedup effect: "
          f"delta={delta:+.4f} z={z:+.2f} p={p:.4f} p_adj={p_adj:.4f}")

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    d18._padmode_bars(ax, [0, 1], vals, errs,
                      [ARMS[0][3], ARMS[1][3]])
    d18._padmode_brackets(ax, [(0, 1, p, p_adj)], vals)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["flank5\n(with duplicate windows)\nn=46,617",
                        "flank5_rmdup\n(deduplicated)\nn=45,278"], fontsize=9.5)
    d18._finish_padmode(
        fig, ax,
        "Does removing duplicate context windows change performance?",
        RUN_ROOT / "plots" / "dedup_flank5_vs_rmdup_auc.png",
        [("flank5 (with dups)", ARMS[0][3]),
         ("flank5_rmdup (canonical split)", ARMS[1][3])],
        footnote=("XGB+BLOSUM50 nested CV, 6 folds; BOTH arms use the canonical "
                  "pep-ham<=1 split rule -- only the dedup differs. Paired DeLong "
                  f"on {len(pairs)} paired rows, Holm over displayed comparisons. "
                  f"delta={delta:+.4f} (z={z:+.2f}). The right bar is the same run "
                  "as the canonical bar in the split-rule figure."))

    out_csv = RUN_ROOT / "plots" / "dedup_flank5_vs_rmdup_auc.csv"
    with open(out_csv, "w") as f:
        f.write("dataset,auc_full,auc_common_subset,n_rows,json\n")
        def auc_mw(y, p):  # Mann-Whitney AUC, no sklearn dependency
            r = pd.Series(p).rank(method="average").values
            n1, n0 = int(sum(y == 1)), int(sum(y == 0))
            return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        f.write(f'"{ARMS[0][0]}",{vals[0]:.6f},{auc_mw(y0, p0):.6f},'
                f'{len(preds[0])},{runs[0].name}\n')
        f.write(f'"{ARMS[1][0]}",{vals[1]:.6f},{auc_mw(y1, p1):.6f},'
                f'{len(preds[1])},{runs[1].name}\n')
        f.write(f"de_long_common,,,{delta:.6f},p={p:.6f},p_adj={p_adj:.6f},"
                f"z={z:.4f},n_paired={len(pairs)}\n")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
