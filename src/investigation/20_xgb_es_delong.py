#!/usr/bin/env python3
"""
20_xgb_es_delong.py
DeLong comparison: XGB early-stopping sweeps vs the fixed-budget baseline.

Arms (all blosum50, same folds/seed):
  baseline  — lr0.1 n1000 no ES           (runs/encodings_sweep, AUC 0.7633)
  es_auc_*  — lr x n5000 es50 eval=auc    (runs/xgb_early_stop_auc)
  es_ll_*   — lr x n5000 es50 eval=logloss(runs/xgb_early_stop_logloss)

Protocol (same as 18_encodings_sweep_delong.py): pooled out-of-fold
probability vector per run (every row predicted exactly once by its outer
bucket), DeLong two-sided z-test of baseline vs each arm, Holm correction
across the full family.

Output:
  runs/xgb_early_stop_auc/results/delong_vs_baseline.tsv
  stdout table (delta, z, p, p_adj, stars)

Usage:
  python src/investigation/20_xgb_es_delong.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH

ALPHA = 0.05

BASELINE = (PROJECT_ROOT / "runs" / "encodings_sweep" / "models" / "flat" /
            "cv_results_xgb_blosum50_flat_20261014_154653.json")
RUNS = {
    "es_auc": PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "models" / "flat",
    "es_ll": PROJECT_ROOT / "runs" / "xgb_early_stop_logloss" / "models" / "flat",
    "fixed_low": PROJECT_ROOT / "runs" / "xgb_fixed_lowlr" / "models" / "flat",
}
OUT_TSV = (PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "results" /
           "delong_vs_baseline.tsv")

# ── stats (inlined from 15_padmode_significance — proven code) ──────────
def delong_structure(y, p):
    pos = p[y == 1]
    neg = p[y == 0]
    m, n = len(pos), len(neg)
    neg_sorted = np.sort(neg)
    left = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (left + 0.5 * (right - left)) / n
    pos_sorted = np.sort(pos)
    leftp = np.searchsorted(pos_sorted, neg, side="left")
    rightp = np.searchsorted(pos_sorted, neg, side="right")
    v01 = 1.0 - (leftp + 0.5 * (rightp - leftp)) / m
    return v10, v01


def delong_test(y, p_a, p_b):
    v10a, v01a = delong_structure(y, p_a)
    v10b, v01b = delong_structure(y, p_b)
    auc_a, auc_b = float(v10a.mean()), float(v10b.mean())
    m, n = len(v10a), len(v01a)
    s10a = np.var(v10a, ddof=1) / m
    s10b = np.var(v10b, ddof=1) / m
    s01a = np.var(v01a, ddof=1) / n
    s01b = np.var(v01b, ddof=1) / n
    cov10 = np.cov(v10a, v10b, ddof=1)[0, 1] / m
    cov01 = np.cov(v01a, v01b, ddof=1)[0, 1] / n
    var_d = s10a + s01a + s10b + s01b - 2.0 * (cov10 + cov01)
    return auc_a, auc_b, float(var_d)


def delong_full(y, pa, pb):
    """Two-sided DeLong test; returns auc_a, auc_b, delta, SE(delta), z, p."""
    auc_a, auc_b, var_d = delong_test(y, pa, pb)
    d = auc_a - auc_b
    if var_d <= 0:
        return auc_a, auc_b, d, 0.0, 0.0, 1.0
    se = float(np.sqrt(var_d))
    z = d / se
    p = float(2 * norm.sf(abs(z)))
    return auc_a, auc_b, d, se, z, p


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full(len(pvals), np.nan)
    valid = ~np.isnan(pvals)
    pv = pvals[valid]
    m = len(pv)
    order = np.argsort(pv)
    running = 0.0
    out = np.empty(m)
    for rank, i in enumerate(order):
        val = (m - rank) * pv[i]
        running = max(running, val)
        out[i] = min(1.0, running)
    adj[valid] = out
    return adj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


# ── data ────────────────────────────────────────────────────────────────
def reconstruct(res, fold_vec, y, tag):
    """Full-length OOF probability vector; validates against the fold CSV."""
    n = len(fold_vec)
    probs = np.full(n, np.nan)
    for fk, entry in res["outer_val_predictions"].items():
        mask = fold_vec == int(fk)
        yt = np.asarray(entry["y_true"], dtype=int)
        if mask.sum() != len(entry["y_prob"]):
            raise ValueError(f"{tag}: fold {fk} size mismatch")
        if not (yt == y[mask]).all():
            raise ValueError(f"{tag}: fold {fk} labels mismatch df")
        probs[mask] = np.asarray(entry["y_prob"], dtype=float)
    if np.isnan(probs).any():
        raise ValueError(f"{tag}: rows never predicted")
    return probs


def load_arm(path, fold_vec, y, group):
    with open(path) as fh:
        res = json.load(fh)
    tag = os.path.basename(path)
    probs = reconstruct(res, fold_vec, y, tag)
    m_lr = re.search(r"xgblr([\dp]+)n", tag)
    lr = float(m_lr.group(1).replace("p", ".")) if m_lr else None
    return {
        "key": tag,
        "group": group,
        "lr": lr,
        "probs": probs,
        "auc": float(roc_auc_score(y, probs)),
    }


def main():
    df = pd.read_csv(SPLIT_DATA_PATH, usecols=["label", "fold"])
    y = df["label"].to_numpy(dtype=int)
    fold_vec = df["fold"].to_numpy(dtype=int)
    print(f"  {SPLIT_DATA_PATH.name}: {len(df)} rows, "
          f"pos={int(y.sum())}, neg={int((y == 0).sum())}")

    base = load_arm(BASELINE, fold_vec, y, "baseline")
    print(f"  baseline: {base['key']}  pooled AUC={base['auc']:.6f}")

    arms = []
    for group, d in RUNS.items():
        for p in sorted(Path(d).glob("cv_results_*xgblr*.json")):
            arms.append(load_arm(p, fold_vec, y, group))
    print(f"  {len(arms)} ES arms loaded\n")

    rows = []
    for a in sorted(arms, key=lambda r: (r["lr"] is None, r["lr"])):
        # delta = arm - baseline (positive = arm better)
        auc_a, auc_b, delta, se, z, p = delong_full(
            y, a["probs"], base["probs"])
        rows.append({
            "arm": f"{a['group']}_lr{a['lr']}",
            "group": a["group"],
            "lr": a["lr"],
            "auc_baseline": auc_b,
            "auc_arm": auc_a,
            "delta": delta,
            "se": se,
            "z": z,
            "p": p,
            "sig_before_adj": stars(p),
        })

    adj = holm_adjust([r["p"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_adj"] = float(pa)
        r["sig"] = stars(pa)

    out = pd.DataFrame(rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TSV, sep="\t", index=False, float_format="%.6g")
    print(f"  saved: {OUT_TSV}\n")

    print(f"  {'arm':<18} {'AUC_arm':>8} {'delta':>9} {'z':>7} "
          f"{'p':>10} {'p_adj':>10} {'sig':>4}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['arm']:<18} {r['auc_arm']:>8.4f} {r['delta']:>+9.4f} "
              f"{r['z']:>7.2f} {fmt_p(r['p']):>10} "
              f"{fmt_p(r['p_adj']):>10} {r['sig']:>4}")

    n_sig = sum(1 for r in rows if r["p_adj"] < ALPHA)
    wins = sum(1 for r in rows if r["delta"] > 0)
    print(f"\n  family: {len(rows)} comparisons, Holm-corrected: "
          f"{n_sig} significant at alpha={ALPHA}; arms above baseline: {wins}")


if __name__ == "__main__":
    main()
