#!/usr/bin/env python3
"""
34_c2st_simple.py
Classifier two-sample test (C2ST) sensitivity check -- ML-standard
"adversarial validation" (Lopez-Paz & Oquab 2017; Rabanser et al. 2019).

Hierarchical, one gate + one family (simple by design):
  GATE: single multiclass omnibus (6-way logistic, balanced accuracy,
        observed minus permutation-null mean). One p-value.
  LOCALIZE (only interpreted if gate rejects): 6x one-vs-rest AUCs,
        each with its own max-T within the family.
Pairwise 15x, GBM sensitivity, BLOSUM-profile repeat, flank repeat are
explicit NON-goals here -- add them only in Phase 2 (see README plan).

Unit/representation (pre-registered, fixed):
  Unique 9-mers (peptide only, labels unused). Position-specific one-hot
  9x20 = 180-d, UNstandardized. L2 logistic C=1.0, class-balanced,
  5-fold CV GROUPED by Hamming-1 cluster (mates never split across CV
  folds -- ungrouped CV leaks fold identity by construction).
  Pooled out-of-fold predictions/scores give the observed statistics.

Null (shared with 33):
  Same stratified cluster shuffle as 33_fold_descriptive
  (within-cluster-size permutation of fold labels, whole clusters move,
  exact per-fold counts preserved). Joint over all 6 buckets; B=200
  default for iteration speed, 2000 for the final run. Every permutation
  re-runs the FULL grouped CV. Tuning is fixed (C=1), so nothing leaks
  outside the loop.

Verdict language (defensible, no hard cutoffs):
  "Inside the re-split null band" = interchangeable. The 0.52/0.55 AUC
  margins from earlier discussion are conventions, not field norms --
  report them as such if used.

Controls (--controls):
  negative: random half-split of one fold's clusters -> AUC ~ 0.5.
  positive: spike P2->L in 5% of fold 0 -> fold-0-vs-rest AUC must lift.

Usage:
    python src/investigation/34_c2st_simple.py
    python src/investigation/34_c2st_simple.py --n-perm 2000
    python src/investigation/34_c2st_simple.py --no-controls --n-perm 500
    python src/investigation/34_c2st_simple.py \
        --split data/processed/df_all_with_folds_h2.csv --tag h2

Outputs:
    results/tables/c2st_multiclass.csv  (obs, null mean/band, p)
    results/tables/c2st_ovr.csv         (per fold: AUC obs, null band, p, maxT-p)
    results/figures/fold_similarity/c2st/ multiclass_null.png, ovr_null.png
"""

import argparse
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
for _p in (PIPELINE_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, SPLIT_DATA_PATH,
    PEPTIDE_COL, FOLD_COL,
    AMINO_ACID_ALPHABET, validate_config,
)

desc33 = importlib.import_module("33_fold_descriptive")
CLUSTER_MAP_DEFAULT = DATA_DIR / "cluster_map_peptide.csv"

AA_ORDER = "".join(AMINO_ACID_ALPHABET)
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}


def one_hot9(peptides):
    n = len(peptides)
    X = np.zeros((n, 9 * 20), dtype=np.float32)
    for i, s in enumerate(peptides):
        for p, a in enumerate(s):
            X[i, p * 20 + AA_INDEX[a]] = 1.0
    return X


def grouped_oof_multiclass(X, y, groups, seed, cv_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.base import clone
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                              random_state=seed)
    base = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                              class_weight="balanced", random_state=seed)
    oof = np.empty_like(y)
    for tr, te in cv.split(X, y, groups):
        clf = clone(base).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return oof


def grouped_oof_ovr_scores(X, ybin, groups, seed, cv_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.base import clone
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                              random_state=seed)
    base = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                              class_weight="balanced", random_state=seed)
    scores = np.empty(len(ybin))
    for tr, te in cv.split(X, ybin, groups):
        clf = clone(base).fit(X[tr], ybin[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return scores


def main():
    ap = argparse.ArgumentParser(description="Simple C2ST gate + OvR.")
    ap.add_argument("--split", default=None)
    ap.add_argument("--cluster-map", default=str(CLUSTER_MAP_DEFAULT))
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="final run: 2000 (default 200 = iterate fast)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--controls", action=argparse.BooleanOptionalAction,
                    default=True, help="run negative + positive controls")
    ap.add_argument("--outdir", default="results/figures/fold_similarity/c2st")
    ap.add_argument("--tables", default="results/tables")
    a = ap.parse_args()

    validate_config()
    split_path = Path(a.split) if a.split else Path(SPLIT_DATA_PATH)
    tag = ("_" + a.tag) if a.tag else ""
    print("=" * 80)
    print("  34_C2ST_SIMPLE -- multiclass gate + 6x OvR (C=1, grouped CV)")
    print("=" * 80)

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    df = pd.read_csv(split_path, usecols=[PEPTIDE_COL, FOLD_COL])
    df = df.drop_duplicates(PEPTIDE_COL).reset_index(drop=True)
    cmap = pd.read_csv(a.cluster_map)
    peptides = df[PEPTIDE_COL].to_numpy()
    folds = df[FOLD_COL].to_numpy()
    buckets = sorted(np.unique(folds))
    print(f"  {len(peptides)} unique peptides, folds {buckets}")

    X = one_hot9(peptides)
    print(f"  X: {X.shape} pos-specific one-hot (unstandardized)")
    cl, pep_clpos = desc33.build_cluster_table(peptides, folds, cmap)
    cl_folds = cl["fold"].to_numpy()
    cl_sizes = cl["size"].to_numpy()
    groups = pep_clpos  # one group per cluster
    y = pd.Index(buckets).get_indexer(folds)  # 0..K-1

    # -- observed --
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    oof_multi = grouped_oof_multiclass(X, y, groups, a.seed, a.cv_folds)
    obs_bal = float(balanced_accuracy_score(y, oof_multi))
    obs_auc = {}
    for f in buckets:
        yb = (folds == f).astype(int)
        s = grouped_oof_ovr_scores(X, yb, groups, a.seed, a.cv_folds)
        obs_auc[f] = float(roc_auc_score(yb, s))
    print(f"  observed: bal_acc={obs_bal:.4f} "
          + " ".join(f"auc({f})={obs_auc[f]:.4f}" for f in buckets))

    # -- null (same stratified permutations as 33) --
    nB = a.n_perm
    null_bal = np.empty(nB)
    null_auc = {f: np.empty(nB) for f in buckets}
    # max-T within OvR family: studentize by per-pair null SD at the end
    for b in range(nB):
        pa = desc33.stratified_perm_assignment(cl_folds, cl_sizes, rng)
        pfolds = pa[pep_clpos]
        py = pd.Index(buckets).get_indexer(pfolds)
        # fresh CV shuffle per permutation, seeded off main rng
        pseed = int(rng.integers(0, 2 ** 31 - 1))
        oof = grouped_oof_multiclass(X, py, groups, pseed, a.cv_folds)
        null_bal[b] = balanced_accuracy_score(py, oof)
        for f in buckets:
            yb = (pfolds == f).astype(int)
            s = grouped_oof_ovr_scores(X, yb, groups, pseed, a.cv_folds)
            null_auc[f][b] = roc_auc_score(yb, s)
        if (b + 1) % 20 == 0:
            print(f"  ... {b + 1}/{nB} perms ({time.time() - t0:.0f}s)")
    print(f"  null done in {time.time() - t0:.1f}s")

    p_bal = float((1 + (null_bal >= obs_bal).sum()) / (1 + nB))
    tables = Path(a.tables)
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "bal_acc_obs": obs_bal,
        "bal_acc_null_mean": float(null_bal.mean()),
        "bal_acc_null_lo": float(np.percentile(null_bal, 2.5)),
        "bal_acc_null_hi": float(np.percentile(null_bal, 97.5)),
        "p_gate": p_bal, "n_perm": nB, "seed": a.seed,
    }]).to_csv(tables / f"c2st_multiclass{tag}.csv", index=False)

    # max-T over the 6 OvR (studentized by own null SD)
    sds = {f: float(null_auc[f].std()) or 1.0 for f in buckets}
    t_obs = {f: (obs_auc[f] - null_auc[f].mean()) / sds[f] for f in buckets}
    t_null_max = np.max(np.array([
        (null_auc[f] - null_auc[f].mean()) / sds[f] for f in buckets]),
        axis=0)
    rows = []
    for f in buckets:
        v = null_auc[f]
        p_raw = float((1 + (v >= obs_auc[f]).sum()) / (1 + nB))
        p_maxt = float((1 + (t_null_max >= t_obs[f]).sum()) / (1 + nB))
        rows.append({"fold": f, "auc_obs": obs_auc[f],
                     "auc_null_mean": float(v.mean()),
                     "auc_null_lo": float(np.percentile(v, 2.5)),
                     "auc_null_hi": float(np.percentile(v, 97.5)),
                     "p_raw": p_raw, "p_maxT": p_maxt})
    pd.DataFrame(rows).to_csv(tables / f"c2st_ovr{tag}.csv", index=False)
    print(f"  gate p={p_bal:.4f}; wrote tables/c2st_*_{tag}.csv")

    # -- controls (cheap, no null) --
    if a.controls:
        # negative: split largest fold's clusters in halves
        big = max(buckets, key=lambda f: (folds == f).sum())
        idx = np.where(folds == big)[0]
        ucl = np.unique(pep_clpos[idx])
        rng2 = np.random.default_rng(a.seed + 1)
        rng2.shuffle(ucl)
        half = set(ucl[:len(ucl) // 2])
        neg = np.isin(pep_clpos[idx], list(half)).astype(int)
        s = grouped_oof_ovr_scores(X[idx], neg, pep_clpos[idx],
                                   a.seed + 1, a.cv_folds)
        print(f"  control-negative ({big} halves): "
              f"AUC={roc_auc_score(neg, s):.4f} (expect ~0.5)")
        # positive: spike P2->L in 5% of fold 0
        f0 = buckets[0]
        i0 = np.where(folds == f0)[0]
        pick = rng2.choice(i0, size=max(1, int(0.05 * len(i0))),
                           replace=False)
        Xsp = X.copy()
        for i in pick:  # P2 = position index 1 -> cols 20..39
            Xsp[i, 20:40] = 0.0
            Xsp[i, 20 + "ACDEFGHIKLMNPQRSTVWY".index("L")] = 1.0
        yb = (folds == f0).astype(int)
        s = grouped_oof_ovr_scores(Xsp, yb, groups, a.seed + 2, a.cv_folds)
        print(f"  control-positive (P2->L 5% in fold {f0}): "
              f"AUC={roc_auc_score(yb, s):.4f} (must lift)")

    # -- figures --
    outdir = Path(a.outdir if not a.tag else f"{a.outdir}{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_bal, bins=30, alpha=0.7, label="null")
    ax.axvline(obs_bal, color="red", label=f"obs {obs_bal:.3f} (p={p_bal:.3f})")
    ax.set_xlabel("multiclass balanced accuracy (OOF)")
    ax.set_title("C2ST gate: omnibus vs cluster null")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"multiclass_null{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(buckets))
    o = [obs_auc[f] for f in buckets]
    lo = [float(np.percentile(null_auc[f], 2.5)) for f in buckets]
    hi = [float(np.percentile(null_auc[f], 97.5)) for f in buckets]
    ax.errorbar(x, o, yerr=[np.array(o) - np.array(lo),
                            np.array(hi) - np.array(o)],
                fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}" for f in buckets])
    ax.set_ylabel("OvR AUC (OOF)")
    ax.set_title("C2ST OvR per fold vs cluster null")
    fig.tight_layout()
    fig.savefig(outdir / f"ovr_null{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {outdir}/multiclass_null{tag}.png, ovr_null{tag}.png")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
