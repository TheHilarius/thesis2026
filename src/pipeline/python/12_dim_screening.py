#!/usr/bin/env python3
"""
12_dim_screening.py
Per-dim ("50-rows") supervised screening for window embeddings — the
supervisor's per-row small-predictor idea, replacing the failed
first21 / top21variance selection.

For every embedding dim d of a (N, 29, D) window tensor, fit a TINY
predictor whose features are that dim's values at all 29 window positions
(X shape (n, 29)) and score its discriminative power as AUC. Ranks all D
dims; the final model keeps the top-N dims RAW (29 x N features, no PCA).

Leak-free by construction: the caller passes ONLY training rows
(prepare_fold's train_indices — held-out bucket and test fold already
excluded), and the screener's internal AUC is leave-one-TRAIN-fold-out
over the EXISTING global FOLD_COL (homology groups from 01_datasplit).
No new random splits; homology separation holds at both levels.

Screeners:
    xgb    small XGBoost (200 trees, depth 4, native NaN) — matches the
           final model class
    lr_l2  median-impute + StandardScaler + LogisticRegression — linear
           baseline, deterministic

Ranking uses auc_disc = max(AUC, 1-AUC): a dim with AUC 0.3 is strongly
informative (inverted); the final XGB exploits direction freely, so
inverted dims must not be dropped.

Results are cached as CSV (deterministic per key/rows/screen model), so
the N-sweep and the 30 nested-CV prepare_fold calls reuse rankings.

Outputs: one CSV per call: dim, auc_raw, auc_disc, rank (rank 1 = best).
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import RANDOM_STATE

WINDOW_CHUNK_ROWS = 2048


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_window_block(prepared_path, indices):
    """
    Load (n, 29, D) float32 window block + (n, 29) bool pad mask for the
    given row indices. Pad slots are NOT modified here (raw fill kept);
    screeners mask them per dim.
    """
    indices = np.asarray(indices)
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        M = f["pad_mask"]
        n, window, d = W.shape
        out = np.empty((len(indices), window, d), dtype=np.float32)
        mask = np.empty((len(indices), window), dtype=bool)
        cur = 0
        for i0 in range(0, len(indices), WINDOW_CHUNK_ROWS):
            idx = indices[i0:i0 + WINDOW_CHUNK_ROWS]
            block = W[idx]                      # (c, window, D)
            out[cur:cur + len(idx)] = block
            mask[cur:cur + len(idx)] = M[idx]
            cur += len(idx)
    return out, mask


# ──────────────────────────────────────────────
# 2. TINY PREDICTORS
# ──────────────────────────────────────────────

def _fit_xgb(X_tr, y_tr, X_sc):
    from xgboost import XGBClassifier
    m = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
        reg_lambda=1.0, reg_alpha=0.0,
        eval_metric="logloss", n_jobs=1, verbosity=0,
        random_state=RANDOM_STATE,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_sc)[:, 1]


def _fit_lr(X_tr, y_tr, X_sc):
    pipe = [
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
    ]
    Xp = X_tr
    for step in pipe[:-1]:
        Xp = step.fit_transform(Xp)
    pipe[-1].fit(Xp, y_tr)
    Xs = X_sc
    for step in pipe[:-1]:
        Xs = step.transform(Xs)
    return pipe[-1].predict_proba(Xs)[:, 1]


_SCREENERS = {"xgb": _fit_xgb, "lr_l2": _fit_lr}


# ──────────────────────────────────────────────
# 3. PER-DIM OOF AUC
# ──────────────────────────────────────────────

def _oof_auc_dim(Xd, y, fold_ids, inner_folds, screen_model):
    """
    Leave-one-TRAIN-fold-out OOF AUC for ONE dim.
    Xd: (n, 29) with pad slots already NaN. inner_folds: sorted unique
    fold ids among the training rows. Scoring folds with a single class
    are skipped (their rows get no OOF prediction). Returns (auc_raw,
    auc_disc) with auc_disc = max(auc, 1-auc).
    """
    fit_fn = _SCREENERS[screen_model]
    oof = np.full(len(y), np.nan)
    for f in inner_folds:
        tr = fold_ids != f
        sc = ~tr
        if tr.sum() == 0 or sc.sum() == 0:
            continue
        if len(np.unique(y[sc])) < 2:
            continue          # single-class scoring fold: no AUC there
        try:
            oof[sc] = fit_fn(Xd[tr], y[tr], Xd[sc])
        except Exception:
            return 0.5, 0.5   # degenerate dim (e.g. all-NaN): neutral
    ok = ~np.isnan(oof)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return 0.5, 0.5
    auc = float(roc_auc_score(y[ok], oof[ok]))
    return auc, max(auc, 1.0 - auc)


def screen_dims(prepared_path, indices, y, fold_ids, screen_model,
                cache_dir=None, cache_key=None, n_jobs=12, tag=""):
    """
    Rank all D dims of the window embedding by leave-one-train-fold-out
    OOF AUC of a tiny per-dim predictor (features = 29 window positions).

    Parameters
    ----------
    prepared_path : prepared H5 ('windows' (N,29,D), 'pad_mask' (N,29))
    indices       : training row indices (leak-free by caller contract)
    y             : labels for those rows
    fold_ids      : global fold id per training row (internal LOO groups)
    screen_model  : 'xgb' | 'lr_l2'
    cache_dir/cache_key : optional CSV cache (deterministic per inputs)
    n_jobs        : joblib workers for the per-dim loop

    Returns: DataFrame[rank, dim, auc_raw, auc_disc] sorted by rank.
    """
    if screen_model not in _SCREENERS:
        raise ValueError(f"unknown screen model: {screen_model}")

    cache_path = None
    if cache_dir is not None and cache_key is not None:
        cache_path = Path(cache_dir) / f"{cache_key}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)

    X, mask = load_window_block(prepared_path, indices)
    n, window, d = X.shape
    X = X.transpose(2, 0, 1).copy()      # (D, n, 29) — contiguous per dim
    # pad slots -> NaN so both screeners ignore them
    # X is (D, n, 29); mask is (n, 29) -> boolean index on the last two axes
    X[:, mask] = np.nan

    y = np.asarray(y)
    fold_ids = np.asarray(fold_ids)
    inner_folds = np.unique(fold_ids)

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_oof_auc_dim)(X[k], y, fold_ids, inner_folds, screen_model)
        for k in range(d)
    )

    auc_raw = np.array([r[0] for r in res])
    auc_disc = np.array([r[1] for r in res])
    order = np.argsort(-auc_disc, kind="stable")
    rank = np.empty(d, dtype=np.int32)
    rank[order] = np.arange(1, d + 1)
    df = pd.DataFrame({
        "rank": rank, "dim": np.arange(d),
        "auc_raw": auc_raw.round(6), "auc_disc": auc_disc.round(6),
    }).sort_values("rank").reset_index(drop=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, cache_path)
        print(f"    [screen {screen_model}] {tag}: cached {cache_path}")
    return df


# ──────────────────────────────────────────────
# 4. CLI (standalone full-data screening; descriptive only)
# ──────────────────────────────────────────────

def main():
    from config import SPLIT_DATA_PATH, FOLD_COL, LABEL_COL, \
        get_embedding_source
    ap = argparse.ArgumentParser(
        description="Per-dim screening (full-data, descriptive only — "
                    "performance claims come from fold-nested dimselect)")
    ap.add_argument("--key", required=True,
                    help="EMBEDDING_SOURCES key (prepared H5)")
    ap.add_argument("--screen-model", default="xgb",
                    choices=sorted(_SCREENERS))
    ap.add_argument("--out", default=None,
                    help="output CSV (default: results/tables/"
                         "dim_screening_<key>_<model>.csv)")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    src = get_embedding_source(args.key)
    prepared = src.get("prepared_path")
    if prepared is None or not Path(prepared).exists():
        raise SystemExit(f"{args.key}: prepared H5 missing ({prepared})")

    df = pd.read_csv(SPLIT_DATA_PATH)
    indices = np.arange(len(df))
    y = df[LABEL_COL].values.astype(np.int32)
    folds = df[FOLD_COL].values

    out = Path(args.out) if args.out else Path("results/tables") / \
        f"dim_screening_{args.key}_{args.screen_model}.csv"
    res = screen_dims(
        str(prepared), indices, y, folds, args.screen_model,
        cache_dir=out.parent, cache_key=out.stem,
        n_jobs=args.jobs, tag=f"{args.key} full-data",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(res.head(20).to_string(index=False))
    print(f"\n  wrote {out} ({len(res)} dims)")


if __name__ == "__main__":
    main()
