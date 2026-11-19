#!/usr/bin/env python3
"""
46_xgb_optuna_tune.py
Optuna TPE tuner for the thesis XGBoost model (blosum50 flat, flank5_rmdup).

Conventions (locked 2026-11-19, decision engram #420):
  - FIXED learning_rate=0.1, n_estimators=1000, NO early stopping
    (campaign convention; 21_xgb_es_diagnostics showed inner ES selection
    hurts on every fold). No Optuna pruning — fixed-budget trials are atomic.
  - scale_pos_weight = auto per fold (neg/pos ratio of the train rows),
    exactly like 04_modelling.train_one_fold.
  - Objective: mean AUC-ROC over the 6 outer buckets (folds 0..5), with
    train = fold != bucket, test = fold == bucket — identical to the
    nested-CV outer evaluation. Mean logloss stored per trial as a
    user_attr (free; AUC stays the selection metric, plots check agreement).

Search space:
  max_depth        int   3..10
  min_child_weight float 1..30 (log)
  gamma            float 1e-8..5 (log; 1e-8 ~ 0)
  subsample        float 0.5..1.0
  colsample_bytree float 0.5..1.0
  reg_alpha        float 1e-8..10 (log)
  reg_lambda       float 1e-3..100 (log)

Usage (repo root; runs in runs/xgb_optuna/):
  python src/investigation/46_xgb_optuna_tune.py --trials 50   # one tune job
  python src/investigation/46_xgb_optuna_tune.py --polish       # low-LR phase 6
  python src/investigation/46_xgb_optuna_tune.py --smoke        # wiring check
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.metrics import log_loss

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit("optuna not installed. On the server: pip install --user optuna")

from config import (  # noqa: E402
    DEFAULT_SPLIT_PATH, FOLD_COL, LABEL_COL, N_CV_FOLDS, PROJECT_ROOT,
    RANDOM_STATE, get_feature_set_config,
)
import pandas as pd  # noqa: E402

RUN_DIR = PROJECT_ROOT / "runs" / "xgb_optuna"
STUDY_NAME = "xgb_blosum50"
FEATURES_KEY = "blosum50"
PCA_MODE = "flat"

# ── import 04_modelling as a module (digit-leading filename) ──
_spec = importlib.util.spec_from_file_location(
    "modelling04", PIPELINE_DIR / "04_modelling.py")
mod04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod04)


def load_matrix():
    """Load split data + blosum50 features exactly like 04_modelling main."""
    df_split = pd.read_csv(DEFAULT_SPLIT_PATH)
    print(f"Split data: {DEFAULT_SPLIT_PATH}  ({df_split.shape[0]} rows)")

    # per-peptide pad-count context features (verbatim from 04_modelling main)
    if "start" in df_split.columns and "n_pad" not in df_split.columns:
        df_split["n_pad"] = 10 - np.minimum(
            10, df_split["start"].astype(int) - 1)
    if ("protein_length" in df_split.columns and "end" in df_split.columns
            and "c_pad" not in df_split.columns):
        df_split["c_pad"] = 10 - np.minimum(
            10, df_split["protein_length"].astype(int)
            - df_split["end"].astype(int))

    feat_cfg = get_feature_set_config(FEATURES_KEY)
    df, csv_cols, emb, _ = mod04.load_all_components(
        df_split, feat_cfg, PCA_MODE)
    assert not emb, f"unexpected embedding components: {list(emb)}"

    X = df[csv_cols].to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan          # XGBoost treats NaN as missing
    folds = df[FOLD_COL].to_numpy()
    y = df[LABEL_COL].to_numpy(dtype=np.int32)
    buckets = sorted(df[FOLD_COL].unique())
    print(f"Features: {X.shape[1]}  |  CV buckets: {buckets}")
    return df, csv_cols, X, folds, y, buckets


def make_model_cfg(trial_params: dict) -> dict:
    """MODEL_REGISTRY['xgb'] + trial overrides (scale_pos_weight stays auto)."""
    cfg = copy.deepcopy(mod04.get_model_config("xgb"))
    cfg["params"].update(trial_params)
    return cfg


def eval_params(params: dict, X, folds, y, buckets, tag="trial") -> dict:
    """6-bucket CV: train on fold != bucket, eval on bucket."""
    aucs, lls = [], []
    for bucket in buckets:
        tr = folds != bucket
        te = ~tr
        model, metrics, y_prob = mod04.train_one_fold(
            make_model_cfg(params), X[tr], y[tr], X[te], y[te], bucket)
        aucs.append(metrics["auc_roc"])
        lls.append(log_loss(y[te], y_prob, labels=[0, 1]))
    res = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "logloss_mean": float(np.mean(lls)),
        "auc_folds": [float(a) for a in aucs],
        "logloss_folds": [float(l) for l in lls],
    }
    print(f"  [{tag}] AUC={res['auc_mean']:.4f}±{res['auc_std']:.4f} "
          f"logloss={res['logloss_mean']:.4f}")
    return res


def suggest(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0,
                                                log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
    }


def open_study(db_name: str, seed: int = RANDOM_STATE) -> optuna.Study:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{RUN_DIR / db_name}"
    return optuna.create_study(
        study_name=STUDY_NAME, storage=storage, load_if_exists=True,
        direction="maximize", sampler=TPESampler(seed=seed,
                                                 multivariate=True),
    )


def dump_trials(study: optuna.Study, path: Path) -> None:
    df_t = study.trials_dataframe()
    df_t.to_csv(path, index=False)
    print(f"Trials table -> {path}")


def run_tune(args) -> None:
    df, csv_cols, X, folds, y, buckets = load_matrix()
    n_est = 100 if args.smoke else 1000
    base = {"n_estimators": n_est}

    if args.smoke:
        study = open_study("study_smoke.db")
        buckets = buckets[:2]          # 2 fits only — wiring check
        n_trials = args.trials or 2
    else:
        study = open_study("study.db", seed=args.seed)
        n_trials = args.trials

    def objective(trial):
        params = {**base, **suggest(trial)}
        res = eval_params(params, X, folds, y, buckets)
        trial.set_user_attr("logloss_mean", res["logloss_mean"])
        trial.set_user_attr("auc_folds", res["auc_folds"])
        trial.set_user_attr("logloss_folds", res["logloss_folds"])
        return res["auc_mean"]

    study.optimize(objective, n_trials=n_trials)

    tag = "smoke" if args.smoke else "tuned"
    out = RUN_DIR / (f"best_params_{tag}.json")
    best = study.best_trial
    payload = {**base, **best.params}
    payload_meta = {
        "params": payload,
        "auc_mean": best.value,
        "logloss_mean": best.user_attrs.get("logloss_mean"),
        "auc_folds": best.user_attrs.get("auc_folds"),
        "study_name": STUDY_NAME,
        "features": FEATURES_KEY,
        "split": str(DEFAULT_SPLIT_PATH),
        "mode": tag,
    }
    out.write_text(json.dumps(payload_meta, indent=2))
    dump_trials(study, RUN_DIR / f"trials_{tag}.csv")
    print(f"\nBest {tag} AUC={best.value:.4f}  params={best.params}")
    print(f"Best params -> {out}")


def run_polish() -> None:
    """Low-LR phase 6: winner's structural params, lr x n_estimators grid."""
    df, csv_cols, X, folds, y, buckets = load_matrix()
    study = open_study("study.db")
    n_trials = len(study.trials)
    assert n_trials > 0, "no trials in study — run tuning first"
    structural = dict(study.best_trial.params)
    tuned_auc = study.best_trial.value
    tuned_logloss = study.best_trial.user_attrs.get("logloss_mean")
    print(f"Study winner: AUC={tuned_auc:.4f}  {structural}")

    candidates = []
    for lr in (0.01, 0.02):
        for n_est in (3000, 5000):
            candidates.append({**structural,
                               "learning_rate": lr, "n_estimators": n_est})

    results = []
    for params in candidates:
        res = eval_params(params, X, folds, y, buckets, tag="polish")
        results.append({"params": params, **res})

    best = max(results, key=lambda r: r["auc_mean"])
    fixed = {"params": {"n_estimators": 1000, **structural},
             "auc_mean": tuned_auc, "logloss_mean": tuned_logloss,
             "auc_std": study.best_trial.user_attrs.get("auc_folds") and
             float(np.std(study.best_trial.user_attrs["auc_folds"])) or None,
             "source": "tune winner (lr=0.1, n=1000)"}
    final = fixed if fixed["auc_mean"] >= best["auc_mean"] else {
        **best, "source": "low-LR polish"}
    out = RUN_DIR / "polish_results.json"
    out.write_text(json.dumps({
        "tuned_winner": fixed, "polish_candidates": results,
        "final": final}, indent=2))
    # final model-ready params (for 04_modelling --xgb-params)
    (RUN_DIR / "best_params.json").write_text(
        json.dumps({"params": final["params"], "auc_mean": final["auc_mean"],
                    "source": final["source"]}, indent=2))
    print(f"\nFINAL AUC={final['auc_mean']:.4f}  ({final['source']})")
    print(f"  params: {final['params']}")
    print(f"  -> {out} + best_params.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--trials", type=int, default=50,
                   help="trials this job contributes (default 50)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE,
                   help="TPE sampler seed; give each parallel job a "
                        "different seed (e.g. 42 / 143) so their random "
                        "startup trials don't duplicate")
    p.add_argument("--polish", action="store_true",
                   help="run low-LR polish on finished study")
    p.add_argument("--smoke", action="store_true",
                   help="wiring check: 100 trees, 2 buckets, smoke study")
    return p.parse_args()


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    if args.polish:
        run_polish()
    else:
        run_tune(args)
