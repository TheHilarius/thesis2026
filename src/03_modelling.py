"""
03_modelling.py
Cross-validation pipeline for MHC-I processing prediction.
Model-agnostic: select a model via --model <key> (default: rf).

Assumes:
  - 01_datasplit.py has been run  (fold column exists).
  - 02_datasplit_analysis.py has been run for data-quality checks.

Usage:
    python 03_modelling.py               # runs default model (rf)
    python 03_modelling.py --model rf    # Random Forest baseline
    python 03_modelling.py --model lr    # Logistic Regression baseline
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import importlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)
from datetime import datetime
import time
import json
import pickle

from config import (
    SPLIT_DATA_PATH, LOG_DIR, MODEL_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    PEPTIDE_COL, LABEL_COL, FOLD_COL,
    POSITION_AA_COLS, RANDOM_STATE,
    DEFAULT_MODEL,
    get_feature_cols, get_model_config, validate_config,
)


# ──────────────────────────────────────────────
# 0. LOGGER
# ──────────────────────────────────────────────

class Logger:
    """Tee: writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


# ──────────────────────────────────────────────
# 1. MODEL FACTORY
# ──────────────────────────────────────────────

def build_model(model_cfg):
    """
    Instantiate a sklearn model from a MODEL_REGISTRY entry.

    The 'model_class' string (e.g. 'sklearn.ensemble.RandomForestClassifier')
    is imported dynamically so config.py stays free of sklearn imports.
    """
    class_path = model_cfg["model_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**model_cfg["params"])


# ──────────────────────────────────────────────
# 2. METRICS
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute all classification metrics from true labels and predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return metrics


def print_metrics(metrics, prefix=""):
    """Pretty-print a metrics dictionary."""
    print(f"{prefix}  AUC-ROC: {metrics['auc_roc']:.4f}  |  AUC-PR: {metrics['auc_pr']:.4f}  |  "
          f"Acc: {metrics['accuracy']:.4f}  |  F1: {metrics['f1']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}")
    print(f"{prefix}  Sens: {metrics['sensitivity']:.4f}  |  Spec: {metrics['specificity']:.4f}  |  "
          f"PPV: {metrics['ppv']:.4f}  |  NPV: {metrics['npv']:.4f}")


# ──────────────────────────────────────────────
# 3. DATA PREPARATION
# ──────────────────────────────────────────────

def impute_nan(X_train, X_test, feature_cols, fold_id):
    """
    Replace Inf with NaN, then fill NaN with column median from train.
    Returns cleaned arrays and the medians used (for held-out imputation).
    """
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: {name}: {n_nan} NaN, {n_inf} Inf values")
            nan_per_col = np.isnan(arr).sum(axis=0)
            for col_idx in np.where(nan_per_col > 0)[0]:
                print(f"    Column '{feature_cols[col_idx]}': "
                      f"{nan_per_col[col_idx]} NaN values")

    col_medians = np.nanmedian(X_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    print(f"  Fold {fold_id}: medians computed from {X_train.shape[0]} training samples")

    n_imputed_train = 0
    n_imputed_test = 0
    for col_idx in range(X_train.shape[1]):
        train_nan = np.isnan(X_train[:, col_idx])
        test_nan = np.isnan(X_test[:, col_idx])
        n_imputed_train += train_nan.sum()
        n_imputed_test += test_nan.sum()
        X_train[train_nan, col_idx] = col_medians[col_idx]
        X_test[test_nan, col_idx] = col_medians[col_idx]

    if n_imputed_train > 0 or n_imputed_test > 0:
        print(f"  Fold {fold_id}: imputed {n_imputed_train} train values, "
              f"{n_imputed_test} test values")

    assert not np.isnan(X_train).any(), "NaN still in X_train"
    assert not np.isnan(X_test).any(), "NaN still in X_test"

    return X_train, X_test, col_medians


def impute_nan_single(X, col_medians):
    """Impute NaN in a single array using pre-computed medians."""
    X = np.where(np.isinf(X), np.nan, X)
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        X[nan_mask, col_idx] = col_medians[col_idx]
    return X


def resolve_feature_cols(df):
    """
    Return the list of numeric feature columns, silently dropping any
    non-numeric ones (the detailed report lives in 02_datasplit_analysis).
    """
    feature_cols = get_feature_cols(df.columns)
    non_numeric = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        feature_cols = [c for c in feature_cols if c not in non_numeric]
        print(f"  Excluded {len(non_numeric)} non-numeric feature column(s)")
    return feature_cols


def prepare_fold_data(df, feature_cols, fold_id, needs_scaling):
    """
    Split CV data into train/test for a given fold.
    Optionally standardises features (fit on train, transform both).

    Returns:
        X_train, y_train, X_test, y_test, col_medians, scaler, fold_info
    """
    cv_df = df[df[FOLD_COL] != HELD_OUT_INDEX].copy()

    train_df = cv_df[cv_df[FOLD_COL] != fold_id].copy()
    test_df = cv_df[cv_df[FOLD_COL] == fold_id].copy()

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[LABEL_COL].values.astype(np.int32)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df[LABEL_COL].values.astype(np.int32)

    # Impute NaN (always before scaling)
    X_train, X_test, col_medians = impute_nan(X_train, X_test, feature_cols, fold_id)

    # Optional scaling
    scaler = None
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print(f"  Fold {fold_id}: features standardised (mean=0, std=1)")

    fold_info = {
        "fold_id": fold_id,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int(len(y_train) - y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int(len(y_test) - y_test.sum()),
        "n_features": len(feature_cols),
    }

    return X_train, y_train, X_test, y_test, col_medians, scaler, fold_info


def prepare_held_out_data(df, feature_cols, col_medians, scaler=None):
    """Prepare the held-out set using pre-computed medians and optional scaler."""
    ho_df = df[df[FOLD_COL] == HELD_OUT_INDEX].copy()

    X_ho = ho_df[feature_cols].values.astype(np.float64)
    y_ho = ho_df[LABEL_COL].values.astype(np.int32)

    X_ho = impute_nan_single(X_ho, col_medians)

    if scaler is not None:
        X_ho = scaler.transform(X_ho)

    return X_ho, y_ho


# ──────────────────────────────────────────────
# 4. MODEL TRAINING (one fold)
# ──────────────────────────────────────────────

def train_one_fold(model_cfg, X_train, y_train, X_test, y_test, fold_id):
    """
    Train a model on one CV fold and evaluate.

    Returns:
        model, test_metrics, y_prob
    """
    model = build_model(model_cfg)
    display = model_cfg["display_name"]

    print(f"  Training {display} ...")
    t_train_start = time.time()
    model.fit(X_train, y_train)
    t_train_end = time.time()
    print(f"  Training time: {t_train_end - t_train_start:.1f}s")

    # Predict probabilities (probability of class 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba (e.g. SVM without probability)
        y_prob = model.decision_function(X_test)

    test_metrics = compute_metrics(y_test, y_prob)

    return model, test_metrics, y_prob


# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE / COEFFICIENTS
# ──────────────────────────────────────────────

def extract_feature_weights(model, model_cfg):
    """
    Extract per-feature weight vector from a trained model.

    Returns:
        weights (1-D array, length n_features) or None
    """
    attr = model_cfg.get("coef_attr")
    if attr is None or not hasattr(model, attr):
        return None

    raw = getattr(model, attr)
    # coef_ is (1, n_features) for binary LR; feature_importances_ is (n_features,)
    weights = np.asarray(raw).ravel()
    return weights


def print_feature_weights(weights, feature_cols, model_cfg, top_n=20):
    """
    Print top N features by absolute weight.
    Label adapts to model type (importance vs coefficient).
    """
    if weights is None:
        print("\n  (model does not expose feature weights)")
        return weights, None

    abs_weights = np.abs(weights)
    indices = np.argsort(abs_weights)[::-1]

    is_coef = model_cfg["coef_attr"] == "coef_"
    col_label = "Coefficient" if is_coef else "Importance"

    print(f"\n  Top {min(top_n, len(feature_cols))} feature {col_label.lower()}s:")
    print(f"  {'Rank':<6} {'Feature':<40} {col_label:>12}"
          f"{'  |Coef|':>10}" if is_coef else "")
    print(f"  {'-' * 60}")

    for rank in range(min(top_n, len(feature_cols))):
        idx = indices[rank]
        line = f"  {rank + 1:<6} {feature_cols[idx]:<40} {weights[idx]:>12.6f}"
        if is_coef:
            line += f"  {abs_weights[idx]:>10.6f}"
        print(line)

    return weights, indices


# ──────────────────────────────────────────────
# 6. AGGREGATE CV RESULTS
# ──────────────────────────────────────────────

def aggregate_cv_results(all_fold_metrics):
    """Compute mean +/- std across CV folds for each metric."""
    metric_names = all_fold_metrics[0].keys()
    summary = {}
    for m in metric_names:
        values = [fold[m] for fold in all_fold_metrics]
        summary[m] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }
    return summary


def print_cv_summary(summary):
    """Pretty-print aggregated CV results."""
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION SUMMARY (mean +/- std)")
    print("=" * 80)
    for metric, stats in summary.items():
        vals = "  ".join(f"{v:.4f}" for v in stats["values"])
        print(f"  {metric:<15} {stats['mean']:.4f} +/- {stats['std']:.4f}    "
              f"[{vals}]")


# ──────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="03_modelling: cross-validation + held-out evaluation",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model key from MODEL_REGISTRY (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    model_key = args.model

    validate_config()
    model_cfg = get_model_config(model_key)
    display_name = model_cfg["display_name"]
    needs_scaling = model_cfg["needs_scaling"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"03_modelling_{model_key}_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    # -- Header --
    print("=" * 80)
    print(f"  03_MODELLING -- {display_name.upper()} BASELINE")
    print("=" * 80)
    print(f"  Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model key:          {model_key}")
    print(f"  Model class:        {model_cfg['model_class']}")
    print(f"  Data path:          {SPLIT_DATA_PATH}")
    print(f"  Log path:           {LOG_PATH}")
    print(f"  CV folds (k):       {N_CV_FOLDS}")
    print(f"  Held-out bucket:    {HELD_OUT_INDEX}")
    print(f"  Random state:       {RANDOM_STATE}")
    print(f"  Needs scaling:      {needs_scaling}")
    print(f"\n  Hyperparameters:")
    for k, v in model_cfg["params"].items():
        print(f"    {k:<25} {v}")
    print("=" * 80)

    # -- Load split data --
    print(f"\nLoading split data from: {SPLIT_DATA_PATH}")
    df = pd.read_csv(SPLIT_DATA_PATH)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # -- Quick fold sanity check --
    if FOLD_COL not in df.columns:
        print(f"FATAL: Column '{FOLD_COL}' not found. Run 01_datasplit.py first.")
        logger.close()
        sys.exit(1)

    actual_folds = sorted(df[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))
    if actual_folds != expected_folds:
        print(f"FATAL: Expected folds {expected_folds}, found {actual_folds}")
        logger.close()
        sys.exit(1)
    print(f"[OK] Fold column validated: {actual_folds}")

    # -- Positional AA column status --
    present_pos = [c for c in POSITION_AA_COLS if c in df.columns]
    if present_pos:
        is_encoded = np.issubdtype(df[present_pos[0]].dtype, np.number)
        if is_encoded:
            print(f"  Positional AA columns appear already encoded (numeric)")
        else:
            print(f"  {len(present_pos)} positional AA columns present but UNENCODED "
                  f"(dtype: object) -- excluded from features")

    # -- Resolve feature columns --
    feature_cols = resolve_feature_cols(df)
    n_features = len(feature_cols)
    print(f"  Using {n_features} numeric feature columns")

    if n_features == 0:
        print("FATAL: No numeric feature columns found. Check config and data.")
        logger.close()
        sys.exit(1)

    print(f"\n  Feature columns entering the model:")
    for i, col in enumerate(feature_cols):
        print(f"    [{i:>3}] {col}")

    print(f"\n{'=' * 80}")
    print(f"PIPELINE READY: {n_features} features, {N_CV_FOLDS} folds, {display_name}")
    print(f"{'=' * 80}")

    # ── Cross-Validation Loop ──
    all_fold_metrics = []
    all_fold_predictions = {}
    all_fold_weights = []
    best_fold_auc = -1
    best_fold_id = -1
    best_fold_model = None
    best_fold_medians = None
    best_fold_scaler = None

    for fold_id in range(N_CV_FOLDS):
        print(f"\n{'-' * 80}")
        print(f"  FOLD {fold_id}/{N_CV_FOLDS - 1}")
        print(f"{'-' * 80}")

        t_fold_start = time.time()

        # Prepare data
        X_train, y_train, X_test, y_test, col_medians, scaler, fold_info = (
            prepare_fold_data(df, feature_cols, fold_id, needs_scaling)
        )

        train_ratio = fold_info["n_train_neg"] / fold_info["n_train_pos"] if fold_info["n_train_pos"] > 0 else float("inf")
        test_ratio = fold_info["n_test_neg"] / fold_info["n_test_pos"] if fold_info["n_test_pos"] > 0 else float("inf")

        print(f"  Train: {fold_info['n_train']:>6} samples "
              f"(pos={fold_info['n_train_pos']}, neg={fold_info['n_train_neg']}, "
              f"ratio={train_ratio:.3f})")
        print(f"  Test:  {fold_info['n_test']:>6} samples "
              f"(pos={fold_info['n_test_pos']}, neg={fold_info['n_test_neg']}, "
              f"ratio={test_ratio:.3f})")
        print(f"  Features: {fold_info['n_features']}")

        # Train
        model, test_metrics, y_prob = train_one_fold(
            model_cfg, X_train, y_train, X_test, y_test, fold_id,
        )

        t_fold_end = time.time()

        print(f"\n  Fold {fold_id} results ({t_fold_end - t_fold_start:.1f}s):")
        print_metrics(test_metrics, prefix="  ")

        # Feature weights for this fold
        weights = extract_feature_weights(model, model_cfg)
        print_feature_weights(weights, feature_cols, model_cfg, top_n=15)
        if weights is not None:
            all_fold_weights.append(weights)

        # Store predictions
        all_fold_predictions[fold_id] = {
            "y_true": y_test.tolist(),
            "y_prob": y_prob.tolist(),
        }

        all_fold_metrics.append(test_metrics)

        # Track best fold
        if test_metrics["auc_roc"] > best_fold_auc:
            best_fold_auc = test_metrics["auc_roc"]
            best_fold_id = fold_id
            best_fold_model = model
            best_fold_medians = col_medians
            best_fold_scaler = scaler

        # Save fold model
        fold_model_path = MODEL_DIR / f"{model_key}_model_fold{fold_id}.pkl"
        with open(fold_model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model: {fold_model_path}")

    # ── CV Summary ──
    summary = aggregate_cv_results(all_fold_metrics)
    print_cv_summary(summary)
    print(f"\n  Best fold: {best_fold_id} (AUC-ROC = {best_fold_auc:.4f})")

    # ── Per-fold comparison table ──
    print(f"\n{'=' * 80}")
    print("PER-FOLD METRIC COMPARISON")
    print("=" * 80)
    metric_names = list(all_fold_metrics[0].keys())
    header = f"  {'Fold':<8}" + "".join(f"{m:<13}" for m in metric_names)
    print(header)
    print(f"  {'-' * (8 + 13 * len(metric_names))}")
    for fold_id, metrics in enumerate(all_fold_metrics):
        row = f"  {fold_id:<8}" + "".join(f"{metrics[m]:<13.4f}" for m in metric_names)
        print(row)
    print(f"  {'-' * (8 + 13 * len(metric_names))}")
    mean_row = f"  {'mean':<8}" + "".join(f"{summary[m]['mean']:<13.4f}" for m in metric_names)
    std_row = f"  {'std':<8}" + "".join(f"{summary[m]['std']:<13.4f}" for m in metric_names)
    print(mean_row)
    print(std_row)

    # ── Averaged feature weights across folds ──
    if all_fold_weights:
        print(f"\n{'=' * 80}")
        is_coef = model_cfg["coef_attr"] == "coef_"
        weight_label = "COEFFICIENTS" if is_coef else "FEATURE IMPORTANCE"
        print(f"AVERAGED {weight_label} ACROSS ALL FOLDS")
        print("=" * 80)

        avg_weights = np.mean(all_fold_weights, axis=0)
        std_weights = np.std(all_fold_weights, axis=0)
        abs_avg = np.abs(avg_weights)
        sorted_indices = np.argsort(abs_avg)[::-1]

        col_label = "Mean Coef" if is_coef else "Mean Imp"
        print(f"  {'Rank':<6} {'Feature':<40} {col_label:>12} {'Std':>12}")
        print(f"  {'-' * 72}")
        for rank in range(min(30, len(feature_cols))):
            idx = sorted_indices[rank]
            print(f"  {rank + 1:<6} {feature_cols[idx]:<40} "
                  f"{avg_weights[idx]:>12.6f} {std_weights[idx]:>12.6f}")

    # ── Held-Out Evaluation ──
    print(f"\n{'=' * 80}")
    print("HELD-OUT VALIDATION SET EVALUATION")
    print(f"{'=' * 80}")
    print(f"  Using model from best fold ({best_fold_id})")

    X_ho, y_ho = prepare_held_out_data(
        df, feature_cols, best_fold_medians, scaler=best_fold_scaler,
    )

    n_ho_pos = int(y_ho.sum())
    n_ho_neg = int(len(y_ho) - y_ho.sum())
    ho_ratio = n_ho_neg / n_ho_pos if n_ho_pos > 0 else float("inf")
    print(f"  Held-out samples: {len(y_ho)} "
          f"(pos={n_ho_pos}, neg={n_ho_neg}, ratio={ho_ratio:.3f})")

    if hasattr(best_fold_model, "predict_proba"):
        ho_probs = best_fold_model.predict_proba(X_ho)[:, 1]
    else:
        ho_probs = best_fold_model.decision_function(X_ho)

    ho_metrics = compute_metrics(y_ho, ho_probs)

    print(f"\n  Held-out results ({len(y_ho)} samples):")
    print_metrics(ho_metrics, prefix="  ")

    ho_preds = (ho_probs >= 0.5).astype(int)
    print(f"\n  Classification Report:")
    print(classification_report(y_ho, ho_preds,
                                target_names=["Negative", "Positive"], digits=4))

    # ── CV vs Held-out comparison ──
    print(f"{'=' * 80}")
    print("CV AVERAGE vs HELD-OUT COMPARISON")
    print("=" * 80)
    print(f"  {'Metric':<15} {'CV mean':>10} {'CV std':>10} {'Held-out':>10} {'Delta':>10}")
    print(f"  {'-' * 57}")
    for m in metric_names:
        cv_mean = summary[m]["mean"]
        cv_std = summary[m]["std"]
        ho_val = ho_metrics[m]
        delta = ho_val - cv_mean
        print(f"  {m:<15} {cv_mean:>10.4f} {cv_std:>10.4f} {ho_val:>10.4f} {delta:>+10.4f}")

    # ── Save All Results ──
    pos_status = "encoded" if (
        present_pos and np.issubdtype(df[present_pos[0]].dtype, np.number)
    ) else "unencoded (excluded)"

    results = {
        "timestamp": timestamp,
        "model_key": model_key,
        "model_type": display_name,
        "model_class": model_cfg["model_class"],
        "config": {
            "n_cv_folds": N_CV_FOLDS,
            "held_out_index": HELD_OUT_INDEX,
            "random_state": RANDOM_STATE,
            "needs_scaling": needs_scaling,
            "hyperparameters": {k: str(v) for k, v in model_cfg["params"].items()},
            "n_features": n_features,
            "feature_cols": feature_cols,
            "positional_aa_cols_status": pos_status,
        },
        "cv_summary": {m: {"mean": s["mean"], "std": s["std"]}
                       for m, s in summary.items()},
        "fold_metrics": all_fold_metrics,
        "held_out_metrics": ho_metrics,
        "best_fold_id": best_fold_id,
        "best_fold_auc": best_fold_auc,
        "fold_predictions": all_fold_predictions,
    }

    # Add averaged feature weights
    if all_fold_weights:
        results["avg_feature_weights"] = {
            feature_cols[i]: float(avg_weights[i])
            for i in sorted_indices[:min(50, len(feature_cols))]
        }

    results_path = MODEL_DIR / f"cv_results_{model_key}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Save best model
    best_model_path = MODEL_DIR / f"best_{model_key}_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_fold_model, f)
    print(f"Best model saved: {best_model_path}")

    # Save best medians
    medians_path = MODEL_DIR / f"best_{model_key}_medians.pkl"
    with open(medians_path, "wb") as f:
        pickle.dump(best_fold_medians, f)
    print(f"Best medians saved: {medians_path}")

    # Save best scaler (if scaling was used)
    if best_fold_scaler is not None:
        scaler_path = MODEL_DIR / f"best_{model_key}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(best_fold_scaler, f)
        print(f"Best scaler saved: {scaler_path}")

    # Save feature column list
    feature_cols_path = MODEL_DIR / f"{model_key}_feature_cols.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Feature columns saved: {feature_cols_path}")

    # ── Footer ──
    t_end = time.time()
    total_minutes = (t_end - t_start) / 60
    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"  Total runtime:    {t_end - t_start:.1f}s ({total_minutes:.1f} min)")
    print(f"  Model:            {display_name} ({model_key})")
    print(f"  Scaling:          {'yes' if needs_scaling else 'no'}")
    print(f"  Features:         {n_features}")
    print(f"  Positional AA:    {len(present_pos)} columns ({pos_status})")
    print(f"  CV folds:         {N_CV_FOLDS}")
    print(f"  Best fold:        {best_fold_id} (AUC = {best_fold_auc:.4f})")
    print(f"  Held-out AUC:     {ho_metrics['auc_roc']:.4f}")
    print(f"  Held-out MCC:     {ho_metrics['mcc']:.4f}")
    print(f"  Models dir:       {MODEL_DIR}/")
    print(f"  Results file:     {results_path}")
    print(f"  Feature cols:     {feature_cols_path}")
    print(f"  Log file:         {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()