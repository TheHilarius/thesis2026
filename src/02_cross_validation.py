"""
02_cross_validation.py
Cross-validation pipeline for MHC-I processing prediction.
Baseline model: Random Forest (no regularization).
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    RANDOM_STATE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES, RF_N_JOBS,
    RF_CLASS_WEIGHT,
    get_feature_cols, validate_config,
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
# 1. METRICS
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
# 2. DATA PREPARATION
# ──────────────────────────────────────────────

def impute_nan(X_train, X_test, feature_cols):
    """
    Replace Inf with NaN, then fill NaN with column median from train.
    Returns cleaned arrays and the medians used (for held-out imputation).
    """
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    # Report
    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING: {name}: {n_nan} NaN, {n_inf} Inf values")
            nan_per_col = np.isnan(arr).sum(axis=0)
            for col_idx in np.where(nan_per_col > 0)[0]:
                print(f"    Column '{feature_cols[col_idx]}': "
                      f"{nan_per_col[col_idx]} NaN values")

    # Compute medians from train only
    col_medians = np.nanmedian(X_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    # Fill NaN
    for col_idx in range(X_train.shape[1]):
        train_nan = np.isnan(X_train[:, col_idx])
        test_nan = np.isnan(X_test[:, col_idx])
        X_train[train_nan, col_idx] = col_medians[col_idx]
        X_test[test_nan, col_idx] = col_medians[col_idx]

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


def prepare_fold_data(df, feature_cols, fold_id):
    """
    Split CV data into train/test for a given fold.
    No scaling needed for Random Forest.

    Returns:
        X_train, y_train, X_test, y_test, col_medians, fold_info
    """
    cv_df = df[df[FOLD_COL] != HELD_OUT_INDEX].copy()

    train_df = cv_df[cv_df[FOLD_COL] != fold_id].copy()
    test_df = cv_df[cv_df[FOLD_COL] == fold_id].copy()

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(np.int32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.int32)

    # Impute NaN
    X_train, X_test, col_medians = impute_nan(X_train, X_test, feature_cols)

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

    return X_train, y_train, X_test, y_test, col_medians, fold_info


def prepare_held_out_data(df, feature_cols, col_medians):
    """Prepare the held-out set using pre-computed medians for imputation."""
    ho_df = df[df[FOLD_COL] == HELD_OUT_INDEX].copy()

    X_ho = ho_df[feature_cols].values.astype(np.float32)
    y_ho = ho_df[LABEL_COL].values.astype(np.int32)

    X_ho = impute_nan_single(X_ho, col_medians)

    return X_ho, y_ho


# ──────────────────────────────────────────────
# 3. MODEL TRAINING (one fold)
# ──────────────────────────────────────────────

def train_one_fold(X_train, y_train, X_test, y_test, fold_id):
    """
    Train a Random Forest on one CV fold and evaluate.

    Returns:
        model, test_metrics, y_prob
    """
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    print(f"  Training Random Forest ({RF_N_ESTIMATORS} trees) ...")
    t_train_start = time.time()
    model.fit(X_train, y_train)
    t_train_end = time.time()
    print(f"  Training time: {t_train_end - t_train_start:.1f}s")

    # Predict probabilities (probability of class 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = compute_metrics(y_test, y_prob)

    return model, test_metrics, y_prob


# ──────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def print_feature_importance(model, feature_cols, top_n=20):
    """Print top N most important features from the Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  Top {min(top_n, len(feature_cols))} feature importances:")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print(f"  {'-' * 60}")
    for rank in range(min(top_n, len(feature_cols))):
        idx = indices[rank]
        print(f"  {rank + 1:<6} {feature_cols[idx]:<40} {importances[idx]:>12.6f}")

    return importances, indices


# ──────────────────────────────────────────────
# 5. AGGREGATE CV RESULTS
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
# 6. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"02_cross_validation_RF_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    # -- Header --
    print("=" * 80)
    print("  02_CROSS_VALIDATION -- RANDOM FOREST BASELINE")
    print("=" * 80)
    print(f"  Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data path:          {SPLIT_DATA_PATH}")
    print(f"  Log path:           {LOG_PATH}")
    print(f"  CV folds (k):       {N_CV_FOLDS}")
    print(f"  Held-out bucket:    {HELD_OUT_INDEX}")
    print(f"  Random state:       {RANDOM_STATE}")
    print(f"  RF n_estimators:    {RF_N_ESTIMATORS}")
    print(f"  RF max_depth:       {RF_MAX_DEPTH}")
    print(f"  RF min_samples_s:   {RF_MIN_SAMPLES_SPLIT}")
    print(f"  RF min_samples_l:   {RF_MIN_SAMPLES_LEAF}")
    print(f"  RF max_features:    {RF_MAX_FEATURES}")
    print(f"  RF class_weight:    {RF_CLASS_WEIGHT}")
    print(f"  RF n_jobs:          {RF_N_JOBS}")
    print("=" * 80)

    # -- Load split data --
    print(f"\nLoading split data from: {SPLIT_DATA_PATH}")
    df = pd.read_csv(SPLIT_DATA_PATH)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # -- Verify fold column --
    if FOLD_COL not in df.columns:
        print(f"FATAL: Column '{FOLD_COL}' not found. Run 01_datasplit.py first.")
        logger.close()
        sys.exit(1)

    actual_folds = sorted(df[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))
    if actual_folds != expected_folds:
        print(f"FATAL: Expected folds {expected_folds}, found {actual_folds}")
        print(f"  Check that config.N_CV_FOLDS={N_CV_FOLDS} matches 01_datasplit.py")
        logger.close()
        sys.exit(1)
    print(f"[OK] Fold column validated: {actual_folds}")

    # -- Label distribution --
    print(f"\nLabel distribution:")
    label_counts = df[LABEL_COL].value_counts().sort_index()
    for val, cnt in label_counts.items():
        lbl = "positive" if val == 1 else "negative" if val == 0 else f"unknown({val})"
        print(f"  {val} ({lbl}): {cnt:>8}  ({cnt / len(df) * 100:.1f}%)")

    # -- Identify feature columns --
    feature_cols = get_feature_cols(df.columns)
    print(f"\nFeature columns identified ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  [{i:>3}] {col:<40} dtype: {df[col].dtype}")

    # Remove non-numeric columns
    non_numeric = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        print(f"\nWARNING: Non-numeric feature columns excluded ({len(non_numeric)}):")
        for c in non_numeric:
            print(f"  - {c} (dtype: {df[c].dtype})")
        feature_cols = [c for c in feature_cols if c not in non_numeric]
        print(f"  Remaining feature columns: {len(feature_cols)}")

    # NaN report
    print(f"\nNaN REPORT:")
    total_nan = 0
    for col in feature_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            pct = n_nan / len(df) * 100
            print(f"  {col:<40} {n_nan:>6} NaN ({pct:.1f}%)")
            total_nan += n_nan
    if total_nan == 0:
        print("  No NaN values found in any feature column")
    else:
        print(f"  TOTAL: {total_nan} NaN values across all feature columns")

    # Constant columns check
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        print(f"\nWARNING: Constant feature columns (zero variance):")
        for c in constant_cols:
            print(f"  - {c} (unique values: {df[c].nunique()})")
    else:
        print(f"[OK] No constant feature columns")

    # Feature summary
    n_features = len(feature_cols)
    print(f"\nFeature summary statistics:")
    feat_stats = df[feature_cols].describe().T
    print(f"  {'Column':<35} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'-' * 77}")
    for col in feature_cols[:20]:
        row = feat_stats.loc[col]
        print(f"  {col:<35} {row['mean']:>10.3f} {row['std']:>10.3f} "
              f"{row['min']:>10.3f} {row['max']:>10.3f}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more columns")

    print(f"\n{'=' * 80}")
    print(f"PIPELINE READY: {n_features} features, {N_CV_FOLDS} folds, Random Forest baseline")
    print(f"{'=' * 80}")

    # ── Cross-Validation Loop ──
    all_fold_metrics = []
    all_fold_predictions = {}
    all_fold_importances = []
    best_fold_auc = -1
    best_fold_id = -1
    best_fold_model = None
    best_fold_medians = None

    for fold_id in range(N_CV_FOLDS):
        print(f"\n{'-' * 80}")
        print(f"  FOLD {fold_id}/{N_CV_FOLDS - 1}")
        print(f"{'-' * 80}")

        t_fold_start = time.time()

        # Prepare data
        X_train, y_train, X_test, y_test, col_medians, fold_info = prepare_fold_data(
            df, feature_cols, fold_id,
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
            X_train, y_train, X_test, y_test, fold_id,
        )

        t_fold_end = time.time()

        print(f"\n  Fold {fold_id} results ({t_fold_end - t_fold_start:.1f}s):")
        print_metrics(test_metrics, prefix="  ")

        # Feature importance for this fold
        importances, indices = print_feature_importance(model, feature_cols, top_n=15)
        all_fold_importances.append(importances)

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

        # Save fold model
        fold_model_path = MODEL_DIR / f"rf_model_fold{fold_id}.pkl"
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

    # ── Averaged feature importance across folds ──
    print(f"\n{'=' * 80}")
    print("AVERAGED FEATURE IMPORTANCE ACROSS ALL FOLDS")
    print("=" * 80)
    avg_importances = np.mean(all_fold_importances, axis=0)
    std_importances = np.std(all_fold_importances, axis=0)
    sorted_indices = np.argsort(avg_importances)[::-1]

    print(f"  {'Rank':<6} {'Feature':<40} {'Mean Imp':>12} {'Std':>12}")
    print(f"  {'-' * 72}")
    for rank in range(min(30, len(feature_cols))):
        idx = sorted_indices[rank]
        print(f"  {rank + 1:<6} {feature_cols[idx]:<40} "
              f"{avg_importances[idx]:>12.6f} {std_importances[idx]:>12.6f}")

    # ── Held-Out Evaluation ──
    print(f"\n{'=' * 80}")
    print("HELD-OUT VALIDATION SET EVALUATION")
    print(f"{'=' * 80}")
    print(f"  Using model from best fold ({best_fold_id})")

    X_ho, y_ho = prepare_held_out_data(df, feature_cols, best_fold_medians)

    n_ho_pos = int(y_ho.sum())
    n_ho_neg = int(len(y_ho) - y_ho.sum())
    ho_ratio = n_ho_neg / n_ho_pos if n_ho_pos > 0 else float("inf")
    print(f"  Held-out samples: {len(y_ho)} "
          f"(pos={n_ho_pos}, neg={n_ho_neg}, ratio={ho_ratio:.3f})")

    ho_probs = best_fold_model.predict_proba(X_ho)[:, 1]
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
    results = {
        "timestamp": timestamp,
        "model_type": "RandomForest",
        "config": {
            "n_cv_folds": N_CV_FOLDS,
            "held_out_index": HELD_OUT_INDEX,
            "random_state": RANDOM_STATE,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "rf_max_depth": str(RF_MAX_DEPTH),
            "rf_min_samples_split": RF_MIN_SAMPLES_SPLIT,
            "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "rf_max_features": RF_MAX_FEATURES,
            "rf_class_weight": str(RF_CLASS_WEIGHT),
            "n_features": n_features,
            "feature_cols": feature_cols,
        },
        "cv_summary": {m: {"mean": s["mean"], "std": s["std"]}
                       for m, s in summary.items()},
        "fold_metrics": all_fold_metrics,
        "held_out_metrics": ho_metrics,
        "best_fold_id": best_fold_id,
        "best_fold_auc": best_fold_auc,
        "fold_predictions": all_fold_predictions,
        "avg_feature_importances": {
            feature_cols[i]: float(avg_importances[i])
            for i in sorted_indices[:50]
        },
    }

    results_path = MODEL_DIR / f"cv_results_RF_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Save best model
    best_model_path = MODEL_DIR / "best_rf_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_fold_model, f)
    print(f"Best model saved: {best_model_path}")

    # Save best medians for future inference
    medians_path = MODEL_DIR / "best_rf_medians.pkl"
    with open(medians_path, "wb") as f:
        pickle.dump(best_fold_medians, f)
    print(f"Best medians saved: {medians_path}")

    # ── Footer ──
    t_end = time.time()
    total_minutes = (t_end - t_start) / 60
    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"  Total runtime:    {t_end - t_start:.1f}s ({total_minutes:.1f} min)")
    print(f"  Model:            Random Forest ({RF_N_ESTIMATORS} trees)")
    print(f"  Features:         {n_features}")
    print(f"  CV folds:         {N_CV_FOLDS}")
    print(f"  Best fold:        {best_fold_id} (AUC = {best_fold_auc:.4f})")
    print(f"  Held-out AUC:     {ho_metrics['auc_roc']:.4f}")
    print(f"  Held-out MCC:     {ho_metrics['mcc']:.4f}")
    print(f"  Models dir:       {MODEL_DIR}/")
    print(f"  Results file:     {results_path}")
    print(f"  Log file:         {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()