"""
02_datasplit_analysis.py
Post-split data quality report: label balance, feature dtypes,
NaN / Inf audit, constant-column check, and summary statistics.
Run AFTER 01_datasplit.py and BEFORE cross-validation.
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    SPLIT_DATA_PATH, LOG_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    LABEL_COL, FOLD_COL,
    get_feature_cols, validate_config,
)


# ──────────────────────────────────────────────
# 0. LOGGER  (same lightweight tee used elsewhere)
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
# 1. FOLD VALIDATION
# ──────────────────────────────────────────────

def validate_folds(df):
    """Check that the fold column exists and contains the expected buckets."""
    if FOLD_COL not in df.columns:
        print(f"FATAL: Column '{FOLD_COL}' not found. Run 01_datasplit.py first.")
        return False

    actual_folds = sorted(df[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))  # 0..k-1 CV + held-out
    if actual_folds != expected_folds:
        print(f"FATAL: Expected folds {expected_folds}, found {actual_folds}")
        print(f"  Check that config.N_CV_FOLDS={N_CV_FOLDS} matches 01_datasplit.py")
        return False

    print(f"[OK] Fold column validated: {actual_folds}")

    # Per-fold sizes
    print(f"\nPer-fold sample counts:")
    print(f"  {'Fold':<8} {'Total':>8} {'Pos':>8} {'Neg':>8} {'Pos %':>8}")
    print(f"  {'-' * 42}")
    for fold in actual_folds:
        subset = df[df[FOLD_COL] == fold]
        n_pos = int(subset[LABEL_COL].sum())
        n_neg = len(subset) - n_pos
        pct = n_pos / len(subset) * 100 if len(subset) > 0 else 0.0
        tag = " (held-out)" if fold == HELD_OUT_INDEX else ""
        print(f"  {fold:<8} {len(subset):>8} {n_pos:>8} {n_neg:>8} {pct:>7.1f}%{tag}")

    return True


# ──────────────────────────────────────────────
# 2. LABEL DISTRIBUTION
# ──────────────────────────────────────────────

def report_label_distribution(df):
    """Print global label counts."""
    print(f"\nLabel distribution (full dataset):")
    label_counts = df[LABEL_COL].value_counts().sort_index()
    for val, cnt in label_counts.items():
        lbl = "positive" if val == 1 else "negative" if val == 0 else f"unknown({val})"
        print(f"  {val} ({lbl}): {cnt:>8}  ({cnt / len(df) * 100:.1f}%)")


# ──────────────────────────────────────────────
# 3. FEATURE-COLUMN IDENTIFICATION
# ──────────────────────────────────────────────

def identify_features(df):
    """
    Resolve feature columns, flag non-numeric ones, and return the
    clean list of numeric feature column names.
    """
    feature_cols = get_feature_cols(df.columns)
    print(f"\nFeature columns identified ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  [{i:>3}] {col:<40} dtype: {df[col].dtype}")

    # Non-numeric guard
    non_numeric = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        print(f"\nWARNING: Non-numeric feature columns ({len(non_numeric)}):")
        for c in non_numeric:
            print(f"  - {c} (dtype: {df[c].dtype})")
        feature_cols = [c for c in feature_cols if c not in non_numeric]
        print(f"  Remaining numeric feature columns: {len(feature_cols)}")

    return feature_cols


# ──────────────────────────────────────────────
# 4. NaN / Inf REPORT
# ──────────────────────────────────────────────

def report_nan(df, feature_cols):
    """Per-column NaN and Inf audit."""
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

    # Inf check (separate pass)
    print(f"\nInf REPORT:")
    total_inf = 0
    for col in feature_cols:
        arr = df[col].values
        n_inf = np.isinf(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0
        if n_inf > 0:
            pct = n_inf / len(df) * 100
            print(f"  {col:<40} {n_inf:>6} Inf ({pct:.1f}%)")
            total_inf += n_inf
    if total_inf == 0:
        print("  No Inf values found in any feature column")
    else:
        print(f"  TOTAL: {total_inf} Inf values across all feature columns")


# ──────────────────────────────────────────────
# 5. CONSTANT-COLUMN CHECK
# ──────────────────────────────────────────────

def report_constant_columns(df, feature_cols):
    """Flag columns with zero variance (≤1 unique value)."""
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        print(f"\nWARNING: Constant feature columns (zero variance):")
        for c in constant_cols:
            print(f"  - {c} (unique values: {df[c].nunique()})")
    else:
        print(f"\n[OK] No constant feature columns")
    return constant_cols


# ──────────────────────────────────────────────
# 6. SUMMARY STATISTICS
# ──────────────────────────────────────────────

def report_feature_summary(df, feature_cols):
    """Print mean / std / min / max for every feature column."""
    print(f"\nFeature summary statistics:")
    feat_stats = df[feature_cols].describe().T
    print(f"  {'Column':<35} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'-' * 77}")
    for col in feature_cols:
        row = feat_stats.loc[col]
        print(f"  {col:<35} {row['mean']:>10.3f} {row['std']:>10.3f} "
              f"{row['min']:>10.3f} {row['max']:>10.3f}")


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"02_datasplit_analysis_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    # -- Header --
    print("=" * 80)
    print("  02_DATASPLIT_ANALYSIS -- POST-SPLIT DATA QUALITY REPORT")
    print("=" * 80)
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data path:       {SPLIT_DATA_PATH}")
    print(f"  Log path:        {LOG_PATH}")
    print(f"  CV folds (k):    {N_CV_FOLDS}")
    print(f"  Held-out bucket: {HELD_OUT_INDEX}")
    print("=" * 80)

    # -- Load --
    print(f"\nLoading split data from: {SPLIT_DATA_PATH}")
    df = pd.read_csv(SPLIT_DATA_PATH)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # -- Fold validation --
    if not validate_folds(df):
        logger.close()
        sys.exit(1)

    # -- Label distribution --
    report_label_distribution(df)

    # -- Feature columns --
    feature_cols = identify_features(df)

    # -- NaN / Inf --
    report_nan(df, feature_cols)

    # -- Constant columns --
    report_constant_columns(df, feature_cols)

    # -- Summary stats --
    report_feature_summary(df, feature_cols)

    # -- Footer --
    n_features = len(feature_cols)
    print(f"\n{'=' * 80}")
    print(f"DATA QUALITY REPORT COMPLETE")
    print(f"  {n_features} numeric features, {N_CV_FOLDS} CV folds + 1 held-out")
    print(f"  Log saved: {LOG_PATH}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()