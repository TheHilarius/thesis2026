"""
02_datasplit_analysis.py
Post-split data quality report: label balance, feature dtypes,
NaN / Inf audit (with per-label breakdown), constant-column check,
positional AA column audit, and summary statistics.
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
    METADATA_COLS, POSITION_AA_COLS,
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
# 1. FOLD VALIDATION
# ──────────────────────────────────────────────

def validate_folds(df):
    """Check that the fold column exists and contains the expected buckets."""
    if FOLD_COL not in df.columns:
        print(f"FATAL: Column '{FOLD_COL}' not found. Run 01_datasplit.py first.")
        return False

    actual_folds = sorted(df[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))
    if actual_folds != expected_folds:
        print(f"FATAL: Expected folds {expected_folds}, found {actual_folds}")
        print(f"  Check that config.N_CV_FOLDS={N_CV_FOLDS} matches 01_datasplit.py")
        return False

    print(f"[OK] Fold column validated: {actual_folds}")

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
# 3. COLUMN INVENTORY
# ──────────────────────────────────────────────

def report_column_inventory(df):
    """
    Print a complete column inventory grouped by role:
    metadata, positional AA (unencoded), and numeric features.
    """
    print(f"\n{'=' * 80}")
    print("COLUMN INVENTORY")
    print("=" * 80)

    present_meta = [c for c in METADATA_COLS if c in df.columns]
    missing_meta = [c for c in METADATA_COLS if c not in df.columns]
    present_pos = [c for c in POSITION_AA_COLS if c in df.columns]
    missing_pos = [c for c in POSITION_AA_COLS if c not in df.columns]

    print(f"\n  Metadata columns ({len(present_meta)} present"
          f"{f', {len(missing_meta)} missing' if missing_meta else ''}):")
    for c in present_meta:
        print(f"    {c:<40} dtype: {df[c].dtype}")
    if missing_meta:
        print(f"    MISSING: {missing_meta}")

    print(f"\n  Positional AA columns ({len(present_pos)} present"
          f"{f', {len(missing_pos)} missing' if missing_pos else ''}):")
    for c in present_pos:
        n_unique = df[c].nunique(dropna=False)
        print(f"    {c:<40} dtype: {df[c].dtype}  unique: {n_unique}")
    if missing_pos:
        print(f"    MISSING: {missing_pos}")

    # Anything in the dataframe that isn't metadata, positional, or fold
    all_known = set(METADATA_COLS) | set(POSITION_AA_COLS)
    unexpected = [c for c in df.columns if c not in all_known
                  and c not in get_feature_cols(df.columns)]
    if unexpected:
        print(f"\n  Unclassified columns ({len(unexpected)}):")
        for c in unexpected:
            print(f"    {c:<40} dtype: {df[c].dtype}")


# ──────────────────────────────────────────────
# 4. FEATURE-COLUMN IDENTIFICATION
# ──────────────────────────────────────────────

def identify_features(df):
    """
    Resolve feature columns, flag non-numeric ones, and return the
    clean list of numeric feature column names.
    """
    feature_cols = get_feature_cols(df.columns)
    print(f"\nNumeric feature columns identified ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  [{i:>3}] {col:<40} dtype: {df[col].dtype}")

    non_numeric = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        print(f"\nWARNING: Non-numeric feature columns ({len(non_numeric)}):")
        for c in non_numeric:
            print(f"  - {c} (dtype: {df[c].dtype})")
        feature_cols = [c for c in feature_cols if c not in non_numeric]
        print(f"  Remaining numeric feature columns: {len(feature_cols)}")

    return feature_cols


# ──────────────────────────────────────────────
# 5. POSITIONAL AA COLUMN AUDIT
# ──────────────────────────────────────────────

def report_position_aa_columns(df):
    """
    Audit the 17 single-residue positional columns:
    unique values, gap/padding characters, and NaN counts.
    """
    present = [c for c in POSITION_AA_COLS if c in df.columns]
    if not present:
        print(f"\n[SKIP] No positional AA columns found in dataframe")
        return

    print(f"\n{'=' * 80}")
    print("POSITIONAL AA COLUMN AUDIT")
    print("=" * 80)

    canonical = set("ACDEFGHIKLMNPQRSTVWY")

    print(f"\n  {'Position':<10} {'NaN':>6} {'NaN+ ':>6} {'NaN- ':>6} "
          f"{'Unique':>7} {'Non-canonical':>15}")
    print(f"  {'-' * 60}")

    for col in present:
        n_nan = df[col].isna().sum()
        nan_mask = df[col].isna()
        n_nan_pos = int((nan_mask & (df[LABEL_COL] == 1)).sum())
        n_nan_neg = int((nan_mask & (df[LABEL_COL] == 0)).sum())
        n_unique = df[col].nunique(dropna=True)

        vals = set(df[col].dropna().unique())
        non_canon = vals - canonical
        non_canon_str = ", ".join(sorted(non_canon)) if non_canon else "-"

        print(f"  {col:<10} {n_nan:>6} {n_nan_pos:>6} {n_nan_neg:>6} "
              f"{n_unique:>7} {non_canon_str:>15}")

    # Overall vocabulary
    all_values = set()
    for col in present:
        all_values.update(df[col].dropna().unique())
    non_canon_all = all_values - canonical
    print(f"\n  Global AA vocabulary: {sorted(all_values)}")
    if non_canon_all:
        print(f"  Non-canonical residues: {sorted(non_canon_all)}")
        print(f"  (These may represent gaps/padding at protein termini)")
    else:
        print(f"  [OK] All residues are canonical amino acids")


# ──────────────────────────────────────────────
# 6. NaN / Inf REPORT (with per-label breakdown)
# ──────────────────────────────────────────────

def report_nan(df, feature_cols):
    """Per-column NaN and Inf audit with positive/negative label breakdown."""
    pos_mask = df[LABEL_COL] == 1
    neg_mask = df[LABEL_COL] == 0
    n_pos_total = int(pos_mask.sum())
    n_neg_total = int(neg_mask.sum())

    # ── NaN ──
    print(f"\nNaN REPORT (numeric features):")
    print(f"  {'Column':<35} {'NaN':>6} {'%':>6}  "
          f"{'NaN+':>6} {'%+':>6}  {'NaN-':>6} {'%-':>6}")
    print(f"  {'-' * 83}")

    total_nan = 0
    cols_with_nan = []

    for col in feature_cols:
        nan_mask = df[col].isna()
        n_nan = int(nan_mask.sum())
        if n_nan == 0:
            continue

        n_nan_pos = int((nan_mask & pos_mask).sum())
        n_nan_neg = int((nan_mask & neg_mask).sum())
        pct = n_nan / len(df) * 100
        pct_pos = n_nan_pos / n_pos_total * 100 if n_pos_total > 0 else 0.0
        pct_neg = n_nan_neg / n_neg_total * 100 if n_neg_total > 0 else 0.0

        print(f"  {col:<35} {n_nan:>6} {pct:>5.1f}%  "
              f"{n_nan_pos:>6} {pct_pos:>5.1f}%  {n_nan_neg:>6} {pct_neg:>5.1f}%")
        total_nan += n_nan
        cols_with_nan.append(col)

    if total_nan == 0:
        print("  No NaN values found in any numeric feature column")
    else:
        print(f"  {'-' * 83}")
        print(f"  TOTAL: {total_nan} NaN values across {len(cols_with_nan)} column(s)")

    # ── Inf ──
    print(f"\nInf REPORT (numeric features):")
    print(f"  {'Column':<35} {'Inf':>6} {'%':>6}  "
          f"{'Inf+':>6} {'%+':>6}  {'Inf-':>6} {'%-':>6}")
    print(f"  {'-' * 83}")

    total_inf = 0
    cols_with_inf = []

    for col in feature_cols:
        arr = df[col].values
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        inf_mask = np.isinf(arr)
        n_inf = int(inf_mask.sum())
        if n_inf == 0:
            continue

        n_inf_pos = int((inf_mask & pos_mask.values).sum())
        n_inf_neg = int((inf_mask & neg_mask.values).sum())
        pct = n_inf / len(df) * 100
        pct_pos = n_inf_pos / n_pos_total * 100 if n_pos_total > 0 else 0.0
        pct_neg = n_inf_neg / n_neg_total * 100 if n_neg_total > 0 else 0.0

        print(f"  {col:<35} {n_inf:>6} {pct:>5.1f}%  "
              f"{n_inf_pos:>6} {pct_pos:>5.1f}%  {n_inf_neg:>6} {pct_neg:>5.1f}%")
        total_inf += n_inf
        cols_with_inf.append(col)

    if total_inf == 0:
        print("  No Inf values found in any numeric feature column")
    else:
        print(f"  {'-' * 83}")
        print(f"  TOTAL: {total_inf} Inf values across {len(cols_with_inf)} column(s)")

    # ── Label-bias check ──
    if total_nan > 0 or total_inf > 0:
        print(f"\n  LABEL-BIAS NOTE:")
        print(f"  If NaN/Inf values are concentrated in one label class,")
        print(f"  median imputation may introduce information leakage.")
        print(f"  Baseline counts — positives: {n_pos_total}, negatives: {n_neg_total}")


# ──────────────────────────────────────────────
# 7. CONSTANT-COLUMN CHECK
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
# 8. SUMMARY STATISTICS
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
# 9. PER-FOLD MISSING-VALUE REPORT
# ──────────────────────────────────────────────

def report_nan_per_fold(df, feature_cols):
    """
    For each fold, report the total number of NaN values across
    feature columns, split by label.
    """
    folds = sorted(df[FOLD_COL].unique())

    print(f"\n{'=' * 80}")
    print("NaN COUNTS PER FOLD (numeric features)")
    print("=" * 80)
    print(f"  {'Fold':<8} {'Role':<22} {'NaN total':>10} "
          f"{'NaN+ ':>10} {'NaN- ':>10}")
    print(f"  {'-' * 62}")

    for fold in folds:
        fold_df = df[df[FOLD_COL] == fold]
        nan_total = int(fold_df[feature_cols].isna().sum().sum())
        nan_pos = int(fold_df.loc[fold_df[LABEL_COL] == 1, feature_cols].isna().sum().sum())
        nan_neg = int(fold_df.loc[fold_df[LABEL_COL] == 0, feature_cols].isna().sum().sum())
        role = "Held-out" if fold == HELD_OUT_INDEX else f"CV fold {fold}"
        print(f"  {fold:<8} {role:<22} {nan_total:>10} "
              f"{nan_pos:>10} {nan_neg:>10}")


# ──────────────────────────────────────────────
# 10. MAIN
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

    # -- Column inventory --
    report_column_inventory(df)

    # -- Fold validation --
    if not validate_folds(df):
        logger.close()
        sys.exit(1)

    # -- Label distribution --
    report_label_distribution(df)

    # -- Feature columns --
    feature_cols = identify_features(df)

    # -- Positional AA audit --
    report_position_aa_columns(df)

    # -- NaN / Inf (with label breakdown) --
    report_nan(df, feature_cols)

    # -- NaN per fold --
    report_nan_per_fold(df, feature_cols)

    # -- Constant columns --
    report_constant_columns(df, feature_cols)

    # -- Summary stats --
    report_feature_summary(df, feature_cols)

    # -- Footer --
    n_features = len(feature_cols)
    n_pos_cols = len([c for c in POSITION_AA_COLS if c in df.columns])
    print(f"\n{'=' * 80}")
    print(f"DATA QUALITY REPORT COMPLETE")
    print(f"  {n_features} numeric features")
    print(f"  {n_pos_cols} positional AA columns (pending encoding)")
    print(f"  {N_CV_FOLDS} CV folds + 1 held-out")
    print(f"  Log saved: {LOG_PATH}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()