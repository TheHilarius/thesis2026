"""
02_datasplit_analysis.py
Post-split data quality report: label balance, feature dtypes,
NaN / Inf audit (with per-label breakdown), constant-column check,
positional AA column audit, and summary statistics.
Run AFTER 01_datasplit.py and BEFORE cross-validation.

Usage:
    python src/investigation/02_datasplit_analysis.py
    python src/investigation/02_datasplit_analysis.py \
        --split data/processed/df_all_with_folds_h2.csv \
        --cluster-map data/processed/cluster_map_peptide_h2.csv \
        --audit-out results/tables/fold_audit_cluster_h2.csv --tag h2
"""

import argparse
import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config import (
    DATA_DIR, SPLIT_DATA_PATH, LOG_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    LABEL_COL, FOLD_COL, PEPTIDE_COL,
    METADATA_COLS, POSITION_AA_COLS,
    get_feature_cols, validate_config,
)

CLUSTER_MAP_PATH = DATA_DIR / "cluster_map_peptide.csv"
FOLD_AUDIT_OUT = "results/tables/fold_audit_cluster.csv"


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
# 10. FOLD-DIFFICULTY AUDIT (cluster-preserving permutation null)
# ──────────────────────────────────────────────

def load_cluster_map():
    """
    Load the peptide -> homology-cluster map built by 29_build_cluster_map.py.
    Returns None (with instructions) if the file does not exist yet.
    """
    if not CLUSTER_MAP_PATH.exists():
        print(f"\n{'=' * 80}")
        print("FOLD-DIFFICULTY AUDIT -- SKIPPED")
        print("=" * 80)
        print(f"  Cluster map not found: {CLUSTER_MAP_PATH}")
        print(f"  Build it once with:")
        print(f"    python src/investigation/29_build_cluster_map.py")
        print(f"  (for a non-default split, also pass --split and the matching")
        print(f"   --hamming-cutoff, then point --cluster-map here)")
        return None
    cmap = pd.read_csv(CLUSTER_MAP_PATH)
    print(f"\n  Loaded cluster map: {len(cmap)} peptides, "
          f"{cmap['cluster_id'].nunique()} clusters")
    return cmap


def _fold_stats_from_assignment(assign_idx, folds, w_rows, w_pos, w_size):
    """
    Compute per-fold stats for one cluster->fold assignment.

    assign_idx : int array, fold position (0..len(folds)-1) per cluster
    Returns dict stat -> array over folds.
    """
    k = len(folds)
    rows = np.bincount(assign_idx, weights=w_rows, minlength=k)
    pos = np.bincount(assign_idx, weights=w_pos, minlength=k)
    ncl = np.bincount(assign_idx, minlength=k).astype(float)
    upep = np.bincount(assign_idx, weights=w_size, minlength=k)
    largest = np.zeros(k)
    for fi in range(k):
        m = assign_idx == fi
        largest[fi] = w_rows[m].max() if m.any() else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = {
            "rows": rows,
            "pos_pct": np.where(rows > 0, pos / rows * 100, np.nan),
            "dup_frac": np.where(rows > 0, 1.0 - upep / rows, np.nan),
            "n_clusters": ncl,
            "mean_cl_size": np.where(ncl > 0, upep / ncl, np.nan),
            "largest_share": np.where(rows > 0, largest / rows * 100, np.nan),
        }
    return stats


def report_fold_difficulty_audit(df, cmap, n_perm=1000, seed=42):
    """
    Per-fold difficulty audit with a cluster-preserving permutation null.

    Null: each homology cluster is independently re-dealt to a CV fold with
    probability = that fold's observed row share (so expected fold sizes
    match). Whole clusters move together, preserving homology structure.
    Held-out fold is reported observed-only and excluded from the null.

    Writes FOLD_AUDIT_OUT (results/tables/fold_audit_cluster.csv).
    """
    print(f"\n{'=' * 80}")
    print("FOLD-DIFFICULTY AUDIT (cluster-preserving permutation null)")
    print("=" * 80)

    cv_folds = [f for f in range(N_CV_FOLDS) if f != HELD_OUT_INDEX]
    k = len(cv_folds)
    fold_pos = {f: i for i, f in enumerate(cv_folds)}

    # -- Per-cluster table (CV folds only for the null) --
    rows = df[[PEPTIDE_COL, LABEL_COL, FOLD_COL]].merge(
        cmap[["peptide", "cluster_id", "cluster_size"]],
        left_on=PEPTIDE_COL, right_on="peptide", how="left",
    )
    if rows["cluster_id"].isna().any():
        missing = rows.loc[rows["cluster_id"].isna(), PEPTIDE_COL].nunique()
        print(f"  WARNING: {missing} peptides in split data have no cluster entry "
              f"(stale map?). Re-run 29_build_cluster_map.py.")
        rows = rows.dropna(subset=["cluster_id"])
    rows["cluster_id"] = rows["cluster_id"].astype(int)

    cv = rows[rows[FOLD_COL].isin(cv_folds)].copy()
    g = cv.groupby("cluster_id")
    cl = pd.DataFrame({
        "cluster_rows": g.size(),
        "cluster_pos": g[LABEL_COL].sum(),
        "cluster_size": g["cluster_size"].first(),
        "fold": g[FOLD_COL].first(),
    }).reset_index()

    w_rows = cl["cluster_rows"].to_numpy(dtype=float)
    w_pos = cl["cluster_pos"].to_numpy(dtype=float)
    w_size = cl["cluster_size"].to_numpy(dtype=float)
    obs_assign = cl["fold"].map(fold_pos).to_numpy(dtype=int)

    obs = _fold_stats_from_assignment(obs_assign, cv_folds, w_rows, w_pos, w_size)

    # -- Null --
    print(f"  Permuting {len(cl)} clusters over {k} folds x {n_perm} reps "
          f"(seed={seed}) ...")
    rng = np.random.default_rng(seed)
    shares = np.array([w_rows[obs_assign == fi].sum() for fi in range(k)])
    shares = shares / shares.sum()
    stat_names = list(obs.keys())
    nulls = {s: np.empty((n_perm, k)) for s in stat_names}
    for r in range(n_perm):
        a = rng.choice(k, p=shares, size=len(cl))
        st = _fold_stats_from_assignment(a, cv_folds, w_rows, w_pos, w_size)
        for s in stat_names:
            nulls[s][r] = st[s]
    lo = {s: np.nanpercentile(nulls[s], 2.5, axis=0) for s in stat_names}
    hi = {s: np.nanpercentile(nulls[s], 97.5, axis=0) for s in stat_names}

    # -- Report --
    print(f"\n  {'Fold':<6} {'Stat':<13} {'Observed':>10} "
          f"{'Null 95%':>22}  Flag")
    print(f"  {'-' * 62}")
    out_records = []
    n_out = 0
    for s in stat_names:
        for fi, f in enumerate(cv_folds):
            o, l, h = obs[s][fi], lo[s][fi], hi[s][fi]
            flag = "OUTLIER" if (o < l or o > h) else "ok"
            if flag == "OUTLIER":
                n_out += 1
            print(f"  {f:<6} {s:<13} {o:>10.2f} "
                  f"[{l:>8.2f}, {h:>8.2f}]  {flag}")
            out_records.append({"fold": f, "role": f"CV fold {f}", "stat": s,
                                "observed": o, "null_lo": l, "null_hi": h,
                                "flag": flag})

    # -- Held-out, observed only --
    ho = df[df[FOLD_COL] == HELD_OUT_INDEX]
    ho_upep = ho[PEPTIDE_COL].nunique()
    ho_pos = int((ho[LABEL_COL] == 1).sum())
    ho_stats = {"rows": float(len(ho)),
                "pos_pct": ho_pos / len(ho) * 100 if len(ho) else float("nan"),
                "dup_frac": 1.0 - ho_upep / len(ho) if len(ho) else float("nan"),
                "n_clusters": float(cmap[cmap["fold"] == HELD_OUT_INDEX]
                                     ["cluster_id"].nunique()),
                "mean_cl_size": float("nan"), "largest_share": float("nan")}
    for s in stat_names:
        out_records.append({"fold": HELD_OUT_INDEX, "role": "Held-out",
                            "stat": s, "observed": ho_stats[s],
                            "null_lo": float("nan"), "null_hi": float("nan"),
                            "flag": "n/a (held-out)"})
    print(f"\n  Held-out fold {HELD_OUT_INDEX} (observed only, excluded from null): "
          f"rows={len(ho)}, pos%={ho_stats['pos_pct']:.1f}, "
          f"dup_frac={ho_stats['dup_frac']:.3f}, "
          f"n_clusters={ho_stats['n_clusters']:.0f}")

    # -- Save --
    audit_df = pd.DataFrame(out_records)
    out_path = Path(FOLD_AUDIT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    if n_out == 0:
        print("  Verdict: all CV-fold stats inside null 95% intervals "
              "-- no fold stands out (H1 not supported by size/composition).")
    else:
        print(f"  Verdict: {n_out} fold-stat(s) outside null 95% intervals "
              f"-- inspect flagged cells above.")
    return audit_df


# ──────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Post-split data quality report + fold-difficulty audit.")
    ap.add_argument("--split", default=None,
                    help="split CSV path (default: config SPLIT_DATA_PATH), "
                         "e.g. data/processed/df_all_with_folds_h2.csv")
    ap.add_argument("--cluster-map", default=None,
                    help="cluster map from 29_build_cluster_map.py "
                         "(default: data/processed/cluster_map_peptide.csv). "
                         "Must match the split's Hamming cutoff.")
    ap.add_argument("--audit-out", default=None,
                    help="fold-audit CSV output path "
                         "(default: results/tables/fold_audit_cluster.csv)")
    ap.add_argument("--tag", default="",
                    help="suffix for the log filename, e.g. 'h2'")
    args = ap.parse_args()

    split_path = Path(args.split) if args.split else Path(SPLIT_DATA_PATH)
    if args.cluster_map:
        CLUSTER_MAP_PATH = Path(args.cluster_map)
    if args.audit_out:
        FOLD_AUDIT_OUT = str(args.audit_out)
    log_tag = f"_{args.tag}" if args.tag else ""

    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"02_datasplit_analysis_log_{timestamp}{log_tag}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    # -- Header --
    print("=" * 80)
    print("  02_DATASPLIT_ANALYSIS -- POST-SPLIT DATA QUALITY REPORT")
    print("=" * 80)
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data path:       {split_path}")
    print(f"  Log path:        {LOG_PATH}")
    print(f"  CV folds (k):    {N_CV_FOLDS}")
    print(f"  Held-out bucket: {HELD_OUT_INDEX}")
    print("=" * 80)

    # -- Load --
    print(f"\nLoading split data from: {split_path}")
    df = pd.read_csv(split_path)
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

    # -- Fold-difficulty audit (needs 29_build_cluster_map.py output) --
    cmap = load_cluster_map()
    if cmap is not None:
        report_fold_difficulty_audit(df, cmap)

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