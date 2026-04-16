"""
01_datasplit.py
Homology-based data splitting for MHC-I processing prediction.
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import time

from config import (
    DATA_DIR, RAW_DATA_PATH, SPLIT_DATA_PATH, LOG_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX, N_BUCKETS, RANDOM_STATE,
    PEPTIDE_COL, LABEL_COL, FOLD_COL, HAMMING_CUTOFF,
    validate_config,
)


# ──────────────────────────────────────────────
# 0. DUAL LOGGER
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
# 1. UNION-FIND DATA STRUCTURE
# ──────────────────────────────────────────────

class UnionFind:
    """Disjoint Set Union with path compression and union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# ──────────────────────────────────────────────
# 2. HOMOLOGY CLUSTERING
# ──────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def cluster_by_hamming(peptides, max_dist=1):
    """
    Cluster unique peptides by Hamming distance.

    Identical peptides are the same node (Hamming distance = 0),
    so duplicate peptide sequences from different proteins are
    trivially in the same cluster.
    """
    uf = UnionFind()
    peptide_set = set(peptides)

    for pep in peptides:
        uf.find(pep)
        for pos in range(len(pep)):
            for aa in AMINO_ACIDS:
                if aa != pep[pos]:
                    neighbour = pep[:pos] + aa + pep[pos + 1:]
                    if neighbour in peptide_set:
                        uf.union(pep, neighbour)

    clusters = defaultdict(set)
    for pep in peptides:
        clusters[uf.find(pep)].add(pep)

    return dict(clusters)


# ──────────────────────────────────────────────
# 3. BALANCED PARTITION
# ──────────────────────────────────────────────

def greedy_balanced_partition(clusters, n_buckets, peptide_sample_counts):
    """
    Greedy bin-packing: assign each cluster to the currently smallest bucket.

    peptide_sample_counts includes ALL rows for each peptide
    (including duplicates from different proteins), so the balancing
    accounts for the true number of data rows, not just unique peptides.
    """
    cluster_info = []
    for root, peps in clusters.items():
        total = sum(peptide_sample_counts.get(p, 0) for p in peps)
        cluster_info.append((root, peps, total))

    cluster_info.sort(key=lambda x: x[2], reverse=True)

    bucket_totals = np.zeros(n_buckets, dtype=int)
    peptide_to_bucket = {}

    for root, peps, size in cluster_info:
        target = int(np.argmin(bucket_totals))
        for p in peps:
            peptide_to_bucket[p] = target
        bucket_totals[target] += size

    return peptide_to_bucket, bucket_totals.tolist()


# ──────────────────────────────────────────────
# 4. VERIFICATION
# ──────────────────────────────────────────────

def verify_no_homology_leakage(df, peptide_col, fold_col):
    """Check that no two peptides in different folds have Hamming distance <= 1."""
    folds = sorted(df[fold_col].unique())
    fold_peptides = {
        f: set(df.loc[df[fold_col] == f, peptide_col].unique()) for f in folds
    }

    violations = 0
    for f1, f2 in combinations(folds, 2):
        set2 = fold_peptides[f2]
        for pep in fold_peptides[f1]:
            for pos in range(len(pep)):
                for aa in AMINO_ACIDS:
                    if aa != pep[pos]:
                        neighbour = pep[:pos] + aa + pep[pos + 1:]
                        if neighbour in set2:
                            violations += 1

    if violations == 0:
        print("No homology leakage detected across folds.")
    else:
        print(f"{violations} cross-fold homologous pairs found!")

    return violations


def verify_no_peptide_split(df, peptide_col, fold_col):
    """
    Verify that every row with the same peptide sequence is in the same fold.
    This catches the case where identical peptides from different proteins
    might have been accidentally split across folds.
    """
    peptide_folds = df.groupby(peptide_col)[fold_col].nunique()
    split_peptides = peptide_folds[peptide_folds > 1]

    if len(split_peptides) == 0:
        print("No peptide split across folds (all duplicates in same fold).")
    else:
        print(f"WARNING: {len(split_peptides)} peptides split across folds!")
        for pep, n_folds in split_peptides.head(10).items():
            folds_in = df.loc[df[peptide_col] == pep, fold_col].unique()
            print(f"  {pep}: appears in folds {sorted(folds_in)}")

    return len(split_peptides)


# ──────────────────────────────────────────────
# 5. DUPLICATE PEPTIDE ANALYSIS
# ──────────────────────────────────────────────

def analyze_duplicate_peptides(df, peptide_col, label_col):
    """
    Report on peptides that appear multiple times in the dataset,
    potentially from different source proteins.
    """
    print("\n" + "=" * 80)
    print("DUPLICATE PEPTIDE ANALYSIS")
    print("=" * 80)

    total_rows = len(df)
    unique_peptides = df[peptide_col].nunique()
    duplicate_rows = total_rows - unique_peptides

    print(f"  Total rows:        {total_rows}")
    print(f"  Unique peptides:   {unique_peptides}")
    print(f"  Duplicate rows:    {duplicate_rows} "
          f"({duplicate_rows / total_rows * 100:.1f}% of all rows)")

    # Count how many times each peptide appears
    pep_counts = df[peptide_col].value_counts()
    multi_occurrence = pep_counts[pep_counts > 1]

    if len(multi_occurrence) == 0:
        print("  No peptides appear more than once.")
        return

    print(f"\n  Peptides appearing more than once: {len(multi_occurrence)}")
    print(f"  Total rows from multi-occurrence peptides: {multi_occurrence.sum()}")

    # Distribution of occurrence counts
    print(f"\n  Occurrence count distribution:")
    print(f"  {'Times seen':<15} {'Peptides':>10} {'Total rows':>12}")
    print(f"  {'-' * 40}")
    for count_val in sorted(pep_counts.unique()):
        n_peps = (pep_counts == count_val).sum()
        n_rows = n_peps * count_val
        if n_peps > 0:
            print(f"  {count_val:<15} {n_peps:>10} {n_rows:>12}")

    # If uniprot_id column exists, show protein overlap
    if "uniprot_id" in df.columns:
        print(f"\n  Peptide-protein relationships:")
        pep_proteins = df.groupby(peptide_col)["uniprot_id"].nunique()
        multi_protein = pep_proteins[pep_proteins > 1]

        print(f"  Peptides in exactly 1 protein:   {(pep_proteins == 1).sum()}")
        print(f"  Peptides in multiple proteins:    {len(multi_protein)}")

        if len(multi_protein) > 0:
            print(f"\n  Top 10 peptides by number of source proteins:")
            print(f"  {'Peptide':<15} {'Proteins':>10} {'Total rows':>12} {'Label dist':>15}")
            print(f"  {'-' * 55}")
            for pep in multi_protein.nlargest(10).index:
                n_prots = multi_protein[pep]
                pep_rows = df[df[peptide_col] == pep]
                n_rows = len(pep_rows)
                n_pos = (pep_rows[label_col] == 1).sum()
                n_neg = (pep_rows[label_col] == 0).sum()
                print(f"  {pep:<15} {n_prots:>10} {n_rows:>12} "
                      f"  pos={n_pos}, neg={n_neg}")

    # Check label consistency for duplicate peptides
    print(f"\n  Label consistency for duplicate peptides:")
    pep_label_nunique = df.groupby(peptide_col)[label_col].nunique()
    mixed_label = pep_label_nunique[pep_label_nunique > 1]
    print(f"  Peptides with consistent label:  {(pep_label_nunique == 1).sum()}")
    print(f"  Peptides with mixed labels:      {len(mixed_label)}")

    if len(mixed_label) > 0:
        print(f"\n  NOTE: {len(mixed_label)} peptides have both positive and negative labels.")
        print(f"  This can happen when the same peptide is processed in one protein")
        print(f"  context but not another. Examples:")
        for pep in list(mixed_label.index)[:5]:
            pep_rows = df[df[peptide_col] == pep]
            n_pos = (pep_rows[label_col] == 1).sum()
            n_neg = (pep_rows[label_col] == 0).sum()
            if "uniprot_id" in df.columns:
                prots = pep_rows["uniprot_id"].unique()
                print(f"    {pep}: pos={n_pos}, neg={n_neg}, "
                      f"proteins={list(prots)}")
            else:
                print(f"    {pep}: pos={n_pos}, neg={n_neg}")


# ──────────────────────────────────────────────
# 6. STATISTICS
# ──────────────────────────────────────────────

def print_bucket_statistics(df, peptide_col, label_col, fold_col,
                            n_cv_folds, held_out_index):
    folds = sorted(df[fold_col].unique())

    print("\n" + "=" * 80)
    print("BUCKET OVERVIEW")
    print("=" * 80)
    header = (f"{'Bucket':<10} {'Role':<22} {'Total':>7} {'Pos':>7} "
              f"{'Neg':>7} {'Pos%':>7} {'Neg:Pos':>9} {'Uniq Pep':>10}")
    print(header)
    print("-" * 80)

    bucket_stats = {}
    for i in folds:
        fold_df = df[df[fold_col] == i]
        n_total = len(fold_df)
        n_pos = int((fold_df[label_col] == 1).sum())
        n_neg = int((fold_df[label_col] == 0).sum())
        n_unique = fold_df[peptide_col].nunique()
        pos_pct = n_pos / n_total * 100 if n_total > 0 else 0
        ratio = n_neg / n_pos if n_pos > 0 else float("inf")
        role = "Held-out validation" if i == held_out_index else f"CV fold {i}"

        bucket_stats[i] = {
            "role": role, "total": n_total, "pos": n_pos,
            "neg": n_neg, "unique": n_unique, "pos_pct": pos_pct, "ratio": ratio,
        }

        print(f"{i:<10} {role:<22} {n_total:>7} {n_pos:>7} "
              f"{n_neg:>7} {pos_pct:>6.1f}% {ratio:>9.3f} {n_unique:>10}")

    total_pos = sum(s["pos"] for s in bucket_stats.values())
    total_neg = sum(s["neg"] for s in bucket_stats.values())
    total_all = sum(s["total"] for s in bucket_stats.values())
    print("-" * 80)
    global_ratio = total_neg / total_pos if total_pos > 0 else float("inf")
    print(f"{'TOTAL':<10} {'':<22} {total_all:>7} {total_pos:>7} "
          f"{total_neg:>7} {total_pos / total_all * 100:>6.1f}% "
          f"{global_ratio:>9.3f}")

    print("\n" + "=" * 80)
    print("CROSS-BUCKET COMPARISON")
    print("=" * 80)

    ratios = [s["ratio"] for s in bucket_stats.values()]
    sizes = [s["total"] for s in bucket_stats.values()]
    pos_pcts = [s["pos_pct"] for s in bucket_stats.values()]

    print(f"  Neg:Pos ratio   -- min: {min(ratios):.3f}  max: {max(ratios):.3f}  "
          f"std: {np.std(ratios):.4f}")
    print(f"  Bucket sizes    -- min: {min(sizes)}  max: {max(sizes)}  "
          f"std: {np.std(sizes):.1f}")
    print(f"  Positive %      -- min: {min(pos_pcts):.1f}%  max: {max(pos_pcts):.1f}%  "
          f"std: {np.std(pos_pcts):.2f}%")

    cv_stats = {k: v for k, v in bucket_stats.items() if k != held_out_index}
    cv_ratios = [v["ratio"] for v in cv_stats.values()]
    cv_sizes = [v["total"] for v in cv_stats.values()]

    print(f"\n  CV folds only (excl. held-out):")
    print(f"    Size range:     {min(cv_sizes)} -- {max(cv_sizes)}  "
          f"(ideal = {sum(cv_sizes) // len(cv_sizes)})")
    print(f"    Ratio range:    {min(cv_ratios):.3f} -- {max(cv_ratios):.3f}")

    ho = bucket_stats.get(held_out_index, {})
    if ho:
        print(f"\n  Held-out set:")
        print(f"    Samples:        {ho['total']}  "
              f"({ho['total'] / total_all * 100:.1f}% of all data)")
        print(f"    Neg:Pos ratio:  {ho['ratio']:.3f}")

    return bucket_stats


def inspect_bucket_clusters(df, clusters, peptide_to_bucket,
                            peptide_col, label_col, fold_col,
                            n_cv_folds, held_out_index):
    bucket_clusters = defaultdict(list)

    for root, peps in clusters.items():
        representative = next(iter(peps))
        bucket_id = peptide_to_bucket[representative]
        bucket_clusters[bucket_id].append((root, peps, len(peps)))

    folds = sorted(df[fold_col].unique())

    print("\n" + "=" * 80)
    print("CLUSTER DISTRIBUTION PER BUCKET")
    print("=" * 80)

    for i in folds:
        role = "Held-out validation" if i == held_out_index else f"CV fold {i}"
        cls_list = bucket_clusters.get(i, [])
        sizes = [s for _, _, s in cls_list]

        if not sizes:
            print(f"\n-- Bucket {i} ({role}): EMPTY --")
            continue

        n_clusters = len(sizes)
        n_singletons = sum(1 for s in sizes if s == 1)
        n_small = sum(1 for s in sizes if 2 <= s <= 5)
        n_medium = sum(1 for s in sizes if 6 <= s <= 20)
        n_large = sum(1 for s in sizes if s > 20)
        max_size = max(sizes)
        mean_size = np.mean(sizes)
        median_size = np.median(sizes)
        total_peptides_in_bucket = sum(sizes)

        print(f"\n-- Bucket {i} ({role}) --")
        print(f"  Total clusters:      {n_clusters}")
        print(f"  Total unique peps:   {total_peptides_in_bucket}")
        print(f"  Cluster size stats:  mean={mean_size:.1f}  "
              f"median={median_size:.1f}  max={max_size}")
        print(f"  Singletons:          {n_singletons:>6}  "
              f"({n_singletons / n_clusters * 100:.1f}% of clusters, "
              f"{n_singletons / total_peptides_in_bucket * 100:.1f}% of peptides)")
        print(f"  Small (2-5):         {n_small:>6}  "
              f"({n_small / n_clusters * 100:.1f}% of clusters)")
        print(f"  Medium (6-20):       {n_medium:>6}  "
              f"({n_medium / n_clusters * 100:.1f}% of clusters)")
        print(f"  Large (>20):         {n_large:>6}  "
              f"({n_large / n_clusters * 100:.1f}% of clusters)")

        top_clusters = sorted(cls_list, key=lambda x: x[2], reverse=True)[:5]
        print(f"\n  Top 5 largest clusters:")
        for rnk, (root, peps, size) in enumerate(top_clusters, 1):
            pep_list = sorted(peps)
            shown = pep_list[:6]
            suffix = f" ... +{size - 6} more" if size > 6 else ""
            cluster_rows = df[df[peptide_col].isin(peps)]
            c_pos = (cluster_rows[label_col] == 1).sum()
            c_neg = (cluster_rows[label_col] == 0).sum()
            print(f"    #{rnk} size={size:>4}  (pos={c_pos}, neg={c_neg})  "
                  f"peptides: {', '.join(shown)}{suffix}")

    all_sizes = [len(peps) for peps in clusters.values()]
    print(f"\n{'=' * 80}")
    print("GLOBAL CLUSTER SIZE HISTOGRAM")
    print("=" * 80)

    bins = [(1, 1, "Singletons"), (2, 2, "Pairs"), (3, 5, "3-5"),
            (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"),
            (51, 100, "51-100"), (101, float("inf"), ">100")]

    print(f"  {'Size range':<15} {'Clusters':>10} {'%':>8} {'Peptides':>10} {'%':>8}")
    print(f"  {'-' * 55}")
    total_clusters = len(all_sizes)
    total_peptides = sum(all_sizes)

    for lo, hi, label in bins:
        if hi == float("inf"):
            count = sum(1 for s in all_sizes if s >= lo)
            pep_count = sum(s for s in all_sizes if s >= lo)
        else:
            count = sum(1 for s in all_sizes if lo <= s <= hi)
            pep_count = sum(s for s in all_sizes if lo <= s <= hi)
        if count > 0:
            print(f"  {label:<15} {count:>10} {count / total_clusters * 100:>7.1f}% "
                  f"{pep_count:>10} {pep_count / total_peptides * 100:>7.1f}%")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"01_datasplit_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    print("=" * 80)
    print("  01_DATASPLIT -- HOMOLOGY-BASED DATA SPLIT")
    print("=" * 80)
    print(f"  Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data path:        {RAW_DATA_PATH}")
    print(f"  Output path:      {SPLIT_DATA_PATH}")
    print(f"  Log path:         {LOG_PATH}")
    print(f"  CV folds (k):     {N_CV_FOLDS}")
    print(f"  Held-out bucket:  {HELD_OUT_INDEX}")
    print(f"  Total buckets:    {N_BUCKETS}")
    print(f"  Random state:     {RANDOM_STATE}")
    print(f"  Hamming cutoff:   <= {HAMMING_CUTOFF}")
    print("=" * 80)

    # -- Load --
    print(f"\nLoading data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  [{i:>3}] {col:<40} dtype: {df[col].dtype}")

    required_cols = [PEPTIDE_COL, LABEL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\nFATAL: Missing required columns: {missing}")
        logger.close()
        sys.exit(1)
    print(f"\nRequired columns present: {required_cols}")

    print(f"\nLabel distribution:")
    label_counts = df[LABEL_COL].value_counts().sort_index()
    for val, cnt in label_counts.items():
        lbl = "positive" if val == 1 else "negative" if val == 0 else f"unknown({val})"
        print(f"  {val} ({lbl}): {cnt:>8}  ({cnt / len(df) * 100:.1f}%)")

    print(f"\nPeptide length distribution:")
    len_counts = df[PEPTIDE_COL].str.len().value_counts().sort_index()
    for length, cnt in len_counts.items():
        print(f"  {length}-mer: {cnt:>8}  ({cnt / len(df) * 100:.1f}%)")

    if "uniprot_id" in df.columns:
        print(f"\nUnique source proteins (uniprot_id): {df['uniprot_id'].nunique()}")

    # -- Duplicate peptide analysis --
    analyze_duplicate_peptides(df, PEPTIDE_COL, LABEL_COL)

    # -- Phase 1: Cluster --
    print("\n" + "=" * 80)
    print("PHASE 1: HOMOLOGY CLUSTERING")
    print("=" * 80)
    unique_peptides = df[PEPTIDE_COL].unique().tolist()
    print(f"Total rows:          {len(df)}")
    print(f"Unique peptides:     {len(unique_peptides)}")
    print(f"Duplicate rows:      {len(df) - len(unique_peptides)} "
          f"(same peptide, different protein context)")

    t_cluster_start = time.time()
    print(f"Clustering by Hamming distance <= {HAMMING_CUTOFF} ...")
    clusters = cluster_by_hamming(unique_peptides, max_dist=HAMMING_CUTOFF)
    t_cluster_end = time.time()

    n_clusters = len(clusters)
    largest_cluster_size = max(len(v) for v in clusters.values())
    print(f"Clusters formed:     {n_clusters}")
    print(f"Largest cluster:     {largest_cluster_size} peptides")
    print(f"Clustering time:     {t_cluster_end - t_cluster_start:.2f}s")

    if largest_cluster_size > len(unique_peptides) * 0.25:
        msg = (f"Largest cluster = {largest_cluster_size} peptides "
               f"({largest_cluster_size / len(unique_peptides) * 100:.1f}% "
               f"of unique peptides) -- may cause imbalanced buckets.")
        print(msg)
        warnings.warn(msg)

    # -- Phase 2: Assign --
    print("\n" + "=" * 80)
    print("PHASE 2: BUCKET ASSIGNMENT")
    print("=" * 80)

    # sample_counts counts ALL rows per peptide, including duplicates
    # from different proteins. This ensures balancing accounts for
    # the true data volume, not just unique sequences.
    sample_counts = df[PEPTIDE_COL].value_counts().to_dict()
    print(f"Sample counts include all rows (duplicates from different proteins counted)")

    np.random.seed(RANDOM_STATE)

    t_assign_start = time.time()
    peptide_to_bucket, bucket_totals = greedy_balanced_partition(
        clusters, N_BUCKETS, sample_counts
    )
    t_assign_end = time.time()

    # Map fold to EVERY row via peptide sequence.
    # All rows with the same peptide (regardless of source protein)
    # get the same fold assignment.
    df[FOLD_COL] = df[PEPTIDE_COL].map(peptide_to_bucket)
    print(f"Assignment time:     {t_assign_end - t_assign_start:.2f}s")
    print(f"Bucket totals:       {bucket_totals}")

    # -- Verify: no homology leakage --
    print("\n" + "=" * 80)
    print("LEAKAGE VERIFICATION")
    print("=" * 80)

    # Verify 1: No Hamming-distance-1 neighbors across folds
    t_verify_start = time.time()
    n_violations = verify_no_homology_leakage(df, PEPTIDE_COL, FOLD_COL)
    t_verify_end = time.time()
    print(f"Verification time: {t_verify_end - t_verify_start:.2f}s")

    # Verify 2: No peptide sequence split across folds
    # (confirms all duplicate rows stay together)
    print()
    n_splits = verify_no_peptide_split(df, PEPTIDE_COL, FOLD_COL)

    # -- Per-fold duplicate peptide report --
    print("\n" + "=" * 80)
    print("DUPLICATE PEPTIDES PER FOLD")
    print("=" * 80)
    print(f"  {'Fold':<10} {'Role':<22} {'Total':>7} {'Unique':>7} "
          f"{'Duplicates':>11} {'Dup %':>7}")
    print(f"  {'-' * 66}")
    for i in sorted(df[FOLD_COL].unique()):
        fold_df = df[df[FOLD_COL] == i]
        n_total = len(fold_df)
        n_unique = fold_df[PEPTIDE_COL].nunique()
        n_dup = n_total - n_unique
        role = "Held-out" if i == HELD_OUT_INDEX else f"CV fold {i}"
        print(f"  {i:<10} {role:<22} {n_total:>7} {n_unique:>7} "
              f"{n_dup:>11} {n_dup / n_total * 100 if n_total > 0 else 0:>6.1f}%")

    # -- Statistics --
    print_bucket_statistics(df, PEPTIDE_COL, LABEL_COL, FOLD_COL,
                            N_CV_FOLDS, HELD_OUT_INDEX)

    inspect_bucket_clusters(df, clusters, peptide_to_bucket,
                            PEPTIDE_COL, LABEL_COL, FOLD_COL,
                            N_CV_FOLDS, HELD_OUT_INDEX)

    # -- CV preview --
    print("\n" + "=" * 80)
    print("CV FOLD TRAIN/TEST SIZES")
    print("=" * 80)
    for fold_id in range(N_CV_FOLDS):
        cv_df = df[df[FOLD_COL] != HELD_OUT_INDEX]
        train = cv_df[cv_df[FOLD_COL] != fold_id]
        test = cv_df[cv_df[FOLD_COL] == fold_id]
        tr_pos = (train[LABEL_COL] == 1).sum()
        tr_neg = (train[LABEL_COL] == 0).sum()
        te_pos = (test[LABEL_COL] == 1).sum()
        te_neg = (test[LABEL_COL] == 0).sum()
        print(f"  Fold {fold_id}:  "
              f"train={len(train):>6} (pos={tr_pos}, neg={tr_neg}, "
              f"ratio={tr_neg / tr_pos:.3f})  "
              f"test={len(test):>6} (pos={te_pos}, neg={te_neg}, "
              f"ratio={te_neg / te_pos:.3f})")

    val_df = df[df[FOLD_COL] == HELD_OUT_INDEX]
    v_pos = (val_df[LABEL_COL] == 1).sum()
    v_neg = (val_df[LABEL_COL] == 0).sum()
    print(f"  Held-out: samples={len(val_df):>6} "
          f"(pos={v_pos}, neg={v_neg}, ratio={v_neg / v_pos:.3f})")

    # -- Save --
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SPLIT_DATA_PATH, index=False)
    print(f"\nSaved to: {SPLIT_DATA_PATH}")

    t_end = time.time()
    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"  Total runtime:    {t_end - t_start:.2f}s")
    print(f"  Input rows:       {len(df)}")
    print(f"  Unique peptides:  {len(unique_peptides)}")
    print(f"  Duplicate rows:   {len(df) - len(unique_peptides)}")
    print(f"  Clusters:         {n_clusters}")
    print(f"  Buckets:          {N_BUCKETS} ({N_CV_FOLDS} CV + 1 held-out)")
    print(f"  Homology leakage: {'NONE' if n_violations == 0 else f'{n_violations} violations'}")
    print(f"  Peptide splits:   {'NONE' if n_splits == 0 else f'{n_splits} peptides split'}")
    print(f"  Output:           {SPLIT_DATA_PATH}")
    print(f"  Log:              {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()

