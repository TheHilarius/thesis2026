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

# Protein column — all peptides from the same protein stay in the same fold.
PROTEIN_COL = "uniprot_id"


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
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


# ──────────────────────────────────────────────
# 2. PROTEIN-LEVEL HOMOLOGY CLUSTERING
# ──────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def cluster_proteins_by_homology(df, peptide_col, protein_col, max_dist=1):
    """
    Build protein-level clusters enforcing three constraints:
      1. All peptides from the same protein → same cluster  (implicit)
      2. Proteins sharing any peptide sequence → same cluster
      3. Proteins connected by Hamming-<=max_dist peptides → same cluster

    The Union-Find operates on protein IDs (not peptides).

    Returns
    -------
    clusters : dict  {root_protein: set of proteins}
    stats    : dict  merge counts for reporting
    """
    uf = UnionFind()

    # Initialise all proteins
    proteins = df[protein_col].unique()
    for prot in proteins:
        uf.find(prot)

    # Build peptide → set of source proteins
    peptide_to_proteins = defaultdict(set)
    for pep, prot in zip(df[peptide_col], df[protein_col]):
        peptide_to_proteins[pep].add(prot)

    # ── Layer 1: Union proteins that share any peptide ──
    shared_peptide_merges = 0
    for pep, prots in peptide_to_proteins.items():
        if len(prots) > 1:
            prot_list = list(prots)
            for i in range(1, len(prot_list)):
                if uf.union(prot_list[0], prot_list[i]):
                    shared_peptide_merges += 1

    # ── Layer 2: Union proteins with Hamming-<=max_dist peptides ──
    all_peptides = set(peptide_to_proteins.keys())
    hamming_merges = 0

    for pep in all_peptides:
        # Representative protein for this peptide's current group
        pep_prots = peptide_to_proteins[pep]
        anchor_prot = next(iter(pep_prots))

        for pos in range(len(pep)):
            for aa in AMINO_ACIDS:
                if aa != pep[pos]:
                    neighbour = pep[:pos] + aa + pep[pos + 1:]
                    if neighbour in all_peptides:
                        neighbour_prot = next(iter(peptide_to_proteins[neighbour]))
                        if uf.union(anchor_prot, neighbour_prot):
                            hamming_merges += 1

    # ── Collect final protein clusters ──
    clusters = defaultdict(set)
    for prot in proteins:
        clusters[uf.find(prot)].add(prot)

    stats = {
        "n_proteins": len(proteins),
        "n_unique_peptides": len(all_peptides),
        "shared_peptide_merges": shared_peptide_merges,
        "hamming_merges": hamming_merges,
        "n_clusters": len(clusters),
    }

    return dict(clusters), stats

# ──────────────────────────────────────────────
# 3. BALANCED PARTITION
# ──────────────────────────────────────────────

def greedy_balanced_partition(protein_clusters, n_buckets, protein_sample_counts):
    """
    Greedy bin-packing: assign each protein cluster to the smallest bucket.

    protein_sample_counts: dict {protein_id: n_rows} from the DataFrame.
    Returns protein_to_bucket mapping and bucket totals.
    """
    cluster_info = []
    for root, prots in protein_clusters.items():
        total = sum(protein_sample_counts.get(p, 0) for p in prots)
        cluster_info.append((root, prots, total))

    cluster_info.sort(key=lambda x: x[2], reverse=True)

    bucket_totals = np.zeros(n_buckets, dtype=int)
    protein_to_bucket = {}

    for root, prots, size in cluster_info:
        target = int(np.argmin(bucket_totals))
        for p in prots:
            protein_to_bucket[p] = target
        bucket_totals[target] += size

    return protein_to_bucket, bucket_totals.tolist()

# ──────────────────────────────────────────────
# 4. VERIFICATION
# ──────────────────────────────────────────────

def verify_no_protein_leakage(df, protein_col, fold_col):
    """Check that no protein appears in more than one fold."""
    prot_folds = df.groupby(protein_col)[fold_col].nunique()
    split_prots = prot_folds[prot_folds > 1]

    if len(split_prots) == 0:
        print("PASS: No protein split across folds.")
    else:
        print(f"FAIL: {len(split_prots)} proteins split across folds!")
        for prot, n_f in split_prots.head(10).items():
            folds_in = df.loc[df[protein_col] == prot, fold_col].unique()
            print(f"  {prot}: appears in folds {sorted(folds_in)}")

    return len(split_prots)


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
        print("PASS: No homology leakage detected across folds.")
    else:
        print(f"FAIL: {violations} cross-fold homologous pairs found!")

    return violations


def verify_no_peptide_split(df, peptide_col, fold_col):
    """
    Verify that every row with the same peptide sequence is in the same fold.
    """
    peptide_folds = df.groupby(peptide_col)[fold_col].nunique()
    split_peptides = peptide_folds[peptide_folds > 1]

    if len(split_peptides) == 0:
        print("PASS: No peptide split across folds (all duplicates in same fold).")
    else:
        print(f"FAIL: {len(split_peptides)} peptides split across folds!")
        for pep, n_folds in split_peptides.head(10).items():
            folds_in = df.loc[df[peptide_col] == pep, fold_col].unique()
            print(f"  {pep}: appears in folds {sorted(folds_in)}")

    return len(split_peptides)


# ──────────────────────────────────────────────
# 5. DUPLICATE PEPTIDE ANALYSIS
# ──────────────────────────────────────────────

def analyze_duplicate_peptides(df, peptide_col, label_col):
    """
    Report on peptides that appear multiple times in the dataset.

    Distinguishes between:
      - Exact duplicates: same peptide, same protein, same flanks, same position
        (true duplicates — likely data-processing artefacts)
      - Positional variants: same peptide, protein, flanks but different start
        position (distinct cleavage sites that happen to share local context)
      - Context variants: same peptide sequence but different protein or flanking
        regions (biologically distinct observations)
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

    # ── Context-aware uniqueness ──
    # Two rows with the same peptide are only true duplicates if they
    # also share the same protein, flanking regions, AND start position.
    # Same peptide + protein + flanks but different position means a
    # repeated local motif at a distinct site in the protein.
    context_cols_available = [
        c for c in ["uniprot_id", "n_flank", "c_flank"] if c in df.columns
    ]
    position_cols_available = [
        c for c in ["start"] if c in df.columns
    ]

    # Full identity = peptide + context + position
    full_id_cols = [peptide_col] + context_cols_available + position_cols_available

    if context_cols_available:
        # -- Level 1: unique (peptide + context) ignoring position --
        context_group_cols = [peptide_col] + context_cols_available
        unique_contexts = df.drop_duplicates(subset=context_group_cols).shape[0]
        context_variants = unique_contexts - unique_peptides

        # -- Level 2: unique (peptide + context + position) --
        if position_cols_available:
            unique_full = df.drop_duplicates(subset=full_id_cols).shape[0]
            true_duplicates = total_rows - unique_full
            positional_variants = unique_full - unique_contexts
        else:
            unique_full = unique_contexts
            true_duplicates = total_rows - unique_contexts
            positional_variants = 0

        print(f"\n  Context-aware breakdown "
              f"(using {', '.join(context_cols_available)}"
              f"{', start' if position_cols_available else ''}):")
        print(f"  Unique (peptide, context, position) tuples: {unique_full}")
        print(f"  Context variants:    {context_variants:>6}  "
              f"(same sequence, different protein/flanks)")
        if position_cols_available:
            print(f"  Positional variants: {positional_variants:>6}  "
                  f"(same sequence + protein + flanks, different start position)")
        print(f"  True duplicates:     {true_duplicates:>6}  "
              f"(identical on all fields — likely data artefacts)")

        # ── Show examples of context variants ──
        if context_variants > 0:
            pep_context_counts = (
                df.groupby(peptide_col)[context_cols_available]
                .apply(lambda g: g.drop_duplicates().shape[0])
            )
            multi_context = pep_context_counts[pep_context_counts > 1]

            print(f"\n  Peptides appearing in multiple distinct contexts: "
                  f"{len(multi_context)}")

            if len(multi_context) > 0:
                print(f"\n  Top 10 peptides by number of distinct contexts:")
                print(f"  {'Peptide':<15} {'Contexts':>9} {'Rows':>6}", end="")
                if "uniprot_id" in df.columns:
                    print(f" {'Proteins':>9}", end="")
                print(f" {'Label dist':>15}")
                print(f"  {'-' * 65}")

                for pep in multi_context.nlargest(10).index:
                    n_ctx = multi_context[pep]
                    pep_rows = df[df[peptide_col] == pep]
                    n_rows = len(pep_rows)
                    n_pos = (pep_rows[label_col] == 1).sum()
                    n_neg = (pep_rows[label_col] == 0).sum()

                    print(f"  {pep:<15} {n_ctx:>9} {n_rows:>6}", end="")
                    if "uniprot_id" in df.columns:
                        n_prots = pep_rows["uniprot_id"].nunique()
                        print(f" {n_prots:>9}", end="")
                    print(f"   pos={n_pos}, neg={n_neg}")

        # ── Show positional variants ──
        if position_cols_available and positional_variants > 0:
            print(f"\n  Positional variant details:")
            print(f"  (Same peptide + protein + flanks but different start position)")

            pos_groups = (
                df.groupby(context_group_cols)["start"]
                .apply(lambda g: g.nunique())
            )
            multi_pos = pos_groups[pos_groups > 1]
            print(f"  Affected (peptide, context) groups: {len(multi_pos)}")

            if len(multi_pos) > 0:
                multi_pos_sorted = multi_pos.sort_values(ascending=False)
                print(f"\n  Top 5 groups with multiple positions:")
                shown = 0
                for keys, n_positions in multi_pos_sorted.items():
                    if shown >= 5:
                        break
                    if isinstance(keys, tuple):
                        pep = keys[0]
                        ctx = dict(zip(context_cols_available, keys[1:]))
                    else:
                        pep = keys
                        ctx = {}

                    # Retrieve the actual start positions
                    mask = df[peptide_col] == pep
                    for col_name, col_val in ctx.items():
                        if pd.isna(col_val):
                            mask = mask & df[col_name].isna()
                        else:
                            mask = mask & (df[col_name] == col_val)
                    starts = sorted(df.loc[mask, "start"].unique())

                    prot_str = ctx.get("uniprot_id", "?")
                    print(f"    {pep}  protein={prot_str}  "
                          f"positions={starts}  ({n_positions} distinct)")
                    shown += 1

        # ── Show true duplicates ──
        if true_duplicates > 0:
            print(f"\n  WARNING: {true_duplicates} true duplicate rows detected")
            print(f"  (identical peptide + protein + flanking regions + start position)")
            dup_mask = df.duplicated(subset=full_id_cols, keep=False)
            dup_df = df[dup_mask]
            dup_groups = dup_df.groupby(full_id_cols).size()
            dup_groups = dup_groups[dup_groups > 1].sort_values(ascending=False)
            print(f"  Affected groups: {len(dup_groups)}")
            print(f"\n  Top 5 duplicated groups:")
            for keys, count in dup_groups.head(5).items():
                if isinstance(keys, tuple):
                    pep = keys[0]
                    extra_keys = context_cols_available + position_cols_available
                    ctx = dict(zip(extra_keys, keys[1:]))
                else:
                    pep = keys
                    ctx = {}
                print(f"    {pep}  x{count}  {ctx}")
    else:
        print(f"\n  NOTE: Context columns (uniprot_id, n_flank, c_flank) not found.")
        print(f"  Cannot distinguish context variants from true duplicates.")

    # ── Occurrence count distribution ──
    pep_counts = df[peptide_col].value_counts()
    multi_occurrence = pep_counts[pep_counts > 1]

    if len(multi_occurrence) == 0:
        print("\n  No peptides appear more than once.")
        return

    print(f"\n  Peptides appearing more than once: {len(multi_occurrence)}")
    print(f"  Total rows from multi-occurrence peptides: {multi_occurrence.sum()}")

    print(f"\n  Occurrence count distribution:")
    print(f"  {'Times seen':<15} {'Peptides':>10} {'Total rows':>12}")
    print(f"  {'-' * 40}")
    for count_val in sorted(pep_counts.unique()):
        n_peps = (pep_counts == count_val).sum()
        n_rows = n_peps * count_val
        if n_peps > 0:
            print(f"  {count_val:<15} {n_peps:>10} {n_rows:>12}")

    # ── Protein overlap ──
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

    # ── Label consistency ──
    print(f"\n  Label consistency for duplicate peptides:")
    pep_label_nunique = df.groupby(peptide_col)[label_col].nunique()
    mixed_label = pep_label_nunique[pep_label_nunique > 1]
    print(f"  Peptides with consistent label:  {(pep_label_nunique == 1).sum()}")
    print(f"  Peptides with mixed labels:      {len(mixed_label)}")

    if len(mixed_label) > 0:
        print(f"\n  NOTE: {len(mixed_label)} peptides have both positive and negative labels.")
        print(f"  This is expected when the same peptide is processed in one protein")
        print(f"  context but not another (different flanking regions / source proteins).")

        # Show whether mixed labels correlate with different contexts
        if context_cols_available:
            full_context_cols = context_cols_available + position_cols_available
            mixed_explained = 0
            for pep in mixed_label.index:
                pep_rows = df[df[peptide_col] == pep]
                ctx_label = pep_rows.groupby(full_context_cols)[label_col].nunique()
                if (ctx_label == 1).all():
                    mixed_explained += 1

            mixed_unexplained = len(mixed_label) - mixed_explained
            print(f"\n  Mixed-label breakdown "
                  f"(grouping by {', '.join(full_context_cols)}):")
            print(f"    Explained by context/position: "
                  f"{mixed_explained}")
            print(f"    Unexplained (same full identity, conflicting labels): "
                  f"{mixed_unexplained}")
            if mixed_unexplained > 0:
                print(f"    WARNING: {mixed_unexplained} peptides have conflicting labels")
                print(f"    within the SAME protein + flanks + position "
                      f"(possible data issue)")

        print(f"\n  Examples:")
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


def inspect_bucket_clusters(df, protein_clusters, protein_to_bucket,
                            protein_col, peptide_col, label_col, fold_col,
                            n_cv_folds, held_out_index):
    bucket_clusters = defaultdict(list)

    for root, prots in protein_clusters.items():
        representative = next(iter(prots))
        bucket_id = protein_to_bucket[representative]
        n_prots = len(prots)
        n_rows = df[df[protein_col].isin(prots)].shape[0]
        bucket_clusters[bucket_id].append((root, prots, n_prots, n_rows))

    folds = sorted(df[fold_col].unique())

    print("\n" + "=" * 80)
    print("CLUSTER DISTRIBUTION PER BUCKET")
    print("=" * 80)

    for i in folds:
        role = "Held-out validation" if i == held_out_index else f"CV fold {i}"
        cls_list = bucket_clusters.get(i, [])

        if not cls_list:
            print(f"\n-- Bucket {i} ({role}): EMPTY --")
            continue

        prot_sizes = [n_p for _, _, n_p, _ in cls_list]
        row_sizes = [n_r for _, _, _, n_r in cls_list]
        n_clusters = len(cls_list)
        n_singletons = sum(1 for s in prot_sizes if s == 1)
        total_prots = sum(prot_sizes)
        total_rows = sum(row_sizes)

        print(f"\n-- Bucket {i} ({role}) --")
        print(f"  Total clusters:      {n_clusters}")
        print(f"  Total proteins:      {total_prots}")
        print(f"  Total rows:          {total_rows}")
        print(f"  Singleton clusters:  {n_singletons}  "
              f"({n_singletons / n_clusters * 100:.1f}% of clusters)")
        print(f"  Cluster size stats (proteins): "
              f"mean={np.mean(prot_sizes):.1f}  "
              f"median={np.median(prot_sizes):.1f}  "
              f"max={max(prot_sizes)}")

        top_clusters = sorted(cls_list, key=lambda x: x[3], reverse=True)[:5]
        print(f"\n  Top 5 largest clusters (by rows):")
        for rnk, (root, prots, n_p, n_r) in enumerate(top_clusters, 1):
            cluster_rows = df[df[protein_col].isin(prots)]
            c_pos = (cluster_rows[label_col] == 1).sum()
            c_neg = (cluster_rows[label_col] == 0).sum()
            prot_list = sorted(prots)[:4]
            suffix = f" ... +{n_p - 4} more" if n_p > 4 else ""
            print(f"    #{rnk} proteins={n_p:>3}  rows={n_r:>5}  "
                  f"(pos={c_pos}, neg={c_neg})  "
                  f"proteins: {', '.join(prot_list)}{suffix}")

    # Global histogram by protein-cluster size
    all_prot_sizes = [len(prots) for prots in protein_clusters.values()]
    print(f"\n{'=' * 80}")
    print("GLOBAL CLUSTER SIZE HISTOGRAM (proteins per cluster)")
    print("=" * 80)

    bins = [(1, 1, "Singletons"), (2, 2, "Pairs"), (3, 5, "3-5"),
            (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"),
            (51, 100, "51-100"), (101, float("inf"), ">100")]

    print(f"  {'Size range':<15} {'Clusters':>10} {'%':>8} {'Proteins':>10} {'%':>8}")
    print(f"  {'-' * 55}")
    total_clusters = len(all_prot_sizes)
    total_proteins = sum(all_prot_sizes)

    for lo, hi, label in bins:
        if hi == float("inf"):
            count = sum(1 for s in all_prot_sizes if s >= lo)
            prot_count = sum(s for s in all_prot_sizes if s >= lo)
        else:
            count = sum(1 for s in all_prot_sizes if lo <= s <= hi)
            prot_count = sum(s for s in all_prot_sizes if lo <= s <= hi)
        if count > 0:
            print(f"  {label:<15} {count:>10} {count / total_clusters * 100:>7.1f}% "
                  f"{prot_count:>10} {prot_count / total_proteins * 100:>7.1f}%")

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
    print("PHASE 1: PROTEIN-LEVEL HOMOLOGY CLUSTERING")
    print("=" * 80)

    # Validate protein column exists
    if PROTEIN_COL not in df.columns:
        print(f"\nFATAL: Protein column '{PROTEIN_COL}' not found in data.")
        print(f"  Available columns: {list(df.columns)}")
        logger.close()
        sys.exit(1)

    unique_proteins = df[PROTEIN_COL].nunique()
    unique_peptides_list = df[PEPTIDE_COL].unique().tolist()
    n_unique_peptides = len(unique_peptides_list)

    print(f"Total rows:            {len(df)}")
    print(f"Unique proteins:       {unique_proteins}")
    print(f"Unique peptides:       {n_unique_peptides}")
    print(f"Duplicate rows:        {len(df) - n_unique_peptides} "
          f"(same peptide, different protein context)")

    t_cluster_start = time.time()
    print(f"\nClustering proteins by:")
    print(f"  1. Shared peptide sequences")
    print(f"  2. Hamming distance <= {HAMMING_CUTOFF} between any peptides")

    protein_clusters, cluster_stats = cluster_proteins_by_homology(
        df, PEPTIDE_COL, PROTEIN_COL, max_dist=HAMMING_CUTOFF
    )
    t_cluster_end = time.time()

    n_clusters = cluster_stats["n_clusters"]
    largest_cluster = max(len(v) for v in protein_clusters.values())
    largest_cluster_rows = max(
        df[df[PROTEIN_COL].isin(prots)].shape[0]
        for prots in protein_clusters.values()
    )

    print(f"\nClustering results:")
    print(f"  Protein clusters:        {n_clusters}")
    print(f"  Shared-peptide merges:   {cluster_stats['shared_peptide_merges']}")
    print(f"  Hamming merges:          {cluster_stats['hamming_merges']}")
    print(f"  Largest cluster:         {largest_cluster} proteins "
          f"({largest_cluster_rows} rows)")
    print(f"  Clustering time:         {t_cluster_end - t_cluster_start:.2f}s")

    if largest_cluster_rows > len(df) * 0.25:
        msg = (f"Largest cluster = {largest_cluster_rows} rows "
               f"({largest_cluster_rows / len(df) * 100:.1f}% of data) "
               f"-- may cause imbalanced buckets.")
        print(f"  WARNING: {msg}")
        warnings.warn(msg)

    # Cluster size distribution (by number of proteins)
    prot_cluster_sizes = sorted([len(v) for v in protein_clusters.values()], reverse=True)
    n_singleton = sum(1 for s in prot_cluster_sizes if s == 1)
    n_multi = sum(1 for s in prot_cluster_sizes if s > 1)
    print(f"\n  Singleton clusters (1 protein):     {n_singleton}")
    print(f"  Multi-protein clusters (>1):        {n_multi}")
    if n_multi > 0:
        print(f"  Top 10 largest clusters (by proteins):")
        for i, size in enumerate(prot_cluster_sizes[:10], 1):
            print(f"    #{i}: {size} proteins")

    # -- Phase 2: Assign --
    print("\n" + "=" * 80)
    print("PHASE 2: BUCKET ASSIGNMENT")
    print("=" * 80)

    # Count rows per protein for balanced bin-packing
    protein_sample_counts = df[PROTEIN_COL].value_counts().to_dict()
    print(f"Balancing by row count per protein (total: {len(df)} rows)")

    np.random.seed(RANDOM_STATE)

    t_assign_start = time.time()
    protein_to_bucket, bucket_totals = greedy_balanced_partition(
        protein_clusters, N_BUCKETS, protein_sample_counts
    )
    t_assign_end = time.time()

    # Map fold to every row via protein ID
    df[FOLD_COL] = df[PROTEIN_COL].map(protein_to_bucket)
    print(f"Assignment time:     {t_assign_end - t_assign_start:.2f}s")
    print(f"Bucket totals:       {bucket_totals}")

    # Sanity: check no unmapped rows
    unmapped = df[FOLD_COL].isna().sum()
    if unmapped > 0:
        print(f"FATAL: {unmapped} rows could not be mapped to a fold!")
        print(f"  Unmapped proteins: "
              f"{df.loc[df[FOLD_COL].isna(), PROTEIN_COL].unique()[:10]}")
        logger.close()
        sys.exit(1)
    df[FOLD_COL] = df[FOLD_COL].astype(int)

    # -- Leakage verification --
    print("\n" + "=" * 80)
    print("LEAKAGE VERIFICATION")
    print("=" * 80)

    # Verify 1: No protein split across folds (strongest guarantee)
    n_prot_splits = verify_no_protein_leakage(df, PROTEIN_COL, FOLD_COL)

    # Verify 2: No peptide sequence split across folds
    print()
    n_pep_splits = verify_no_peptide_split(df, PEPTIDE_COL, FOLD_COL)

    # Verify 3: No Hamming-distance-1 neighbors across folds
    print()
    t_verify_start = time.time()
    n_violations = verify_no_homology_leakage(df, PEPTIDE_COL, FOLD_COL)
    t_verify_end = time.time()
    print(f"Homology verification time: {t_verify_end - t_verify_start:.2f}s")

    
    # -- Statistics --
    print_bucket_statistics(df, PEPTIDE_COL, LABEL_COL, FOLD_COL,
                            N_CV_FOLDS, HELD_OUT_INDEX)

    inspect_bucket_clusters(df, protein_clusters, protein_to_bucket,
                            PROTEIN_COL, PEPTIDE_COL, LABEL_COL, FOLD_COL,
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
    print(f"  Total runtime:           {t_end - t_start:.2f}s")
    print(f"  Input rows:              {len(df)}")
    print(f"  Unique proteins:         {unique_proteins}")
    print(f"  Unique peptides:         {n_unique_peptides}")
    print(f"  Protein clusters:        {n_clusters}")
    print(f"  Shared-peptide merges:   {cluster_stats['shared_peptide_merges']}")
    print(f"  Hamming merges:          {cluster_stats['hamming_merges']}")
    print(f"  Buckets:                 {N_BUCKETS} ({N_CV_FOLDS} CV + 1 held-out)")
    print(f"  Protein leakage:         {'NONE' if n_prot_splits == 0 else f'{n_prot_splits} proteins split'}")
    print(f"  Peptide leakage:         {'NONE' if n_pep_splits == 0 else f'{n_pep_splits} peptides split'}")
    print(f"  Homology leakage:        {'NONE' if n_violations == 0 else f'{n_violations} violations'}")
    print(f"  Output:                  {SPLIT_DATA_PATH}")
    print(f"  Log:                     {LOG_PATH}")
    print(f"  Completed:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()

