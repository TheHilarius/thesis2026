"""
01_datasplit.py
Homology-based data splitting for MHC-I processing prediction.

Usage:
    python src/pipeline/python/01_datasplit.py
    python src/pipeline/python/01_datasplit.py --hamming-cutoff 3 \
        --split-out data/processed/df_all_with_folds_h3.csv

    # Robust Hamming sweep over the FULL context window (19-mer or 29-mer,
    # width inferred from the data), one marked split file per cutoff:
    python src/pipeline/python/01_datasplit.py --group-by context --sweep 1,2,3
    THESIS_VARIANT=flank5 python src/pipeline/python/01_datasplit.py \
        --group-by context --sweep 1,2,3

--group-by context is a ROBUST split: one union-find over the union of
three edge sets —
    (A) window edges:     ham(full_context) <= c
    (B) peptide edges:    ham(peptide) <= 1  (fixed, independent of c)
    (C) flank edges:      EITHER flank byte-identical (n_flank equal OR
                          c_flank equal, X-padding included)
GUARANTEES:
  - No window pair at distance <= c crosses folds.
  - No peptide pair at distance <= 1 crosses folds.
  - No pair sharing an identical flank crosses folds (audit-41 rule).
  - NOT guaranteed: peptide pairs at distance 2..c with divergent flanks
    can still be split (that case is the canonical peptide-hc split).
  Relations: pure-window-hc and peptide-h1 are both subsets of
  robust-hc; robust-hc and peptide-hc are INCOMPARABLE (robust adds
  flank-identity, peptide-hc adds far-flank pairs at distance 2..c).
  At c=1, robust == peptide-h1 ∪ flank-identity.
All three violations are checked HARD (FATAL + exit 1) after fold
assignment.

Outputs are additive marked files (e.g. df_all_with_folds_updated_h2.csv);
the canonical df_all_with_folds.csv is only written by the default
invocation (peptide grouping, cutoff == config HAMMING_CUTOFF, no sweep).
Pure-window baseline files df_all_with_folds*_ctx-h*.csv are never
overwritten (produced by src/pipeline/python/01_datasplit.py.ctx_baseline.bak).
"""

import argparse
import sys
import os
from pathlib import Path

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
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
# 'X' = flank padding char inside full_context windows (context mode).
# Distinct code so the pigeonhole encoder never KeyErrors on padded
# contexts; also part of the substitution alphabet so A<->X
# single-substitution pairs are found from BOTH sides.
AA_TO_INT["X"] = 20
NEIGHBOR_ALPHABET = AMINO_ACIDS + ["X"]


def iter_close_pairs(peps, max_dist):
    """
    Yield (i, j) index pairs (i < j; a pair can be yielded more than
    once when it shares several bucket keys) of peptides at DIRECT
    Hamming distance <= max_dist, for max_dist in {2, 3}.

    Pigeonhole candidate index: positions are split into 3 contiguous
    blocks. Two strings at Hamming distance <= 3 mismatch in at most 3
    positions, so at least one block has <= 1 mismatch. Indexing each
    peptide per block by its exact block string plus every
    single-position wildcard variant ('*') collects a superset of all
    close pairs into shared buckets; every candidate pair is verified
    with the true Hamming distance before being yielded.
    """
    if max_dist not in (2, 3):
        raise ValueError(
            f"iter_close_pairs supports max_dist 2 or 3, got {max_dist}")
    L = len(peps[0])
    if any(len(p) != L for p in peps):
        raise ValueError(
            "All peptides must have equal length for Hamming clustering")
    if L < 3:
        raise ValueError("Pigeonhole index needs at least 3 positions")

    bounds = [0, L // 3, (2 * L) // 3, L]

    # Bucket key -> peptide indices. Exact keys are (block, string)
    # 2-tuples, wildcard keys (block, wildcard position, string) with a
    # '*' that cannot collide with a real residue.
    buckets = defaultdict(list)
    for i, pep in enumerate(peps):
        for b in range(3):
            s, e = bounds[b], bounds[b + 1]
            block = pep[s:e]
            buckets[(b, block)].append(i)          # exact block
            for p in range(s, e):                  # one wildcard position
                wc = block[:p - s] + "*" + block[p - s + 1:]
                buckets[(b, p, wc)].append(i)

    enc = np.array([[AA_TO_INT[c] for c in p] for p in peps], dtype=np.uint8)

    chunk = 512  # bucket rows per verification block (bounds memory)
    for members in buckets.values():
        m = len(members)
        if m < 2:
            continue
        idx = np.array(members, dtype=np.int64)
        B = enc[idx]
        for c0 in range(0, m, chunk):
            c1 = min(c0 + chunk, m)
            # Hamming distances of the chunk rows against all bucket members
            D = (B[c0:c1, None, :] != B[None, :, :]).sum(axis=2, dtype=np.uint8)
            for a, b in np.argwhere(D <= max_dist):
                gi, gj = int(idx[c0 + a]), int(idx[b])
                if gi < gj:
                    yield gi, gj


def cluster_by_hamming(peptides, max_dist=1):
    """
    Cluster unique peptides by DIRECT Hamming distance <= max_dist.

    Identical peptides are the same node (Hamming distance = 0), so
    duplicate peptide sequences from different proteins are trivially
    in the same cluster.

    max_dist = 1     : exhaustive single-substitution neighbour enumeration.
    max_dist = 2 or 3: pigeonhole candidate index (iter_close_pairs).
    max_dist > 3     : not supported (3-block pigeonhole bound fails).

    Note: union-find merges transitively, so a cluster can contain pairs
    farther apart than max_dist that are connected through bridging
    peptides. The split guarantee is that no DIRECT <= max_dist pair is
    separated across folds.
    """
    uf = UnionFind()  # Initialize the union-find structure to manage clusters of peptides
    peptide_set = set(peptides)

    for pep in peptides:
        uf.find(pep)  # Ensure every peptide is initialized in the union-find structure

    if max_dist == 1:
        for pep in peptides:
            for pos in range(len(pep)):
                for aa in NEIGHBOR_ALPHABET:  # substitutions incl. X<->AA
                    if aa != pep[pos]:  # Only substitutions that change the sequence (distance 1)
                        neighbour = pep[:pos] + aa + pep[pos + 1:]
                        if neighbour in peptide_set:  # Neighbour exists -> homologous, same cluster
                            uf.union(pep, neighbour)
    elif max_dist in (2, 3):
        peps = sorted(peptide_set)  # deterministic order for the index
        for i, j in iter_close_pairs(peps, max_dist):
            uf.union(peps[i], peps[j])
    else:
        raise NotImplementedError(
            f"HAMMING_CUTOFF {max_dist} not supported: use 1, 2 or 3 "
            f"(pigeonhole index requires <= 3 mismatches over 3 blocks)")

    clusters = defaultdict(set) # initialize a dictionary to hold clusters, where each key is a cluster representative (root) and the value is a set of peptides in that cluster
    for pep in peptides: # For each peptide
        clusters[uf.find(pep)].add(pep) # Add the peptide to the cluster corresponding to its root representative

    return dict(clusters)


def cluster_context_robust(contexts, ctx_to_peptide, max_dist):
    """
    Robust context clustering: ONE union-find over the union of three
    edge sets, taken as connected components of A ∪ B ∪ C (never
    sequential — no phase's output is final on its own):

      (A) window edges:  ham(full_context) <= max_dist   (reuses
          cluster_by_hamming unchanged)
      (B) peptide edges: ham(peptide) <= 1, fixed at 1 and independent
          of max_dist — lifts to every context of those peptides
      (C) flank edges:   two contexts are linked if EITHER flank is
          byte-identical (n_flank equal OR c_flank equal, X-padding
          included); flanks are sliced from the context string itself

    Returns (clusters, pep_class) where clusters maps root -> set of
    contexts, and pep_class maps each context -> its peptide-h1 class
    root (used for the c=1 bijection assertion).

    Guarantees (see module docstring): no window pair at distance
    <= max_dist, no peptide pair at distance <= 1, and no pair sharing
    an identical flank crosses the resulting groups.
    """
    # Edge set A: components of window-ham <= max_dist
    win_clusters = cluster_by_hamming(contexts, max_dist=max_dist)
    # Edge set B: components of peptide-ham <= 1
    unique_peptides = list(dict.fromkeys(ctx_to_peptide[c] for c in contexts))
    pep_clusters = cluster_by_hamming(unique_peptides, max_dist=1)

    pep_class = {}
    for root, peps in pep_clusters.items():
        for p in peps:
            pep_class[p] = root

    uf = UnionFind()
    for c in contexts:
        uf.find(c)
    # Add A edges (union within each window component; members sorted so
    # the pass is order-stable — components do not depend on it)
    for members in win_clusters.values():
        ordered = sorted(members)
        for m in ordered[1:]:
            uf.union(ordered[0], m)
    # Add B edges: all contexts whose peptides share a peptide-h1 class
    by_class = defaultdict(list)
    for c in contexts:
        by_class[pep_class[ctx_to_peptide[c]]].append(c)
    for members in by_class.values():
        ordered = sorted(members)
        for m in ordered[1:]:
            uf.union(ordered[0], m)

    # Add C edges: EITHER flank byte-identical links the contexts.
    # Flanks sliced from the string: (len - 9) / 2 per side (peptide = 9).
    L = (len(contexts[0]) - 9) // 2
    by_nflank, by_cflank = defaultdict(list), defaultdict(list)
    for c in contexts:
        by_nflank[c[:L]].append(c)
        by_cflank[c[len(c) - L:]].append(c)
    for groups in (by_nflank, by_cflank):
        for members in groups.values():
            if len(members) < 2:
                continue
            ordered = sorted(members)
            for m in ordered[1:]:
                uf.union(ordered[0], m)

    clusters = defaultdict(set)
    for c in contexts:
        clusters[uf.find(c)].add(c)

    ctx_class = {c: pep_class[ctx_to_peptide[c]] for c in contexts}
    return dict(clusters), ctx_class


# ──────────────────────────────────────────────
# 3. BALANCED PARTITION
# ──────────────────────────────────────────────

def greedy_balanced_partition(clusters, n_buckets, peptide_sample_counts,
                              key_min_row):
    """
    Greedy bin-packing: assign each cluster to the currently smallest bucket.

    peptide_sample_counts includes ALL rows for each cluster member
    (including duplicates from different proteins), so the balancing
    accounts for the true number of data rows, not just unique keys.

    Deterministic: clusters are ordered by (-size, min row index), so
    tie-breaking never depends on dict/union-find construction order.
    """
    cluster_info = []
    for root, peps in clusters.items(): # For each cluster
        total = sum(peptide_sample_counts.get(p, 0) for p in peps) # Calculate the total number of samples (rows) that correspond to the keys in this cluster, including duplicates from different proteins
        min_row = min(key_min_row[p] for p in peps)
        cluster_info.append((root, peps, total, min_row)) # Append (representative, members, total row count, first row index)

    cluster_info.sort(key=lambda x: (-x[2], x[3])) # Largest first; ties broken by earliest row — deterministic across construction orders
    # x[2] is the second column, which contains the total sample count for each cluster. Reverse = True, so largest clusters are first. 
    bucket_totals = np.zeros(n_buckets, dtype=int) 
    peptide_to_bucket = {} 

    print_count = 0
    for root, peps, size, _min_row in cluster_info:
        if print_count < 5:
            print(f"Assigning cluster with root '{root}' and {len(peps)} peptides "
                  f"({size} total samples) to a bucket...")
        print_count += 1
        target = int(np.argmin(bucket_totals)) # Bucket with smallest current size 
        for p in peps: #
            peptide_to_bucket[p] = target # Assign all peptides in the cluster to the target bucket
        bucket_totals[target] += size # Bucket size with duplicates 

    return peptide_to_bucket, bucket_totals.tolist() # 


# ──────────────────────────────────────────────
# 4. VERIFICATION
# ──────────────────────────────────────────────

def verify_no_homology_leakage(df, peptide_col, fold_col, hamming_cutoff=1):
    """
    Check that no two peptides in different folds have DIRECT Hamming
    distance <= hamming_cutoff. Identical peptides appearing in
    different folds are covered separately by verify_no_peptide_split.
    """
    folds = sorted(df[fold_col].unique())
    fold_peptides = {
        f: set(df.loc[df[fold_col] == f, peptide_col].unique()) for f in folds
    }

    violations = 0
    if hamming_cutoff == 1:
        for f1, f2 in combinations(folds, 2):
            set2 = fold_peptides[f2]
            for pep in fold_peptides[f1]:
                for pos in range(len(pep)):
                    for aa in NEIGHBOR_ALPHABET:  # incl. X<->AA substitutions
                        if aa != pep[pos]:
                            neighbour = pep[:pos] + aa + pep[pos + 1:]
                            if neighbour in set2:
                                violations += 1
    elif hamming_cutoff in (2, 3):
        pep_to_fold = {}
        for f, peps in fold_peptides.items():
            for p in peps:
                pep_to_fold[p] = f
        peps = sorted(pep_to_fold)
        seen = set()  # a close pair can be yielded from several buckets
        for i, j in iter_close_pairs(peps, hamming_cutoff):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            fi, fj = pep_to_fold[peps[i]], pep_to_fold[peps[j]]
            if fi != fj:
                violations += 1
                if violations <= 10:
                    print(f"  leak: {peps[i]} (fold {fi}) ~ "
                          f"{peps[j]} (fold {fj})")
    else:
        raise NotImplementedError(
            f"Leakage verification not implemented for cutoff {hamming_cutoff}")

    if violations == 0:
        print(f"No homology leakage detected across folds "
              f"(direct Hamming <= {hamming_cutoff}).")
    else:
        print(f"{violations} cross-fold homologous pairs found "
              f"(direct Hamming <= {hamming_cutoff})!")

    return violations


def verify_no_peptide_split(df, peptide_col, fold_col, key_label="peptide"):
    """
    Verify that every row with the same key sequence (peptide, or
    full_context in context mode) is in the same fold.
    This catches the case where identical keys from different proteins
    might have been accidentally split across folds.
    """
    peptide_folds = df.groupby(peptide_col)[fold_col].nunique()
    split_peptides = peptide_folds[peptide_folds > 1]

    if len(split_peptides) == 0:
        print(f"No {key_label} split across folds "
              f"(all duplicates in same fold).")
    else:
        print(f"WARNING: {len(split_peptides)} {key_label}s split across folds!")
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


def inspect_bucket_clusters(df, clusters, peptide_to_bucket,
                            peptide_col, label_col, fold_col,
                            n_cv_folds, held_out_index,
                            key_label="peptide"):
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
        print(f"  Total unique {key_label}s:   {total_peptides_in_bucket}")
        print(f"  Cluster size stats:  mean={mean_size:.1f}  "
              f"median={median_size:.1f}  max={max_size}")
        print(f"  Singletons:          {n_singletons:>6}  "
              f"({n_singletons / n_clusters * 100:.1f}% of clusters, "
              f"{n_singletons / total_peptides_in_bucket * 100:.1f}% of {key_label}s)")
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
                  f"{key_label}s: {', '.join(shown)}{suffix}")

    all_sizes = [len(peps) for peps in clusters.values()]
    print(f"\n{'=' * 80}")
    print("GLOBAL CLUSTER SIZE HISTOGRAM")
    print("=" * 80)

    bins = [(1, 1, "Singletons"), (2, 2, "Pairs"), (3, 5, "3-5"),
            (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"),
            (51, 100, "51-100"), (101, float("inf"), ">100")]

    print(f"  {'Size range':<15} {'Clusters':>10} {'%':>8} "
          f"{key_label.capitalize() + 's':>10} {'%':>8}")
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

    ap = argparse.ArgumentParser(
        description="Homology-based data splitting for MHC-I processing prediction.")
    ap.add_argument("--hamming-cutoff", type=int, default=HAMMING_CUTOFF,
                    help="cluster keys at DIRECT Hamming distance <= this "
                         "cutoff (1-3). Default: config HAMMING_CUTOFF (%(default)s).")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated cutoffs to generate in one run, "
                         "e.g. 1,2,3 — one marked split file per cutoff "
                         "(overrides --hamming-cutoff; cannot be combined "
                         "with --split-out).")
    ap.add_argument("--group-by", choices=["peptide", "context"],
                    default="peptide",
                    help="cluster on 'peptide' (9-mer only, legacy behavior) "
                         "or 'context' (ROBUST: window Hamming <= c union "
                         "peptide Hamming <= 1 union identical-flank "
                         "grouping over full_context; width inferred from "
                         "the data; all checks fail hard). "
                         "Default: %(default)s.")
    ap.add_argument("--split-out", default=None,
                    help="output split CSV path (single cutoff only). "
                         "Default: auto marked path such as data/processed/"
                         "df_all_with_folds_ctx-h2.csv; the canonical "
                         "df_all_with_folds.csv is written only by the "
                         "default invocation (peptide, config cutoff, no sweep).")
    args = ap.parse_args()

    if args.sweep is not None:
        if args.split_out:
            ap.error("--split-out cannot be combined with --sweep "
                     "(sweep outputs are auto-named marked files)")
        try:
            cutoffs = [int(c) for c in args.sweep.split(",")]
        except ValueError:
            ap.error(f"--sweep must be a comma-separated list of ints, "
                     f"got: {args.sweep}")
        if not cutoffs or any(c not in (1, 2, 3) for c in cutoffs):
            ap.error("--sweep cutoffs must be 1, 2 or 3 "
                     "(3-block pigeonhole index supports at most 3)")
        if len(set(cutoffs)) != len(cutoffs):
            ap.error(f"--sweep contains duplicate cutoffs: {args.sweep}")
    else:
        if args.hamming_cutoff not in (1, 2, 3):
            ap.error("--hamming-cutoff must be 1, 2 or 3 "
                     "(3-block pigeonhole index supports at most 3)")
        cutoffs = [args.hamming_cutoff]

    group_by = args.group_by
    key_col = PEPTIDE_COL if group_by == "peptide" else "full_context"
    key_label = "peptide" if group_by == "peptide" else "window"
    tag = "pep" if group_by == "peptide" else "updated"

    def resolve_split_out(cutoff):
        """Marked auto-name for every non-default run; the canonical
        SPLIT_DATA_PATH is only touched by the legacy default
        (peptide grouping, config cutoff, single run)."""
        if args.split_out:
            return Path(args.split_out)
        marked = (group_by != "peptide" or cutoff != HAMMING_CUTOFF
                  or args.sweep is not None)
        if not marked:
            return Path(SPLIT_DATA_PATH)
        stem = Path(SPLIT_DATA_PATH).stem
        return Path(SPLIT_DATA_PATH).with_name(
            f"{stem}_{tag}_h{cutoff}{Path(SPLIT_DATA_PATH).suffix}")

    validate_config()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if group_by == "peptide" and args.sweep is None \
            and cutoffs == [HAMMING_CUTOFF]:
        log_tag = ""
    elif len(cutoffs) == 1:
        log_tag = f"_{tag}_h{cutoffs[0]}"
    else:
        log_tag = f"_{tag}_sweep"
    LOG_PATH = LOG_DIR / f"01_datasplit_log_{timestamp}{log_tag}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    print("=" * 80)
    print("  01_DATASPLIT -- HOMOLOGY-BASED DATA SPLIT")
    print("=" * 80)
    print(f"  Timestamp:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data path:        {RAW_DATA_PATH}")
    print(f"  Log path:         {LOG_PATH}")
    print(f"  Group-by:         {group_by}"
          + ("" if group_by == "peptide"
             else " (ROBUST: window-ham <= c  ∪  peptide-ham <= 1  "
                  "∪  identical flank)"))
    print(f"  Hamming cutoffs:  {', '.join(f'<= {c}' for c in cutoffs)}")
    if group_by == "context":
        print("  Guarantees:       no window pair at distance <= c, no")
        print("                    peptide pair at distance <= 1, and no")
        print("                    pair sharing an identical flank cross")
        print("                    folds; peptide pairs 2..c with divergent")
        print("                    flanks MAY split (peptide-hc case).")
        print("                    At c=1 robust == peptide-h1 ∪ flank-id.")
    print(f"  CV folds (k):     {N_CV_FOLDS}")
    print(f"  Held-out bucket:  {HELD_OUT_INDEX}")
    print(f"  Total buckets:    {N_BUCKETS}")
    print(f"  Random state:     {RANDOM_STATE}")
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

    # -- Context-mode checks (fail fast, before any clustering) --
    win_len = None
    if group_by == "context":
        missing = [c for c in ("full_context", "n_flank", "c_flank")
                   if c not in df.columns]
        if missing:
            print(f"\nFATAL: --group-by context requires columns {missing} "
                  f"(not found in input)")
            logger.close()
            sys.exit(1)
        if df["full_context"].isna().any():
            n_na = int(df["full_context"].isna().sum())
            print(f"\nFATAL: 'full_context' has {n_na} missing values")
            logger.close()
            sys.exit(1)
        widths = sorted(df["full_context"].str.len().unique().tolist())
        if widths not in ([19], [29]):
            print(f"\nFATAL: full_context must be a single width of 19 or 29, "
                  f"got widths {widths}")
            logger.close()
            sys.exit(1)
        win_len = widths[0]
        print(f"\nContext mode: full_context width = {win_len} "
              f"({win_len}-mer window) — Hamming computed over the whole "
              f"window including X flank padding")

    # First row index per grouping key — canonical tie-break input for
    # deterministic bucket ordering (greedy sorts by (-size, min row)).
    key_min_row = {}
    for _i, _k in enumerate(df[key_col].tolist()):
        if _k not in key_min_row:
            key_min_row[_k] = _i
    print(f"Key -> first-row index map: {len(key_min_row)} keys")

    # Context -> peptide (a function: the peptide sits inside
    # full_context), input to the robust peptide-edge union.
    ctx_to_peptide = dict(zip(df[key_col], df[PEPTIDE_COL]))

    # -- Duplicate peptide analysis --
    analyze_duplicate_peptides(df, PEPTIDE_COL, LABEL_COL)

    # ================== per-cutoff loop ==================
    written = []
    for cutoff in cutoffs:
        split_out_path = resolve_split_out(cutoff)

        print("\n" + "=" * 80)
        print(f"  SPLIT: group-by={group_by}  Hamming <= {cutoff}  ->  "
              f"{split_out_path.name}")
        print("=" * 80)

        # -- Phase 1: Cluster --
        print("\n" + "=" * 80)
        print("PHASE 1: HOMOLOGY CLUSTERING")
        print("=" * 80)
        unique_keys = df[key_col].unique().tolist()
        print(f"Total rows:          {len(df)}")
        print(f"Unique {key_label}s:     {len(unique_keys)}")
        print(f"Duplicate rows:      {len(df) - len(unique_keys)} "
              f"(same {key_label}, different context)")

        t_cluster_start = time.time()
        if group_by == "context":
            print(f"Clustering ROBUST: window Hamming <= {cutoff}  ∪  "
                  f"peptide Hamming <= 1  ∪  identical flank ...")
            clusters, ctx_class = cluster_context_robust(
                unique_keys, ctx_to_peptide, max_dist=cutoff)
        else:
            print(f"Clustering {key_label}s by Hamming distance <= {cutoff} ...")
            clusters = cluster_by_hamming(unique_keys, max_dist=cutoff)
            ctx_class = None
        t_cluster_end = time.time()

        # sample_counts counts ALL rows per key (duplicates from other
        # proteins included) so balancing uses the true data volume.
        sample_counts = df[key_col].value_counts().to_dict()

        n_clusters = len(clusters)
        largest_cluster_size = max(len(v) for v in clusters.values())
        cluster_rows = [
            sum(sample_counts.get(k, 0) for k in members)
            for members in clusters.values()
        ]
        largest_cluster_rows = max(cluster_rows)
        largest_share = largest_cluster_rows / len(df)
        print(f"Clusters formed:     {n_clusters}")
        print(f"Largest cluster:     {largest_cluster_size} {key_label}s "
              f"({largest_cluster_rows} rows = {largest_share * 100:.1f}% "
              f"of rows)")
        print(f"Clustering time:     {t_cluster_end - t_cluster_start:.2f}s")

        if largest_share > 1 / 6:
            msg = (f"Largest cluster = {largest_cluster_rows} rows "
                   f"({largest_share * 100:.1f}% of rows) exceeds 1/6 — "
                   f"perfect balance across {N_BUCKETS} buckets is "
                   f"impossible.")
            print(msg)
            warnings.warn(msg)

        # c=1 acceptance: robust components must match an INDEPENDENTLY
        # constructed expectation (peptide-h1 ∪ flank-identity) one-to-one.
        # The expectation is built row-based from the n_flank/c_flank
        # COLUMNS + peptide clusters, while the wrapper slices flanks from
        # the context STRING — two implementations must agree.
        # (At c=1 window edges are redundant: win-1 ⊆ pep-1.)
        if group_by == "context" and cutoff == 1:
            exp = UnionFind()
            for k in unique_keys:
                exp.find(k)
            pep_clusters = cluster_by_hamming(
                list(dict.fromkeys(ctx_to_peptide.values())), max_dist=1)
            pep_cls = {}
            for root, members in pep_clusters.items():
                for pep in members:
                    pep_cls[pep] = root
            by_cls = defaultdict(list)
            for k in unique_keys:
                by_cls[pep_cls[ctx_to_peptide[k]]].append(k)
            for members in by_cls.values():
                for x in members[1:]:
                    exp.union(members[0], x)
            for col in ("n_flank", "c_flank"):
                for _, idx in df.groupby(col).groups.items():
                    ks = df.loc[idx, key_col].unique().tolist()
                    for x in ks[1:]:
                        exp.union(ks[0], x)
            exp_root = {k: exp.find(k) for k in unique_keys}

            comp_of_exp = {}
            for comp, members in clusters.items():
                roots = {exp_root[m] for m in members}
                if len(roots) != 1:
                    print(f"\nFATAL: component spans {len(roots)} expected "
                          f"classes — union logic broken")
                    logger.close()
                    sys.exit(1)
                r = next(iter(roots))
                if r in comp_of_exp and comp_of_exp[r] != comp:
                    print("\nFATAL: expected class split across components "
                          "— edge sets B/C not applied")
                    logger.close()
                    sys.exit(1)
                comp_of_exp[r] = comp
            if len(comp_of_exp) != n_clusters:
                print(f"\nFATAL: bijection failed — {len(comp_of_exp)} "
                      f"expected classes vs {n_clusters} components")
                logger.close()
                sys.exit(1)
            print(f"c=1 cluster bijection OK: {n_clusters} components == "
                  f"{n_clusters} (peptide-h1 ∪ flank-identity) classes")

        # -- Phase 2: Assign --
        print("\n" + "=" * 80)
        print("PHASE 2: BUCKET ASSIGNMENT")
        print("=" * 80)
        print(f"Sample counts include all rows (duplicates from different "
              f"proteins counted); clusters ordered by (-size, min row)")

        np.random.seed(RANDOM_STATE)

        t_assign_start = time.time()
        key_to_bucket, bucket_totals = greedy_balanced_partition(
            clusters, N_BUCKETS, sample_counts, key_min_row
        )
        t_assign_end = time.time()

        # Map fold to EVERY row via the grouping key (peptide sequence,
        # or full_context in context mode). All rows with the same key
        # get the same fold assignment.
        df[FOLD_COL] = df[key_col].map(key_to_bucket)
        n_unmapped = int(df[FOLD_COL].isna().sum())
        if n_unmapped:
            print(f"\nFATAL: {n_unmapped} rows have no bucket assignment "
                  f"(key not found in clusters)")
            logger.close()
            sys.exit(1)
        print(f"Assignment time:     {t_assign_end - t_assign_start:.2f}s")
        print(f"Bucket totals:       {bucket_totals}")

        # -- Verify: no homology leakage --
        print("\n" + "=" * 80)
        print("LEAKAGE VERIFICATION")
        print("=" * 80)

        # Verify 1: no close key pairs across folds (window-c)
        t_verify_start = time.time()
        n_violations = verify_no_homology_leakage(df, key_col, FOLD_COL,
                                                  hamming_cutoff=cutoff)
        t_verify_end = time.time()
        print(f"Verification time: {t_verify_end - t_verify_start:.2f}s")

        # Verify 2: no identical key split across folds
        # (confirms all duplicate rows stay together)
        print()
        n_splits = verify_no_peptide_split(df, key_col, FOLD_COL,
                                           key_label=key_label)

        # FAILSAFE (robust mode): no peptide pair at Hamming <= 1 may
        # cross folds — this is edge set B's guarantee, HARD fail.
        n_pep1_violations = 0
        if group_by == "context":
            print("\nFAILSAFE: peptide Hamming <= 1 across folds")
            n_pep1_violations = verify_no_homology_leakage(
                df, PEPTIDE_COL, FOLD_COL, hamming_cutoff=1)
            if n_pep1_violations > 0:
                print(f"\nFATAL: FAILSAFE violated — {n_pep1_violations} "
                      f"peptide-ham<=1 pairs cross folds")
                logger.close()
                sys.exit(1)
            # flank-identity failsafe (edge set C): every n_flank and
            # c_flank value must live in exactly one fold
            n_nf = int((df.groupby("n_flank")[FOLD_COL].nunique() > 1).sum())
            n_cf = int((df.groupby("c_flank")[FOLD_COL].nunique() > 1).sum())
            if n_nf or n_cf:
                print(f"\nFATAL: FLANK FAILSAFE violated — {n_nf} n_flank "
                      f"and {n_cf} c_flank values straddle folds")
                logger.close()
                sys.exit(1)
            print("Flank-identity failsafe PASSED (0 straddling flanks)")
            if n_violations > 0:
                print(f"\nFATAL: {n_violations} window-ham<={cutoff} pairs "
                      f"cross folds")
                logger.close()
                sys.exit(1)

        # Identical PEPTIDES split across folds — must be 0 in robust
        # mode (implied by the pep-h1 failsafe; asserted anyway).
        pep_fold_n = df.groupby(PEPTIDE_COL)[FOLD_COL].nunique()
        n_pep_split = int((pep_fold_n > 1).sum())
        if group_by == "context":
            if n_pep_split > 0:
                print(f"\nFATAL: {n_pep_split} identical peptides split "
                      f"across folds (must be 0 under robust grouping)")
                logger.close()
                sys.exit(1)
            print(f"\nOK: identical peptides split across folds: "
                  f"{n_pep_split} (must be 0)")
        else:
            print(f"\nIdentical peptides split across folds: {n_pep_split}")

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

        inspect_bucket_clusters(df, clusters, key_to_bucket,
                                key_col, LABEL_COL, FOLD_COL,
                                N_CV_FOLDS, HELD_OUT_INDEX,
                                key_label=key_label)

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
        split_out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(split_out_path, index=False)
        written.append(split_out_path)
        print(f"\nSaved to: {split_out_path}")

        t_end = time.time()
        print("\n" + "=" * 80)
        print(f"RUN SUMMARY: group-by={group_by}, Hamming <= {cutoff}")
        print("=" * 80)
        print(f"  Runtime so far:   {t_end - t_start:.2f}s")
        print(f"  Input rows:       {len(df)}")
        print(f"  Unique {key_label}s:  {len(unique_keys)}")
        print(f"  Duplicate rows:   {len(df) - len(unique_keys)}")
        print(f"  Clusters:         {n_clusters}")
        print(f"  Largest cluster:  {largest_cluster_rows} rows "
              f"({largest_share * 100:.1f}% of rows)"
              f"{'  [>1/6: balance capped]' if largest_share > 1 / 6 else ''}")
        print(f"  Bucket sizes:     {bucket_totals}")
        print(f"  Buckets:          {N_BUCKETS} ({N_CV_FOLDS} CV + 1 held-out)")
        print(f"  Homology leakage: {'NONE' if n_violations == 0 else f'{n_violations} violations'}")
        print(f"  {key_label} splits:  {'NONE' if n_splits == 0 else f'{n_splits} {key_label}s split'}")
        if group_by == "context":
            print(f"  Pep-h1 failsafe:  PASSED (0 violations, hard-checked)")
            print(f"  Flank failsafe:   PASSED (0 straddling flanks, hard-checked)")
            print(f"  Identical peptides split: {n_pep_split} (must be 0)")
        print(f"  Output:           {split_out_path}")
        print("=" * 80)

    # ================== final summary ==================
    t_end = time.time()
    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"  Total runtime:    {t_end - t_start:.2f}s")
    print(f"  Group-by:         {group_by}"
          + (f" ({win_len}-mer window)" if win_len else ""))
    print(f"  Splits written:   {len(written)}")
    for p in written:
        print(f"    {p}")
    print(f"  Log:              {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()

