"""
03_prepare_embeddings.py
Align HDF5 embeddings with the split dataset (labels, folds, metadata).

Reads the raw embedding HDF5 (ESM-C or ESM-IF), aligns rows with
df_all_with_folds.csv, validates the join, and saves a prepared HDF5
with canonical dataset names that 04_modelling.py can load directly.

Usage:
    python 03_prepare_embeddings.py --embedding esmc
    python 03_prepare_embeddings.py --embedding esmif
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import time

from config import (
    SPLIT_DATA_PATH, LOG_DIR, PREPARED_EMBEDDING_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    PEPTIDE_COL, LABEL_COL, FOLD_COL,
    EMBEDDING_REGIONS,
    get_embedding_source, validate_config,
)


# ──────────────────────────────────────────────
# 0. LOGGER
# ──────────────────────────────────────────────

class Logger:
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
# 1. LOAD RAW EMBEDDINGS
# ──────────────────────────────────────────────

def load_raw_embeddings(h5_path, emb_source):
    """
    Load a raw embedding HDF5 using the schema defined in
    EMBEDDING_SOURCES.  Returns (data_dict, metadata_dict).

    The data_dict uses CANONICAL keys:
      - "peptide_emb", "n_flank_emb", "c_flank_emb" (remapped from source names)
      - "peptide_ids", "uniprot_ids" (remapped from source column names)
      - "row_indices", "start", "end" (if available)
    """
    region_map = emb_source["region_map"]         # canonical → HDF5 name
    pep_id_col = emb_source["peptide_id_col"]
    uni_id_col = emb_source["uniprot_id_col"]

    data = {}
    metadata = {}

    with h5py.File(h5_path, "r") as f:
        # Metadata attributes
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            metadata[key] = val

        # List all datasets for reporting
        all_datasets = list(f.keys())

        # Embedding regions → canonical names
        for canonical, h5_name in region_map.items():
            canonical_key = f"{canonical}_emb"
            if h5_name in f:
                data[canonical_key] = f[h5_name][:]
            else:
                print(f"  WARNING: Region '{h5_name}' not found in HDF5")

        # Peptide IDs
        if pep_id_col in f:
            arr = f[pep_id_col][:]
            if arr.dtype.kind in ("S", "O"):
                arr = np.array([x.decode("utf-8") if isinstance(x, bytes) else x
                                for x in arr])
            data["peptide_ids"] = arr

        # UniProt IDs
        if uni_id_col in f:
            arr = f[uni_id_col][:]
            if arr.dtype.kind in ("S", "O"):
                arr = np.array([x.decode("utf-8") if isinstance(x, bytes) else x
                                for x in arr])
            data["uniprot_ids"] = arr

        # Row indices (ESM-C has them, ESM-IF does not)
        if emb_source["has_row_indices"] and "row_indices" in f:
            data["row_indices"] = f["row_indices"][:]

        # Start/end positions (ESM-C has them, ESM-IF does not)
        if emb_source["has_start_end"]:
            if "start" in f:
                data["start"] = f["start"][:]
            if "end" in f:
                data["end"] = f["end"][:]

        # Fallback flags (optional)
        if "fallback_flag" in f:
            data["fallback_flag"] = f["fallback_flag"][:]

    return data, metadata, all_datasets


# ──────────────────────────────────────────────
# 2. ALIGNMENT
# ──────────────────────────────────────────────

def align_by_row_indices(emb_data, df):
    """
    Align using row_indices stored in the HDF5 (point to positions
    in the original df_all.csv / df_all_with_folds.csv).
    """
    emb_row_idx = emb_data["row_indices"]
    n_df = len(df)

    valid_mask = (emb_row_idx >= 0) & (emb_row_idx < n_df)
    emb_positions = np.where(valid_mask)[0]
    df_rows = emb_row_idx[valid_mask].astype(int)

    report = {
        "method": "row_indices",
        "n_emb_total": len(emb_row_idx),
        "n_df_total": n_df,
        "n_aligned": int(valid_mask.sum()),
        "n_emb_unmatched": int((~valid_mask).sum()),
        "n_df_unmatched": n_df - len(set(df_rows)),
    }

    return emb_positions, df_rows, report


def align_by_peptide_uniprot(emb_data, df):
    """
    Align by joining on (peptide, uniprot_id).
    Used when row_indices are not available (e.g. ESM-IF).
    """
    emb_peps = emb_data["peptide_ids"]
    emb_unis = emb_data["uniprot_ids"]

    emb_df = pd.DataFrame({
        "peptide": emb_peps,
        "uniprot_id": emb_unis,
        "_emb_idx": np.arange(len(emb_peps)),
    })

    df_indexed = df.reset_index().rename(columns={"index": "_df_idx"})

    # Join on peptide + uniprot_id
    merged = emb_df.merge(
        df_indexed[["peptide", "uniprot_id", "_df_idx"]],
        on=["peptide", "uniprot_id"],
        how="inner",
    )

    # If multiple df rows match one embedding row (e.g. same peptide+protein
    # at different positions), keep all — each is a distinct sample in the
    # split data with its own fold assignment
    # If multiple embedding rows match one df row, keep first
    merged = merged.drop_duplicates(subset=["_df_idx"], keep="first")

    emb_positions = merged["_emb_idx"].values.astype(int)
    df_rows = merged["_df_idx"].values.astype(int)

    # Sort by df row index for consistent ordering
    sort_order = np.argsort(df_rows)
    emb_positions = emb_positions[sort_order]
    df_rows = df_rows[sort_order]

    report = {
        "method": "peptide+uniprot_id",
        "n_emb_total": len(emb_peps),
        "n_df_total": len(df),
        "n_aligned": len(merged),
        "n_emb_unmatched": len(emb_peps) - len(set(emb_positions)),
        "n_df_unmatched": len(df) - len(set(df_rows)),
    }

    return emb_positions, df_rows, report


# ──────────────────────────────────────────────
# 3. VALIDATION
# ──────────────────────────────────────────────

def validate_alignment(emb_data, df, emb_positions, df_rows, emb_source):
    """
    Spot-check that aligned rows match on peptide sequence.
    Adapts to available ID columns.
    """
    n_check = min(500, len(emb_positions))
    check_idx = np.random.choice(len(emb_positions), n_check, replace=False)

    mismatches_pep = 0
    mismatches_uni = 0

    for i in check_idx:
        emb_idx = emb_positions[i]
        df_idx = df_rows[i]

        emb_pep = emb_data["peptide_ids"][emb_idx]
        df_pep = df.iloc[df_idx][PEPTIDE_COL]

        if emb_pep != df_pep:
            mismatches_pep += 1

        if "uniprot_ids" in emb_data and "uniprot_id" in df.columns:
            emb_uni = emb_data["uniprot_ids"][emb_idx]
            df_uni = df.iloc[df_idx]["uniprot_id"]
            if emb_uni != df_uni:
                mismatches_uni += 1

    return mismatches_pep, mismatches_uni, n_check


def report_zero_vectors(emb_data, emb_positions):
    """Count and report zero embedding vectors per region."""
    print(f"\n  Zero vector report:")
    print(f"  {'Region':<20} {'Zeros':>8} {'%':>8}")
    print(f"  {'-' * 38}")

    zero_counts = {}
    for region in EMBEDDING_REGIONS:
        if region not in emb_data:
            continue
        arr = emb_data[region][emb_positions]
        norms = np.linalg.norm(arr, axis=1)
        n_zero = int((norms == 0).sum())
        pct = n_zero / len(emb_positions) * 100
        zero_counts[region] = n_zero
        print(f"  {region:<20} {n_zero:>8} {pct:>7.2f}%")

    total_zero = sum(zero_counts.values())
    if total_zero > 0:
        print(f"\n  NOTE: {total_zero} total zero vectors detected.")
        print(f"  These represent samples without structural/sequence coverage.")
        print(f"  They will map to the PCA mean after transformation.")

    return zero_counts


def report_label_distribution(labels, folds):
    """Print label counts per fold."""
    print(f"\n  Label distribution per fold:")
    print(f"  {'Fold':<8} {'Total':>8} {'Pos':>8} {'Neg':>8} {'Pos %':>8}")
    print(f"  {'-' * 42}")
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        n = int(mask.sum())
        n_pos = int(labels[mask].sum())
        n_neg = n - n_pos
        pct = n_pos / n * 100 if n > 0 else 0
        tag = " (held-out)" if fold == HELD_OUT_INDEX else ""
        print(f"  {fold:<8} {n:>8} {n_pos:>8} {n_neg:>8} {pct:>7.1f}%{tag}")


# ──────────────────────────────────────────────
# 4. SAVE PREPARED HDF5
# ──────────────────────────────────────────────

def save_prepared(out_path, emb_data, emb_positions, df, df_rows,
                  emb_metadata, embedding_key, emb_source):
    """
    Save a self-contained HDF5 with:
      - Embedding arrays under canonical names (peptide_emb, n_flank_emb, c_flank_emb)
      - Labels, folds, identifiers
      - Full provenance metadata
    """
    labels = df.iloc[df_rows][LABEL_COL].values.astype(np.int32)
    folds = df.iloc[df_rows][FOLD_COL].values.astype(np.int32)
    peptides = df.iloc[df_rows][PEPTIDE_COL].values
    uniprot_ids = (df.iloc[df_rows]["uniprot_id"].values
                   if "uniprot_id" in df.columns
                   else np.array([""] * len(df_rows)))
    starts = (df.iloc[df_rows]["start"].values.astype(np.int32)
              if "start" in df.columns
              else np.zeros(len(df_rows), dtype=np.int32))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as f:
        # Embedding arrays under CANONICAL names
        for region in EMBEDDING_REGIONS:
            if region in emb_data:
                arr = emb_data[region][emb_positions].astype(np.float32)
                f.create_dataset(region, data=arr,
                                 compression="gzip", compression_opts=4)
                print(f"    {region}: shape={arr.shape}")

        # Labels and folds
        f.create_dataset("labels", data=labels)
        f.create_dataset("folds", data=folds)
        f.create_dataset("starts", data=starts)

        # Identifiers
        dt_str = h5py.string_dtype()
        f.create_dataset("peptide_seqs", data=peptides.astype(str), dtype=dt_str)
        f.create_dataset("uniprot_ids", data=uniprot_ids.astype(str), dtype=dt_str)

        # Row mapping back to df_all_with_folds.csv
        f.create_dataset("df_row_indices", data=df_rows.astype(np.int64))

        # Metadata
        f.attrs["embedding_key"] = embedding_key
        f.attrs["display_name"] = emb_source["display_name"]
        f.attrs["n_samples"] = len(df_rows)
        f.attrs["emb_dim"] = emb_source["emb_dim"]
        f.attrs["n_regions"] = sum(1 for r in EMBEDDING_REGIONS if r in emb_data)
        f.attrs["n_folds"] = int(np.max(folds)) + 1
        f.attrs["held_out_index"] = HELD_OUT_INDEX
        f.attrs["prepared_timestamp"] = datetime.now().isoformat()
        f.attrs["source_csv"] = str(SPLIT_DATA_PATH)
        f.attrs["source_h5"] = str(emb_source["raw_path"])

        # Copy original embedding metadata
        for k, v in emb_metadata.items():
            f.attrs[f"orig_{k}"] = str(v)

    return labels, folds


# ──────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="03_prepare_embeddings: align embeddings with split data",
    )
    parser.add_argument(
        "--embedding", type=str, required=True,
        help="Embedding key (e.g. 'esmc', 'esmif')",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    embedding_key = args.embedding

    validate_config()
    emb_source = get_embedding_source(embedding_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"03_prepare_embeddings_{embedding_key}_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    # -- Header --
    print("=" * 80)
    print(f"  03_PREPARE_EMBEDDINGS -- {emb_source['display_name'].upper()}")
    print("=" * 80)
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Embedding key:   {embedding_key}")
    print(f"  Raw HDF5:        {emb_source['raw_path']}")
    print(f"  Split data:      {SPLIT_DATA_PATH}")
    print(f"  Output:          {emb_source['prepared_path']}")
    print(f"  Log:             {LOG_PATH}")
    print(f"\n  Source schema:")
    print(f"    Embedding dim:   {emb_source['emb_dim']}")
    print(f"    Region mapping:")
    for canonical, h5_name in emb_source["region_map"].items():
        print(f"      {canonical:<12} → {h5_name}")
    print(f"    Peptide ID col:  {emb_source['peptide_id_col']}")
    print(f"    UniProt ID col:  {emb_source['uniprot_id_col']}")
    print(f"    Has row_indices: {emb_source['has_row_indices']}")
    print(f"    Has start/end:   {emb_source['has_start_end']}")
    print("=" * 80)

    # -- Load split data --
    print(f"\nLoading split data: {SPLIT_DATA_PATH}")
    df = pd.read_csv(SPLIT_DATA_PATH)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Folds: {sorted(df[FOLD_COL].unique())}")

    n_pos = int((df[LABEL_COL] == 1).sum())
    n_neg = int((df[LABEL_COL] == 0).sum())
    print(f"  Positives: {n_pos}  Negatives: {n_neg}")

    # -- Load raw embeddings --
    raw_path = emb_source["raw_path"]
    if not raw_path.exists():
        print(f"\nFATAL: Embedding file not found: {raw_path}")
        logger.close()
        sys.exit(1)

    print(f"\nLoading raw embeddings: {raw_path}")
    emb_data, emb_metadata, all_datasets = load_raw_embeddings(raw_path, emb_source)

    # Report
    print(f"\n  HDF5 metadata:")
    for k, v in sorted(emb_metadata.items()):
        print(f"    {k:<25} {v}")

    print(f"\n  HDF5 datasets (raw):")
    for ds in sorted(all_datasets):
        print(f"    {ds}")

    print(f"\n  Loaded canonical datasets:")
    for key, arr in sorted(emb_data.items()):
        if hasattr(arr, "shape"):
            print(f"    {key:<25} shape={str(arr.shape):<20} dtype={arr.dtype}")
        else:
            print(f"    {key:<25} type={type(arr).__name__}")

    # Verify regions
    available_regions = [r for r in EMBEDDING_REGIONS if r in emb_data]
    missing_regions = [r for r in EMBEDDING_REGIONS if r not in emb_data]
    print(f"\n  Canonical regions available: {available_regions}")
    if missing_regions:
        print(f"  WARNING: Missing canonical regions: {missing_regions}")

    if not available_regions:
        print(f"\nFATAL: No embedding regions found after remapping")
        print(f"  Raw datasets: {all_datasets}")
        print(f"  Region map: {emb_source['region_map']}")
        logger.close()
        sys.exit(1)

    emb_dim = emb_data[available_regions[0]].shape[1]
    n_emb_samples = emb_data[available_regions[0]].shape[0]
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Embedding samples: {n_emb_samples}")

    # Verify dim matches config
    if emb_dim != emb_source["emb_dim"]:
        print(f"  WARNING: Config says emb_dim={emb_source['emb_dim']}, "
              f"actual={emb_dim}")

    # -- Alignment --
    print(f"\n{'=' * 80}")
    print("ALIGNMENT")
    print("=" * 80)

    if emb_source["has_row_indices"] and "row_indices" in emb_data:
        print(f"  Strategy: row_indices (direct positional mapping)")
        emb_positions, df_rows, align_report = align_by_row_indices(emb_data, df)
    else:
        print(f"  Strategy: peptide + uniprot_id join "
              f"(row_indices not available)")
        emb_positions, df_rows, align_report = align_by_peptide_uniprot(emb_data, df)

    print(f"\n  Alignment results:")
    print(f"    Method:              {align_report['method']}")
    print(f"    Embedding samples:   {align_report['n_emb_total']}")
    print(f"    Split data rows:     {align_report['n_df_total']}")
    print(f"    Successfully aligned: {align_report['n_aligned']}")
    print(f"    Embedding unmatched: {align_report['n_emb_unmatched']}")
    print(f"    Split data unmatched: {align_report['n_df_unmatched']}")

    coverage = align_report["n_aligned"] / align_report["n_df_total"] * 100
    print(f"    Coverage: {coverage:.1f}% of split data")

    if align_report["n_aligned"] == 0:
        print(f"\nFATAL: No rows could be aligned.")
        print(f"  Check that the embedding HDF5 corresponds to the same dataset")
        print(f"  version as {SPLIT_DATA_PATH}")
        logger.close()
        sys.exit(1)

    if coverage < 95:
        print(f"\n  WARNING: Coverage below 95% ({coverage:.1f}%)")
        print(f"  {align_report['n_df_unmatched']} split-data rows have no embedding")
    elif coverage < 100:
        print(f"\n  NOTE: {align_report['n_df_unmatched']} split-data rows unmatched "
              f"({100 - coverage:.1f}%)")

    # -- Validate alignment --
    print(f"\n  Spot-checking alignment ...")
    np.random.seed(42)
    mis_pep, mis_uni, n_checked = validate_alignment(
        emb_data, df, emb_positions, df_rows, emb_source,
    )
    if mis_pep == 0 and mis_uni == 0:
        print(f"  [OK] {n_checked} spot-checks passed "
              f"(0 peptide mismatches, 0 uniprot mismatches)")
    else:
        print(f"  Peptide mismatches:  {mis_pep}/{n_checked}")
        print(f"  UniProt mismatches:  {mis_uni}/{n_checked}")
        if mis_pep > 0:
            print(f"  WARNING: Peptide mismatches detected — alignment unreliable!")

    # -- Zero vector report --
    zero_counts = report_zero_vectors(emb_data, emb_positions)

    # -- Save --
    print(f"\n{'=' * 80}")
    print("SAVING PREPARED EMBEDDINGS")
    print("=" * 80)

    out_path = emb_source["prepared_path"]
    print(f"  Output: {out_path}")
    labels, folds = save_prepared(
        out_path, emb_data, emb_positions, df, df_rows,
        emb_metadata, embedding_key, emb_source,
    )
    print(f"  [OK] Saved successfully")

    # -- Label distribution per fold --
    report_label_distribution(labels, folds)

    # -- Embedding stats per fold --
    print(f"\n  Embedding norm stats per fold ({available_regions[0]}):")
    aligned_emb = emb_data[available_regions[0]][emb_positions]
    norms = np.linalg.norm(aligned_emb, axis=1)

    print(f"  {'Fold':<8} {'Samples':>8} {'Norm med':>10} {'Norm std':>10} "
          f"{'Zero vecs':>10}")
    print(f"  {'-' * 50}")
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_norms = norms[mask]
        n_zero = int((fold_norms == 0).sum())
        print(f"  {fold:<8} {int(mask.sum()):>8} "
              f"{np.median(fold_norms):>10.4f} {np.std(fold_norms):>10.4f} "
              f"{n_zero:>10}")

    # -- Footer --
    t_end = time.time()
    file_size_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"  Runtime:          {t_end - t_start:.1f}s")
    print(f"  Embedding:        {emb_source['display_name']}")
    print(f"  Embedding dim:    {emb_dim}")
    print(f"  Aligned samples:  {align_report['n_aligned']}")
    print(f"  Coverage:         {coverage:.1f}%")
    print(f"  Regions:          {available_regions}")
    print(f"  Zero vectors:     {zero_counts}")
    print(f"  Output:           {out_path}")
    print(f"  Output size:      {file_size_mb:.1f} MB")
    print(f"  Log:              {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()