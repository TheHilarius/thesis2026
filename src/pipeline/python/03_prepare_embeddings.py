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
    get_embedding_source, validate_config,
)


def get_embedding_regions(emb_source):
    """Derive canonical region names from emb_source region_map.
    
    For 3-region entries: {"peptide": "peptide_emb", ...} → ["peptide_emb", ...]
    For context entries:  {"context": "context_emb"}     → ["context_emb"]
    """
    return [f"{canonical}_emb" for canonical in emb_source["region_map"]]


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
    EMBEDDING_SOURCES.  Returns (data_dict, metadata_dict, all_dataset_names).

    The data_dict uses CANONICAL keys:
      - "context_emb" (single mean-pooled n-flank+peptide+c-flank window)
      - "peptide_ids", "uniprot_ids"
      - "row_indices", "start", "end" (if available)
    """
    region_map = emb_source["region_map"]
    pep_id_col = emb_source["peptide_id_col"]
    uni_id_col = emb_source["uniprot_id_col"]

    data = {}
    metadata = {}

    with h5py.File(h5_path, "r") as f:
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            metadata[key] = val

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

        # Row indices
        if emb_source["has_row_indices"] and "row_indices" in f:
            data["row_indices"] = f["row_indices"][:]

        # Start/end positions
        if emb_source["has_start_end"]:
            if "start" in f:
                data["start"] = f["start"][:]
            if "end" in f:
                data["end"] = f["end"][:]

        # Fallback flags
        if "fallback_flag" in f:
            data["fallback_flag"] = f["fallback_flag"][:]

    return data, metadata, all_datasets


# ──────────────────────────────────────────────
# 2. ALIGNMENT
# ──────────────────────────────────────────────

def align_by_row_indices(emb_data, df):
    """
    Align using row_indices stored in the HDF5.
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
    Used when row_indices are not available or unreliable.
    """
    emb_peps = emb_data["peptide_ids"]
    emb_unis = emb_data["uniprot_ids"]

    emb_df = pd.DataFrame({
        "peptide": emb_peps,
        "uniprot_id": emb_unis,
        "_emb_idx": np.arange(len(emb_peps)),
    })

    df_indexed = df.reset_index().rename(columns={"index": "_df_idx"})

    merged = emb_df.merge(
        df_indexed[["peptide", "uniprot_id", "_df_idx"]],
        on=["peptide", "uniprot_id"],
        how="inner",
    )

    # Keep one embedding per df row (first match)
    merged = merged.drop_duplicates(subset=["_df_idx"], keep="first")

    emb_positions = merged["_emb_idx"].values.astype(int)
    df_rows = merged["_df_idx"].values.astype(int)

    # Sort by df row index
    sort_order = np.argsort(df_rows)
    emb_positions = emb_positions[sort_order]
    df_rows = df_rows[sort_order]

    report = {
        "method": "peptide+uniprot_id",
        "n_emb_total": len(emb_peps),
        "n_df_total": len(df),
        "n_aligned": len(df_rows),
        "n_emb_unmatched": len(emb_peps) - len(set(emb_positions)),
        "n_df_unmatched": len(df) - len(set(df_rows)),
    }

    return emb_positions, df_rows, report


def align_by_peptide_uniprot_start(emb_data, df):
    """
    Align by joining on (peptide, uniprot_id, start).
    Most precise — used when embeddings have start positions.
    """
    emb_peps = emb_data["peptide_ids"]
    emb_unis = emb_data["uniprot_ids"]
    emb_starts = emb_data["start"]

    emb_df = pd.DataFrame({
        "peptide": emb_peps,
        "uniprot_id": emb_unis,
        "start": emb_starts,
        "_emb_idx": np.arange(len(emb_peps)),
    })

    df_indexed = df.reset_index().rename(columns={"index": "_df_idx"})

    merged = emb_df.merge(
        df_indexed[["peptide", "uniprot_id", "start", "_df_idx"]],
        on=["peptide", "uniprot_id", "start"],
        how="inner",
    )

    merged = merged.drop_duplicates(subset=["_df_idx"], keep="first")

    emb_positions = merged["_emb_idx"].values.astype(int)
    df_rows = merged["_df_idx"].values.astype(int)

    sort_order = np.argsort(df_rows)
    emb_positions = emb_positions[sort_order]
    df_rows = df_rows[sort_order]

    report = {
        "method": "peptide+uniprot_id+start",
        "n_emb_total": len(emb_peps),
        "n_df_total": len(df),
        "n_aligned": len(df_rows),
        "n_emb_unmatched": len(emb_peps) - len(set(emb_positions)),
        "n_df_unmatched": len(df) - len(set(df_rows)),
    }

    return emb_positions, df_rows, report


# ──────────────────────────────────────────────
# 3. VALIDATION
# ──────────────────────────────────────────────

def spot_check_alignment(emb_data, df, emb_positions, df_rows, n_check=500):
    """
    Spot-check that aligned rows match on peptide sequence and uniprot ID.
    Returns (peptide_mismatches, uniprot_mismatches, n_checked).
    """
    n_check = min(n_check, len(emb_positions))
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


def report_zero_vectors(emb_data, emb_positions, embedding_regions):
    """Count and report zero embedding vectors per region."""
    print(f"\n  Zero vector report:")
    print(f"  {'Region':<20} {'Zeros':>8} {'%':>8}")
    print(f"  {'-' * 38}")

    zero_counts = {}
    for region in embedding_regions:
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
                  emb_metadata, embedding_key, emb_source, embedding_regions):
    """
    Save a self-contained HDF5 with canonical dataset names.
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

    # Convert strings to bytes for h5py compatibility
    peptides_bytes = np.array([s.encode("utf-8") for s in peptides])
    uniprot_bytes = np.array([str(s).encode("utf-8") for s in uniprot_ids])

    with h5py.File(out_path, "w") as f:
        # Embedding arrays under CANONICAL names
        for region in embedding_regions:
            if region in emb_data:
                arr = emb_data[region][emb_positions].astype(np.float32)
                f.create_dataset(region, data=arr,
                                 compression="gzip", compression_opts=4)
                print(f"    {region}: shape={arr.shape}")

        # Labels and folds
        f.create_dataset("labels", data=labels)
        f.create_dataset("folds", data=folds)
        f.create_dataset("starts", data=starts)

        # Identifiers (as byte strings — universally compatible with h5py)
        f.create_dataset("peptide_seqs", data=peptides_bytes)
        f.create_dataset("uniprot_ids", data=uniprot_bytes)

        # Row mapping back to df_all_with_folds.csv
        f.create_dataset("df_row_indices", data=np.array(df_rows).astype(np.int64))

        # Metadata
        f.attrs["embedding_key"] = embedding_key
        f.attrs["display_name"] = emb_source["display_name"]
        f.attrs["n_samples"] = len(df_rows)
        f.attrs["emb_dim"] = emb_source["emb_dim"]
        f.attrs["n_regions"] = len(embedding_regions)
        f.attrs["n_folds"] = int(np.max(folds)) + 1
        f.attrs["held_out_index"] = HELD_OUT_INDEX
        f.attrs["prepared_timestamp"] = datetime.now().isoformat()
        f.attrs["source_csv"] = str(SPLIT_DATA_PATH)
        f.attrs["source_h5"] = str(emb_source["raw_path"])

        for k, v in emb_metadata.items():
            f.attrs[f"orig_{k}"] = str(v)

    return labels, folds


# ──────────────────────────────────────────────
# 5. WINDOW EMBEDDING PATH (kind == "windows")
# ──────────────────────────────────────────────
#
# Handles the fixed-29 per-residue window HDF5s (B pipeline). Differs from the
# legacy mean-pooled path in three ways:
#   - The payload is a full (N, 29, D) tensor, copied through by streaming
#     (never held whole in RAM twice).
#   - pad_mask (True = padded slot) is copied and reduced to 2 pad-count
#     features (N-flank count, C-flank count).
#   - Alignment uses row_indices directly (a validated permutation 0..N-1),
#     and the output is reordered into df row order so downstream modelling
#     can index prepared row i == df row i.

# Slot layout for the 29-slot window (fixed by embed_windows_esmif.py):
#   0-9 N-flank (right-aligned), 10-18 peptide (9), 19-28 C-flank (left-aligned)
WINDOW_FLANK = 10


def load_raw_window_embeddings(h5_path, emb_source):
    """
    Load a window HDF5 (kind == "windows").

    Returns (data, metadata, all_datasets) where data holds the small arrays
    (pad_mask, pad_counts, ids, row_indices, start/end, status) plus the
    window shape/dtype, but NOT the windows tensor itself.  The tensor is
    streamed at save time.
    """
    ws = emb_source["window_ds"]
    mask_ds = emb_source["pad_mask_ds"]
    n_pad_ds = emb_source.get("n_pad_ds")
    c_pad_ds = emb_source.get("c_pad_ds")
    status_ds = emb_source.get("status_ds")
    pep_id_col = emb_source["peptide_id_col"]
    uni_id_col = emb_source["uniprot_id_col"]

    data = {}
    metadata = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.attrs:
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            metadata[key] = val

        all_datasets = list(f.keys())

        if ws not in f:
            raise KeyError(
                f"Window dataset '{ws}' not found in {h5_path}. "
                f"Available: {all_datasets}"
            )

        data["window_shape"] = tuple(f[ws].shape)
        data["window_dtype"] = f[ws].dtype

        # pad_mask (small: N x 29 bool)
        if mask_ds in f:
            data["pad_mask"] = f[mask_ds][:].astype(bool)
        else:
            raise KeyError(
                f"pad_mask dataset '{mask_ds}' not found in {h5_path}"
            )

        # pad-count features derived from the mask (True = padded)
        mask = data["pad_mask"]
        n_pad = mask[:, :WINDOW_FLANK].sum(axis=1).astype(np.int32)
        c_pad = mask[:, WINDOW_FLANK + 9:].sum(axis=1).astype(np.int32)
        data["pad_counts"] = np.stack([n_pad, c_pad], axis=1)  # (N, 2)

        # status (optional)
        if status_ds and status_ds in f:
            data["status"] = f[status_ds][:]
        else:
            data["status"] = None

        # cross-check derived pad counts against stored n_pad/c_pad
        # (n_pad/c_pad use -1 sentinel for notfound rows; skip those)
        if data["status"] is not None and n_pad_ds and n_pad_ds in f:
            raw_n = f[n_pad_ds][:]
            ok = data["status"] == 0
            if ok.sum() > 0:
                bad_n = int((raw_n[ok] != data["pad_counts"][ok, 0]).sum())
                if bad_n > 0:
                    print(f"  WARNING: {bad_n} rows have n_pad != mask-derived "
                          f"n-flank pad count (status==0)")
        if data["status"] is not None and c_pad_ds and c_pad_ds in f:
            raw_c = f[c_pad_ds][:]
            ok = data["status"] == 0
            if ok.sum() > 0:
                bad_c = int((raw_c[ok] != data["pad_counts"][ok, 1]).sum())
                if bad_c > 0:
                    print(f"  WARNING: {bad_c} rows have c_pad != mask-derived "
                          f"c-flank pad count (status==0)")

        # identifiers
        if pep_id_col in f:
            arr = f[pep_id_col][:]
            if arr.dtype.kind in ("S", "O"):
                arr = np.array([x.decode("utf-8") if isinstance(x, bytes) else x
                                for x in arr])
            data["peptide_ids"] = arr
        if uni_id_col in f:
            arr = f[uni_id_col][:]
            if arr.dtype.kind in ("S", "O"):
                arr = np.array([x.decode("utf-8") if isinstance(x, bytes) else x
                                for x in arr])
            data["uniprot_ids"] = arr

        # row_indices (authoritative for windows)
        if emb_source["has_row_indices"] and "row_indices" in f:
            data["row_indices"] = f["row_indices"][:]

        if emb_source["has_start_end"]:
            if "start" in f:
                data["start"] = f["start"][:]
            if "end" in f:
                data["end"] = f["end"][:]

    return data, metadata, all_datasets


def align_window_by_row_indices(emb_data, df):
    """
    Align window rows to df rows via row_indices (permutation 0..N-1).

    Returns (emb_positions, df_rows, report) with BOTH arrays in df row order,
    so prepared row i corresponds to df row i.
    """
    ri = emb_data["row_indices"]
    n_df = len(df)
    n_emb = len(ri)

    valid = (ri >= 0) & (ri < n_df)
    n_valid = int(valid.sum())
    n_unique = len(np.unique(ri[valid])) if n_valid > 0 else 0

    inv = np.full(n_df, -1, dtype=np.int64)
    inv[ri[valid]] = np.where(valid)[0]

    df_rows = np.where(inv >= 0)[0]
    emb_positions = inv[df_rows]

    report = {
        "method": "row_indices (window)",
        "n_emb_total": n_emb,
        "n_df_total": n_df,
        "n_aligned": n_valid,
        "n_emb_unmatched": n_emb - n_valid,
        "n_df_unmatched": n_df - n_unique,
    }

    return emb_positions, df_rows, report


def report_window_coverage(emb_data, emb_positions):
    """Report status breakdown + zero-window count for a window embedding."""
    status = emb_data.get("status")
    pad_counts = emb_data["pad_counts"][emb_positions]

    print(f"\n  Window coverage report:")
    if status is not None:
        st = status[emb_positions]
        print(f"  {'Status':<8} {'Count':>8}  Legend")
        for code, label in [(0, "primary"), (1, "af2_fallback"),
                            (2, "missing"), (3, "notfound")]:
            n = int((st == code).sum())
            if n > 0:
                print(f"  {code:<8} {n:>8}  {label}")
    else:
        print(f"  (no status dataset)")

    print(f"  Pad-count stats (mask-derived, aligned rows):")
    print(f"    N-flank pad: mean={pad_counts[:, 0].mean():.3f} "
          f"max={pad_counts[:, 0].max()}")
    print(f"    C-flank pad: mean={pad_counts[:, 1].mean():.3f} "
          f"max={pad_counts[:, 1].max()}")
    n_fullpad = int((pad_counts[:, 0] == WINDOW_FLANK).sum())
    if n_fullpad > 0:
        print(f"    Fully-padded N-flank rows: {n_fullpad}")
    return pad_counts


def save_prepared_windows(out_path, raw_path, emb_source, emb_data,
                          emb_positions, df, df_rows, metadata, embedding_key):
    """
    Stream the (N, 29, D) window tensor into a prepared HDF5 in df row order,
    alongside pad_mask, pad_counts, labels, folds and identifiers.
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

    peptides_bytes = np.array([s.encode("utf-8") for s in peptides])
    uniprot_bytes = np.array([str(s).encode("utf-8") for s in uniprot_ids])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ws = emb_source["window_ds"]
    mask_ds = emb_source["pad_mask_ds"]
    n_out = len(emb_positions)
    window = int(metadata.get("window", emb_data["window_shape"][1]))
    emb_dim = int(metadata.get("emb_dim", emb_data["window_shape"][2]))
    pad_mode = metadata.get("pad_mode", emb_source.get("pad_mode", "unknown"))

    CHUNK = 1024

    with h5py.File(raw_path, "r") as fsrc, h5py.File(out_path, "w") as fdst:
        src = fsrc[ws]

        # windows: streamed copy in df row order
        dst = fdst.create_dataset(
            "windows", shape=(n_out, window, emb_dim), dtype=np.float32,
            chunks=(64, window, emb_dim),
        )
        for i0 in range(0, n_out, CHUNK):
            idx = emb_positions[i0:i0 + CHUNK]
            order = np.argsort(idx)
            buf = src[np.sort(idx)]              # increasing h5py read
            dst[i0:i0 + CHUNK] = buf[np.argsort(order)]  # contiguous write

        # pad_mask + pad_counts (small)
        mask = fsrc[mask_ds][:].astype(bool)
        fdst.create_dataset("pad_mask", data=mask[emb_positions])
        fdst.create_dataset("pad_counts",
                            data=emb_data["pad_counts"][emb_positions])

        # labels / folds / identifiers
        fdst.create_dataset("labels", data=labels)
        fdst.create_dataset("folds", data=folds)
        fdst.create_dataset("starts", data=starts)
        fdst.create_dataset("peptide_seqs", data=peptides_bytes)
        fdst.create_dataset("uniprot_ids", data=uniprot_bytes)
        fdst.create_dataset("df_row_indices",
                            data=np.array(df_rows).astype(np.int64))

        # metadata
        fdst.attrs["embedding_key"] = embedding_key
        fdst.attrs["display_name"] = emb_source["display_name"]
        fdst.attrs["kind"] = "windows"
        fdst.attrs["pad_mode"] = pad_mode
        fdst.attrs["window"] = window
        fdst.attrs["emb_dim"] = emb_dim
        fdst.attrs["n_samples"] = n_out
        fdst.attrs["n_folds"] = int(np.max(folds)) + 1
        fdst.attrs["held_out_index"] = HELD_OUT_INDEX
        fdst.attrs["prepared_timestamp"] = datetime.now().isoformat()
        fdst.attrs["source_csv"] = str(SPLIT_DATA_PATH)
        fdst.attrs["source_h5"] = str(raw_path)

        for k, v in metadata.items():
            fdst.attrs[f"orig_{k}"] = str(v)

    return labels, folds


# ──────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="03_prepare_embeddings: align embeddings with split data",
    )
    parser.add_argument(
        "--embedding", type=str, required=True,
        help="Embedding key (e.g. 'esmc', 'esmif', 'esmif_zero')",
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
    print(f"    Kind:            {emb_source.get('kind', 'legacy')}")
    print(f"    Embedding dim:   {emb_source['emb_dim']}")
    if emb_source.get("kind") == "windows":
        print(f"    Window dataset:  {emb_source['window_ds']}")
        print(f"    Pad mask dataset:{emb_source['pad_mask_ds']}")
        print(f"    Pad mode:        {emb_source.get('pad_mode', '?')}")
    else:
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

    # ══════════════════════════════════════════════════════════════════
    # WINDOW BRANCH (kind == "windows")
    # ══════════════════════════════════════════════════════════════════
    if emb_source.get("kind") == "windows":
        raw_path = emb_source["raw_path"]
        if not raw_path.exists():
            print(f"\nFATAL: Embedding file not found: {raw_path}")
            logger.close()
            sys.exit(1)

        print(f"\nLoading raw window embeddings: {raw_path}")
        emb_data, emb_metadata, all_datasets = load_raw_window_embeddings(
            raw_path, emb_source,
        )

        print(f"\n  HDF5 metadata:")
        for k, v in sorted(emb_metadata.items()):
            print(f"    {k:<25} {v}")
        print(f"\n  HDF5 datasets (raw):")
        for ds in sorted(all_datasets):
            print(f"    {ds}")
        print(f"\n  Window tensor: {emb_data['window_shape']} "
              f"({emb_data['window_dtype']})")
        print(f"  pad_counts shape: {emb_data['pad_counts'].shape}")

        # emb_dim is authoritative from the H5 attr, config value is expected
        actual_dim = emb_data["window_shape"][2]
        if actual_dim != emb_source["emb_dim"]:
            print(f"  WARNING: config emb_dim={emb_source['emb_dim']}, "
                  f"H5 emb_dim={actual_dim}")

        # -- Alignment via row_indices (df-ordered) --
        print(f"\n{'=' * 80}")
        print("ALIGNMENT (row_indices → df order)")
        print("=" * 80)
        emb_positions, df_rows, ar = align_window_by_row_indices(emb_data, df)
        print(f"    Aligned: {ar['n_aligned']} / {ar['n_df_total']} df rows")
        print(f"    Unmatched embeddings: {ar['n_emb_unmatched']}")
        print(f"    Unmatched df rows:    {ar['n_df_unmatched']}")

        if ar["n_aligned"] == 0:
            print(f"\nFATAL: No rows aligned via row_indices.")
            logger.close()
            sys.exit(1)

        # -- Coverage / pad report --
        report_window_coverage(emb_data, emb_positions)

        # -- Save --
        print(f"\n{'=' * 80}")
        print("SAVING PREPARED WINDOW EMBEDDINGS")
        print("=" * 80)
        out_path = emb_source["prepared_path"]
        print(f"  Output: {out_path}")
        labels, folds = save_prepared_windows(
            out_path, raw_path, emb_source, emb_data,
            emb_positions, df, df_rows, emb_metadata, embedding_key,
        )
        print(f"  [OK] Saved successfully")

        report_label_distribution(labels, folds)

        t_end = time.time()
        file_size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"\n{'=' * 80}")
        print("RUN SUMMARY (WINDOWS)")
        print("=" * 80)
        print(f"  Runtime:          {t_end - t_start:.1f}s")
        print(f"  Embedding:        {emb_source['display_name']}")
        print(f"  Pad mode:         {emb_metadata.get('pad_mode', '?')}")
        print(f"  Window dim:       {emb_data['window_shape']}")
        print(f"  Aligned samples:  {ar['n_aligned']}")
        print(f"  Output:           {out_path}")
        print(f"  Output size:      {file_size_mb:.1f} MB")
        print(f"  Log:              {LOG_PATH}")
        print("=" * 80)

        logger.close()
        sys.exit(0)

    # -- Load raw embeddings --
    raw_path = emb_source["raw_path"]
    if not raw_path.exists():
        print(f"\nFATAL: Embedding file not found: {raw_path}")
        logger.close()
        sys.exit(1)

    print(f"\nLoading raw embeddings: {raw_path}")
    emb_data, emb_metadata, all_datasets = load_raw_embeddings(raw_path, emb_source)

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

    available_regions = get_embedding_regions(emb_source)
    missing_regions = [r for r in available_regions if r not in emb_data]
    print(f"\n  Canonical regions available: {available_regions}")
    if missing_regions:
        print(f"  WARNING: Missing canonical regions: {missing_regions}")

    if not available_regions:
        print(f"\nFATAL: No embedding regions found after remapping")
        logger.close()
        sys.exit(1)

    emb_dim = emb_data[available_regions[0]].shape[1]
    n_emb_samples = emb_data[available_regions[0]].shape[0]
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Embedding samples: {n_emb_samples}")

    if emb_dim != emb_source["emb_dim"]:
        print(f"  WARNING: Config says emb_dim={emb_source['emb_dim']}, actual={emb_dim}")

    # -- Alignment --
    # Strategy: try the most precise method available, validate it,
    # and fall back to less precise methods if validation fails.
    print(f"\n{'=' * 80}")
    print("ALIGNMENT")
    print("=" * 80)

    np.random.seed(42)
    emb_positions = None
    df_rows = None
    align_report = None

    # Build list of alignment strategies in order of preference
    strategies = []

    if (emb_source["has_start_end"] and "start" in emb_data
            and "peptide_ids" in emb_data and "uniprot_ids" in emb_data):
        strategies.append(("peptide+uniprot_id+start", align_by_peptide_uniprot_start))

    if emb_source["has_row_indices"] and "row_indices" in emb_data:
        strategies.append(("row_indices", align_by_row_indices))

    if "peptide_ids" in emb_data and "uniprot_ids" in emb_data:
        strategies.append(("peptide+uniprot_id", align_by_peptide_uniprot))

    for strategy_name, strategy_fn in strategies:
        print(f"\n  Trying strategy: {strategy_name}")

        try:
            ep, dr, ar = strategy_fn(emb_data, df)
        except Exception as e:
            print(f"  Strategy failed with error: {e}")
            continue

        if ar["n_aligned"] == 0:
            print(f"  No rows aligned — skipping")
            continue

        # Spot-check
        mis_pep, mis_uni, n_checked = spot_check_alignment(
            emb_data, df, ep, dr, n_check=500,
        )

        coverage = ar["n_aligned"] / ar["n_df_total"] * 100
        print(f"    Aligned: {ar['n_aligned']} ({coverage:.1f}% coverage)")
        print(f"    Spot-check: {mis_pep} peptide mismatches, "
              f"{mis_uni} uniprot mismatches out of {n_checked}")

        if mis_pep == 0:
            print(f"  [OK] Strategy '{strategy_name}' passed validation")
            emb_positions = ep
            df_rows = dr
            align_report = ar
            break
        else:
            print(f"  Strategy '{strategy_name}' FAILED validation — "
                  f"trying next strategy")

    if emb_positions is None:
        print(f"\nFATAL: All alignment strategies failed.")
        print(f"  The embedding HDF5 and split CSV may be from different "
              f"data versions.")
        logger.close()
        sys.exit(1)

    # Print final alignment report
    print(f"\n  Final alignment results:")
    print(f"    Method:              {align_report['method']}")
    print(f"    Embedding samples:   {align_report['n_emb_total']}")
    print(f"    Split data rows:     {align_report['n_df_total']}")
    print(f"    Successfully aligned: {align_report['n_aligned']}")
    print(f"    Embedding unmatched: {align_report['n_emb_unmatched']}")
    print(f"    Split data unmatched: {align_report['n_df_unmatched']}")

    coverage = align_report["n_aligned"] / align_report["n_df_total"] * 100
    print(f"    Coverage: {coverage:.1f}% of split data")

    if coverage < 100:
        n_missing = align_report["n_df_unmatched"]
        print(f"\n  NOTE: {n_missing} split-data rows have no embedding match")
        print(f"  These samples will be excluded from embedding-based models.")
        print(f"  This is a TEMPORARY limitation until embeddings are recomputed")
        print(f"  on the current version of the split data.")

    # -- Zero vector report --
    zero_counts = report_zero_vectors(emb_data, emb_positions, available_regions)

    # -- Save --
    print(f"\n{'=' * 80}")
    print("SAVING PREPARED EMBEDDINGS")
    print("=" * 80)

    out_path = emb_source["prepared_path"]
    print(f"  Output: {out_path}")
    labels, folds = save_prepared(
        out_path, emb_data, emb_positions, df, df_rows,
        emb_metadata, embedding_key, emb_source, available_regions,
    )
    print(f"  [OK] Saved successfully")

    # -- Label distribution --
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
    print(f"  Alignment method: {align_report['method']}")
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
