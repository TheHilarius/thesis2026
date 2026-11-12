#!/usr/bin/env python3
"""
prepare_variant_h5.py
Build variant prepared-embedding HDF5s by SUBSETTING the base prepared
files with a variant row mask — no re-embedding, no re-run of
03_prepare_embeddings.

Why not 03_prepare: align_window_by_row_indices maps raw-H5 row_indices
positionally into the (shorter) variant df with no permutation/identity
validation — it silently misaligns on row-dropped variants. The base
prepared H5s are already in base-df row order, and a variant df is the
base df minus rows (same relative order), so a positional subset is exact.

Inputs (per variant, built by make_variant_dataset.py + 01_datasplit.py):
    data/processed/variant_mask_<variant>.csv     kept base row indices
    data/processed/df_all_with_folds_<variant>.csv  variant rows + folds
    data/processed/embeddings_prepared/*_prepared.h5  (base files only)

Output (per base prepared file):
    embeddings_prepared/<stem>_<variant>.h5  — same name config.py's
    _variant() redirect produces for prepared_path under
    THESIS_VARIANT=<variant>.

Rewrites from the variant fold CSV: folds, labels, peptide_seqs,
uniprot_ids, starts, df_row_indices (renumbered 0..n-1). Every other
dataset is subset with the mask. Self-checks after write: row count,
folds/labels/peptide identity vs variant CSV, sampled windows bit-identical
to base rows.

Usage (from repo root):
    python src/pipeline/python/prepare_variant_h5.py --variant remove_dup_windows
    python src/pipeline/python/prepare_variant_h5.py --variant X --only esmc300m_win_padtoken_prepared
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
PREPARED_DIR = DATA_DIR / "embeddings_prepared"

# Base prepared files only — derived variants (flank5, remove715negonly,
# first21, ...) are rebuilt from their own base, not re-derived here.
DERIVED_MARKERS = ("remove715", "flank5", "first21", "dim21")

# datasets rewritten from the variant fold CSV instead of subset
REWRITE_KEYS = {"folds", "labels"}


def variant_name(base: str, variant: str) -> str:
    """Mirror config.py _variant(): stem + '_' + variant + suffix."""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{variant}{p.suffix}"))


def load_variant(variant: str):
    mask_path = DATA_DIR / f"variant_mask_{variant}.csv"
    fold_path = DATA_DIR / f"df_all_with_folds_{variant}.csv"
    for p in (mask_path, fold_path):
        if not p.exists():
            sys.exit(f"FATAL: {p} not found — run make_variant_dataset.py "
                     f"--variant {variant} and 01_datasplit with "
                     f"THESIS_VARIANT={variant} first.")
    base_idx = pd.read_csv(mask_path)["base_row_index"].to_numpy()
    vdf = pd.read_csv(fold_path)
    if len(base_idx) != len(vdf):
        sys.exit(f"FATAL: mask ({len(base_idx)}) vs variant df ({len(vdf)}) "
                 "length mismatch")
    return base_idx, vdf


def subset_file(src: Path, dst: Path, base_idx: np.ndarray, vdf: pd.DataFrame,
                chunk: int, base_n: int) -> dict:
    t0 = time.time()
    n = len(vdf)

    def arr_from_variant(name: str):
        if name == "folds":
            return vdf["fold"].to_numpy().astype(np.int32)
        if name == "labels":
            return vdf["label"].to_numpy().astype(np.int32)
        if name == "peptide_seqs":
            return np.char.encode(vdf["peptide"].to_numpy(dtype=str))
        if name == "uniprot_ids":
            col = "uniprot_id" if "uniprot_id" in vdf.columns else None
            return (np.char.encode(vdf[col].to_numpy(dtype=str))
                    if col else None)
        if name == "starts":
            col = "start" if "start" in vdf.columns else None
            return vdf[col].to_numpy().astype(np.int32) if col else None
        if name == "df_row_indices":
            return np.arange(n, dtype=np.int64)
        return None

    checks = {}
    with h5py.File(src, "r") as f1, h5py.File(dst, "w") as f2:
        for k, v in f1.attrs.items():
            f2.attrs[k] = v
        f2.attrs["variant"] = Path(dst).name
        f2.attrs["built_from"] = src.name
        f2.attrs["build_script"] = "prepare_variant_h5.py"
        f2.attrs["n_samples"] = n

        for name in f1.keys():
            d1 = f1[name]
            if not isinstance(d1, h5py.Dataset):
                continue
            if d1.shape[0] != base_n and name not in REWRITE_KEYS:
                # legacy/aux dataset not aligned to df rows — skip loudly
                print(f"    skip {name}: first dim {d1.shape[0]} != base rows "
                      f"{base_n}")
                continue
            rewrite = arr_from_variant(name)
            if name in REWRITE_KEYS and rewrite is None:
                continue
            if rewrite is not None:
                f2.create_dataset(name, data=rewrite)
                continue
            # subset in chunks (window tensors are GB-scale)
            dtype = d1.dtype
            shape = (n,) + d1.shape[1:]
            opts = {}
            if d1.compression:
                opts = dict(compression=d1.compression,
                            compression_opts=d1.compression_opts)
            d2 = f2.create_dataset(name, shape=shape, dtype=dtype, **opts)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                d2[s:e] = d1[base_idx[s:e]]
            checks[name] = True

        # identity self-checks vs variant CSV
        for ds, col in (("peptide_seqs", "peptide"), ("uniprot_ids", "uniprot_id"),
                        ("labels", "label"), ("folds", "fold")):
            if ds in f2 and col in vdf.columns:
                got = f2[ds][:]
                if isinstance(got[0], (bytes, np.bytes_)):
                    got = np.char.decode(got)
                want = vdf[col].to_numpy()
                if not (got == want).all():
                    bad = int((got != want).sum())
                    sys.exit(f"FATAL: {dst.name}: {ds} mismatch on {bad} rows")

    # sampled window bit-identity vs base prepared
    with h5py.File(src, "r") as f1, h5py.File(dst, "r") as f2:
        tensor_keys = [k for k in f2.keys()
                       if isinstance(f2[k], h5py.Dataset)
                       and f2[k].ndim >= 2 and f2[k].shape[0] == n
                       and k not in REWRITE_KEYS
                       and k not in ("peptide_seqs", "uniprot_ids",
                                     "df_row_indices", "starts")]
        if tensor_keys:
            k = tensor_keys[0]
            rows = np.linspace(0, n - 1, num=min(5, n), dtype=int)
            if not np.array_equal(f1[k][base_idx[rows]], f2[k][rows]):
                sys.exit(f"FATAL: {dst.name}: sampled rows of '{k}' "
                         "differ from base prepared — subset misaligned")
            print(f"    identity check OK: {k} {f2[k].shape}, "
                  f"{n} rows, {time.time() - t0:.1f}s")

    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, help="variant name")
    ap.add_argument("--only", action="append", default=None,
                    help="subset only these prepared stems (repeatable)")
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    base_idx, vdf = load_variant(args.variant)
    base_n = len(pd.read_csv(DATA_DIR / "df_all_with_folds.csv", usecols=["peptide"]))
    assert base_idx.max() < base_n, "mask exceeds base df length"
    print(f"Variant {args.variant}: {len(vdf)} rows "
          f"({int((vdf['label'] == 1).sum())} pos / {int((vdf['label'] == 0).sum())} neg), "
          f"base rows {base_n}, mask covers {base_idx.min()}..{base_idx.max()} "
          f"(ordered: {np.all(np.diff(base_idx) > 0)})")

    base_files = sorted(
        p for p in PREPARED_DIR.glob("*_prepared.h5")
        if not any(m in p.name for m in DERIVED_MARKERS)
    )
    if args.only:
        base_files = [p for p in base_files
                      if any(o in p.name for o in args.only)]
    if not base_files:
        sys.exit("FATAL: no base prepared files matched")

    for src in base_files:
        dst = Path(variant_name(str(src), args.variant))
        print(f"\n{src.name} -> {dst.name}")
        subset_file(src, dst, base_idx, vdf, args.chunk, base_n)
        print(f"  wrote {dst.name} ({dst.stat().st_size / 1e9:.2f} GB)")

    print("\nDone. Run modelling with THESIS_VARIANT="
          f"{args.variant} to pick these up.")


if __name__ == "__main__":
    main()
