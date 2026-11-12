#!/usr/bin/env python3
"""
make_variant_dataset.py
Build dataset variants by applying a row mask to df_all.csv.

Rules (selected by --variant):
    remove715negonly    negatives whose protein contributes no positive
                        (715 orphan proteins, 2,271 negatives).
    remove_dup_windows  one row per identical context window
                        (full_context = n_flank|peptide|c_flank); a window
                        carrying both labels keeps the positive row. Drops
                        1,107 rows (361 duplicate positive windows, 746
                        negatives: 352 label-conflicting + 394 same-label).
                        Same peptide with DIFFERENT flanks is allowed and
                        kept.

Also filters every per-peptide encoding CSV with the identical row mask
(same order) so load_alternate_csv()'s strict row-alignment check in
04_modelling stays satisfied.

Read-only on existing files; writes *_<variant>.csv only.

Usage:
    python src/pipeline/python/make_variant_dataset.py [--variant remove715negonly]
    python src/pipeline/python/make_variant_dataset.py --variant remove_dup_windows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

ENCODING_CSVS = [
    "df_sparse.csv", "df_blosum50.csv", "df_blosum62.csv", "df_blosum80.csv",
    "df_kidera.csv", "df_atchley.csv", "df_compositional.csv",
]


def orphan_mask(df: pd.DataFrame) -> pd.Series:
    """Keep all positives + negatives from proteins that contribute a positive."""
    pos_prot = set(df.loc[df["label"] == 1, "uniprot_id"])
    return (df["label"] == 1) | df["uniprot_id"].isin(pos_prot)


def dup_window_mask(df: pd.DataFrame) -> pd.Series:
    """Keep one row per identical context window; prefer a positive row.

    A window (full_context = n_flank|peptide|c_flank) seen multiple times
    keeps its first positive occurrence if any positive exists, else its
    first row. Same peptide with different flanks is untouched.
    """
    key = df["full_context"].fillna("<NA>")
    is_pos = df["label"] == 1
    first_any = ~key.duplicated()
    first_pos = pd.Series(False, index=df.index)
    first_pos.loc[key[is_pos].drop_duplicates().index] = True
    group_has_pos = key.isin(set(key[is_pos]))
    keep = first_any.copy()
    keep[group_has_pos] = first_pos[group_has_pos]
    return keep


RULES = {
    "remove715negonly": (orphan_mask, 2271),
    "remove_dup_windows": (dup_window_mask, 1107),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="remove715negonly",
                    choices=sorted(RULES),
                    help="variant name (selects the row-mask rule)")
    ap.add_argument("--expect-dropped", type=int, default=None,
                    help="expected dropped rows (default: per-rule; "
                         "guards against upstream drift)")
    args = ap.parse_args()

    mask_fn, default_expect = RULES[args.variant]
    expect = args.expect_dropped if args.expect_dropped is not None else default_expect

    src = DATA_DIR / "df_all.csv"
    df = pd.read_csv(src)
    print(f"Loaded {src.name}: {len(df)} rows, {df['uniprot_id'].nunique()} proteins, rule={args.variant}")

    mask = mask_fn(df)

    n_dropped = int((~mask).sum())
    assert n_dropped == expect, (
        f"expected {expect} dropped rows, got {n_dropped} — "
        "df_all.csv differs from the audited version"
    )

    out = df[mask].reset_index(drop=True)
    if args.variant == "remove715negonly":
        orphan_check = set(out.loc[out["label"] == 0, "uniprot_id"]) - set(
            out.loc[out["label"] == 1, "uniprot_id"])
        assert not orphan_check, f"orphan proteins remain: {len(orphan_check)}"
    else:  # remove_dup_windows
        assert not out["full_context"].duplicated().any(), \
            "duplicate context windows remain"
        assert out["full_context"].nunique() == len(out), "full_context not unique"
        n_peps_before, n_peps_after = df["peptide"].nunique(), out["peptide"].nunique()
        assert n_peps_after == n_peps_before, \
            f"peptides lost: {n_peps_before} -> {n_peps_after}"
        # breakdown for the report
        pre_keys = df.groupby("full_context")["label"].nunique()
        n_conflicted = int((pre_keys > 1).sum())
        n_pos_dup = int((df["label"] == 1)[~mask].sum())
        n_neg_dup = n_dropped - n_pos_dup
        print(f"  label-conflicting windows resolved: {n_conflicted}")
        print(f"  dropped: {n_pos_dup} duplicate positive windows, "
              f"{n_neg_dup} negative windows ({n_dropped} total)")

    out_path = DATA_DIR / f"df_all_{args.variant}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path.name}: {len(out)} rows "
          f"({int((out['label'] == 1).sum())} pos / {int((out['label'] == 0).sum())} neg), "
          f"{out['uniprot_id'].nunique()} proteins")

    # Base-row mask (reused by prepare_variant_h5.py to subset prepared H5s)
    mask_path = DATA_DIR / f"variant_mask_{args.variant}.csv"
    pd.DataFrame({"base_row_index": df.index[mask]}).to_csv(mask_path, index=False)
    print(f"Wrote {mask_path.name}: {len(out)} kept base row indices")

    # Encoding CSVs: identical mask, strict peptide+label alignment to df_all
    for name in ENCODING_CSVS:
        enc_path = DATA_DIR / name
        if not enc_path.exists():
            print(f"  skip {name} (not found)")
            continue
        enc = pd.read_csv(enc_path)
        assert len(enc) == len(df), f"{name}: row count {len(enc)} != {len(df)}"
        assert (enc["peptide"] == df["peptide"]).all(), f"{name}: peptide misaligned"
        assert (enc["label"] == df["label"]).all(), f"{name}: label misaligned"
        out_enc = enc[mask].reset_index(drop=True)
        enc_out_path = DATA_DIR / name.replace(".csv", f"_{args.variant}.csv")
        out_enc.to_csv(enc_out_path, index=False)
        print(f"  wrote {enc_out_path.name}: {len(out_enc)} rows")


if __name__ == "__main__":
    sys.exit(main())
