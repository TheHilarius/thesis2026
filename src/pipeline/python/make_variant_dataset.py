#!/usr/bin/env python3
"""
make_variant_dataset.py
Build the "remove715negonly" dataset variant: df_all.csv minus the
negatives whose protein contributes no positive (715 orphan proteins,
2,271 negatives). Also filters every per-peptide encoding CSV with the
identical row mask (same order) so load_alternate_csv()'s strict
row-alignment check in 04_modelling stays satisfied.

Read-only on existing files; writes *_remove715negonly.csv only.

Usage:
    python src/pipeline/python/make_variant_dataset.py [--variant remove715negonly]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="remove715negonly")
    ap.add_argument("--expect-orphans", type=int, default=2271,
                    help="expected dropped negatives (guard against upstream drift)")
    args = ap.parse_args()

    src = DATA_DIR / "df_all.csv"
    df = pd.read_csv(src)
    print(f"Loaded {src.name}: {len(df)} rows, {df['uniprot_id'].nunique()} proteins")

    # Row mask: keep all positives + negatives from positive proteins
    pos_prot = set(df.loc[df["label"] == 1, "uniprot_id"])
    mask = (df["label"] == 1) | df["uniprot_id"].isin(pos_prot)

    n_dropped = int((~mask).sum())
    assert n_dropped == args.expect_orphans, (
        f"expected {args.expect_orphans} orphan negatives, got {n_dropped} — "
        "df_all.csv differs from the audited version"
    )

    out = df[mask].reset_index(drop=True)
    orphan_check = set(out.loc[out["label"] == 0, "uniprot_id"]) - set(
        out.loc[out["label"] == 1, "uniprot_id"])
    assert not orphan_check, f"orphan proteins remain: {len(orphan_check)}"

    out_path = DATA_DIR / f"df_all_{args.variant}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path.name}: {len(out)} rows "
          f"({int((out['label'] == 1).sum())} pos / {int((out['label'] == 0).sum())} neg), "
          f"{out['uniprot_id'].nunique()} proteins, 0 orphans")

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
