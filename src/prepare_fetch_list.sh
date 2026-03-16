#!/usr/bin/env bash
# prepare_fetch_list.sh
# Usage: bash src/prepare_fetch_list.sh \
#            data/processed/pos_EL_8-to-14mers_epitopes_hla0201.csv \
#            data/processed/structures

set -euo pipefail

INPUT_CSV="${1}"
OUT_DIR="${2}"

mkdir -p "${OUT_DIR}/logs"

python3 - <<'PYEOF' "${INPUT_CSV}" "${OUT_DIR}"
import sys
import pandas as pd

csv_path = sys.argv[1]
out_dir  = sys.argv[2]

df = pd.read_csv(csv_path)

print("=== COLUMNS ===")
print(df.columns.tolist())
print(f"\n=== SHAPE: {df.shape} ===")

# ── Drop missing uniprot_ids ─────────────────────────────────────────────────
before = len(df)
df = df.dropna(subset=['uniprot_id'])
print(f"Dropped {before - len(df)} rows with missing uniprot_id")

# ── Classify UniProt IDs ─────────────────────────────────────────────────────
canonical_mask = df['uniprot_id'].str.match(r'^[A-Z0-9]{6,10}$',       na=False)
isoform_mask   = df['uniprot_id'].str.match(r'^[A-Z0-9]{6,10}-\d+$',   na=False)
other_mask     = ~canonical_mask & ~isoform_mask

print(f"\nCanonical UniProt IDs : {canonical_mask.sum()} rows")
print(f"Isoform UniProt IDs   : {isoform_mask.sum()} rows")
print(f"Truly invalid         : {other_mask.sum()} rows")

# ── Handle truly invalid ─────────────────────────────────────────────────────
if other_mask.sum() > 0:
    print("\nTruly invalid examples:")
    print(df[other_mask]['uniprot_id'].head(10).tolist())
    df[other_mask].to_csv(f"{out_dir}/logs/truly_invalid_ids.tsv",
                          sep='\t', index=False)
    print(f"Saved to {out_dir}/logs/truly_invalid_ids.tsv")

# ── Split canonical and isoform ──────────────────────────────────────────────
df_canonical = df[canonical_mask].copy()
df_isoform   = df[isoform_mask].copy()

df_isoform['base_uniprot_id'] = (
    df_isoform['uniprot_id']
    .str.extract(r'^([A-Z0-9]{6,10})-\d+$')
)

# ── Aggregate: one row per unique ID ────────────────────────────────────────
# Take min(start) and max(end) across all peptides from same protein
# so coverage check later covers ALL peptides from that source protein
canonical_grouped = df_canonical.groupby('uniprot_id').agg(
    start=('start', 'min'),
    end=('end', 'max'),
    n_peptides=('peptide', 'count')
).reset_index()
# base_id == uniprot_id for canonical
canonical_grouped['base_id'] = canonical_grouped['uniprot_id']

isoform_grouped = df_isoform.groupby('uniprot_id').agg(
    start=('start', 'min'),
    end=('end', 'max'),
    n_peptides=('peptide', 'count'),
    base_id=('base_uniprot_id', 'first')
).reset_index()

print(f"\nUnique canonical proteins : {len(canonical_grouped)}")
print(f"Unique isoform proteins   : {len(isoform_grouped)}")

# ── Show samples ─────────────────────────────────────────────────────────────
print("\n--- Canonical sample ---")
print(canonical_grouped.head(5).to_string())
print("\n--- Isoform sample ---")
print(isoform_grouped.head(5).to_string())

# ── Save ─────────────────────────────────────────────────────────────────────
# Columns: uniprot_id | start | end | n_peptides | base_id
canonical_grouped.to_csv(f"{out_dir}/logs/fetch_list_canonical.tsv",
                         sep='\t', index=False, header=False)
isoform_grouped.to_csv(f"{out_dir}/logs/fetch_list_isoforms.tsv",
                       sep='\t', index=False, header=False)

print(f"\nSaved:")
print(f"  {out_dir}/logs/fetch_list_canonical.tsv  ({len(canonical_grouped)} entries)")
print(f"  {out_dir}/logs/fetch_list_isoforms.tsv   ({len(isoform_grouped)} entries)")
print(f"\nTotal proteins to fetch structures for: {len(canonical_grouped) + len(isoform_grouped)}")
PYEOF
