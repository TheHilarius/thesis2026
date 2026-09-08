#!/usr/bin/env python3
"""
fetch_uniprot_fastas.py — Download full-length protein sequences from UniProt.

Reads unique uniprot_ids from pos_EL_all_epitopes_hla0201.csv (which includes
isoforms resolved by 02_load_iedb_data.R). Queries UniProt with
includeIsoform=true to ensure isoform-specific sequences are returned as
separate FASTA entries.

Outputs:
  data/raw/fasta/combined_full_length.fasta       — all accessions (with isoforms)
  data/raw/fasta/combined_positives_only.fasta    — proteins with positive epitopes only
"""
import pandas as pd
import requests
import re
import sys
from pathlib import Path
import time


def main():
    # Paths
    pos_el_path = Path("data/processed/pos_EL_all_epitopes_hla0201.csv")
    output_fasta = Path("data/raw/fasta/combined_full_length.fasta")
    positives_fasta = Path("data/raw/fasta/combined_positives_only.fasta")

    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading processed epitope data: {pos_el_path}...")

    try:
        df = pd.read_csv(pos_el_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if 'uniprot_id' not in df.columns:
        print("[ERROR] 'uniprot_id' column not found!")
        sys.exit(1)

    # Strict UniProt accession filter
    UNIPROT_PATTERN = re.compile(r'^(?![A-Z]{3}[0-9])[A-Z][A-Z0-9]{5,9}(-[0-9]+)?$')
    unique_ids = [uid for uid in df['uniprot_id'].dropna().unique()
                  if UNIPROT_PATTERN.match(str(uid))]

    n_isoforms = sum(1 for uid in unique_ids if re.search(r'-\d+$', str(uid)))
    print(f"Found {len(unique_ids)} unique UniProt IDs "
          f"({n_isoforms} isoforms, {len(unique_ids) - n_isoforms} canonical)")

    # Batch download from UniProt REST API
    batch_size = 50
    batches = [unique_ids[i:i + batch_size] for i in range(0, len(unique_ids), batch_size)]

    print(f"Downloading from UniProt in {len(batches)} batches (includeIsoform=true)...")

    total_downloaded = 0
    with open(output_fasta, "w") as out_f:
        for i, batch in enumerate(batches, 1):
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            url = "https://rest.uniprot.org/uniprotkb/stream"
            params = {
                "query": query,
                "format": "fasta",
                "includeIsoform": "true",
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                fasta_data = response.text
                if fasta_data.strip():
                    out_f.write(fasta_data)
                    if not fasta_data.endswith("\n"):
                        out_f.write("\n")
                    total_downloaded += fasta_data.count(">")

                if i % 10 == 0 or i == len(batches):
                    print(f"  Batch {i}/{len(batches)} ({total_downloaded} sequences so far)")
                time.sleep(1)

            except Exception as e:
                print(f"[ERROR] Failed to fetch batch {i}: {e}")

    print(f"\n=== SUMMARY ===")
    print(f"Queried:             {len(unique_ids)} IDs")
    print(f"Downloaded:          {total_downloaded} FASTA sequences")
    print(f"Output:              {output_fasta}")

    # ── Filter to proteins with positive epitopes ──────────────────────────────
    print(f"\nFiltering to proteins with positive epitopes...")
    pos_ids = set(df['uniprot_id'].dropna().unique())

    from Bio import SeqIO
    all_records = list(SeqIO.parse(str(output_fasta), "fasta"))
    pos_records = [r for r in all_records if r.id.split("|")[1] in pos_ids]
    SeqIO.write(pos_records, str(positives_fasta), "fasta")

    print(f"Full FASTA:           {len(all_records)} entries")
    print(f"Positives-only FASTA: {len(pos_records)} entries")
    print(f"Output:               {positives_fasta}")

    if total_downloaded < len(unique_ids):
        print(f"\n[NOTE] {len(unique_ids) - total_downloaded} IDs not found on UniProt "
              f"(obsolete or deleted).")


if __name__ == "__main__":
    main()
