#!/usr/bin/env python3
"""
fetch_uniprot_fastas.py — Download full-length protein sequences from UniProt.

Reads unique uniprot_ids from pos_EL_all_epitopes csv (which includes
isoforms resolved by 02_load_iedb_data.R). Queries UniProt with
includeIsoform=true to ensure isoform-specific sequences are returned as
separate FASTA entries.

Allele switch (args only, logic untouched — same exact code per allele).
Each allele downloads its FULL set to SEPARATE files (duplicates across
files OK — R unions with distinct()). No skip-list, no mixing on disk.
  legacy (no args = old run byte-identical):
    python src/fetch/fetch_uniprot_fastas.py
  a0101 (full download, separated files):
    python src/fetch/fetch_uniprot_fastas.py \
      --input-csv data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv \
      --output-fasta data/raw/fasta/hla_a0101_full_length.fasta \
      --positives-fasta data/raw/fasta/hla_a0101_positives_only.fasta

Outputs (defaults = legacy):
  data/raw/fasta/combined_full_length.fasta       — all accessions (with isoforms)
  data/raw/fasta/combined_positives_only.fasta    — proteins with positive epitopes only
"""
import argparse
import pandas as pd
import requests
import re
import sys
from pathlib import Path
import time


def parse_args():
    p = argparse.ArgumentParser(description="Fetch UniProt FASTAs (allele-aware paths, identical logic).")
    p.add_argument("--input-csv", default="data/processed/pos_EL_all_epitopes_hla0201.csv")
    p.add_argument("--output-fasta", default="data/raw/fasta/combined_full_length.fasta")
    p.add_argument("--positives-fasta", default="data/raw/fasta/combined_positives_only.fasta")
    return p.parse_args()


def main(args=None):
    opts = parse_args() if args is None else args
    # Paths
    pos_el_path = Path(opts.input_csv)
    output_fasta = Path(opts.output_fasta)
    positives_fasta = Path(opts.positives_fasta)

    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading processed epitope data: {pos_el_path}...")

    # Read csv and confirm uniprot-ID exists
    try:
        df = pd.read_csv(pos_el_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    if 'uniprot_id' not in df.columns:
        print("[ERROR] 'uniprot_id' column not found!")
        sys.exit(1)

    # UniProt filter that also catches isoforms
    UNIPROT_PATTERN = re.compile(r'^(?![A-Z]{3}[0-9])[A-Z][A-Z0-9]{5,9}(-[0-9]+)?$') #
    unique_ids = [uid for uid in df['uniprot_id'].dropna().unique()
                  if UNIPROT_PATTERN.match(str(uid))]
    # Count isoforms
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
            # Query string for the entire batch, e.g. accession:O14730 OR accession:O15042 (etc.)
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            url = "https://rest.uniprot.org/uniprotkb/stream"
            params = {
                "query": query,
                "format": "fasta",
                "includeIsoform": "true", # Also return isoform entries
            }

            try:
                # Get from Uniprot Rest API
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                fasta_data = response.text
                if fasta_data.strip():
                    out_f.write(fasta_data)
                    # Only write if content is non-empty
                    if not fasta_data.endswith("\n"):
                        out_f.write("\n")
                    total_downloaded += fasta_data.count(">")

                # Print statement for every 10-batch progress
                if i % 10 == 0 or i == len(batches):
                    print(f"  Batch {i}/{len(batches)} ({total_downloaded} sequences so far)")
                time.sleep(1)

            except Exception as e:
                print(f"[ERROR] Failed to fetch batch {i}: {e}")

    print(f"\n=== SUMMARY ===")
    print(f"Queried:             {len(unique_ids)} IDs")
    print(f"Downloaded:          {total_downloaded} FASTA sequences")
    print(f"Output:              {output_fasta}")

    # Positives filter runs for both alleles (separate full + positives files).
    # Guard: same path passed twice → stream file already is the positives file.
    if positives_fasta.resolve() == output_fasta.resolve():
        print(f"\nPositives-only FASTA: {total_downloaded} entries (stream file, no filter)")
        print(f"Output:               {positives_fasta}")
    else:
        # Filter to proteins with positive epitopes
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
