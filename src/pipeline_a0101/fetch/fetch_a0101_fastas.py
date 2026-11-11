#!/usr/bin/env python3
"""
fetch_a0101_fastas.py — Download full-length protein sequences from UniProt
for the HLA-A*01:01 external-validation allele.

Edition of legacy src/fetch/fetch_uniprot_fastas.py. Reads unique uniprot_ids
from data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv (produced by
src/pipeline_a0101/r/02_load_iedb_data.R) and downloads the sequences with
includeIsoform=true.

Output:
  data/raw/fasta/hla_a0101_positives_only.fasta

The legacy combined_positives_only.fasta (A0201 proteins) is NOT modified.
Proteins already covered by the legacy FASTA are skipped; only a0101-specific
proteins are fetched (verified 2026-11-11: ~2000 of ~7000 a0101 proteins are
outside the legacy database — without this fetch they would be silently
dropped by the semi_join in 04).

Usage:
    python src/pipeline_a0101/fetch/fetch_a0101_fastas.py
"""
import pandas as pd
import requests
import re
import sys
from pathlib import Path
import time


def read_legacy_accessions():
    """Accessions already present in the legacy A0201 FASTA."""
    fasta = Path("data/raw/fasta/combined_positives_only.fasta")
    accs = set()
    if not fasta.exists():
        return accs
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                m = re.search(r"(?<=\|)[^|]+(?=\|)", line)
                if m:
                    accs.add(m.group())
    return accs


def main():
    pos_el_path = Path("data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv")
    output_fasta = Path("data/raw/fasta/hla_a0101_positives_only.fasta")

    if not pos_el_path.exists():
        print(f"[ERROR] {pos_el_path} not found. "
              f"Run src/pipeline_a0101/r/02_load_iedb_data.R first.")
        sys.exit(1)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading processed epitope data: {pos_el_path}...")
    df = pd.read_csv(pos_el_path)
    if 'uniprot_id' not in df.columns:
        print("[ERROR] 'uniprot_id' column not found!")
        sys.exit(1)

    UNIPROT_PATTERN = re.compile(r'^(?![A-Z]{3}[0-9])[A-Z][A-Z0-9]{5,9}(-[0-9]+)?$')
    unique_ids = [uid for uid in df['uniprot_id'].dropna().unique()
                  if UNIPROT_PATTERN.match(str(uid))]
    n_isoforms = sum(1 for uid in unique_ids if re.search(r'-\d+$', str(uid)))
    print(f"Found {len(unique_ids)} unique UniProt IDs "
          f"({n_isoforms} isoforms, {len(unique_ids) - n_isoforms} canonical)")

    # Skip proteins already in the legacy FASTA (shared database)
    legacy_accs = read_legacy_accessions()
    to_fetch = [uid for uid in unique_ids if uid not in legacy_accs]
    print(f"Already in legacy FASTA (skipped): {len(unique_ids) - len(to_fetch)}")
    print(f"To download:                      {len(to_fetch)}")

    if not to_fetch:
        print("Nothing to fetch — writing empty output for pipeline "
              "consistency.")
        output_fasta.write_text("")
        return

    batch_size = 50
    batches = [to_fetch[i:i + batch_size]
               for i in range(0, len(to_fetch), batch_size)]
    print(f"Downloading from UniProt in {len(batches)} batches "
          f"(includeIsoform=true)...")

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
                    print(f"  Batch {i}/{len(batches)} "
                          f"({total_downloaded} sequences so far)")
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Failed to fetch batch {i}: {e}")

    print(f"\n=== SUMMARY ===")
    print(f"Queried:             {len(to_fetch)} IDs (a0101-specific)")
    print(f"Downloaded:          {total_downloaded} FASTA sequences")
    print(f"Output:              {output_fasta}")

    if total_downloaded < len(to_fetch):
        print(f"\n[NOTE] {len(to_fetch) - total_downloaded} IDs not found on "
              f"UniProt (obsolete or deleted).")

    print("\n[NOTE] AlphaFold structures + NetSurfP-3 runs for these proteins "
          "are NOT fetched here. Proteins without structures are excluded "
          "automatically by the ledger in 04_evaluate_netmhcpan_sensitivity.R "
          "(no_pdb), mirroring legacy behaviour. To maximise coverage, run the "
          "legacy fetch/structure scripts for the missing proteins first.")


if __name__ == "__main__":
    main()
