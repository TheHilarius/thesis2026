#!/usr/bin/env python3

import pandas as pd
import requests
from pathlib import Path
import sys
import time


def download_fasta(uniprot_id):
    """
    Download FASTA from UniProt REST API
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        return response.text
    else:
        print(f"[WARNING] Failed to download {uniprot_id} (status {response.status_code})")
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python export_fasta_files.py <input_csv> <output_dir>")
        print("Example: python export_fasta_files.py data/epitopes.csv data/fasta_files")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # Validate input file
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".csv":
        print(f"[ERROR] Input file must be a CSV: {input_file}")
        sys.exit(1)

    # Check output dir doesn't already exist
    if output_dir.exists():
        print(f"[ERROR] Output directory already exists: {output_dir}")
        print("Please specify a new directory name to avoid overwriting.")
        sys.exit(1)

    print(f"Reading: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(1)

    # Check uniprot_id column exists
    if "uniprot_id" not in df.columns:
        print("[ERROR] Column 'uniprot_id' not found in CSV.")
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit(1)

    unique_ids = df["uniprot_id"].dropna().unique()
    print(f"Found {len(unique_ids)} unique UniProt IDs")

    output_dir.mkdir(parents=True)
    print(f"Created output directory: {output_dir}")

    # Track results
    success, failed = 0, 0

    for uid in unique_ids:
        output_path = output_dir / f"{uid}.fasta"

        print(f"Downloading {uid}...")
        fasta_text = download_fasta(uid)

        if fasta_text:
            with open(output_path, "w") as f:
                f.write(fasta_text)
            print(f"[OK] Saved to {output_path}")
            success += 1
        else:
            failed += 1

        time.sleep(0.5)

    print(f"\nDone. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
