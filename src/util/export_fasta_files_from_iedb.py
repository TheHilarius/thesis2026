#!/usr/bin/env python3

import pandas as pd
import requests
from pathlib import Path
import sys
import time


def download_fasta(uniprot_id, retries=3, wait=5):
    """
    Download FASTA from UniProt REST API with retry logic
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                print(f"[WARNING] Failed to download {uniprot_id} (status {response.status_code})")
                return None
        except requests.exceptions.ConnectionError as e:
            print(f"[WARNING] Connection error for {uniprot_id} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"[ERROR] Giving up on {uniprot_id} after {retries} attempts")
                return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python export_fasta_files.py <input_csv> <output_dir>")
        print("Example: python export_fasta_files.py data/epitopes.csv data/fasta_files")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)

    if not input_file.suffix == ".csv":
        print(f"[ERROR] Input file must be a CSV: {input_file}")
        sys.exit(1)

    if output_dir.exists():
        print(f"[WARNING] Output directory already exists: {output_dir}")
        print("Will skip already downloaded files and continue.")
    else:
        output_dir.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")

    print(f"Reading: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(1)

    if "uniprot_id" not in df.columns:
        print("[ERROR] Column 'uniprot_id' not found in CSV.")
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit(1)

    unique_ids = df["uniprot_id"].dropna().unique()
    print(f"Found {len(unique_ids)} unique UniProt IDs")

    success, skipped = 0, 0
    failed_ids = []

    for uid in unique_ids:
        output_path = output_dir / f"{uid}.fasta"

        if output_path.exists():
            print(f"[SKIP] {uid} already exists")
            skipped += 1
            continue

        print(f"Downloading {uid}...")
        fasta_text = download_fasta(uid)

        if fasta_text:
            with open(output_path, "w") as f:
                f.write(fasta_text)
            print(f"[OK] Saved to {output_path}")
            success += 1
        else:
            failed_ids.append(uid)

        time.sleep(0.5)

    # Save failed IDs
    failed_log = output_dir / "failed_downloads.txt"
    if failed_ids:
        with open(failed_log, "w") as f:
            for uid in failed_ids:
                f.write(f"{uid}\n")
        print(f"\n[WARNING] {len(failed_ids)} failed downloads saved to {failed_log}")
    else:
        print("\nAll downloads successful — no failures.")

    print(f"Done. Success: {success}, Failed: {len(failed_ids)}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
