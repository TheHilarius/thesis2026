#!/usr/bin/env python3

import pandas as pd
import requests
import re
from pathlib import Path
import sys
import time


INPUT_FILE = "/mnt/c/Users/Hilarius/OneDrive - Danmarks Tekniske Universitet/Skrivebord/special_course_spring2026/special_course_spring2026/data/iedb_522_EL_epitopes.csv"
OUTPUT_DIR = Path("/mnt/c/Users/Hilarius/OneDrive - Danmarks Tekniske Universitet/Skrivebord/special_course_spring2026/special_course_spring2026/data/fasta_files")


def extract_uniprot_id(iri):
    """
    Extract UniProt accession from a URL like:
    http://www.uniprot.org/uniprot/Q6SW84
    """
    if pd.isna(iri):
        return None

    match = re.search(r'uniprot/([A-Z0-9]+)', str(iri))
    if match:
        return match.group(1)
    return None


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
    print(f"Reading: {INPUT_FILE}")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    COLUMN_NAME = "Epitope - Molecule Parent IRI"
    
    if COLUMN_NAME not in df.columns:
        print(f"Column '{COLUMN_NAME}' not found in CSV.")
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit(1)

    print("Extracting UniProt IDs...")
    df["uniprot_id"] = df[COLUMN_NAME].apply(extract_uniprot_id)

    unique_ids = df["uniprot_id"].dropna().unique()

    print(f"Found {len(unique_ids)} unique UniProt IDs")

    OUTPUT_DIR.mkdir(exist_ok=True)

    for uid in unique_ids:
        print(f"Downloading {uid}...")

        fasta_text = download_fasta(uid)

        if fasta_text:
            output_path = OUTPUT_DIR / f"{uid}.fasta"
            with open(output_path, "w") as f:
                f.write(fasta_text)

            print(f"Saved to {output_path}")

        # be polite to UniProt API
        time.sleep(0.5)

    print("Done.")


if __name__ == "__main__":
    main()
