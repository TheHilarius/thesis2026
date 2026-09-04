#!/usr/bin/env python3
import pandas as pd
import requests
import re
import sys
from pathlib import Path
from io import StringIO
import time

def main():
    # Paths
    iedb_csv_path = Path("data/raw/iedb_200K_EL_epitopes.csv")
    output_fasta = Path("data/raw/fasta/combined_9mer.fasta")
    
    # Ensure output directory exists
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw IEDB dataset: {iedb_csv_path}...")
    
    # Read the CSV (IEDB uses comma or semicolon depending on the export, usually comma but with quoted strings)
    # The snippet provided uses semicolons but standard IEDB uses commas. We'll try comma first, fallback to semicolon.
    try:
        # Read with semicolon separator and latin1 encoding. No skipped rows.
        df = pd.read_csv(iedb_csv_path, sep=";", low_memory=False, encoding="latin1")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if 'Epitope - Source Molecule IRI' not in df.columns:
        print("[ERROR] 'Epitope - Source Molecule IRI' column not found!")
        sys.exit(1)

    # Extract UniProt IDs
    # IEDB URLs look like: https://www.uniprot.org/uniprot/Q562F6.2 or Q14094-7
    # We want to extract the ID, keeping isoforms (e.g., -7) but dropping sequence versions (e.g., .2)
    def extract_uniprot_id(iri):
        if pd.isna(iri):
            return None
        match = re.search(r'uniprot/([A-Z0-9]+(?:-\d+)?)', str(iri))
        return match.group(1) if match else None

    print("Extracting unique UniProt IDs...")
    df['uniprot_id'] = df['Epitope - Source Molecule IRI'].apply(extract_uniprot_id)
    
    # Strict UniProt accession filter: 6-10 alphanumeric chars starting with
    # a letter, optionally followed by -isoform. Rejects GenBank (AAX38256),
    # RefSeq (NP_, XP_), and other non-UniProt IDs.
    UNIPROT_PATTERN = re.compile(r'^[A-Z][A-Z0-9]{5,9}(-[0-9]+)?$')
    unique_ids = [uid for uid in df['uniprot_id'].dropna().unique()
                  if UNIPROT_PATTERN.match(uid)]
    print(f"Found {len(unique_ids)} unique UniProt IDs (including isoforms) in the raw IEDB data.")

    # Batch download from UniProt REST API
    # UniProt allows batches of up to 500 for fasta retrieval
    batch_size = 50
    batches = [unique_ids[i:i + batch_size] for i in range(0, len(unique_ids), batch_size)]
    
    print(f"Downloading FASTA sequences from UniProt in {len(batches)} batches...")
    
    total_downloaded = 0
    with open(output_fasta, "w") as out_f:
        for i, batch in enumerate(batches, 1):
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            url = "https://rest.uniprot.org/uniprotkb/stream"
            params = {
                "query": query,
                "format": "fasta"
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                # Write directly to file
                fasta_data = response.text
                if fasta_data.strip():
                    out_f.write(fasta_data)
                    if not fasta_data.endswith("\n"):
                        out_f.write("\n")
                    
                    # Count how many '>' headers we got in this batch
                    total_downloaded += fasta_data.count(">")
                
                print(f"  Batch {i}/{len(batches)} complete...")
                time.sleep(1) # Be polite to the API
                
            except Exception as e:
                print(f"[ERROR] Failed to fetch batch {i}: {e}")

    print("\n=== SUMMARY ===")
    print(f"Extracted {len(unique_ids)} required IDs from raw IEDB data.")
    print(f"Successfully downloaded {total_downloaded} FASTA sequences to:")
    print(f" -> {output_fasta}")
    
    if total_downloaded < len(unique_ids):
        print("\n[NOTE] Some IDs were not found on UniProt. This usually happens if an ID is obsolete or deleted.")

if __name__ == "__main__":
    main()
