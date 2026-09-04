#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys

def main():
    # Define our paths
    csv_path = Path("data/processed/df_combined_9mer_pos_and_neg.csv")
    source_fasta_dir = Path("data/raw/fasta/fasta_all_hla0201")
    output_fasta = Path("data/raw/fasta/combined_9mer.fasta")
    
    # 1. Read the exact UniProt IDs we need
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if "uniprot_id" not in df.columns:
        print("[ERROR] 'uniprot_id' column not found!")
        sys.exit(1)
        
    required_ids = df["uniprot_id"].dropna().unique()
    print(f"Found {len(required_ids)} unique UniProt IDs required for the 9-mer dataset.")
    
    # 2. Extract and combine the FASTAs
    print(f"Searching for local FASTA files in {source_fasta_dir}...")
    
    found_count = 0
    missing_ids = []
    
    with open(output_fasta, "w") as out_f:
        for uid in required_ids:
            fasta_file = source_fasta_dir / f"{uid}.fasta"
            
            if fasta_file.exists():
                with open(fasta_file, "r") as in_f:
                    content = in_f.read()
                    out_f.write(content)
                    # Ensure there is a newline between files
                    if not content.endswith("\n"):
                        out_f.write("\n")
                found_count += 1
            else:
                missing_ids.append(uid)
                
    # 3. Print the summary
    print(f"\n=== SUMMARY ===")
    print(f"Successfully combined {found_count} FASTA files into:")
    print(f" -> {output_fasta}")
    
    if missing_ids:
        print(f"\n[WARNING] Could not find local FASTA files for {len(missing_ids)} IDs.")
        print(f"First 10 missing: {missing_ids[:10]}")
        
if __name__ == "__main__":
    main()
