#!/usr/bin/env python3
"""
clean_fasta_headers.py — Truncate FASTA headers to >type|ACCESSION|NAME

NetSurfP creates output directories named after FASTA headers.
Full UniProt headers can reach 256 chars, exceeding the 255-char
Linux ext4 filename limit. This script keeps only the first three
pipe-delimited fields (e.g., >sp|O14730|RIOK3_HUMAN), which preserves
isoform suffixes (e.g., Q9NYP3-3) while staying under 35 chars.

Usage:
    python src/tools/netsurfp/clean_fasta_headers.py
"""
from pathlib import Path

INPUT  = Path("data/raw/fasta/combined_positives_only.fasta")
OUTPUT = Path("data/raw/fasta/combined_positives_only_clean.fasta")

max_len_before = 0
max_len_after  = 0
n_entries      = 0

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            n_entries += 1
            raw = line.rstrip("\n")
            max_len_before = max(max_len_before, len(raw))
            # >db|ACC|NAME rest... → first token (split on whitespace)
            cleaned = raw.split()[0]
            max_len_after = max(max_len_after, len(cleaned))
            fout.write(cleaned + "\n")
        else:
            fout.write(line)

print(f"Entries processed: {n_entries}")
print(f"Max header length: {max_len_before} → {max_len_after}")
print(f"Output: {OUTPUT}")
