#!/bin/bash
# Filters FASTA by length AND shortens headers to UniProt ID only
# Usage: bash src/nsp3_clean_fasta.sh

INPUT="data/raw/fasta/combined_8_to_14mer.fasta"
OUTPUT="data/raw/fasta/combined_8_to_14mer_nsp3_clean.fasta"

python3 - <<'EOF'
from Bio import SeqIO

input_fasta  = "data/raw/fasta/combined_8_to_14mer.fasta"
output_clean = "data/raw/fasta/combined_8_to_14mer_nsp3_clean.fasta"

MIN_LEN = 130
MAX_LEN = 5000
kept = skipped = 0

with open(output_clean, 'w') as out:
    for record in SeqIO.parse(input_fasta, "fasta"):
        # >tr|A0A024R6N5|A0A024R6N5_HUMAN ... → >A0A024R6N5
        parts = record.id.split("|")
        uid   = parts[1] if len(parts) >= 2 else parts[0]
        seq   = str(record.seq)

        if MIN_LEN <= len(seq) <= MAX_LEN:
            out.write(f">{uid}\n{seq}\n")
            kept += 1
        else:
            skipped += 1

print(f"Kept    : {kept}")
print(f"Skipped : {skipped}  (< {MIN_LEN} or > {MAX_LEN} aa)")
print(f"Output  : {output_clean}")
EOF
