#!/bin/bash
# Filters FASTA by length AND shortens headers to UniProt ID only
# (prepares the NSP3 input: NetSurfP output dirs are named after headers,
# so full UniProt headers exceed the 255-char filename limit).
#
# Usage: bash src/tools/netsurfp/nsp3_clean_fasta.sh [hla0201|hla_a0101]
# No arg = hla0201 (legacy call preserved).
# Logic identical per allele — only IN/OUT paths switch.
#
# hla0201  (legacy): data/raw/fasta/combined_positives_only.fasta
#                 -> data/raw/fasta/combined_positives_only_nsp3_clean.fasta
# hla_a0101       : data/raw/fasta/hla_a0101_positives_only.fasta
#                 -> data/raw/fasta/hla_a0101_positives_only_nsp3_clean.fasta

set -uo pipefail

ALLELE="${1:-hla0201}"

case "$ALLELE" in
  hla0201)
    INPUT="data/raw/fasta/combined_positives_only.fasta"
    OUTPUT="data/raw/fasta/combined_positives_only_nsp3_clean.fasta" ;;
  hla_a0101)
    INPUT="data/raw/fasta/hla_a0101_positives_only.fasta"
    OUTPUT="data/raw/fasta/hla_a0101_positives_only_nsp3_clean.fasta" ;;
  *)
    echo "Usage: bash src/tools/netsurfp/nsp3_clean_fasta.sh [hla0201|hla_a0101]"
    echo "  Got: '$ALLELE'"
    exit 1 ;;
esac

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found — run the UniProt FASTA fetch first"
    exit 1
fi

echo "Allele:  $ALLELE"
echo "Input:   $INPUT"
echo "Output:  $OUTPUT"

python3 - "$INPUT" "$OUTPUT" <<'EOF'
from Bio import SeqIO
import sys

input_fasta  = sys.argv[1]
output_clean = sys.argv[2]

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
