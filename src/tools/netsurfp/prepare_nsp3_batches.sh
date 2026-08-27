#!/bin/bash

INPUT="data/raw/fasta/combined_8_to_14mer_nsp3_clean.fasta"
BATCH_DIR="data/raw/fasta/batches"
MAX_SEQS=5000
MAX_RESIDUES=10000000

mkdir -p "$BATCH_DIR"

awk -v max_seqs="$MAX_SEQS" -v max_res="$MAX_RESIDUES" -v dir="$BATCH_DIR" '
/^>/ {
    if (seq != "") {
        seq_count++
        res_count += length(seq)
        if (seq_count > max_seqs || res_count > max_res) {
            batch++
            seq_count = 1
            res_count = length(seq)
        }
        print header >> dir "/batch_" batch ".fasta"
        print seq >> dir "/batch_" batch ".fasta"
    }
    header = $0
    seq = ""
    next
}
{
    seq = seq $0
}
END {
    if (seq != "") {
        seq_count++
        res_count += length(seq)
        if (seq_count > max_seqs || res_count > max_res) {
            batch++
        }
        print header >> dir "/batch_" batch ".fasta"
        print seq >> dir "/batch_" batch ".fasta"
    }
}
' batch=1 seq_count=0 res_count=0 "$INPUT"

echo "Done! Batches saved to $BATCH_DIR"
ls -lh "$BATCH_DIR"
