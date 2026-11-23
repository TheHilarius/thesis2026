#!/bin/bash
# Split the NSP3-clean FASTA into batch files (max seqs / max residues
# per batch — NetSurfP-3 is run per batch on the GPU).
#
# Usage: bash src/tools/netsurfp/prepare_nsp3_batches.sh [hla0201|hla_a0101]
# No arg = hla0201 (legacy call preserved).
# Splitting algorithm identical per allele — only IN/BATCH_DIR/prefix switch.
#
# hla0201  (legacy): in  data/raw/fasta/combined_positives_only_nsp3_clean.fasta
#                    out data/raw/fasta/batches/           (batch_*.fasta)
# hla_a0101       : in  data/raw/fasta/hla_a0101_positives_only_nsp3_clean.fasta
#                    out data/raw/fasta/a0101_nsp3_batches/ (a0101_batch_*.fasta)
#
# NOTE (legacy quirk, kept for byte-identical no-arg behavior): the awk
# counter starts empty, so the FIRST batch file is named "batch_.fasta"
# (resp. "a0101_batch_.fasta") and numbering starts at batch_1. Both
# run_netsurfp3.sh (*batch_*.fasta glob) and the output-dir naming
# handle this fine.
#
# Run nsp3_clean_fasta.sh <allele> first — this script does NOT filter
# or shorten headers.

set -uo pipefail

ALLELE="${1:-hla0201}"
MAX_SEQS=250
MAX_RESIDUES=10000000

case "$ALLELE" in
  hla0201)
    INPUT="data/raw/fasta/combined_positives_only_nsp3_clean.fasta"
    BATCH_DIR="data/raw/fasta/batches"
    PREFIX="batch" ;;
  hla_a0101)
    INPUT="data/raw/fasta/hla_a0101_positives_only_nsp3_clean.fasta"
    BATCH_DIR="data/raw/fasta/a0101_nsp3_batches"
    PREFIX="a0101_batch" ;;
  *)
    echo "Usage: bash src/tools/netsurfp/prepare_nsp3_batches.sh [hla0201|hla_a0101]"
    echo "  Got: '$ALLELE'"
    exit 1 ;;
esac

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found — run nsp3_clean_fasta.sh $ALLELE first"
    exit 1
fi

mkdir -p "$BATCH_DIR"

echo "Allele:   $ALLELE"
echo "Input:    $INPUT"
echo "Batches:  $BATCH_DIR (max $MAX_SEQS seqs / $MAX_RESIDUES residues)"

awk -v max_seqs="$MAX_SEQS" -v max_res="$MAX_RESIDUES" -v dir="$BATCH_DIR" -v prefix="$PREFIX" '
/^>/ {
    if (seq != "") {
        seq_count++
        res_count += length(seq)
        if (seq_count > max_seqs || res_count > max_res) {
            batch++
            seq_count = 1
            res_count = length(seq)
        }
        print header >> dir "/" prefix "_" batch ".fasta"
        print seq >> dir "/" prefix "_" batch ".fasta"
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
        print header >> dir "/" prefix "_" batch ".fasta"
        print seq >> dir "/" prefix "_" batch ".fasta"
    }
}
' "$INPUT"

N_BATCHES=$(ls "$BATCH_DIR" | wc -l)
echo "Done: $N_BATCHES batch files in $BATCH_DIR"
echo "Next: bash src/tools/netsurfp/run_netsurfp3.sh $BATCH_DIR"
