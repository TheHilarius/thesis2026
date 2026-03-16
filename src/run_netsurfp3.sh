#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nsp3

BATCH_DIR="data/raw/fasta/batches"
NSP3_DIR="/home/hilarius/NetSurfP-3.0_standalone"
OUTPUT_BASE="data/processed/nsp3"

mkdir -p "$OUTPUT_BASE"

for batch_file in "$BATCH_DIR"/batch_*.fasta; do
    batch=$(basename "$batch_file" .fasta)
    seq_count=$(grep -c "^>" "$batch_file")
    echo "Running $batch ($seq_count sequences)..."
    python "$NSP3_DIR/nsp3.py" -m "$NSP3_DIR/models/nsp3.pth" -i "$batch_file" -o "${OUTPUT_BASE}/${batch}"
done

echo "All done! Results in $OUTPUT_BASE"
