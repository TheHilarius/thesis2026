#!/bin/bash
# run_netsurfp3.sh — Run NetSurfP-3.0 on all batches sequentially with GPU
#
# Usage: bash src/tools/netsurfp/run_netsurfp3.sh

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nsp3_gpu

BATCH_DIR="data/raw/fasta/batches"
NSP3_DIR="/home/hilarius/NetSurfP-3.0_standalone"
OUTPUT_BASE="data/processed/nsp3"

mkdir -p "$OUTPUT_BASE"

echo "Running NetSurfP-3.0 on all batches (GPU, sequential)..."
echo "Batch dir: $BATCH_DIR"
echo "Output:    $OUTPUT_BASE"
echo ""

for batch_file in "$BATCH_DIR"/batch_*.fasta; do
    batch=$(basename "$batch_file" .fasta)
    seq_count=$(grep -c "^>" "$batch_file")

    # Skip if already done
    if [ -d "${OUTPUT_BASE}/${batch}" ] && [ "$(ls "${OUTPUT_BASE}/${batch}"/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[SKIP] $batch — already complete"
        continue
    fi

    echo "[RUN] $batch ($seq_count sequences, GPU) — starting at $(date +%H:%M:%S)"
    python "$NSP3_DIR/nsp3.py" \
        -m "$NSP3_DIR/models/nsp3.pth" \
        -i "$batch_file" \
        -o "${OUTPUT_BASE}/${batch}" \
        -gpu True
    echo "[DONE] $batch — finished at $(date +%H:%M:%S)"
done

echo ""
echo "All done! Results in $OUTPUT_BASE"
