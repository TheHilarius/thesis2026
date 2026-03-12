#!/bin/bash

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
    echo "Usage: bash run_netmhcpan.sh <fasta_batch_dir> <output_dir> <prefix> [BA]"
    echo "Example: bash src/run_netmhcpan.sh data/raw/fasta/combined_batches data/processed/netmhcpan my_run"
    echo "Example: bash src/run_netmhcpan.sh data/raw/fasta/combined_batches data/processed/netmhcpan my_run BA"
    exit 1
fi

BATCH_DIR=$1
OUTPUT_DIR=$2
PREFIX=$3

# Validate batch directory exists
if [ ! -d "$BATCH_DIR" ]; then
    echo "[ERROR] Batch directory not found: $BATCH_DIR"
    exit 1
fi

# Create output directory if it doesn't exist (no longer errors if it does)
mkdir -p "$OUTPUT_DIR"

BA_FLAG=""
if [ "\$4" == "BA" ]; then
    BA_FLAG="-BA"
    echo "Mode: EL + BA"
else
    echo "Mode: EL only"
fi

echo "Running netMHCpan on batches in: $BATCH_DIR"
echo "Output dir: $OUTPUT_DIR"

SKIPPED=0
PROCESSED=0
TOTAL=0

for batch in "$BATCH_DIR"/batch_*.fasta; do
    TOTAL=$((TOTAL + 1))
    batch_name=$(basename "$batch" .fasta)
    output_csv="$OUTPUT_DIR/${PREFIX}_${batch_name}.csv"
    output_txt="$OUTPUT_DIR/${PREFIX}_${batch_name}.txt"

    # Skip if BOTH .csv and .txt already exist
    if [ -f "$output_csv" ] && [ -f "$output_txt" ]; then
        echo "[SKIP] $batch_name — already complete ($output_csv and $output_txt exist)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Clean up partial outputs before re-running
    [ -f "$output_csv" ] && rm "$output_csv"
    [ -f "$output_txt" ] && rm "$output_txt"

    echo "[RUN] Processing $batch_name..."

    netMHCpan \
        -a HLA-A02:01 \
        -f "$batch" \
        -l 8,9,10,11,12,13,14 \
        $BA_FLAG \
        -xls \
        -xlsfile "$output_csv" \
        > "$output_txt"

    PROCESSED=$((PROCESSED + 1))
    echo "[OK] Done: $batch_name"
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total batches:   $TOTAL"
echo "  Skipped (done):  $SKIPPED"
echo "  Processed (new): $PROCESSED"
echo "========================================"
echo "All batches done."
