#!/bin/bash

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
    echo "Usage: bash run_netmhcpan.sh <fasta_batch_dir> <output_dir> <prefix> [BA]"
    echo "Example: bash run_netmhcpan.sh data/fasta_batches_hla0201 results/netmhcpan my_run"
    echo "Example: bash run_netmhcpan.sh data/fasta_batches_hla0201 results/netmhcpan my_run BA"
    exit 1
fi

BATCH_DIR=\$1
OUTPUT_DIR=\$2
PREFIX=\$3

# Validate batch directory exists
if [ ! -d "$BATCH_DIR" ]; then
    echo "[ERROR] Batch directory not found: $BATCH_DIR"
    exit 1
fi

# Check output directory doesn't already exist
if [ -d "$OUTPUT_DIR" ]; then
    echo "[ERROR] Output directory already exists: $OUTPUT_DIR"
    exit 1
fi

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

for batch in "$BATCH_DIR"/batch_*.fasta; do
    batch_name=$(basename "$batch" .fasta)
    output_csv="$OUTPUT_DIR/${PREFIX}_${batch_name}.csv"
    output_txt="$OUTPUT_DIR/${PREFIX}_${batch_name}.txt"

    echo "Processing $batch_name..."

    netMHCpan \
        -a HLA-A02:01 \
        -f "$batch" \
        -l 8,9,10,11,12,13,14 \
        $BA_FLAG \
        -xls \
        -xlsfile "$output_csv" \
        > "$output_txt"

    echo "[OK] Done: $batch_name"
done

echo "All batches done."

### Without BA
# bash src/run_netmhcpan.sh data/fasta_batches_hla0201 results/netmhcpan my_run

### With BA
# bash src/run_netmhcpan.sh data/fasta_batches_hla0201 results/netmhcpan my_run BA
