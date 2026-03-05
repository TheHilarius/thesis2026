#!/bin/bash

# Test script - runs netMHCpan on a single small FASTA file
# Usage: bash src/run_netmhcpan_test.sh [BA]

INPUT_FASTA="data/raw/fasta/test_10.fasta"
OUTPUT_DIR="data/processed/netmhcpan_test"
PREFIX="test"

if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

BA_FLAG=""
if [ "\$1" == "BA" ]; then
    BA_FLAG="-BA"
    echo "Mode: EL + BA"
else
    echo "Mode: EL only"
fi

echo "Running netMHCpan test on: $INPUT_FASTA"

netMHCpan \
    -a HLA-A02:01 \
    -f "$INPUT_FASTA" \
    -l 8,9,10,11,12,13,14 \
    $BA_FLAG \
    -xls \
    -xlsfile "$OUTPUT_DIR/${PREFIX}_output.csv" \
    > "$OUTPUT_DIR/${PREFIX}_output.txt"

echo "[OK] Done. Output in $OUTPUT_DIR"

### Without BA
# bash src/run_netmhcpan_test.sh

### With BA
# bash src/run_netmhcpan_test.sh BA
