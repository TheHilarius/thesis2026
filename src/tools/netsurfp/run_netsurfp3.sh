#!/bin/bash
# run_netsurfp3.sh — Run NetSurfP-3.0 on all batches sequentially with GPU
#
# Usage: bash src/tools/netsurfp/run_netsurfp3.sh [batch_dir] [output_base]
#   batch_dir:   dir with the NSP3 batch FASTAs
#                (default: data/raw/fasta/batches — legacy HLA-A*02:01)
#   output_base: per-protein CSV output root
#                (default: data/processed/nsp3 — SHARED across alleles;
#                per-batch output dirs are named after the batch files, so
#                per-allele batch prefixes keep outputs separate)
#
# Example (a0101 batches built by prepare_nsp3_batches.sh hla_a0101):
#   bash src/tools/netsurfp/run_netsurfp3.sh data/raw/fasta/a0101_nsp3_batches
#
# No args = legacy call, byte-identical.
#
# hub1 locations (verified 2026-11-24 — Sep 2026 nsp3_h100_* jobs used
# these): NetSurfP checkout + model live in the PROJECT dir (a ctools
# copy also exists); the nsp3_gpu conda env lives in the project
# partner's ENTER install and is readable/executable by s204692.
# Override via env vars on other machines.

# Conda env with torch (GPU build)
NSP3_CONDA_SH="${NSP3_CONDA_SH:-/home/people/s204581/ENTER/etc/profile.d/conda.sh}"
NSP3_ENV="${NSP3_ENV:-nsp3_gpu}"
source "$NSP3_CONDA_SH"
conda activate "$NSP3_ENV"

BATCH_DIR="${1:-data/raw/fasta/batches}"
NSP3_DIR="${NSP3_DIR:-/home/projects1/thesis_s204692_s204581/NetSurfP-3.0_standalone}"
OUTPUT_BASE="${2:-data/processed/nsp3}"

mkdir -p "$OUTPUT_BASE"

echo "Running NetSurfP-3.0 on all batches (GPU, sequential)..."
echo "Batch dir: $BATCH_DIR"
echo "Output:    $OUTPUT_BASE"
echo ""

# *batch_*.fasta matches legacy batch_.fasta / batch_1..N and per-allele
# prefixed batches (e.g. a0101_batch_.fasta) in their own dirs.
for batch_file in "$BATCH_DIR"/*batch_*.fasta; do
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
