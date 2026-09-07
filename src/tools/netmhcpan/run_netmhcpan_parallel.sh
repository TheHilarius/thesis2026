#!/bin/bash
# run_netmhcpan_parallel.sh — Parallel netMHCpan batch runner
#
# Usage: bash run_netmhcpan_parallel.sh <fasta_batch_dir> <output_dir> <prefix> [max_jobs] [BA]
# Example: bash src/tools/netmhcpan/run_netmhcpan_parallel.sh data/raw/fasta/combined_batches data/processed/netmhcpan hla_a0201 10

set -uo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: bash run_netmhcpan_parallel.sh <fasta_batch_dir> <output_dir> <prefix> [max_jobs] [BA]"
    echo "  max_jobs: parallel netMHCpan processes (default: 10)"
    echo "  BA: add 'BA' as 5th arg to include binding affinity prediction"
    exit 1
fi

BATCH_DIR=$1
OUTPUT_DIR=$2
PREFIX=$3
MAX_JOBS=${4:-10}
BA_FLAG=""

if [ "$#" -eq 5 ] && [ "$5" == "BA" ]; then
    BA_FLAG="-BA"
fi

# ── Validate ─────────────────────────────────────────────────────────────────
if [ ! -d "$BATCH_DIR" ]; then
    echo "[ERROR] Batch directory not found: $BATCH_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Discover batches ──────────────────────────────────────────────────────────
BATCHES=()
for batch in "$BATCH_DIR"/batch_*.fasta; do
    [ -f "$batch" ] && BATCHES+=("$batch")
done
TOTAL=${#BATCHES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "[ERROR] No batch files found in $BATCH_DIR"
    exit 1
fi

# ── Classify batches ──────────────────────────────────────────────────────────
SKIP=0
BATCHES_TO_RUN=()

for batch in "${BATCHES[@]}"; do
    batch_name=$(basename "$batch" .fasta)
    output_csv="$OUTPUT_DIR/${PREFIX}_${batch_name}.csv"
    output_txt="$OUTPUT_DIR/${PREFIX}_${batch_name}.txt"

    if [ -s "$output_csv" ] && [ -s "$output_txt" ]; then
        SKIP=$((SKIP + 1))
    else
        # Clean up partial outputs
        [ -f "$output_csv" ] && rm "$output_csv"
        [ -f "$output_txt" ] && rm "$output_txt"
        BATCHES_TO_RUN+=("$batch")
    fi
done
RUN=${#BATCHES_TO_RUN[@]}

# ── Print plan ───────────────────────────────────────────────────────────────
echo "========================================"
echo "  netMHCpan Parallel Runner"
echo "========================================"
echo "  Batch dir:    $BATCH_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Prefix:       $PREFIX"
echo "  Mode:         $([ -n "$BA_FLAG" ] && echo 'EL + BA' || echo 'EL only')"
echo "  Max parallel: $MAX_JOBS"
echo "----------------------------------------"
echo "  Total batches:     $TOTAL"
echo "  Already done:      $SKIP"
echo "  To process:        $RUN"
echo "========================================"
echo ""

if [ "$RUN" -eq 0 ]; then
    echo "All batches already complete."
    exit 0
fi

# ── Run in parallel ───────────────────────────────────────────────────────────
START_TIME=$(date +%s)
COMPLETED=0
FAILED=0
PIDS=()
BATCH_NAMES=()

run_batch() {
    local batch=$1
    local batch_name=$(basename "$batch" .fasta)
    local output_csv="$OUTPUT_DIR/${PREFIX}_${batch_name}.csv"
    local output_txt="$OUTPUT_DIR/${PREFIX}_${batch_name}.txt"
    local batch_start=$(date +%s)

    netMHCpan \
        -a HLA-A02:01 \
        -f "$batch" \
        -l 8,9,10,11,12,13,14 \
        $BA_FLAG \
        -xls \
        -xlsfile "$output_csv" \
        > "$output_txt" 2>&1

    local exit_code=$?
    local batch_end=$(date +%s)
    local elapsed=$((batch_end - batch_start))

    if [ $exit_code -eq 0 ] && [ -s "$output_csv" ]; then
        echo "[OK] $batch_name completed in ${elapsed}s"
    else
        echo "[FAIL] $batch_name failed (exit=$exit_code, ${elapsed}s)"
    fi
    return $exit_code
}

for i in "${!BATCHES_TO_RUN[@]}"; do
    batch="${BATCHES_TO_RUN[$i]}"
    batch_name=$(basename "$batch" .fasta)

    # Wait if at max parallelism
    while [ ${#PIDS[@]} -ge "$MAX_JOBS" ]; do
        NEW_PIDS=()
        NEW_NAMES=()
        for j in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$j]}" 2>/dev/null; then
                wait "${PIDS[$j]}" 2>/dev/null
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    COMPLETED=$((COMPLETED + 1))
                else
                    FAILED=$((FAILED + 1))
                fi
                DONE=$((COMPLETED + FAILED))
                ELAPSED=$(( $(date +%s) - START_TIME ))
                if [ "$DONE" -gt 0 ]; then
                    ETA=$(( ELAPSED * (RUN - DONE) / DONE ))
                    ETA_MIN=$((ETA / 60))
                    echo "[PROGRESS] $DONE/$RUN done (${COMPLETED} ok, ${FAILED} fail) | Elapsed: $((ELAPSED/60))m | ETA: ~${ETA_MIN}m"
                fi
            else
                NEW_PIDS+=("${PIDS[$j]}")
                NEW_NAMES+=("${BATCH_NAMES[$j]}")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        BATCH_NAMES=("${NEW_NAMES[@]}")
        [ ${#PIDS[@]} -ge "$MAX_JOBS" ] && sleep 5
    done

    echo "[LAUNCH] $batch_name (slot $(( ${#PIDS[@]} + 1 ))/$MAX_JOBS)"
    run_batch "$batch" &
    PIDS+=($!)
    BATCH_NAMES+=("$batch_name")
done

# Wait for remaining jobs
echo ""
echo "All jobs launched. Waiting for remaining ${#PIDS[@]} jobs..."
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" 2>/dev/null
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_SEC=$((TOTAL_TIME % 60))

echo ""
echo "========================================"
echo "  COMPLETE"
echo "========================================"
echo "  Total time:   ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Skipped:      $SKIP"
echo "  Completed:    $COMPLETED"
echo "  Failed:       $FAILED"
echo "  Total:        $((SKIP + COMPLETED + FAILED)) / $TOTAL"
echo "========================================"

# List failed batches if any
if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "Failed batches:"
    for batch in "${BATCHES_TO_RUN[@]}"; do
        batch_name=$(basename "$batch" .fasta)
        output_csv="$OUTPUT_DIR/${PREFIX}_${batch_name}.csv"
        if [ ! -s "$output_csv" ]; then
            echo "  $batch_name"
        fi
    done
fi
