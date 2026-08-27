#!/usr/bin/env bash
# audit_hybrid_structures.sh
# Usage: bash src/fetch/audit_hybrid_structures.sh

set -euo pipefail

STRUCT_DIR="data/processed/structures_hybrid/selected"
LOG_FILE="data/processed/structures_hybrid/logs/fetch_log.tsv"
CSV_FILE="data/processed/df_all.csv"
AUDIT_DIR="data/processed/structures_hybrid/logs"

echo "════════════════════════════════════════"
echo "  HYBRID STRUCTURE AUDIT"
echo "════════════════════════════════════════"

# 1. Get unique UniProt IDs from the current dataset
echo ""
echo "Extracting UniProt IDs from df_all.csv..."
cut -d',' -f6 "${CSV_FILE}" | tail -n +2 | sort -u > "${AUDIT_DIR}/dataset_uniprots.txt"
N_DATASET=$(wc -l < "${AUDIT_DIR}/dataset_uniprots.txt")
echo "  Unique proteins in dataset: ${N_DATASET}"

# 2. Get UniProt IDs that have structures on disk
echo ""
echo "Listing structures on disk..."
ls "${STRUCT_DIR}"/*.pdb 2>/dev/null | \
    xargs -I{} basename {} .pdb | \
    sort -u > "${AUDIT_DIR}/ondisk_uniprots.txt"
N_ONDISK=$(wc -l < "${AUDIT_DIR}/ondisk_uniprots.txt")
echo "  Structures on disk: ${N_ONDISK}"

# 3. Find orphans (on disk but NOT in dataset)
comm -23 "${AUDIT_DIR}/ondisk_uniprots.txt" "${AUDIT_DIR}/dataset_uniprots.txt" \
    > "${AUDIT_DIR}/orphan_uniprots.txt"
N_ORPHAN=$(wc -l < "${AUDIT_DIR}/orphan_uniprots.txt")
echo "  Orphaned structures (not in dataset): ${N_ORPHAN}"

# 4. Find missing (in dataset but NOT on disk)
comm -23 "${AUDIT_DIR}/dataset_uniprots.txt" "${AUDIT_DIR}/ondisk_uniprots.txt" \
    > "${AUDIT_DIR}/missing_uniprots.txt"
N_MISSING=$(wc -l < "${AUDIT_DIR}/missing_uniprots.txt")
echo "  Dataset proteins without structure: ${N_MISSING}"

# 5. Find matched (in both)
comm -12 "${AUDIT_DIR}/dataset_uniprots.txt" "${AUDIT_DIR}/ondisk_uniprots.txt" \
    > "${AUDIT_DIR}/matched_uniprots.txt"
N_MATCHED=$(wc -l < "${AUDIT_DIR}/matched_uniprots.txt")
echo "  Matched (dataset ∩ disk): ${N_MATCHED}"

# 6. Source breakdown for matched proteins only
echo ""
echo "Source breakdown (dataset proteins only):"
if [[ -f "${LOG_FILE}" ]]; then
    # Join log with matched list
    grep -Ff "${AUDIT_DIR}/matched_uniprots.txt" "${LOG_FILE}" | \
        awk -F'\t' '{print $4}' | sort | uniq -c | sort -rn
fi

# 7. Coverage summary
echo ""
echo "════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════"
echo "  Dataset proteins:     ${N_DATASET}"
echo "  With structure:       ${N_MATCHED} ($(python3 -c "print(f'{${N_MATCHED}/${N_DATASET}*100:.1f}')")%)"
echo "  Without structure:    ${N_MISSING} ($(python3 -c "print(f'{${N_MISSING}/${N_DATASET}*100:.1f}')")%)"
echo "  Orphaned (to remove): ${N_ORPHAN}"
echo ""

# 8. Optional: remove orphans
if [[ ${N_ORPHAN} -gt 0 ]]; then
    echo "To remove orphaned structures, run:"
    echo "  while read uid; do rm -f \"${STRUCT_DIR}/\${uid}.pdb\"; done < ${AUDIT_DIR}/orphan_uniprots.txt"
    echo ""
    echo "Or to preview what would be removed:"
    echo "  head ${AUDIT_DIR}/orphan_uniprots.txt"
fi

echo "════════════════════════════════════════"
