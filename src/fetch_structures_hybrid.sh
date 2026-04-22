#!/usr/bin/env bash
# fetch_structures_hybrid.sh
# Priority: experimental PDB (full coverage, best resolution) → AlphaFold fallback
#
# Usage: bash src/fetch_structures_hybrid.sh <out_dir>
# Example: bash src/fetch_structures_hybrid.sh data/processed/structures_hybrid

set -uo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
if [[ $# -ne 1 ]]; then
    echo "Usage: bash src/fetch_structures_hybrid.sh <out_dir>"
    exit 1
fi

OUT_DIR="${1}"

# Reuse the same fetch lists from the original pipeline
ORIGINAL_STRUCT_DIR="data/processed/structures"
CANONICAL_LIST="${ORIGINAL_STRUCT_DIR}/logs/fetch_list_canonical.tsv"
ISOFORM_LIST="${ORIGINAL_STRUCT_DIR}/logs/fetch_list_isoforms.tsv"

if [[ ! -f "${CANONICAL_LIST}" ]]; then
    echo "ERROR: ${CANONICAL_LIST} not found — run prepare_fetch_list.sh first"
    exit 1
fi
if [[ ! -f "${ISOFORM_LIST}" ]]; then
    echo "ERROR: ${ISOFORM_LIST} not found — run prepare_fetch_list.sh first"
    exit 1
fi

# ── Config ───────────────────────────────────────────────────────────────────
AF2_BASE="https://alphafold.ebi.ac.uk/files"
PDB_SEARCH="https://search.rcsb.org/rcsbsearch/v2/query"
PDB_BASE="https://files.rcsb.org/download"

# Maximum number of PDB candidates to check per protein
MAX_PDB_CANDIDATES=10

mkdir -p "${OUT_DIR}/selected"
mkdir -p "${OUT_DIR}/logs"

LOG="${OUT_DIR}/logs/fetch_log.tsv"
MISSING="${OUT_DIR}/logs/missing_structures.tsv"
SUMMARY="${OUT_DIR}/logs/source_summary.tsv"

# ── Resume logic ─────────────────────────────────────────────────────────────
declare -A ALREADY_DONE

if [[ -f "${LOG}" ]]; then
    echo "Existing log found — collecting already-processed entries..."
    while IFS=$'\t' read -r uid _rest; do
        ALREADY_DONE["${uid}"]=1
    done < <(tail -n +2 "${LOG}")
    echo "Already processed: ${#ALREADY_DONE[@]} entries"
else
    echo -e "uniprot_id\tstart\tend\tsource\tpdb_id\tfile\tcoverage\tresolution" > "${LOG}"
    echo -e "uniprot_id\treason" > "${MISSING}"
    echo "Fresh run — logs initialized"
fi

# ── Counters ─────────────────────────────────────────────────────────────────
COUNT_PDB=0
COUNT_AF2=0
COUNT_NONE=0

# ── Helper: strip .0 from pandas float coords ────────────────────────────────
strip_float() {
    printf "%.0f" "${1}"
}

# ── Helper: check PDB residue coverage ────────────────────────────────────────
check_coverage() {
    local pdb_file="${1}"
    local start="${2}"
    local end="${3}"

    local min_res max_res

    min_res=$(grep "^ATOM" "${pdb_file}" | \
              awk '{print substr($0,23,4)+0}' | \
              sort -n | head -1)
    max_res=$(grep "^ATOM" "${pdb_file}" | \
              awk '{print substr($0,23,4)+0}' | \
              sort -n | tail -1)

    if [[ -z "${min_res}" || -z "${max_res}" ]]; then
        echo "no_atoms"
        return
    fi

    if (( min_res <= start && max_res >= end )); then
        echo "ok"
    else
        # Calculate overlap fraction
        local req_len=$(( end - start + 1 ))
        local cov_start=$(( start > min_res ? start : min_res ))
        local cov_end=$(( end < max_res ? end : max_res ))
        local overlap=0

        if (( cov_end >= cov_start )); then
            overlap=$(( cov_end - cov_start + 1 ))
        fi

        local pct=$(python3 -c "print(f'{${overlap}/${req_len}*100:.1f}')")

        echo "partial:${min_res}-${max_res}:${pct}pct"
    fi
}

# ── Helper: extract resolution from PDB header ───────────────────────────────
get_resolution() {
    local pdb_file="${1}"
    local res

    res=$(grep "^REMARK   2 RESOLUTION" "${pdb_file}" | \
          head -1 | \
          awk '{for(i=1;i<=NF;i++) if($i+0==$i && $i>0) {print $i; exit}}')

    if [[ -z "${res}" ]]; then
        echo "NA"
    else
        echo "${res}"
    fi
}

# ── Helper: fetch AF2 structure, try v6 → v5 → v4 ───────────────────────────
fetch_af2() {
    local uniprot_id="${1}"
    local out_file="${2}"
    local http_status

    for version in 6 5 4; do
        local url="${AF2_BASE}/AF-${uniprot_id}-F1-model_v${version}.pdb"

        http_status=$(curl -s -o "${out_file}" \
                           -w "%{http_code}" \
                           --retry 2 \
                           --retry-delay 1 \
                           --max-time 30 \
                           "${url}" || echo "000")

        if [[ "${http_status}" == "200" ]]; then
            echo "${version}"
            return 0
        fi

        rm -f "${out_file}"
    done

    echo "none"
    return 1
}

# ── Helper: try PDB first, then AF2 fallback ─────────────────────────────────
fetch_one() {
    local UNIPROT="${1}"
    local START="${2}"
    local END="${3}"
    local BASE_ID="${4}"

    local SELECTED_FILE=""
    local SELECTED_SOURCE=""
    local SELECTED_PDB_ID="NA"
    local SELECTED_COVERAGE=""
    local SELECTED_RESOLUTION="NA"

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Try experimental PDB structures (sorted by resolution)
    # ══════════════════════════════════════════════════════════════════════

    # Track best partial PDB in case AF2 also fails
    local BEST_PARTIAL_FILE=""
    local BEST_PARTIAL_PDB_ID=""
    local BEST_PARTIAL_COVERAGE=""
    local BEST_PARTIAL_RESOLUTION=""
    local BEST_PARTIAL_PCT=0

    QUERY=$(printf '{
      "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
          "operator": "exact_match",
          "value": "%s"
        }
      },
      "request_options": {
        "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        "results_content_type": ["experimental"],
        "paginate": {"start": 0, "rows": %d}
      },
      "return_type": "entry"
    }' "${BASE_ID}" "${MAX_PDB_CANDIDATES}")

    PDB_IDS=$(curl -s -X POST \
                   -H "Content-Type: application/json" \
                   -d "${QUERY}" \
                   --max-time 15 \
                   "${PDB_SEARCH}" | \
              python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    ids = [r['identifier'] for r in data.get('result_set', [])]
    print('\n'.join(ids))
except:
    pass
" 2>/dev/null || true)

    if [[ -n "${PDB_IDS}" ]]; then
        while IFS= read -r PDB_ID; do
            [[ -z "${PDB_ID}" ]] && continue

            local TMP_PDB="${OUT_DIR}/selected/.tmp_${BASE_ID}_${PDB_ID}.pdb"

            HTTP_PDB=$(curl -s -o "${TMP_PDB}" \
                            -w "%{http_code}" \
                            --retry 2 \
                            --max-time 30 \
                            "${PDB_BASE}/${PDB_ID}.pdb" || echo "000")

            if [[ "${HTTP_PDB}" != "200" ]]; then
                rm -f "${TMP_PDB}"
                continue
            fi

            COVERAGE=$(check_coverage "${TMP_PDB}" "${START}" "${END}")

            if [[ "${COVERAGE}" == "ok" ]]; then
                RESOLUTION=$(get_resolution "${TMP_PDB}")
                FINAL_NAME="${OUT_DIR}/selected/${UNIPROT}.pdb"
                mv "${TMP_PDB}" "${FINAL_NAME}"

                SELECTED_FILE="${FINAL_NAME}"
                SELECTED_SOURCE="pdb"
                SELECTED_PDB_ID="${PDB_ID}"
                SELECTED_COVERAGE="ok"
                SELECTED_RESOLUTION="${RESOLUTION}"

                echo "  [PDB] ✓ ${PDB_ID} (resolution: ${RESOLUTION}Å, coverage: ok)"
                break

            elif [[ "${COVERAGE}" == partial:*pct ]]; then
                local pct_val
                pct_val=$(echo "${COVERAGE}" | grep -oP '[0-9.]+(?=pct)')

                if (( $(echo "${pct_val} >= 80.0" | bc -l) )); then
                    RESOLUTION=$(get_resolution "${TMP_PDB}")
                    FINAL_NAME="${OUT_DIR}/selected/${UNIPROT}.pdb"
                    mv "${TMP_PDB}" "${FINAL_NAME}"

                    SELECTED_FILE="${FINAL_NAME}"
                    SELECTED_SOURCE="pdb"
                    SELECTED_PDB_ID="${PDB_ID}"
                    SELECTED_COVERAGE="${COVERAGE}"
                    SELECTED_RESOLUTION="${RESOLUTION}"

                    echo "  [PDB] ✓ ${PDB_ID} (resolution: ${RESOLUTION}Å, coverage: ${COVERAGE} — accepted ≥80%)"
                    break
                else
                    # Track best partial for last-resort fallback
                    if (( $(echo "${pct_val} > ${BEST_PARTIAL_PCT}" | bc -l) )); then
                        # Save this as best partial so far
                        rm -f "${BEST_PARTIAL_FILE}"
                        BEST_PARTIAL_FILE="${TMP_PDB}"
                        BEST_PARTIAL_PDB_ID="${PDB_ID}"
                        BEST_PARTIAL_COVERAGE="${COVERAGE}"
                        BEST_PARTIAL_RESOLUTION=$(get_resolution "${TMP_PDB}")
                        BEST_PARTIAL_PCT=$(echo "${pct_val}" | bc -l)
                    else
                        rm -f "${TMP_PDB}"
                    fi

                    echo "  [PDB] ✗ ${PDB_ID} coverage: ${COVERAGE} — below 80% threshold"
                fi
            else
                echo "  [PDB] ✗ ${PDB_ID} coverage: ${COVERAGE}"
                rm -f "${TMP_PDB}"
            fi

        done <<< "${PDB_IDS}"
    else
        echo "  [PDB] No experimental structures found for ${BASE_ID}"
    fi

    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: AlphaFold fallback (if no suitable PDB found)
    # ══════════════════════════════════════════════════════════════════════

    if [[ -z "${SELECTED_FILE}" ]]; then
        echo "  [AF2] Falling back to AlphaFold..."

        AF2_FILE="${OUT_DIR}/selected/${UNIPROT}.pdb"
        WORKED_VERSION=$(fetch_af2 "${UNIPROT}" "${AF2_FILE}" || true)

        if [[ "${WORKED_VERSION}" != "none" ]]; then
            COVERAGE=$(check_coverage "${AF2_FILE}" "${START}" "${END}")
            echo "  [AF2] ✓ v${WORKED_VERSION} (coverage: ${COVERAGE})"

            SELECTED_FILE="${AF2_FILE}"
            SELECTED_SOURCE="alphafold_v${WORKED_VERSION}"
            SELECTED_COVERAGE="${COVERAGE}"
            SELECTED_RESOLUTION="NA"

        elif [[ "${UNIPROT}" != "${BASE_ID}" ]]; then
            echo "  [AF2] Trying canonical base: ${BASE_ID}"
            BASE_AF2_FILE="${OUT_DIR}/selected/${UNIPROT}.pdb"
            BASE_VERSION=$(fetch_af2 "${BASE_ID}" "${BASE_AF2_FILE}" || true)

            if [[ "${BASE_VERSION}" != "none" ]]; then
                COVERAGE=$(check_coverage "${BASE_AF2_FILE}" "${START}" "${END}")
                echo "  [AF2] ✓ Base ${BASE_ID} v${BASE_VERSION} (coverage: ${COVERAGE})"

                SELECTED_FILE="${BASE_AF2_FILE}"
                SELECTED_SOURCE="alphafold_base:${BASE_ID}_v${BASE_VERSION}"
                SELECTED_COVERAGE="${COVERAGE}"
                SELECTED_RESOLUTION="NA"
            fi
        fi
    fi

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Last resort — best partial PDB (if AF2 also failed)
    # ══════════════════════════════════════════════════════════════════════

    if [[ -z "${SELECTED_FILE}" && -n "${BEST_PARTIAL_FILE}" ]]; then
        FINAL_NAME="${OUT_DIR}/selected/${UNIPROT}.pdb"
        mv "${BEST_PARTIAL_FILE}" "${FINAL_NAME}"

        SELECTED_FILE="${FINAL_NAME}"
        SELECTED_SOURCE="pdb_partial"
        SELECTED_PDB_ID="${BEST_PARTIAL_PDB_ID}"
        SELECTED_COVERAGE="${BEST_PARTIAL_COVERAGE}"
        SELECTED_RESOLUTION="${BEST_PARTIAL_RESOLUTION}"

        echo "  [PDB] ✓ Last resort: ${BEST_PARTIAL_PDB_ID} (resolution: ${BEST_PARTIAL_RESOLUTION}Å, coverage: ${BEST_PARTIAL_COVERAGE})"
    fi

    # Clean up any remaining temp partial file
    if [[ -n "${BEST_PARTIAL_FILE}" && -f "${BEST_PARTIAL_FILE}" ]]; then
        rm -f "${BEST_PARTIAL_FILE}"
    fi

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Log result
    # ══════════════════════════════════════════════════════════════════════

    if [[ -n "${SELECTED_FILE}" ]]; then
        echo -e "${UNIPROT}\t${START}\t${END}\t${SELECTED_SOURCE}\t${SELECTED_PDB_ID}\t${SELECTED_FILE}\t${SELECTED_COVERAGE}\t${SELECTED_RESOLUTION}" >> "${LOG}"

        if [[ "${SELECTED_SOURCE}" == pdb* ]]; then
            COUNT_PDB=$((COUNT_PDB + 1))
        else
            COUNT_AF2=$((COUNT_AF2 + 1))
        fi

        if [[ "${SELECTED_COVERAGE}" != "ok" ]]; then
            echo -e "${UNIPROT}\tSelected structure has ${SELECTED_COVERAGE}" >> "${MISSING}"
        fi
    else
        echo "  [WARN] No structure found at all"
        echo -e "${UNIPROT}\t${START}\t${END}\tnone\tNA\tNA\tNA\tNA" >> "${LOG}"
        echo -e "${UNIPROT}\tNo PDB or AF2 structure found" >> "${MISSING}"
        COUNT_NONE=$((COUNT_NONE + 1))
    fi

    sleep 0.3
}
# ── Process one fetch list ────────────────────────────────────────────────────
process_list() {
    local fetch_list="${1}"
    local list_label="${2}"

    local total
    total=$(wc -l < "${fetch_list}")
    local count=0

    echo ""
    echo "════════════════════════════════════════"
    echo "Processing ${list_label} (${total} entries)"
    echo "════════════════════════════════════════"

    while IFS=$'\t' read -r UNIPROT START END N_PEPTIDES BASE_ID; do

        count=$((count + 1))

        START=$(strip_float "${START}")
        END=$(strip_float "${END}")

        # ── Skip if already processed ────────────────────────────────────
        if [[ -n "${ALREADY_DONE[${UNIPROT}]+_}" ]]; then
            echo "[${count}/${total}] SKIP: ${UNIPROT}"
            continue
        fi

        echo "[${count}/${total}] ${UNIPROT} (region: ${START}-${END}, n_pep: ${N_PEPTIDES})"

        fetch_one "${UNIPROT}" "${START}" "${END}" "${BASE_ID}"

        ALREADY_DONE["${UNIPROT}"]=1

    done < "${fetch_list}"
}

# ── Run both lists ────────────────────────────────────────────────────────────
process_list "${CANONICAL_LIST}" "canonical IDs"
process_list "${ISOFORM_LIST}"   "isoform IDs"

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$((COUNT_PDB + COUNT_AF2 + COUNT_NONE))

echo ""
echo "════════════════════════════════════════"
echo "DONE — HYBRID STRUCTURE FETCHING"
echo "════════════════════════════════════════"
echo "Experimental PDB  : ${COUNT_PDB}"
echo "AlphaFold fallback : ${COUNT_AF2}"
echo "Nothing found      : ${COUNT_NONE}"
echo "Total processed    : ${TOTAL}"
echo ""

if [[ ${TOTAL} -gt 0 ]]; then
    PDB_PCT=$(python3 -c "print(f'{${COUNT_PDB}/${TOTAL}*100:.1f}')")
    AF2_PCT=$(python3 -c "print(f'{${COUNT_AF2}/${TOTAL}*100:.1f}')")
    echo "PDB fraction       : ${PDB_PCT}%"
    echo "AF2 fraction       : ${AF2_PCT}%"
fi

echo ""
echo "Output dir : ${OUT_DIR}/selected/"
echo "Log        : ${LOG}"
echo "Missing    : ${MISSING}"

# Write machine-readable summary
echo -e "source\tcount\tpercent" > "${SUMMARY}"
echo -e "pdb\t${COUNT_PDB}\t${PDB_PCT:-0}" >> "${SUMMARY}"
echo -e "alphafold\t${COUNT_AF2}\t${AF2_PCT:-0}" >> "${SUMMARY}"
echo -e "none\t${COUNT_NONE}\t0" >> "${SUMMARY}"
echo "Summary    : ${SUMMARY}"
echo "════════════════════════════════════════"
