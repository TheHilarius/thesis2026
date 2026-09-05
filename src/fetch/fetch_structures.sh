#!/usr/bin/env bash
# fetch_structures.sh
# Usage: bash src/fetch/fetch_structures.sh <out_dir>

set -uo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
if [[ $# -ne 1 ]]; then
    echo "Usage: bash src/fetch/fetch_structures.sh <out_dir>"
    echo "Example: bash src/fetch/fetch_structures.sh data/processed/structures"
    exit 1
fi

OUT_DIR="${1}"

CANONICAL_LIST="${OUT_DIR}/logs/fetch_list_canonical.tsv"
ISOFORM_LIST="${OUT_DIR}/logs/fetch_list_isoforms.tsv"

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
FASTA_PATH="data/raw/fasta/combined_full_length.fasta"

mkdir -p "${OUT_DIR}/alphafold"
mkdir -p "${OUT_DIR}/pdb_fallback"

LOG="${OUT_DIR}/logs/fetch_log.tsv"
MISSING="${OUT_DIR}/logs/missing_structures.tsv"

# ── Parse FASTA for ground-truth sequence lengths ────────────────────────────
declare -A FASTA_LENGTHS
if [[ -f "${FASTA_PATH}" ]]; then
    echo "Parsing ${FASTA_PATH} for sequence lengths..."
    while IFS= read -r header; do
        IFS= read -r seq_line
        # Accumulate multi-line sequences
        seq="${seq_line}"
        while IFS= read -r next_line; do
            if [[ "${next_line}" == ">"* ]]; then
                # Extract accession between pipes: >sp|O43236|SEPT4... -> O43236
                uid=$(echo "${header}" | awk -F'|' '{print $2}')
                FASTA_LENGTHS["${uid}"]=${#seq}
                header="${next_line}"
                seq=""
            else
                seq="${seq}${next_line}"
            fi
        done < /dev/null  # dummy — use process substitution below
        # This approach doesn't work for multi-line; use awk instead
    done < /dev/null
    # Use awk for reliable multi-line FASTA parsing
    while IFS=$'\t' read -r uid length; do
        FASTA_LENGTHS["${uid}"]="${length}"
    done < <(awk '/^>/{if(seq && hdr){
        split(hdr, a, "|");
        uid=a[2];
        printf "%s\t%d\n", uid, length(seq);
    } hdr=$0; seq=""; next}
    {seq=seq$0}
    END{if(seq && hdr){
        split(hdr, a, "|");
        uid=a[2];
        printf "%s\t%d\n", uid, length(seq);
    }}' "${FASTA_PATH}")
    echo "FASTA lengths loaded: ${#FASTA_LENGTHS[@]} entries"
else
    echo "WARNING: ${FASTA_PATH} not found — length validation disabled"
fi

# ── Resume logic ─────────────────────────────────────────────────────────────
declare -A ALREADY_DONE

if [[ -f "${LOG}" ]]; then
    echo "Existing log found — collecting already-processed entries..."
    while IFS=$'\t' read -r uid _rest; do
        ALREADY_DONE["${uid}"]=1
    done < <(tail -n +2 "${LOG}")
    echo "Already processed: ${#ALREADY_DONE[@]} entries"
else
    echo -e "uniprot_id\tstart\tend\tsource\tfile\tcoverage" > "${LOG}"
    echo -e "uniprot_id\treason" > "${MISSING}"
    echo "Fresh run — logs initialized"
fi

# ── Helper: strip .0 from pandas float coords ────────────────────────────────
strip_float() {
    printf "%.0f" "${1}"
}

# ── Helper: fetch AF2 structure, try v6 → v5 → v4 ───────────────────────────
# Returns the version number that worked, or "none"
# Usage: version=$(fetch_af2 "P04637" "/path/to/output.pdb")
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

# ── Helper: check PDB coverage ───────────────────────────────────────────────
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
        echo "partial:${min_res}-${max_res}"
    fi
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

        # ── Skip if already processed ────────────────────────────────────────
        if [[ -n "${ALREADY_DONE[${UNIPROT}]+_}" ]]; then
            echo "[${count}/${total}] SKIP: ${UNIPROT}"
            continue
        fi

        echo "[${count}/${total}] ${UNIPROT} (region: ${START}-${END}, n_pep: ${N_PEPTIDES})"

        # ── Step 1: AlphaFold2 ───────────────────────────────────────────────
        AF2_FILE="${OUT_DIR}/alphafold/${UNIPROT}.pdb"

        # Strict: fetch exact ID only, no canonical fallback
        WORKED_VERSION=$(fetch_af2 "${UNIPROT}" "${AF2_FILE}" || true)

        if [[ "${WORKED_VERSION}" != "none" ]]; then
            # ── Length validation: does AF2 model match FASTA ground truth? ──
            FASTA_LENGTH="${FASTA_LENGTHS[${UNIPROT}]:-}"
            if [[ -n "${FASTA_LENGTH}" ]]; then
                AF_LENGTH=$(grep "^SEQRES" "${AF2_FILE}" | \
                            awk '{print $4}' | sort -n | tail -1)
                if [[ -n "${AF_LENGTH}" && "${AF_LENGTH}" != "${FASTA_LENGTH}" ]]; then
                    echo "  [AF2] ⚠ Length mismatch: AF2=${AF_LENGTH}, FASTA=${FASTA_LENGTH}"
                    CORRECT_MODEL=$(python3 src/fetch/find_correct_af_model.py \
                                    "${UNIPROT}" "${FASTA_LENGTH}" 2>/dev/null || true)
                    if [[ -n "${CORRECT_MODEL}" ]]; then
                        echo "  [AF2] ✓ Found correct model: ${CORRECT_MODEL}"
                        ALT_URL="${AF2_BASE}/${CORRECT_MODEL}-model_v${WORKED_VERSION}.pdb"
                        HTTP_ALT=$(curl -s -o "${AF2_FILE}" \
                                       -w "%{http_code}" \
                                       --retry 2 \
                                       --max-time 30 \
                                       "${ALT_URL}" || echo "000")
                        if [[ "${HTTP_ALT}" == "200" ]]; then
                            echo "  [AF2] ✓ Downloaded ${CORRECT_MODEL} (saved as ${UNIPROT}.pdb)"
                        else
                            echo "  [AF2] ✗ Failed to download ${CORRECT_MODEL}, keeping default"
                            # Re-download the default
                            fetch_af2 "${UNIPROT}" "${AF2_FILE}" >/dev/null 2>&1 || true
                        fi
                    else
                        echo "  [AF2] ✗ No matching model found, keeping default"
                    fi
                fi
            fi
            COVERAGE=$(check_coverage "${AF2_FILE}" "${START}" "${END}")
            echo "  [AF2] ✓ Coverage: ${COVERAGE}"
            echo -e "${UNIPROT}\t${START}\t${END}\talphafold_v${WORKED_VERSION}\t${AF2_FILE}\t${COVERAGE}" >> "${LOG}"
            if [[ "${COVERAGE}" != "ok" ]]; then
                echo -e "${UNIPROT}\tAF2 partial coverage: ${COVERAGE}" >> "${MISSING}"
            fi
            ALREADY_DONE["${UNIPROT}"]=1
            sleep 0.2
            continue
        fi

        # AF2 failed — no canonical fallback, strict failure
        echo "  [AF2] ✗ Not found in AlphaFold (tried v6, v5, v4)"

        echo "  Falling through to PDB..."

        # ── Step 2: PDB fallback using BASE_ID ───────────────────────────────
        echo "  [PDB] Querying RCSB for ${BASE_ID}..."

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
            "paginate": {"start": 0, "rows": 5}
          },
          "return_type": "entry"
        }' "${BASE_ID}")

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

        if [[ -z "${PDB_IDS}" ]]; then
            echo "  [PDB] No structures found"
            echo -e "${UNIPROT}\tNo AF2 or PDB structure found" >> "${MISSING}"
            echo -e "${UNIPROT}\t${START}\t${END}\tnone\tNA\tNA" >> "${LOG}"
            ALREADY_DONE["${UNIPROT}"]=1
            sleep 0.3
            continue
        fi

        FOUND=0
        while IFS= read -r PDB_ID; do
            [[ -z "${PDB_ID}" ]] && continue

            PDB_FILE="${OUT_DIR}/pdb_fallback/${BASE_ID}_${PDB_ID}.pdb"

            HTTP_PDB=$(curl -s -o "${PDB_FILE}" \
                            -w "%{http_code}" \
                            --retry 2 \
                            --max-time 30 \
                            "${PDB_BASE}/${PDB_ID}.pdb" || echo "000")

            if [[ "${HTTP_PDB}" != "200" ]]; then
                rm -f "${PDB_FILE}"
                continue
            fi

            COVERAGE=$(check_coverage "${PDB_FILE}" "${START}" "${END}")
            echo "  [PDB] ${PDB_ID} coverage: ${COVERAGE}"

            if [[ "${COVERAGE}" == "ok" ]]; then
                echo -e "${UNIPROT}\t${START}\t${END}\tpdb:${PDB_ID}\t${PDB_FILE}\tok" >> "${LOG}"
                FOUND=1
                ALREADY_DONE["${UNIPROT}"]=1
                break
            fi

        done <<< "${PDB_IDS}"

        if [[ "${FOUND}" -eq 0 ]]; then
            echo "  [WARN] No structure covers full region ${START}-${END}"
            echo -e "${UNIPROT}\tNo structure covers ${START}-${END}" >> "${MISSING}"
            echo -e "${UNIPROT}\t${START}\t${END}\tpartial_only\tNA\tpartial" >> "${LOG}"
            ALREADY_DONE["${UNIPROT}"]=1
        fi

        sleep 0.3

    done < "${fetch_list}"
}

# ── Run both lists ────────────────────────────────────────────────────────────
process_list "${CANONICAL_LIST}" "canonical IDs"
process_list "${ISOFORM_LIST}"   "isoform IDs"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "DONE"
echo "════════════════════════════════════════"
echo "AlphaFold successes : $(grep -c 'alphafold' "${LOG}" 2>/dev/null || echo 0)"
echo "PDB fallback        : $(grep -c 'pdb:'      "${LOG}" 2>/dev/null || echo 0)"
echo "Nothing found       : $(grep -c 'none'      "${LOG}" 2>/dev/null || echo 0)"
echo "Partial only        : $(grep -c 'partial'   "${LOG}" 2>/dev/null || echo 0)"
echo ""
echo "Log     : ${LOG}"
echo "Missing : ${MISSING}"
