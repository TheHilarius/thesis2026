#!/usr/bin/env bash
# fetch_structures.sh
# Usage: bash src/fetch/fetch_structures.sh <out_dir> [allele] [list_tag]
#   out_dir:   shared structures dir (data/processed/structures)
#   allele:    hla0201 (default — legacy call preserved) or hla_a0101.
#              Selects the positives-only FASTA used for length
#              validation; PDBs and logs stay in the SHARED out_dir.
#   list_tag:  optional suffix used by prepare_fetch_list.sh (e.g.
#              "a0101"). Empty/omitted = legacy list names. Must match
#              the tag the fetch lists were prepared with.
#
# Example (a0101):
#   bash src/fetch/prepare_fetch_list.sh \
#       data/processed/hla_a0101/pos_EL_all_epitopes_hla_a0101.csv \
#       data/processed/structures a0101
#   bash src/fetch/fetch_structures.sh data/processed/structures hla_a0101 a0101
set -uo pipefail

# Arguments
if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: bash src/fetch/fetch_structures.sh <out_dir> [hla0201|hla_a0101] [list_tag]"
    echo "  Example (legacy):  bash src/fetch/fetch_structures.sh data/processed/structures"
    echo "  Example (a0101):   bash src/fetch/fetch_structures.sh data/processed/structures hla_a0101 a0101"
    exit 1
fi

OUT_DIR="${1}"
ALLELE="${2:-hla0201}"
TAG="${3:-}"

case "${ALLELE}" in
  hla0201)
    FASTA_PATH="data/raw/fasta/combined_positives_only.fasta" ;;
  hla_a0101)
    FASTA_PATH="data/raw/fasta/hla_a0101_positives_only.fasta" ;;
  *)
    echo "ERROR: unknown allele '${ALLELE}' (expected hla0201 or hla_a0101)"
    exit 1 ;;
esac

_LIST_SUFFIX=""
if [[ -n "${TAG}" ]]; then _LIST_SUFFIX="_${TAG}"; fi

CANONICAL_LIST="${OUT_DIR}/logs/fetch_list_canonical${_LIST_SUFFIX}.tsv"
ISOFORM_LIST="${OUT_DIR}/logs/fetch_list_isoforms${_LIST_SUFFIX}.tsv"

if [[ ! -f "${CANONICAL_LIST}" ]]; then
    echo "ERROR: ${CANONICAL_LIST} not found — run prepare_fetch_list.sh first"
    exit 1
fi
if [[ ! -f "${ISOFORM_LIST}" ]]; then
    echo "ERROR: ${ISOFORM_LIST} not found — run prepare_fetch_list.sh first"
    exit 1
fi

# Config
AF2_BASE="https://alphafold.ebi.ac.uk/files"
# FASTA_PATH set per-allele in the case block above
mkdir -p "${OUT_DIR}/alphafold"
# Hybrid shared-pool design: the resume READS the shared legacy fetch_log
# (proteins already fetched for another allele are skipped — AlphaFold PDBs
# are per-UNIPROT, allele-independent), but all NEW entries append to the
# per-allele TAGGED log. The shared files are owned by the legacy run
# (mode 644) and must never be written to. Empty tag = legacy call:
# LOG == LOG_READ, byte-identical legacy behavior.
LOG_READ="${OUT_DIR}/logs/fetch_log.tsv"
LOG="${OUT_DIR}/logs/fetch_log${_LIST_SUFFIX}.tsv"
MISSING="${OUT_DIR}/logs/missing_structures${_LIST_SUFFIX}.tsv"

# Parse FASTA for ground-truth sequence lengths 
declare -A FASTA_LENGTHS
# Lengths live in FASTA_LENGTHS[uid] for this run only (bash assoc array).
# Real parser is the awk block after the unused while-loop (reads /dev/null).
# Same idea: uid from header pipe-field, length of accumulated seq.
if [[ -f "${FASTA_PATH}" ]]; then
    echo "Parsing ${FASTA_PATH} for sequence lengths..."
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

# Resume logic: union of the shared legacy log (read-only) and this
# allele's own tagged log — identical for the legacy call (same file).
declare -A ALREADY_DONE

for _resume_log in "${LOG_READ}" "${LOG}"; do
    if [[ -f "${_resume_log}" ]]; then
        echo "Existing log found (${_resume_log}) — collecting already-processed entries..."
        while IFS=$'\t' read -r uid _rest; do
            ALREADY_DONE["${uid}"]=1
        done < <(tail -n +2 "${_resume_log}")
    fi
done
echo "Already processed: ${#ALREADY_DONE[@]} entries"

if [[ ! -f "${LOG}" ]]; then
    echo -e "uniprot_id\tstart\tend\tsource\tfile\tcoverage" > "${LOG}"
    echo "Fresh run — tagged log initialized: ${LOG}"
fi
if [[ ! -f "${MISSING}" ]]; then
    echo -e "uniprot_id\treason" > "${MISSING}"
fi

# Helper: strip .0 from pandas float coords 
strip_float() {
    printf "%.0f" "${1}"
}

# Helper: fetch AF2 structure, try v6 → v5 → v4 
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

# Helper: check PDB coverage 
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

# Process one fetch list 
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
        # Skip if already processed 
        if [[ -n "${ALREADY_DONE[${UNIPROT}]+_}" ]]; then
            echo "[${count}/${total}] SKIP: ${UNIPROT}"
            continue
        fi
        echo "[${count}/${total}] ${UNIPROT} (region: ${START}-${END}, n_pep: ${N_PEPTIDES})"

        # Step 1: AlphaFold2 
        AF2_FILE="${OUT_DIR}/alphafold/${UNIPROT}.pdb"

        # Strict: fetch exact ID only, no canonical fallback
        WORKED_VERSION=$(fetch_af2 "${UNIPROT}" "${AF2_FILE}" || true)

        if [[ "${WORKED_VERSION}" != "none" ]]; then
            # Length validation: does AF2 model match FASTA ground truth? 
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

        # AF2 failed — no model available for this exact accession
        echo "  [AF2] ✗ Not found in AlphaFold (tried v6, v5, v4)"
        echo -e "${UNIPROT}\t${START}\t${END}\tnone\tNA\tNA" >> "${LOG}"
        echo -e "${UNIPROT}\tNo AlphaFold model" >> "${MISSING}"
        ALREADY_DONE["${UNIPROT}"]=1
        sleep 0.3

    done < "${fetch_list}"
}

# Run both lists 
process_list "${CANONICAL_LIST}" "canonical IDs"
process_list "${ISOFORM_LIST}"   "isoform IDs"

# Summary 
echo ""
echo "════════════════════════════════════════"
echo "DONE"
echo "════════════════════════════════════════"
echo "AlphaFold successes : $(grep -c 'alphafold' "${LOG}" 2>/dev/null || echo 0)"
echo "Nothing found       : $(grep -c 'none'      "${LOG}" 2>/dev/null || echo 0)"
echo "Partial only        : $(grep -c 'partial'   "${LOG}" 2>/dev/null || echo 0)"
echo ""
echo "Log     : ${LOG}"
echo "Missing : ${MISSING}"
