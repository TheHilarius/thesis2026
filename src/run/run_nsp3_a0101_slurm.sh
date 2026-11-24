#!/bin/bash
#SBATCH --job-name=a0101_nsp3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/netsurfp/a0101_nsp3_%j.out
#SBATCH --error=logs/netsurfp/a0101_nsp3_%j.err

# NetSurfP-3.0 HLA-A*01:01 Slurm wrapper (Stage B, step 7 of the a0101 rerun)
# See docs/hla_a0101_run_recipe.md
#
# Modeled on the proven ../run_h100.sh pattern (the Sep 2026 legacy runs:
# ~0.2 s/sequence on L40S). GPU type is PINNED to L40S: the nsp3_gpu env
# is torch 2.5.1+cu121, which has NO kernels for the Blackwell nodes
# (compute07-08) — do not switch to plain gpu:1 unless the env is
# upgraded. H100 nodes (cu121, sm_90) also work.
#
# Usage (submit from repo root, after nsp3_clean_fasta.sh +
# prepare_nsp3_batches.sh hla_a0101):
#   sbatch src/run/run_nsp3_a0101_slurm.sh
#
# Inputs:
#   data/raw/fasta/a0101_nsp3_batches/*batch_*.fasta  (~29 batches, 250 seqs)
# Outputs:
#   data/processed/nsp3/a0101_batch_*/  (per-protein CSVs, shared root,
#                                        per-allele prefixed batch dirs)
#   logs/netsurfp/a0101_nsp3_<jobid>.out/.err  (receipts)
#
# Resume-safe: the runner skips batches whose output dir already has CSVs.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs/netsurfp data/processed/nsp3

# ── Receipt header ───────────────────────────────────────────────────────────
echo "=== NetSurfP-3.0 a0101 — Slurm receipt ==="
echo "  Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "  Node:          ${SLURMD_NODENAME:-unknown}"
echo "  GPU:           $(nvidia-smi -L 2>/dev/null | head -1 || echo 'nvidia-smi unavailable')"
echo "  Submit dir:    ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "  Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:         ${NSP3_DIR:-/home/projects1/thesis_s204692_s204581/NetSurfP-3.0_standalone}/models/nsp3.pth"
echo ""

# ── Run ─────────────────────────────────────────────────────────────────────
bash src/tools/netsurfp/run_netsurfp3.sh data/raw/fasta/a0101_nsp3_batches

# ── Verify + receipt footer ────────────────────────────────────────────────
EXPECTED=$(ls data/raw/fasta/a0101_nsp3_batches/*batch_*.fasta 2>/dev/null | wc -l)
DONE=0
MISSING=""
for f in data/raw/fasta/a0101_nsp3_batches/*batch_*.fasta; do
    batch=$(basename "$f" .fasta)
    if [ -d "data/processed/nsp3/${batch}" ] && \
       [ "$(ls data/processed/nsp3/${batch}/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
        DONE=$((DONE + 1))
    else
        MISSING="${MISSING} ${batch}"
    fi
done

echo ""
echo "=== Receipt footer ==="
echo "  Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Batches found: ${EXPECTED}"
echo "  With CSVs:     ${DONE}"
if [ "${DONE}" -lt "${EXPECTED}" ]; then
    echo "  Status:        INCOMPLETE — missing:${MISSING}"
    echo "  (resubmit this same script; completed batches are skipped)"
    exit 1
fi
echo "  Status:        OK — all ${EXPECTED} batches have CSVs"
exit 0
