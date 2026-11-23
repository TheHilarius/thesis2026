#!/bin/bash
#SBATCH --job-name=a0101_nmpan
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/netmhcpan/a0101_nmpan_%j.out
#SBATCH --error=logs/netmhcpan/a0101_nmpan_%j.err

# netMHCpan HLA-A*01:01 Slurm wrapper (Stage A, step 4 of the a0101 rerun)
# https://docs/hla_a0101_run_recipe.md
#
# Purpose: run the allele-aware netMHCpan batch runner on a compute node
# and leave RECEIPTS: Slurm .out/.err log + per-batch netMHCpan .csv/.txt.
# The version banner netMHCpan writes into every .txt doubles as the
# binary-version record for the thesis methods section (server runs 4.2e
# vs legacy 4.2b — same family, note the drift).
#
# Usage (submit from repo root, after step 3 batching):
#   NMPAN_DIR=$(dirname "$(which netMHCpan)") \
#     sbatch --export=ALL src/run/run_netmhcpan_a0101_slurm.sh
# NMPAN_DIR is only needed if netMHCpan is not on the compute node's
# default PATH (batch shells do not source your login-shell PATH).
# Override partition/time on the command line if needed:
#   sbatch --partition=other --time=48:00:00 src/run/run_netmhcpan_a0101_slurm.sh
#
# Inputs:
#   data/raw/fasta/a0101_batches/batch_*.fasta   (src/util/batch_fasta.py, 50/batch)
# Outputs:
#   data/processed/hla_a0101/netmhcpan/hla_a0101_batch_*.csv/.txt
#   logs/netmhcpan/a0101_nmpan_<jobid>.out/.err   (receipts)
#
# Resume-safe: the runner skips batches whose csv+txt are non-empty, so a
# failed/timed-out job is resubmitted as-is — only missing batches rerun.

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ── netMHCpan binary resolution ──────────────────────────────────────────────
if [ -n "${NMPAN_DIR:-}" ]; then
    export PATH="${NMPAN_DIR}:${PATH}"
fi
if ! command -v netMHCpan >/dev/null 2>&1; then
    echo "[FATAL] netMHCpan not on PATH in this batch shell."
    echo "        On the login shell, find it:  which netMHCpan"
    echo "        Then resubmit:"
    echo "        NMPAN_DIR=\$(dirname \$(which netMHCpan)) sbatch --export=ALL src/run/run_netmhcpan_a0101_slurm.sh"
    exit 1
fi

mkdir -p logs/netmhcpan

# ── Receipt header ───────────────────────────────────────────────────────────
echo "=== netMHCpan a0101 — Slurm receipt ==="
echo "  Job ID:        ${SLURM_JOB_ID:-unknown}"
echo "  Node:          ${SLURMD_NODENAME:-unknown}"
echo "  Submit dir:    ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "  Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Binary:        $(command -v netMHCpan)"
echo "  Binary info:   $(netMHCpan 2>&1 | head -n 1 || true)"
echo "  Allele:        HLA-A01:01  (-l 9, EL only)"
echo ""

# ── Run ─────────────────────────────────────────────────────────────────────
bash src/tools/netmhcpan/run_netmhcpan_parallel.sh \
    data/raw/fasta/a0101_batches \
    data/processed/hla_a0101/netmhcpan \
    hla_a0101 10 HLA-A01:01 9

# NOTE: the runner exits 0 even if individual batches fail (it prints a
# failed list instead) — so verify outputs explicitly:
EXPECTED=$(ls data/raw/fasta/a0101_batches/batch_*.fasta 2>/dev/null | wc -l)
N_CSV=$(ls   data/processed/hla_a0101/netmhcpan/hla_a0101_batch_*.csv 2>/dev/null | wc -l)
N_TXT=$(ls   data/processed/hla_a0101/netmhcpan/hla_a0101_batch_*.txt 2>/dev/null | wc -l)

# ── Receipt footer ───────────────────────────────────────────────────────────
echo ""
echo "=== Receipt footer ==="
echo "  Finished:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Batches found: ${EXPECTED}"
echo "  CSV outputs:   ${N_CSV}"
echo "  TXT outputs:   ${N_TXT}"

if [ "${N_CSV}" -lt "${EXPECTED}" ] || [ "${N_TXT}" -lt "${EXPECTED}" ]; then
    echo "  Status:        INCOMPLETE — missing $((EXPECTED - N_CSV)) csv / $((EXPECTED - N_TXT)) txt"
    echo "  (resubmit this same script; completed batches are skipped)"
    exit 1
fi
echo "  Status:        OK — all ${EXPECTED} batches have csv+txt"
exit 0
