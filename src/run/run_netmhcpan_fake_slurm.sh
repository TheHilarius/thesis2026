#!/bin/bash
# run_netmhcpan_fake_slurm.sh — array job: NetMHCpan 4.2 on fake-protein
# FASTA batches (32_fake_protein_netmhcpan_logo.py --generate output).
#
# One task per batch file, resume-safe: a task whose output CSV already
# exists and is complete (>2 lines) is skipped, so the array can be
# re-submitted after partial failure with the same command.
#
# Usage (from repo root, after rsync of runs/fake_protein_netmhcpan/fasta/):
#   sbatch src/run/run_netmhcpan_fake_slurm.sh
#
# netMHCpan is NOT in the cluster default PATH (verified Nov 2026) — the
# shared install wrapper (tcsh script that sets $NETMHCpan) is prepended.
# NOTE: call `netMHCpan` (the wrapper), NOT Linux_x86_64/bin/netMHCpan-4.2
# (the bare binary needs $NETMHCpan set manually).

#SBATCH --job-name=nmp_fake
#SBATCH --partition=cpu
#SBATCH --array=0-153%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=runs/fake_protein_netmhcpan/slurm/%x_%a.out

set -euo pipefail

export PATH="/home/projects/netMHCpan4.2/netMHCpan-4.2:$PATH"

BATCHES=(runs/fake_protein_netmhcpan/fasta/batch_*.fasta)
BATCH="${BATCHES[$SLURM_ARRAY_TASK_ID]}"
OUT_CSV="runs/fake_protein_netmhcpan/netmhcpan/$(basename "${BATCH}" .fasta).csv"

mkdir -p runs/fake_protein_netmhcpan/netmhcpan

if [ -s "$OUT_CSV" ] && [ "$(wc -l < "$OUT_CSV")" -gt 2 ]; then
    echo "[SKIP] $OUT_CSV already complete"
    exit 0
fi

echo "[RUN] task $SLURM_ARRAY_TASK_ID on $(hostname): $BATCH"
time netMHCpan \
    -a HLA-A02:01 \
    -l 9 \
    -xls \
    -xlsfile "$OUT_CSV" \
    -f "$BATCH"
