#!/usr/bin/env python3
"""
Create chunked prediction plan for OOM AlphaFold proteins (af_0041 to af_0154).

Reads both FASTA files, splits long sequences into 1000 aa chunks with 100 aa overlap,
writes per-chunk FASTA files, a combined FASTA, and generates the runner script.
"""

import os
import re
import stat

# ── Config ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
OVERLAP = 100
STEP = CHUNK_SIZE - OVERLAP  # 900

SHORT_FASTA = "data/raw/fasta/af_local_batch_short.fasta"
FULL_FASTA = "data/raw/fasta/af_local_batch.fasta"
CHUNKS_DIR = "data/processed/alphafold_output/chunks"
COLABFOLD_BIN = "/home/hilarius/tools/alphafold_localcolabfold/.pixi/envs/default/bin/colabfold_batch"
LOG_FILE = "data/processed/af_chunks_output.log"

FAILED_INDICES = list(range(41, 155))  # af_0041 to af_0154


# ── Helpers ─────────────────────────────────────────────────────────────────────
def parse_fasta(path):
    """Parse a 2-line-per-entry FASTA (header + sequence). Returns list of (id, seq)."""
    entries = []
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    for i in range(0, len(lines), 2):
        header = lines[i]  # >af_0001 or >sp|P06856|...
        seq = lines[i + 1]
        # Extract the short ID (af_NNNN) from header
        m = re.search(r">(af_\d+)", header)
        if m:
            af_id = m.group(1)
        else:
            af_id = header.lstrip(">")
        entries.append((af_id, seq))
    return entries


def parse_full_fasta_accessions(path):
    """Extract UniProt accessions from the full FASTA, in order."""
    accessions = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(">"):
                continue
            # >sp|P06856|... or >tr|H7C525|...
            parts = line.split("|")
            if len(parts) >= 2:
                accessions.append(parts[1])
    return accessions


def split_sequence(seq, chunk_size=CHUNK_SIZE, step=STEP):
    """Yield (chunk_idx, subsequence) for a protein sequence."""
    idx = 0
    start = 0
    while start < len(seq):
        end = start + chunk_size
        chunk_seq = seq[start:end]
        yield (idx, chunk_seq)
        idx += 1
        start += step


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    # Parse FASTA files
    short_entries = parse_fasta(SHORT_FASTA)
    short_ids = [e[0] for e in short_entries]
    id_to_seq = {e[0]: e[1] for e in short_entries}

    accessions = parse_full_fasta_accessions(FULL_FASTA)
    id_to_acc = {}
    for i, af_id in enumerate(short_ids):
        if i < len(accessions):
            id_to_acc[af_id] = accessions[i]

    # Filter to failed proteins only
    failed_ids = [f"af_{idx:04d}" for idx in FAILED_INDICES]
    failed_ids = [af for af in failed_ids if af in id_to_seq]

    print(f"Total proteins in FASTA: {len(short_ids)}")
    print(f"Proteins to chunk: {len(failed_ids)} (af_{FAILED_INDICES[0]:04d}–af_{FAILED_INDICES[-1]:04d})")

    # Create chunks directory
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    total_chunks = 0
    all_chunk_lines = []

    for af_id in failed_ids:
        seq = id_to_seq[af_id]
        acc = id_to_acc.get(af_id, "unknown")
        n_chunks = 0

        for chunk_idx, chunk_seq in split_sequence(seq):
            chunk_name = f"{af_id}_chunk_{chunk_idx:03d}"
            # Write per-chunk FASTA
            chunk_fasta = os.path.join(CHUNKS_DIR, f"{chunk_name}.fasta")
            with open(chunk_fasta, "w") as f:
                f.write(f">{chunk_name}\n{chunk_seq}\n")

            # Accumulate for combined FASTA
            all_chunk_lines.append(f">{chunk_name}\n{chunk_seq}\n")

            n_chunks += 1
            total_chunks += 1

        print(f"  {af_id} ({acc}): {len(seq)} aa → {n_chunks} chunks")

    # Write combined FASTA
    combined_path = os.path.join(CHUNKS_DIR, "all_chunks.fasta")
    with open(combined_path, "w") as f:
        f.writelines(all_chunk_lines)

    print(f"\nTotal chunks: {total_chunks}")
    print(f"Combined FASTA: {combined_path}")

    # ── Write runner script ──────────────────────────────────────────────────────
    runner_path = "src/tools/alphafold/af_run_chunks.sh"
    runner_content = f"""#!/usr/bin/env bash
set -euo pipefail

export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true

{COLABFOLD_BIN} \\
    --num-models 1 \\
    --num-recycle 3 \\
    --max-extra-seq 512 \\
    --max-seq 128 \\
    --use-pallas \\
    --stop-at-score 85 \\
    --random-seed 0 \\
    {combined_path} \\
    data/processed/alphafold_output/chunks/ \\
    2>&1 | tee {LOG_FILE}
"""
    with open(runner_path, "w") as f:
        f.write(runner_content)
    os.chmod(runner_path, os.stat(runner_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"\nRunner script written: {runner_path}")

    # ── Write stitch script (File 3) ────────────────────────────────────────────
    stitch_path = "src/tools/alphafold/af_stitch_plddt.py"
    if not os.path.exists(stitch_path):
        # This should be created by the user separately; just note it
        print(f"NOTE: {stitch_path} should be created separately")

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Proteins to process:  {len(failed_ids)}")
    print(f"  Total chunks:         {total_chunks}")
    print(f"  Chunk size:           {CHUNK_SIZE} aa (overlap: {OVERLAP} aa, step: {STEP} aa)")
    print(f"  Output directory:     {CHUNKS_DIR}/")
    print(f"\nTo launch:")
    print(f"  tmux new-session -d -s af-chunks 'bash src/tools/alphafold/af_run_chunks.sh'")


if __name__ == "__main__":
    main()
