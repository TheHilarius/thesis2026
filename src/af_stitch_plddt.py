#!/usr/bin/env python3
"""
Stitch per-residue pLDDT from chunked AlphaFold predictions.

Reads *_scores_rank_001_*.json from the chunks output directory,
groups by protein ID, sorts by chunk index, and reconstructs full pLDDT
using max-overlap for overlapping residues.

Outputs:
  - Per-residue CSVs: data/processed/alphafold_plddt_af_NNNN.csv
  - Summary CSV:      data/processed/alphafold_plddt_stitched.csv
"""

import csv
import glob
import json
import os
import re

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────────
CHUNKS_DIR = "data/processed/alphafold_output/chunks"
OUTDIR = "data/processed"
FULL_FASTA = "data/raw/fasta/af_local_batch.fasta"
STEP = 900  # chunk_size - overlap = 1000 - 100


# ── Helpers ─────────────────────────────────────────────────────────────────────
def parse_fasta_accessions(path):
    """Extract (af_id_index, accession) pairs from full FASTA, in order."""
    accessions = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(">"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                accessions.append(parts[1])
    return accessions


def parse_scores_file(filepath):
    """Extract af_id and chunk_idx from a scores JSON filename."""
    basename = os.path.basename(filepath)
    # Pattern: af_NNNN_chunk_NNN_scores_rank_001_...json
    m = re.match(r"(af_\d+)_chunk_(\d+)_scores_rank_001_", basename)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    # Find all score files
    pattern = os.path.join(CHUNKS_DIR, "af_*_chunk_*_scores_rank_001_*.json")
    score_files = sorted(glob.glob(pattern))

    if not score_files:
        print(f"No score files found matching: {pattern}")
        print("Run af_run_chunks.sh first, then re-run this script.")
        return

    print(f"Found {len(score_files)} score files")

    # Parse accessions from full FASTA
    accessions = parse_fasta_accessions(FULL_FASTA)

    # Group chunks by protein ID
    proteins = {}  # af_id -> {chunk_idx: plddt_array}
    for fp in score_files:
        af_id, chunk_idx = parse_scores_file(fp)
        if af_id is None:
            print(f"  SKIP (unparseable): {fp}")
            continue

        with open(fp) as f:
            data = json.load(f)

        plddt = np.array(data["plddt"], dtype=np.float32)

        if af_id not in proteins:
            proteins[af_id] = {}
        proteins[af_id][chunk_idx] = plddt

    print(f"Proteins with chunks: {len(proteins)}")

    # Build af_id → accession mapping (by index)
    # Extract numeric ID from af_NNNN
    af_id_to_acc = {}
    for i, acc in enumerate(accessions):
        af_id = f"af_{i+1:04d}"
        af_id_to_acc[af_id] = acc

    # Stitch each protein
    summary_rows = []
    for af_id in sorted(proteins.keys()):
        chunks = proteins[af_id]
        sorted_indices = sorted(chunks.keys())
        n_chunks = len(sorted_indices)

        # Determine total length: last chunk start + its length
        last_idx = sorted_indices[-1]
        last_len = len(chunks[last_idx])
        total_len = last_idx * STEP + last_len

        # Initialize pLDDT array (use 0 as sentinel; will be overwritten)
        full_plddt = np.zeros(total_len, dtype=np.float32)
        coverage = np.zeros(total_len, dtype=np.int32)

        for ci in sorted_indices:
            arr = chunks[ci]
            start = ci * STEP
            end = start + len(arr)
            # Extend if needed
            if end > total_len:
                full_plddt = np.pad(full_plddt, (0, end - total_len))
                coverage = np.pad(coverage, (0, end - total_len))
                total_len = end
            # Max overlap
            full_plddt[start:end] = np.maximum(full_plddt[start:end], arr)
            coverage[start:end] += 1

        max_coverage = int(coverage.max())

        # Per-residue CSV
        per_residue_path = os.path.join(OUTDIR, f"alphafold_plddt_{af_id}.csv")
        with open(per_residue_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["residue_idx", "plddt"])
            for idx, val in enumerate(full_plddt):
                writer.writerow([idx, f"{val:.2f}"])

        # Summary stats
        accession = af_id_to_acc.get(af_id, "unknown")
        mean_plddt = float(np.mean(full_plddt))
        frac_above_70 = float(np.mean(full_plddt >= 70))

        summary_rows.append({
            "af_id": af_id,
            "accession": accession,
            "length": len(full_plddt),
            "mean_plddt": round(mean_plddt, 2),
            "frac_above_70": round(frac_above_70, 4),
            "n_chunks": n_chunks,
            "max_coverage": max_coverage,
        })

        print(f"  {af_id} ({accession}): {len(full_plddt)} residues, "
              f"mean pLDDT={mean_plddt:.1f}, >70={frac_above_70:.1%}, "
              f"chunks={n_chunks}, max_cov={max_coverage}")

    # Write summary CSV
    summary_path = os.path.join(OUTDIR, "alphafold_plddt_stitched.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "af_id", "accession", "length", "mean_plddt",
            "frac_above_70", "n_chunks", "max_coverage"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"STITCHING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total proteins stitched: {len(summary_rows)}")
    print(f"  Per-residue CSVs:  {OUTDIR}/alphafold_plddt_af_NNNN.csv")
    print(f"  Summary CSV:       {summary_path}")

    # Distribution of mean pLDDT
    if summary_rows:
        means = [r["mean_plddt"] for r in summary_rows]
        print(f"\n  Mean pLDDT distribution:")
        print(f"    Min:    {min(means):.1f}")
        print(f"    Median: {np.median(means):.1f}")
        print(f"    Max:    {max(means):.1f}")
        print(f"    Std:    {np.std(means):.1f}")
        # Bucket distribution
        buckets = {"<50": 0, "50-70": 0, "70-80": 0, "80-90": 0, ">90": 0}
        for m in means:
            if m < 50:
                buckets["<50"] += 1
            elif m < 70:
                buckets["50-70"] += 1
            elif m < 80:
                buckets["70-80"] += 1
            elif m < 90:
                buckets["80-90"] += 1
            else:
                buckets[">90"] += 1
        for label, count in buckets.items():
            print(f"    {label:>5}: {count} proteins")


if __name__ == "__main__":
    main()
