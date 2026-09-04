#!/usr/bin/env python3
"""
09_sequence_consistency_audit.py — Step 1: Global Alignment Audit

Compares the canonical UniProt sequence (from df_all['sequence']) against
the full intended sequence from AlphaFold PDB files (SEQRES records).

Purpose: Detect silent coordinate shifts caused by isoform canonicalization
in the R pipeline (stripping suffixes like O43829-7 → O43829, then attaching
isoform-mapped coordinates to the canonical sequence string).

Input:
  data/processed/df_all.csv
  data/processed/structures/alphafold/

Output:
  data/processed/sequence_audit/alignment_audit.csv
  data/processed/sequence_audit/alignment_summary.txt
  results/figures/sequence_consistency/sequence_identity_distribution.png
  results/figures/sequence_consistency/length_difference_distribution.png
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from Bio.Align import PairwiseAligner, substitution_matrices

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DF_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "df_all.csv"
AF_DIR = PROJECT_ROOT / "data" / "processed" / "structures" / "alphafold"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sequence_audit"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "sequence_consistency"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 3-letter → 1-letter amino acid mapping
AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'UNK': 'X',
}


# ── SEQRES Parser ─────────────────────────────────────────────────────────────
def parse_seqres(pdb_path):
    """
    Extract the full intended sequence from PDB SEQRES records.

    SEQRES format:
      SEQRES   1 A  437  MET THR THR SER THR LEU GLN LYS ALA ILE ...
      SEQRES   2 A  437  THR LYS ALA THR GLU GLU ASP LYS ALA LYS ...

    Returns 1-letter amino acid sequence string, or None on error.
    """
    try:
        residues = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('SEQRES'):
                    # Columns: SEQRES serial chain num_residues res1 res2 ...
                    parts = line.split()
                    if len(parts) >= 5:
                        residues.extend(parts[4:])
        if not residues:
            return None
        return ''.join(AA3_TO_1.get(r, 'X') for r in residues)
    except Exception as e:
        print(f"  ERROR parsing SEQRES from {pdb_path}: {e}")
        return None


# ── Alignment ─────────────────────────────────────────────────────────────────
def create_aligner():
    """Create a global pairwise aligner with BLOSUM62."""
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def compute_alignment_stats(aligner, seq1, seq2):
    """
    Align seq1 (UniProt canonical) vs seq2 (AF2 SEQRES).
    Returns dict with alignment statistics.
    """
    alignments = aligner.align(seq1, seq2)
    a = alignments[0]
    counts = a.counts()

    aligned_len = a.shape[1]
    identities = counts.identities
    mismatches = counts.mismatches
    gaps = counts.gaps

    pct_identity = (identities / aligned_len * 100) if aligned_len > 0 else 0.0

    return {
        'score': a.score,
        'aligned_length': aligned_len,
        'identities': identities,
        'mismatches': mismatches,
        'gaps': gaps,
        'pct_identity': round(pct_identity, 2),
        'length_diff': len(seq2) - len(seq1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  SEQUENCE CONSISTENCY AUDIT — Step 1: Global Alignment")
    print("=" * 65)
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading {DF_ALL_PATH}...")
    df = pd.read_csv(DF_ALL_PATH)
    print(f"  Total rows: {len(df)}")

    # Get unique proteins with their canonical sequences
    unique_proteins = df.groupby('uniprot_id').first().reset_index()
    n_proteins = len(unique_proteins)
    print(f"  Unique proteins: {n_proteins}")

    # Check which have AF2 files
    af_files = set(f.replace('.pdb', '') for f in os.listdir(AF_DIR) if f.endswith('.pdb'))
    n_with_af = unique_proteins['uniprot_id'].isin(af_files).sum()
    n_missing_af = n_proteins - n_with_af
    print(f"  Proteins with AF2 structure: {n_with_af}")
    print(f"  Proteins without AF2 structure: {n_missing_af}")

    # ── Create aligner ─────────────────────────────────────────────────────────
    print("\nCreating BLOSUM62 global aligner...")
    aligner = create_aligner()

    # ── Run alignment audit ────────────────────────────────────────────────────
    print(f"\nRunning alignment audit on {n_with_af} proteins...")
    results = []
    n_exact = 0
    n_mismatch = 0
    n_no_af = 0
    n_parse_error = 0

    for i, row in unique_proteins.iterrows():
        uid = row['uniprot_id']
        uniprot_seq = str(row['sequence'])

        # Check AF2 file exists
        af_path = AF_DIR / f"{uid}.pdb"
        if not af_path.exists():
            n_no_af += 1
            results.append({
                'uniprot_id': uid,
                'uniprot_len': len(uniprot_seq),
                'seqres_len': None,
                'length_diff': None,
                'pct_identity': None,
                'identities': None,
                'mismatches': None,
                'gaps': None,
                'score': None,
                'status': 'no_af_structure',
            })
            continue

        # Parse SEQRES
        seqres_seq = parse_seqres(af_path)
        if seqres_seq is None or len(seqres_seq) == 0:
            n_parse_error += 1
            results.append({
                'uniprot_id': uid,
                'uniprot_len': len(uniprot_seq),
                'seqres_len': 0,
                'length_diff': None,
                'pct_identity': None,
                'identities': None,
                'mismatches': None,
                'gaps': None,
                'score': None,
                'status': 'parse_error',
            })
            continue

        # Align
        stats = compute_alignment_stats(aligner, uniprot_seq, seqres_seq)

        # Determine status
        if stats['pct_identity'] == 100.0 and stats['length_diff'] == 0:
            status = 'exact_match'
            n_exact += 1
        else:
            status = 'mismatch'
            n_mismatch += 1

        results.append({
            'uniprot_id': uid,
            'uniprot_len': len(uniprot_seq),
            'seqres_len': len(seqres_seq),
            'length_diff': stats['length_diff'],
            'pct_identity': stats['pct_identity'],
            'identities': stats['identities'],
            'mismatches': stats['mismatches'],
            'gaps': stats['gaps'],
            'score': stats['score'],
            'status': status,
        })

        # Progress
        done = n_exact + n_mismatch + n_parse_error
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_with_af - done) / rate if rate > 0 else 0
            print(f"  [{done:>5}/{n_with_af}]  "
                  f"{rate:.1f} prot/s  "
                  f"ETA: {eta/60:.1f}min  "
                  f"exact: {n_exact}  mismatch: {n_mismatch}")

    # ── Build results DataFrame ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    # ── Save audit CSV ─────────────────────────────────────────────────────────
    audit_path = OUT_DIR / "alignment_audit.csv"
    results_df.to_csv(audit_path, index=False)
    print(f"\nSaved: {audit_path}")

    # ── Summary statistics ─────────────────────────────────────────────────────
    n_total = len(results_df)
    n_checked = n_exact + n_mismatch + n_parse_error
    n_exact_only = results_df[results_df['status'] == 'exact_match']
    n_mismatch_only = results_df[results_df['status'] == 'mismatch']

    # Identity stats for mismatches
    if len(n_mismatch_only) > 0:
        mismatch_identities = n_mismatch_only['pct_identity'].dropna()
        mismatch_lengths = n_mismatch_only['length_diff'].dropna()
    else:
        mismatch_identities = pd.Series()
        mismatch_lengths = pd.Series()

    # Length diff stats for mismatches
    if len(mismatch_lengths) > 0:
        len_longer = (mismatch_lengths > 0).sum()
        len_shorter = (mismatch_lengths < 0).sum()
        len_same = (mismatch_lengths == 0).sum()
    else:
        len_longer = len_shorter = len_same = 0

    summary_lines = [
        "=" * 65,
        "SEQUENCE CONSISTENCY AUDIT — SUMMARY",
        "=" * 65,
        f"",
        f"Total unique proteins:        {n_total}",
        f"  With AF2 structure:         {n_with_af}",
        f"  Without AF2 structure:      {n_no_af}",
        f"",
        f"Alignment results (proteins with AF2):",
        f"  Exact match:                {n_exact} ({n_exact/n_with_af*100:.1f}%)",
        f"  Mismatch:                   {n_mismatch} ({n_mismatch/n_with_af*100:.1f}%)",
        f"  Parse error:                {n_parse_error}",
        f"",
    ]

    if len(n_mismatch_only) > 0:
        summary_lines.extend([
            f"Mismatch details:",
            f"  Mean identity:              {mismatch_identities.mean():.2f}%",
            f"  Median identity:            {mismatch_identities.median():.2f}%",
            f"  Min identity:               {mismatch_identities.min():.2f}%",
            f"  Mean length diff:           {mismatch_lengths.mean():.1f} residues",
            f"  AF2 longer than UniProt:    {len_longer}",
            f"  AF2 shorter than UniProt:   {len_shorter}",
            f"  Same length but diff seq:   {len_same}",
            f"",
            f"  Top 20 worst mismatches (by identity):",
        ])
        worst = n_mismatch_only.nsmallest(20, 'pct_identity')
        for _, r in worst.iterrows():
            summary_lines.append(
                f"    {r['uniprot_id']:<12}  "
                f"UniProt={r['uniprot_len']:>5}  "
                f"AF2={r['seqres_len']:>5}  "
                f"diff={r['length_diff']:>+5}  "
                f"identity={r['pct_identity']:.1f}%"
            )

    # Worst by length diff
    if len(mismatch_lengths) > 0:
        worst_len = n_mismatch_only.reindex(
            n_mismatch_only['length_diff'].abs().sort_values(ascending=False).index
        ).head(20)
        summary_lines.extend([
            f"",
            f"  Top 20 worst mismatches (by length diff):",
        ])
        for _, r in worst_len.iterrows():
            summary_lines.append(
                f"    {r['uniprot_id']:<12}  "
                f"UniProt={r['uniprot_len']:>5}  "
                f"AF2={r['seqres_len']:>5}  "
                f"diff={r['length_diff']:>+5}  "
                f"identity={r['pct_identity']:.1f}%"
            )

    summary_lines.extend([
        f"",
        f"{'=' * 65}",
        f"Runtime: {time.time() - t0:.1f}s",
        f"{'=' * 65}",
    ])

    summary_text = '\n'.join(summary_lines)

    # Save summary
    summary_path = OUT_DIR / "alignment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Saved: {summary_path}")

    # Print summary
    print(f"\n{summary_text}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    # Only plot proteins with AF2 structures
    checked = results_df[results_df['status'].isin(['exact_match', 'mismatch'])]

    if len(checked) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Identity distribution
        ax1 = axes[0]
        identities = checked['pct_identity'].dropna()
        ax1.hist(identities, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
        ax1.axvline(x=100, color='red', linestyle='--', linewidth=1, label='100% identity')
        ax1.set_xlabel('Sequence Identity (%)')
        ax1.set_ylabel('Number of Proteins')
        ax1.set_title('Sequence Identity: UniProt vs AF2 SEQRES')
        ax1.legend()

        # Add text annotation
        n_exact_pct = (checked['status'] == 'exact_match').sum() / len(checked) * 100
        ax1.text(0.02, 0.95, f'{n_exact_pct:.1f}% exact match',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Length difference distribution
        ax2 = axes[1]
        length_diffs = checked['length_diff'].dropna()
        ax2.hist(length_diffs, bins=50, color='#FF9800', edgecolor='white', alpha=0.8)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, label='0 difference')
        ax2.set_xlabel('Length Difference (AF2 - UniProt)')
        ax2.set_ylabel('Number of Proteins')
        ax2.set_title('Sequence Length: UniProt vs AF2 SEQRES')
        ax2.legend()

        plt.tight_layout()
        fig_path = FIG_DIR / "sequence_consistency_audit.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fig_path}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
