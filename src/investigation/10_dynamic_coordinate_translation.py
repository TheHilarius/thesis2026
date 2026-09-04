#!/usr/bin/env python3
"""
10_dynamic_coordinate_translation.py — Step 2: Dynamic Coordinate Translation

For the 28 mismatched proteins identified in Step 1:
1. Run global alignment between UniProt and AF2 SEQRES sequences
2. Build uniprot_index → af_index mapping from the alignment
3. Translate IEDB coordinates to AF2 coordinates
4. Biological Integrity Check: extracted AF2 peptide must match original
5. Partition df_all into clean/rescued/quarantined cohorts

Input:
  data/processed/df_all.csv
  data/processed/structures/alphafold/
  data/processed/sequence_audit/alignment_audit.csv

Output:
  data/processed/sequence_audit/cohort_clean.csv
  data/processed/sequence_audit/cohort_rescued.csv
  data/processed/sequence_audit/cohort_quarantined.csv
  data/processed/sequence_audit/translation_report.txt
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.Align import PairwiseAligner, substitution_matrices

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DF_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "df_all.csv"
AF_DIR = PROJECT_ROOT / "data" / "processed" / "structures" / "alphafold"
AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "sequence_audit" / "alignment_audit.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sequence_audit"

AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'UNK': 'X',
}


# ── SEQRES Parser ─────────────────────────────────────────────────────────────
def parse_seqres(pdb_path):
    """Extract full intended sequence from PDB SEQRES records."""
    try:
        residues = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('SEQRES'):
                    parts = line.split()
                    if len(parts) >= 5:
                        residues.extend(parts[4:])
        if not residues:
            return None
        return ''.join(AA3_TO_1.get(r, 'X') for r in residues)
    except Exception as e:
        print(f"  ERROR parsing SEQRES: {e}")
        return None


# ── Alignment ─────────────────────────────────────────────────────────────────
def create_aligner():
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


# ── Index Map Builder ─────────────────────────────────────────────────────────
def build_index_map(aligner, uniprot_seq, af_seq):
    """
    Build a mapping: uniprot_1based_pos → af_0based_pos (or None if gap).

    Iterates over aligned sequences simultaneously, tracking running indices.
    When a UniProt residue aligns to an AF residue, map them.
    When a UniProt residue aligns to a gap in AF, map to None.
    """
    alignments = aligner.align(uniprot_seq, af_seq)
    a = alignments[0]

    # Extract aligned sequences from alignment object
    aligned = str(a).split('\n')
    aligned_uniprot = aligned[0]  # top sequence
    aligned_af = aligned[2]       # bottom sequence

    index_map = {}  # 1-based uniprot pos → 0-based af pos (or None)
    uniprot_idx = 0  # running 0-based index into uniprot_seq
    af_idx = 0       # running 0-based index into af_seq

    for u_res, a_res in zip(aligned_uniprot, aligned_af):
        if u_res == '-' and a_res == '-':
            # Both gaps — skip (shouldn't happen in global alignment)
            continue
        elif u_res == '-':
            # Gap in UniProt, residue in AF — AF residue has no UniProt counterpart
            af_idx += 1
        elif a_res == '-':
            # Residue in UniProt, gap in AF — this UniProt position has no AF counterpart
            index_map[uniprot_idx + 1] = None  # 1-based
            uniprot_idx += 1
        else:
            # Both residues align
            index_map[uniprot_idx + 1] = af_idx  # 1-based UniProt → 0-based AF
            uniprot_idx += 1
            af_idx += 1

    return index_map, aligned_uniprot, aligned_af


# ── Coordinate Translator ─────────────────────────────────────────────────────
def translate_coordinates(start_1based, end_1based, index_map, af_seq_len):
    """
    Translate 1-based UniProt coordinates to 0-based AF2 coordinates.

    Returns (af_start_0, af_end_0, success) where:
      af_start_0 = 0-based start in AF2 sequence
      af_end_0   = 0-based end (exclusive) in AF2 sequence
      success    = True if all positions mapped to valid AF indices
    """
    # Convert 1-based inclusive to range of positions
    positions = list(range(start_1based, end_1based + 1))

    af_positions = []
    failed = False
    for pos in positions:
        af_pos = index_map.get(pos)
        if af_pos is None:
            af_positions.append(None)
            failed = True
        else:
            af_positions.append(af_pos)

    if failed:
        return None, None, False

    af_start = min(af_positions)
    af_end = max(af_positions) + 1  # exclusive end

    # Bounds check
    if af_start < 0 or af_end > af_seq_len:
        return None, None, False

    return af_start, af_end, True


# ── Biological Integrity Check ────────────────────────────────────────────────
def biological_integrity_check(af_seq, af_start, af_end, original_peptide):
    """
    Extract peptide from AF2 sequence using translated coordinates.
    Returns (extracted_string, match) where match is True if exact match.
    """
    if af_start is None or af_end is None:
        return None, False

    extracted = af_seq[af_start:af_end]
    match = (extracted == original_peptide)
    return extracted, match


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  DYNAMIC COORDINATE TRANSLATION — Step 2")
    print("=" * 65)
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    df_all = pd.read_csv(DF_ALL_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    print(f"  df_all: {len(df_all)} rows, {df_all['uniprot_id'].nunique()} proteins")
    print(f"  audit: {len(audit)} proteins")

    # ── Identify mismatched proteins ───────────────────────────────────────────
    mismatched = audit[audit['status'] == 'mismatch']['uniprot_id'].tolist()
    exact_match = audit[audit['status'] == 'exact_match']['uniprot_id'].tolist()
    n_mismatched = len(mismatched)
    print(f"\n  Exact match proteins: {len(exact_match)}")
    print(f"  Mismatched proteins: {n_mismatched}")

    # Get peptides from mismatched proteins
    df_mismatched = df_all[df_all['uniprot_id'].isin(mismatched)].copy()
    df_clean = df_all[df_all['uniprot_id'].isin(exact_match)].copy()
    print(f"  Peptides from mismatched proteins: {len(df_mismatched)}")
    print(f"  Peptides from exact-match proteins: {len(df_clean)}")

    # ── Create aligner ─────────────────────────────────────────────────────────
    print("\nCreating aligner...")
    aligner = create_aligner()

    # ── Process mismatched proteins ────────────────────────────────────────────
    print(f"\nProcessing {n_mismatched} mismatched proteins...")
    rescued_rows = []
    quarantined_rows = []

    for uid in mismatched:
        # Get AF2 SEQRES
        af_path = AF_DIR / f"{uid}.pdb"
        af_seq = parse_seqres(af_path)
        if af_seq is None:
            # Can't parse — quarantine all peptides
            prot_df = df_mismatched[df_mismatched['uniprot_id'] == uid].copy()
            prot_df['translation_status'] = 'quarantined'
            prot_df['reason'] = 'seqres_parse_error'
            quarantined_rows.append(prot_df)
            print(f"  {uid}: SEQRES parse error → quarantined {len(prot_df)} peptides")
            continue

        # Get UniProt canonical sequence
        uniprot_seq = df_mismatched[df_mismatched['uniprot_id'] == uid]['sequence'].iloc[0]

        # Build index map
        index_map, aligned_u, aligned_a = build_index_map(aligner, uniprot_seq, af_seq)

        # Get all peptides for this protein
        prot_df = df_mismatched[df_mismatched['uniprot_id'] == uid].copy()
        prot_rescued = []
        prot_quarantined = []

        for idx, row in prot_df.iterrows():
            peptide = str(row['peptide'])
            start = int(row['start'])
            end = int(row['end'])

            # Translate coordinates
            af_start, af_end, trans_ok = translate_coordinates(
                start, end, index_map, len(af_seq)
            )

            if not trans_ok:
                prot_quarantined.append({
                    'idx': idx,
                    'reason': 'translation_failed',
                    'af_start': af_start,
                    'af_end': af_end,
                })
                continue

            # Biological Integrity Check
            extracted, match = biological_integrity_check(
                af_seq, af_start, af_end, peptide
            )

            if match:
                prot_rescued.append({
                    'idx': idx,
                    'af_start': af_start,
                    'af_end': af_end,
                    'extracted': extracted,
                })
            else:
                prot_quarantined.append({
                    'idx': idx,
                    'reason': f'integrity_check_failed',
                    'af_start': af_start,
                    'af_end': af_end,
                    'extracted': extracted,
                })

        # Build rescued DataFrame
        if prot_rescued:
            rescued_df = prot_df.loc[[r['idx'] for r in prot_rescued]].copy()
            rescued_df['af_start'] = [r['af_start'] for r in prot_rescued]
            rescued_df['af_end'] = [r['af_end'] for r in prot_rescued]
            rescued_df['af_extracted'] = [r['extracted'] for r in prot_rescued]
            rescued_df['translation_status'] = 'rescued'
            rescued_rows.append(rescued_df)

        # Build quarantined DataFrame
        if prot_quarantined:
            quarantined_df = prot_df.loc[[r['idx'] for r in prot_quarantined]].copy()
            quarantined_df['af_start'] = [r['af_start'] for r in prot_quarantined]
            quarantined_df['af_end'] = [r['af_end'] for r in prot_quarantined]
            quarantined_df['reason'] = [r['reason'] for r in prot_quarantined]
            if 'extracted' in prot_quarantined[0]:
                quarantined_df['af_extracted'] = [r.get('extracted', None) for r in prot_quarantined]
            quarantined_df['translation_status'] = 'quarantined'
            quarantined_rows.append(quarantined_df)

        n_res = len(prot_rescued)
        n_qua = len(prot_quarantined)
        print(f"  {uid}: {len(prot_df)} peptides → {n_res} rescued, {n_qua} quarantined")

    # ── Combine rescued and quarantined ────────────────────────────────────────
    if rescued_rows:
        df_rescued = pd.concat(rescued_rows, ignore_index=True)
    else:
        df_rescued = pd.DataFrame()

    if quarantined_rows:
        df_quarantined = pd.concat(quarantined_rows, ignore_index=True)
    else:
        df_quarantined = pd.DataFrame()

    # ── Add status column to clean cohort ──────────────────────────────────────
    df_clean['translation_status'] = 'clean'
    df_clean['af_start'] = df_clean['start']
    df_clean['af_end'] = df_clean['end']
    df_clean['af_extracted'] = df_clean['peptide']

    # ── Save cohorts ───────────────────────────────────────────────────────────
    print(f"\nSaving cohorts...")
    df_clean.to_csv(OUT_DIR / "cohort_clean.csv", index=False)
    print(f"  cohort_clean.csv: {len(df_clean)} peptides")

    if len(df_rescued) > 0:
        df_rescued.to_csv(OUT_DIR / "cohort_rescued.csv", index=False)
        print(f"  cohort_rescued.csv: {len(df_rescued)} peptides")
    else:
        print(f"  cohort_rescued.csv: 0 peptides (empty)")

    if len(df_quarantined) > 0:
        df_quarantined.to_csv(OUT_DIR / "cohort_quarantined.csv", index=False)
        print(f"  cohort_quarantined.csv: {len(df_quarantined)} peptides")
    else:
        print(f"  cohort_quarantined.csv: 0 peptides (empty)")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_total = len(df_all)
    n_clean = len(df_clean)
    n_res = len(df_rescued)
    n_qua = len(df_quarantined)

    summary_lines = [
        "=" * 65,
        "DYNAMIC COORDINATE TRANSLATION — SUMMARY",
        "=" * 65,
        f"",
        f"Input dataset:               {n_total} peptides from {df_all['uniprot_id'].nunique()} proteins",
        f"",
        f"Protein-level results:",
        f"  Exact match (clean):       {len(exact_match)} proteins",
        f"  Mismatched (processed):    {n_mismatched} proteins",
        f"",
        f"Peptide-level results:",
        f"  Clean cohort:              {n_clean} peptides ({n_clean/n_total*100:.1f}%)",
        f"  Rescued cohort:            {n_res} peptides ({n_res/n_total*100:.1f}%)",
        f"  Quarantined cohort:        {n_qua} peptides ({n_qua/n_total*100:.1f}%)",
        f"  Total accounted:           {n_clean + n_res + n_qua} peptides",
        f"",
    ]

    # Rescued details
    if n_res > 0:
        summary_lines.append("Rescued peptide details:")
        for uid in df_rescued['uniprot_id'].unique():
            prot_res = df_rescued[df_rescued['uniprot_id'] == uid]
            summary_lines.append(
                f"  {uid}: {len(prot_res)} peptides rescued "
                f"(AF coords: {prot_res['af_start'].min()}-{prot_res['af_end'].max()})"
            )
        summary_lines.append("")

    # Quarantined details
    if n_qua > 0:
        summary_lines.append("Quarantined peptide details:")
        if 'reason' in df_quarantined.columns:
            reason_counts = df_quarantined['reason'].value_counts()
            for reason, count in reason_counts.items():
                summary_lines.append(f"  {reason}: {count} peptides")
        summary_lines.append("")
        for uid in df_quarantined['uniprot_id'].unique():
            prot_q = df_quarantined[df_quarantined['uniprot_id'] == uid]
            summary_lines.append(
                f"  {uid}: {len(prot_q)} peptides quarantined"
            )
        summary_lines.append("")

    summary_lines.extend([
        "=" * 65,
        f"Runtime: {time.time() - t0:.1f}s",
        "=" * 65,
    ])

    summary_text = '\n'.join(summary_lines)

    # Save summary
    summary_path = OUT_DIR / "translation_report.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\nSaved: {summary_path}")

    # Print summary
    print(f"\n{summary_text}")


if __name__ == '__main__':
    main()
