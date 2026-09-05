#!/usr/bin/env python3
"""
11_isoform_structure_audit.py — UniProt ↔ AlphaFold sequence consistency audit.

Validates that every downloaded AlphaFold structure contains the correct
peptide context for the epitopes mapped onto it. Runs after the fetch
pipeline and 04_evaluate_netmhcpan_sensitivity.R produce valid data.

Checks:
  1. Whole-protein sanity: UniProt length vs AF2 SEQRES length
  2. Context-window validation (PRIMARY): peptide ± flanks from UniProt vs AF2
  3. Full-sequence comparison (diagnostic, only for failures)
  4. Isoform investigation (only for failures)

Outputs:
  audit_summary.tsv       — one row per protein
  audit_peptide_details.tsv — one row per peptide
  audit_mismatches.tsv    — failures only
  audit_statistics.txt    — aggregate counts
  O43236_report.txt       — dedicated case report
"""
import os
import re
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AF_DIR = PROJECT_ROOT / "data" / "processed" / "structures" / "alphafold"
FETCH_LOG = PROJECT_ROOT / "data" / "processed" / "structures" / "logs" / "fetch_log.tsv"
POS_EL = PROJECT_ROOT / "data" / "processed" / "pos_EL_all_epitopes_hla0201.csv"
FASTA_PATH = PROJECT_ROOT / "data" / "raw" / "fasta" / "combined_full_length.fasta"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sequence_audit"
FLANK_SIZE = 8  # matches extract_flanks_safe default in functions.R

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3-letter → 1-letter amino acid mapping
AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'UNK': 'X',
}

# UniProt cache
UNIPROT_CACHE_PATH = OUT_DIR / "uniprot_cache.json"
uniprot_cache = {}

# Rate limiting
UNIPROT_DELAY = 0.35
ALPHAFOLD_DELAY = 0.5
last_api_call = 0


# ── Helper functions ──────────────────────────────────────────────────────────

def rate_limit(delay: float):
    global last_api_call
    elapsed = time.time() - last_api_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    last_api_call = time.time()


def parse_seqres(pdb_path: str) -> str:
    """Extract 1-letter AA sequence from PDB SEQRES records."""
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
        print(f"  ERROR parsing SEQRES from {pdb_path}: {e}")
        return None


def parse_fasta(fasta_path: str) -> dict[str, str]:
    """Parse multi-line FASTA into {accession: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
                # Extract accession between pipes: >sp|O43236|SEPT4... -> O43236
                if '|' in line:
                    current_id = line.split('|')[1]
                else:
                    current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id and current_seq:
        sequences[current_id] = ''.join(current_seq)

    return sequences


def fetch_uniprot_data(uniprot_id: str) -> dict | None:
    """Fetch UniProt entry data. Returns {length, sequence, canonical_isoform}."""
    if uniprot_id in uniprot_cache:
        return uniprot_cache[uniprot_id]

    rate_limit(UNIPROT_DELAY)

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        req = urllib.request.urlopen(url, timeout=15)
        data = json.loads(req.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError,
            urllib.error.HTTPError):
        uniprot_cache[uniprot_id] = None
        return None

    seq_info = data.get('sequence', {})
    result = {
        'length': seq_info.get('length'),
        'sequence': seq_info.get('value', ''),
    }
    uniprot_cache[uniprot_id] = result
    return result


def fetch_alphafold_models(uniprot_id: str) -> list[dict]:
    """Fetch all AlphaFold models for a protein."""
    rate_limit(ALPHAFOLD_DELAY)

    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        req = urllib.request.urlopen(url, timeout=15)
        return json.loads(req.read())
    except Exception:
        return []


def extract_context_window(sequence: str, start: int, end: int,
                           flank_size: int = FLANK_SIZE) -> str | None:
    """Extract n_flank + peptide + c_flank from a protein sequence.
    Mirrors R's extract_flanks_safe logic."""
    if not sequence or start < 1 or end > len(sequence) or start > end:
        return None

    n_flank_start = max(0, start - 1 - flank_size)
    n_flank_end = start - 1
    n_flank = sequence[n_flank_start:n_flank_end]
    n_flank_padded = 'X' * (flank_size - len(n_flank)) + n_flank

    c_flank_start = end
    c_flank_end = min(len(sequence), end + flank_size)
    c_flank = sequence[c_flank_start:c_flank_end]
    c_flank_padded = c_flank + 'X' * (flank_size - len(c_flank))

    return n_flank_padded + sequence[start - 1:end] + c_flank_padded


def read_fetch_log(log_path: Path) -> dict[str, dict]:
    """Parse fetch log into {uniprot_id: {source, file, coverage}}."""
    lookup = {}
    if not log_path.exists():
        return lookup

    with open(log_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                uid = parts[0]
                lookup[uid] = {
                    'source': parts[3],
                    'file': parts[4],
                    'coverage': parts[5],
                }
    return lookup


def load_cache():
    global uniprot_cache
    if UNIPROT_CACHE_PATH.exists():
        with open(UNIPROT_CACHE_PATH, 'r') as f:
            uniprot_cache = json.load(f)
        print(f"Loaded UniProt cache: {len(uniprot_cache)} entries")


def save_cache():
    with open(UNIPROT_CACHE_PATH, 'w') as f:
        json.dump(uniprot_cache, f)


# ── Main audit ────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ISOFORM STRUCTURE AUDIT")
    print("  UniProt ↔ AlphaFold sequence consistency")
    print("=" * 65)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    load_cache()

    print(f"\nLoading FASTA: {FASTA_PATH}")
    fasta_seqs = parse_fasta(str(FASTA_PATH))
    print(f"  FASTA entries: {len(fasta_seqs)}")

    print(f"\nLoading peptide data: {POS_EL}")
    df_pep = pd.read_csv(POS_EL)
    df_pep = df_pep[df_pep['pep_length'] == 9]  # 9-mers only
    print(f"  9-mer peptides: {len(df_pep)}")

    # Get proteins with AF2 structures
    af_files = [f.stem for f in AF_DIR.glob("*.pdb")]
    print(f"  AF2 structures on disk: {len(af_files)}")

    # Read fetch log
    fetch_log = read_fetch_log(FETCH_LOG)
    print(f"  Fetch log entries: {len(fetch_log)}")

    # Unique proteins to audit
    proteins_with_af = [uid for uid in af_files if uid in fasta_seqs]
    print(f"  Proteins with AF2 + FASTA: {len(proteins_with_af)}")

    # ── Layer 1: Whole-protein sanity check ───────────────────────────────────
    print(f"\n--- Layer 1: Whole-protein length check ---")
    protein_results = {}
    for i, uid in enumerate(proteins_with_af):
        if i % 500 == 0 and i > 0:
            print(f"  [{i}/{len(proteins_with_af)}]")

        fasta_seq = fasta_seqs[uid]
        fasta_length = len(fasta_seq)

        af_seq = parse_seqres(str(AF_DIR / f"{uid}.pdb"))
        af_length = len(af_seq) if af_seq else None

        length_match = (af_length == fasta_length) if af_length else None

        # Isoform detection
        has_isoform_suffix = bool(re.match(r'^[A-Z][A-Z0-9]{5,9}-\d+$', uid))

        # Fetch log info
        log_entry = fetch_log.get(uid, {})

        protein_results[uid] = {
            'uniprot_id': uid,
            'uniprot_length': fasta_length,
            'af_length': af_length,
            'length_match': length_match,
            'has_isoform_suffix': has_isoform_suffix,
            'af_sequence': af_seq,
            'uniprot_sequence': fasta_seq,
            'downloaded_model': log_entry.get('source', ''),
            'coverage': log_entry.get('coverage', ''),
        }

    n_length_match = sum(1 for r in protein_results.values() if r['length_match'] is True)
    n_length_mismatch = sum(1 for r in protein_results.values() if r['length_match'] is False)
    n_no_af = sum(1 for r in protein_results.values() if r['af_length'] is None)
    print(f"  Length matches:    {n_length_match}")
    print(f"  Length mismatches: {n_length_mismatch}")
    print(f"  No AF2 sequence:   {n_no_af}")

    # ── Layer 2: Context-window validation ────────────────────────────────────
    print(f"\n--- Layer 2: Context-window validation (PRIMARY CHECK) ---")

    # Build FASTA sequence lookup for peptides
    pep_results = []
    n_context_match = 0
    n_context_mismatch = 0
    n_context_skip = 0

    for i, (_, row) in enumerate(df_pep.iterrows()):
        if i % 5000 == 0 and i > 0:
            print(f"  [{i}/{len(df_pep)}]")

        uid = row['uniprot_id']
        start = int(row['start'])
        end = int(row['end'])
        peptide = row['peptide']

        pr = protein_results.get(uid)

        # Skip if no AF2 structure or no FASTA
        if pr is None or pr['af_sequence'] is None:
            n_context_skip += 1
            pep_results.append({
                'peptide': peptide,
                'uniprot_id': uid,
                'start': start,
                'end': end,
                'expected_window': None,
                'af_window': None,
                'context_match': None,
                'peptide_found_in_af': None,
                'failure_reason': 'no_structure' if uid not in af_files else 'no_fasta',
            })
            continue

        expected_window = extract_context_window(pr['uniprot_sequence'], start, end)
        af_window = extract_context_window(pr['af_sequence'], start, end)

        if expected_window is None or af_window is None:
            n_context_skip += 1
            pep_results.append({
                'peptide': peptide,
                'uniprot_id': uid,
                'start': start,
                'end': end,
                'expected_window': expected_window,
                'af_window': af_window,
                'context_match': None,
                'peptide_found_in_af': None,
                'failure_reason': 'window_out_of_range',
            })
            continue

        context_match = (expected_window == af_window)
        peptide_found = peptide in pr['af_sequence'] if pr['af_sequence'] else None

        if context_match:
            n_context_match += 1
        else:
            n_context_mismatch += 1

        pep_results.append({
            'peptide': peptide,
            'uniprot_id': uid,
            'start': start,
            'end': end,
            'expected_window': expected_window,
            'af_window': af_window,
            'context_match': context_match,
            'peptide_found_in_af': peptide_found,
            'failure_reason': None if context_match else 'context_mismatch',
        })

    print(f"  Context matches:    {n_context_match}")
    print(f"  Context mismatches: {n_context_mismatch}")
    print(f"  Skipped:            {n_context_skip}")

    # ── Layer 3: Full-sequence comparison (diagnostic) ────────────────────────
    print(f"\n--- Layer 3: Full-sequence comparison (diagnostic) ---")
    mismatch_proteins = [uid for uid, r in protein_results.items()
                         if r['length_match'] is False or
                         any(p['context_match'] is False
                             for p in pep_results if p['uniprot_id'] == uid)]

    print(f"  Proteins needing full comparison: {len(mismatch_proteins)}")

    # ── Layer 4: Isoform investigation ────────────────────────────────────────
    print(f"\n--- Layer 4: Isoform investigation ---")
    isoform_investigations = []
    for uid in mismatch_proteins:
        uniprot_data = fetch_uniprot_data(uid)
        if uniprot_data is None:
            continue

        models = fetch_alphafold_models(uid)
        expected_length = uniprot_data['length']
        af_length = protein_results[uid]['af_length']

        # Find matching model
        matching_model = None
        for model in models:
            model_length = model['uniprotEnd'] - model['uniprotStart'] + 1
            if model_length == expected_length:
                matching_model = model['entryId']
                break

        isoform_investigations.append({
            'uniprot_id': uid,
            'uniprot_length': expected_length,
            'downloaded_length': af_length,
            'n_models_available': len(models),
            'models': [m['entryId'] for m in models],
            'matching_model': matching_model,
            'alternative_available': matching_model is not None,
        })

    n_alt_available = sum(1 for inv in isoform_investigations
                          if inv['alternative_available'])
    print(f"  Mismatches investigated: {len(isoform_investigations)}")
    print(f"  Alternative model available: {n_alt_available}")

    # ── Generate outputs ──────────────────────────────────────────────────────
    print(f"\n--- Writing outputs ---")

    # audit_summary.tsv
    summary_rows = []
    for uid, pr in protein_results.items():
        n_pep = sum(1 for p in pep_results if p['uniprot_id'] == uid)
        n_ctx_match = sum(1 for p in pep_results
                          if p['uniprot_id'] == uid and p['context_match'] is True)
        n_ctx_mismatch = sum(1 for p in pep_results
                             if p['uniprot_id'] == uid and p['context_match'] is False)
        summary_rows.append({
            'uniprot_id': uid,
            'uniprot_length': pr['uniprot_length'],
            'af_length': pr['af_length'],
            'length_match': pr['length_match'],
            'has_isoform_suffix': pr['has_isoform_suffix'],
            'downloaded_model': pr['downloaded_model'],
            'coverage': pr['coverage'],
            'n_peptides': n_pep,
            'n_context_matches': n_ctx_match,
            'n_context_mismatches': n_ctx_mismatch,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "audit_summary.tsv", sep='\t', index=False)
    print(f"  audit_summary.tsv: {len(df_summary)} proteins")

    # audit_peptide_details.tsv
    df_pep_detail = pd.DataFrame(pep_results)
    df_pep_detail.to_csv(OUT_DIR / "audit_peptide_details.tsv", sep='\t', index=False)
    print(f"  audit_peptide_details.tsv: {len(df_pep_detail)} peptides")

    # audit_mismatches.tsv
    mismatch_rows = [r for r in pep_results if r['context_match'] is False]
    df_mismatches = pd.DataFrame(mismatch_rows)
    df_mismatches.to_csv(OUT_DIR / "audit_mismatches.tsv", sep='\t', index=False)
    print(f"  audit_mismatches.tsv: {len(df_mismatches)} context mismatches")

    # audit_statistics.txt
    stats_lines = [
        "=" * 50,
        "  AUDIT STATISTICS",
        "=" * 50,
        "",
        f"Total proteins audited:             {len(protein_results)}",
        f"Whole-protein length matches:       {n_length_match}",
        f"Whole-protein length mismatches:    {n_length_mismatch}",
        f"No AF2 sequence:                    {n_no_af}",
        "",
        f"Total peptides audited:             {len(df_pep_detail)}",
        f"Context-window matches:             {n_context_match}",
        f"Context-window mismatches:          {n_context_mismatch}",
        f"Skipped (no structure/FASTA):       {n_context_skip}",
        "",
        f"Isoform suffix proteins:            {sum(1 for r in protein_results.values() if r['has_isoform_suffix'])}",
        f"Mismatches investigated:            {len(isoform_investigations)}",
        f"Alternative model available:        {n_alt_available}",
        "",
    ]
    with open(OUT_DIR / "audit_statistics.txt", 'w') as f:
        f.write('\n'.join(stats_lines))
    print(f"  audit_statistics.txt")

    # O43236_report.txt
    write_o43236_report(protein_results, isoform_investigations, pep_results)

    # Save cache
    save_cache()

    elapsed = time.time() - t0
    print(f"\nAudit complete in {elapsed / 60:.1f} minutes")


def write_o43236_report(protein_results: dict, isoform_investigations: list,
                        pep_results: list):
    """Always generate a dedicated O43236 report."""
    uid = "O43236"
    pr = protein_results.get(uid)
    inv = next((i for i in isoform_investigations if i['uniprot_id'] == uid), None)

    lines = [
        "=" * 50,
        "  O43236 STRUCTURE AUDIT",
        "=" * 50,
        "",
    ]

    if pr:
        lines.extend([
            "UniProt Entry:",
            f"  Accession:        {uid}",
            f"  Canonical Length: {pr['uniprot_length']} aa",
            f"  Sequence (first 30): {pr['uniprot_sequence'][:30]}",
            "",
            "Downloaded AlphaFold Model:",
            f"  File:             {AF_DIR / f'{uid}.pdb'}",
            f"  Source:           {pr['downloaded_model']}",
            f"  SEQRES Length:    {pr['af_length']} aa",
            f"  Sequence (first 30): {pr['af_sequence'][:30] if pr['af_sequence'] else 'N/A'}",
            "",
            "Sequence Comparison:",
            f"  Length Match:     {pr['length_match']}",
            f"  Length Difference: {pr['af_length'] - pr['uniprot_length'] if pr['af_length'] else 'N/A'} aa",
            "",
        ])
    else:
        lines.append(f"  {uid} not found in audit results")
        lines.append("")

    if inv:
        lines.extend([
            "Available AlphaFold Models:",
        ])
        for model_id in inv['models']:
            tag = " (DOWNLOADED)" if model_id == pr.get('downloaded_model', '') else ""
            tag = tag or " (MATCHES FASTA)" if model_id == inv['matching_model'] else tag
            lines.append(f"  {model_id}{tag}")
        lines.extend([
            "",
            f"Correct model found:  {inv['matching_model'] or 'NONE'}",
            f"Alternative available: {inv['alternative_available']}",
            "",
        ])

    # Peptide context for O43236
    o43236_peps = [p for p in pep_results if p['uniprot_id'] == uid]
    if o43236_peps:
        lines.append("Peptide Context:")
        for p in o43236_peps:
            status = "MATCH" if p['context_match'] else "MISMATCH"
            lines.append(f"  {p['peptide']} ({p['start']}-{p['end']}): {status}")
            if p['expected_window'] and p['af_window']:
                lines.append(f"    Expected: {p['expected_window']}")
                lines.append(f"    AF2:      {p['af_window']}")
        lines.append("")

    lines.extend([
        "NOTE: Coverage check passes (peptide within AF2 coordinate range)",
        "but the AF2 model may correspond to a different isoform/sequence.",
        "This is why context-window validation is the primary check.",
    ])

    report_path = OUT_DIR / "O43236_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  O43236_report.txt")


if __name__ == "__main__":
    main()
