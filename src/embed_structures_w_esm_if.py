#!/usr/bin/env python3
"""
embed_structures_w_esm_if.py — Hybrid PDB + AlphaFold compatible

Three fixes over the original:
  1. Auto-detects actual file format (CIF vs PDB) regardless of extension
  2. Multi-chain resolution with sequence validation
  3. Finds peptide position in structure sequence instead of trusting
     UniProt indices (fixes experimental PDB numbering offsets)
  4. Handles variable-length flanks (0–4 aa) and missing flanks

Usage:
    python src/embed_structures_w_esm_if.py \
        data/processed/df_all.csv \
        data/processed/structures_hybrid/selected/ \
        data/processed/embeddings/esmif_hybrid_structure_embeddings.h5
"""

import argparse
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob

import esm
import esm.inverse_folding.util as esm_util

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('csv_path', help="Path to your final ML dataset CSV")
parser.add_argument('pdb_dir',  help="Directory containing .pdb or .cif files")
parser.add_argument('out_h5',   help="Path to save the output HDF5 embeddings")
args = parser.parse_args()

DEVICE = "cpu"
Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"BioPython: {'available' if HAS_BIOPYTHON else 'NOT available (fallback mode)'}")

# ── Load ESM-IF1 ──────────────────────────────────────────────────────────────
print("Loading ESM-IF1 (142M parameters)...")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to(DEVICE)
print("ESM-IF1 loaded successfully.")

EMB_DIM = 512

# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(args.csv_path)
before = len(df)
df = df.dropna(subset=['peptide', 'uniprot_id', 'start', 'end'])
print(f"Total rows to process: {len(df)} (dropped {before - len(df)} NaNs)")
df = df.reset_index(drop=True)

# Pre-group: all unique peptides per protein (for chain selection scoring)
uid_peptides = df.groupby('uniprot_id')['peptide'].apply(set).to_dict()

# ── FIX 1: File format detection ─────────────────────────────────────────────

def detect_file_format(filepath):
    """
    Detect actual format by reading the first non-blank line.
    Needed because hybrid fetcher names AlphaFold CIF files as .pdb.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(('data_', 'loop_', '#')):
                    return 'cif'
                if stripped.startswith(('ATOM', 'HETATM', 'HEADER', 'REMARK',
                                       'MODEL', 'CRYST', 'TITLE', 'COMPND',
                                       'SOURCE', 'KEYWDS', 'EXPDTA', 'DBREF')):
                    return 'pdb'
                break
    except Exception:
        pass
    return 'cif' if filepath.endswith('.cif') else 'pdb'


def load_coords_safe(struct_path, chain):
    """
    Wrapper around esm_util.load_coords that handles format/extension
    mismatches by creating a temporary symlink with the correct extension.
    """
    actual_fmt = detect_file_format(struct_path)
    ext_fmt = 'cif' if struct_path.endswith('.cif') else 'pdb'

    if actual_fmt == ext_fmt:
        return esm_util.load_coords(struct_path, chain=chain)

    # Mismatch: create temp symlink with correct extension
    tmp_path = struct_path + f".tmp_esm.{actual_fmt}"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.symlink(os.path.abspath(struct_path), tmp_path)
        return esm_util.load_coords(tmp_path, chain=chain)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Helper: find structure file ───────────────────────────────────────────────

def find_structure_file(uniprot_id, search_dir):
    """Find a PDB/CIF file matching the UniProt ID."""
    pattern = os.path.join(search_dir, f"*{uniprot_id}*.*")
    matches = glob.glob(pattern)
    matches = [m for m in matches if m.endswith('.pdb') or m.endswith('.cif')]
    return matches[0] if matches else None


# ── FIX 2: Chain resolution ──────────────────────────────────────────────────

def get_chain_ids(struct_path):
    """Get all chain IDs from a structure file."""
    if HAS_BIOPYTHON:
        try:
            fmt = detect_file_format(struct_path)
            if fmt == 'cif':
                p = MMCIFParser(QUIET=True)
            else:
                p = PDBParser(QUIET=True)
            structure = p.get_structure("tmp", struct_path)
            return [c.id for c in structure.get_chains()]
        except Exception:
            pass
    return ["A", "B", "C", "D", "E", "L", "H", "1", "2"]


def load_structure_best_chain(struct_path, uid, peptide_set):
    """
    Load the chain that contains the most peptides from this protein.
    If NO chain contains any peptides, reject the structure entirely.
    """
    # --- Fast path: chain A ---
    try:
        coords, seq = load_coords_safe(struct_path, chain="A")
        rep = esm_util.get_encoder_output(model, alphabet, coords).cpu().numpy()
        hits = sum(1 for p in peptide_set if p in seq)
        if hits > 0:
            return rep, seq, "A"
        chain_a_result = (rep, seq)
    except Exception:
        chain_a_result = None

    # --- Score all other chains ---
    chain_ids = get_chain_ids(struct_path)
    best_validated = None

    for chain_id in chain_ids:
        if chain_id == "A":
            continue
        try:
            coords, seq = load_coords_safe(struct_path, chain=chain_id)
            rep = esm_util.get_encoder_output(model, alphabet, coords).cpu().numpy()
            hits = sum(1 for p in peptide_set if p in seq)

            if hits > 0:
                if best_validated is None or hits > best_validated[3]:
                    best_validated = (rep, seq, chain_id, hits)
        except Exception:
            continue

    if best_validated is not None:
        return best_validated[0], best_validated[1], best_validated[2]

    # No chain contains any peptides — reject entirely.
    # Could be: MHC complex (wrong protein), or structure doesn't
    # cover the peptide region. Either way, zero vector is honest.
    return None, None, None


# ── FIX 3: Peptide locator with full-window search + duplicate handling ───────

def _get_flank(value):
    """
    Safely extract flank string. Handles:
      - NaN / None         -> ''
      - float (from NaN)   -> ''
      - 'nan' string       -> ''
      - normal string      -> as-is
    """
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if s.lower() == 'nan' or s == '':
        return ''
    return s


def _find_all(sequence, query):
    """Find ALL start positions of query in sequence."""
    positions = []
    start = 0
    while True:
        idx = sequence.find(query, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _pick_closest(positions, target_idx):
    """Pick the position closest to the UniProt-derived target index."""
    return min(positions, key=lambda p: abs(p - target_idx))


def find_peptide_in_structure(pdb_seq, rep_len, row):
    """
    Locate the peptide + flanks in the structure sequence.

    Strategy:
      1. Search for full window (n_flank + peptide + c_flank) — most specific
      2. Fall back to peptide-only if full window not found
      3. If multiple matches at any stage, use UniProt start to disambiguate

    Handles variable-length flanks:
      - Full 4aa flanks on both sides (most common)
      - Short flanks (1-3aa, e.g. near protein termini)
      - Missing flanks (0aa, e.g. peptide starts at residue 1)

    Returns (pep_start, pep_end, n_start, c_end, match_type) or None.
    """
    peptide = str(row['peptide'])
    n_flank = _get_flank(row.get('n_flank'))
    c_flank = _get_flank(row.get('c_flank'))
    uniprot_start_0 = int(row['start']) - 1  # 0-indexed hint for disambiguation

    n_len = len(n_flank)
    c_len = len(c_flank)

    # ── Strategy 1: Full window search (most specific) ────────────────────
    # Only attempt if at least one flank exists (otherwise identical to pep-only)
    full_window = n_flank + peptide + c_flank
    if len(full_window) > len(peptide):
        positions = _find_all(pdb_seq, full_window)

        if len(positions) == 1:
            win_start = positions[0]
            pep_start = win_start + n_len
            pep_end   = pep_start + len(peptide)
            n_start   = win_start
            c_end     = min(rep_len, win_start + len(full_window))
            return pep_start, pep_end, n_start, c_end, "full_window_unique"

        elif len(positions) > 1:
            expected_win_start = uniprot_start_0 - n_len
            win_start = _pick_closest(positions, max(0, expected_win_start))
            pep_start = win_start + n_len
            pep_end   = pep_start + len(peptide)
            n_start   = win_start
            c_end     = min(rep_len, win_start + len(full_window))
            return pep_start, pep_end, n_start, c_end, "full_window_positional"

    # ── Strategy 2: Peptide-only search (fallback) ────────────────────────
    positions = _find_all(pdb_seq, peptide)

    if len(positions) == 0:
        return None  # peptide not in structure → zero vector

    if len(positions) == 1:
        pep_start = positions[0]
        match_type = "peptide_unique"
    else:
        pep_start = _pick_closest(positions, uniprot_start_0)
        match_type = "peptide_positional"

    pep_end = pep_start + len(peptide)

    # Compute flank boundaries from the found peptide position.
    # Flanks may be shorter than requested if peptide is near structure edge.
    n_start = max(0, pep_start - n_len)
    c_end   = min(rep_len, pep_end + c_len)

    return pep_start, pep_end, n_start, c_end, match_type


# ── Embed ─────────────────────────────────────────────────────────────────────
print(f"\nExtracting 3D embeddings for {len(df)} sequences...")

peptide_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
n_flank_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
c_flank_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)

structure_cache = {}       # uid -> (rep, seq)
missing_structures = set()
peptide_not_found = 0      # rows where peptide isn't in structure seq
match_stats = {}           # tracks how each row was resolved

chain_stats = {"A": 0, "other_validated": 0, "rejected": 0, "failed": 0}

with torch.no_grad():
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  [{i}/{len(df)}]  "
                  f"(cached: {len(structure_cache)}, "
                  f"missing: {len(missing_structures)}, "
                  f"pep_not_found: {peptide_not_found})")

        uid = row['uniprot_id']

        if uid in missing_structures:
            continue

        # ── Load and cache structure ──────────────────────────────────────
        if uid not in structure_cache:
            struct_path = find_structure_file(uid, args.pdb_dir)
            if not struct_path:
                missing_structures.add(uid)
                chain_stats["failed"] += 1
                continue

            rep, seq, chain_used = load_structure_best_chain(
                struct_path, uid, uid_peptides.get(uid, set())
            )

            if rep is None:
                # Log WHY it failed: no file, or structure loaded but no peptides?
                if struct_path:
                    print(f"  REJECTED {uid}: structure loaded but no peptides "
                          f"found in any chain (MHC complex or uncovered region)")
                    chain_stats["rejected"] += 1
                else:
                    chain_stats["failed"] += 1
                missing_structures.add(uid)
                continue

            structure_cache[uid] = (rep, seq)

            if chain_used == "A":
                chain_stats["A"] += 1
            else:
                chain_stats["other_validated"] += 1
                if chain_stats["other_validated"] <= 50 or \
                   chain_stats["other_validated"] % 200 == 0:
                    print(f"  {uid}: validated chain '{chain_used}' "
                          f"(chain A unavailable/wrong protein)")

        rep, pdb_seq = structure_cache[uid]

        # ── Locate peptide in structure sequence ──────────────────────────
        result = find_peptide_in_structure(pdb_seq, len(rep), row)

        if result is None:
            peptide_not_found += 1
            continue

        pep_start, pep_end, n_start, c_end, match_type = result
        match_stats[match_type] = match_stats.get(match_type, 0) + 1

        # Slice and average
        n_slice   = rep[n_start:pep_start]
        pep_slice = rep[pep_start:pep_end]
        c_slice   = rep[pep_end:c_end]

        n_emb   = n_slice.mean(axis=0) if len(n_slice) > 0 else np.zeros(EMB_DIM)
        pep_emb = pep_slice.mean(axis=0) if len(pep_slice) > 0 else np.zeros(EMB_DIM)
        c_emb   = c_slice.mean(axis=0) if len(c_slice) > 0 else np.zeros(EMB_DIM)

        peptide_embs[i] = pep_emb
        n_flank_embs[i] = n_emb
        c_flank_embs[i] = c_emb

# ── Summary ───────────────────────────────────────────────────────────────────
total_loaded = len(structure_cache)
total_proteins = df['uniprot_id'].nunique()

# Count rows belonging to missing proteins
missing_rows = (df['uniprot_id'].isin(missing_structures)).sum()
embedded_rows = len(df) - missing_rows - peptide_not_found

print(f"\n{'='*60}")
print(f"STRUCTURE LOADING ({total_loaded} / {total_proteins} proteins):")
print(f"  Chain A (validated):         {chain_stats['A']}")
print(f"  Other chain (validated):     {chain_stats['other_validated']}")
print(f"  Rejected (no peptides):      {chain_stats['rejected']}")
print(f"  Failed (no file):            {chain_stats['failed']}")
print(f"")
print(f"PEPTIDE ALIGNMENT ({len(df)} rows):")
print(f"  Embedded successfully:       {embedded_rows}")
print(f"  Peptide NOT in structure:    {peptide_not_found}  (-> zero vector)")
print(f"  Protein missing entirely:    {len(missing_structures)} proteins ({missing_rows} rows)")
print(f"")
print(f"MATCH RESOLUTION BREAKDOWN:")
for mtype in ['full_window_unique', 'full_window_positional',
              'peptide_unique', 'peptide_positional']:
    count = match_stats.get(mtype, 0)
    pct = (count / max(embedded_rows, 1)) * 100
    print(f"  {mtype:30s} {count:>7d}  ({pct:5.1f}%)")
# Any unexpected match types
for mtype, count in match_stats.items():
    if mtype not in ['full_window_unique', 'full_window_positional',
                     'peptide_unique', 'peptide_positional']:
        print(f"  {mtype:30s} {count:>7d}")
print(f"{'='*60}")

# ── Zero-vector summary ──────────────────────────────────────────────────────
zero_pep = np.sum(np.linalg.norm(peptide_embs, axis=1) == 0.0)
zero_n   = np.sum(np.linalg.norm(n_flank_embs, axis=1) == 0.0)
zero_c   = np.sum(np.linalg.norm(c_flank_embs, axis=1) == 0.0)
print(f"\nZero vectors: peptide={zero_pep}, n_flank={zero_n}, c_flank={zero_c}")
print(f"  (out of {len(df)} rows = {zero_pep/len(df)*100:.1f}% / "
      f"{zero_n/len(df)*100:.1f}% / {zero_c/len(df)*100:.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {args.out_h5}...")
with h5py.File(args.out_h5, 'w') as f:
    f.create_dataset('peptide_if_struct', data=peptide_embs,  dtype='float32')
    f.create_dataset('n_flank_if_struct', data=n_flank_embs,  dtype='float32')
    f.create_dataset('c_flank_if_struct', data=c_flank_embs,  dtype='float32')
    f.create_dataset('peptide_ids',
                     data=np.array(df['peptide'].tolist(), dtype='S20'))
    f.create_dataset('uniprot_ids',
                     data=np.array(df['uniprot_id'].tolist(), dtype='S20'))

    f.attrs['model']     = 'esm_if1_gvp4_t16_142M_UR50'
    f.attrs['emb_dim']   = EMB_DIM
    f.attrs['n_samples'] = len(df)

print("Done.")
