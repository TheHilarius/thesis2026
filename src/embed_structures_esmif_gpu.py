#!/usr/bin/env python3
"""
embed_structures_esmif_gpu.py — Hybrid PDB + AlphaFold with per-peptide AF2 fallback

Fixes over the original:
  1. Auto-detects actual file format (CIF vs PDB) regardless of extension
  2. Multi-chain resolution with sequence validation
  3. Finds peptide position in structure sequence instead of trusting
     UniProt indices (fixes experimental PDB numbering offsets)
  4. Handles variable-length flanks (0–4 aa) and missing flanks
  5. LRU cache prevents OOM on large datasets
  6. Sorts by protein for cache locality, writes to original row order
  7. GPU support with device-aware encoder output
  8. Per-peptide AF2 fallback: if peptide not found in primary structure,
     tries AlphaFold structure before giving up → fewer zero vectors

Usage:
    python src/embed_structures_esmif_gpu.py \
        data/processed/df_all.csv \
        data/processed/structures_hybrid_80/selected/ \
        data/processed/embeddings/esmif_hybrid_80_embeddings.h5 \
        --af2-fallback-dir data/processed/structures/alphafold/
"""

import argparse
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
from collections import OrderedDict

import esm
import esm.inverse_folding.util as esm_util

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ── FIX 7: GPU-aware encoder output ──────────────────────────────────────────

try:
    from esm.inverse_folding.util import CoordBatchConverter
    HAS_COORD_CONVERTER = True
except ImportError:
    HAS_COORD_CONVERTER = False


def get_encoder_output_gpu(model, alphabet, coords, device):
    """
    GPU-compatible replacement for esm_util.get_encoder_output().
    Adds the missing .to(device) calls on CoordBatchConverter outputs.
    """
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]
    coords_t, confidence, strs, tokens, padding_mask = batch_converter(batch)

    coords_t = coords_t.to(device)
    confidence = confidence.to(device)
    padding_mask = padding_mask.to(device)
    tokens = tokens.to(device)

    encoder_out = model.encoder(coords_t, padding_mask, confidence,
                                return_all_hiddens=False)

    rep = encoder_out['encoder_out'][0].transpose(0, 1)
    rep = rep[0]

    return rep.cpu().numpy()


# ── LRU Cache ─────────────────────────────────────────────────────────────────

class LRUCache(OrderedDict):
    """Simple LRU cache that evicts oldest entries when full."""
    def __init__(self, maxsize=500):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('csv_path', help="Path to your final ML dataset CSV")
parser.add_argument('pdb_dir',  help="Directory containing primary .pdb or .cif files")
parser.add_argument('out_h5',   help="Path to save the output HDF5 embeddings")
parser.add_argument('--af2-fallback-dir', default=None,
                    help="Directory with AlphaFold structures for per-peptide fallback")
parser.add_argument('--cache-size', type=int, default=500,
                    help="Max proteins to keep in memory (default: 500)")
parser.add_argument('--force-cpu', action='store_true',
                    help="Force CPU even if CUDA is available")
args = parser.parse_args()

# ── Device selection ──────────────────────────────────────────────────────────
if args.force_cpu:
    DEVICE = "cpu"
    USE_GPU = False
    print("Forced CPU mode via --force-cpu flag.")
elif torch.cuda.is_available() and HAS_COORD_CONVERTER:
    DEVICE = "cuda"
    USE_GPU = True
    print(f"GPU mode enabled: {torch.cuda.get_device_name(0)}")
elif torch.cuda.is_available() and not HAS_COORD_CONVERTER:
    DEVICE = "cpu"
    USE_GPU = False
    print(f"WARNING: GPU available ({torch.cuda.get_device_name(0)}) but "
          f"CoordBatchConverter import failed. Falling back to CPU.")
else:
    DEVICE = "cpu"
    USE_GPU = False
    print("No GPU available, using CPU.")

AF2_FALLBACK = args.af2_fallback_dir is not None
if AF2_FALLBACK:
    if not os.path.isdir(args.af2_fallback_dir):
        print(f"ERROR: AF2 fallback dir not found: {args.af2_fallback_dir}")
        AF2_FALLBACK = False
    else:
        af2_file_count = len(glob.glob(os.path.join(args.af2_fallback_dir, "*.pdb"))) + \
                         len(glob.glob(os.path.join(args.af2_fallback_dir, "*.cif")))
        print(f"AF2 fallback  : ENABLED ({af2_file_count} files in {args.af2_fallback_dir})")

Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)

print(f"Device        : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"BioPython     : {'available' if HAS_BIOPYTHON else 'NOT available (fallback mode)'}")
print(f"Cache size    : {args.cache_size} proteins")
print(f"AF2 fallback  : {'ENABLED' if AF2_FALLBACK else 'DISABLED (use --af2-fallback-dir)'}")

# ── Load ESM-IF1 ──────────────────────────────────────────────────────────────
print("Loading ESM-IF1 (142M parameters)...")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to(DEVICE)
print(f"ESM-IF1 loaded successfully on {DEVICE}.")

EMB_DIM = 512


# ── Encoder wrapper (GPU or CPU) ─────────────────────────────────────────────
def encode_structure(coords):
    if USE_GPU:
        return get_encoder_output_gpu(model, alphabet, coords, DEVICE)
    else:
        return esm_util.get_encoder_output(model, alphabet, coords)


GPU_VALIDATED = False


def encode_structure_safe(coords):
    global USE_GPU, DEVICE, GPU_VALIDATED

    if GPU_VALIDATED or not USE_GPU:
        return encode_structure(coords)

    try:
        rep = get_encoder_output_gpu(model, alphabet, coords, DEVICE)
        GPU_VALIDATED = True
        print(f"  ✓ GPU inference validated! Output shape: {rep.shape}")
        return rep
    except Exception as e:
        print(f"\n  ✗ GPU inference FAILED: {e}")
        print(f"    Falling back to CPU for entire run.")
        USE_GPU = False
        model.to("cpu")
        DEVICE = "cpu"
        return esm_util.get_encoder_output(model, alphabet, coords)


# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(args.csv_path)
before = len(df)
df = df.dropna(subset=['peptide', 'uniprot_id', 'start', 'end'])
print(f"Total rows to process: {len(df)} (dropped {before - len(df)} NaNs)")

df['_orig_idx'] = np.arange(len(df))
df = df.sort_values('uniprot_id').reset_index(drop=True)
print(f"Sorted by uniprot_id for cache locality.")

uid_peptides = df.groupby('uniprot_id')['peptide'].apply(set).to_dict()

# ── FIX 1: File format detection ─────────────────────────────────────────────

def detect_file_format(filepath):
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
    actual_fmt = detect_file_format(struct_path)
    ext_fmt = 'cif' if struct_path.endswith('.cif') else 'pdb'

    if actual_fmt == ext_fmt:
        return esm_util.load_coords(struct_path, chain=chain)

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
    pattern = os.path.join(search_dir, f"*{uniprot_id}*.*")
    matches = glob.glob(pattern)
    matches = [m for m in matches if m.endswith('.pdb') or m.endswith('.cif')]
    return matches[0] if matches else None


# ── FIX 2: Chain resolution ──────────────────────────────────────────────────

def get_chain_ids(struct_path):
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
    try:
        coords, seq = load_coords_safe(struct_path, chain="A")
        rep = encode_structure_safe(coords)
        hits = sum(1 for p in peptide_set if p in seq)
        if hits > 0:
            return rep, seq, "A"
        chain_a_result = (rep, seq)
    except Exception:
        chain_a_result = None

    chain_ids = get_chain_ids(struct_path)
    best_validated = None

    for chain_id in chain_ids:
        if chain_id == "A":
            continue
        try:
            coords, seq = load_coords_safe(struct_path, chain=chain_id)
            rep = encode_structure_safe(coords)
            hits = sum(1 for p in peptide_set if p in seq)

            if hits > 0:
                if best_validated is None or hits > best_validated[3]:
                    best_validated = (rep, seq, chain_id, hits)
        except Exception:
            continue

    if best_validated is not None:
        return best_validated[0], best_validated[1], best_validated[2]

    return None, None, None


# ── FIX 8: AF2 fallback loader ────────────────────────────────────────────────

def load_af2_fallback(uid, af2_dir):
    """
    Load AlphaFold structure as fallback. Always chain A, no validation needed
    since AF2 structures always contain the full protein.

    Returns (rep, seq) or (None, None).
    """
    af2_path = find_structure_file(uid, af2_dir)
    if af2_path is None:
        return None, None

    try:
        coords, seq = load_coords_safe(af2_path, chain="A")
        rep = encode_structure_safe(coords)
        return rep, seq
    except Exception:
        return None, None


# ── FIX 3: Peptide locator ───────────────────────────────────────────────────

def _get_flank(value):
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if s.lower() == 'nan' or s == '':
        return ''
    return s


def _find_all(sequence, query):
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
    return min(positions, key=lambda p: abs(p - target_idx))


def find_peptide_in_structure(pdb_seq, rep_len, row):
    peptide = str(row['peptide'])
    n_flank = _get_flank(row.get('n_flank'))
    c_flank = _get_flank(row.get('c_flank'))
    uniprot_start_0 = int(row['start']) - 1

    n_len = len(n_flank)
    c_len = len(c_flank)

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

    positions = _find_all(pdb_seq, peptide)

    if len(positions) == 0:
        return None

    if len(positions) == 1:
        pep_start = positions[0]
        match_type = "peptide_unique"
    else:
        pep_start = _pick_closest(positions, uniprot_start_0)
        match_type = "peptide_positional"

    pep_end = pep_start + len(peptide)
    n_start = max(0, pep_start - n_len)
    c_end   = min(rep_len, pep_end + c_len)

    return pep_start, pep_end, n_start, c_end, match_type


# ── Embed ─────────────────────────────────────────────────────────────────────
n_rows = len(df)
print(f"\nExtracting 3D embeddings for {n_rows} sequences...")
print(f"Unique proteins: {df['uniprot_id'].nunique()}")

peptide_embs = np.zeros((n_rows, EMB_DIM), dtype=np.float32)
n_flank_embs = np.zeros((n_rows, EMB_DIM), dtype=np.float32)
c_flank_embs = np.zeros((n_rows, EMB_DIM), dtype=np.float32)

orig_peptides = [''] * n_rows
orig_uniprots = [''] * n_rows

# Separate caches for primary and AF2 fallback structures
structure_cache = LRUCache(maxsize=args.cache_size)
af2_cache = LRUCache(maxsize=args.cache_size) if AF2_FALLBACK else None

missing_structures = set()
peptide_not_found = 0
match_stats = {}
cache_reloads = 0

# Track source of each embedding
source_stats = {"primary": 0, "af2_fallback": 0}
chain_stats = {"A": 0, "other_validated": 0, "rejected": 0, "failed": 0}

with torch.no_grad():
    for i, row in df.iterrows():
        orig_i = int(row['_orig_idx'])

        orig_peptides[orig_i] = str(row['peptide'])
        orig_uniprots[orig_i] = str(row['uniprot_id'])

        if i % 500 == 0:
            fb_str = f", af2_fallback: {source_stats['af2_fallback']}" if AF2_FALLBACK else ""
            print(f"  [{i}/{n_rows}]  "
                  f"(cached: {len(structure_cache)}, "
                  f"missing: {len(missing_structures)}, "
                  f"pep_not_found: {peptide_not_found}{fb_str}, "
                  f"reloads: {cache_reloads})")

        uid = row['uniprot_id']

        if uid in missing_structures:
            # Even if primary is missing, try AF2 fallback
            if AF2_FALLBACK:
                af2_cached = af2_cache.get(uid)
                if af2_cached is None:
                    af2_rep, af2_seq = load_af2_fallback(uid, args.af2_fallback_dir)
                    if af2_rep is not None:
                        af2_cache.put(uid, (af2_rep, af2_seq))
                        af2_cached = (af2_rep, af2_seq)

                if af2_cached is not None:
                    af2_rep, af2_seq = af2_cached
                    result = find_peptide_in_structure(af2_seq, len(af2_rep), row)
                    if result is not None:
                        pep_start, pep_end, n_start, c_end, match_type = result
                        match_stats[match_type] = match_stats.get(match_type, 0) + 1
                        source_stats["af2_fallback"] += 1

                        n_slice   = af2_rep[n_start:pep_start]
                        pep_slice = af2_rep[pep_start:pep_end]
                        c_slice   = af2_rep[pep_end:c_end]

                        peptide_embs[orig_i] = pep_slice.mean(axis=0) if len(pep_slice) > 0 else np.zeros(EMB_DIM)
                        n_flank_embs[orig_i] = n_slice.mean(axis=0) if len(n_slice) > 0 else np.zeros(EMB_DIM)
                        c_flank_embs[orig_i] = c_slice.mean(axis=0) if len(c_slice) > 0 else np.zeros(EMB_DIM)
                        continue

            continue

        # ── Load and cache primary structure ──────────────────────────────
        cached = structure_cache.get(uid)
        if cached is None:
            struct_path = find_structure_file(uid, args.pdb_dir)
            if not struct_path:
                missing_structures.add(uid)
                chain_stats["failed"] += 1

                # Immediately try AF2 fallback for this first row
                if AF2_FALLBACK:
                    af2_rep, af2_seq = load_af2_fallback(uid, args.af2_fallback_dir)
                    if af2_rep is not None:
                        af2_cache.put(uid, (af2_rep, af2_seq))
                        result = find_peptide_in_structure(af2_seq, len(af2_rep), row)
                        if result is not None:
                            pep_start, pep_end, n_start, c_end, match_type = result
                            match_stats[match_type] = match_stats.get(match_type, 0) + 1
                            source_stats["af2_fallback"] += 1

                            n_slice   = af2_rep[n_start:pep_start]
                            pep_slice = af2_rep[pep_start:pep_end]
                            c_slice   = af2_rep[pep_end:c_end]

                            peptide_embs[orig_i] = pep_slice.mean(axis=0) if len(pep_slice) > 0 else np.zeros(EMB_DIM)
                            n_flank_embs[orig_i] = n_slice.mean(axis=0) if len(n_slice) > 0 else np.zeros(EMB_DIM)
                            c_flank_embs[orig_i] = c_slice.mean(axis=0) if len(c_slice) > 0 else np.zeros(EMB_DIM)
                continue

            rep, seq, chain_used = load_structure_best_chain(
                struct_path, uid, uid_peptides.get(uid, set())
            )

            if rep is None:
                print(f"  REJECTED {uid}: structure loaded but no peptides "
                      f"found in any chain (MHC complex or uncovered region)")
                chain_stats["rejected"] += 1
                missing_structures.add(uid)

                # Try AF2 fallback for rejected structures too
                if AF2_FALLBACK:
                    af2_rep, af2_seq = load_af2_fallback(uid, args.af2_fallback_dir)
                    if af2_rep is not None:
                        af2_cache.put(uid, (af2_rep, af2_seq))
                        result = find_peptide_in_structure(af2_seq, len(af2_rep), row)
                        if result is not None:
                            pep_start, pep_end, n_start, c_end, match_type = result
                            match_stats[match_type] = match_stats.get(match_type, 0) + 1
                            source_stats["af2_fallback"] += 1

                            n_slice   = af2_rep[n_start:pep_start]
                            pep_slice = af2_rep[pep_start:pep_end]
                            c_slice   = af2_rep[pep_end:c_end]

                            peptide_embs[orig_i] = pep_slice.mean(axis=0) if len(pep_slice) > 0 else np.zeros(EMB_DIM)
                            n_flank_embs[orig_i] = n_slice.mean(axis=0) if len(n_slice) > 0 else np.zeros(EMB_DIM)
                            c_flank_embs[orig_i] = c_slice.mean(axis=0) if len(c_slice) > 0 else np.zeros(EMB_DIM)
                continue

            structure_cache.put(uid, (rep, seq))
            cached = (rep, seq)

            if chain_used == "A":
                chain_stats["A"] += 1
            else:
                chain_stats["other_validated"] += 1
                if chain_stats["other_validated"] <= 50 or \
                   chain_stats["other_validated"] % 200 == 0:
                    print(f"  {uid}: validated chain '{chain_used}' "
                          f"(chain A unavailable/wrong protein)")

        rep, pdb_seq = cached

        # ── Locate peptide in primary structure ───────────────────────────
        result = find_peptide_in_structure(pdb_seq, len(rep), row)

        # ── FIX 8: AF2 fallback if peptide not in primary ────────────────
        used_fallback = False
        if result is None and AF2_FALLBACK:
            af2_cached = af2_cache.get(uid)
            if af2_cached is None:
                af2_rep, af2_seq = load_af2_fallback(uid, args.af2_fallback_dir)
                if af2_rep is not None:
                    af2_cache.put(uid, (af2_rep, af2_seq))
                    af2_cached = (af2_rep, af2_seq)

            if af2_cached is not None:
                af2_rep, af2_seq = af2_cached
                result = find_peptide_in_structure(af2_seq, len(af2_rep), row)
                if result is not None:
                    rep = af2_rep  # use AF2 rep for slicing
                    used_fallback = True

        if result is None:
            peptide_not_found += 1
            continue

        pep_start, pep_end, n_start, c_end, match_type = result
        match_stats[match_type] = match_stats.get(match_type, 0) + 1

        if used_fallback:
            source_stats["af2_fallback"] += 1
        else:
            source_stats["primary"] += 1

        # Slice and average
        n_slice   = rep[n_start:pep_start]
        pep_slice = rep[pep_start:pep_end]
        c_slice   = rep[pep_end:c_end]

        n_emb   = n_slice.mean(axis=0) if len(n_slice) > 0 else np.zeros(EMB_DIM)
        pep_emb = pep_slice.mean(axis=0) if len(pep_slice) > 0 else np.zeros(EMB_DIM)
        c_emb   = c_slice.mean(axis=0) if len(c_slice) > 0 else np.zeros(EMB_DIM)

        peptide_embs[orig_i] = pep_emb
        n_flank_embs[orig_i] = n_emb
        c_flank_embs[orig_i] = c_emb

# ── Summary ───────────────────────────────────────────────────────────────────
total_loaded = chain_stats["A"] + chain_stats["other_validated"]
total_proteins = df['uniprot_id'].nunique()

missing_rows = (df['uniprot_id'].isin(missing_structures)).sum()
embedded_rows = n_rows - missing_rows - peptide_not_found
# Add back the AF2 fallback recoveries for proteins that were "missing"
af2_recovered = source_stats["af2_fallback"]
true_embedded = source_stats["primary"] + source_stats["af2_fallback"]

print(f"\n{'='*60}")
print(f"DEVICE: {DEVICE}" + (f" (GPU validated)" if GPU_VALIDATED else ""))
print(f"{'='*60}")
print(f"STRUCTURE LOADING ({total_loaded} / {total_proteins} proteins):")
print(f"  Chain A (validated):         {chain_stats['A']}")
print(f"  Other chain (validated):     {chain_stats['other_validated']}")
print(f"  Rejected (no peptides):      {chain_stats['rejected']}")
print(f"  Failed (no file):            {chain_stats['failed']}")
print(f"")
print(f"PEPTIDE ALIGNMENT ({n_rows} rows):")
print(f"  Embedded (primary):          {source_stats['primary']}")
if AF2_FALLBACK:
    print(f"  Embedded (AF2 fallback):     {source_stats['af2_fallback']}")
print(f"  Embedded TOTAL:              {true_embedded}")
print(f"  Peptide NOT in any struct:   {peptide_not_found}  (-> zero vector)")
print(f"  Protein missing entirely:    {len(missing_structures)} proteins")
print(f"")
print(f"MATCH RESOLUTION BREAKDOWN:")
for mtype in ['full_window_unique', 'full_window_positional',
              'peptide_unique', 'peptide_positional']:
    count = match_stats.get(mtype, 0)
    pct = (count / max(true_embedded, 1)) * 100
    print(f"  {mtype:30s} {count:>7d}  ({pct:5.1f}%)")
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
print(f"  (out of {n_rows} rows = {zero_pep/n_rows*100:.1f}% / "
      f"{zero_n/n_rows*100:.1f}% / {zero_c/n_rows*100:.1f}%)")

if AF2_FALLBACK:
    print(f"\nAF2 fallback recovered {source_stats['af2_fallback']} peptides "
          f"that would have been zero vectors.")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {args.out_h5}...")
with h5py.File(args.out_h5, 'w') as f:
    f.create_dataset('peptide_if_struct', data=peptide_embs,  dtype='float32')
    f.create_dataset('n_flank_if_struct', data=n_flank_embs,  dtype='float32')
    f.create_dataset('c_flank_if_struct', data=c_flank_embs,  dtype='float32')
    f.create_dataset('peptide_ids',
                     data=np.array(orig_peptides, dtype='S20'))
    f.create_dataset('uniprot_ids',
                     data=np.array(orig_uniprots, dtype='S20'))

    f.attrs['model']     = 'esm_if1_gvp4_t16_142M_UR50'
    f.attrs['emb_dim']   = EMB_DIM
    f.attrs['n_samples'] = n_rows
    f.attrs['device']    = DEVICE
    f.attrs['cache_size'] = args.cache_size
    f.attrs['af2_fallback'] = AF2_FALLBACK
    f.attrs['af2_fallback_recoveries'] = source_stats['af2_fallback']
    f.attrs['primary_embeddings'] = source_stats['primary']

print("Done.")
