#!/usr/bin/env python3
"""
embed_structures_w_esm_if.py
Usage:
    python src/embed_structures_w_esm_if.py \
        data/processed/final_ml_dataset.csv \
        data/raw/alphafold_structures/ \
        data/processed/embeddings/esm_if_embeddings.h5
"""

import argparse
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob

# Meta's ESM imports
import esm
import esm.inverse_folding.util as esm_util

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('csv_path', help="Path to your final ML dataset CSV")
parser.add_argument('pdb_dir', help="Directory containing AlphaFold .pdb or .cif files")
parser.add_argument('out_h5', help="Path to save the output HDF5 embeddings")
args = parser.parse_args()

DEVICE = "cpu"
Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)

print(f"Device : {DEVICE}")

# ── Load ESM-IF1 ──────────────────────────────────────────────────────────────
print("Loading ESM-IF1 (142M parameters)...")
# This downloads the model weights the first time you run it
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to(DEVICE)
print("ESM-IF1 loaded successfully.")

EMB_DIM = 512 # ESM-IF1 hidden dimension size

# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(args.csv_path)
before = len(df)
df = df.dropna(subset=['peptide', 'uniprot_id', 'start', 'end'])
print(f"Total rows to process: {len(df)} (dropped {before - len(df)} NaNs)")
df = df.reset_index(drop=True)

# ── Helper to find AlphaFold file ─────────────────────────────────────────────
def find_structure_file(uniprot_id, search_dir):
    # AlphaFold files are usually named AF-<UNIPROT>-F1-model_v4.cif
    pattern = os.path.join(search_dir, f"*{uniprot_id}*.*")
    matches = glob.glob(pattern)
    # Filter to valid structure extensions
    matches = [m for m in matches if m.endswith('.pdb') or m.endswith('.cif')]
    return matches[0] if matches else None

# ── Embed ─────────────────────────────────────────────────────────────────────
print(f"\nExtracting 3D embeddings for {len(df)} sequences...")

peptide_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
n_flank_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
c_flank_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)

# Cache structures to avoid reloading the same protein multiple times
structure_cache = {}
missing_structures = set()

with torch.no_grad():
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  [{i}/{len(df)}]")

        uid = row['uniprot_id']
        
        # 1. Load Structure and get full-protein representations
        if uid in missing_structures:
            continue
            
        if uid not in structure_cache:
            struct_path = find_structure_file(uid, args.pdb_dir)
            if not struct_path:
                missing_structures.add(uid)
                continue
            
            try:
                # AlphaFold files usually map everything to chain 'A'
                coords, seq = esm_util.load_coords(struct_path, chain="A")
                # Move to GPU and get the structural embedding for every residue
                rep = esm_util.get_encoder_output(model, alphabet, coords).cpu().numpy()
                structure_cache[uid] = (rep, seq)
            except Exception as e:
                print(f"  Error loading {uid}: {e}")
                missing_structures.add(uid)
                continue
                
        rep, pdb_seq = structure_cache[uid]

        # 2. Calculate indices (R is 1-indexed, Python is 0-indexed)
        # We use the raw 'start' and 'end' from the CSV
        pep_start = int(row['start']) - 1
        pep_end   = int(row['end'])
        
        n_len = len(str(row['n_flank'])) if pd.notna(row['n_flank']) else 0
        c_len = len(str(row['c_flank'])) if pd.notna(row['c_flank']) else 0
        
        n_start = pep_start - n_len
        c_end   = pep_end + c_len
        
        # Safety check: ensure we don't slice outside the protein bounds
        n_start = max(0, n_start)
        c_end   = min(len(rep), c_end)

        # 3. Slice the structural embeddings and average them
        # 3. Safely slice the structural embeddings
        n_slice   = rep[n_start:pep_start]
        pep_slice = rep[pep_start:pep_end]
        c_slice   = rep[pep_end:c_end]

        # Average them, but only if the slice isn't empty (prevents NaN errors)
        n_emb   = n_slice.mean(axis=0) if len(n_slice) > 0 else np.zeros(EMB_DIM)
        pep_emb = pep_slice.mean(axis=0) if len(pep_slice) > 0 else np.zeros(EMB_DIM)
        c_emb   = c_slice.mean(axis=0) if len(c_slice) > 0 else np.zeros(EMB_DIM)
        
        peptide_embs[i] = pep_emb
        n_flank_embs[i] = n_emb
        c_flank_embs[i] = c_emb

print(f"\nMissing structures for {len(missing_structures)} unique proteins.")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {args.out_h5}...")
with h5py.File(args.out_h5, 'w') as f:
    f.create_dataset('peptide_if_struct', data=peptide_embs,  dtype='float32')
    f.create_dataset('n_flank_if_struct', data=n_flank_embs,  dtype='float32')
    f.create_dataset('c_flank_if_struct', data=c_flank_embs,  dtype='float32')
    f.create_dataset('peptide_ids', data=np.array(df['peptide'].tolist(), dtype='S20'))
    f.create_dataset('uniprot_ids', data=np.array(df['uniprot_id'].tolist(), dtype='S20'))
    
    f.attrs['model']      = 'esm_if1_gvp4_t16_142M_UR50'
    f.attrs['emb_dim']    = EMB_DIM
    f.attrs['n_samples']  = len(df)

print("Done.")
