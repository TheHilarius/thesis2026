#!/usr/bin/env python3
"""
embed_peptides_w_esm.py
Usage:
    python src/embed_peptides_w_esm.py \
        data/processed/epitopes_with_features.csv \
        data/processed/embeddings/esmc_embeddings.h5 \
        --model esmc_300m
"""

import argparse
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('csv_path')
parser.add_argument('out_h5')
parser.add_argument('--model', default='esmc_300m',
                    choices=['esmc_300m', 'esmc_600m'])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)

print(f"Model  : {args.model}")
print(f"Device : {DEVICE}")

# ── Load ESM-C ────────────────────────────────────────────────────────────────
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

print("Loading ESM-C...")
client = ESMC.from_pretrained(args.model).to(DEVICE)
print("ESM-C loaded")

EMB_DIM = 960 if args.model == 'esmc_300m' else 1152

# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv(args.csv_path)
print(f"Total rows: {len(df)}")

before = len(df)
df = df.dropna(subset=['peptide', 'full_context', 'n_flank', 'c_flank'])
print(f"After dropping NaN: {len(df)} (dropped {before - len(df)})")
df = df.reset_index(drop=True)

# ── Validate full_context ─────────────────────────────────────────────────────
print("Validating full_context...")
mismatches = 0
for _, row in df.iterrows():
    expected = str(row['n_flank']) + str(row['peptide']) + str(row['c_flank'])
    if str(row['full_context']) != expected:
        mismatches += 1
        if mismatches <= 3:
            print(f"  WARNING: {row['uniprot_id']}")
            print(f"    expected: {expected}")
            print(f"    got:      {row['full_context']}")
print(f"Mismatches: {mismatches} / {len(df)}")

# ── Check BOS/EOS offset — NOW df is defined ─────────────────────────────────
print("\nChecking ESM-C token offset...")
test_seq = str(df.iloc[0]['full_context'])
test_out = client.logits(
               client.encode(ESMProtein(sequence=test_seq)),
               LogitsConfig(sequence=True, return_embeddings=True))
test_emb = test_out.embeddings.squeeze(0)
offset   = test_emb.shape[0] - len(test_seq)
print(f"Sequence length : {len(test_seq)}")
print(f"Embedding length: {test_emb.shape[0]}")
print(f"Offset          : {offset}  (BOS + EOS)")
assert offset == 2, f"Unexpected offset {offset}"

# ── Embed ─────────────────────────────────────────────────────────────────────
print(f"\nEmbedding {len(df)} sequences...")

peptide_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
n_flank_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
c_flank_embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)

with torch.no_grad():
    for i, (_, row) in enumerate(df.iterrows()):

        if i % 500 == 0:
            print(f"  [{i}/{len(df)}]")

        n = len(str(row['n_flank']))
        p = len(str(row['peptide']))
        c = len(str(row['c_flank']))

        protein        = ESMProtein(sequence=str(row['full_context']))
        protein_tensor = client.encode(protein)
        logits_output  = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )

        # emb shape: [seq_len+2, EMB_DIM]
        # index 0          = BOS  → skip
        # index 1..1+n     = n_flank
        # index 1+n..1+n+p = peptide
        # index 1+n+p..end = c_flank (EOS ignored by slicing)
        emb = logits_output.embeddings.squeeze(0).cpu().numpy()

        n_emb   = emb[1       : 1+n    ].mean(axis=0) if n > 0 else np.zeros(EMB_DIM)
        pep_emb = emb[1+n     : 1+n+p  ].mean(axis=0)
        c_emb   = emb[1+n+p   : 1+n+p+c].mean(axis=0) if c > 0 else np.zeros(EMB_DIM)

        peptide_embs[i] = pep_emb
        n_flank_embs[i] = n_emb
        c_flank_embs[i] = c_emb

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {args.out_h5}...")
with h5py.File(args.out_h5, 'w') as f:
    f.create_dataset('peptide_in_context', data=peptide_embs,  dtype='float32')
    f.create_dataset('n_flank_in_context', data=n_flank_embs,  dtype='float32')
    f.create_dataset('c_flank_in_context', data=c_flank_embs,  dtype='float32')
    f.create_dataset('peptide_ids',
                     data=np.array(df['peptide'].tolist(),    dtype='S20'))
    f.create_dataset('uniprot_ids',
                     data=np.array(df['uniprot_id'].tolist(), dtype='S20'))
    f.create_dataset('row_indices',
                     data=np.array(df.index.tolist()))
    f.attrs['model']      = args.model
    f.attrs['emb_dim']    = EMB_DIM
    f.attrs['n_samples']  = len(df)
    f.attrs['csv_source'] = args.csv_path

print("Done.")
print(f"  peptide_in_context : {peptide_embs.shape}")
print(f"  n_flank_in_context : {n_flank_embs.shape}")
print(f"  c_flank_in_context : {c_flank_embs.shape}")
