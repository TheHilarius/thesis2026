#!/usr/bin/env python3
"""
Probe ESM-IF: append/prepend NaN-coordinate "gap" residues.

Answers: do NaN-coordinate "empty" residues produce finite, sane embeddings?
NaN coords are ESM-IF's in-distribution signal for "no structure here"
(model trained with span-masking over missing backbone coords).

Run in the FAIR-ESM env (the one with fair-esm, NOT the `esm` ESM-C package).

This env is on the H100 cluster (`conda activate esm_gpu`), not the local Windows
anaconda (which has the new `esm` ESM-C package instead).

    python src/tools/esm/embed_test_esmif.py --pdb path/to/small.pdb
    python src/tools/esm/embed_test_esmif.py   # auto-finds a small AF2 pdb

If you get "No module named 'esm.inverse_folding'", you are in the wrong env.
"""
import argparse
import glob
import os

import numpy as np
import torch

import esm
import esm.inverse_folding.util as util

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", default=None,
                help="path to a small .pdb/.cif (default: auto-find a small AF2 pdb)")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# -- resolve structure ---------------------------------------------------------
pdb_path = args.pdb
if pdb_path is None:
    cands = glob.glob("data/processed/structures/alphafold/*.pdb")
    if not cands:
        raise SystemExit("No default pdb found and --pdb not given. "
                         "Pass --pdb path/to/small.pdb")
    pdb_path = min(cands, key=os.path.getsize)
print(f"using structure: {pdb_path} ({os.path.getsize(pdb_path)} bytes)")

# -- load model ----------------------------------------------------------------
print("loading ESM-IF1 (142M)...")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to(DEVICE)

coords, seq = util.load_coords(pdb_path, chain="A")  # [L, 4, 3]
print(f"real coords: shape={coords.shape} dtype={coords.dtype}")
print(f"seq head: {seq[:20]}...  (L={len(seq)})")

with torch.no_grad():
    rep = util.get_encoder_output(model, alphabet, coords)
print(f"real rep: shape={rep.shape}  finite={np.isfinite(rep).all()}")

NAN_ROWS = 10

def probe(name: str, coords_padded: np.ndarray, gap_slice):
    with torch.no_grad():
        rep2 = util.get_encoder_output(model, alphabet, coords_padded)
    finite = np.isfinite(rep2).all()
    gap = rep2[gap_slice]
    gap_norms = np.linalg.norm(gap, axis=1)
    real_norms = np.linalg.norm(rep2[:len(rep)] if gap_slice.start >= len(rep)
                                else rep2[len(gap):], axis=1)
    print(f"\n[{name}] padded rep shape={rep2.shape}  finite={finite}")
    print(f"  gap norms     {np.round(gap_norms, 3)}")
    print(f"  real median   {np.median(real_norms):.3f}  "
          f"(gap median {np.median(gap_norms):.3f})")
    return finite

nan_rows = np.full((NAN_ROWS, 4, 3), np.nan, dtype=coords.dtype)

# C-flank pad: NaN after the real chain
c_pad = np.concatenate([coords, nan_rows], axis=0)
finite_c = probe("C-flank (append NaN)", c_pad, slice(len(coords), len(coords) + NAN_ROWS))

# N-flank pad: NaN before the real chain
n_pad = np.concatenate([nan_rows, coords], axis=0)
finite_n = probe("N-flank (prepend NaN)", n_pad, slice(0, NAN_ROWS))

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  finite (append):  {finite_c}")
print(f"  finite (prepend): {finite_n}")
if finite_c and finite_n:
    print("  -> NaN-coord 'gap' residues produce finite embeddings.")
    print("     If gap norms ~ real norms, this is in-distribution and usable as")
    print("     the ESM-IF pad representation (mirrors AntiFold's gap tokens).")
else:
    print("  -> NaN propagation detected. Use post-model zero-vector padding instead.")
