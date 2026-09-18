#!/usr/bin/env python3
"""
embed_test_esmif_pad.py — ESM-IF NaN-gap padding metrics.

Ports the metrics from embed_test_esmc_pad.py to ESM-IF, where the "pad token"
is a NaN-coordinate gap residue (the AntiFold / in-distribution "no structure
here" signal).  Runs on a sample of small real structures and answers:

  1. finite?            do NaN-gap positions yield finite embeddings?
  2. gapmask inert?     cos(real@gapmasked, real@nogap)  ~1.0 -> appending
                        MASKED gaps does NOT change real-residue embeddings.
                        This is the property our full run relies on (we grab
                        real reps from the full chain, then post-fill).
  3. unmasked leak?     cos(real@unmasked, real@gapmasked) <1 -> gaps contaminate
                        real residues if padding_mask is NOT applied.
  4. pad_norm           magnitude of gap-position embeddings vs real residues.
  5. pad_real_cos       gap embedding vs mean real embedding similarity
                        (high = hard to distinguish from real, low = distinct).
  6. eos_norm / eos_cos norm and similarity of the end-token output — this is
                        the "eos" constant used by the eos-repeat fill mode.

The pad/eos constants probed here are derived exactly the same way as in
embed_windows_esmif.py (end-token output = eos_rep, first gap output = pad_rep),
so this script directly validates those constants before the full run.

Run in the fair-esm env (conda activate esm_gpu), NOT the esm (ESM-C) env:

    python src/tools/esm/embed_test_esmif_pad.py
    python src/tools/esm/embed_test_esmif_pad.py --n 20 --k 10

Args:
    --n N          number of structures to sample (default 20)
    --k K          number of NaN tail rows to append (default 10)
    --pdb-dir DIR  structure dir (default data/processed/structures/alphafold)
"""
import argparse
import glob
import os

import numpy as np
import torch

import esm
import esm.inverse_folding.util as esm_util

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--pdb-dir", default="data/processed/structures/alphafold")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}, loading ESM-IF1 (142M)...")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to(DEVICE)

try:
    from esm.inverse_folding.util import CoordBatchConverter
    batch_converter = CoordBatchConverter(alphabet)
except ImportError:
    raise SystemExit("CoordBatchConverter not found — wrong env (need fair-esm)")

# ── sample small structures (small = fast to encode) ──────────────────────────
cands = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")),
               key=os.path.getsize)
if not cands:
    raise SystemExit(f"No .pdb found in {args.pdb_dir}")

rng = np.random.default_rng(42)
pool = cands[: max(args.n * 4, 40)]
print(f"sampling from {len(pool)} smallest structures in {args.pdb_dir}\n")


def forward(coords, unmasked=False):
    """Run the ESM-IF encoder on one chain, return raw reps [T, D].

    unmasked=True forces padding_mask to all-ones, simulating the case where
    NaN gaps are allowed to attend (the leaky condition we want to quantify).
    """
    coords_t, confidence, strs, tokens, padding_mask = batch_converter(
        [(coords, None, None)])
    coords_t = coords_t.to(DEVICE)
    confidence = confidence.to(DEVICE)
    padding_mask = padding_mask.to(DEVICE)
    if unmasked:
        padding_mask = torch.ones_like(padding_mask)
    with torch.no_grad():
        out = model.encoder(coords_t, padding_mask, confidence,
                            return_all_hiddens=False)
    rep = out['encoder_out'][0].transpose(0, 1)
    return rep[0].cpu().numpy()  # [T, D]


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())


rows = []

for prot_i in range(args.n):
    if not pool:
        break
    pdb = pool[prot_i]

    try:
        coords, seq = esm_util.load_coords(pdb, chain="A")
    except Exception as e:
        print(f"  skip {os.path.basename(pdb)}: {e}")
        continue

    L = len(seq)
    if L < 20:
        print(f"  skip {os.path.basename(pdb)}: too short (L={L})")
        continue

    rep_nogap = forward(coords)
    offset = rep_nogap.shape[0] - L
    if offset != 2:
        print(f"  skip {os.path.basename(pdb)}: unexpected offset {offset}")
        continue

    # coords is L x 3 x 3 (N, CA, C) — derive gap shape from it, don't hardcode
    nan_rows = np.full((args.k,) + coords.shape[1:], np.nan, dtype=coords.dtype)
    rep_gap = forward(np.concatenate([coords, nan_rows], axis=0))
    rep_unmasked = forward(np.concatenate([coords, nan_rows], axis=0),
                           unmasked=True)

    real_idx = slice(1, 1 + L)             # real residues
    gap_idx = slice(1 + L, 1 + L + args.k)  # NaN gaps

    finite_gap = bool(np.isfinite(rep_gap[gap_idx]).all())
    unmasked_finite = bool(np.isfinite(rep_unmasked[real_idx]).all())
    eos_vec = rep_nogap[-1]                 # end-token output

    rows.append({
        "name": os.path.basename(pdb),
        "L": L,
        "finite": finite_gap,
        "unmasked_finite": unmasked_finite,
        "real_cos_gapmask_vs_nogap": cos(rep_gap[real_idx], rep_nogap[real_idx]),
        "real_cos_unmasked_vs_gapmask": (
            cos(rep_unmasked[real_idx], rep_gap[real_idx])
            if unmasked_finite else float("nan")),
        "pad_norm": np.linalg.norm(rep_gap[gap_idx], axis=-1).mean(),
        "real_norm": np.linalg.norm(rep_nogap[real_idx], axis=-1).mean(),
        "pad_real_cos": cos(rep_gap[gap_idx].mean(0, keepdims=True),
                            rep_nogap[real_idx].mean(0, keepdims=True)),
        "eos_norm": float(np.linalg.norm(eos_vec)),
        "eos_real_cos": cos(eos_vec[None], rep_nogap[real_idx].mean(0, keepdims=True)),
    })
    print(f"  [{len(rows)}] {os.path.basename(pdb)} (L={L}) done")

# ── summary ────────────────────────────────────────────────────────────────────
if not rows:
    raise SystemExit("No usable structures sampled.")

print("\n" + "=" * 78)
print(f"SUMMARY (n={len(rows)} structures, k={args.k} NaN tail rows)")
print("=" * 78)

def agg(key):
    return float(np.mean([r[key] for r in rows]))

n_catastrophic = sum(1 for r in rows if not r["unmasked_finite"])
print(f"  finite (NaN gap embeddings)      : {all(r['finite'] for r in rows)}")
print(f"  unmasked -> NaN leak (real pos)  : {n_catastrophic}/{len(rows)} catastrophic")
print(f"  {'metric':<32} {'mean':>10}  {'min':>10}  {'max':>10}")
print("  " + "-" * 66)
for key, label in [
    ("real_cos_gapmask_vs_nogap", "gapmasked vs nogap (inert?)"),
    ("real_cos_unmasked_vs_gapmask", "unmasked vs gapmasked (leak?)"),
    ("pad_norm", "pad_norm"),
    ("real_norm", "real_norm"),
    ("pad_real_cos", "pad~real cos"),
    ("eos_norm", "eos_norm"),
    ("eos_real_cos", "eos~real cos"),
]:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    if vals:
        print(f"  {label:<32} {np.mean(vals):>10.4f}  {np.min(vals):>10.4f}  {np.max(vals):>10.4f}")
    else:
        print(f"  {label:<32} {'—':>10}  {'—':>10}  {'—':>10}")

print("""
INTERPRETATION
  gapmasked vs nogap   ~1.0000 means appending MASKED NaN gaps leaves real-residue
                       embeddings unchanged -> safe to embed the full chain and
                       post-fill (our full run's assumption).
  unmasked vs gapmasked <1 means NaN gaps contaminate real residues if padding_mask
                       is NOT applied.  NaN in the cosine (or "catastrophic" count
                       > 0) means forcing padding_mask=1 lets NaN coords propagate
                       into real positions -> never override the auto mask.
  finite                True means the pad_rep/eos_rep constants are usable; if
                       False the fill modes would inject NaN and must fall back
                       to zero vectors.
  pad_norm vs real_norm  if pad_norm ~ real_norm, gaps are in-distribution (hard
                       to distinguish by magnitude alone); if ~0, clearly distinct.
  pad~real cos          low = gap embedding clearly distinct from real residues.
  eos_norm / eos~real   sanity on the eos-repeat fill constant.
""")
