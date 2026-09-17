#!/usr/bin/env python3
"""
Larger ESM-C padding test with explicit attention masking.

Tests, on ~100 real protein sequences:
  1. Does the model auto-mask pad tokens?  (`<pad>` yes, `X`/`-`/`<mask>` no)
  2. Does setting attention_mask=0 on pads (explicit sequence_id) isolate pads
     from real residues?  (real-position embeddings unchanged vs unmasked)
  3. Pad-token embedding magnitude and how distinct it is from real residues.

The mask semantics (esm/layers/attention.py):
    mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
  real==real -> attend;  real==pad -> no;  pad==pad -> attend (#299 quirk).
So explicit seq_id with False at pads stops real residues from attending pads.

Run in the ESM-C env (Windows base anaconda has torch + esm):

    /mnt/c/Users/olive/anaconda3/python.exe src/tools/esm/embed_test_esmc_pad.py

Args:
    --n N          number of proteins to sample (default 100)
    --k K          number of tail positions to pad (default 10)
"""
import argparse

import numpy as np
import pandas as pd
import torch

from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=10,
                help="number of tail (C-flank) positions replaced by pad")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = EsmSequenceTokenizer()
PAD_CHARS = ["<pad>", "X", "-", "<mask>"]

print(f"device={DEVICE}, loading ESMC-600M...")
model = ESMC.from_pretrained("esmc_600m").eval()

# ---- sample real proteins ---------------------------------------------------
df = pd.read_csv("data/processed/df_all.csv")
seqs = df["sequence"].dropna().astype(str).unique()
canonical = set("ACDEFGHIKLMNPQRSTVWY")
seqs = [s for s in seqs if len(s) >= 40 and set(s) <= canonical]
rng = np.random.default_rng(42)
seqs = rng.choice(seqs, size=min(args.n, len(seqs)), replace=False)
print(f"sampled {len(seqs)} proteins (canonical AA only, len>=40)\n")


def embed(tokens: torch.Tensor, seq_id: torch.Tensor | None) -> np.ndarray:
    """Run ESMC.forward on a [1, L] token tensor, return embeddings [L, D]."""
    tokens = tokens.unsqueeze(0).to(DEVICE)
    if seq_id is not None:
        seq_id = seq_id.unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        out = model.forward(sequence_tokens=tokens, sequence_id=seq_id)
    return out.embeddings[0].float().cpu().numpy()  # [L, D]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean())


# ---- per pad-char accumulators ---------------------------------------------
results = {c: {"real_cos_auto_vs_mask": [],   # ~1 if auto already masks
               "real_cos_full_vs_mask": [],   # how much pads disturb reals if NOT masked
               "pad_norm": [],                # embedding magnitude at pad slots
               "pad_real_cos": []}            # pad embedding vs real embedding
           for c in PAD_CHARS}

for si, prot in enumerate(seqs):
    L = len(prot)
    # fixed real core: 10 N-flank + 9 peptide = 19 residues at a mid position
    p = rng.integers(10, L - 9 - args.k)   # leave room for 10 n-flank + 9 pep + k tail
    core = prot[p - 10:p + 9]              # 19 chars
    tail_real = prot[p + 9:p + 9 + args.k]  # the k real residues we'll replace

    for c in PAD_CHARS:
        window = core + c * args.k          # 19 real + k pad chars
        tokens = torch.tensor(tok.encode(window), dtype=torch.long)  # [19+k+2]
        pad_id = tok.encode(c)[1]           # token id of pad char
        Ltot = tokens.shape[0]

        # pad positions = the k chars at the tail (exclude cls/eos)
        is_real = (tokens != pad_id)        # True = real (cls, eos, core)
        # explicit mask: attention_mask=0 at pad positions
        mask = is_real.clone()              # bool, True=participate

        e_auto = embed(tokens, None)                          # model default
        e_mask = embed(tokens, mask)                          # explicit pad mask
        e_full = embed(tokens, torch.ones_like(tokens, dtype=torch.bool))  # nothing masked

        # real positions = all except cls/eos/pads
        real_idx = [i for i in range(Ltot)
                    if i not in (0, Ltot - 1) and tokens[i] != pad_id]
        pad_idx = [i for i in range(1, Ltot - 1) if tokens[i] == pad_id]

        r = results[c]
        r["real_cos_auto_vs_mask"].append(
            cos(e_mask[real_idx], e_auto[real_idx]))
        r["real_cos_full_vs_mask"].append(
            cos(e_mask[real_idx], e_full[real_idx]))
        r["pad_norm"].append(np.linalg.norm(e_mask[pad_idx], axis=-1).mean())
        # pad embedding vs real core embedding (both masked)
        r["pad_real_cos"].append(
            cos(e_mask[pad_idx].mean(0, keepdims=True),
                e_mask[real_idx].mean(0, keepdims=True)))

    if (si + 1) % 25 == 0:
        print(f"  [{si+1}/{len(seqs)}] done")

# ---- summary -----------------------------------------------------------------
print("\n" + "=" * 78)
print(f"SUMMARY  (n={len(seqs)} proteins, k={args.k} tail pads)")
print("=" * 78)
print(f"  {'pad':<8} {'auto=mask?':>12} {'unmasked disturb':>18} "
      f"{'pad_norm':>10} {'pad~real cos':>13}")
print("  " + "-" * 60)
for c in PAD_CHARS:
    r = results[c]
    auto = float(np.mean(r["real_cos_auto_vs_mask"]))
    full = float(np.mean(r["real_cos_full_vs_mask"]))
    pnorm = float(np.mean(r["pad_norm"]))
    prcos = float(np.mean(r["pad_real_cos"]))
    print(f"  {c:<8} {auto:>12.4f} {full:>18.4f} {pnorm:>10.3f} {prcos:>13.4f}")

print("""
INTERPRETATION
  auto=mask?      ~1.0000 means the model's default already masks that pad char
                  (only <pad> is expected to). <1.0 means the pad char ATTENDS
                  to real residues unless you pass an explicit mask.
  unmasked disturb  ~1.0000 means pads do NOT change real residues even when
                  forced to attend (inert); lower means pads leak signal into
                  real residues when unmasked -> you MUST mask them.
  pad_norm        magnitude of pad-position embedding vs ~1.5 for real residues.
  pad~real cos    how similar the pad embedding is to a real-residue embedding
                  (high = hard to distinguish from real, low = clearly distinct).
""")
