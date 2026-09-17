#!/usr/bin/env python3
"""
Probe ESM-C tokenizer + pad-token embeddings.

Answers three questions:
  1. What token id does each literal string map to (X / - / <pad> / space / ...)?
  2. What is the embedding magnitude at a pad slot vs a real residue?
  3. Is <pad> a real, injectable token, or a string that becomes <unk> chars?

Run in the ESM-C env (the one with `esm` / esm.models.esmc).

From WSL, use the WINDOWS anaconda base python (WSL `python` has no torch):

    /mnt/c/Users/olive/anaconda3/python.exe src/tools/esm/embed_test_esmc.py

Note: this is the `esm` (ESM-C/ESM3) package, NOT fair-esm.
"""
import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = EsmSequenceTokenizer()

print("=" * 70)
print("1) VOCAB (id -> literal)")
print("=" * 70)
for i in range(33):
    print(f"  {i:2d} -> {tok.decode(i)!r}")

print("\n" + "=" * 70)
print("2) LITERAL STRING -> TOKEN IDS  (tokenizer adds <cls>...<eos>)")
print("=" * 70)
for s in ["X", "-", ".", "|", "<mask>", "<pad>", "<cls>", "<eos>", "<unk>", " "]:
    ids = tok.encode(s)
    print(f"  {s!r:10s} -> {ids}")

print("\n" + "=" * 70)
print("3) EMBEDDING MAGNITUDE: pad slot vs real residue")
print("=" * 70)
print(f"  device: {DEVICE}")
model = ESMC.from_pretrained("esmc_600m").eval().to(DEVICE)
BASE = "ACDEFGHIK"  # 9-mer


def emb(seq: str) -> np.ndarray:
    with torch.no_grad():
        t = model.encode(ESMProtein(sequence=seq))
        out = model.logits(t, LogitsConfig(sequence=True, return_embeddings=True))
    return out.embeddings.squeeze(0).cpu().numpy()  # [L+2, D]


e_base = emb(BASE)         # [11, 1152] = <cls> + 9 + <eos>
e_real = e_base[1:-1]      # strip <cls>/<eos>
real_norm = np.linalg.norm(e_real, axis=1)
print(f"  real residue norms: median={np.median(real_norm):.3f} "
      f"min={real_norm.min():.3f} max={real_norm.max():.3f}")
print()

for pad in ["X", "-", "<pad>", "<mask>", "<unk>"]:
    e = emb(BASE + pad)
    e_pad = e[-2]  # pad token sits just before <eos>
    pad_id = tok.encode(BASE + pad)[-2]
    print(f"  pad {pad!r:8s} id={pad_id:2d}  norm={np.linalg.norm(e_pad):.3f}  "
          f"(real median {np.median(real_norm):.3f})")

print("\n" + "=" * 70)
print("4) SANITY")
print("=" * 70)
print("  encode('X')      =", tok.encode("X"),       " (expect [0, 24, 2])")
print("  encode('-')      =", tok.encode("-"),       " (expect [0, 30, 2])")
print("  encode(' ')      =", tok.encode(" "),       " (space = <unk>, NOT <pad>)")
print("  encode('<pad>')  =", tok.encode("<pad>"),   " (expect [0, 1, 2])")
print("  encode('<mask>') =", tok.encode("<mask>"),  " (expect [0, 32, 2])")
print()
print("  INTERPRETATION")
print("    - <pad> is injectable via the literal string '<pad>' (id 1).")
print("    - space is <unk> (id 3), NOT pad.")
print("    - a real residue has median norm ~1.5; see how each pad token compares.")
print("    - if all pad norms are ~ real-norm scale, the CNN cannot detect 'empty'")
print("      by magnitude alone -> still need an explicit is_pad mask channel.")
