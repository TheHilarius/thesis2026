#!/usr/bin/env python3
"""Tiny smoke test for embed_peptides_esmc6b helpers.

Uses a randomly-initialised tiny EsmcForMaskedLM (no weights downloaded) to
check the one thing that is easy to get wrong: the integer sequence_id padding
semantics and output shapes. Run with the env_esmc6b interpreter:

    env_esmc6b/bin/python src/tools/esm/test_embed_esmc6b.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer  # noqa: E402
from esm.models.esmc.config import EsmcConfig  # noqa: E402

import embed_peptides_esmc6b as e  # noqa: E402


def main() -> None:
    cfg = EsmcConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=2)
    model = EsmcForMaskedLM(cfg).eval()
    tok = EsmcTokenizer()
    dev = torch.device("cpu")
    pad_id = tok.pad_token_id
    assert pad_id == 1, pad_id

    seq = "ACDEFGHIKL"
    full = e.embed_full_protein(model, tok, seq, dev)
    assert full.shape == (len(seq) + 2, 64), full.shape

    padded = "XXX" + seq + "XX"
    emb, n_left = e.embed_padded_protein(model, tok, padded, pad_id, dev)
    assert emb.shape == (len(padded) + 2, 64), emb.shape
    assert n_left == 3, n_left

    # Pad slots must be model-derived (non-zero), not zero-fill.
    x = torch.tensor([ch in ("X", "x") for ch in padded])[None, :, None].numpy()
    pad_norm = float((np.abs(emb[1:-1]) * x).sum())
    assert pad_norm > 0, "pad slots are all zero"

    # pad_mask flags exactly the out-of-protein window slots.
    w, pm = e.extract_pad_window(emb, win_start=0, n_left=n_left, L_orig=len(seq))
    assert w.shape == (e.WINDOW_LEN, 64), w.shape
    assert not pm[: len(seq)].any() and pm[len(seq) :].all(), pm

    print("OK: esmc6b embed helpers (shapes, int sequence_id, pad vectors)")


if __name__ == "__main__":
    main()
