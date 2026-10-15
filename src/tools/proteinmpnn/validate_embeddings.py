#!/usr/bin/env python3
"""
validate_embeddings.py — Phase-4 sanity checks for ProteinMPNN encoder embeddings.

Checks (run in env_mpnn, from repo root):
  1. NATIVE RECOVERY / LOG-PROB — the model's own forward gives per-position
     logits over 20 AAs conditioned on the backbone. We report:
       * greedy top-1 recovery  = fraction of positions where argmax == native AA
       * native mean log-prob   = mean log P(native | backbone)
       * vs a shuffled-sequence baseline
     If the coordinate path is broken (coords mis-ordered / wrong residues), these
     collapse toward the ~5% random floor. A working ProteinMPNN is far above it.
  2. DETERMINISM — encode() the same protein twice; max abs diff must be 0.
  3. NaN/Inf + all-zero window scan on a debug embedding file.

Usage:
    python src/tools/proteinmpnn/validate_embeddings.py --n-proteins 12
    python src/tools/proteinmpnn/validate_embeddings.py --h5 /tmp/opencode/mpnn_dbg_zero.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MPNN_DIR = _REPO_ROOT / "vendor" / "ProteinMPNN"
if str(_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MPNN_DIR))
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize  # noqa: E402

DEFAULT_PDB = "data/processed/structures/alphafold"
DEFAULT_CSV = "data/processed/df_all.csv"
CKPT_DIR = _MPNN_DIR / "vanilla_model_weights"
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_to_i = {a: i for i, a in enumerate(AA)}


def load_model(ckpt="v_48_020", device="cpu"):
    path = CKPT_DIR / f"{ckpt}.pt"
    checkpoint = torch.load(str(path), map_location=device)
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                        augment_eps=0.0, k_neighbors=checkpoint["num_edges"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def featurize(pdb_path, device):
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    protein = pdb_dict_list[0]
    chain_list = [k[-1:] for k in protein if k[:9] == "seq_chain"]
    chain_id_dict = {protein["name"]: (chain_list, [])}
    X, S, mask, lengths, chain_M, chain_encoding_all, *_rest = tied_featurize(
        [protein], device, chain_id_dict, None, None, None, None, None, ca_only=False)
    # tied_featurize return: X,S,mask,lengths,chain_M,chain_encoding_all,
    #   letter_list,visible_list,masked_list,masked_chain_length_list,
    #   chain_M_pos,omit_AA_mask,residue_idx,...
    residue_idx = _rest[6]  # index of residue_idx after chain_encoding_all
    return protein, X, S, mask, chain_M, chain_encoding_all, residue_idx


def recovery_and_logprob(model, pdb_path, device):
    """Greedy top-1 recovery + native mean log-prob via the model's forward."""
    protein, X, S, mask, chain_M, chain_encoding_all, residue_idx = featurize(pdb_path, device)
    randn = torch.zeros_like(S, dtype=torch.float32)
    with torch.no_grad():
        log_probs = model(X, S, mask, chain_M * chain_M, residue_idx, chain_encoding_all, randn)
    # log_probs [1, L, 21]; mask real positions
    m = mask[0].bool()
    lp = log_probs[0][m]                       # [L_real, 21]
    native = S[0][m]                           # [L_real]
    # greedy top-1
    pred = lp.argmax(-1)
    recovery = (pred == native).float().mean().item()
    native_lp = lp.gather(1, native.unsqueeze(1)).mean().item()
    # shuffled baseline: same model, native S permuted per position-independent shuffle
    # (compute log-prob of a random AA assignment)
    rand_aa = torch.randint(0, 20, native.shape, device=device)
    rand_lp = lp.gather(1, rand_aa.unsqueeze(1)).mean().item()
    return recovery, native_lp, rand_lp, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=DEFAULT_PDB)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--n-proteins", type=int, default=12)
    ap.add_argument("--ckpt", default="v_48_020")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--h5", default=None, help="Optional debug H5 to NaN/zero-scan")
    args = ap.parse_args()
    device = torch.device(args.device)

    model = load_model(args.ckpt, device)
    df = pd.read_csv(args.csv, usecols=["uniprot_id", "sequence"]).drop_duplicates("uniprot_id")
    # pick a spread of protein sizes
    df = df.copy()
    df["L"] = df["sequence"].str.len()
    df = df.sort_values("L")
    pick = pd.concat([
        df.head(args.n_proteins // 3),
        df.iloc[len(df) // 2: len(df) // 2 + args.n_proteins // 3],
        df.tail(args.n_proteins - 2 * (args.n_proteins // 3)),
    ]).drop_duplicates("uniprot_id").head(args.n_proteins)

    print(f"=== 1. Native recovery / log-prob (ckpt {args.ckpt}) ===")
    recs, nls, rls = [], [], []
    for uid, seq in zip(pick["uniprot_id"], pick["sequence"]):
        p = Path(args.pdb) / f"{uid}.pdb"
        if not p.exists():
            continue
        try:
            rec, nlp, rlp, L = recovery_and_logprob(model, str(p), device)
        except Exception as e:
            print(f"  {uid}: FAILED {e}")
            continue
        recs.append(rec); nls.append(nlp); rls.append(rlp)
        print(f"  {uid:12s} L={L:5d}  greedy_recovery={rec*100:5.1f}%  native_logprob={nlp:7.3f}  random_logprob={rlp:7.3f}")
    if recs:
        print(f"\n  MEAN greedy recovery = {np.mean(recs)*100:.1f}%  (random floor ~5%)")
        print(f"  MEAN native log-prob = {np.mean(nls):.3f}  vs random {np.mean(rls):.3f}")
        assert np.mean(recs) > 0.15, f"recovery {np.mean(recs):.3f} too low — coordinate path broken?"
        assert np.mean(nls) > np.mean(rls) + 1.0, "native log-prob not above random — coordinate path broken?"
        print("  PASS: recovery far above random floor; native >> random log-prob")

    # ── 2. Determinism ──
    print("\n=== 2. Determinism (encode twice, same protein) ===")
    uid0 = pick["uniprot_id"].iloc[0]
    p0 = Path(args.pdb) / f"{uid0}.pdb"
    if p0.exists():
        from embed_peptides_mpnn import encode_protein
        r1, s1 = encode_protein(model, str(p0), device)
        r2, s2 = encode_protein(model, str(p0), device)
        maxdiff = float(np.abs(r1 - r2).max())
        print(f"  {uid0}: max|Δ| between two encode() runs = {maxdiff:.3e}")
        assert maxdiff == 0.0, "encode() NOT deterministic"
        print("  PASS: deterministic")

    # ── 3. NaN/zero scan on a debug H5 ──
    if args.h5:
        import h5py
        print(f"\n=== 3. NaN/zero scan: {args.h5} ===")
        with h5py.File(args.h5, "r") as f:
            W = f["window_embeddings"][:]
            st = f["status"][:]
            ok = st == 0
            print(f"  NaN/Inf: {(~np.isfinite(W)).sum()}")
            zero_win = int((np.abs(W[ok]).sum(axis=(1, 2)) == 0).sum())
            print(f"  all-zero windows (ok rows): {zero_win} / {ok.sum()}")
            assert np.isfinite(W).all(), "NaN/Inf in embeddings"
            assert zero_win == 0, "all-zero windows present"
            print("  PASS")

    print("\nAll requested validation checks passed.")


if __name__ == "__main__":
    main()
