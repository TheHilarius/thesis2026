#!/usr/bin/env python3
"""
embed_peptides_mpnn.py — ProteinMPNN encoder-only per-residue embeddings for the
fixed 29-mer full_context windows.

For each protein: parse the AFDB backbone once, run the ENCODER only (added
ProteinMPNN.encode() in vendor/ProteinMPNN — no decoder, so no sequence leak),
giving per-residue h_V [L, 128]. Then slice the same 29-slot windows the ESM-C
script uses (win_start = start0 - 10) into two pad modes:

  zero     — pad slots = 0
  boundary — pad slots = nearest real boundary residue (terminus fallback = the
             peptide's own boundary residue)

ProteinMPNN has NO BOS/EOS tokens and NO pad/NaN-gap convention, so pad_token
and eos_bos_repeat are NOT implemented (would invent semantics). Documented here
rather than faked.

Mapping is validated per protein: if the MPNN-parsed sequence equals the df
`sequence` column, the coordinate window (win_start = start0 - 10) is used
(status=0 positional). If they differ (13 known isoform-mismatch proteins), the
peptide is located by sequence search (the same logic ESM-IF uses; status=1
string-search) so all 9148 proteins are mapped. A peptide genuinely absent from
its structure is zero-filled and flagged (status=3), never silently mis-mapped.
Windowed pLDDT (AFDB CA B-factor) is stored as an extra dataset for downstream
confidence stratification.

Output (never overwrites):
  <out-prefix>_zero.h5, <out-prefix>_boundary.h5
  datasets: window_embeddings [N,29,128], pad_mask [N,29], window_plddt [N,29],
            fallback_flag [N], peptide, uniprot_id, row_indices, status

Usage:
    python src/tools/proteinmpnn/embed_peptides_mpnn.py --dry-run
    python src/tools/proteinmpnn/embed_peptides_mpnn.py --n-debug 200 --out-prefix /tmp/mpnn_dbg
    python src/tools/proteinmpnn/embed_peptides_mpnn.py --checkpoint v_48_020
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

# ── Vendored ProteinMPNN (pinned + encode() patch) ────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MPNN_DIR = _REPO_ROOT / "vendor" / "ProteinMPNN"
if str(_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MPNN_DIR))
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize  # noqa: E402

# ── Shared windowing (positional, residue-aligned offset 0) ──────────────────
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from windowing import (  # noqa: E402
    WINDOW_LEN,
    span_0idx,
    win_start_of,
    make_zero_window,
    make_impute_boundary_window,
    make_plddt_window,
    mean_plddt_real,
    find_peptide_in_structure,
    win_start_from_found,
)

# ── Constants ────────────────────────────────────────────────────────────────
EMB_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 3
DEFAULT_CSV = "data/processed/df_all.csv"
DEFAULT_PDB = "data/processed/structures/alphafold"
DEFAULT_OUTDIR = "data/processed/embeddings"
MODES = ["zero", "boundary"]
CKPT_DIR = _MPNN_DIR / "vanilla_model_weights"
# status legend
# 0=positional (structure_seq == df_seq), 1=string-search (isoform mismatch, peptide located by sequence),
# 2=missing_structure, 3=peptide_notfound, 4=parse_fail
ST_POSITIONAL, ST_STRINGSEARCH, ST_MISSING, ST_NOTFOUND, ST_PARSEFAIL = 0, 1, 2, 3, 4


def parse_args():
    p = argparse.ArgumentParser(description="ProteinMPNN encoder-only 29-mer window embeddings")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--pdb", default=DEFAULT_PDB)
    p.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    p.add_argument("--out-prefix", default=None,
                   help="Override output prefix (default: <out-dir>/mpnn_<ckpt>_context_embeddings)")
    p.add_argument("--checkpoint", default="v_48_020",
                   help="Checkpoint name in vanilla_model_weights/ (e.g. v_48_020) or a full .pt path")
    p.add_argument("--modes", default="zero,boundary", help="Comma-separated subset of zero,boundary")
    p.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs + model load only")
    p.add_argument("--n-debug", type=int, default=0, help="Process first N rows only")
    return p.parse_args()


def resolve_ckpt(spec: str) -> Path:
    if spec.endswith(".pt") and os.path.exists(spec):
        return Path(spec)
    cand = CKPT_DIR / f"{spec}.pt"
    if cand.exists():
        return cand
    raise SystemExit(f"Checkpoint not found: {spec} (looked at {cand})")


def load_model(ckpt_path: Path, device: torch.device) -> ProteinMPNN:
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    num_edges = checkpoint["num_edges"]
    noise = checkpoint.get("noise_level", "?")
    model = ProteinMPNN(
        ca_only=False, num_letters=21,
        node_features=HIDDEN_DIM, edge_features=HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS,
        augment_eps=0.0,  # no random rotation -> deterministic
        k_neighbors=num_edges,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, num_edges, noise


def get_plddt_backbone(path: str) -> np.ndarray | None:
    """CA B-factor (pLDDT) for residues with a COMPLETE backbone (N,CA,C,O),
    matching ProteinMPNN's residue selection so indices align with h_V."""
    try:
        st = PDBParser(QUIET=True).get_structure("x", path)
        chain = next(st.get_chains())
    except Exception:
        return None
    vals = []
    for res in chain:
        if res.id[0] != " ":
            continue
        if "N" in res and "CA" in res and "C" in res and "O" in res:
            vals.append(float(res["CA"].bfactor))
    if not vals:
        return None
    return np.asarray(vals, dtype=np.float32)


def encode_protein(model, pdb_path: str, device: torch.device):
    """Return (rep [L,128] residue-aligned, mpnn_seq str) or (None, None)."""
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    if not pdb_dict_list:
        return None, None
    protein = pdb_dict_list[0]
    chain_list = [k[-1:] for k in protein if k[:9] == "seq_chain"]
    chain_id_dict = {protein["name"]: (chain_list, [])}
    batch = [protein]
    X, S, mask, lengths, chain_M, chain_encoding_all, \
        chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
            batch, device, chain_id_dict, None, None, None, None, None, ca_only=False
        )
    with torch.no_grad():
        h_V = model.encode(X, mask, residue_idx, chain_encoding_all)  # [1, L, 128]
    rep = h_V[0].detach().cpu().numpy().astype(np.float32)  # [L, 128]
    mpnn_seq = str(protein["seq"])
    return rep, mpnn_seq


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",")]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"Unknown mode {m}; allowed {MODES}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = resolve_ckpt(args.checkpoint)
    ckpt_tag = ckpt_path.stem  # e.g. v_48_020
    prefix = args.out_prefix or str(Path(args.out_dir) / f"mpnn_{ckpt_tag}_context_embeddings")

    # no-overwrite guard
    if not args.dry_run:
        for m in modes:
            outp = Path(f"{prefix}_{m}.h5")
            if outp.exists():
                raise SystemExit(f"{outp} exists; will not overwrite.")

    # load CSV
    df = pd.read_csv(args.csv)
    required = ["peptide", "uniprot_id", "full_context", "start", "end", "sequence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")
    if args.n_debug > 0:
        df = df.iloc[: args.n_debug].reset_index(drop=True)
    lens = df["full_context"].astype(str).str.len()
    bad = int((lens != WINDOW_LEN).sum())
    if bad:
        raise SystemExit(f"{bad} rows have full_context != {WINDOW_LEN}")

    n_rows = len(df)
    groups = df.groupby("uniprot_id").indices
    print(f"rows={n_rows} proteins={len(groups)} device={device}")
    print(f"checkpoint={ckpt_path.name} modes={modes} out_prefix={prefix}")

    # load model
    print("Loading ProteinMPNN ...")
    model, num_edges, noise = load_model(ckpt_path, device)
    print(f"  loaded; num_edges(K)={num_edges} train_noise={noise}A")

    if args.dry_run:
        print("Dry run OK. Inputs + model validated.")
        return

    outdir = Path(prefix).parent
    outdir.mkdir(parents=True, exist_ok=True)
    h5s = {m: h5py.File(f"{prefix}_{m}.h5", "w") for m in modes}
    for m, f in h5s.items():
        f.create_dataset("peptide", data=df["peptide"].astype(str).tolist(), dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("uniprot_id", data=df["uniprot_id"].astype(str).tolist(), dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("row_indices", data=df.index.to_numpy(dtype=np.int64))
        f.attrs["model"] = f"proteinmpnn_{ckpt_tag}"
        f.attrs["emb_dim"] = EMB_DIM
        f.attrs["window_len"] = WINDOW_LEN
        f.attrs["pad_mode"] = m
        f.attrs["csv_source"] = str(args.csv)
        f.attrs["checkpoint"] = str(ckpt_path)
        f.attrs["train_noise_A"] = str(noise)
        f.attrs["k_neighbors"] = int(num_edges)
        f.attrs["encoder_only"] = "h_V after encoder layers; no decoder (no sequence leak)"
        f.attrs["plddt_source"] = "AFDB CA B-factor (complete-backbone residues)"
        f.attrs["status_legend"] = "0=positional 1=stringsearch 2=missing_structure 3=peptide_notfound 4=parse_fail"

    emb_ds = {m: h5s[m].create_dataset(
        "window_embeddings", shape=(n_rows, WINDOW_LEN, EMB_DIM), dtype="float32",
        chunks=(min(64, n_rows), WINDOW_LEN, EMB_DIM), compression="gzip", compression_opts=4
    ) for m in modes}
    pad_ds = {m: h5s[m].create_dataset("pad_mask", shape=(n_rows, WINDOW_LEN), dtype="uint8") for m in modes}
    plddt_ds = {m: h5s[m].create_dataset("window_plddt", shape=(n_rows, WINDOW_LEN), dtype="float32") for m in modes}
    fallback_ds = {m: h5s[m].create_dataset("fallback_flag", shape=(n_rows,), dtype="uint8") for m in modes}
    status_arr = np.zeros(n_rows, dtype=np.int8)

    FULL_PAD = np.ones(WINDOW_LEN, dtype=np.uint8)
    ZERO_WIN = np.zeros((WINDOW_LEN, EMB_DIM), dtype=np.float32)
    NAN_PLDDT = np.full(WINDOW_LEN, np.nan, dtype=np.float32)

    stats = {}  # row-level; computed from status_arr after the loop
    t0 = time.time()
    prot_n = 0

    for prot_id, row_idxs in groups.items():
        prot_n += 1
        pdb_path = Path(args.pdb) / f"{prot_id}.pdb"
        df_seq = str(df.iloc[row_idxs[0]]["sequence"])

        rep = plddt = None
        mpnn_seq = None
        seq_matched = False
        prot_status = ST_MISSING
        if pdb_path.exists():
            rep, mpnn_seq = encode_protein(model, str(pdb_path), device)
            if rep is None:
                prot_status = ST_PARSEFAIL
            else:
                seq_matched = (mpnn_seq == df_seq and len(mpnn_seq) == rep.shape[0])
                plddt = get_plddt_backbone(str(pdb_path))
                if plddt is not None and len(plddt) != rep.shape[0]:
                    plddt = None  # safety; shouldn't happen on AFDB
                prot_status = ST_POSITIONAL if seq_matched else ST_STRINGSEARCH
        else:
            prot_status = ST_MISSING

        L = rep.shape[0] if rep is not None else 0
        for ridx in row_idxs:
            row = df.iloc[ridx]

            if rep is None:
                # missing structure or parse failure: zero-fill every row of this protein
                status_arr[ridx] = prot_status
                for m in modes:
                    emb_ds[m][ridx] = ZERO_WIN
                    pad_ds[m][ridx] = FULL_PAD
                    plddt_ds[m][ridx] = NAN_PLDDT
                    fallback_ds[m][ridx] = 1
                continue

            if seq_matched:
                # positional: structure_seq == df_seq, so coordinate window is valid
                ws = win_start_of(row)
                status_arr[ridx] = ST_POSITIONAL
            else:
                # isoform mismatch: locate the peptide by sequence (ESM-IF logic)
                result = find_peptide_in_structure(mpnn_seq, L, row)
                if result is None:
                    # peptide genuinely absent from this (mismatched) structure
                    status_arr[ridx] = ST_NOTFOUND
                    for m in modes:
                        emb_ds[m][ridx] = ZERO_WIN
                        pad_ds[m][ridx] = FULL_PAD
                        plddt_ds[m][ridx] = NAN_PLDDT
                        fallback_ds[m][ridx] = 1
                    continue
                ws = win_start_from_found(result)
                status_arr[ridx] = ST_STRINGSEARCH

            pw, pmask = make_plddt_window(plddt, ws, L) if plddt is not None else (NAN_PLDDT, None)
            w0, pm0 = make_zero_window(rep, ws, L)
            wb, pmb = make_impute_boundary_window(rep, ws, L)
            wins = {"zero": (w0, pm0), "boundary": (wb, pmb)}
            for m in modes:
                w, pm = wins[m]
                emb_ds[m][ridx] = w
                pad_ds[m][ridx] = pm.astype(np.uint8)
                plddt_ds[m][ridx] = pw
                fallback_ds[m][ridx] = 0

        del rep, plddt
        if prot_n % 100 == 0:
            elapsed = time.time() - t0
            rate = prot_n / elapsed
            eta = (len(groups) - prot_n) / rate / 60
            print(f"  [{prot_n}/{len(groups)}] {rate:.2f} prot/s ETA {eta:.0f}min")
            for f in h5s.values():
                f.flush()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for m, f in h5s.items():
        f.create_dataset("status", data=status_arr)
        f.attrs["n_rows"] = n_rows
        f.attrs["total_time_sec"] = time.time() - t0
        f.close()

    # row-level status breakdown
    _label = {ST_POSITIONAL: "positional", ST_STRINGSEARCH: "stringsearch",
              ST_MISSING: "missing_structure", ST_NOTFOUND: "peptide_notfound",
              ST_PARSEFAIL: "parse_fail"}
    u, c = np.unique(status_arr, return_counts=True)
    stats = {_label.get(int(k), str(k)): int(v) for k, v in zip(u, c)}

    print(f"\nDone. {stats}")
    print(f"Files: {prefix}_{{{','.join(modes)}}}.h5")


if __name__ == "__main__":
    main()
