#!/usr/bin/env python3
"""
extract_window_plddt.py — windowed per-residue pLDDT sidecar for the EXISTING
ESM-IF window embeddings.

The ESM-IF pipeline (embed_peptides_esmif.py) already produced 4 pad-mode window
files but did NOT store pLDDT. This script recomputes the SAME peptide→structure
mapping (importing find_peptide_in_structure / find_structure_file / get_chain_ids
from that module, so windows align slot-for-slot) and writes a single sidecar HDF5
with the windowed pLDDT vector [N,29] (NaN at pad slots) plus the shared pad_mask.

Rationale for one shared file: the pad_mask is IDENTICAL across the 4 ESM-IF mode
files (built once per peptide, written to all four), so one sidecar serves all of
them.

Per-residue pLDDT = CA atom B-factor in the AFDB model. Sequence + pLDDT are read
from the SAME Bio.PDB pass (consistent residue ordering); the chain is chosen by
the same peptide-containment rule the embedder used (replicated here WITHOUT
encoding, since load_structure_best_chain in the embedder requires the model).

Structure-only + string-matching; no torch/esm model load. Needs Biopython.
Output: data/processed/embeddings/esmif_window_plddt.h5

Usage:
    python src/tools/esm/extract_window_plddt.py
    python src/tools/esm/extract_window_plddt.py --limit 200 --out /tmp/plddt_test.h5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser

# Import the ESM-IF mapping helpers (module top-level imports esm, but main()
# is not triggered, so the model is never loaded).
_TOOLS_ESM = os.path.join(os.path.dirname(__file__))
if _TOOLS_ESM not in sys.path:
    sys.path.insert(0, _TOOLS_ESM)
from embed_peptides_esmif import (  # noqa: E402
    find_structure_file,
    get_chain_ids,
    find_peptide_in_structure,
)

WINDOW_LEN = 29
FLANK = 10
DEFAULT_CSV = "data/processed/df_all.csv"
DEFAULT_PDB = "data/processed/structures/alphafold"
DEFAULT_OUT = "data/processed/embeddings/esmif_window_plddt.h5"

_parser = PDBParser(QUIET=True)


def load_chain_seq_plddt(path: str, chain_id: str):
    """Return (one_letter_seq, plddt[N]) for a chain's ATOM CA residues, or (None,None)."""
    try:
        st = _parser.get_structure("x", path)
        chain = st[0][chain_id]
    except Exception:
        return None, None
    from Bio.PDB.Polypeptide import protein_letters_3to1
    chars, plddt = [], []
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        chars.append(protein_letters_3to1.get(res.get_resname(), "X"))
        plddt.append(float(res["CA"].bfactor))
    if not chars:
        return None, None
    return "".join(chars), np.asarray(plddt, dtype=np.float32)


def pick_chain_seq_plddt(path: str, uid: str, peptide_set: set):
    """Replicate embed_peptides_esmif.load_structure_best_chain's chain choice,
    but returning (seq, plddt) without any encoding. Prefers chain A if it
    contains >=1 peptide, else the chain with the most peptide hits, else A."""
    # try A first
    seqA, plddtA = load_chain_seq_plddt(path, "A")
    if seqA is not None and any(p in seqA for p in peptide_set):
        return seqA, plddtA, "A"
    best = None
    for cid in get_chain_ids(path):
        if cid == "A":
            continue
        seq, plddt = load_chain_seq_plddt(path, cid)
        if seq is None:
            continue
        hits = sum(1 for p in peptide_set if p in seq)
        if hits > 0 and (best is None or hits > best[3]):
            best = (seq, plddt, cid, hits)
    if best is not None:
        return best[0], best[1], best[2]
    if seqA is not None:
        return seqA, plddtA, "A"
    return None, None, None


def plddt_window_from_mapping(plddt: np.ndarray, ns: int, ps: int, pe: int, ce: int):
    """Build [29] pLDDT window + pad_mask from the ESM-IF mapping (ns,ps,pe,ce),
    mirroring embed_peptides_esmif.build_windows' slot layout:
        n_real = ps-ns, c_real = ce-pe
        slots [10-n_real:10] <- plddt[ns:ps]
        slots [10:19]        <- plddt[ps:pe]
        slots [19:19+c_real] <- plddt[pe:ce]
        pads = [0:10-n_real] and [19+c_real:]  (NaN)
    """
    n_real = ps - ns
    c_real = ce - pe
    w = np.full(WINDOW_LEN, np.nan, dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    if n_real > 0:
        w[FLANK - n_real:FLANK] = plddt[ns:ps]
    w[FLANK:FLANK + 9] = plddt[ps:pe]
    if c_real > 0:
        w[FLANK + 9:FLANK + 9 + c_real] = plddt[pe:ce]
    n_pad = FLANK - n_real
    c_pad = FLANK - c_real
    if n_pad > 0:
        pm[0:n_pad] = True
    if c_pad > 0:
        pm[WINDOW_LEN - c_pad:] = True
    return w, pm


def main():
    ap = argparse.ArgumentParser(description="Windowed pLDDT sidecar for ESM-IF windows")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--pdb", default=DEFAULT_PDB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="First N CSV rows only (debug)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["_csv_row"] = np.arange(len(df))
    before = len(df)
    df = df.dropna(subset=["peptide", "uniprot_id", "start", "end"])
    df = df.sort_values("uniprot_id").reset_index(drop=True)
    df["_pos"] = np.arange(len(df))
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        df["_pos"] = np.arange(len(df))
        df["_csv_row"] = df["_csv_row"].astype(np.int64)
    n_rows = len(df)
    print(f"Rows: {n_rows} (dropped {before - len(df)} NaN pre-limit); "
          f"proteins={df['uniprot_id'].nunique()}")

    uid_peptides = df.groupby("uniprot_id")["peptide"].apply(set).to_dict()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; will not overwrite. Delete it first.")

    f = h5py.File(args.out, "w")
    f.create_dataset("window_plddt", shape=(n_rows, WINDOW_LEN), dtype="float32",
                     chunks=(min(64, n_rows), WINDOW_LEN), compression="gzip",
                     compression_opts=4)
    f.create_dataset("pad_mask", shape=(n_rows, WINDOW_LEN), dtype="uint8")
    row_indices = np.zeros(n_rows, dtype=np.int64)
    statuses = np.zeros(n_rows, dtype=np.int8)  # 0 ok, 2 missing structure, 3 notfound
    uniprot_ids = np.zeros(n_rows, dtype="S20")

    stats = {"ok": 0, "missing": 0, "notfound": 0}
    for gi, (uid, group) in enumerate(df.groupby("uniprot_id", sort=True)):
        struct_path = find_structure_file(uid, args.pdb)
        seq = plddt = None
        if struct_path:
            seq, plddt, _chain = pick_chain_seq_plddt(struct_path, uid, uid_peptides.get(uid, set()))
        for j, (_, row) in enumerate(group.iterrows()):
            oi = row["_pos"]
            row_indices[oi] = row["_csv_row"]
            uniprot_ids[oi] = str(uid).encode()
            if seq is None:
                statuses[oi] = 2
                stats["missing"] += 1
                f["pad_mask"][oi] = np.ones(WINDOW_LEN, dtype=np.uint8)
                continue  # window_plddt stays NaN (preallocated)
            L_rep = len(seq)
            result = find_peptide_in_structure(seq, L_rep, row)
            if result is None:
                statuses[oi] = 3
                stats["notfound"] += 1
                f["pad_mask"][oi] = np.ones(WINDOW_LEN, dtype=np.uint8)
                continue
            ps, pe, ns, ce, _match = result
            w, pm = plddt_window_from_mapping(plddt, ns, ps, pe, ce)
            f["window_plddt"][oi] = w
            f["pad_mask"][oi] = pm.astype(np.uint8)
            statuses[oi] = 0
            stats["ok"] += 1
        if (gi + 1) % 500 == 0:
            f.flush()
            print(f"  [{gi+1}/{df['uniprot_id'].nunique()}] {uid}")

    f.create_dataset("row_indices", data=row_indices)
    f.create_dataset("status", data=statuses)
    f.create_dataset("uniprot_ids", data=uniprot_ids)
    f.attrs["source_model_windows"] = "esm-if_test_{zero,pad,boundary,eos_bos_repeat}.h5"
    f.attrs["mapping"] = "find_peptide_in_structure (imported from embed_peptides_esmif)"
    f.attrs["plddt_source"] = "AFDB CA atom B-factor"
    f.attrs["window_len"] = WINDOW_LEN
    f.attrs["pad_mask_note"] = "identical across all 4 ESM-IF mode files; True=pad"
    f.attrs["plddt_pad_note"] = "pad slots are NaN; average over finite slots only"
    f.attrs["status_legend"] = "0=ok 2=missing_structure 3=peptide_not_found"
    f.attrs["n_rows"] = n_rows
    f.close()

    print(f"\nDone. {stats}")
    print(f"File: {args.out}")


if __name__ == "__main__":
    main()
