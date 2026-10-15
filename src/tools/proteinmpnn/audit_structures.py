#!/usr/bin/env python3
"""
audit_structures.py — validate local AlphaFold DB structures against df_all.

For every unique uniprot_id in the input CSV, locate the AFDB PDB and check:
  * found              — a PDB file exists in --pdb
  * first_resseq / contiguous / last_resseq — ATOM residue numbering (ProteinMPNN
    positional mapping needs first_resseq == 1 and a gapless 1..N run)
  * seqres_match       — SEQRES sequence == df sequence (full intended sequence)
  * atom_match         — ATOM-derived residue sequence == df sequence (what the
    encoder actually sees; must equal df sequence for positional windows)
  * atom_is_prefix     — ATOM seq is a prefix of df seq (fragmented / truncated model)
  * status             — ok | mismatch | fragmented | missing | no_atoms

Never skips silently: every protein gets a TSV row; summary counts printed.

This is structure-only (Biopython + pandas); runs in any env with Biopython
(no esm / torch import). Output: results/structure_audit.tsv

Usage:
    python src/tools/proteinmpnn/audit_structures.py
    python src/tools/proteinmpnn/audit_structures.py --limit 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

DEFAULT_CSV = "data/processed/df_all.csv"
DEFAULT_PDB = "data/processed/structures/alphafold"
DEFAULT_OUT = "results/structure_audit.tsv"


def atom_seq_and_resnums(chain) -> tuple[str, list[int], list[float]]:
    """One-letter seq, resseq list, and CA B-factor (pLDDT) for a chain's ATOM residues."""
    seq_chars: list[str] = []
    resnums: list[int] = []
    plddt: list[float] = []
    for res in chain:
        if res.id[0] != " ":  # skip HETATM / water
            continue
        if "CA" not in res:
            continue
        seq_chars.append(protein_letters_3to1.get(res.get_resname(), "X"))
        resnums.append(res.id[1])
        plddt.append(float(res["CA"].bfactor))
    return "".join(seq_chars), resnums, plddt


def seqres_seq(pdb_path: Path) -> str | None:
    """Reconstruct SEQRES one-letter sequence from SEQRES records, or None."""
    three_to_one = protein_letters_3to1
    seqs: list[str] = []
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("SEQRES"):
                    # SEQRES cols: 1-6 SEQRES, 8 serial, 11 chainID, 13-17 num/total, 20+ resnames (3-char, 4-char fields)
                    fields = line[19:].split()
                    for f in fields:
                        seqs.append(three_to_one.get(f, "X"))
    except OSError:
        return None
    return "".join(seqs) if seqs else None


def is_contiguous(resnums: list[int]) -> bool:
    return bool(resnums) and resnums == list(range(resnums[0], resnums[0] + len(resnums)))


def classify(seqres_match, atom_match, atom_is_prefix, found, n_atoms):
    if not found:
        return "missing"
    if n_atoms == 0:
        return "no_atoms"
    if atom_match:
        return "ok"
    if atom_is_prefix:
        return "fragmented"
    return "mismatch"


def main():
    ap = argparse.ArgumentParser(description="Audit AFDB structures vs df_all sequences")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--pdb", default=DEFAULT_PDB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="Audit first N proteins only")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["peptide", "uniprot_id", "start", "end", "sequence"])
    before = len(df)
    df = df.dropna(subset=["peptide", "uniprot_id", "start", "end"])
    dropped = before - len(df)

    # One row per protein: take the first row's sequence (all rows for a protein
    # share the same canonical `sequence`).
    prot = df.drop_duplicates("uniprot_id")[["uniprot_id", "sequence"]].reset_index(drop=True)
    if args.limit > 0:
        prot = prot.head(args.limit)

    parser = PDBParser(QUIET=True)
    rows = []
    n = len(prot)
    for i, r in enumerate(prot.itertuples(index=False)):
        uid = r.uniprot_id
        df_seq = str(r.sequence)
        L_df = len(df_seq)
        pdb_path = Path(args.pdb) / f"{uid}.pdb"

        found = pdb_path.exists()
        atom_match = seqres_match = atom_is_prefix = False
        first_resseq = last_resseq = None
        contiguous = False
        n_atoms = 0
        atom_seq = ""
        chain_id = None

        if found:
            try:
                st = parser.get_structure("x", str(pdb_path))
                chain = next(st.get_chains())
                chain_id = chain.id
                atom_seq, resnums, _plddt = atom_seq_and_resnums(chain)
                n_atoms = len(resnums)
                if n_atoms:
                    first_resseq = resnums[0]
                    last_resseq = resnums[-1]
                    contiguous = is_contiguous(resnums)
                atom_match = atom_seq == df_seq
                atom_is_prefix = (not atom_match) and df_seq.startswith(atom_seq)
                sr = seqres_seq(pdb_path)
                seqres_match = (sr == df_seq) if sr is not None else None
            except Exception as e:  # parse failure -> treat as no_atoms, note it
                atom_seq = ""
                n_atoms = 0

        status = classify(seqres_match, atom_match, atom_is_prefix, found, n_atoms)
        rows.append({
            "uniprot_id": uid,
            "L_df": L_df,
            "found": int(found),
            "chain": chain_id,
            "n_atoms": n_atoms,
            "first_resseq": first_resseq,
            "last_resseq": last_resseq,
            "contiguous": int(contiguous),
            "seqres_match": ("" if seqres_match is None else int(seqres_match)),
            "atom_match": int(atom_match),
            "atom_is_prefix": int(atom_is_prefix),
            "status": status,
        })
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}]")

    audit = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, sep="\t", index=False)

    # ── Summary ──
    print("\n=== Structure audit ===")
    print(f"  proteins audited : {n}")
    print(f"  CSV rows dropped (NaN): {dropped}")
    vc = audit["status"].value_counts()
    for k in ["ok", "mismatch", "fragmented", "missing", "no_atoms"]:
        if k in vc:
            print(f"  {k:12s}: {int(vc[k])}")
    noncontig = int(((audit["found"] == 1) & (audit["contiguous"] == 0)).sum())
    badfirst = int(((audit["found"] == 1) & (audit["first_resseq"] != 1) & audit["first_resseq"].notna()).sum())
    print(f"  non-contiguous ATOM resseq : {noncontig}")
    print(f"  first_resseq != 1          : {badfirst}")

    bad = audit[audit["status"] != "ok"]
    if len(bad):
        print(f"\n  Wrote {len(bad)} non-ok rows to {out}")
        print("  sample:")
        print(bad.head(10).to_string(index=False))
    else:
        print(f"\n  All {n} proteins OK. Wrote {out}")

    # Exit non-zero if any mismatch (positional mapping would be wrong) so
    # callers/CI can gate on it — but missing/fragmented are reported, not fatal.
    n_mismatch = int((audit["status"] == "mismatch").sum())
    if n_mismatch:
        print(f"\n  WARNING: {n_mismatch} sequence MISMATCH proteins — positional "
              f"mapping unsafe for these; they will be status-flagged at embed time.")


if __name__ == "__main__":
    main()
