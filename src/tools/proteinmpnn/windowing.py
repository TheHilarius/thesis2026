#!/usr/bin/env python3
"""
windowing.py — shared fixed-29 window slot logic for structure-derived embeddings.

Convention (identical to the ESM-C script embed_peptides_esmc.py):
  * peptide coords in df_all are 1-indexed inclusive  (start, end)
  * start_0 = start - 1                       (0-indexed inclusive)
  * win_start = start_0 - 10                  (10 N-flank slots before peptide)
  * WINDOW_LEN = 29 = 10 N-flank + 9 peptide + 10 C-flank
  * slot i (0..28) maps to protein residue index  win_start + i  (0-indexed)
  * slots where win_start + i is outside [0, L) are PADS

Difference from the ESM-C script's own make_*_window functions: those index the
embedding as emb_full[j + 1] because ESM-C embeddings carry a leading <cls>/BOS
row at index 0. Structure-model encoder outputs (ESM-IF stripped reps, ProteinMPNN
h_V) are RESIDUE-ALIGNED — index 0 is the first residue, no special tokens. So the
builders here index emb[j] directly (offset 0).

These builders are what the ProteinMPNN embedder (and the structure audit) use.
The existing ESM-IF pipeline built its windows by STRING search (find_peptide_in_
structure), not positional mapping, so it does NOT use this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_LEN = 29
FLANK = 10  # N-flank slots before the 9-mer peptide; also C-flank slots after


def span_0idx(row: pd.Series) -> tuple[int, int]:
    """Return (start_0, end_0_exclusive) for the peptide in its protein.

    df_all coords are 1-indexed inclusive (verified by
    src/investigation/12_indexing_audit.py).
    """
    s, e = int(row["start"]), int(row["end"])
    return (s - 1, e)


def win_start_of(row: pd.Series) -> int:
    """0-indexed protein position of window slot 0 (may be negative on overhang)."""
    start_0, _ = span_0idx(row)
    return start_0 - FLANK


def pad_mask_for(win_start: int, L: int) -> np.ndarray:
    """Boolean pad mask [29]; True = PADDED slot (outside the protein)."""
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    for i in range(WINDOW_LEN):
        j = win_start + i
        if not (0 <= j < L):
            pm[i] = True
    return pm


def make_zero_window(emb: np.ndarray, win_start: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-fill pad slots. emb is residue-aligned [L, D]. Returns (window [29,D], pad_mask [29])."""
    D = emb.shape[1]
    w = np.zeros((WINDOW_LEN, D), dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    for i in range(WINDOW_LEN):
        j = win_start + i
        if 0 <= j < L:
            w[i] = emb[j]
        else:
            pm[i] = True
    return w, pm


def make_impute_boundary_window(emb: np.ndarray, win_start: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Repeat nearest real boundary residue into pad slots (terminus fallback =
    the peptide's own boundary residue, matching ESM-C impute_boundary).

    emb is residue-aligned [L, D]. Returns (window [29,D], pad_mask [29])."""
    D = emb.shape[1]
    w = np.zeros((WINDOW_LEN, D), dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    valid: list[int] = []
    for i in range(WINDOW_LEN):
        j = win_start + i
        if 0 <= j < L:
            w[i] = emb[j]
            valid.append(i)
        else:
            pm[i] = True
    if valid:
        for i in range(valid[0]):
            w[i] = w[valid[0]]
        for i in range(valid[-1] + 1, WINDOW_LEN):
            w[i] = w[valid[-1]]
    return w, pm


def make_plddt_window(plddt: np.ndarray, win_start: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Window a per-residue pLDDT vector [L] into [29]. Pad slots = NaN.

    Returns (plddt_window [29] float32, pad_mask [29]). pLDDT is scalar per
    residue so no boundary imputation is meaningful here — pads stay NaN so the
    consumer can average over real slots only."""
    w = np.full(WINDOW_LEN, np.nan, dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    for i in range(WINDOW_LEN):
        j = win_start + i
        if 0 <= j < L:
            w[i] = plddt[j]
        else:
            pm[i] = True
    return w, pm


def mean_plddt_real(plddt_window: np.ndarray) -> float:
    """Mean pLDDT over non-NaN (real) slots; NaN if all slots are pad."""
    real = plddt_window[np.isfinite(plddt_window)]
    return float(real.mean()) if real.size else float("nan")


# ── Peptide locator (isoform-mismatch fallback) ─────────────────────────────
# Copied VERBATIM from src/tools/esm/embed_peptides_esmif.py (pure string logic,
# no esm/torch import) so this module stays usable in env_mpnn, which must NOT
# have fair-esm installed. Used to locate the peptide in a structure whose
# sequence differs from df `sequence` (the 13 isoform-mismatch proteins).
# Returns the SAME (ps, pe, ns, ce, match) tuple as the ESM-IF embedder so the
# mapping is slot-for-slot identical to what ESM-IF produced.

def _get_flank(value):
    if pd.isna(value):
        return ''
    s = str(value).strip()
    return '' if s.lower() == 'nan' else s


def _find_all(sequence, query):
    positions, start = [], 0
    while True:
        idx = sequence.find(query, start)
        if idx == -1:
            return positions
        positions.append(idx)
        start = idx + 1


def _pick_closest(positions, target_idx):
    return min(positions, key=lambda p: abs(p - target_idx))


def find_peptide_in_structure(pdb_seq, rep_len, row):
    """Locate peptide/flanks in a structure sequence (0-indexed, half-open)."""
    peptide = str(row['peptide'])
    n_flank = _get_flank(row.get('n_flank'))
    c_flank = _get_flank(row.get('c_flank'))
    uniprot_start_0 = int(row['start']) - 1
    n_len, c_len = len(n_flank), len(c_flank)

    full_window = n_flank + peptide + c_flank
    if len(full_window) > len(peptide):
        positions = _find_all(pdb_seq, full_window)
        if len(positions) == 1:
            ws = positions[0]
            return (ws + n_len, ws + n_len + len(peptide), ws,
                    min(rep_len, ws + len(full_window)), "full_window_unique")
        elif len(positions) > 1:
            ws = _pick_closest(positions, max(0, uniprot_start_0 - n_len))
            return (ws + n_len, ws + n_len + len(peptide), ws,
                    min(rep_len, ws + len(full_window)), "full_window_positional")

    positions = _find_all(pdb_seq, peptide)
    if len(positions) == 0:
        return None
    if len(positions) == 1:
        ps = positions[0]
        match = "peptide_unique"
    else:
        ps = _pick_closest(positions, uniprot_start_0)
        match = "peptide_positional"

    return (ps, ps + len(peptide), max(0, ps - n_len),
            min(rep_len, ps + len(peptide) + c_len), match)


def win_start_from_found(result, flank: int = FLANK) -> int:
    """Derive the window start (win_start for the positional builders) from a
    find_peptide_in_structure result. result[0] = peptide start (ps) in the
    structure; the canonical window is ps-flank .. ps+9+flank. Reusing the
    positional builders with this win_start gives a slot layout identical to
    ESM-IF's mapping."""
    ps = result[0]
    return ps - flank
