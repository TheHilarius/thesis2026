#!/usr/bin/env python3
"""
embed_peptides_full_protein.py

Embeds FULL protein sequences with ESM-C, then extracts mean-pooled
residue embeddings for the peptide, n-flank and c-flank regions using
start/end coordinates within the protein.

Each unique protein is embedded once; every peptide from that protein
is then extracted from the shared protein embedding.

If a protein causes OOM, a window centred on each peptide is used
as automatic fallback (flagged in output).

Usage:
    python src/tools/esm/embed_peptides_w_esm.py \
        data/processed/df_all.csv \
        data/processed/embeddings/esmc_protein_embeddings.h5 \
        --model esmc_600m
"""

import argparse
import gc
import time
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Embed full proteins with ESM-C; extract peptide/flank embeddings"
)
parser.add_argument("csv_path", help="Input CSV (e.g. data/processed/df_all.csv)")
parser.add_argument("out_h5",   help="Output HDF5 file")
parser.add_argument("--model",  default="esmc_600m",
                    choices=["esmc_300m", "esmc_600m"])
parser.add_argument("--max_seq_len", type=int, default=None,
                    help="Force windowed fallback above this length. "
                         "Default: None (try full protein, fall back on OOM)")
parser.add_argument("--fallback_window", type=int, default=2048,
                    help="Window size for fallback when full protein fails/too long")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  ESM-C Full-Protein Peptide Embedding")
print("=" * 65)
print(f"{'Model':<20}: {args.model}")
print(f"{'Device':<20}: {DEVICE}")
print(f"{'Max seq length':<20}: {args.max_seq_len or 'None (try all, OOM fallback)'}")
print(f"{'Fallback window':<20}: {args.fallback_window}")
print(f"{'Input CSV':<20}: {args.csv_path}")
print(f"{'Output HDF5':<20}: {args.out_h5}")
print()

# ── Column names (matched to df_all.csv) ─────────────────────────────────────
COL_PROTEIN = "sequence"
COL_START   = "start"
COL_END     = "end"
COL_PEPTIDE = "peptide"
COL_UNIPROT = "uniprot_id"
COL_NFLANK  = "n_flank"
COL_CFLANK  = "c_flank"

# ── Load ESM-C ────────────────────────────────────────────────────────────────
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

print("Loading ESM-C...")
t0 = time.time()
client = ESMC.from_pretrained(args.model).to(DEVICE)
client.eval()
print(f"ESM-C loaded ✓  ({time.time() - t0:.1f}s)")

EMB_DIM = 960 if args.model == "esmc_300m" else 1152
print(f"{'Embedding dim':<20}: {EMB_DIM}")

# ── Load CSV ──────────────────────────────────────────────────────────────────
print("\nLoading CSV...")
df = pd.read_csv(args.csv_path)
print(f"Total rows: {len(df)}")

required = [COL_PEPTIDE, COL_PROTEIN, COL_START, COL_END, COL_UNIPROT]
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

before = len(df)
df = df.dropna(subset=required).reset_index(drop=True)
print(f"After dropping NaN in required cols: {len(df)}  (dropped {before - len(df)})")

# ── Detect coordinate convention ──────────────────────────────────────────────
print("\nDetecting coordinate convention (checking first 200 rows)...")
n_check = min(200, len(df))
n_0idx = n_1idx_inclusive = 0

for i in range(n_check):
    row  = df.iloc[i]
    prot = str(row[COL_PROTEIN])
    pep  = str(row[COL_PEPTIDE])
    s, e = int(row[COL_START]), int(row[COL_END])

    # Convention A: 0-indexed, half-open [s, e)
    if 0 <= s < e <= len(prot) and prot[s:e] == pep:
        n_0idx += 1

    # Convention B: 1-indexed inclusive [s, e] → 0-indexed [s-1, e)
    if s >= 1 and e <= len(prot) and prot[s-1 : e] == pep:
        n_1idx_inclusive += 1

print(f"  0-indexed [s, e) matches       : {n_0idx}/{n_check}")
print(f"  1-indexed inclusive [s, e] hits : {n_1idx_inclusive}/{n_check}")

if n_1idx_inclusive >= n_0idx and n_1idx_inclusive > n_check * 0.8:
    CONVENTION = "1idx_inclusive"
    print("  → Using 1-indexed inclusive convention")
elif n_0idx > n_1idx_inclusive and n_0idx > n_check * 0.8:
    CONVENTION = "0idx_halfopen"
    print("  → Using 0-indexed half-open convention")
else:
    raise RuntimeError(
        f"Cannot reliably detect coordinate convention "
        f"(0idx: {n_0idx}, 1idx_inclusive: {n_1idx_inclusive} / {n_check}). "
        f"Check start/end columns and the protein sequence column."
    )


def get_peptide_span_0idx(row):
    """Return (start_0, end_0_exclusive) for the peptide in the protein."""
    s = int(row[COL_START])
    e = int(row[COL_END])
    if CONVENTION == "1idx_inclusive":
        return s - 1, e          # prot[s-1 : e]
    else:
        return s, e              # prot[s : e]


# ── Validate ALL coordinates ─────────────────────────────────────────────────
print("\nValidating peptide coordinates against protein sequences...")
mismatches = 0
for i in range(len(df)):
    row  = df.iloc[i]
    prot = str(row[COL_PROTEIN])
    pep  = str(row[COL_PEPTIDE])
    s0, e0 = get_peptide_span_0idx(row)
    if s0 < 0 or e0 > len(prot) or prot[s0:e0] != pep:
        mismatches += 1
        if mismatches <= 5:
            print(f"  MISMATCH row {i}: prot[{s0}:{e0}]='{prot[s0:e0][:20]}' "
                  f"!= peptide='{pep}' (prot_len={len(prot)})")

if mismatches:
    print(f"  ⚠  {mismatches}/{len(df)} coordinate mismatches — check your data!")
else:
    print(f"  All {len(df)} coordinates validated ✓")

# ── Validate flank consistency ────────────────────────────────────────────────
print("\nValidating flanks against protein sequence (first 500 rows)...")
flank_mismatches = 0
n_check_flank = min(500, len(df))
for i in range(n_check_flank):
    row  = df.iloc[i]
    prot = str(row[COL_PROTEIN])
    s0, e0 = get_peptide_span_0idx(row)

    nf = str(row[COL_NFLANK]) if pd.notna(row[COL_NFLANK]) else ""
    cf = str(row[COL_CFLANK]) if pd.notna(row[COL_CFLANK]) else ""

    expected_nf = prot[s0 - len(nf) : s0] if len(nf) > 0 else ""
    expected_cf = prot[e0 : e0 + len(cf)] if len(cf) > 0 else ""

    if nf != expected_nf or cf != expected_cf:
        flank_mismatches += 1
        if flank_mismatches <= 3:
            print(f"  row {i}: n_flank expected='{expected_nf}' got='{nf}' | "
                  f"c_flank expected='{expected_cf}' got='{cf}'")

if flank_mismatches:
    print(f"  ⚠  {flank_mismatches}/{n_check_flank} flank mismatches")
else:
    print(f"  All {n_check_flank} flank checks passed ✓")

# ── Check BOS/EOS offset ─────────────────────────────────────────────────────
print("\nVerifying ESM-C BOS/EOS offset...")
with torch.no_grad():
    _test_seq = "ACDEFGHIKLMNPQRST"
    _test_out = client.logits(
        client.encode(ESMProtein(sequence=_test_seq)),
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    _test_emb = _test_out.embeddings.squeeze(0)
    _offset = _test_emb.shape[0] - len(_test_seq)
    print(f"  Sequence length  : {len(_test_seq)}")
    print(f"  Embedding length : {_test_emb.shape[0]}")
    print(f"  Offset (BOS+EOS) : {_offset}")
    assert _offset == 2, f"Unexpected offset {_offset} — script assumes BOS + EOS = 2"
    del _test_out, _test_emb

# ── Group peptides by protein ─────────────────────────────────────────────────
print("\nGrouping peptides by protein...")
protein_groups = df.groupby(COL_UNIPROT).indices
n_proteins = len(protein_groups)
print(f"  {len(df)} peptides from {n_proteins} unique proteins")
print(f"  Avg peptides/protein : {len(df)/n_proteins:.1f}")

prot_lens = df.groupby(COL_UNIPROT)[COL_PROTEIN].first().str.len()
print(f"  Protein lengths      : "
      f"min={prot_lens.min()}, median={int(prot_lens.median())}, "
      f"max={prot_lens.max()}")

if args.max_seq_len is not None:
    n_long = (prot_lens > args.max_seq_len).sum()
    print(f"  Proteins > {args.max_seq_len} aa  : {n_long} → forced windowed fallback")

# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_sequence(seq: str) -> np.ndarray:
    """Embed a sequence → residue embeddings [L, D] with BOS/EOS stripped."""
    protein = ESMProtein(sequence=seq)
    tensor  = client.encode(protein)
    out     = client.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
    emb     = out.embeddings.squeeze(0).cpu().numpy()   # [L+2, D]
    return emb[1:-1]                                     # [L, D]


def try_embed_full(seq: str):
    """Try embedding the full sequence. Returns embeddings or None on OOM."""
    try:
        emb = embed_sequence(seq)
        return emb
    except (RuntimeError, MemoryError) as e:
        err_str = str(e).lower()
        if "out of memory" in err_str or "memoryerror" in err_str or isinstance(e, MemoryError):
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return None
        raise


def embed_window(prot_seq: str, s0: int, e0: int, nf_len: int, cf_len: int):
    """Embed a window centred on the peptide. Returns (residue_emb, win_start)."""
    prot_len  = len(prot_seq)
    pep_mid   = (s0 + e0) // 2
    win_half  = args.fallback_window // 2
    win_start = max(0, pep_mid - win_half)
    win_end   = min(prot_len, win_start + args.fallback_window)
    win_start = max(0, win_end - args.fallback_window)

    win_seq     = prot_seq[win_start:win_end]
    residue_emb = embed_sequence(win_seq)
    return residue_emb, win_start


def mean_pool(emb_slice: np.ndarray) -> np.ndarray:
    """Mean-pool [N, D] → [D]. Returns zeros if N == 0."""
    if emb_slice.shape[0] == 0:
        return np.zeros(EMB_DIM, dtype=np.float32)
    return emb_slice.mean(axis=0).astype(np.float32)


def get_flank_len(row, col):
    """Get flank length, handling NaN / empty."""
    val = row[col]
    if pd.isna(val):
        return 0
    s = str(val)
    if s in ("", "nan"):
        return 0
    return len(s)


# ── Main embedding loop ──────────────────────────────────────────────────────
print(f"\nEmbedding {n_proteins} proteins...")
print(f"(On CPU this will be slow — expect ~1-5 sec per protein)\n")

peptide_embs   = np.zeros((len(df), EMB_DIM), dtype=np.float32)
n_flank_embs   = np.zeros((len(df), EMB_DIM), dtype=np.float32)
c_flank_embs   = np.zeros((len(df), EMB_DIM), dtype=np.float32)
fallback_flags = np.zeros(len(df), dtype=np.int8)

n_done       = 0
n_fallback   = 0
oom_proteins = set()
t_start      = time.time()

with torch.no_grad():
    for prot_id, row_idxs in protein_groups.items():

        # ── Progress ──────────────────────────────────────────────────────
        if n_done % 50 == 0:
            elapsed = time.time() - t_start
            rate    = n_done / elapsed if elapsed > 0 else 0
            eta     = (n_proteins - n_done) / rate if rate > 0 else float('inf')
            eta_str = f"{eta/60:.0f}min" if eta < float('inf') else "?"
            print(f"  [{n_done:>5}/{n_proteins}]  "
                  f"{rate:.2f} prot/s  "
                  f"ETA: {eta_str}  "
                  f"fallbacks: {n_fallback}")

        prot_seq = str(df.iloc[row_idxs[0]][COL_PROTEIN])
        prot_len = len(prot_seq)

        # ── Decide strategy ───────────────────────────────────────────────
        force_fallback = (args.max_seq_len is not None) and (prot_len > args.max_seq_len)

        residue_emb = None
        if not force_fallback:
            residue_emb = try_embed_full(prot_seq)
            if residue_emb is None:
                oom_proteins.add(prot_id)
                print(f"    ⚠ OOM on {prot_id} (len={prot_len}) → windowed fallback")

        # ── Case 1: full protein embedding succeeded ──────────────────────
        if residue_emb is not None:
            assert residue_emb.shape[0] == prot_len, \
                f"{prot_id}: emb {residue_emb.shape[0]} != prot {prot_len}"

            for idx in row_idxs:
                row = df.iloc[idx]
                s0, e0 = get_peptide_span_0idx(row)

                peptide_embs[idx] = mean_pool(residue_emb[s0:e0])

                nf_len   = get_flank_len(row, COL_NFLANK)
                nf_start = max(0, s0 - nf_len)
                n_flank_embs[idx] = mean_pool(residue_emb[nf_start:s0])

                cf_len = get_flank_len(row, COL_CFLANK)
                cf_end = min(prot_len, e0 + cf_len)
                c_flank_embs[idx] = mean_pool(residue_emb[e0:cf_end])

            # Free memory immediately
            del residue_emb
            gc.collect()

        # ── Case 2: fallback — per-peptide windowed embedding ─────────────
        else:
            for idx in row_idxs:
                row = df.iloc[idx]
                s0, e0 = get_peptide_span_0idx(row)
                nf_len = get_flank_len(row, COL_NFLANK)
                cf_len = get_flank_len(row, COL_CFLANK)

                win_emb, win_start = embed_window(prot_seq, s0, e0, nf_len, cf_len)

                s_w = s0 - win_start
                e_w = e0 - win_start

                peptide_embs[idx] = mean_pool(win_emb[s_w:e_w])

                nf_start_w = max(0, s_w - nf_len)
                n_flank_embs[idx] = mean_pool(win_emb[nf_start_w:s_w])

                cf_end_w = min(len(win_emb), e_w + cf_len)
                c_flank_embs[idx] = mean_pool(win_emb[e_w:cf_end_w])

                fallback_flags[idx] = 1
                n_fallback += 1

                del win_emb
                gc.collect()

        n_done += 1

        if DEVICE == "cuda" and n_done % 200 == 0:
            torch.cuda.empty_cache()

elapsed_total = time.time() - t_start
print(f"\n{'─' * 50}")
print(f"  Proteins embedded        : {n_done}")
print(f"  Peptides total           : {len(df)}")
print(f"  Windowed-fallback used   : {n_fallback}/{len(df)}")
print(f"  Total time               : {elapsed_total/60:.1f} min")
print(f"  Avg time per protein     : {elapsed_total/n_done:.2f} s")
if oom_proteins:
    print(f"  OOM proteins             : {len(oom_proteins)}")
    oom_lens = [len(str(df[df[COL_UNIPROT] == p].iloc[0][COL_PROTEIN]))
                for p in oom_proteins]
    print(f"    Lengths                : min={min(oom_lens)}, max={max(oom_lens)}")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {args.out_h5}...")
with h5py.File(args.out_h5, "w") as f:
    f.create_dataset("peptide_emb",   data=peptide_embs,   dtype="float32")
    f.create_dataset("n_flank_emb",   data=n_flank_embs,   dtype="float32")
    f.create_dataset("c_flank_emb",   data=c_flank_embs,   dtype="float32")
    f.create_dataset("fallback_flag", data=fallback_flags,  dtype="int8")

    f.create_dataset("peptide_seqs",
                     data=np.array(df[COL_PEPTIDE].tolist(), dtype="S50"))
    f.create_dataset("uniprot_ids",
                     data=np.array(df[COL_UNIPROT].tolist(), dtype="S20"))
    f.create_dataset("row_indices",
                     data=np.array(df.index.tolist()))
    f.create_dataset("start", data=df[COL_START].values.astype(np.int32))
    f.create_dataset("end",   data=df[COL_END].values.astype(np.int32))

    f.attrs["model"]            = args.model
    f.attrs["emb_dim"]          = EMB_DIM
    f.attrs["n_samples"]        = len(df)
    f.attrs["n_proteins"]       = n_proteins
    f.attrs["n_fallback"]       = int(n_fallback)
    f.attrs["n_oom_proteins"]   = len(oom_proteins)
    f.attrs["fallback_window"]  = args.fallback_window
    f.attrs["max_seq_len"]      = str(args.max_seq_len)
    f.attrs["csv_source"]       = args.csv_path
    f.attrs["coord_convention"] = CONVENTION
    f.attrs["device"]           = DEVICE
    f.attrs["total_time_sec"]   = elapsed_total

print("\nDone ✓")
print(f"  peptide_emb  : {peptide_embs.shape}")
print(f"  n_flank_emb  : {n_flank_embs.shape}")
print(f"  c_flank_emb  : {c_flank_embs.shape}")
print(f"  fallback_flag: {fallback_flags.shape}  (sum={fallback_flags.sum()})")
