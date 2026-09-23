#!/usr/bin/env python3
"""
embed_windows_esmif.py — ESM-IF fixed-29 per-residue windows, 4 padding modes.

Embeds each protein structure ONCE with ESM-IF1, then for every peptide builds a
fixed 29-slot window (10 N-flank + 9 peptide + 10 C-flank) of per-residue
encoder reps [29, 512].  Peptides near a protein terminus have missing flank
slots; those slots are filled with ONE of four strategies, each written to its
own HDF5 file so the four can be ablated later:

  *zero*           — every missing slot = the zero vector.
  *pad*            — every missing slot = a single "<pad>"-analog vector
                     (the encoder output at a NaN-coordinate gap position, the
                     ESM-IF / AntiFold convention for "no structure here").
  *boundary*       — every missing slot = the nearest real residue at that
                     boundary, repeated (terminus fallback: the peptide's own
                     boundary residue).  No EOS/BOS marker anywhere.
  *eos_bos_repeat* — every missing slot = the "<eos>"-analog vector on the C
                     side / "<bos>"-analog vector on the N side.

All four are uniform in structure (k pad slots, all identical fill); none
reserves a slot for a boundary marker.  This keeps the ablation a pure
fill-value comparison.

No mean-pooling is done: every row is a full 29 x 512 tensor for a downstream
model that consumes the (29, 512) sequence.

Memory / IO handling:
  - Windows are streamed straight to preallocated HDF5 datasets, one protein
    group at a time, so the full N x 29 x 512 arrays are never held in RAM.
  - Each protein's rep [L, 512] is freed after its group is written.

CLI (kept minimal on purpose).  Additional knobs can be added here later if
needed — the likely candidates are:
  --chunk-rows N   HDF5 chunk size along the row axis (currently 64)
  --compress       enable gzip on the window dataset (currently off for speed)
  --modes LIST     subset of {zero,pad,boundary,eos_bos_repeat}
  --probe-k K      number of NaN rows used to derive the pad vector (currently 8)

Usage:
    python src/tools/esm/embed_windows_esmif.py
    python src/tools/esm/embed_windows_esmif.py --out-prefix path/to/esmif_windows
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

import esm
import esm.inverse_folding.util as esm_util

try:
    from esm.inverse_folding.util import CoordBatchConverter
    HAS_COORD_CONVERTER = True
except ImportError:
    HAS_COORD_CONVERTER = False

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--csv', default="data/processed/df_all.csv")
parser.add_argument('--pdb',
                    default="data/processed/structures/alphafold/")
parser.add_argument('--af2', default="data/processed/structures/alphafold_supplement/")
parser.add_argument('--out-prefix',
                    default="data/processed/embeddings/esm-if_test")
parser.add_argument('--force-cpu', action='store_true')
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
if args.force_cpu:
    DEVICE, USE_GPU = "cpu", False
elif torch.cuda.is_available() and HAS_COORD_CONVERTER:
    DEVICE, USE_GPU = "cuda", True
elif torch.cuda.is_available() and not HAS_COORD_CONVERTER:
    DEVICE, USE_GPU = "cpu", False
    print("WARN: GPU present but CoordBatchConverter import failed -> CPU")
else:
    DEVICE, USE_GPU = "cpu", False

EMB_DIM = 512
CHUNK_ROWS = 64
PROBE_K = 8

# AF2 fallback only when the --af2 dir exists
AF2_FALLBACK = os.path.isdir(args.af2)
os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

print(f"Device        : {DEVICE}")
print(f"AF2 fallback  : {'ENABLED' if AF2_FALLBACK else 'DISABLED'}")

# ── Load ESM-IF1 ──────────────────────────────────────────────────────────────
print("Loading ESM-IF1 (142M)...")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to(DEVICE)


# ── GPU-aware encoder output (raw, [T, D] WITH special tokens) ───────────────
def get_encoder_output_gpu(coords):
    batch_converter = CoordBatchConverter(alphabet)
    coords_t, confidence, strs, tokens, padding_mask = batch_converter(
        [(coords, None, None)])
    coords_t = coords_t.to(DEVICE)
    confidence = confidence.to(DEVICE)
    padding_mask = padding_mask.to(DEVICE)
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        encoder_out = model.encoder(coords_t, padding_mask, confidence,
                                    return_all_hiddens=False)
    rep = encoder_out['encoder_out'][0].transpose(0, 1)
    return rep[0].cpu().numpy()  # [T, D]


def encode_structure(coords):
    """Return raw encoder rep [T, D] (may include BOS/EOS special tokens)."""
    if USE_GPU:
        return get_encoder_output_gpu(coords)
    out = esm_util.get_encoder_output(model, alphabet, coords)
    if isinstance(out, tuple):
        out = out[0]
    return out


def strip_special(rep, seq_len):
    """Strip leading/trailing special tokens so rep aligns 1:1 with seq."""
    n = rep.shape[0] - seq_len
    if n == 0:
        return rep
    if n == 1:
        return rep[1:]
    if n == 2:
        return rep[1:-1]
    raise ValueError(f"Unexpected rep offset {n} (rep {rep.shape[0]} vs seq {seq_len})")


# ── Probe: derive the pad-token and eos-token constant vectors ────────────────
def probe_constants():
    """Return (pad_rep, eos_rep, bos_rep) as [512] float32 constants.

    eos_rep = encoder output at the final (end-token) position of a real chain.
    bos_rep = encoder output at the leading (start-token) position of a real chain.
    pad_rep = encoder output at the first NaN-coordinate gap position of a
              chain with PROBE_K NaN rows appended (ESM-IF "no structure" gap).
    Falls back to zeros if no structure is available or the outputs are
    non-finite.
    """
    pad_rep = np.zeros(EMB_DIM, dtype=np.float32)
    eos_rep = np.zeros(EMB_DIM, dtype=np.float32)
    bos_rep = np.zeros(EMB_DIM, dtype=np.float32)

    cands = (glob.glob(os.path.join(args.pdb, "*.pdb"))
             + glob.glob(os.path.join(args.af2, "*.pdb")))
    if not cands:
        print("WARN: no AF2 pdb for probe; pad/eos/bos constants = zero vector")
        return pad_rep, eos_rep, bos_rep

    pdb = min(cands, key=os.path.getsize)
    try:
        coords, seq = load_coords_safe(pdb, chain="A")
    except Exception as e:
        print(f"WARN: probe structure failed ({e}); pad/eos/bos constants = zero")
        return pad_rep, eos_rep, bos_rep

    L = len(seq)
    rep = encode_structure(coords)
    offset = rep.shape[0] - L

    if offset >= 1:
        eos_rep = rep[-1].astype(np.float32)
    else:
        print("WARN: no special tokens detected; eos constant = zero")

    if offset >= 2:
        # leading special position (BOS / <cls>); verified against fair-esm's
        # get_encoder_output, which strips [1:-1] as "bos and eos tokens".
        bos_rep = rep[0].astype(np.float32)
    else:
        print("WARN: no leading BOS detected; bos constant = zero")

    nan_rows = np.full((PROBE_K,) + coords.shape[1:], np.nan, dtype=coords.dtype)
    coords_pad = np.concatenate([coords, nan_rows], axis=0)
    rep_pad = encode_structure(coords_pad)
    pad_idx = L + (1 if offset == 2 else 0)
    if pad_idx < rep_pad.shape[0]:
        pad_rep = rep_pad[pad_idx].astype(np.float32)

    for name, v in [("pad_rep", pad_rep), ("eos_rep", eos_rep), ("bos_rep", bos_rep)]:
        if not np.isfinite(v).all():
            print(f"WARN: {name} non-finite -> zero vector")
            if name == "pad_rep":
                pad_rep = np.zeros(EMB_DIM, dtype=np.float32)
            elif name == "eos_rep":
                eos_rep = np.zeros(EMB_DIM, dtype=np.float32)
            else:
                bos_rep = np.zeros(EMB_DIM, dtype=np.float32)

    print(f"probe pad_rep norm = {np.linalg.norm(pad_rep):.3f}  "
          f"eos_rep norm = {np.linalg.norm(eos_rep):.3f}  "
          f"bos_rep norm = {np.linalg.norm(bos_rep):.3f}")
    return pad_rep, eos_rep, bos_rep


# ── Structure file helpers (copied from embed_structures_esmif_gpu.py) ───────
def detect_file_format(filepath):
    try:
        with open(filepath, 'r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith(('data_', 'loop_', '#')):
                    return 'cif'
                if s.startswith(('ATOM', 'HETATM', 'HEADER', 'REMARK',
                                 'MODEL', 'CRYST', 'TITLE', 'COMPND',
                                 'SOURCE', 'KEYWDS', 'EXPDTA', 'DBREF')):
                    return 'pdb'
                break
    except Exception:
        pass
    return 'cif' if filepath.endswith('.cif') else 'pdb'


def load_coords_safe(struct_path, chain):
    actual_fmt = detect_file_format(struct_path)
    ext_fmt = 'cif' if struct_path.endswith('.cif') else 'pdb'
    if actual_fmt == ext_fmt:
        return esm_util.load_coords(struct_path, chain=chain)
    tmp = struct_path + f".tmp_esm.{actual_fmt}"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        os.symlink(os.path.abspath(struct_path), tmp)
        return esm_util.load_coords(tmp, chain=chain)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def find_structure_file(uniprot_id, search_dir):
    pattern = os.path.join(search_dir, f"*{uniprot_id}*.*")
    matches = [m for m in glob.glob(pattern)
               if m.endswith('.pdb') or m.endswith('.cif')]
    return matches[0] if matches else None


def get_chain_ids(struct_path):
    if HAS_BIOPYTHON:
        try:
            fmt = detect_file_format(struct_path)
            p = MMCIFParser(QUIET=True) if fmt == 'cif' else PDBParser(QUIET=True)
            return [c.id for c in p.get_structure("tmp", struct_path).get_chains()]
        except Exception:
            pass
    return ["A", "B", "C", "D", "E", "L", "H", "1", "2"]


def load_structure_best_chain(struct_path, uid, peptide_set):
    try:
        coords, seq = load_coords_safe(struct_path, chain="A")
        rep = encode_structure(coords)
        if sum(1 for p in peptide_set if p in seq) > 0:
            return rep, seq, "A"
        chain_a_result = (rep, seq)
    except Exception:
        chain_a_result = None

    best = None
    for chain_id in get_chain_ids(struct_path):
        if chain_id == "A":
            continue
        try:
            coords, seq = load_coords_safe(struct_path, chain=chain_id)
            rep = encode_structure(coords)
            hits = sum(1 for p in peptide_set if p in seq)
            if hits > 0 and (best is None or hits > best[3]):
                best = (rep, seq, chain_id, hits)
        except Exception:
            continue
    if best is not None:
        return best[0], best[1], best[2]
    if chain_a_result is not None:
        return chain_a_result[0], chain_a_result[1], "A"
    return None, None, None


def load_af2_fallback(uid, af2_dir):
    af2_path = find_structure_file(uid, af2_dir)
    if af2_path is None:
        return None, None
    try:
        coords, seq = load_coords_safe(af2_path, chain="A")
        return encode_structure(coords), seq
    except Exception:
        return None, None


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


# ── Fixed-29 window builder ───────────────────────────────────────────────────
def build_windows(rep, n_start, pep_start, pep_end, c_end, pad_rep, eos_rep, bos_rep):
    """Build (W_zero, W_pad, W_boundary, W_eosbos, pad_mask, n_pad, c_pad).

    rep is the [L, D] residue-aligned encoder output.
    Slot layout: 0-9 N (right-aligned), 10-18 peptide (9), 19-28 C (left-aligned).

    Every mode fills EVERY pad slot with one uniform value (no reserved marker
    slot): zero -> 0, pad -> pad_rep, boundary -> nearest real residue,
    eos_bos_repeat -> eos_rep on the C side / bos_rep on the N side.
    Terminus fallback (n_real==0 / c_real==0): the boundary copy is the
    peptide's own boundary residue, matching ESM-C's impute_boundary.
    """
    D = rep.shape[1]
    n_real = pep_start - n_start
    c_real = c_end - pep_end

    base = np.zeros((29, D), dtype=np.float32)
    base[10 - n_real:10] = rep[n_start:pep_start]
    base[10:19] = rep[pep_start:pep_end]
    base[19:19 + c_real] = rep[pep_end:c_end]

    n_pad = 10 - n_real
    c_pad = 10 - c_real
    pad_mask = np.zeros(29, dtype=bool)
    pad_mask[0:n_pad] = True
    pad_mask[29 - c_pad:] = True

    # nearest real residue at each boundary (terminus fallback -> peptide residue)
    first_real_idx = 10 if n_real == 0 else (10 - n_real)
    last_real_idx = 18 if c_real == 0 else (19 + c_real - 1)

    # zero
    W_zero = base.copy()

    # pad
    W_pad = base.copy()
    if n_pad > 0:
        W_pad[0:n_pad] = pad_rep
    if c_pad > 0:
        W_pad[29 - c_pad:] = pad_rep

    # boundary: every pad slot = nearest real residue
    W_boundary = base.copy()
    if n_pad > 0:
        W_boundary[0:n_pad] = base[first_real_idx]
    if c_pad > 0:
        W_boundary[29 - c_pad:] = base[last_real_idx]

    # eos/bos-repeat: every pad slot = EOS (C side) / BOS (N side)
    W_eosbos = base.copy()
    if n_pad > 0:
        W_eosbos[0:n_pad] = bos_rep
    if c_pad > 0:
        W_eosbos[29 - c_pad:] = eos_rep

    return W_zero, W_pad, W_boundary, W_eosbos, pad_mask, n_pad, c_pad


# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
df['_csv_row'] = np.arange(len(df))  # original 0-based row index in the CSV
before = len(df)
df = df.dropna(subset=['peptide', 'uniprot_id', 'start', 'end'])
print(f"Rows: {len(df)} (dropped {before - len(df)} NaNs)")

df = df.sort_values('uniprot_id').reset_index(drop=True)
df['_pos'] = np.arange(len(df))  # contiguous write position
n_rows = len(df)

uid_peptides = df.groupby('uniprot_id')['peptide'].apply(set).to_dict()

# ── Probe constants ───────────────────────────────────────────────────────────
pad_rep, eos_rep, bos_rep = probe_constants()

# ── Preallocate 3 HDF5 files ──────────────────────────────────────────────────
MODES = ["zero", "pad", "boundary", "eos_bos_repeat"]
files = {}
for m in MODES:
    f = h5py.File(f"{args.out_prefix}_{m}.h5", "w")
    f.create_dataset("window_if_struct", shape=(n_rows, 29, EMB_DIM),
                     dtype="float32", chunks=(CHUNK_ROWS, 29, EMB_DIM))
    f.create_dataset("pad_mask", shape=(n_rows, 29), dtype="bool")
    files[m] = f

# Small metadata arrays (held in RAM; tiny)
peptide_ids = np.zeros(n_rows, dtype="S20")
uniprot_ids = np.zeros(n_rows, dtype="S20")
row_indices = np.zeros(n_rows, dtype=np.int64)
starts = np.zeros(n_rows, dtype=np.int32)
ends = np.zeros(n_rows, dtype=np.int32)
n_pads = np.zeros(n_rows, dtype=np.int8)
c_pads = np.zeros(n_rows, dtype=np.int8)
statuses = np.zeros(n_rows, dtype=np.int8)  # 0 primary,1 af2,2 missing,3 notfound

ZERO_WINDOW = np.zeros((29, EMB_DIM), dtype=np.float32)
FULL_MASK = np.ones(29, dtype=bool)

stats = {"primary": 0, "af2_fallback": 0, "missing": 0, "notfound": 0}

# ── Main loop: one protein group at a time ────────────────────────────────────
print(f"\nEmbedding {df['uniprot_id'].nunique()} proteins -> 3 window files ...")

for gi, (uid, group) in enumerate(df.groupby('uniprot_id', sort=True)):
    if gi % 200 == 0:
        print(f"  [{gi}] protein {uid}  (rows={len(group)})")

    orig_indices = group['_pos'].values       # contiguous write positions
    csv_rows = group['_csv_row'].values       # original CSV row indices
    rows = list(group.iterrows())

    # Resolve structure once per protein
    rep = seq = None
    source = 2  # default missing

    struct_path = find_structure_file(uid, args.pdb)
    if struct_path:
        rep_raw, seq, chain = load_structure_best_chain(
            struct_path, uid, uid_peptides.get(uid, set()))
        if rep_raw is not None:
            rep = strip_special(rep_raw, len(seq))
            source = 0
    if rep is None and AF2_FALLBACK:
        rep_raw, seq = load_af2_fallback(uid, args.af2)
        if rep_raw is not None:
            rep = strip_special(rep_raw, len(seq))
            source = 1

    # Build windows for every peptide in this protein
    W_blocks = {m: np.zeros((len(group), 29, EMB_DIM), dtype=np.float32)
                for m in MODES}
    mask_blocks = np.zeros((len(group), 29), dtype=bool)

    for j, (_, row) in enumerate(rows):
        oi = orig_indices[j]
        peptide_ids[oi] = str(row['peptide']).encode()
        uniprot_ids[oi] = str(uid).encode()
        row_indices[oi] = csv_rows[j]
        starts[oi] = int(row['start'])
        ends[oi] = int(row['end'])

        if rep is None:
            statuses[oi] = 2
            stats["missing"] += 1
            n_pads[oi] = c_pads[oi] = -1
            mask_blocks[j] = FULL_MASK
            continue

        result = find_peptide_in_structure(seq, rep.shape[0], row)
        if result is None:
            statuses[oi] = 3
            stats["notfound"] += 1
            n_pads[oi] = c_pads[oi] = -1
            mask_blocks[j] = FULL_MASK
            continue

        ps, pe, ns, ce, match = result
        Wz, Wp, Wb, We, pm, npad, cpad = build_windows(
            rep, ns, ps, pe, ce, pad_rep, eos_rep, bos_rep)

        W_blocks["zero"][j] = Wz
        W_blocks["pad"][j] = Wp
        W_blocks["boundary"][j] = Wb
        W_blocks["eos_bos_repeat"][j] = We
        mask_blocks[j] = pm
        n_pads[oi] = npad
        c_pads[oi] = cpad
        statuses[oi] = source
        stats["primary" if source == 0 else "af2_fallback"] += 1

    # Write this protein's block to the 3 files, then free the rep
    for m in MODES:
        files[m]["window_if_struct"][orig_indices] = W_blocks[m]
        files[m]["pad_mask"][orig_indices] = mask_blocks
    del W_blocks, mask_blocks, rep
    if gi % 100 == 0:
        for f in files.values():
            f.flush()
    if USE_GPU and gi % 200 == 0:
        torch.cuda.empty_cache()

# ── Write metadata + attrs, close ─────────────────────────────────────────────
for m, f in files.items():
    f.create_dataset("peptide_ids", data=peptide_ids)
    f.create_dataset("uniprot_ids", data=uniprot_ids)
    f.create_dataset("row_indices", data=row_indices)
    f.create_dataset("start", data=starts)
    f.create_dataset("end", data=ends)
    f.create_dataset("n_pad", data=n_pads)
    f.create_dataset("c_pad", data=c_pads)
    f.create_dataset("status", data=statuses)
    f.attrs["model"] = "esm_if1_gvp4_t16_142M_UR50"
    f.attrs["emb_dim"] = EMB_DIM
    f.attrs["window"] = 29
    f.attrs["pad_mode"] = m
    f.attrs["pad_rep_norm"] = float(np.linalg.norm(pad_rep))
    f.attrs["eos_rep_norm"] = float(np.linalg.norm(eos_rep))
    f.attrs["n_samples"] = n_rows
    f.attrs["device"] = DEVICE
    f.attrs["status_legend"] = "0=primary 1=af2_fallback 2=missing 3=notfound"
    f.attrs["n_pad_sentinel"] = "-1 means no structure/peptide found"
    f.close()

print("\nDone. Source breakdown:", stats)
print(f"Files: {args.out_prefix}_{{zero,pad,boundary,eos_bos_repeat}}.h5")
