#!/usr/bin/env python3
"""
Embed fixed 29-residue full_context windows with ESM-C (full-protein first).

Four pad-handling modes produce four separate HDF5 files:
  zero              - zero-fill pad positions
  impute_boundary   - repeat nearest boundary residue (leftmost/rightmost real)
  impute_bos_eos    - repeat BOS/<cls> for left pads, EOS/<eos> for right pads
  pad_token         - replace X pads with <pad> token id, use sequence_id mask

Default pad-mode = impute_boundary.
Default padded FASTA = data/processed/positives_clean_padded.fasta.

Requirements:
  - df_all.csv must have columns: peptide, uniprot_id, full_context, start, end, sequence
  - full_context must be exactly 29 characters for ALL rows (script aborts otherwise)
  - Target .h5 files must NOT already exist (script errors, no overwrite)

Usage:
    # dry run (validate only)
    python src/tools/esm/embed_peptides_w_esm.py --dry-run

    # small debug run
    python src/tools/esm/embed_peptides_w_esm.py --n-debug 200

    # full run
    python src/tools/esm/embed_peptides_w_esm.py
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_LEN = 29
BOS_EOS_OFFSET = 2  # <cls> + <eos>
DEFAULT_CSV = "data/processed/df_all.csv"
DEFAULT_PAD_FASTA = "data/processed/positives_clean_padded.fasta"
DEFAULT_OUTDIR = "data/processed/embeddings"
DEFAULT_MODEL = "esmc_600m"
DEFAULT_FALLBACK_WINDOW = 2048


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Embed 29-mer full_context with ESM-C (4 pad modes)"
    )
    p.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV")
    p.add_argument("--out-dir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["esmc_300m", "esmc_600m"],
    )
    p.add_argument("--fallback-window", type=int, default=DEFAULT_FALLBACK_WINDOW)
    p.add_argument(
        "--modes",
        default="zero,impute_boundary,impute_bos_eos,pad_token",
        help="Comma-separated modes to run",
    )
    p.add_argument("--pad-token-fasta", default=DEFAULT_PAD_FASTA)
    p.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    p.add_argument("--n-debug", type=int, default=0, help="Process first N rows only")
    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────
def read_fasta(path: Path) -> dict[str, str]:
    """Read FASTA into {header: sequence} dict."""
    dd: dict[str, str] = {}
    cur_h: str | None = None
    cur_seq: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur_h is not None:
                    dd[cur_h] = "".join(cur_seq)
                cur_h = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_h is not None:
            dd[cur_h] = "".join(cur_seq)
    return dd


def detect_convention(df: pd.DataFrame) -> str:
    """Detect whether (start, end) is 0-indexed half-open or 1-indexed inclusive."""
    n = min(200, len(df))
    a, b = 0, 0
    for i in range(n):
        r = df.iloc[i]
        prot, pep = str(r["sequence"]), str(r["peptide"])
        s, e = int(r["start"]), int(r["end"])
        if 0 <= s < e <= len(prot) and prot[s:e] == pep:
            a += 1
        if s >= 1 and e <= len(prot) and prot[s - 1 : e] == pep:
            b += 1
    if b >= a and b > n * 0.8:
        return "1idx_inclusive"
    if a > b and a > n * 0.8:
        return "0idx_halfopen"
    raise RuntimeError(
        f"Cannot detect coord convention (0idx={a}, 1idx={b}/{n})"
    )


def span_0idx(row: pd.Series, conv: str) -> tuple[int, int]:
    """Return (start_0, end_0_exclusive) for peptide in protein."""
    s, e = int(row["start"]), int(row["end"])
    return (s - 1, e) if conv == "1idx_inclusive" else (s, e)


# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_full_protein(client, seq: str) -> np.ndarray:
    """Full protein embedding via client.logits. Returns [L+2, D] numpy."""
    protein = ESMProtein(sequence=seq)
    tensor = client.encode(protein)
    out = client.logits(
        tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return out.embeddings.squeeze(0).cpu().numpy()  # [L+2, D]


def embed_padded_protein(
    client, tokenizer, padded_seq: str, pad_token_id: int, device: torch.device
) -> tuple[np.ndarray, int]:
    """Embed padded protein with <pad> tokens + sequence_id mask.

    Returns (emb [L_padded+2, D], n_left) where n_left = number of leading X's.
    """
    L_padded = len(padded_seq)
    encoded = tokenizer.encode(padded_seq, add_special_tokens=True)
    tokens = torch.tensor([encoded], dtype=torch.long, device=device)  # [1, L_padded+2]

    if tokens.shape[1] != L_padded + 2:
        raise RuntimeError(
            f"Token length {tokens.shape[1]} != padded_seq len {L_padded} + 2"
        )

    # identify X pad positions in the residue slots (exclude BOS/EOS)
    pad_mask = torch.tensor(
        [ch in ("X", "x") for ch in padded_seq],
        dtype=torch.bool,
        device=device,
    )

    tokens = tokens.clone()
    tokens[0, 1:-1][pad_mask] = pad_token_id

    # sequence_id: True for real + specials, False for pads
    seq_id = torch.ones_like(tokens, dtype=torch.bool)
    seq_id[0, 1:-1][pad_mask] = False

    with torch.inference_mode():
        out = client.forward(sequence_tokens=tokens, sequence_id=seq_id)
    emb = out.embeddings[0].float().cpu().numpy()  # [L_padded+2, D]

    # count leading X's
    n_left = 0
    for ch in padded_seq:
        if ch in ("X", "x"):
            n_left += 1
        else:
            break

    return emb, n_left


# ── Window builders (zero / impute) ──────────────────────────────────────────
def make_zero_window(emb_full: np.ndarray, win_start: int, L: int):
    """Zero-fill pads. Returns (window [29,D], pad_mask [29])."""
    D = emb_full.shape[1]
    w = np.zeros((WINDOW_LEN, D), dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    for i in range(WINDOW_LEN):
        j = win_start + i
        if 0 <= j < L:
            w[i] = emb_full[j + 1]  # +1: skip BOS
        else:
            pm[i] = True
    return w, pm


def make_impute_boundary_window(emb_full: np.ndarray, win_start: int, L: int):
    """Repeat nearest boundary residue. Returns (window [29,D], pad_mask [29])."""
    D = emb_full.shape[1]
    w = np.zeros((WINDOW_LEN, D), dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    valid: list[int] = []
    for i in range(WINDOW_LEN):
        j = win_start + i
        if 0 <= j < L:
            w[i] = emb_full[j + 1]
            valid.append(i)
        else:
            pm[i] = True
    if valid:
        # fill left pads with first real, right pads with last real
        for i in range(valid[0]):
            w[i] = w[valid[0]]
        for i in range(valid[-1] + 1, WINDOW_LEN):
            w[i] = w[valid[-1]]
    return w, pm


def make_impute_bos_eos_window(emb_full: np.ndarray, win_start: int, L: int):
    """BOS for left pads, EOS for right pads. Returns (window [29,D], pad_mask [29])."""
    D = emb_full.shape[1]
    w = np.zeros((WINDOW_LEN, D), dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    for i in range(WINDOW_LEN):
        j = win_start + i
        if j < 0:
            w[i] = emb_full[0]  # BOS / <cls>
            pm[i] = True
        elif j >= L:
            w[i] = emb_full[L + 1]  # EOS / <eos>
            pm[i] = True
        else:
            w[i] = emb_full[j + 1]
    return w, pm


# ── Pad-token window extractor ────────────────────────────────────────────────
def extract_pad_window(
    emb_padded: np.ndarray,
    win_start: int,
    n_left: int,
    L_orig: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 29-mer from padded-protein embeddings. Returns (window, pad_mask)."""
    D = emb_padded.shape[1]
    L_padded = len(emb_padded) - 2  # subtract BOS/EOS
    w = np.zeros((WINDOW_LEN, D), dtype=np.float32)
    pm = np.zeros(WINDOW_LEN, dtype=bool)
    for i in range(WINDOW_LEN):
        j_orig = win_start + i
        j_padded = j_orig + n_left
        if 0 <= j_padded < L_padded and 0 <= j_orig < L_orig:
            w[i] = emb_padded[j_padded + 1]  # +1 for BOS
        else:
            pm[i] = True
    return w, pm


# ── HDF5 helpers ──────────────────────────────────────────────────────────────
def ensure_datasets(h5f: h5py.File, n_rows: int, emb_dim: int):
    """Create window_embeddings, pad_mask, fallback_flag datasets if missing."""
    if "window_embeddings" in h5f:
        return
    h5f.create_dataset(
        "window_embeddings",
        shape=(n_rows, WINDOW_LEN, emb_dim),
        dtype="float32",
        chunks=(min(64, n_rows), WINDOW_LEN, emb_dim),
        compression="gzip",
        compression_opts=4,
    )
    h5f.create_dataset(
        "pad_mask",
        shape=(n_rows, WINDOW_LEN),
        dtype="uint8",
        compression="gzip",
    )
    h5f.create_dataset(
        "fallback_flag",
        shape=(n_rows,),
        dtype="uint8",
    )
    h5f.attrs["emb_dim"] = emb_dim


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",")]
    allowed = {"zero", "impute_boundary", "impute_bos_eos", "pad_token"}
    for m in modes:
        if m not in allowed:
            raise SystemExit(f"Unknown mode {m}; allowed: {allowed}")

    out_paths = {
        "zero": outdir / "esmc_context_embeddings_zeropad.h5",
        "impute_boundary": outdir / "esmc_context_embeddings_impute_boundary.h5",
        "impute_bos_eos": outdir / "esmc_context_embeddings_impute_bos_eos.h5",
        "pad_token": outdir / "esmc_context_embeddings_padtoken.h5",
    }

    # ── check no overwrite ──
    for m in modes:
        if out_paths[m].exists():
            raise SystemExit(f"{out_paths[m]} exists; will not overwrite.")

    # ── load CSV ──
    df = pd.read_csv(args.csv)
    required = ["peptide", "uniprot_id", "full_context", "start", "end", "sequence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    if args.n_debug > 0:
        df = df.iloc[: args.n_debug].reset_index(drop=True)

    # ── validate full_context length ──
    lens = df["full_context"].astype(str).str.len()
    bad_count = (lens != WINDOW_LEN).sum()
    if bad_count > 0:
        bad_idx = df.index[lens != WINDOW_LEN].tolist()[:10]
        raise SystemExit(
            f"{bad_count} rows have full_context != {WINDOW_LEN}. "
            f"Example indices: {bad_idx}"
        )

    conv = detect_convention(df)
    groups = df.groupby("uniprot_id").indices
    n_rows = len(df)

    print(f"  {n_rows} rows, {len(groups)} unique proteins, coord={conv}")

    # ── load padded FASTA ──
    fasta_map: dict[str, str] = {}
    if "pad_token" in modes:
        fp = Path(args.pad_token_fasta)
        if fp.exists():
            fasta_map = read_fasta(fp)
            print(f"  padded FASTA: {len(fasta_map)} entries from {fp}")
        else:
            print(
                f"  Warning: {fp} not found. "
                f"Will construct padded sequences from n_pad_len/c_pad_len."
            )

    # ── dry run ──
    if args.dry_run:
        print("Dry run OK. Inputs validated.")
        print(f"  modes: {modes}")
        return

    # ── load model ──
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"  Loading {args.model} on {device}...")
    client = ESMC.from_pretrained(args.model).to(device)
    client.eval()
    tokenizer = get_esmc_model_tokenizers()
    pad_token_id = tokenizer.pad_token_id
    print(f"  pad_token_id={pad_token_id}")

    # ── open HDF5 files ──
    h5s: dict[str, h5py.File] = {}
    for m in modes:
        h5s[m] = h5py.File(out_paths[m], "w")
        h5s[m].create_dataset(
            "peptide",
            data=df["peptide"].astype(str).tolist(),
            dtype=h5py.string_dtype("utf-8"),
        )
        h5s[m].create_dataset(
            "uniprot_id",
            data=df["uniprot_id"].astype(str).tolist(),
            dtype=h5py.string_dtype("utf-8"),
        )
        h5s[m].create_dataset(
            "row_indices", data=df.index.to_numpy(dtype=np.int64)
        )
        h5s[m].attrs["model"] = args.model
        h5s[m].attrs["csv_source"] = str(args.csv)
        h5s[m].attrs["pad_mode"] = m
        h5s[m].attrs["window_len"] = WINDOW_LEN
        h5s[m].attrs["device"] = str(device)

    emb_dim: int | None = None
    stats = {
        m: {"padded_rows": 0, "pad_slots": 0, "fallback": 0}
        for m in modes
    }
    t0 = time.time()
    prot_n = 0

    # ── process proteins ──
    for prot_id, row_idxs in groups.items():
        prot_n += 1
        prot_seq = str(df.iloc[row_idxs[0]]["sequence"])
        L = len(prot_seq)

        # ── full protein embedding (zero / impute modes) ──
        emb_full: np.ndarray | None = None
        try:
            emb_full = embed_full_protein(client, prot_seq)
            if emb_full.shape[0] - L != BOS_EOS_OFFSET:
                raise RuntimeError(
                    f"Offset mismatch {prot_id}: "
                    f"emb={emb_full.shape[0]} prot={L}"
                )
        except (RuntimeError, MemoryError) as exc:
            emb_full = None
            print(f"  ⚠ full-protein fail {prot_id} (len={L}): {exc}")

        if emb_full is not None and emb_dim is None:
            emb_dim = int(emb_full.shape[1])
            for m in modes:
                ensure_datasets(h5s[m], n_rows, emb_dim)

        # ── padded protein embedding (pad_token mode) ──
        emb_padded: np.ndarray | None = None
        n_left = 0
        if "pad_token" in modes:
            padded_seq: str | None = None

            # try FASTA first
            if prot_id in fasta_map:
                padded_seq = fasta_map[prot_id]
            # otherwise construct from n_pad_len / c_pad_len
            elif "n_pad_len" in df.columns:
                prot_rows = df.loc[row_idxs]
                n_pad_max = int(prot_rows["n_pad_len"].max())
                c_pad_max = int(prot_rows["c_pad_len"].max())
                if n_pad_max > 0 or c_pad_max > 0:
                    padded_seq = "X" * n_pad_max + prot_seq + "X" * c_pad_max

            if padded_seq is not None:
                try:
                    emb_padded, n_left = embed_padded_protein(
                        client, tokenizer, padded_seq, pad_token_id, device
                    )
                    L_padded = len(padded_seq)
                    if emb_padded.shape[0] - L_padded != BOS_EOS_OFFSET:
                        raise RuntimeError(
                            f"Padded offset {prot_id}: "
                            f"emb={emb_padded.shape[0]} padded={L_padded}"
                        )
                except (RuntimeError, MemoryError) as exc:
                    emb_padded = None
                    print(f"  ⚠ padded fail {prot_id}: {exc}")

            if emb_padded is not None and emb_dim is None:
                emb_dim = int(emb_padded.shape[1])
                for m in modes:
                    ensure_datasets(h5s[m], n_rows, emb_dim)

        # ── extract windows per peptide ──
        for ridx in row_idxs:
            row = df.iloc[ridx]
            s0, _e0 = span_0idx(row, conv)
            win_start = s0 - 10
            fallback = 0

            for m in modes:
                w: np.ndarray
                pm: np.ndarray

                if m == "pad_token":
                    # ── pad-token: use padded protein embeddings ──
                    if emb_padded is not None:
                        w, pm = extract_pad_window(
                            emb_padded, win_start, n_left, L
                        )
                    else:
                        # fallback: embed the 29-mer window directly
                        fc = str(row["full_context"])
                        encoded = tokenizer.encode(
                            fc, add_special_tokens=True
                        )
                        tokens = torch.tensor(
                            [encoded], dtype=torch.long, device=device
                        )
                        x_mask = torch.tensor(
                            [ch in ("X", "x") for ch in fc],
                            dtype=torch.bool,
                            device=device,
                        )
                        tokens = tokens.clone()
                        tokens[0, 1:-1][x_mask] = pad_token_id
                        seq_id = torch.ones_like(tokens, dtype=torch.bool)
                        seq_id[0, 1:-1][x_mask] = False
                        with torch.inference_mode():
                            out = client.forward(
                                sequence_tokens=tokens,
                                sequence_id=seq_id,
                            )
                        w = out.embeddings[0].float().cpu().numpy()[1:-1].astype(
                            np.float32
                        )
                        pm = x_mask.cpu().numpy()
                        fallback = 1
                else:
                    # ── zero / impute: use full-protein embeddings ──
                    if emb_full is not None:
                        if m == "zero":
                            w, pm = make_zero_window(emb_full, win_start, L)
                        elif m == "impute_boundary":
                            w, pm = make_impute_boundary_window(
                                emb_full, win_start, L
                            )
                        elif m == "impute_bos_eos":
                            w, pm = make_impute_bos_eos_window(
                                emb_full, win_start, L
                            )
                    else:
                        # OOM fallback: embed centred window
                        pep_mid = s0 + 4
                        wh = args.fallback_window // 2
                        ws = max(0, pep_mid - wh)
                        we = min(L, ws + args.fallback_window)
                        ws = max(0, we - args.fallback_window)
                        try:
                            emb_win_full = embed_full_protein(
                                client, prot_seq[ws:we]
                            )
                            emb_win = emb_win_full[1:-1]  # strip BOS/EOS
                            Lw = emb_win.shape[0]
                            emb_local = np.zeros(
                                (Lw + 2, emb_win.shape[1]), dtype=np.float32
                            )
                            emb_local[1:-1] = emb_win
                            wl = win_start - ws
                            if m == "zero":
                                w, pm = make_zero_window(emb_local, wl, Lw)
                            elif m == "impute_boundary":
                                w, pm = make_impute_boundary_window(
                                    emb_local, wl, Lw
                                )
                            elif m == "impute_bos_eos":
                                w, pm = make_impute_bos_eos_window(
                                    emb_local, wl, Lw
                                )
                            fallback = 1
                        except (RuntimeError, MemoryError):
                            w = np.zeros(
                                (WINDOW_LEN, emb_dim or 1152), dtype=np.float32
                            )
                            pm = np.ones(WINDOW_LEN, dtype=bool)
                            fallback = 1

                # ── write row ──
                h5s[m]["window_embeddings"][ridx] = w
                h5s[m]["pad_mask"][ridx] = pm.astype(np.uint8)
                h5s[m]["fallback_flag"][ridx] = fallback

                if pm.any():
                    stats[m]["padded_rows"] += 1
                    stats[m]["pad_slots"] += int(pm.sum())
                if fallback:
                    stats[m]["fallback"] += 1

        # ── free per-protein ──
        del emb_full, emb_padded
        gc.collect()

        if prot_n % 100 == 0:
            elapsed = time.time() - t0
            rate = prot_n / elapsed if elapsed > 0 else 0
            eta = (len(groups) - prot_n) / rate if rate > 0 else float("inf")
            eta_s = f"{eta / 60:.0f}min" if eta < float("inf") else "?"
            print(
                f"  [{prot_n}/{len(groups)}] "
                f"{rate:.2f} prot/s  ETA: {eta_s}"
            )

    # ── finalize ──
    elapsed = time.time() - t0
    for m in modes:
        h5s[m].attrs["n_rows"] = n_rows
        h5s[m].attrs["total_time_sec"] = elapsed
        h5s[m].attrs["padded_rows"] = stats[m]["padded_rows"]
        h5s[m].attrs["pad_slots_total"] = stats[m]["pad_slots"]
        h5s[m].attrs["fallback_rows"] = stats[m]["fallback"]
        if m == "pad_token":
            h5s[m].attrs["pad_token_id"] = int(pad_token_id)
        h5s[m].close()

    print(f"\nDone. {n_rows} rows, {elapsed:.1f}s")
    for m in modes:
        print(
            f"  {m}: padded_rows={stats[m]['padded_rows']}, "
            f"pad_slots={stats[m]['pad_slots']}, "
            f"fallback={stats[m]['fallback']}"
        )


if __name__ == "__main__":
    main()
