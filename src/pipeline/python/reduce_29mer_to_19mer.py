#!/usr/bin/env python3
"""
reduce_29mer_to_19mer.py
Derive 19mer (5aa flank) dataset variants from the existing 29mer (10aa flank)
artifacts — no re-embedding, no structure re-fetch.

Motivation: the differential sequence logo showed weak signal at the terminal
positions furthest from the peptide; this produces a flank5 dataset to test
whether trimming those positions helps.

Why slicing is exact (not an approximation):
  * n_flank is X-padded on the LEFT (right-aligned to the cut), c_flank on the
    RIGHT. So flank5 = last5(n_flank) + peptide + first5(c_flank), and the
    position columns N5..N1 / C1..C5 of the 29mer are byte-identical to the
    flank5 positions — only N6..N10 / C6..C10 are dropped.
  * Window embeddings lay out slots [0-9 N | 10-18 peptide | 19-28 C], so the
    19-wide window is exactly slots 5..23 -> windows[:, 5:24, :].

Outputs (all named so config.py's THESIS_VARIANT=flank5 mechanism picks them
up, i.e. <stem>_flank5<suffix>):
  data/processed/df_all_with_folds_flank5.csv    (split data, folds unchanged)
  data/processed/df_all_flank5.csv               (raw, no fold col)
  data/processed/df_<enc>_flank5.csv             (sparse/blosum*/kidera/atchley,
                                                  far-position columns dropped)
  data/processed/df_compositional_flank5.csv     (frac_* recomputed over 19 pos)
  data/processed/embeddings_prepared/<key>_prepared_flank5.h5
                                                  (windows [N,19,D], attrs
                                                   window=19)

Known caveat (deliberate): structural flank aggregates in the handcrafted
feature set (mean_rsa_nflank, frac_q8_*_nflank, mean_plddt_nflank, ...) were
computed over the original 10-residue windows and are kept AS-IS. Only
n_pad_len / c_pad_len / n_flank_len_real (pure width features) and the
flank coordinates are updated for width 5. Re-running R steps 06/07 would be
needed for a fully consistent 19mer structural set.

Usage:
    python src/pipeline/python/reduce_29mer_to_19mer.py                  # CSVs + default key
    python src/pipeline/python/reduce_29mer_to_19mer.py --keys esmc300m_win_padtoken mpnn_v48020_zeropad
    python src/pipeline/python/reduce_29mer_to_19mer.py --skip-h5         # CSVs only
    python src/pipeline/python/reduce_29mer_to_19mer.py --skip-csv        # H5 only
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# This script always READS the base 29mer artifacts and WRITES _flank5 ones.
# config.py applies the THESIS_VARIANT path redirect at import time, so drop
# the env var before importing config or we'd read (nonexistent) flank5 inputs.
import os
os.environ.pop("THESIS_VARIANT", None)

from config import PROJECT_ROOT, DATA_DIR, get_embedding_source

FLANK = 5                 # new flank width (was 10)
OLD_FLANK = 10
NEW_WINDOW = 19           # 5 + 9 peptide + 5
OLD_WINDOW = 29
SLICE = slice(FLANK, FLANK + NEW_WINDOW)   # 5:24 — N5..N1 | P1..P9 | C1..C5
CHUNK = 1024

# position columns kept / dropped
KEEP_POS = ([f"N{i}" for i in range(FLANK, 0, -1)] +      # N5..N1
            [f"P{i}" for i in range(1, 10)] +             # P1..P9
            [f"C{i}" for i in range(1, FLANK + 1)])       # C1..C5
DROP_POS = ([f"N{i}" for i in range(FLANK + 1, OLD_FLANK + 1)] +  # N6..N10
            [f"C{i}" for i in range(FLANK + 1, OLD_FLANK + 1)])    # C6..C10
DROP_COL_RE = re.compile(r"^(?:N10|N[6-9]|C10|C[6-9])_")

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]   # matches functions.R

ENCODED_SOLO = ["df_sparse", "df_blosum50", "df_blosum62", "df_blosum80",
                "df_kidera", "df_atchley"]

DEFAULT_KEYS = ["esmc300m_win_padtoken"]


def variant_path(path: Path) -> Path:
    """Mirror config._variant() with VARIANT=flank5: <stem>_flank5<suffix>."""
    return path.with_name(f"{path.stem}_flank5{path.suffix}")


# ──────────────────────────────────────────────────────────────
# 1. CONTEXT DATAFRAME (df_all / df_all_with_folds)
# ──────────────────────────────────────────────────────────────
def reduce_context_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()

    for col in ("n_flank", "c_flank", "full_context"):
        if col not in out:
            raise KeyError(f"{name}: missing expected column '{col}'")

    bad = out["n_flank"].str.len().ne(OLD_FLANK) | out["c_flank"].str.len().ne(OLD_FLANK)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} rows with flank length != {OLD_FLANK}")

    # ── flanks: right-aligned N (keep last 5), left-aligned C (keep first 5) ──
    out["n_flank"] = out["n_flank"].str[-FLANK:]
    out["c_flank"] = out["c_flank"].str[:FLANK]
    out["n_flank_raw"] = out["n_flank_raw"].str[-FLANK:]
    out["c_flank_raw"] = out["c_flank_raw"].str[:FLANK]

    # ── real-length / pad bookkeeping for width 5 ──
    out["n_flank_len_real"] = out["n_flank_len_real"].clip(upper=FLANK)
    out["c_flank_len_real"] = out["c_flank_len_real"].clip(upper=FLANK)
    out["n_pad_len"] = FLANK - out["n_flank_len_real"]
    out["c_pad_len"] = FLANK - out["c_flank_len_real"]

    # ── combined context must be exactly 19 ──
    out["full_context"] = (out["n_flank"] + out["peptide"] + out["c_flank"])
    ctx_bad = out["full_context"].str.len().ne(NEW_WINDOW)
    if ctx_bad.any():
        raise ValueError(f"{name}: {int(ctx_bad.sum())} rows with full_context != {NEW_WINDOW}")
    # padding direction sanity: X only at N-left / C-right edges
    n_x = out["n_flank"].str.len() - out["n_flank"].str.lstrip("X").str.len()
    if not (n_x == FLANK - out["n_flank_len_real"]).all():
        raise ValueError(f"{name}: n_flank X-padding inconsistent with n_flank_len_real")
    c_x = out["c_flank"].str.len() - out["c_flank"].str.rstrip("X").str.len()
    if not (c_x == FLANK - out["c_flank_len_real"]).all():
        raise ValueError(f"{name}: c_flank X-padding inconsistent with c_flank_len_real")

    # ── flank coordinates follow the 5-wide windows ──
    # (structural aggregates stay 10-wide — see module docstring caveat)
    out["nflank_start"] = out["start"] - out["n_flank_len_real"]
    out["nflank_end"] = out["start"] - 1
    out["cflank_start"] = out["end"] + 1
    out["cflank_end"] = out["end"] + out["c_flank_len_real"]

    # ── drop far position columns ──
    present_drop = [c for c in DROP_POS if c in out.columns]
    if len(present_drop) != len(DROP_POS):
        missing = set(DROP_POS) - set(present_drop)
        raise ValueError(f"{name}: expected position columns missing: {sorted(missing)}")
    out = out.drop(columns=present_drop)

    absent_keep = [c for c in KEEP_POS if c not in out.columns]
    if absent_keep:
        raise ValueError(f"{name}: kept position columns missing: {absent_keep}")

    return out


def run_csvs() -> None:
    print("=" * 70)
    print("CSV reduction: 29mer -> 19mer (flank5)")
    print("=" * 70)

    # --- context dataframes ---
    for src_name in ("df_all_with_folds.csv", "df_all.csv"):
        src = DATA_DIR / src_name
        dst = variant_path(src)
        df = pd.read_csv(src)
        n_in = len(df)
        red = reduce_context_df(df, src_name)
        assert len(red) == n_in, "row count changed"
        for key in ("peptide", "label"):
            if not red[key].equals(df[key]):
                raise ValueError(f"{src_name}: '{key}' changed")
        if "fold" in red.columns and not red["fold"].equals(df["fold"]):
            raise ValueError(f"{src_name}: fold assignment changed")
        red.to_csv(dst, index=False)
        print(f"  {src.name} -> {dst.name}  ({len(red)} rows x {red.shape[1]} cols)")

    # --- encoded solo CSVs: drop far-position columns only ---
    for stem in ENCODED_SOLO:
        src = DATA_DIR / f"{stem}.csv"
        if not src.exists():
            print(f"  SKIP {src.name} (not found)")
            continue
        dst = variant_path(src)
        df = pd.read_csv(src)
        drop = [c for c in df.columns if DROP_COL_RE.match(c)]
        # every far position must be represented (N6..N10 x dims, C6..C10 x dims)
        covered = {c.split("_", 1)[0] for c in drop}
        missing_pos = [p for p in DROP_POS if p not in covered]
        if missing_pos:
            raise ValueError(f"{src.name}: no columns found for {missing_pos}")
        red = df.drop(columns=drop)
        # kept positions must survive
        keep_present = {c.split("_", 1)[0] for c in red.columns
                        if c.split("_", 1)[0] in KEEP_POS}
        missing_keep = [p for p in KEEP_POS if p not in keep_present]
        if missing_keep:
            raise ValueError(f"{src.name}: kept positions missing: {missing_keep}")
        red.to_csv(dst, index=False)
        print(f"  {src.name} -> {dst.name}  "
              f"({len(red)} rows x {red.shape[1]} cols, -{len(drop)} cols)")

    # --- compositional: recompute frac_* over the 19 kept positions ---
    src = DATA_DIR / "df_compositional.csv"
    if src.exists():
        dst = variant_path(src)
        keys = pd.read_csv(src, usecols=["peptide", "label"])
        split = pd.read_csv(DATA_DIR / "df_all_with_folds.csv",
                            usecols=["peptide", "label"] + KEEP_POS)
        if not keys.equals(split[["peptide", "label"]]):
            raise ValueError("df_compositional row order != split row order")
        mat = split[KEEP_POS].to_numpy()
        n_pos = len(KEEP_POS)   # 19
        comp = pd.DataFrame(
            {f"frac_{aa}": (mat == aa).sum(axis=1) / n_pos for aa in AA_ORDER})
        # row sums = non-NA positions / 19 (mirrors 08_aa_encoding.R)
        expected = (~pd.isna(mat)).sum(axis=1) / n_pos
        if not np.allclose(comp.sum(axis=1).to_numpy(), expected, atol=1e-9):
            raise ValueError("compositional row-sum validation failed")
        out = pd.concat([keys.reset_index(drop=True), comp], axis=1)
        out.to_csv(dst, index=False)
        print(f"  {src.name} -> {dst.name}  ({len(out)} rows x {out.shape[1]} cols, "
              f"frac over {n_pos} positions)")
    else:
        print(f"  SKIP {src.name} (not found)")


# ──────────────────────────────────────────────────────────────
# 2. WINDOW EMBEDDING HDF5s: [N,29,D] -> [N,19,D]
# ──────────────────────────────────────────────────────────────
def reduce_h5(key: str) -> None:
    src_cfg = get_embedding_source(key)
    if src_cfg.get("kind") != "windows":
        raise ValueError(f"'{key}' is not a window embedding (kind="
                         f"{src_cfg.get('kind')}) — nothing to slice")
    src = src_cfg.get("prepared_path")
    if src is None or not Path(src).exists():
        raise FileNotFoundError(f"prepared H5 not found for '{key}': {src}")
    src = Path(src)
    dst = variant_path(src)

    with h5py.File(src, "r") as fin:
        W = fin["windows"]
        M = fin["pad_mask"]
        P = fin["pad_counts"]
        n, w, d = W.shape
        if w != OLD_WINDOW:
            raise ValueError(f"{src.name}: window dim {w} != {OLD_WINDOW}")
        if M.shape != (n, OLD_WINDOW):
            raise ValueError(f"{src.name}: pad_mask shape {M.shape} != {(n, OLD_WINDOW)}")

        with h5py.File(dst, "w") as fout:
            # windows: stream chunk-wise, slice slots 5..23
            ws = fout.create_dataset(
                "windows", shape=(n, NEW_WINDOW, d), dtype=W.dtype,
                chunks=(min(CHUNK, n), NEW_WINDOW, d))
            ms = fout.create_dataset(
                "pad_mask", shape=(n, NEW_WINDOW), dtype=M.dtype,
                chunks=(min(CHUNK, n), NEW_WINDOW))
            for i0 in range(0, n, CHUNK):
                sl = slice(i0, min(i0 + CHUNK, n))
                ws[sl] = W[sl, SLICE, :]
                ms[sl] = M[sl, SLICE]

            # pad_counts: pads lost by trimming = max(0, old - 5) per side
            pc = P[...]
            pc_out = np.clip(pc - FLANK, 0, None).astype(P.dtype)
            fout.create_dataset("pad_counts", data=pc_out)

            # everything else copied verbatim (row-aligned metadata)
            for name, obj in fin.items():
                if name in ("windows", "pad_mask", "pad_counts"):
                    continue
                fout.create_dataset(name, data=obj[...])

            # attrs: window=19, keep provenance
            for ak, av in fin.attrs.items():
                fout.attrs[ak] = av
            fout.attrs["window"] = NEW_WINDOW
            fout.attrs["orig_window"] = OLD_WINDOW
            fout.attrs["reduced_by"] = "reduce_29mer_to_19mer.py"
            fout.attrs["reduced_timestamp"] = pd.Timestamp.now().isoformat()

    # ── post-write validation ──
    with h5py.File(dst, "r") as f:
        assert f["windows"].shape == (n, NEW_WINDOW, d)
        assert f["pad_mask"].shape == (n, NEW_WINDOW)
        assert int(f.attrs["window"]) == NEW_WINDOW
        assert f["pad_counts"][...].max() <= FLANK
        # pad_mask must agree with pad_counts on both sides
        m = f["pad_mask"][...]
        if not (m[:, :FLANK].sum(axis=1) == f["pad_counts"][..., 0]).all():
            raise ValueError("N-side pad_mask != pad_counts after slice")
        if not (m[:, 9 + FLANK:].sum(axis=1) == f["pad_counts"][..., 1]).all():
            raise ValueError("C-side pad_mask != pad_counts after slice")
        # labels/folds identical to source
        with h5py.File(src, "r") as fin:
            for ds in ("labels", "folds"):
                if ds in fin and not np.array_equal(f[ds][...], fin[ds][...]):
                    raise ValueError(f"{ds} changed during slice")
    print(f"  {src.name} -> {dst.name}  "
          f"(windows [{n},{NEW_WINDOW},{d}], pad_counts <= {FLANK})")


def run_h5(keys: list[str]) -> None:
    print("=" * 70)
    print("HDF5 reduction: 29-wide windows -> 19-wide (slots 5:24)")
    print("=" * 70)
    for key in keys:
        reduce_h5(key)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS,
                    help=f"EMBEDDING_SOURCES window keys to slice "
                         f"(default: {' '.join(DEFAULT_KEYS)})")
    ap.add_argument("--skip-csv", action="store_true",
                    help="skip the dataframe reduction")
    ap.add_argument("--skip-h5", action="store_true",
                    help="skip the embedding HDF5 slicing")
    args = ap.parse_args()

    if not args.skip_csv:
        run_csvs()
    if not args.skip_h5:
        run_h5(args.keys)
    print("\nDone. Run modelling with THESIS_VARIANT=flank5 to pick these up.")


if __name__ == "__main__":
    main()
