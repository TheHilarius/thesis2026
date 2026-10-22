#!/usr/bin/env python3
"""
11_reduce_prepared_windows.py
Build reduced prepared window embeddings: keep all 29 positions, keep only
N embedding dims (29x21 = 609, AA-encoding shape — supervisor's dim21 test).

Two dim selections:
    --dims PATH|LIST   explicit global dim list (var21 top-N JSON from
                       10_dim_variance.py, or '3,17,208'), or
    --first N          the first N dims (simple21: positions 0..N-1).

Output: a prepared HDF5 with the SAME schema as the source (windows
[N,29,n_dims], pad_mask, pad_counts, labels, folds, identifiers, attrs),
so modelling consumes it untouched — no --dims flag needed at train time.
Dataset copies are streamed chunk-wise; the dim gather applies per chunk.

Example (server, env_esmc_server):
    python src/pipeline/python/11_reduce_prepared_windows.py \
        --key esmc300m_win_padtoken --first 21
    python src/pipeline/python/11_reduce_prepared_windows.py \
        --key esmc300m_win_padtoken \
        --dims results/tables/dim_variance_esmc300m_win_padtoken_top21.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import h5py
from pathlib import Path

from config import get_embedding_source, PREPARED_EMBEDDING_DIR

CHUNK = 1024

# copied verbatim, row-aligned
META_DATASETS = ("pad_mask", "pad_counts", "labels", "folds", "starts",
                 "peptide_seqs", "uniprot_ids", "df_row_indices")


def parse_dims_arg(args):
    """Resolve --dims (JSON/list) or --first N into a sorted unique dim list."""
    if args.dims is not None and args.first is not None:
        raise SystemExit("use only one of --dims / --first")
    if args.dims is not None:
        if args.dims.endswith(".json"):
            with open(args.dims) as fh:
                data = json.load(fh)
            dims = data.get("dims") if isinstance(data, dict) else data
            if dims is None:
                raise SystemExit(f"no 'dims' key in {args.dims}")
        else:
            dims = [t for t in args.dims.split(",") if t.strip() != ""]
        dims = sorted({int(d) for d in dims})
    elif args.first is not None:
        dims = list(range(int(args.first)))
    else:
        raise SystemExit("pick --first N or --dims PATH|LIST")
    if not dims or min(dims) < 0:
        raise SystemExit(f"bad dims: {dims[:5]}")
    return dims


def reduce_one(key, dims, out_path=None, overwrite=False, chunk=CHUNK):
    src = get_embedding_source(key)
    prepared = src.get("prepared_path")
    if prepared is None or not Path(prepared).exists():
        raise SystemExit(
            f"{key}: prepared H5 missing ({prepared}) — "
            f"run 03_prepare_embeddings.py first")
    out_path = Path(out_path) if out_path else Path(str(prepared).replace(
        "_prepared.h5",
        f"_prepared_{'first' if dims == list(range(len(dims))) else 'var'}"
        f"{len(dims)}.h5"))
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} exists (use --overwrite)")

    with h5py.File(prepared, "r") as f:
        if "windows" not in f:
            raise SystemExit(f"{prepared}: no 'windows' dataset")
        W = f["windows"]
        n, window, d = W.shape
        if max(dims) >= d:
            raise SystemExit(f"dim {max(dims)} >= D={d}")
        tag = "first" if dims == list(range(len(dims))) else "var"

        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with h5py.File(tmp, "w") as g:
            dst = g.create_dataset(
                "windows", shape=(n, window, len(dims)), dtype=np.float32,
                chunks=(64, window, min(len(dims), d)),
            )
            for i0 in range(0, n, chunk):
                i1 = min(i0 + chunk, n)
                dst[i0:i1] = W[i0:i1, :, dims]

            copied = []
            for name in META_DATASETS:
                if name in f:
                    g.create_dataset(name, data=f[name][:])
                    copied.append(name)

            for k, v in f.attrs.items():
                g.attrs[k] = v
            g.attrs["emb_dim"] = len(dims)
            g.attrs["display_name"] = (
                f"{src['display_name']} ({tag}-{len(dims)}-dim reduced)")
            g.attrs["source_prepared"] = str(prepared)
            g.attrs["dim_selection"] = tag
            g.attrs["selected_dims"] = np.array(dims, dtype=np.int32)
            g.attrs["reduce_timestamp"] = datetime.now().isoformat()

        os.replace(tmp, out_path)

    print(f"{key}: {n}x{window}x{d} -> {n}x{window}x{len(dims)} "
          f"({tag}-{len(dims)})")
    print(f"  wrote {out_path}")
    print(f"  copied: {', '.join(copied)}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--key", required=True,
                    help="EMBEDDING_SOURCES key (prepared H5 source)")
    ap.add_argument("--first", type=int, default=None,
                    help="Keep the first N dims (simple21: 21)")
    ap.add_argument("--dims", default=None,
                    help="Dim selection: 10_dim_variance top-N JSON path "
                         "or comma list ('3,17,208')")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <prepared>_d<N>.h5)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing output file")
    args = ap.parse_args()

    dims = parse_dims_arg(args)
    reduce_one(args.key, dims, args.out, args.overwrite)


if __name__ == "__main__":
    main()
