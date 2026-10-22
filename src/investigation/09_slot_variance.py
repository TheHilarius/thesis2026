#!/usr/bin/env python3
"""
09_slot_variance.py
Rank the 29 window slots by variance to pick ONE global top-21 set per
embedding (same slots for every sample — supervisor's var21 test).

Metric (default): mean per-sample row variance. For each sample and each
non-padded slot, variance across the D embedding dims; averaged per slot
over all samples. Generalizes the supervisor's single-sample example
("rows with most variance for that sample") to all 46617 windows.
A plain global per-slot variance column is reported alongside for free.

Pads are masked via pad_mask (fill values would corrupt the ranking).

Inputs: prepared window H5s (fallback: raw), resolved via config
EMBEDDING_SOURCES, e.g. esmc300m_win_padtoken (29x960),
esmif_eosrepeat (29x512).

Outputs (results/tables/):
    slot_variance_<key>.csv       — slot, mean_row_var, global_var, rank
    slot_variance_<key>_top21.json — selected slots, ascending (model order)
Plus bar plot in results/figures/models/slot_variance/ if matplotlib exists.
"""

import argparse
import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import numpy as np
import h5py
from pathlib import Path

from config import (
    PROJECT_ROOT,
    get_embedding_source,
)

CHUNK = 2048
DEFAULT_KEYS = ["esmc300m_win_padtoken", "esmif_eosrepeat"]


def resolve_h5(emb_key):
    """Prepared path preferred (what modelling uses); else raw. Returns
    (path, window_ds, pad_mask_ds)."""
    src = get_embedding_source(emb_key)
    for p in (src.get("prepared_path"), src.get("raw_path")):
        if p is not None and Path(p).exists():
            return (Path(p), src["window_ds"], src["pad_mask_ds"],
                    "prepared" if p == src.get("prepared_path") else "raw")
    raise FileNotFoundError(
        f"No H5 found for '{emb_key}': "
        f"{src.get('prepared_path')} / {src.get('raw_path')}")


def rank_slots(h5_path, window_ds, mask_ds, top=21, chunk=CHUNK):
    """Stream (N, 29, D) + (N, 29) mask. Returns (mean_row_var[29],
    global_var[29], n_used[29])."""
    with h5py.File(h5_path, "r") as f:
        W = f[window_ds]
        M = f[mask_ds]
        n, window, d = W.shape
        row_var_sum = np.zeros(window, dtype=np.float64)
        row_var_cnt = np.zeros(window, dtype=np.int64)
        g_sum = np.zeros(window, dtype=np.float64)
        g_sumsq = np.zeros(window, dtype=np.float64)
        g_cnt = np.zeros(window, dtype=np.int64)
        for i0 in range(0, n, chunk):
            idx = slice(i0, min(i0 + chunk, n))
            block = W[idx].astype(np.float64)       # (c, 29, D)
            mblock = M[idx].astype(bool)            # (c, 29) True=pad
            real = ~mblock
            # mean per-sample row variance (pads excluded)
            rv = block.var(axis=2)                  # (c, 29)
            row_var_sum += np.where(real, rv, 0.0).sum(axis=0)
            row_var_cnt += real.sum(axis=0)
            # global per-slot variance accumulators (pads excluded)
            b = np.where(real[:, :, None], block, 0.0)
            g_sum += b.sum(axis=(0, 2))
            g_sumsq += (b ** 2).sum(axis=(0, 2))
            g_cnt += real.sum(axis=0) * d
    mean_row_var = row_var_sum / np.maximum(row_var_cnt, 1)
    mean = g_sum / np.maximum(g_cnt, 1)
    global_var = g_sumsq / np.maximum(g_cnt, 1) - mean ** 2
    return mean_row_var, global_var, row_var_cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                    help="Comma-separated EMBEDDING_SOURCES keys")
    ap.add_argument("--top", type=int, default=21,
                    help="Number of slots to select (default: 21)")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: results/tables)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PROJECT_ROOT / "results" / "figures" / "models" / "slot_variance"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir.mkdir(parents=True, exist_ok=True)
        has_mpl = True
    except ImportError:
        has_mpl = False

    for key in [k.strip() for k in args.keys.split(",") if k.strip()]:
        path, wds, mds, kind = resolve_h5(key)
        print(f"\n{key}: {path.name} ({kind})")
        mean_rv, gvar, n_used = rank_slots(path, wds, mds, top=args.top)
        order = np.argsort(-mean_rv)                # desc by score
        top_slots = sorted(int(s) for s in order[:args.top])
        print(f"  mean_row_var range: {mean_rv.min():.4g} .. {mean_rv.max():.4g}")
        print(f"  top-{args.top} slots (model order): {top_slots}")

        csv_path = out_dir / f"slot_variance_{key}.csv"
        with open(csv_path, "w") as fh:
            fh.write("slot,mean_row_var,global_var,n_samples_used,rank\n")
            for rank, s in enumerate(order, 1):
                fh.write(f"{s},{mean_rv[s]:.10g},{gvar[s]:.10g},"
                         f"{n_used[s]},{rank}\n")
        print(f"  wrote {csv_path}")
        json_path = out_dir / f"slot_variance_{key}_top{args.top}.json"
        json_path.write_text(json.dumps({
            "embedding_key": key,
            "metric": "mean per-sample row variance, pads masked",
            "slots": top_slots,
        }, indent=1) + "\n")
        print(f"  wrote {json_path}")

        if has_mpl:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            xs = np.arange(len(mean_rv))
            ax.bar(xs, mean_rv, color="#3498db")
            ax.bar(top_slots, mean_rv[top_slots], color="#e74c3c")
            ax.set_xlabel("window slot")
            ax.set_ylabel("mean per-sample row variance")
            ax.set_title(f"{key} — top-{args.top} slots (red)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"slot_variance_{key}.png", dpi=120)
            plt.close(fig)
            print(f"  wrote {fig_dir / f'slot_variance_{key}.png'}")


if __name__ == "__main__":
    main()
