#!/usr/bin/env python3
"""
10_dim_variance.py
Rank the D embedding dims by variance to pick ONE global top-21 dim set per
embedding (same dims for every sample AND every position — var21-dim test).

Target shape per sample: 29 positions x 21 dims (609 features), mirroring
the AA-encoding schemes so embeddings compare fairly against them.

Metric (default): mean per-sample across-position variance. For each sample
and each dim, variance across the 29 positional slots; averaged per dim
over all samples. The transpose of the supervisor's single-sample row
example. A plain global per-dim variance column is reported alongside.

Pads are masked via pad_mask (fill values would corrupt the ranking).

Inputs: prepared window H5s (fallback: raw), resolved via config
EMBEDDING_SOURCES, e.g. esmc300m_win_padtoken (29x960),
esmif_eosrepeat (29x512).

Outputs (results/tables/):
    dim_variance_<key>.csv       — dim, mean_col_var, global_var, rank
    dim_variance_<key>_top21.json — selected dims, ascending (model order)
Plus bar plot in results/figures/models/dim_variance/ if matplotlib exists.
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

CHUNK = 512
DEFAULT_KEYS = ["esmc300m_win_padtoken", "esmif_eosrepeat"]


def resolve_h5(emb_key):
    """Prepared path preferred (what modelling uses); else raw. Returns
    (path, window_ds, pad_mask_ds). Prepared files store the tensor as
    'windows'/'pad_mask'; raw files use the config dataset names."""
    src = get_embedding_source(emb_key)
    for p in (src.get("prepared_path"), src.get("raw_path")):
        if p is not None and Path(p).exists():
            if p == src.get("prepared_path"):
                return (Path(p), "windows", "pad_mask", "prepared")
            return (Path(p), src["window_ds"], src["pad_mask_ds"], "raw")
    raise FileNotFoundError(
        f"No H5 found for '{emb_key}': "
        f"{src.get('prepared_path')} / {src.get('raw_path')}")


def rank_dims(h5_path, window_ds, mask_ds, top=21, chunk=CHUNK):
    """Stream (N, 29, D) + (N, 29) mask. Returns (mean_col_var[D],
    global_var[D], n_used[D])."""
    with h5py.File(h5_path, "r") as f:
        W = f[window_ds]
        M = f[mask_ds]
        n, window, d = W.shape
        col_var_sum = np.zeros(d, dtype=np.float64)
        col_var_cnt = np.zeros((), dtype=np.int64)
        g_sum = np.zeros(d, dtype=np.float64)
        g_sumsq = np.zeros(d, dtype=np.float64)
        g_cnt = np.zeros((), dtype=np.int64)
        for i0 in range(0, n, chunk):
            idx = slice(i0, min(i0 + chunk, n))
            block = W[idx].astype(np.float64)       # (c, 29, D)
            mblock = M[idx].astype(bool)            # (c, 29) True=pad
            real = (~mblock).astype(np.float64)     # (c, 29)
            n_real = real.sum(axis=1, keepdims=True)  # (c, 1)
            ok = (n_real[:, 0] > 1)
            # masked mean/var across positions, per (sample, dim)
            s = (block * real[:, :, None]).sum(axis=1)          # (c, D)
            ss = ((block ** 2) * real[:, :, None]).sum(axis=1)  # (c, D)
            nrf = np.maximum(n_real, 1)
            mean = s / nrf
            var = ss / nrf - mean ** 2
            col_var_sum += np.where(ok[:, None], var, 0.0).sum(axis=0)
            col_var_cnt += int(ok.sum())
            g_sum += s.sum(axis=0)
            g_sumsq += ss.sum(axis=0)
            g_cnt += int(real.sum())
    mean_col_var = col_var_sum / max(int(col_var_cnt), 1)
    mean = g_sum / max(int(g_cnt), 1)
    global_var = g_sumsq / max(int(g_cnt), 1) - mean ** 2
    n_used = np.full(d, int(g_cnt), dtype=np.int64)
    return mean_col_var, global_var, n_used


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                    help="Comma-separated EMBEDDING_SOURCES keys")
    ap.add_argument("--top", type=int, default=21,
                    help="Number of dims to select (default: 21)")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: results/tables)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "results" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PROJECT_ROOT / "results" / "figures" / "models" / "dim_variance"
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
        print(f"\n{key}: {path.name} ({kind})", flush=True)
        mean_cv, gvar, n_used = rank_dims(path, wds, mds, top=args.top)
        order = np.argsort(-mean_cv)                # desc by score
        top_dims = sorted(int(s) for s in order[:args.top])
        print(f"  mean_col_var range: {mean_cv.min():.4g} .. {mean_cv.max():.4g}")
        print(f"  top-{args.top} dims (model order): {top_dims}")

        csv_path = out_dir / f"dim_variance_{key}.csv"
        with open(csv_path, "w") as fh:
            fh.write("dim,mean_col_var,global_var,n_values_used,rank\n")
            for rank, s in enumerate(order, 1):
                fh.write(f"{s},{mean_cv[s]:.10g},{gvar[s]:.10g},"
                         f"{n_used[s]},{rank}\n")
        print(f"  wrote {csv_path}")
        json_path = out_dir / f"dim_variance_{key}_top{args.top}.json"
        json_path.write_text(json.dumps({
            "embedding_key": key,
            "metric": "mean per-sample across-position variance, pads masked",
            "dims": top_dims,
        }, indent=1) + "\n")
        print(f"  wrote {json_path}")

        if has_mpl:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            xs = np.arange(len(mean_cv))
            ax.bar(xs, mean_cv, color="#3498db")
            ax.bar(top_dims, mean_cv[top_dims], color="#e74c3c")
            ax.set_xlabel("embedding dim")
            ax.set_ylabel("mean per-sample across-position variance")
            ax.set_title(f"{key} — top-{args.top} dims (red)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"dim_variance_{key}.png", dpi=120)
            plt.close(fig)
            print(f"  wrote {fig_dir / f'dim_variance_{key}.png'}")


if __name__ == "__main__":
    main()
