#!/usr/bin/env python3
"""
summarize_embeddings.py — Compare zero-vector rates across embedding HDF5 files.

Usage:
    python src/tools/esm/summarize_embeddings.py data/processed/embeddings/esmif_*.h5
"""

import sys
import numpy as np
import h5py
from pathlib import Path


def summarize(h5_path):
    """Extract summary stats from an embedding HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        if 'context_if_struct' in f:
            ctx = f['context_if_struct'][:]
        elif 'context_emb' in f:
            ctx = f['context_emb'][:]
        else:
            raise KeyError(f"No 'context_if_struct' or 'context_emb' dataset in {h5_path}")

        n_rows = len(ctx)
        emb_dim = ctx.shape[1]

        zero_ctx = np.sum(np.linalg.norm(ctx, axis=1) == 0.0)

        embedded = n_rows - zero_ctx

        # Mean norm of non-zero embeddings (sanity check)
        nonzero_mask = np.linalg.norm(ctx, axis=1) > 0
        mean_norm = np.mean(np.linalg.norm(ctx[nonzero_mask], axis=1)) if nonzero_mask.any() else 0.0
        std_norm = np.std(np.linalg.norm(ctx[nonzero_mask], axis=1)) if nonzero_mask.any() else 0.0

        # Unique proteins
        if 'uniprot_ids' in f:
            uids = f['uniprot_ids'][:]
            uids_decoded = [u.decode() if isinstance(u, bytes) else u for u in uids]
            total_proteins = len(set(uids_decoded) - {''})
            # Proteins with at least one non-zero embedding
            nonzero_uids = set(u for u, m in zip(uids_decoded, nonzero_mask) if m)
            embedded_proteins = len(nonzero_uids)
        else:
            total_proteins = None
            embedded_proteins = None

        # Attrs
        device = f.attrs.get('device', '?')
        cache_size = f.attrs.get('cache_size', '?')

    return {
        'path': h5_path,
        'name': Path(h5_path).stem,
        'n_rows': n_rows,
        'emb_dim': emb_dim,
        'embedded': embedded,
        'zero_ctx': zero_ctx,
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'total_proteins': total_proteins,
        'embedded_proteins': embedded_proteins,
        'device': device,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/tools/esm/summarize_embeddings.py <file1.h5> [file2.h5 ...]")
        sys.exit(1)

    paths = sorted(sys.argv[1:])
    results = []

    for p in paths:
        try:
            results.append(summarize(p))
        except Exception as e:
            print(f"ERROR reading {p}: {e}")

    if not results:
        print("No valid files found.")
        sys.exit(1)

    # ── Per-file detail ───────────────────────────────────────────────────
    for r in results:
        print(f"\n{'='*60}")
        print(f"  {r['name']}")
        print(f"{'='*60}")
        print(f"  Rows:              {r['n_rows']}")
        print(f"  Embedding dim:     {r['emb_dim']}")
        if r['total_proteins'] is not None:
            print(f"  Proteins total:    {r['total_proteins']}")
            print(f"  Proteins embedded: {r['embedded_proteins']}  "
                  f"({r['embedded_proteins']/r['total_proteins']*100:.1f}%)")
        print(f"  Embedded rows:     {r['embedded']}  "
              f"({r['embedded']/r['n_rows']*100:.1f}%)")
        print(f"  Zero vectors:      ctx={r['zero_ctx']}")
        print(f"  Zero rate:         {r['zero_ctx']/r['n_rows']*100:.1f}%")
        print(f"  Mean context norm: {r['mean_norm']:.3f} ± {r['std_norm']:.3f}")
        print(f"  Device used:       {r['device']}")

    # ── Comparison table ──────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*60}")

        # Header
        name_w = max(len(r['name']) for r in results) + 2
        print(f"  {'Source':<{name_w}} {'Rows':>6} {'Embedded':>9} {'Zero%':>7} "
              f"{'Proteins':>9} {'MeanNorm':>9}")
        print(f"  {'-'*name_w} {'-'*6} {'-'*9} {'-'*7} {'-'*9} {'-'*9}")

        for r in results:
            prot_str = (f"{r['embedded_proteins']}/{r['total_proteins']}"
                        if r['total_proteins'] else "?")
            print(f"  {r['name']:<{name_w}} {r['n_rows']:>6} {r['embedded']:>9} "
                  f"{r['zero_ctx']/r['n_rows']*100:>6.1f}% "
                  f"{prot_str:>9} {r['mean_norm']:>9.3f}")

    print()


if __name__ == '__main__':
    main()
