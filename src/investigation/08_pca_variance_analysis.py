#!/usr/bin/env python3
"""
08_pca_variance_analysis.py
PCA variance analysis for per-residue window embeddings (fixed 29-mer).

Fits IncrementalPCA on pooled real residue slots (N×29_real, D) — matching
the PCA fit in 04_modelling.fit_window_pca — and plots cumulative explained
variance vs number of components.

Inputs:
    data/processed/embeddings/esmc_context_embeddings_{zeropad,impute_boundary,
        impute_bos_eos,padtoken}.h5   — window_embeddings [N,29,1152]
    data/processed/embeddings/esm-if_test_{zero,pad,boundary,eos_bos_repeat}.h5
                                        — window_if_struct [N,29,512]

PCA modes:
    slot     — per-position IncrementalPCA (default, matches 04_modelling slot path)
    flat     — flattened-window randomized SVD, capped at --flat-n-components;
               representative set only (FLAT_SETS: zeropad/zero)
    flat-full — EXACT full spectrum via streaming Gram + eigvalsh over ALL 8
               files (4 pad modes × 2 toolkits, WINDOW_SETS — parity with
               slot). Peak RAM ≈ 18 GB per ESM-C fit (D=1152), ≈4 GB ESM-IF.
               Outputs get `_full` suffix; cross-mode identity asserted on
               zero/boundary/eos_bos_repeat (padtoken/pad exempt — shifted
               reals).

Caching: finished curves (validated CSV: exact row count, monotonic
cumulative) are reused on rerun — only missing/aborted modes recompute.
--force recomputes everything.

Outputs (results/figures/models/pca_optimization/):
    pca_variance_windows_{key}.png       — individual plot with threshold annotations
    pca_variance_windows_{key}.csv       — per-component explained variance
    pca_variance_windows_{key}_full.*    — flat-full only, complete spectrum
    pca_variance_windows_esmc_{mode}_combined.png — ESM-C curves overlaid
    pca_variance_windows_esmif_{mode}_combined.png — ESM-IF curves overlaid
"""

import sys
import os
import argparse

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import numpy as np
import h5py
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
from sklearn.decomposition import IncrementalPCA, PCA
from pathlib import Path
from datetime import datetime

from config import EMBEDDING_DIR, FIGURES_DIR

# ── Window embedding sources ──────────────────────────────────────────────────
# Each entry: (filename, window_dataset_name, emb_dim, display_name, color, group)
WINDOW_SETS = {
    # ── ESM-C (D=1152) ──
    "esmc_zeropad": (
        "esmc_context_embeddings_zeropad.h5",
        "window_embeddings", 1152,
        "ESM-C zero", "#e74c3c", "esmc",
    ),
    "esmc_impute_boundary": (
        "esmc_context_embeddings_impute_boundary.h5",
        "window_embeddings", 1152,
        "ESM-C boundary", "#e67e22", "esmc",
    ),
    "esmc_impute_bos_eos": (
        "esmc_context_embeddings_impute_bos_eos.h5",
        "window_embeddings", 1152,
        "ESM-C EOS/BOS-repeat", "#c0392b", "esmc",
    ),
    "esmc_padtoken": (
        "esmc_context_embeddings_padtoken.h5",
        "window_embeddings", 1152,
        "ESM-C pad", "#8e44ad", "esmc",
    ),
    # ── ESM-IF (D=512) ──
    "esmif_zero": (
        "esm-if_test_zero.h5",
        "window_if_struct", 512,
        "ESM-IF zero", "#3498db", "esmif",
    ),
    "esmif_pad": (
        "esm-if_test_pad.h5",
        "window_if_struct", 512,
        "ESM-IF pad", "#2980b9", "esmif",
    ),
    "esmif_boundary": (
        "esm-if_test_boundary.h5",
        "window_if_struct", 512,
        "ESM-IF boundary", "#1abc9c", "esmif",
    ),
    "esmif_eos_bos_repeat": (
        "esm-if_test_eos_bos_repeat.h5",
        "window_if_struct", 512,
        "ESM-IF EOS/BOS-repeat", "#16a085", "esmif",
    ),
}

THRESHOLDS = [0.50, 0.80, 0.85, 0.90, 0.95, 0.99]

OUT_DIR = FIGURES_DIR / "pca_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 2048
PCA_BATCH = 32768

# Representative sources for capped `flat` mode only (--pca-mode flat).
# flat-full iterates WINDOW_SETS (all 8 files) for parity with slot;
# slot/flat-full do not consult this dict.
FLAT_SETS = {
    # ESM-IF first: cheap (~4 GB), validates numerics before ESM-C.
    "esmif_flat": (
        "esm-if_test_zero.h5",
        "window_if_struct", 512,
        "ESM-IF flat", "#3498db", "esmif",
    ),
    "esmc_flat": (
        "esmc_context_embeddings_zeropad.h5",
        "window_embeddings", 1152,
        "ESM-C flat", "#e74c3c", "esmc",
    ),
}
FLAT_N_COMPONENTS = 1024


# ── Stream real (non-pad, non-zero-norm) residue slots ────────────────────────
def stream_real_slots(raw_path: Path, ds_name: str, chunk: int = CHUNK):
    """
    Yield blocks of real residue slot vectors [m, D] from the raw window HDF5.

    Matches 04_modelling._stream_window_slots: reshape (N,29,D) -> (N*29, D),
    exclude pad_mask=True and zero-norm rows.
    """
    with h5py.File(raw_path, "r") as f:
        W = f[ds_name]
        M = f["pad_mask"]
        n = W.shape[0]

        for i0 in range(0, n, chunk):
            block = W[i0 : i0 + chunk]          # (c, 29, D)
            mblock = M[i0 : i0 + chunk]         # (c, 29) bool/uint8
            c = block.shape[0]

            flat = block.reshape(c * 29, block.shape[-1]).astype(np.float32)
            flat_mask = mblock.reshape(-1)

            # convert uint8 mask to bool if needed
            if flat_mask.dtype != bool:
                flat_mask = flat_mask.astype(bool)

            real = ~flat_mask
            slots = flat[real]

            # exclude zero-norm rows
            norms = np.linalg.norm(slots, axis=1)
            nonzero = norms > 0.0
            yield slots[nonzero]


# ── Fit PCA streaming ─────────────────────────────────────────────────────────
def fit_pca_streaming(raw_path: Path, ds_name: str, emb_dim: int):
    """
    Fit IncrementalPCA with n_components=emb_dim on streamed real slots.
    Returns (explained, cumulative) numpy arrays.
    """
    ipca = IncrementalPCA(n_components=emb_dim, batch_size=PCA_BATCH)

    n_slots = 0
    n_kept = 0
    for block in stream_real_slots(raw_path, ds_name):
        n_slots += block.shape[0]
        n_kept += block.shape[0]
        ipca.partial_fit(block.astype(np.float64))

    explained = ipca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"    slots streamed: {n_slots}, kept (real+nonzero): {n_kept}")
    print(f"    top-10 PCs explain: {cumulative[min(9, len(cumulative)-1)] * 100:.1f}%")

    return explained, cumulative


# ── Flat (flattened-window) streaming + PCA fit ───────────────────────────────
def stream_real_rows_flat(raw_path: Path, ds_name: str, chunk: int = CHUNK):
    """
    Yield fully-real flattened row vectors [m, 29*D] from the raw window HDF5.

    A row is kept only if it has no pad slot anywhere and no zero-norm slot —
    the same mode-invariant subset 04_modelling.fit_window_pca_flat fits on.
    """
    with h5py.File(raw_path, "r") as f:
        W = f[ds_name]
        M = f["pad_mask"]
        n = W.shape[0]

        for i0 in range(0, n, chunk):
            block = W[i0 : i0 + chunk].astype(np.float32)   # (c, 29, D)
            mblock = M[i0 : i0 + chunk]                     # (c, 29)
            c = block.shape[0]

            if mblock.dtype != bool:
                mblock = mblock.astype(bool)

            flat = block.reshape(c, -1)                     # (c, 29*D)
            fully_real = ~mblock.any(axis=1)
            norms = np.linalg.norm(flat, axis=1)
            keep = fully_real & (norms > 0.0)
            if keep.any():
                yield flat[keep]


def fit_pca_streaming_flat(raw_path: Path, ds_name: str, n_components: int):
    """
    Fit PCA (randomized SVD) over flattened (29*D,) fully-real rows.
    Returns (explained, cumulative) numpy arrays.

    Uses full-memory randomized PCA: IncrementalPCA is O(n_batch^2 * n_features)
    per batch and is intractable for n_features (29*D) >> n_batch.
    """
    rows = []
    n_rows = 0
    for block in stream_real_rows_flat(raw_path, ds_name):
        n_rows += block.shape[0]
        rows.append(block.astype(np.float64))

    X = np.concatenate(rows, axis=0)
    del rows
    n_components = int(min(n_components, X.shape[1], X.shape[0]))
    pca = PCA(n_components=n_components, svd_solver="randomized",
              random_state=42)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"    flat rows kept (fully-real): {n_rows}")
    print(f"    top-10 PCs explain: {cumulative[min(9, len(cumulative)-1)] * 100:.1f}%")

    return explained, cumulative


# ── Flat-full: exact complete spectrum via Gram matrix + eigvalsh ─────────────
def fit_pca_full_spectrum(raw_path: Path, ds_name: str):
    """
    Exact full variance spectrum over flattened (29*D,) fully-real rows.

    Unlike fit_pca_streaming_flat (randomized SVD, capped at n_components),
    this computes ALL eigenvalues: stream fully-real rows while accumulating
    the Gram matrix G = sum x x^T and column sums, center-correct
    C = G - n mu mu^T, then symmetric eigendecomposition. No random_state,
    deterministic. Peak RAM ≈ 2x the Gram matrix (~18 GB for ESM-C D=1152,
    ~4 GB for ESM-IF D=512). Requires the same inputs as flat mode.

    Returns (explained, cumulative, n_rows).
    """
    import gc

    G = None  # (d, d) float64 accumulator
    s = None  # (d,) float64 column sums
    n_rows = 0

    for block in stream_real_rows_flat(raw_path, ds_name):
        b = block.astype(np.float64)
        n = b.shape[0]
        if G is None:
            d = b.shape[1]
            G = np.zeros((d, d), dtype=np.float64)
            s = np.zeros(d, dtype=np.float64)
        s += b.sum(axis=0)
        G += b.T @ b
        n_rows += n

    if G is None or n_rows == 0:
        raise RuntimeError(f"No fully-real rows in {raw_path}")

    # center-correct: C = G - n * outer(mu, mu)
    mu = s / n_rows
    G -= n_rows * np.outer(mu, mu)
    trace_c = float(np.trace(G))
    del s, mu
    gc.collect()

    evals = np.linalg.eigvalsh(G)  # ascending
    evals = np.clip(evals, 0.0, None)
    del G
    gc.collect()
    evals = evals[::-1]  # descending

    total = float(evals.sum())
    if total <= 0:
        raise RuntimeError(f"Non-positive total variance in {raw_path}")

    explained = evals / total
    cumulative = np.cumsum(explained)

    print(f"    flat rows kept (fully-real): {n_rows}")
    print(f"    full spectrum: {len(explained)} components (all dims)")
    print(f"    sum(eigenvalues)={total:.6g}  trace(C)={trace_c:.6g}  "
          f"rel_diff={abs(total - trace_c) / max(trace_c, 1e-12):.2e}")
    print(f"    top-10 PCs explain: {cumulative[min(9, len(cumulative)-1)] * 100:.1f}%")

    return explained, cumulative, n_rows


# ── Find n_components for thresholds ──────────────────────────────────────────
def find_threshold_components(cumulative, thresholds):
    """Find n_components needed for each variance threshold."""
    results = []
    for t in thresholds:
        n = np.searchsorted(cumulative, t) + 1
        n = min(n, len(cumulative))
        actual_var = cumulative[n - 1]
        results.append({
            "threshold": t,
            "n_components": n,
            "actual_variance": actual_var,
        })
    return results


# ── Individual plot (with threshold annotations) ─────────────────────────────
def plot_individual(cumulative, thresholds_info, display_name, color, out_path):
    """Single-curve plot with threshold arrows."""
    if not HAS_MPL:
        print(f"    [skip plot] matplotlib unavailable: {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(1, len(cumulative) + 1)

    ax.plot(x, cumulative, color=color, linewidth=1.5, alpha=0.9)
    ax.fill_between(x, cumulative, alpha=0.1, color=color)

    for info in thresholds_info:
        t = info["threshold"]
        n = info["n_components"]
        actual = info["actual_variance"]

        ax.axhline(y=t, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axvline(x=n, color=color, linestyle=":", linewidth=0.8, alpha=0.6)

        label = f"{t*100:.0f}% -> {n} PCs"
        ax.annotate(
            label,
            xy=(n, actual),
            xytext=(n + 15, actual - 0.03),
            fontsize=7, color="grey",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"PCA Variance — {display_name}")
    ax.set_xlim(0, len(cumulative) + 10)
    ax.set_ylim(0, 1.05)

    max_pc = len(cumulative)
    tick_step = (5000 if max_pc > 5000 else
                 1000 if max_pc > 2000 else
                 100 if max_pc > 500 else (50 if max_pc > 200 else 25))
    ax.set_xticks(range(0, max_pc + 1, tick_step))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ── Combined plot (overlay curves + legend only) ─────────────────────────────
def plot_combined(curves, title, out_path):
    """Overlay multiple curves with legend. No threshold annotations."""
    if not HAS_MPL:
        print(f"    [skip plot] matplotlib unavailable: {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, (cumulative, display_name, color) in curves.items():
        x = np.arange(1, len(cumulative) + 1)
        ax.plot(x, cumulative, label=display_name, color=color,
                linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ── Save CSV ──────────────────────────────────────────────────────────────────
def save_csv(explained, cumulative, key, out_dir, suffix=""):
    """Save per-component variance to CSV."""
    csv_path = out_dir / f"pca_variance_windows_{key}{suffix}.csv"
    with open(csv_path, "w") as f:
        f.write("component_idx,explained_variance_ratio,cumulative_variance\n")
        for i, (e, c) in enumerate(zip(explained, cumulative)):
            f.write(f"{i + 1},{e:.8f},{c:.8f}\n")
    print(f"    Saved: {csv_path}")


def load_existing_curve(csv_path: Path, expected_n: int):
    """Return (explained, cumulative) from a finished CSV, else None.

    'Finished' = exactly expected_n rows, cumulative non-decreasing,
    first > 0, last <= 1 + 1e-6. A partial/aborted write fails these
    and the curve is recomputed (and overwritten).
    """
    if not csv_path.exists():
        return None
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(1, 2))
    except Exception:
        return None
    if data.ndim != 2 or data.shape[0] != expected_n:
        return None
    explained, cumulative = data[:, 0], data[:, 1]
    if not np.all(np.diff(cumulative) >= -1e-9):
        return None
    if explained[0] <= 0 or cumulative[-1] > 1.0 + 1e-6:
        return None
    return explained, cumulative


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PCA variance analysis for window embeddings",
    )
    ap.add_argument(
        "--pca-mode", type=str, default="slot",
        choices=["slot", "flat", "flat-full"],
        help="'slot' (per-position PCA, default), 'flat' (flattened-window, "
             "capped at --flat-n-components, representative set), or "
             "'flat-full' (exact full spectrum via Gram + eigvalsh over all "
             "8 files; ~18 GB RAM per ESM-C fit)",
    )
    ap.add_argument(
        "--flat-n-components", type=int, default=FLAT_N_COMPONENTS,
        help="Max components for the flat variance curve",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Recompute every curve even if a finished *_full/CSV output "
             "already exists (default: skip finished curves, recompute "
             "only missing/aborted ones)",
    )
    args = ap.parse_args()

    if args.pca_mode == "flat":
        sets = FLAT_SETS
    else:
        # slot AND flat-full: all 8 (4 modes × 2 toolkits) for parity
        sets = WINDOW_SETS

    print("=" * 65)
    print("  PCA VARIANCE ANALYSIS — Per-Residue Window Embeddings")
    print("=" * 65)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PCA mode:  {args.pca_mode}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Modes: {list(sets.keys())}")
    print("=" * 65)

    # ── Check all files exist before starting ──
    missing = []
    for key, (fname, *_) in sets.items():
        p = EMBEDDING_DIR / fname
        if not p.exists():
            missing.append(f"  {key}: {p}")
    if missing:
        print("\n  ERROR — missing input files:")
        print("\n".join(missing))
        print("\n  Transfer these files before running.")
        sys.exit(1)

    # ── Fit PCA for each version ──
    results = {}  # key -> (explained, cumulative)
    group_curves = {"esmc": {}, "esmif": {}}

    suffix = "_full" if args.pca_mode == "flat-full" else ""
    for key, (fname, ds_name, emb_dim, display_name, color, group) in sets.items():
        print(f"\n{'-' * 65}")
        print(f"  {display_name}")
        print(f"{'-' * 65}")

        raw_path = EMBEDDING_DIR / fname
        print(f"  File: {raw_path}")

        # ── Skip-if-exists guard: reuse finished curves, recompute the rest ──
        out_csv = OUT_DIR / f"pca_variance_windows_{key}{suffix}.csv"
        if args.pca_mode == "flat-full":
            expected_n = 29 * emb_dim
        elif args.pca_mode == "flat":
            expected_n = int(min(args.flat_n_components, 29 * emb_dim))
        else:
            expected_n = int(emb_dim)
        cached = None if args.force else load_existing_curve(out_csv, expected_n)
        if cached is not None:
            explained, cumulative = cached
            print(f"  SKIP (cached, {len(cumulative)} components): "
                  f"{out_csv.name}")
        elif args.pca_mode == "flat-full":
            n_feat = 29 * emb_dim
            print(f"  flat dims: {n_feat}, fitting FULL spectrum "
                  f"({n_feat} components)")
            explained, cumulative, _n_rows = fit_pca_full_spectrum(
                raw_path, ds_name)
        elif args.pca_mode == "flat":
            n_feat = 29 * emb_dim
            n_comp = int(min(args.flat_n_components, n_feat))
            print(f"  flat dims: {n_feat}, fitting n_components={n_comp}")
            explained, cumulative = fit_pca_streaming_flat(
                raw_path, ds_name, n_comp)
        else:
            explained, cumulative = fit_pca_streaming(raw_path, ds_name, emb_dim)

        thresholds_info = find_threshold_components(cumulative, THRESHOLDS)

        print(f"\n    {'Threshold':>10} {'Components':>12} {'Actual Var':>12}")
        print(f"    {'-' * 36}")
        for info in thresholds_info:
            print(f"    {info['threshold']*100:>9.0f}% "
                  f"{info['n_components']:>10} "
                  f"{info['actual_variance']*100:>10.2f}%")

        # Save individual CSV
        save_csv(explained, cumulative, key, OUT_DIR, suffix=suffix)

        # Save individual plot
        plot_individual(
            cumulative, thresholds_info,
            f"{display_name} ({args.pca_mode})", color,
            OUT_DIR / f"pca_variance_windows_{key}{suffix}.png",
        )

        results[key] = (explained, cumulative, thresholds_info)
        group_curves[group][key] = (cumulative, display_name, color)

    # ── flat-full cross-mode validation ──
    # zero/boundary/eos_bos_repeat embed identical real rows → identical
    # spectra; padtoken/pad exempt (RoPE-shifted reals, different input).
    if args.pca_mode == "flat-full":
        print(f"\n{'=' * 65}")
        print("  CROSS-MODE VALIDATION (flat-full)")
        print(f"{'=' * 65}")
        identical_sets = {
            "ESM-C": ["esmc_zeropad", "esmc_impute_boundary",
                      "esmc_impute_bos_eos"],
            "ESM-IF": ["esmif_zero", "esmif_boundary",
                       "esmif_eos_bos_repeat"],
        }
        for label, keys in identical_sets.items():
            ref_cum = results[keys[0]][1]
            for k in keys[1:]:
                diff = float(np.max(np.abs(results[k][1] - ref_cum)))
                print(f"  {label}: {keys[0]} vs {k}: max |Δcum| = {diff:.2e}")
                # 1e-6 tolerates eigvalsh thread-order noise (observed up to
                # ~4e-08 on the 14848-dim ESM-IF fit); real inconsistencies
                # (wrong file/mask) differ at ≥1e-3 scale.
                if diff > 1e-6:
                    print(f"    ERROR: identical-mode spectra diverged > 1e-6 "
                          f"— embeddings or run inconsistent")
                    sys.exit(1)
        print("  padtoken/pad exempt from identity check (shifted reals)")

    # ── Combined plots ──
    print(f"\n{'=' * 65}")
    print("  COMBINED PLOTS")
    print(f"{'=' * 65}")

    for group, label in [("esmc", "ESM-C"), ("esmif", "ESM-IF")]:
        if not group_curves[group]:
            continue
        out_path = OUT_DIR / f"pca_variance_windows_{group}_{args.pca_mode}_combined.png"
        plot_combined(
            group_curves[group],
            f"PCA Variance — {label} Window Embeddings "
            f"({args.pca_mode} pad-mode comparison)",
            out_path,
        )

    print(f"\n{'=' * 65}")
    print("  DONE")
    print(f"{'=' * 65}")
