#!/usr/bin/env python3
"""
08_pca_variance_analysis.py
Fit PCA on full context window embeddings and plot cumulative explained
variance vs number of components. Informs how many PCs to use for modelling.

Input:
    data/processed/embeddings/{esmc,esmif}_context_embeddings.h5
    (expects 'context_emb' / 'context_if_struct' dataset — mean-pooled vector per sample)

Output:
    results/figures/models/pca_variance_analysis.png
    results/figures/models/pca_variance_components_{esmc,esmif}.csv
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from datetime import datetime

from config import EMBEDDING_DIR, FIGURES_DIR

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_SETS = {
    "esmc": {
        "display_name": "ESM-C (600M)",
        "emb_key": "context_emb",
        "emb_dim": 1152,
    },
    "esmif": {
        "display_name": "ESM-IF1",
        "emb_key": "context_if_struct",
        "emb_dim": 512,
    },
}

THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]

OUT_DIR = FIGURES_DIR / "pca_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_embeddings(feature_key):
    """Load full context embeddings from raw HDF5."""
    cfg = FEATURE_SETS[feature_key]
    h5_path = EMBEDDING_DIR / f"{feature_key}_context_embeddings.h5"

    print(f"\n  Loading {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        emb_key = cfg["emb_key"]

        if emb_key not in f:
            available = [k for k in f.keys() if k not in
                        ("fallback_flag", "peptide_seqs", "uniprot_ids",
                         "row_indices", "start", "end")]
            raise KeyError(
                f"'{emb_key}' not found in {h5_path}.\n"
                f"  Available datasets: {available}\n"
                f"  Run extraction scripts first to produce "
                f"'{emb_key}'."
            )

        X = f[emb_key][:]

        labels = f["labels"][:] if "labels" in f else None
        folds = f["folds"][:] if "folds" in f else None

    print(f"  Shape: {X.shape}")
    return X, labels, folds


# ── Remove zero vectors ──────────────────────────────────────────────────────
def remove_zeros(X):
    """Remove rows where norm == 0."""
    norms = np.linalg.norm(X, axis=1)
    mask = norms > 0.0
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  Removed {n_removed} zero-vector rows "
              f"({n_removed / len(X) * 100:.1f}%)")
    return X[mask], mask


# ── Fit PCA and compute variance ─────────────────────────────────────────────
def fit_pca_full(X, feature_key):
    """Fit PCA with all components, return explained variance ratios."""
    cfg = FEATURE_SETS[feature_key]
    n_components = min(cfg["emb_dim"], X.shape[0])

    print(f"  Fitting PCA with {n_components} components "
          f"(data: {X.shape[0]} samples × {X.shape[1]} features)...")

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"  Done. Top 10 components explain "
          f"{cumulative[9] * 100:.1f}% variance.")

    return explained, cumulative, pca


# ── Find n_components for thresholds ──────────────────────────────────────────
def find_threshold_components(cumulative, thresholds):
    """Find n_components needed for each variance threshold."""
    results = []
    for t in thresholds:
        n = np.searchsorted(cumulative, t) + 1  # +1 because 1-indexed
        n = min(n, len(cumulative))
        actual_var = cumulative[n - 1]
        results.append({
            "threshold": t,
            "n_components": n,
            "actual_variance": actual_var,
        })
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_variance_analysis(results_all, out_path):
    """Create combined 2-panel plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"esmc": "#e74c3c", "esmif": "#3498db"}

    for ax, (feature_key, data) in zip(axes, results_all.items()):
        cumulative = data["cumulative"]
        thresholds_info = data["thresholds"]
        cfg = FEATURE_SETS[feature_key]

        x = np.arange(1, len(cumulative) + 1)

        # Main line
        ax.plot(x, cumulative, color=colors[feature_key],
                linewidth=1.5, alpha=0.9)

        # Fill under curve
        ax.fill_between(x, cumulative, alpha=0.1, color=colors[feature_key])

        # Threshold lines
        for info in thresholds_info:
            t = info["threshold"]
            n = info["n_components"]
            actual = info["actual_variance"]

            ax.axhline(y=t, color="grey", linestyle="--",
                       linewidth=0.7, alpha=0.5)
            ax.axvline(x=n, color=colors[feature_key],
                       linestyle=":", linewidth=0.8, alpha=0.6)

            # Annotate
            label = f"{t*100:.0f}% → {n} PCs"
            ax.annotate(
                label,
                xy=(n, actual),
                xytext=(n + 15, actual - 0.03),
                fontsize=7,
                color="grey",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            )

        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title(f"{cfg['display_name']}")
        ax.set_xlim(0, len(cumulative) + 10)
        ax.set_ylim(0, 1.05)

        # Smart tick spacing
        max_pc = len(cumulative)
        if max_pc > 500:
            tick_step = 100
        elif max_pc > 200:
            tick_step = 50
        else:
            tick_step = 25
        ax.set_xticks(range(0, max_pc + 1, tick_step))
        ax.tick_params(axis='x', rotation=45)

        ax.grid(True, alpha=0.3)

    fig.suptitle("PCA Variance Analysis — Full Context Window Embeddings",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ── Save CSV ──────────────────────────────────────────────────────────────────
def save_csv(explained, cumulative, feature_key, out_dir):
    """Save per-component variance to CSV."""
    csv_path = out_dir / f"pca_variance_components_{feature_key}.csv"

    with open(csv_path, "w") as f:
        f.write("component_idx,explained_variance_ratio,cumulative_variance\n")
        for i, (e, c) in enumerate(zip(explained, cumulative)):
            f.write(f"{i + 1},{e:.8f},{c:.8f}\n")

    print(f"  Saved: {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PCA VARIANCE ANALYSIS — Full Context Window")
    print("=" * 60)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Feature sets: {list(FEATURE_SETS.keys())}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    results_all = {}

    for feature_key in FEATURE_SETS:
        print(f"\n{'─' * 60}")
        print(f"  {FEATURE_SETS[feature_key]['display_name']}")
        print(f"{'─' * 60}")

        # Load
        X, labels, folds = load_embeddings(feature_key)

        # Remove zeros
        X_clean, zero_mask = remove_zeros(X)

        # Fit PCA
        explained, cumulative, pca = fit_pca_full(X_clean, feature_key)

        # Find thresholds
        thresholds_info = find_threshold_components(cumulative, THRESHOLDS)

        print(f"\n  Threshold results:")
        print(f"  {'Threshold':>10} {'Components':>12} {'Actual Var':>12}")
        print(f"  {'─' * 36}")
        for info in thresholds_info:
            print(f"  {info['threshold']*100:>9.0f}% "
                  f"{info['n_components']:>10} "
                  f"{info['actual_variance']*100:>10.2f}%")

        # Save CSV
        save_csv(explained, cumulative, feature_key, OUT_DIR)

        results_all[feature_key] = {
            "explained": explained,
            "cumulative": cumulative,
            "thresholds": thresholds_info,
        }

    # Combined plot
    out_png = OUT_DIR / "pca_variance_analysis.png"
    plot_variance_analysis(results_all, out_png)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")
