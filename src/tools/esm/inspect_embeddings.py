#!/usr/bin/env python3
"""
inspect_embeddings.py

Summarise and visualise ESM protein embeddings (ESM-C or ESM-IF) stored in HDF5.
Supports both the current single context-embedding format and the legacy
3-region (peptide / n_flank / c_flank) format.

Usage:
    python src/tools/esm/inspect_embeddings.py \
        data/processed/embeddings/esmc_protein_embeddings.h5 \
        data/processed/df_all.csv \
        --out_dir results/embedding_inspection/esmc

    python src/tools/esm/inspect_embeddings.py \
        data/processed/embeddings/esmif_structure_embeddings.h5 \
        data/processed/df_all.csv \
        --out_dir results/embedding_inspection/esmif
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inspect ESM embedding HDF5 file")
parser.add_argument("h5_path",  help="HDF5 embedding file")
parser.add_argument("csv_path", help="Original CSV (for labels, metadata)")
parser.add_argument("--out_dir", default="results/embedding_inspection",
                    help="Directory for output plots")
parser.add_argument("--max_tsne", type=int, default=5000,
                    help="Max samples for t-SNE (subsampled for speed)")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load HDF5 ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  ESM Embedding Inspection")
print("=" * 65)

print(f"\nLoading {args.h5_path}...")
with h5py.File(args.h5_path, "r") as f:

    # ── File-level metadata ───────────────────────────────────────────────
    print("\n── HDF5 Metadata ──────────────────────────────────────────")
    for key, val in f.attrs.items():
        print(f"  {key:<22}: {val}")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\n── Datasets ───────────────────────────────────────────────")
    available_keys = list(f.keys())
    for key in available_keys:
        ds = f[key]
        print(f"  {key:<24}: shape={ds.shape}  dtype={ds.dtype}")

    # ── Auto-detect format ────────────────────────────────────────────────
    if "context_emb" in available_keys:
        MODEL_TYPE = "ESM-C"
        KEY_CTX     = "context_emb"
        KEY_PEPSEQ  = "peptide_seqs"
    elif "context_if_struct" in available_keys:
        MODEL_TYPE  = "ESM-IF"
        KEY_CTX     = "context_if_struct"
        KEY_PEPSEQ  = "peptide_ids"
    elif "peptide_emb" in available_keys:
        # Legacy 3-region ESM-C format
        MODEL_TYPE = "ESM-C (legacy 3-region)"
        KEY_CTX    = None
        KEY_PEPSEQ = "peptide_seqs"
    elif "peptide_if_struct" in available_keys:
        # Legacy 3-region ESM-IF format
        MODEL_TYPE = "ESM-IF (legacy 3-region)"
        KEY_CTX    = None
        KEY_PEPSEQ = "peptide_ids"
    else:
        raise ValueError(
            f"Cannot detect embedding format. "
            f"Available keys: {available_keys}\n"
            f"Expected 'context_emb' / 'context_if_struct' (single) or "
            f"'peptide_emb' / 'peptide_if_struct' (legacy 3-region)")

    print(f"\n  Detected format: {MODEL_TYPE}")

    # ── Load arrays ───────────────────────────────────────────────────────
    if KEY_CTX is not None:
        emb_dict = {"context": f[KEY_CTX][:]}
    elif MODEL_TYPE.startswith("ESM-C"):
        emb_dict = {
            "peptide": f["peptide_emb"][:],
            "n_flank": f["n_flank_emb"][:],
            "c_flank": f["c_flank_emb"][:],
        }
    else:
        emb_dict = {
            "peptide": f["peptide_if_struct"][:],
            "n_flank": f["n_flank_if_struct"][:],
            "c_flank": f["c_flank_if_struct"][:],
        }

    pep_seqs = np.array([s.decode() for s in f[KEY_PEPSEQ][:]])
    uid_seqs = np.array([s.decode() for s in f["uniprot_ids"][:]])

    # Optional fields
    has_fallback = "fallback_flag" in available_keys
    fallback     = f["fallback_flag"][:] if has_fallback else None

    has_row_idx  = "row_indices" in available_keys
    row_idx      = f["row_indices"][:] if has_row_idx else None

primary_name = next(iter(emb_dict))
primary_emb  = emb_dict[primary_name]
n_samples, emb_dim = primary_emb.shape
n_regions = len(emb_dict)
print(f"\n  Loaded {n_samples} samples, embedding dim = {emb_dim}, "
      f"{n_regions} region(s)")
print(f"  Model type: {MODEL_TYPE}")

# ── Load CSV for labels ───────────────────────────────────────────────────────
print(f"\nLoading {args.csv_path}...")
df = pd.read_csv(args.csv_path)

# Align rows: use row_indices if available, otherwise assume 1:1 alignment
if has_row_idx:
    labels      = df.loc[row_idx, "label"].values.astype(int)
    pep_lengths = df.loc[row_idx, "pep_length"].values.astype(int)
else:
    # ESM-IF file may not have row_indices — match by position
    # (only valid if the CSV was not reordered relative to the embedding)
    if len(df) == n_samples:
        labels      = df["label"].values.astype(int)
        pep_lengths = df["pep_length"].values.astype(int)
    else:
        # Fallback: match on peptide + uniprot_id
        print("  Row count mismatch — matching by peptide + uniprot_id...")
        df["_key"] = df["peptide"].astype(str) + "|" + df["uniprot_id"].astype(str)
        h5_keys    = [f"{p}|{u}" for p, u in zip(pep_seqs, uid_seqs)]

        key_to_idx = {k: i for i, k in enumerate(df["_key"])}
        matched_idx = []
        unmatched   = 0
        for k in h5_keys:
            if k in key_to_idx:
                matched_idx.append(key_to_idx[k])
            else:
                matched_idx.append(None)
                unmatched += 1

        if unmatched > 0:
            print(f"  ⚠  {unmatched}/{n_samples} H5 rows could not be matched to CSV")

        # Build aligned arrays (use -1 for unmatched, filter later)
        valid_mask  = np.array([m is not None for m in matched_idx])
        matched_idx = np.array([m if m is not None else 0 for m in matched_idx])
        labels      = df.loc[matched_idx, "label"].values.astype(int)
        pep_lengths = df.loc[matched_idx, "pep_length"].values.astype(int)

        # Zero out unmatched
        labels[~valid_mask]      = -1
        pep_lengths[~valid_mask] = -1

        n_valid = valid_mask.sum()
        print(f"  Matched {n_valid}/{n_samples} samples")
        df.drop(columns=["_key"], inplace=True)

print(f"  Positives (label=1) : {(labels == 1).sum()}")
print(f"  Negatives (label=0) : {(labels == 0).sum()}")
if (labels == -1).any():
    print(f"  Unmatched           : {(labels == -1).sum()}")

# Filter to valid labels for all analyses
valid = labels >= 0
if not valid.all():
    print(f"  Restricting analyses to {valid.sum()} matched samples")
    emb_dict = {k: v[valid] for k, v in emb_dict.items()}
    labels      = labels[valid]
    pep_lengths = pep_lengths[valid]
    pep_seqs    = pep_seqs[valid]
    uid_seqs    = uid_seqs[valid]
    if fallback is not None:
        fallback = fallback[valid]
    n_samples = valid.sum()

primary_emb = emb_dict[primary_name]

# ═════════════════════════════════════════════════════════════════════════════
#  1. BASIC EMBEDDING STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  1. Basic Embedding Statistics")
print("=" * 65)

for name, emb in emb_dict.items():
    norms  = np.linalg.norm(emb, axis=1)
    n_zero = (norms == 0).sum()
    nonzero = emb[norms > 0]

    print(f"\n  ── {name} embeddings ({emb.shape}) ──")
    print(f"    Zero vectors       : {n_zero} / {len(emb)}"
          f"  ({100*n_zero/len(emb):.1f}%)")
    print(f"    Value range        : [{emb.min():.4f}, {emb.max():.4f}]")
    print(f"    Mean (all dims)    : {emb.mean():.6f}")
    print(f"    Std  (all dims)    : {emb.std():.6f}")
    print(f"    L2 norm  min/med/max : "
          f"{norms.min():.3f} / {np.median(norms):.3f} / {norms.max():.3f}")
    if len(nonzero) > 0:
        print(f"    Per-dim mean range : "
              f"[{nonzero.mean(axis=0).min():.4f}, "
              f"{nonzero.mean(axis=0).max():.4f}]")
        print(f"    Per-dim std range  : "
              f"[{nonzero.std(axis=0).min():.4f}, "
              f"{nonzero.std(axis=0).max():.4f}]")

# ── Fallback summary ──────────────────────────────────────────────────────────
if fallback is not None:
    print(f"\n  ── Fallback flags ──")
    print(f"    Full protein       : {(fallback == 0).sum()}")
    print(f"    Windowed fallback  : {(fallback == 1).sum()}")
else:
    print(f"\n  ── Fallback flags: not available ({MODEL_TYPE}) ──")

# ═════════════════════════════════════════════════════════════════════════════
#  2. L2 NORM DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  2. Plotting L2 norm distributions...")
print("=" * 65)

fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4),
                         squeeze=False)
fig.suptitle(f"{MODEL_TYPE}: L2 Norm Distributions", fontsize=13, y=1.02)
for ax, (name, emb) in zip(axes[0], emb_dict.items()):
    norms     = np.linalg.norm(emb, axis=1)
    norms_pos = norms[labels == 1]
    norms_neg = norms[labels == 0]

    ax.hist(norms_pos, bins=60, alpha=0.6, label=f"Positive (n={len(norms_pos)})",
            density=True, color="steelblue")
    ax.hist(norms_neg, bins=60, alpha=0.6, label=f"Negative (n={len(norms_neg)})",
            density=True, color="salmon")
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}")
    ax.legend(fontsize=8)

    stat, pval = mannwhitneyu(norms_pos, norms_neg, alternative="two-sided")
    ax.text(0.98, 0.95, f"MW p={pval:.2e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(out_dir / "l2_norm_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'l2_norm_distributions.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  3. PER-DIMENSION MEAN COMPARISON (POS vs NEG)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  3. Per-dimension mean difference (positive vs negative)...")
print("=" * 65)

fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4),
                         squeeze=False)
fig.suptitle(f"{MODEL_TYPE}: Per-Dimension Mean Difference", fontsize=13, y=1.02)
for ax, (name, emb) in zip(axes[0], emb_dict.items()):
    mean_pos = emb[labels == 1].mean(axis=0)
    mean_neg = emb[labels == 0].mean(axis=0)
    diff     = mean_pos - mean_neg

    ax.bar(range(emb_dim), diff, width=1.0, color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Mean(pos) − Mean(neg)")
    ax.set_title(f"{name}")
    ax.text(0.98, 0.95,
            f"max |Δ| = {np.abs(diff).max():.4f}\n"
            f"mean |Δ| = {np.abs(diff).mean():.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(out_dir / "per_dim_mean_diff.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'per_dim_mean_diff.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  4. CROSS-REGION COSINE SIMILARITY (legacy 3-region format only)
# ═════════════════════════════════════════════════════════════════════════════
if n_regions == 3:
    print("\n" + "=" * 65)
    print("  4. Cosine similarity between peptide / n_flank / c_flank...")
    print("=" * 65)

    pep_emb, nf_emb, cf_emb = (emb_dict["peptide"], emb_dict["n_flank"],
                               emb_dict["c_flank"])
    pairs = [
        ("peptide↔n_flank", pep_emb, nf_emb),
        ("peptide↔c_flank", pep_emb, cf_emb),
        ("n_flank↔c_flank", nf_emb,  cf_emb),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{MODEL_TYPE}: Cross-Region Cosine Similarity", fontsize=13, y=1.02)
    for ax, (pair_name, emb_a, emb_b) in zip(axes, pairs):
        norms_a = np.linalg.norm(emb_a, axis=1)
        norms_b = np.linalg.norm(emb_b, axis=1)
        valid   = (norms_a > 0) & (norms_b > 0)

        cos_sim = np.sum(emb_a[valid] * emb_b[valid], axis=1) / (
                  norms_a[valid] * norms_b[valid])

        cos_pos = cos_sim[labels[valid] == 1]
        cos_neg = cos_sim[labels[valid] == 0]

        ax.hist(cos_pos, bins=60, alpha=0.6, label="Positive", density=True,
                color="steelblue")
        ax.hist(cos_neg, bins=60, alpha=0.6, label="Negative", density=True,
                color="salmon")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title(pair_name)
        ax.legend(fontsize=8)

        print(f"  {pair_name:<22}: "
              f"median(pos)={np.median(cos_pos):.3f}  "
              f"median(neg)={np.median(cos_neg):.3f}  "
              f"(n_valid={valid.sum()})")

    plt.tight_layout()
    plt.savefig(out_dir / "cosine_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'cosine_similarity.png'}")
else:
    print("\n" + "=" * 65)
    print("  4. Cross-region cosine similarity: skipped (single embedding)")
    print("=" * 65)

# ═════════════════════════════════════════════════════════════════════════════
#  5. INTER-SAMPLE COSINE SIMILARITY (WITHIN-CLASS vs BETWEEN-CLASS)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  5. Inter-sample cosine similarity (sampling 2000 pairs)...")
print("=" * 65)

rng = np.random.default_rng(42)
n_pairs = 2000

pos_idx = np.where(labels == 1)[0]
neg_idx = np.where(labels == 0)[0]

def sample_cosines(idx_a, idx_b, emb, n=n_pairs):
    """Sample n random pairs and compute cosine similarities."""
    i = rng.choice(idx_a, size=n, replace=True)
    j = rng.choice(idx_b, size=n, replace=True)
    dot  = np.sum(emb[i] * emb[j], axis=1)
    norm = np.linalg.norm(emb[i], axis=1) * np.linalg.norm(emb[j], axis=1)
    mask = norm > 0
    return dot[mask] / norm[mask]

fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 4),
                         squeeze=False)
fig.suptitle(f"{MODEL_TYPE}: Inter-Sample Cosine Similarity", fontsize=13, y=1.02)
for ax, (name, emb) in zip(axes[0], emb_dict.items()):
    cos_pp = sample_cosines(pos_idx, pos_idx, emb)
    cos_nn = sample_cosines(neg_idx, neg_idx, emb)
    cos_pn = sample_cosines(pos_idx, neg_idx, emb)

    ax.hist(cos_pp, bins=50, alpha=0.5, label="pos↔pos", density=True)
    ax.hist(cos_nn, bins=50, alpha=0.5, label="neg↔neg", density=True)
    ax.hist(cos_pn, bins=50, alpha=0.5, label="pos↔neg", density=True)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}")
    ax.legend(fontsize=8)

    print(f"  {name}: pos↔pos={np.median(cos_pp):.3f}  "
          f"neg↔neg={np.median(cos_nn):.3f}  "
          f"pos↔neg={np.median(cos_pn):.3f}")

plt.tight_layout()
plt.savefig(out_dir / "inter_sample_cosine.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'inter_sample_cosine.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  6. PCA — VARIANCE EXPLAINED + SCATTER
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  6. PCA analysis...")
print("=" * 65)

fig = plt.figure(figsize=(5 * max(n_regions, 1) + 1, 10))
fig.suptitle(f"{MODEL_TYPE}: PCA Analysis", fontsize=14, y=1.01)
gs  = gridspec.GridSpec(2, n_regions, figure=fig, hspace=0.35, wspace=0.3)

for col, (name, emb) in enumerate(emb_dict.items()):
    norms = np.linalg.norm(emb, axis=1)
    valid_mask = norms > 0
    emb_v = emb[valid_mask]
    lab_v = labels[valid_mask]

    n_components = min(50, emb_v.shape[1], emb_v.shape[0])
    pca = PCA(n_components=n_components)
    pc  = pca.fit_transform(emb_v)

    # Top row: variance explained
    ax_var = fig.add_subplot(gs[0, col])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax_var.plot(range(1, len(cumvar)+1), cumvar, "o-", markersize=3)
    ax_var.set_xlabel("# PCs")
    ax_var.set_ylabel("Cumulative variance (%)")
    ax_var.set_title(f"{name}: PCA variance")
    ax_var.axhline(90, color="red", linestyle="--", alpha=0.5)

    for thresh in [50, 75, 90]:
        idx_t = np.searchsorted(cumvar, thresh)
        n_pc  = idx_t + 1 if idx_t < len(cumvar) else len(cumvar)
        ax_var.text(0.98, 0.05 + (thresh-50)*0.012,
                    f"{thresh}% → {n_pc} PCs",
                    transform=ax_var.transAxes, ha="right", fontsize=7)
        print(f"  {name}: {thresh}% variance in {n_pc} PCs")

    # Bottom row: PC1 vs PC2 scatter
    ax_sc = fig.add_subplot(gs[1, col])
    for lab, color, lname in [(0, "salmon", "Neg"), (1, "steelblue", "Pos")]:
        mask = lab_v == lab
        ax_sc.scatter(pc[mask, 0], pc[mask, 1], c=color, alpha=0.15,
                      s=5, label=lname, rasterized=True)
    ax_sc.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_sc.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_sc.set_title(f"{name}: PC1 vs PC2")
    ax_sc.legend(fontsize=8, markerscale=3)

plt.savefig(out_dir / "pca_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'pca_analysis.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  7. t-SNE (primary embedding, subsampled)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  7. t-SNE on {primary_name} embeddings...")
print("=" * 65)

norms_pri = np.linalg.norm(primary_emb, axis=1)
valid_pri = norms_pri > 0
n_valid   = valid_pri.sum()
n_tsne    = min(args.max_tsne, n_valid)

sub_idx  = rng.choice(np.where(valid_pri)[0], size=n_tsne, replace=False)
sub_emb  = primary_emb[sub_idx]
sub_lab  = labels[sub_idx]

print(f"  Running t-SNE on {n_tsne} samples (perplexity=30)...")
n_pca_pre = min(50, sub_emb.shape[1], sub_emb.shape[0])
pca_pre   = PCA(n_components=n_pca_pre)
sub_pc    = pca_pre.fit_transform(sub_emb)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_coords = tsne.fit_transform(sub_pc)

fig, ax = plt.subplots(figsize=(8, 7))
for lab, color, lname in [(0, "salmon", "Negative"), (1, "steelblue", "Positive")]:
    mask = sub_lab == lab
    ax.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
               c=color, alpha=0.3, s=8, label=lname, rasterized=True)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title(f"{MODEL_TYPE}: t-SNE of {primary_name} embeddings (n={n_tsne})")
ax.legend(markerscale=3)
plt.savefig(out_dir / "tsne_peptide.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'tsne_peptide.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  8. DIMENSION ACTIVATION HEATMAP (top discriminative dims)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  8. Top discriminative dimensions (Mann-Whitney per dim)...")
print("=" * 65)

fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions + 1, 5),
                         squeeze=False)
fig.suptitle(f"{MODEL_TYPE}: Top Discriminative Dimensions", fontsize=13, y=1.02)
for ax, (name, emb) in zip(axes[0], emb_dict.items()):
    pvals  = np.zeros(emb_dim)
    effect = np.zeros(emb_dim)

    pos_data = emb[labels == 1]
    neg_data = emb[labels == 0]
    n1, n2   = len(pos_data), len(neg_data)

    for d in range(emb_dim):
        stat, pv = mannwhitneyu(pos_data[:, d], neg_data[:, d],
                                alternative="two-sided")
        pvals[d]  = pv
        effect[d] = 1 - 2 * stat / (n1 * n2)

    sig = pvals < (0.05 / emb_dim)  # Bonferroni
    top_k = 20
    top_dims = np.argsort(np.abs(effect))[-top_k:][::-1]

    ax.barh(range(top_k), effect[top_dims], color=[
        "steelblue" if effect[d] > 0 else "salmon" for d in top_dims
    ], alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"dim {d}" + (" *" if sig[d] else "")
                        for d in top_dims], fontsize=7)
    ax.set_xlabel("Rank-biserial r (pos vs neg)")
    ax.set_title(f"{name}")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    n_sig = sig.sum()
    print(f"  {name}: {n_sig}/{emb_dim} dims significant (Bonferroni p<0.05)")
    print(f"    Top 5 effect sizes: "
          f"{', '.join(f'd{d}={effect[d]:.4f}' for d in top_dims[:5])}")

plt.tight_layout()
plt.savefig(out_dir / "top_discriminative_dims.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'top_discriminative_dims.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  9. EMBEDDING NORM vs PEPTIDE LENGTH
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  9. Embedding norm vs peptide length...")
print("=" * 65)

fig, ax = plt.subplots(figsize=(8, 5))
pri_norms   = np.linalg.norm(primary_emb, axis=1)
unique_lens = sorted(np.unique(pep_lengths[pep_lengths > 0]))

bp_data = [pri_norms[pep_lengths == l] for l in unique_lens]
# Filter out empty groups
valid_lens = [l for l, d in zip(unique_lens, bp_data) if len(d) > 0]
bp_data    = [d for d in bp_data if len(d) > 0]

if bp_data:
    bp = ax.boxplot(bp_data, positions=valid_lens, widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Peptide length")
    ax.set_ylabel(f"{primary_name} embedding L2 norm")
    ax.set_title(f"{MODEL_TYPE}: Embedding norm vs peptide length")

    for l in valid_lens:
        n = (pep_lengths == l).sum()
        ax.text(l, ax.get_ylim()[0], f"n={n}", ha="center", va="top", fontsize=7)

plt.savefig(out_dir / "norm_vs_length.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_dir / 'norm_vs_length.png'}")

# ═════════════════════════════════════════════════════════════════════════════
#  10. SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  10. Summary")
print("=" * 65)

summary = []
for name, emb in emb_dict.items():
    norms  = np.linalg.norm(emb, axis=1)
    n_zero = (norms == 0).sum()
    summary.append({
        "model":        MODEL_TYPE,
        "region":       name,
        "n_samples":    len(emb),
        "emb_dim":      emb.shape[1],
        "n_zero_vecs":  n_zero,
        "norm_median":  f"{np.median(norms):.3f}",
        "norm_std":     f"{norms.std():.3f}",
        "value_mean":   f"{emb.mean():.6f}",
        "value_std":    f"{emb.std():.6f}",
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(out_dir / "embedding_summary.csv", index=False)
print(f"\n  Saved: {out_dir / 'embedding_summary.csv'}")

print(f"\n{'=' * 65}")
print(f"  All outputs saved to: {out_dir}/")
print(f"{'=' * 65}")
print(f"\nFiles:")
for p in sorted(out_dir.glob("*")):
    print(f"  {p.name}")


# ── Structure coverage (per protein) ──────────────────────────────────────────
df_cov = pd.DataFrame({
    "uniprot": uid_seqs,
    "zero": np.linalg.norm(primary_emb, axis=1) == 0
})

protein_cov = df_cov.groupby("uniprot")["zero"].mean()

print(protein_cov.describe())
print("\nProteins with 100% missing structure:", (protein_cov == 1).sum())
print("Proteins with any structure:", (protein_cov < 1).sum())

row_missing = np.linalg.norm(primary_emb, axis=1) == 0

print("Row-level missing %:", row_missing.mean())
print("Rows missing:", row_missing.sum())

df_tmp = pd.DataFrame({
    "uniprot": uid_seqs,
    "missing": row_missing
})

protein_missing_rate = df_tmp.groupby("uniprot")["missing"].mean()

print(protein_missing_rate.describe())
print("Proteins fully missing:", (protein_missing_rate == 1).sum())
