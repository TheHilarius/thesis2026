#!/usr/bin/env python3
"""
33_fold_descriptive.py
Field-idiom fold comparison (main-text numbers, DTU / Kim et al. style).

What it computes (peptide 9-mers only, labels unused):
  - per-fold 9x20 residue-frequency matrix (unique peptides per fold)
  - mean per-position Shannon entropy (bits), averaged over 9 positions.
    Kim et al. 2014 coverage measure: low entropy = narrow coverage.
  - per-position KLD (bits) of each fold vs the pooled data, plus the
    mean over positions. Pseudocount 0.5 per AA per position so zeros
    stay finite.
  - Pearson correlation (PCC) between flattened fold frequency matrices
    (pairwise 6x6) and each fold vs pooled.

Null (cluster-preserving, shared with 34):
  Stratified shuffle of fold labels across clusters: clusters are grouped
  by cluster size (n unique peptides); within each size stratum the
  observed fold labels are randomly permuted. Whole clusters move
  together, so Hamming<=1 mates never separate, and per-fold peptide
  counts are preserved EXACTLY. One joint permutation gives all metrics,
  so the same seeds reproduce the same null in 34_c2st_simple.py.
  (Differs from 02_datasplit_analysis.py, which re-deals clusters with
  observed row-share probabilities instead of exact-count permutation.)

Usage:
    python src/investigation/33_fold_descriptive.py
    python src/investigation/33_fold_descriptive.py --n-perm 2000 --seed 42
    python src/investigation/33_fold_descriptive.py \
        --split data/processed/df_all_with_folds_h2.csv --tag h2

Outputs:
    results/tables/fold_descriptive.csv          observed per fold + null bands
    results/tables/fold_descriptive_kld_pos.csv  per-position KLD (fold, pos, obs, null99.9, flag)
    results/tables/fold_descriptive_pcc.csv      pairwise PCC observed + null bands
    results/figures/fold_similarity/descriptive/ entropy_null.png, kld_heatmap.png,
                                                 pcc_heatmap.png
    results/tables/fold_descriptive_null.npz    null arrays (shared seeds with 34)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
for _p in (PIPELINE_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, SPLIT_DATA_PATH,
    PEPTIDE_COL, FOLD_COL, HELD_OUT_INDEX,
    AMINO_ACID_ALPHABET, validate_config,
)

AA_ORDER = "".join(AMINO_ACID_ALPHABET)
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}
N_POS = 9
PSEUDO = 0.5
CLUSTER_MAP_DEFAULT = DATA_DIR / "cluster_map_peptide.csv"


# ---------------------------------------------------------------------
# helpers (imported by 34 to share the exact null)
# ---------------------------------------------------------------------

def encode9(peptides):
    return np.array([[AA_INDEX[a] for a in p] for p in peptides], dtype=np.int64)


def freq_matrix(enc):
    """enc (n,9) int -> (9,20) frequency matrix."""
    n = len(enc)
    F = np.zeros((N_POS, 20))
    for p in range(N_POS):
        F[p] = np.bincount(enc[:, p], minlength=20)
    return F / n if n else F


def freq_matrix_smooth(enc):
    """With pseudocount PSEUDO per AA per position (for KLD)."""
    n = len(enc)
    F = np.zeros((N_POS, 20))
    for p in range(N_POS):
        F[p] = np.bincount(enc[:, p], minlength=20) + PSEUDO
    return F / (n + PSEUDO * 20)


def mean_entropy(F):
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(F > 0, F * np.log2(F), 0.0)).sum(axis=1)
    return float(H.mean()), H


def kld_vs_pooled(F_fold, F_pool):
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = (np.where(F_fold > 0, F_fold * np.log2(F_fold / F_pool), 0.0)
              ).sum(axis=1)
    return float(kl.mean()), kl


def build_cluster_table(peptides, folds, cmap):
    """One row per cluster: size + observed fold. Peptides must be unique."""
    m = pd.DataFrame({"peptide": peptides, "fold": folds}).merge(
        cmap[["peptide", "cluster_id", "cluster_size"]],
        on="peptide", how="left")
    if m["cluster_id"].isna().any():
        missing = int(m.loc[m["cluster_id"].isna(), "peptide"].nunique())
        raise FileNotFoundError(
            f"{missing} peptides lack a cluster entry (stale map?). "
            f"Re-run 29_build_cluster_map.py.")
    # homology-clean check: one fold per cluster
    nun = m.groupby("cluster_id")["fold"].nunique()
    if (nun > 1).any():
        bad = int((nun > 1).sum())
        raise ValueError(f"{bad} clusters span folds (split leakage!).")
    g = m.groupby("cluster_id")
    cl = pd.DataFrame({
        "size": g["cluster_size"].first(),
        "fold": g["fold"].first(),
    }).reset_index(drop=True)
    # peptide -> cluster position map for fast expansion
    # (ngroup numbers clusters 0..C-1 in sorted-key order, matching cl
    # row order above since both use the default sorted groupby)
    m["clpos"] = m.groupby("cluster_id").ngroup()
    return cl, m["clpos"].to_numpy()


def stratified_perm_assignment(cl_folds, cl_sizes, rng):
    """Permute fold labels within each cluster-size stratum."""
    out = cl_folds.copy()
    for s in np.unique(cl_sizes):
        idx = np.where(cl_sizes == s)[0]
        out[idx] = rng.permutation(cl_folds[idx])
    return out


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Field-idiom fold comparison.")
    ap.add_argument("--split", default=None)
    ap.add_argument("--cluster-map", default=str(CLUSTER_MAP_DEFAULT))
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir",
                    default="results/figures/fold_similarity/descriptive")
    ap.add_argument("--tables", default="results/tables")
    a = ap.parse_args()

    validate_config()
    split_path = Path(a.split) if a.split else Path(SPLIT_DATA_PATH)
    tag = ("_" + a.tag) if a.tag else ""
    print("=" * 80)
    print("  33_FOLD_DESCRIPTIVE -- entropy / KLD / PCC + cluster null")
    print("=" * 80)

    df = pd.read_csv(split_path, usecols=[PEPTIDE_COL, FOLD_COL])
    df = df.drop_duplicates(PEPTIDE_COL).reset_index(drop=True)
    if not ((df[PEPTIDE_COL].str.len() == 9).all()):
        print("  WARNING: non-9-mers present; entropy/KLD assume aligned 9-mers.")
    cmap = pd.read_csv(a.cluster_map)

    peptides = df[PEPTIDE_COL].to_numpy()
    folds = df[FOLD_COL].to_numpy()
    buckets = sorted(np.unique(folds))
    print(f"  {len(peptides)} unique peptides, folds {buckets}")

    enc = encode9(peptides)
    cl, pep_clpos = build_cluster_table(peptides, folds, cmap)
    cl_folds = cl["fold"].to_numpy()
    cl_sizes = cl["size"].to_numpy()
    print(f"  {len(cl)} clusters "
          f"({(cl_sizes == 1).mean() * 100:.1f}% singletons)")

    # -- observed --
    F = {f: freq_matrix(enc[folds == f]) for f in buckets}
    Fs = {f: freq_matrix_smooth(enc[folds == f]) for f in buckets}
    Fpool = freq_matrix(enc)
    Fspool = freq_matrix_smooth(enc)
    obs_ent, obs_kld, obs_pcc_pool = {}, {}, {}
    obs_kld_pos = {}
    for f in buckets:
        e, _ = mean_entropy(F[f])
        k, kpos = kld_vs_pooled(Fs[f], Fspool)
        obs_ent[f], obs_kld[f], obs_kld_pos[f] = e, k, kpos
        v, w = F[f].ravel(), Fpool.ravel()
        obs_pcc_pool[f] = float(np.corrcoef(v, w)[0, 1])
    pcc_obs = pd.DataFrame(np.nan, index=buckets, columns=buckets)
    for i in buckets:
        for j in buckets:
            pcc_obs.loc[i, j] = float(
                np.corrcoef(F[i].ravel(), F[j].ravel())[0, 1])

    # -- null --
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    nB = a.n_perm
    null_ent = {f: np.empty(nB) for f in buckets}
    null_kld = {f: np.empty(nB) for f in buckets}
    null_kld_pos = {f: np.empty((nB, N_POS)) for f in buckets}
    null_pcc = {(i, j): np.empty(nB) for i in buckets for j in buckets}
    for b in range(nB):
        pa = stratified_perm_assignment(cl_folds, cl_sizes, rng)
        pfolds = pa[pep_clpos]
        Fp = {f: freq_matrix(enc[pfolds == f]) for f in buckets}
        Fps = {f: freq_matrix_smooth(enc[pfolds == f]) for f in buckets}
        for f in buckets:
            e, _ = mean_entropy(Fp[f])
            k, kp = kld_vs_pooled(Fps[f], Fspool)
            null_ent[f][b], null_kld[f][b] = e, k
            null_kld_pos[f][b] = kp
        for i in buckets:
            for j in buckets:
                null_pcc[(i, j)][b] = float(
                    np.corrcoef(Fp[i].ravel(), Fp[j].ravel())[0, 1])
        if (b + 1) % 100 == 0:
            print(f"  ... {b + 1}/{nB} perms")
    print(f"  null done in {time.time() - t0:.1f}s")

    # -- tables --
    tables = Path(a.tables)
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in buckets:
        rows.append({
            "fold": f,
            "n_peptides": int((folds == f).sum()),
            "mean_entropy_obs": obs_ent[f],
            "mean_entropy_null_lo": float(np.percentile(null_ent[f], 2.5)),
            "mean_entropy_null_hi": float(np.percentile(null_ent[f], 97.5)),
            "mean_kld_obs": obs_kld[f],
            "mean_kld_null_lo": float(np.percentile(null_kld[f], 2.5)),
            "mean_kld_null_hi": float(np.percentile(null_kld[f], 97.5)),
            "pcc_vs_pooled_obs": obs_pcc_pool[f],
        })
    pd.DataFrame(rows).to_csv(
        tables / f"fold_descriptive{tag}.csv", index=False)

    krows = []
    for f in buckets:
        thr = np.percentile(null_kld_pos[f], 99.9, axis=0)
        for p in range(N_POS):
            krows.append({"fold": f, "position": p + 1,
                          "kld_obs": float(obs_kld_pos[f][p]),
                          "kld_null_99_9": float(thr[p]),
                          "flag": bool(obs_kld_pos[f][p] > thr[p])})
    pd.DataFrame(krows).to_csv(
        tables / f"fold_descriptive_kld_pos{tag}.csv", index=False)

    grows = []
    for i in buckets:
        for j in buckets:
            v = null_pcc[(i, j)]
            grows.append({"fold_i": i, "fold_j": j,
                          "pcc_obs": float(pcc_obs.loc[i, j]),
                          "pcc_null_lo": float(np.percentile(v, 2.5)),
                          "pcc_null_hi": float(np.percentile(v, 97.5))})
    pd.DataFrame(grows).to_csv(
        tables / f"fold_descriptive_pcc{tag}.csv", index=False)

    np.savez_compressed(
        tables / f"fold_descriptive_null{tag}.npz",
        buckets=np.array(buckets),
        null_ent=np.array([null_ent[f] for f in buckets]),
        null_kld=np.array([null_kld[f] for f in buckets]),
        seed=np.array([a.seed]), n_perm=np.array([nB]))
    print(f"  saved tables to {tables}/fold_descriptive*{tag}.csv")

    # -- figures --
    outdir = Path(a.outdir if not a.tag else f"{a.outdir}{tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(buckets))
    o = [obs_ent[f] for f in buckets]
    lo = [np.percentile(null_ent[f], 2.5) for f in buckets]
    hi = [np.percentile(null_ent[f], 97.5) for f in buckets]
    ax.errorbar(x, o, yerr=[np.array(o) - np.array(lo),
                            np.array(hi) - np.array(o)],
                fmt="o", capsize=4, label="observed (95% null whiskers)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {f}" for f in buckets])
    ax.set_ylabel("mean per-position entropy (bits)")
    ax.set_title("Fold coverage vs cluster-preserving null")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"entropy_null{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    K = np.array([[obs_kld_pos[f][p] for p in range(N_POS)]
                  for f in buckets])
    im = ax.imshow(K, aspect="auto")
    ax.set_xticks(range(N_POS))
    ax.set_xticklabels([f"P{i + 1}" for i in range(N_POS)])
    ax.set_yticks(range(len(buckets)))
    ax.set_yticklabels([f"fold {f}" for f in buckets])
    ax.set_title("KLD vs pooled per position (bits)")
    for i in range(len(buckets)):
        for j in range(N_POS):
            ax.text(j, i, f"{K[i, j]:.3f}", ha="center", va="center",
                    fontsize=7)
    fig.colorbar(im, ax=ax, label="bits")
    fig.tight_layout()
    fig.savefig(outdir / f"kld_heatmap{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    M = pcc_obs.to_numpy(dtype=float)
    im = ax.imshow(M, vmin=0.99, vmax=1.0)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f"{f}" for f in buckets])
    ax.set_yticks(range(len(buckets)))
    ax.set_yticklabels([f"{f}" for f in buckets])
    ax.set_title("PCC between fold frequency matrices")
    for i in range(len(buckets)):
        for j in range(len(buckets)):
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="PCC")
    fig.tight_layout()
    fig.savefig(outdir / f"pcc_heatmap{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {outdir}/entropy_null{tag}.png, "
          f"kld_heatmap{tag}.png, pcc_heatmap{tag}.png")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
