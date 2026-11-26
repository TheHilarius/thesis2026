#!/usr/bin/env python3
"""
31_random_peptide_logo.py — score 1M random 29-mers with the baseline
BLOSUM50 XGB ensemble and slice the confident extremes for sequence logos.

Question: what patterns has the model learned? Score 1M random context
windows (uniform over 20 AAs, none seen in training), take the top 10%
(most "presented") and bottom 10% (least "presented"), and hand per-position
PPMs to the logo scripts (31_random_peptide_logos.R / DiffLogo).

Model: the nested-CV inner-fold pickles of the encodings_sweep baseline
(xgb_blosum50_flat, lr=0.1, n=1000, depth=6, 609 features) — ensemble
average of predict_proba over all 30 pickles, same averaging the pipeline
uses on validation buckets (04_modelling.predict_with_averaging).

Outputs (runs/random_peptide_logo/):
  scores.csv.gz        sequence, prob (1M rows)
  top10.csv            100k highest-prob sequences
  bottom10.csv         100k lowest-prob sequences
  ppm_top10.csv        20x29 position probability matrix (top)
  ppm_bottom10.csv     20x29 (bottom)
  train_scores.csv.gz  training-set scores (sanity reference)
  score_hist.png       random vs training score distributions

Usage:  python src/investigation/31_random_peptide_logo.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import PROJECT_ROOT

OUT = PROJECT_ROOT / "runs" / "random_peptide_logo"

# Two model regimes:
#   29mer : baseline encodings_sweep blosum50 flat (lr=0.1), N10..C10 window
#   flank5_rmdup : runs/r19_rmdup/canonical blosum50 flat (lr=0.1), N5..C5 window
MODEL_REGIMES = {
    "29mer": {
        "model_dir": PROJECT_ROOT / "runs" / "encodings_sweep" / "models" / "flat",
        "train_csv": PROJECT_ROOT / "data" / "processed" / "df_all.csv",
        "out_base": "random_peptide_logo",
    },
    "flank5_rmdup": {
        "model_dir": PROJECT_ROOT / "runs" / "r19_rmdup" / "canonical" / "models" / "flat",
        "train_csv": PROJECT_ROOT / "data" / "processed" / "df_all_flank5_rmdup.csv",
        "out_base": "random_peptide_logo_rmdup",
    },
}

N_SEQS = 1_000_000
SEED = 20261108
CHUNK = 100_000

AA_ORDER = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
POSITIONS = ([f"N{i}" for i in range(10, 0, -1)] +
             [f"P{i}" for i in range(1, 10)] +
             [f"C{i}" for i in range(1, 11)])

# BLOSUM50 verbatim from src/pipeline/r/functions.R:1269 (NCBI, NetMHCpan
# encoding; NA -> all-zero vector, irrelevant here since we only emit AAs).
_BLOSUM50_ROWS = [
    [5, -1, -2, -1, -3, 0, -2, -1, -1, -2, -1, -1, -1, -1, -2, 1, 0, 0, -3, -2, -1],
    [-1, 13, -4, -3, -2, -3, -3, -2, -3, -2, -2, -2, -4, -3, -4, -1, -1, -1, -5, -3, -1],
    [-2, -4, 8, 2, -5, -1, -1, -4, -1, -4, -4, 2, -1, 0, -2, 0, -1, -4, -5, -3, -1],
    [-1, -3, 2, 6, -3, -3, 0, -4, 1, -3, -2, 0, -1, 2, 0, -1, -1, -3, -3, -2, -1],
    [-3, -2, -5, -3, 8, -4, -1, 0, -4, 1, 0, -4, -4, -4, -3, -3, -2, -1, 1, 4, -1],
    [0, -3, -1, -3, -4, 8, -2, -4, -2, -4, -3, 0, -2, -2, -3, 0, -2, -4, -3, -3, -1],
    [-2, -3, -1, 0, -1, -2, 10, -4, 0, -3, -1, 1, -2, 1, 0, -1, -2, -4, -3, 2, -1],
    [-1, -2, -4, -4, 0, -4, -4, 5, -3, 2, 2, -3, -3, -3, -4, -3, -1, 4, -3, -1, -1],
    [-1, -3, -1, 1, -4, -2, 0, -3, 6, -3, -2, 0, -1, 2, 3, 0, -1, -3, -3, -2, -1],
    [-2, -2, -4, -3, 1, -4, -3, 2, -3, 5, 3, -4, -4, -2, -3, -3, -1, 1, -2, -1, -1],
    [-1, -2, -4, -2, 0, -3, -1, 2, -2, 3, 7, -2, -3, 0, -2, -2, -1, 1, -1, 0, -1],
    [-1, -2, 2, 0, -4, 0, 1, -3, 0, -4, -2, 7, -2, 0, -1, 1, 0, -3, -4, -2, -1],
    [-1, -4, -1, -1, -4, -2, -2, -3, -1, -4, -3, -2, 10, -1, -3, -1, -1, -3, -4, -3, -1],
    [-1, -3, 0, 2, -4, -2, 1, -3, 2, -2, 0, 0, -1, 7, 1, 0, -1, -3, -1, -1, -1],
    [-2, -4, -2, 0, -3, -3, 0, -4, 3, -3, -2, -1, -3, 1, 7, -1, -1, -3, -3, -1, -1],
    [1, -1, 0, -1, -3, 0, -1, -3, 0, -3, -2, 1, -1, 0, -1, 5, 2, -2, -4, -2, -1],
    [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 2, 5, 0, -3, -2, -1],
    [0, -1, -4, -3, -1, -4, -4, 4, -3, 1, 1, -3, -3, -3, -3, -2, 0, 5, -3, -1, -1],
    [-3, -5, -5, -3, 1, -3, -3, -3, -3, -2, -1, -4, -4, -1, -3, -4, -3, -3, 15, 2, -1],
    [-2, -3, -3, -2, 4, -3, 2, -1, -2, -1, 0, -2, -3, -1, -1, -2, -2, -1, 2, 8, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]
BLOSUM50 = np.asarray(_BLOSUM50_ROWS, dtype=np.float32)  # rows = AA_ORDER


def parse_positions(feature_names: list[str]) -> list[str]:
    pos = []
    for c in feature_names:
        p = c.rsplit("_", 1)[0]
        if not pos or pos[-1] != p:
            pos.append(p)
    return pos


def load_feature_names(regime: dict) -> tuple[list[str], list[str]]:
    fn = json.loads((regime["model_dir"] / "xgb_blosum50_flat_feature_names.json").read_text())
    positions = parse_positions(fn)
    expected = [f"{p}_{a}" for p in positions for a in AA_ORDER]
    assert fn == expected, "feature_names.json does not match expected order"
    return fn, positions


def generate_sequences(n: int, n_positions: int, train_csv: Path,
                       anchors: dict[int, list[str]] | None = None
                       ) -> list[str]:
    """Unique random context windows over 20 canonical AAs; training excluded.

    anchors: optional {position index -> allowed AAs} forced at those
    positions (run 2: HLA-A*02:01 anchors at P2/P9).
    """
    rng = np.random.default_rng(SEED)
    train = set()
    with open(train_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            train.add(row["full_context"])
    print(f"training windows to exclude: {len(train)}")

    canon = np.array([a for a in AA_ORDER if a != "X"])
    anchors = anchors or {}
    seqs: set[str] = set()
    while len(seqs) < n:
        need = n - len(seqs)
        idx = rng.integers(0, 20, size=(need + need // 10, n_positions),
                           dtype=np.int8)
        for pos, allowed in anchors.items():
            pool = np.array([AA_ORDER.index(a) for a in allowed], dtype=np.int8)
            idx[:, pos] = pool[rng.integers(0, len(pool), size=len(idx))]
        for row in idx:
            s = "".join(canon[row])
            if s not in train and len(seqs) < n:
                seqs.add(s)
    out = list(seqs)
    rng.shuffle(out)
    assert len(out) == n and len(set(out)) == n
    return out


def encode(seqs: list[str]) -> np.ndarray:
    """(n, n_positions*21) BLOSUM50 features, column order = feature_names."""
    aa_idx = {a: i for i, a in enumerate(AA_ORDER)}
    idx = np.array([[aa_idx[c] for c in s] for s in seqs], dtype=np.int8)
    feats = BLOSUM50[idx]                 # (n, positions, 21)
    return feats.reshape(len(seqs), -1)   # pos-major, matches feature order


def load_models(regime: dict) -> list:
    paths = sorted(glob.glob(str(regime["model_dir"] /
                                "xgb_blosum50_flat_outer*_inner*.pkl")))
    paths = [p for p in paths if not p.endswith("_artifacts.pkl")]
    if len(paths) != 30:
        raise SystemExit(f"expected 30 pickles, found {len(paths)}")
    models = []
    for p in paths:
        with open(p, "rb") as fh:
            models.append(pickle.load(fh))
    print(f"loaded {len(models)} models")
    return models


def score(models: list, seqs: list[str]) -> np.ndarray:
    probs = np.empty(len(seqs), dtype=np.float32)
    for i in range(0, len(seqs), CHUNK):
        chunk = seqs[i:i + CHUNK]
        X = encode(chunk)
        acc = np.zeros(len(chunk), dtype=np.float64)
        for m in models:
            acc += m.predict_proba(X)[:, 1]
        probs[i:i + CHUNK] = acc / len(models)
        print(f"scored {min(i + CHUNK, len(seqs))}/{len(seqs)}")
    return probs


def score_training(models: list, feature_names: list[str],
                   blosum_csv: Path) -> np.ndarray:
    df = pd.read_csv(blosum_csv, usecols=feature_names)
    X = df[feature_names].to_numpy(dtype=np.float32)
    acc = np.zeros(len(X), dtype=np.float64)
    for m in models:
        acc += m.predict_proba(X)[:, 1]
    return acc / len(models)


def ppm(seqs: list[str], positions: list[str]) -> pd.DataFrame:
    """Per-position probability matrix, 20 AAs x positions."""
    alphabet = [a for a in AA_ORDER if a != "X"]
    counts = np.zeros((20, len(positions)), dtype=np.int64)
    aa_idx = {a: i for i, a in enumerate(alphabet)}
    for s in seqs:
        for j, c in enumerate(s):
            i = aa_idx.get(c)
            if i is not None:      # X pads etc. dropped, same convention
                counts[i, j] += 1  # as 12_/13_ R scripts
    ppm_mat = counts / counts.sum(axis=0, keepdims=True)
    return pd.DataFrame(ppm_mat, index=alphabet, columns=positions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a02", action="store_true",
                    help="run 2: force HLA-A*02:01 anchors "
                         "P2 in {L,I,V,M,Q}, P9 in {V,L,I,A}")
    ap.add_argument("--flank5-rmdup", action="store_true",
                    help="use the r19_rmdup/canonical model (19-mer windows, "
                         "df_blosum50_flank5_rmdup) instead of the 29mer "
                         "baseline")
    args = ap.parse_args()

    regime_key = "flank5_rmdup" if args.flank5_rmdup else "29mer"
    regime = MODEL_REGIMES[regime_key]

    out_dir = OUT.parent / (
        regime["out_base"] + ("_a02" if args.a02 else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names, positions = load_feature_names(regime)
    print(f"regime={regime_key}  window={len(positions)}  "
          f"n_features={len(feature_names)}")

    anchors = None
    if args.a02:
        # run 2: force the A*02:01 anchors read off the presented-class logo
        anchors = {positions.index("P2"): ["L", "I", "V", "M", "Q"],
                   positions.index("P9"): ["V", "L", "I", "A"]}
        print(f"A*02:01 anchors: P2 in {anchors[positions.index('P2')]}, "
              f"P9 in {anchors[positions.index('P9')]}")

    models = load_models(regime)

    print("generating sequences ...")
    seqs = generate_sequences(N_SEQS, len(positions), regime["train_csv"],
                              anchors)

    print("scoring random sequences ...")
    probs = score(models, seqs)
    with gzip.open(out_dir / "scores.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "prob"])
        w.writerows(zip(seqs, probs))

    print("scoring training set (sanity) ...")
    train_probs = score_training(models, feature_names,
                                 PROJECT_ROOT / "data" / "processed" /
                                 ("df_blosum50.csv" if regime_key == "29mer"
                                  else "df_blosum50_flank5_rmdup.csv"))
    with gzip.open(out_dir / "train_scores.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prob"])
        w.writerows((f"{p:.6f}",) for p in train_probs)

    k = N_SEQS // 10
    order = np.argsort(probs)
    bottom_idx, top_idx = order[:k], order[-k:]
    top_seqs = [seqs[i] for i in top_idx]
    bot_seqs = [seqs[i] for i in bottom_idx]

    pd.DataFrame({"sequence": top_seqs, "prob": probs[top_idx]}) \
        .to_csv(out_dir / "top10.csv", index=False)
    pd.DataFrame({"sequence": bot_seqs, "prob": probs[bottom_idx]}) \
        .to_csv(out_dir / "bottom10.csv", index=False)

    ppm(top_seqs, positions).to_csv(out_dir / "ppm_top10.csv")
    ppm(bot_seqs, positions).to_csv(out_dir / "ppm_bottom10.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs, bins=100, density=True, alpha=0.6, label="random (1M)")
    ax.hist(train_probs, bins=100, density=True, alpha=0.6,
            label=f"training ({len(train_probs):,})")
    ax.axvline(probs[top_idx[0]], color="k", ls="--", lw=1,
               label=f"top10% cutoff = {probs[top_idx[0]]:.3f}")
    ax.axvline(probs[bottom_idx[-1]], color="k", ls=":", lw=1,
               label=f"bottom10% cutoff = {probs[bottom_idx[-1]]:.3f}")
    ax.set_xlabel("Predicted P(presented)")
    ax.set_ylabel("Density")
    title = "XGB BLOSUM50 ensemble — "
    title += ("A*02:01-anchored random" if args.a02 else "random")
    title += f" ({regime_key}) vs training score distributions"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_hist.png", dpi=200)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
