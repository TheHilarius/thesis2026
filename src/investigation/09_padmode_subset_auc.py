#!/usr/bin/env python3
"""
09_padmode_subset_auc.py
Padded-subset AUC comparison across embedding pad-fill modes.

Reuses saved nested-CV predictions (cv_results_*.json) from 04_modelling.py —
no retraining. For each run the full out-of-fold probability vector is
reconstructed from outer_val_predictions[fold 0..5]; rows align to
df_all_with_folds.csv row order via the fold column (verified against the
prepared-embedding HDF5). AUC is then computed on the padded subset only.

Because the ~45k unpadded rows are bit-identical across the four pad-fill
modes, any difference in padded-subset AUC is attributable to the ~1443
padded rows' mode-specific representations.

Usage:
    python 09_padmode_subset_auc.py

    python 09_padmode_subset_auc.py \
        --results 'runs/xgb_*_flat_*/models/flat/cv_results_*.json' \
        --since '2026-10-07 10:00:00' \
        --out '/net/kvasir/pool/projects1/thesis_s204692_s204581/thesis2026/results/tables/padmode_base_auc.tsv'

Outputs:
    --out                       Main per-run AUC summary TSV
    <out stem>_pivot.tsv        Padmode x PCA pivot table
    <out stem>_pairwise.tsv     Pairwise |Δprob| summary on padded rows
    <out stem>_rows.tsv         Per-row probabilities (--save-rows)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from config import SPLIT_DATA_PATH, PREPARED_EMBEDDING_DIR


DEFAULT_PAD_H5 = PREPARED_EMBEDDING_DIR / "esmc_win_zeropad_prepared.h5"


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_pad_metadata(h5_path):
    """Load pad_mask (N,29) bool + folds/labels from a prepared window HDF5."""
    with h5py.File(h5_path, "r") as f:
        pad_mask = f["pad_mask"][:].astype(bool)
        folds = f["folds"][:].astype(int)
        labels = f["labels"][:].astype(int)
        pad_counts = f["pad_counts"][:].astype(int)
    return pad_mask, folds, labels, pad_counts


def load_split_df():
    df = pd.read_csv(SPLIT_DATA_PATH)
    return df


def parse_timestamp(ts):
    """Parse timestamps used in cv_results JSON files."""
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except (TypeError, ValueError):
            pass
    return None


def mode_from_result(r):
    """Extract the pad-fill mode tag from a cv_results JSON."""
    emb = (
        r.get("config", {})
        .get("component_info", {})
        .get("emb_components", [])
    )

    if emb:
        emb_key = emb[0].get("embedding_key", "")

        if emb_key.startswith("esmc_win_"):
            return emb_key[len("esmc_win_"):]

        if emb_key.startswith("esmif_"):
            return emb_key[len("esmif_"):]

        return emb_key

    feat = r.get("features_key", "")

    for prefix in (
        "handcrafted_sparse_esmc_win_",
        "handcrafted_sparse_esmif_",
        "handcrafted_sparse_",
    ):
        if feat.startswith(prefix):
            return feat[len(prefix):]

    return feat


# ──────────────────────────────────────────────
# Prediction reconstruction
# ──────────────────────────────────────────────

def reconstruct_probs(r, folds):
    """
    Rebuild the full-length out-of-fold probability vector.

    outer_val_predictions[f] = {y_true, y_prob} holds rows of the validation
    bucket f in df row order (prepare_validation uses np.where(df.fold==f)[0]).

    So we scatter each bucket's probabilities back onto its fold mask.
    """
    n = len(folds)
    probs = np.full(n, np.nan)
    y_recon = np.full(n, -1)
    ovp = r["outer_val_predictions"]

    fold_ids = sorted(ovp.keys(), key=int)

    for fk in fold_ids:
        f = int(fk)
        rec = ovp[fk]
        mask = folds == f
        n_here = int(mask.sum())

        if len(rec["y_true"]) != n_here:
            raise ValueError(
                f"fold {f}: JSON has {len(rec['y_true'])} rows, "
                f"fold mask has {n_here} — misaligned"
            )

        probs[mask] = np.asarray(rec["y_prob"], dtype=np.float64)
        y_recon[mask] = np.asarray(rec["y_true"])

    if np.isnan(probs).any():
        missing = np.isnan(probs).sum()
        raise ValueError(
            f"{missing} rows never covered by any validation bucket"
        )

    return probs, y_recon.astype(int)


def auc_safe(y, p):
    """AUC-ROC with a single-class guard."""
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results",
        nargs="+",
        default=["runs/xgb_*_flat_*/models/flat/cv_results_*.json"],
        help="Result JSON paths or globs",
    )

    parser.add_argument(
        "--since",
        default=None,
        help="Only include JSONs >= this timestamp (YYYY-MM-DD HH:MM:SS)",
    )

    parser.add_argument(
        "--out",
        default="results/tables/pad_subset_auc.tsv",
        help="Output TSV file",
    )

    parser.add_argument(
        "--pad-h5",
        default=str(DEFAULT_PAD_H5),
        help="Prepared window HDF5 supplying pad_mask/folds/labels",
    )

    parser.add_argument(
        "--save-rows",
        action="store_true",
        help="Also write the per-row probability table",
    )

    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.strptime(
            args.since,
            "%Y-%m-%d %H:%M:%S",
        )

    # ── Expand result globs ──
    paths = []

    for pat in args.results:
        matches = sorted(Path(".").glob(pat))
        paths.extend(matches)

    paths = sorted(set(paths))

    if not paths:
        print("No JSONs matched the supplied --results patterns")
        sys.exit(1)

    # ── Filter by timestamp stored inside each JSON ──
    if since:
        filtered_paths = []

        for path in paths:
            with open(path) as f:
                r = json.load(f)

            ts_raw = r.get("timestamp", "")
            ts_dt = parse_timestamp(ts_raw)

            if ts_dt is not None and ts_dt >= since:
                filtered_paths.append(path)

        paths = filtered_paths

    if not paths:
        print("No JSONs remained after applying --since")
        sys.exit(1)

    print(f"Found {len(paths)} result JSON(s)")

    # ── Load shared metadata ──
    pad_mask, folds, labels, pad_counts = load_pad_metadata(args.pad_h5)
    df = load_split_df()
    n = len(folds)

    if len(df) != n:
        raise ValueError(
            f"split CSV has {len(df)} rows, pad h5 has {n}"
        )

    pad_row = pad_mask.any(axis=1)
    real_row = ~pad_row

    print(f"  rows total        : {n}")
    print(
        f"  rows with any pad : {pad_row.sum()} "
        f"({pad_row.mean() * 100:.2f}%)"
    )
    print(f"  pad slots total   : {pad_mask.sum()}")

    n_only = int(
        ((pad_counts[:, 0] > 0) & (pad_counts[:, 1] == 0)).sum()
    )
    c_only = int(
        ((pad_counts[:, 0] == 0) & (pad_counts[:, 1] > 0)).sum()
    )
    both = int(
        ((pad_counts[:, 0] > 0) & (pad_counts[:, 1] > 0)).sum()
    )

    print(
        f"  N-only / C-only / both: "
        f"{n_only} / {c_only} / {both}"
    )

    # ── Load each run, reconstruct, evaluate ──
    summary = []
    prob_rows = {}

    for p in paths:
        with open(p) as f:
            r = json.load(f)

        mode = mode_from_result(r)

        probs, y_recon = reconstruct_probs(r, folds)

        if not (y_recon == labels).all():
            raise ValueError(
                f"{p.name}: reconstructed labels mismatch HDF5"
            )

        prob_rows[mode] = probs

        n_pad_pos = int((labels[pad_row] == 1).sum())
        n_pad_neg = int((labels[pad_row] == 0).sum())

        summary.append({
            "mode": mode,
            "result_file": p.name,
            "timestamp": r.get("timestamp", ""),
            "model_key": r.get("model_key", ""),
            "features_key": r.get("features_key", ""),
            "pca_mode": r.get("config", {}).get("pca_mode", ""),
            "n_features": r.get("config", {}).get("n_features", ""),
            "n_pad_rows": int(pad_row.sum()),
            "n_pad_pos": n_pad_pos,
            "n_pad_neg": n_pad_neg,
            "pad_auc_roc": auc_safe(
                labels[pad_row],
                probs[pad_row],
            ),
            "pad_auc_pr": average_precision_score(
                labels[pad_row],
                probs[pad_row],
            ),
            "real_auc_roc": auc_safe(
                labels[real_row],
                probs[real_row],
            ),
            "global_auc_roc": auc_safe(
                labels,
                probs,
            ),
            "pad_prob_mean_pos": float(
                probs[pad_row & (labels == 1)].mean()
            ),
            "pad_prob_mean_neg": float(
                probs[pad_row & (labels == 0)].mean()
            ),
        })

        print(
            f"  [{mode:<16}] pad AUC-ROC = "
            f"{summary[-1]['pad_auc_roc']:.4f} "
            f"(pad rows {n_pad_pos} pos / {n_pad_neg} neg)"
        )

    if not prob_rows:
        print("No modes loaded")
        sys.exit(1)

    # ── Output paths ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Main summary TSV ──
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(
        out_path,
        sep="\t",
        index=False,
    )

    print(f"\nSaved summary: {out_path}")

    # ── Padmode x PC decision table ──
    df_sum["pca"] = pd.to_numeric(
        df_sum["result_file"]
        .str.extract(r"_pca(\d+)_")[0],
        errors="coerce",
    )

    piv = df_sum.pivot_table(
        index="pca",
        columns="mode",
        values="pad_auc_roc",
    )

    print("\nPadmode x PC subset AUC (pad_auc_roc):")
    print(piv.round(4).to_string())

    piv_path = out_path.with_name(
        f"{out_path.stem}_pivot.tsv"
    )

    piv.to_csv(
        piv_path,
        sep="\t",
    )

    print(f"Saved pivot table: {piv_path}")

    # ── Pairwise |Δprob| on padded rows ──
    modes = sorted(prob_rows.keys())
    rows = []

    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]

            d = np.abs(
                prob_rows[a][pad_row] -
                prob_rows[b][pad_row]
            )

            rows.append({
                "mode_a": a,
                "mode_b": b,
                "mean_abs_delta_pad": float(d.mean()),
                "std_abs_delta_pad": float(d.std()),
                "pct_delta_gt_0.05": float(
                    (d > 0.05).mean()
                ),
                "pct_delta_gt_0.10": float(
                    (d > 0.10).mean()
                ),
            })

    df_pair = pd.DataFrame(rows)

    pair_path = out_path.with_name(
        f"{out_path.stem}_pairwise.tsv"
    )

    df_pair.to_csv(
        pair_path,
        sep="\t",
        index=False,
    )

    print(f"Saved pairwise deltas: {pair_path}")
    print(df_pair.to_string(index=False))

    # ── Optional per-row table ──
    if args.save_rows:
        row_tbl = df[
            [
                "peptide",
                "uniprot_id",
                "start",
                "end",
                "protein_length",
                "fold",
            ]
        ].copy()

        row_tbl["label"] = labels
        row_tbl["pad"] = pad_row
        row_tbl["n_pad"] = pad_counts[:, 0]
        row_tbl["c_pad"] = pad_counts[:, 1]

        for m in modes:
            row_tbl[f"prob_{m}"] = prob_rows[m]

        rows_path = out_path.with_name(
            f"{out_path.stem}_rows.tsv"
        )

        row_tbl.to_csv(
            rows_path,
            sep="\t",
            index=False,
        )

        print(f"Saved per-row table: {rows_path}")


if __name__ == "__main__":
    main()
