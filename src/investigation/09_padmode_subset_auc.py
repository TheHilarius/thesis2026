#!/usr/bin/env python3
"""
09_padmode_subset_auc.py
AUC on the padded-row subset (padtype comparison for the flat modelling grid).

The overall Val AUC cannot separate pad modes — real-slot signal is identical
across zero/boundary/eos_bos_repeat and near-identical for padtoken. Only the
padded rows carry mode signal (~1443 terminus peptides per toolkit), so this
script recomputes AUC on exactly those rows from the saved nested-CV
predictions, per toolkit x padtype x PCA dim, XGBOOST ONLY (the production
learner — lr/rf rows excluded; JSONs unchanged on disk). Per-toolkit subset =
rows with >=1 pad AND >=1 real slot (excludes notfound zero-windows, which
ESM-IF marks all-pad). Decision (which padtype to keep) is the user's; this
script only reports. pad_token for ESM-C is provisional — revertable if
phase-4 performance is bad.

Inputs (all local after rsync):
    models/flat/cv_results_*.json          — flat modelling runs (72 on disk;
        this script reads the 24 xgb ones). outer_val_predictions[fold] =
        {y_true, y_prob}, folds 0..5 rotate all buckets (incl. held-out) as
        validation, so every row is predicted exactly once.
    data/processed/df_all_with_folds.csv   — columns label, fold (SPLIT_DATA_PATH)
    data/processed/embeddings/esmc_context_embeddings_zeropad.h5  — pad_mask
    data/processed/embeddings/esm-if_test_zero.h5                 — pad_mask
        (pad_mask identical across modes within a toolkit; one file per
        toolkit suffices. Toolkits' masks differ on notfound rows.)

Output:
    results/padmode_subset_auc.csv
    stdout table per toolkit

Usage:
    python src/investigation/09_padmode_subset_auc.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH, EMBEDDING_DIR

MODELS_DIR = PROJECT_ROOT / "models" / "flat"
OUT_CSV = PROJECT_ROOT / "results" / "padmode_subset_auc.csv"

PADMASK_FILES = {
    "ESM-C": EMBEDDING_DIR / "esmc_context_embeddings_zeropad.h5",
    "ESM-IF": EMBEDDING_DIR / "esm-if_test_zero.h5",
}

JSON_RE = re.compile(
    r"cv_results_(lr_l2|rf|xgb)_handcrafted_sparse_"
    r"(esmc_win_(?:zeropad|impute_boundary|impute_bos_eos|padtoken)"
    r"|esmif_(?:win|pad|boundary|eos_bos_repeat))"
    r"_pca(\d+)_flat_"
)

# log/config name -> canonical padtype label
PADTYPE_MAP = {
    "esmc_win_zeropad": ("ESM-C", "zero"),
    "esmc_win_impute_boundary": ("ESM-C", "boundary"),
    "esmc_win_impute_bos_eos": ("ESM-C", "eos_bos_repeat"),
    "esmc_win_padtoken": ("ESM-C", "pad_token"),
    "esmif_win": ("ESM-IF", "zero"),
    "esmif_pad": ("ESM-IF", "pad"),
    "esmif_boundary": ("ESM-IF", "boundary"),
    "esmif_eos_bos_repeat": ("ESM-IF", "eos_bos_repeat"),
}


def load_padded_masks() -> dict[str, np.ndarray]:
    """Per-toolkit informative padded flag in df order (dict toolkit -> mask).

    Informative = >=1 pad slot AND >=1 real slot — rows where pad fill
    actually differs across modes. Excludes notfound zero-windows (ESM-IF
    marks those all-pad; ESM-C leaves them all-real). ESM-C row_indices are
    df-ordered; ESM-IF's are a processing-order permutation — scatter both
    into df order. The two toolkits' masks need NOT agree: they define
    independent per-toolkit subsets (the padtype pick is per toolkit).
    """
    masks = {}
    for toolkit, path in PADMASK_FILES.items():
        if not path.exists():
            raise SystemExit(f"Missing pad_mask source: {path}")
        with h5py.File(path, "r") as f:
            pm = f["pad_mask"][:].astype(bool)   # (rows, 29)
            ri = f["row_indices"][:]
        n = len(ri)
        if not np.array_equal(np.sort(ri), np.arange(n)):
            raise SystemExit(f"{path.name}: row_indices not a permutation "
                             f"of 0..{n-1} — cannot align with split-df rows")
        informative_by_h5row = pm.any(axis=1) & ~pm.all(axis=1)
        by_df = np.zeros(n, dtype=bool)
        by_df[ri] = informative_by_h5row          # scatter to df order
        masks[toolkit] = by_df
        print(f"  {toolkit}: informative padded rows = {int(by_df.sum())}")
    return masks


def parse_json_name(path: Path):
    m = JSON_RE.search(path.name)
    if not m:
        return None
    learner, key, pca = m.group(1), m.group(2), int(m.group(3))
    toolkit, padtype = PADTYPE_MAP[key]
    return toolkit, padtype, learner, pca


def subset_auc_for_json(path: Path, df: pd.DataFrame,
                        masks: dict[str, np.ndarray]) -> dict | None:
    parsed = parse_json_name(path)
    if parsed is None:
        print(f"  [skip] unrecognized filename: {path.name}")
        return None
    toolkit, padtype, learner, pca = parsed
    padded = masks[toolkit]

    with open(path) as fh:
        res = json.load(fh)
    preds = res["outer_val_predictions"]

    n = len(df)
    y_true = np.full(n, -1, dtype=int)
    y_prob = np.full(n, np.nan, dtype=float)

    for fold_key, entry in preds.items():
        fold = int(fold_key)
        bucket = df[df["fold"] == fold]
        pos = bucket.index.to_numpy()
        yt = np.asarray(entry["y_true"], dtype=int)
        yp = np.asarray(entry["y_prob"], dtype=float)
        if len(yt) != len(pos):
            raise SystemExit(
                f"{path.name} fold {fold}: len(y_true)={len(yt)} != "
                f"bucket size {len(pos)}")
        if not np.array_equal(yt, bucket["label"].to_numpy()):
            raise SystemExit(
                f"{path.name} fold {fold}: y_true does not match df labels")
        y_true[pos] = yt
        y_prob[pos] = yp

    if (y_true < 0).any() or np.isnan(y_prob).any():
        raise SystemExit(f"{path.name}: not every row received a prediction")

    auc_pad = float(roc_auc_score(y_true[padded], y_prob[padded]))
    auc_rest = float(roc_auc_score(y_true[~padded], y_prob[~padded]))
    auc_all = float(roc_auc_score(y_true, y_prob))

    # per-fold spread on padded rows (uncertainty at n≈1443)
    fold_aucs = []
    for fold in sorted(int(k) for k in preds):
        idx = df.index[df["fold"] == fold].to_numpy()
        sel = idx[padded[idx]]
        if len(sel) and y_true[sel].min() != y_true[sel].max():
            fold_aucs.append(roc_auc_score(y_true[sel], y_prob[sel]))
    fold_std = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else float("nan")

    return {
        "toolkit": toolkit,
        "padtype": padtype,
        "learner": learner,
        "pca": pca,
        "auc_padded": auc_pad,
        "n_padded": int(padded.sum()),
        "auc_nonpadded": auc_rest,
        "auc_overall": auc_all,
        "fold_padded_auc_std": fold_std,
    }


def main():
    if not MODELS_DIR.is_dir():
        raise SystemExit(
            f"Missing {MODELS_DIR} — rsync the JSONs first:\n"
            f"  rsync -avz s204581@hub1:/home/projects/"
            f"thesis_s204692_s204581/thesis2026/models/flat/*.json "
            f"{MODELS_DIR}/")
    json_paths = sorted(MODELS_DIR.glob("cv_results_xgb_*_flat_*.json"))
    print(f"  {len(json_paths)} xgb result JSONs in {MODELS_DIR}")

    df_path = SPLIT_DATA_PATH
    df = pd.read_csv(df_path, usecols=["label", "fold"])
    print(f"  {df_path.name}: {len(df)} rows, "
          f"pos={(df['label'] == 1).sum()}, neg={(df['label'] == 0).sum()}")

    masks = load_padded_masks()
    for toolkit, m in masks.items():
        if len(m) != len(df):
            raise SystemExit(
                f"{toolkit} pad_mask rows {len(m)} != df rows {len(df)}")

    rows = []
    for p in json_paths:
        r = subset_auc_for_json(p, df, masks)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit("No JSONs parsed — check models/flat/ contents")

    out = pd.DataFrame(rows).sort_values(
        ["toolkit", "learner", "pca", "padtype"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    col_order = {
        "ESM-C": ["zero", "boundary", "eos_bos_repeat", "pad_token"],
        "ESM-IF": ["zero", "boundary", "eos_bos_repeat", "pad"],
    }
    for toolkit in ["ESM-C", "ESM-IF"]:
        sub = out[out["toolkit"] == toolkit]
        n_pad_rows = int(masks[toolkit].sum())
        print(f"\n{'=' * 72}")
        print(f"  {toolkit} — XGB AUC on the {n_pad_rows} informative "
              f"padded rows")
        print(f"{'=' * 72}")
        piv = sub.pivot_table(index="pca", columns="padtype",
                              values="auc_padded")
        piv = piv.reindex(columns=col_order[toolkit])
        print(piv.round(4).to_string())

    print(f"\n  Saved: {OUT_CSV}")
    print(f"  NOTE: non-padded control AUC in the CSV must be ~identical "
          f"across padtypes (mode-invariant reals).")


if __name__ == "__main__":
    main()
