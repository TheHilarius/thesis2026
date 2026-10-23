"""
04_modelling.py
Cross-validation pipeline for MHC-I processing prediction.
Model-agnostic with composable feature sets.

Usage:
    python 04_modelling.py --model rf  --features handcrafted
    python 04_modelling.py --model lr_l2 --features handcrafted_sparse
    python 04_modelling.py --model lr_elasticnet --features handcrafted_sparse
    python 04_modelling.py --model rf  --features handcrafted_sparse_esmc
    python 04_modelling.py --model rf  --features all_sparse
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import importlib
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)
from datetime import datetime
import time
import json
import pickle
from pathlib import Path

from config import (
    SPLIT_DATA_PATH, LOG_DIR, MODEL_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    PEPTIDE_COL, LABEL_COL, FOLD_COL,
    POSITION_AA_COLS, EMBEDDING_REGIONS,
    RANDOM_STATE, DEFAULT_MODEL, DEFAULT_FEATURE_SET,
    get_feature_cols, get_model_config,
    get_feature_set_config, get_feature_component,
    get_embedding_source, validate_config, validate_feature_set,
)


# ──────────────────────────────────────────────
# 0. LOGGER
# ──────────────────────────────────────────────

class Logger:
    """Tee: writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


# ──────────────────────────────────────────────
# 1. MODEL FACTORY
# ──────────────────────────────────────────────

def build_model(model_cfg, param_overrides=None):
    """Instantiate a sklearn model from a MODEL_REGISTRY entry."""
    class_path = model_cfg["model_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    params = dict(model_cfg["params"])
    if param_overrides:
        params.update(param_overrides)
    # early_stopping_rounds is a valid XGBClassifier constructor param
    # (xgboost >= 1.6; server runs 3.4.1) — passed through untouched.
    return cls(**params)


# ──────────────────────────────────────────────
# 2. METRICS
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return metrics


def print_metrics(metrics, prefix=""):
    print(f"{prefix}  AUC-ROC: {metrics['auc_roc']:.4f}  |  AUC-PR: {metrics['auc_pr']:.4f}  |  "
          f"Acc: {metrics['accuracy']:.4f}  |  F1: {metrics['f1']:.4f}  |  "
          f"MCC: {metrics['mcc']:.4f}")
    print(f"{prefix}  Sens: {metrics['sensitivity']:.4f}  |  Spec: {metrics['specificity']:.4f}  |  "
          f"PPV: {metrics['ppv']:.4f}  |  NPV: {metrics['npv']:.4f}")


# ──────────────────────────────────────────────
# 3. DATA LOADING
# ──────────────────────────────────────────────

def load_embedding_data(embedding_key):
    """
    Load a prepared embedding HDF5 into memory.

    For kind=="windows" (B pipeline) this returns a LAZY descriptor: small
    arrays (labels, folds, pad_counts) + metadata are loaded, but the (N,29,D)
    tensor is NOT materialized — it is streamed at PCA fit/transform time.

    Returns dict with keys:
        regions  -> dict of {region_name: np.array (N, D)}   (legacy only)
        kind     -> "legacy" or "windows"
        labels   -> np.array (N,)
        folds    -> np.array (N,)
        emb_dim  -> int
        window   -> int (windows only)
        pad_counts -> np.array (N, 2)  (windows only)
        n_samples -> int
    """
    emb_source = get_embedding_source(embedding_key)
    prepared_path = emb_source["prepared_path"]

    if not prepared_path.exists():
        raise FileNotFoundError(
            f"Prepared embedding not found: {prepared_path}\n"
            f"Run: python 03_prepare_embeddings.py --embedding {embedding_key}"
        )

    print(f"    Loading prepared embeddings: {prepared_path}")

    if emb_source.get("kind") == "windows":
        with h5py.File(prepared_path, "r") as f:
            data = {
                "kind": "windows",
                "regions": {},
                "labels": f["labels"][:],
                "folds": f["folds"][:],
                "pad_counts": f["pad_counts"][:].astype(np.float32),
                "emb_dim": int(f.attrs.get("emb_dim", 0)),
                "window": int(f.attrs.get("window", 29)),
                "n_samples": int(f.attrs.get("n_samples", len(f["labels"]))),
                "prepared_path": str(prepared_path),
                "embedding_key": embedding_key,
            }
        print(f"    Loaded (windows): {data['n_samples']} samples, "
              f"window={data['window']}, dim={data['emb_dim']} "
              f"(tensor streamed lazily)")
        return data

    data = {"regions": {}, "kind": "legacy"}
    with h5py.File(prepared_path, "r") as f:
        data["labels"] = f["labels"][:]
        data["folds"] = f["folds"][:]
        data["emb_dim"] = int(f.attrs.get("emb_dim", 0))
        data["n_samples"] = int(f.attrs.get("n_samples", len(data["labels"])))

        for region in EMBEDDING_REGIONS:
            if region in f:
                data["regions"][region] = f[region][:]

    first_region = list(data["regions"].values())[0]
    emb_dim = first_region.shape[1]
    data["emb_dim"] = emb_dim

    print(f"    Loaded: {data['n_samples']} samples, "
          f"dim={emb_dim}")

    return data


def load_alternate_csv(csv_path, df_split):
    """
    Load an alternate CSV (sparse or BLOSUM) and extract only the
    new columns that are not already in the split dataframe.
    Validates row alignment before returning.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Alternate CSV not found: {csv_path}")

    print(f"    Loading: {csv_path.name}")
    df_alt = pd.read_csv(csv_path)
    print(f"    Shape: {df_alt.shape[0]} rows x {df_alt.shape[1]} columns")

    if len(df_alt) != len(df_split):
        raise ValueError(
            f"Row count mismatch — split data has {len(df_split)} rows, "
            f"alternate CSV has {len(df_alt)} rows"
        )

    # Spot-check alignment
    n_pep_mismatch = (df_split[PEPTIDE_COL] != df_alt[PEPTIDE_COL]).sum()
    n_label_mismatch = (df_split[LABEL_COL] != df_alt[LABEL_COL]).sum()
    if n_pep_mismatch > 0 or n_label_mismatch > 0:
        raise ValueError(
            f"Alignment check failed — {n_pep_mismatch} peptide mismatches, "
            f"{n_label_mismatch} label mismatches."
        )
    print(f"    [OK] Alignment verified")

    # Extract only new columns
    new_cols = [c for c in df_alt.columns if c not in df_split.columns]
    print(f"    New columns: {len(new_cols)}")

    return df_alt[new_cols]


# ──────────────────────────────────────────────
# 4. COMPONENT-AWARE DATA ASSEMBLY
# ──────────────────────────────────────────────

def resolve_components(feat_cfg):
    """
    Parse a feature set config into its component parts.
    Returns:
        csv_components   -> list of component configs with type=="csv"
        emb_components   -> list of component configs with type=="embedding"
    """
    csv_components = []
    emb_components = []

    for comp_key in feat_cfg["components"]:
        comp = get_feature_component(comp_key)
        comp = dict(comp)       # copy so we can annotate
        comp["_key"] = comp_key

        if comp["type"] == "csv":
            csv_components.append(comp)
        elif comp["type"] in ("embedding", "embedding_windows"):
            emb_components.append(comp)
        else:
            raise ValueError(f"Unknown component type: {comp['type']}")

    return csv_components, emb_components


def load_all_components(df_split, feat_cfg, pca_mode="flat",
                          pool=None, pool_mask="masked"):
    """
    Load everything required by the feature set.

    Returns:
        df           -> the split dataframe, possibly enriched with new CSV columns
        csv_feature_cols -> list of column names to use from the dataframe
        emb_data_dict    -> dict of {comp_key: emb_data} for each embedding component
        component_info   -> summary dict for logging
    """
    csv_components, emb_components = resolve_components(feat_cfg)

    df = df_split
    component_info = {
        "csv_components": [],
        "emb_components": [],
    }

    # ── Load CSV components ──
    new_col_frames = []      # collect DataFrames to concat at the end
    comp_cols_by_key = {}    # comp_key -> column names loaded from its CSV

    for comp in csv_components:
        comp_key = comp["_key"]
        csv_path = comp.get("csv_path")

        if csv_path is not None:
            print(f"\n  Loading CSV component: {comp['display_name']}")
            new_cols_df = load_alternate_csv(csv_path, df)
            new_col_frames.append(new_cols_df)
            comp_cols_by_key[comp_key] = set(new_cols_df.columns)

            component_info["csv_components"].append({
                "key": comp_key,
                "display_name": comp["display_name"],
                "source": str(csv_path),
                "n_new_cols": len(new_cols_df.columns),
            })
        else:
            component_info["csv_components"].append({
                "key": comp_key,
                "display_name": comp["display_name"],
                "source": "split_data (built-in)",
            })

    # Merge all new columns at once (avoids DataFrame fragmentation)
    if new_col_frames:
        df = pd.concat([df] + new_col_frames, axis=1)
        total_new = sum(len(f.columns) for f in new_col_frames)
        print(f"\n  Merged {total_new} new columns → "
              f"{df.shape[0]} rows x {df.shape[1]} columns")
        del new_col_frames

    # Resolve CSV feature columns (component-aware).
    # Structural (handcrafted) columns always live in the split data, so
    # they must only be used when the 'handcrafted' component is selected.
    # AA-encoded columns ({pos}_{AA}/KF*/AF*, 29 positions) come from the
    # alternate CSVs (df_all_sparse.csv / df_all_blosum50.csv / ...) and are
    # only used when an AA-encoding component is selected.
    AA_ENCODING_COMPONENTS = {
        "sparse", "blosum50", "blosum62", "blosum80", "kidera", "atchley",
        "compositional"
    }
    csv_feature_cols = []
    if csv_components:
        selected = {comp["_key"] for comp in csv_components}
        include_structural = "handcrafted" in selected
        include_encoded = bool(selected & AA_ENCODING_COMPONENTS)

        # frac_* columns do not carry a {pos}_ prefix, so they must be
        # recognised explicitly or they'd be misfiled as structural
        compositional_cols = set()
        for _k, _cols in comp_cols_by_key.items():
            if _k == "compositional":
                compositional_cols |= _cols

        all_feature_cols = get_feature_cols(df.columns)
        numeric_cols = [
            c for c in all_feature_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        non_numeric_dropped = len(all_feature_cols) - len(numeric_cols)
        if non_numeric_dropped > 0:
            print(f"  Excluded {non_numeric_dropped} non-numeric feature column(s)")

        structural_cols = [
            c for c in numeric_cols
            if not any(c.startswith(f"{pos}_") for pos in POSITION_AA_COLS)
            and c not in compositional_cols
        ]
        encoded_cols = [c for c in numeric_cols if c not in structural_cols]

        if include_structural:
            csv_feature_cols.extend(structural_cols)
        else:
            print(f"  Excluded {len(structural_cols)} structural column(s) "
                  f"('handcrafted' component not selected)")
        if include_encoded:
            csv_feature_cols.extend(encoded_cols)
        else:
            print(f"  Excluded {len(encoded_cols)} AA-encoded column(s) "
                  f"(no AA-encoding component selected)")

    # ── Load embedding components ──
    emb_data_dict = {}
    for comp in emb_components:
        comp_key = comp["_key"]
        emb_key = comp["embedding_key"]
        pca_components = comp.get("pca_components", 50)

        print(f"\n  Loading embedding component: {comp['display_name']}")
        emb_data = load_embedding_data(emb_key)

        # Verify sample count matches
        if emb_data["n_samples"] != len(df):
            raise ValueError(
                f"Embedding '{emb_key}' has {emb_data['n_samples']} samples "
                f"but split data has {len(df)} rows. "
                f"Rerun 03_prepare_embeddings.py --embedding {emb_key}"
            )

        emb_data["pca_components"] = pca_components
        emb_data["pca_mode"] = pca_mode
        emb_data["pool"] = pool
        emb_data["pool_mask"] = pool_mask
        emb_data_dict[comp_key] = emb_data

        if pool is not None and emb_data.get("kind") != "windows":
            raise ValueError(
                f"--pool '{pool}' requires window embeddings, but "
                f"component '{comp_key}' is legacy (kind="
                f"{emb_data.get('kind')}). Pooling is windows-only."
            )

        emb_dim = emb_data["emb_dim"]
        if emb_data.get("kind") == "windows":
            window = emb_data["window"]
            if pool is not None:
                mult = 2 if pool in ("mean_max", "mean_median", "mean_std") else 1
                pca_total = mult * emb_dim       # no PCA — pooled dims
                raw_dim_desc = (f"{window}x{emb_dim} "
                                f"(pool-{pool} {pool_mask} {pca_total})")
            elif pca_mode == "flat_raw":
                pca_total = window * emb_dim          # no PCA — all raw dims
                raw_dim_desc = f"{window}x{emb_dim} (flat_raw {window * emb_dim})"
            elif pca_mode == "simple21":
                pca_total = window * SIMPLE21_DIMS  # placeholder; --dims fixes it
                raw_dim_desc = f"{window}x{SIMPLE21_DIMS} (simple21 dim21)"
            elif pca_mode == "dimselect":
                # placeholder; --top-n fixes it in main
                pca_total = window * DIMSELECT_DEFAULT_TOP_N
                raw_dim_desc = (f"{window}x{emb_dim} (dimselect top"
                                f"{DIMSELECT_DEFAULT_TOP_N})")
            elif pca_mode == "varfold":
                # placeholder; --top-n fixes it in main
                pca_total = window * VARFOLD_DEFAULT_TOP_N
                raw_dim_desc = (f"{window}x{emb_dim} (varfold top"
                                f"{VARFOLD_DEFAULT_TOP_N}, fold-nested)")
            elif pca_mode == "autoencoder":
                # placeholder; --ae-latent fixes it in main
                pca_total = window * AE_DEFAULT_LATENT
                raw_dim_desc = (f"{window}x{emb_dim} (1x1-conv AE latent"
                                f"{AE_DEFAULT_LATENT}, fold-nested)")
            else:  # flat
                pca_total = int(pca_components)
                raw_dim_desc = f"{window}x{emb_dim} (flat {window * emb_dim})"
        else:
            window = None
            pca_total = pca_components
            raw_dim_desc = str(emb_dim)
        component_info["emb_components"].append({
            "key": comp_key,
            "display_name": comp["display_name"],
            "embedding_key": emb_key,
            "emb_dim": emb_dim,
            "raw_dim": emb_dim,
            "raw_dim_desc": raw_dim_desc,
            "window": window,
            "pca_components": pca_components,
            "pca_total": pca_total,
            "pca_mode": pca_mode,
            "pool": pool,
            "pool_mask": pool_mask,
        })

    return df, csv_feature_cols, emb_data_dict, component_info


# ──────────────────────────────────────────────
# 5. DATA PREPARATION — CSV PART
# ──────────────────────────────────────────────

def impute_nan(X_train, X_test, feature_cols, fold_id):
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    col_medians = np.nanmedian(X_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    # Per-column imputation report
    train_nan_counts = np.isnan(X_train).sum(axis=0)
    test_nan_counts  = np.isnan(X_test).sum(axis=0)

    cols_with_nan = np.where((train_nan_counts > 0) | (test_nan_counts > 0))[0]

    if len(cols_with_nan) > 0:
        print(f"\n    NaN imputation report — "
              f"{len(cols_with_nan)} column(s) affected:")
        print(f"    {'Column':<45} {'Train NaN':>10} {'Test NaN':>10} {'Median':>10}")
        print(f"    {'-' * 77}")
        for col_idx in cols_with_nan:
            col_name = feature_cols[col_idx] if col_idx < len(feature_cols) else f"col_{col_idx}"
            print(f"    {col_name:<45} {train_nan_counts[col_idx]:>10} "
                  f"{test_nan_counts[col_idx]:>10} {col_medians[col_idx]:>10.4f}")
        train_total = train_nan_counts.sum()
        test_total = test_nan_counts.sum()
        train_pct = train_total / X_train.size * 100 if X_train.size > 0 else 0.0
        test_pct = test_total / X_test.size * 100 if X_test.size > 0 else 0.0
        print(f"    Total NaN — train: {train_total} ({train_pct:.2f}%), "
              f"test: {test_total} ({test_pct:.2f}%)")
    else:
        print(f"    no NaN values found in CSV features")

    for col_idx in range(X_train.shape[1]):
        train_nan = np.isnan(X_train[:, col_idx])
        test_nan  = np.isnan(X_test[:, col_idx])
        X_train[train_nan, col_idx] = col_medians[col_idx]
        X_test[test_nan, col_idx]   = col_medians[col_idx]

    return X_train, X_test, col_medians


def impute_nan_single(X, col_medians):
    X = np.where(np.isinf(X), np.nan, X)
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        X[nan_mask, col_idx] = col_medians[col_idx]
    return X


# ──────────────────────────────────────────────
# 6. DATA PREPARATION — EMBEDDING PART
# ──────────────────────────────────────────────

def get_context_matrix(emb_data, indices):
    """
    Return the single context embedding matrix (n_flank + peptide + c_flank,
    mean-pooled at generation time) for the given sample indices.
    """
    region_name = EMBEDDING_REGIONS[0]
    return emb_data["regions"][region_name][indices].astype(np.float64)


# ──────────────────────────────────────────────
# 6b. WINDOW EMBEDDING STREAMING (kind == "windows")
# ──────────────────────────────────────────────
#
# The (N, 29, D) window tensor is never fully materialized in RAM.  Reads are
# chunked; h5py requires increasing index order, which is guaranteed here
# because every indices array passed in is an increasing subset of 0..N-1.

WINDOW_CHUNK_ROWS = 2048

# 29x21 test: keep all 29 slots, gather 21 embedding dims (AA-sized).
SIMPLE21_DIMS = 21

# dimselect placeholder: real N comes from --top-n (patched in main).
DIMSELECT_DEFAULT_TOP_N = 50

# varfold placeholder: real N comes from --top-n (patched in main).
VARFOLD_DEFAULT_TOP_N = 21

# autoencoder placeholder: real latent comes from --ae-latent (patched in main).
AE_DEFAULT_LATENT = 64


def _stream_window_slots(prepared_path, indices):
    """
    Generator over window rows for `indices`.  Yields
    (slots (c*window, D) float32, mask_flat (c*window,) bool) per chunk.
    mask convention: True = padded slot.
    """
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        M = f["pad_mask"]
        for i0 in range(0, len(indices), WINDOW_CHUNK_ROWS):
            idx = indices[i0:i0 + WINDOW_CHUNK_ROWS]
            block = W[idx]                      # (c, window, D)
            mblock = M[idx]                     # (c, window) bool
            c = block.shape[0]
            flat = block.reshape(c * block.shape[1], block.shape[-1])
            flatm = mblock.reshape(-1)
            yield flat, flatm


def concat_window_slots(emb_data, indices, dims=None):
    """
    Full flattened (n, window*D) float32 block for `indices` — flat_raw mode.
    Reuses the slot streamer; ALL slots kept (pads carry their raw fill),
    which is exactly what flat PCA would feed minus the rotation.
    dims gathers explicit embedding dims from every slot (29x21 mode,
    original order, no PCA mixing).
    """
    prepared_path = emb_data["prepared_path"]
    W = emb_data["window"]
    D = emb_data["emb_dim"]
    if dims is not None:
        dims = [int(d) for d in dims]
        ndim = len(dims)
    else:
        ndim = D
    n = len(indices)
    out = np.empty((n, W * ndim), dtype=np.float32)
    cur = 0
    for flat, _flatm in _stream_window_slots(prepared_path, indices):
        w = flat.shape[0] // W
        full = flat.reshape(w, W, D)
        if dims is not None:
            full = full[:, :, dims]
        out[cur:cur + w] = full.reshape(w, W * ndim)
        cur += w
    return out


def variance_rank_dims(prepared_path, indices):
    """
    Per-dim variance ranking over `indices` ONLY (fold-nested, leakage-free).

    Same metric as 10_dim_variance.py rank_dims: per sample, variance of each
    dim across the 29 positional slots (pad slots excluded via pad_mask,
    samples with <=1 real slot skipped), averaged over samples. Streams the
    H5 in chunks; `indices` must be an increasing subset of 0..N-1.

    Returns mean_col_var (D,) float64.
    """
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        M = f["pad_mask"]
        d = W.shape[2]
        col_var_sum = np.zeros(d, dtype=np.float64)
        col_var_cnt = 0
        for i0 in range(0, len(indices), WINDOW_CHUNK_ROWS):
            idx = indices[i0:i0 + WINDOW_CHUNK_ROWS]
            block = W[idx].astype(np.float64)     # (c, window, D)
            mblock = M[idx].astype(bool)          # (c, window) True=pad
            real = (~mblock).astype(np.float64)
            n_real = real.sum(axis=1, keepdims=True)   # (c, 1)
            ok = (n_real[:, 0] > 1)
            s = (block * real[:, :, None]).sum(axis=1)          # (c, D)
            ss = ((block ** 2) * real[:, :, None]).sum(axis=1)  # (c, D)
            mean = s / np.maximum(n_real, 1)
            var = ss / np.maximum(n_real, 1) - mean ** 2
            col_var_sum += np.where(ok[:, None], var, 0.0).sum(axis=0)
            col_var_cnt += int(ok.sum())
    return col_var_sum / max(col_var_cnt, 1)


def fit_window_pca_flat(emb_data, indices, k):
    """
    Fit PCA (randomized SVD) over flattened (window*D,) window vectors, using
    ONLY fully-real rows (no pad slot anywhere AND no zero-norm slot).

    Why: in the flattened scheme every row is a fixed (window*D,) vector, so a
    pad slot is part of the input rather than an appended extra.  The pad-fill
    values differ by mode (zero / pad / boundary / eos_bos_repeat), so fitting
    on rows that contain pads would leak mode-specific signal into the PCA
    basis and make the four-mode comparison confounded.  Rows with no pads are
    identical across the zero/boundary/eos_bos_repeat modes by construction,
    so fitting on them gives a single mode-invariant basis; the fill only
    enters at transform time.

    Uses full-memory randomized PCA rather than IncrementalPCA: IncrementalPCA
    is O(n_batch^2 * n_features) per batch (QR-based) and is intractable when
    n_features (window*D) >> n_batch, whereas randomized PCA is
    O(n_samples * n_features * k).
    """
    prepared_path = emb_data["prepared_path"]
    window = emb_data["window"]
    D = emb_data["emb_dim"]
    n_feat = window * D
    k = int(min(k, n_feat))

    rows = []
    n_rows = 0
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        M = f["pad_mask"]
        for i0 in range(0, len(indices), WINDOW_CHUNK_ROWS):
            idx = indices[i0:i0 + WINDOW_CHUNK_ROWS]
            block = W[idx].astype(np.float64)     # (c, window, D)
            mblock = M[idx].astype(bool)          # (c, window)
            n_rows += block.shape[0]
            flat = block.reshape(block.shape[0], n_feat)
            fully_real = ~mblock.any(axis=1)      # no pad slot anywhere
            norms = np.linalg.norm(flat, axis=1)
            keep = fully_real & (norms > 0.0)
            if keep.any():
                rows.append(flat[keep])

    X = np.concatenate(rows, axis=0)
    del rows
    k = int(min(k, X.shape[0]))
    pca = PCA(n_components=k, svd_solver="randomized",
              random_state=RANDOM_STATE)
    pca.fit(X)

    print(f"      window PCA (flat) fit: {n_rows} rows streamed, "
          f"{X.shape[0]} fully-real rows kept (k={k} over {n_feat} dims)")
    return pca


def transform_window_block_flat(emb_data, indices, ipca, k):
    """
    Project flattened (window*D,) vectors through the fitted PCA.  Pad slots
    keep their raw mode-fill values (NOT zeroed), so the mode-specific signal
    enters here.  Returns (n, k) float64.
    """
    prepared_path = emb_data["prepared_path"]
    n = len(indices)
    out = np.empty((n, k), dtype=np.float64)
    cur = 0
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        for i0 in range(0, n, WINDOW_CHUNK_ROWS):
            idx = indices[i0:i0 + WINDOW_CHUNK_ROWS]
            block = W[idx].astype(np.float64)   # (c, window, D)
            c = block.shape[0]
            flat = block.reshape(c, -1)         # (c, window*D)
            proj = ipca.transform(flat)         # (c, k)
            out[cur:cur + c] = proj
            cur += c
    return out


def window_feature_names_flat(comp_key, k):
    """k whole-window PC names (order matches the block)."""
    return [f"{comp_key}_PC{i + 1:03d}" for i in range(k)]


# ──────────────────────────────────────────────
# 6c. WINDOW POOLING (kind == "windows", no PCA)
# ──────────────────────────────────────────────
#
# Pooled embeddings collapse the 29-slot window to one (or two) D-dim
# vectors per sample: mean / median / max / min over slots, or a concat
# of two (mean+max, mean+median, mean+std). Standalone chunked reader —
# deliberately NOT reusing _stream_window_slots (different mask semantics).
# No PCA is fitted on pooled output; it goes straight to the model.
#
# pool_mask="masked" (default): reduce over real slots only (pad_mask False).
#   Fully-padded fallback rows (no real slot) use all slots.
# pool_mask="naive": reduce over all 29 slots including filler.

POOL_METHODS = ("mean", "median", "max", "min",
                "mean_max", "mean_median", "mean_std")
POOL_MASKS = ("masked", "naive")
POOL_CHUNK_ROWS = 2048


def _pool_reduce(work, pool):
    """Reduce a (c, W, D) float64 block (pads already NaN or absent).

    Returns list of one (single) or two (concat) (c, D) arrays.
    """
    if pool == "mean":
        return [np.nanmean(work, axis=1)]
    if pool == "median":
        return [np.nanmedian(work, axis=1)]
    if pool == "max":
        return [np.nanmax(work, axis=1)]
    if pool == "min":
        return [np.nanmin(work, axis=1)]
    if pool == "mean_max":
        return [np.nanmean(work, axis=1), np.nanmax(work, axis=1)]
    if pool == "mean_median":
        return [np.nanmean(work, axis=1), np.nanmedian(work, axis=1)]
    if pool == "mean_std":
        return [np.nanmean(work, axis=1), np.nanstd(work, axis=1)]
    raise ValueError(f"Unknown pool method '{pool}'. Valid: {POOL_METHODS}")


def pool_window_block(emb_data, indices, pool, pool_mask="masked"):
    """Pool window rows to (n, D) or (n, 2D) float64 for `indices`.

    `indices` must be increasing (h5py requirement) — guaranteed by all
    callers (boolean-mask subsets of df order).
    """
    if pool not in POOL_METHODS:
        raise ValueError(f"Unknown pool method '{pool}'. Valid: {POOL_METHODS}")
    prepared_path = emb_data["prepared_path"]
    D = emb_data["emb_dim"]
    two = pool in ("mean_max", "mean_median", "mean_std")
    n = len(indices)
    out = np.empty((n, 2 * D if two else D), dtype=np.float64)
    cur = 0
    with h5py.File(prepared_path, "r") as f:
        Wd = f["windows"]
        M = f["pad_mask"] if pool_mask == "masked" else None
        for i0 in range(0, n, POOL_CHUNK_ROWS):
            idx = indices[i0:i0 + POOL_CHUNK_ROWS]
            block = Wd[idx].astype(np.float64)   # (c, W, D)
            c = block.shape[0]
            if M is not None:
                m = M[idx].astype(bool)          # True = padded slot
                full = m.all(axis=1)             # no real slot at all
                work = np.where(m[..., None], np.nan, block)
                if full.any():
                    work[full] = block[full]     # fallback: all slots
            else:
                work = block
            parts = _pool_reduce(work, pool)
            out[cur:cur + c] = (np.concatenate(parts, axis=1)
                                if len(parts) == 2 else parts[0])
            cur += c
    return out


def pool_feature_names(comp_key, pool, D):
    """Pooled feature names (order matches pool_window_block columns)."""
    if pool in ("mean_max", "mean_median", "mean_std"):
        second = pool.split("_")[1]
        names = ([f"{comp_key}_pool-mean_d{d + 1:04d}" for d in range(D)]
                 + [f"{comp_key}_pool-{second}_d{d + 1:04d}" for d in range(D)])
        return names
    return [f"{comp_key}_pool-{pool}_d{d + 1:04d}" for d in range(D)]


# ──────────────────────────────────────────────
# 7. UNIFIED FOLD PREPARATION (composable)
# ──────────────────────────────────────────────

def prepare_fold(df, csv_feature_cols, emb_data_dict, model_cfg, fold_id,
                 validation_bucket=None):
    """
    Prepare train/test data for one CV fold by assembling all components.

    Steps:
      1. Split rows into train/test by fold
      2. Build CSV feature matrix (if any CSV components)
      3. Build PCA-reduced embedding matrices (if any embedding components)
      4. Concatenate all parts horizontally
      5. Apply scaling if required by the model

    Returns:
        X_train, y_train, X_test, y_test, fold_info, fold_artifacts
    """
    needs_scaling = model_cfg["needs_scaling"]

    # ── Row split ──
    # validation_bucket: which bucket to exclude from CV entirely
    # (used in nested CV outer loop). None = use HELD_OUT_INDEX (legacy).
    excl_bucket = validation_bucket if validation_bucket is not None else HELD_OUT_INDEX
    cv_mask = df[FOLD_COL] != excl_bucket
    cv_df = df[cv_mask].copy()
    cv_indices = np.where(cv_mask.values)[0]

    train_mask_cv = cv_df[FOLD_COL] != fold_id
    test_mask_cv = cv_df[FOLD_COL] == fold_id

    train_indices = cv_indices[train_mask_cv.values]
    test_indices = cv_indices[test_mask_cv.values]

    y_train = df.iloc[train_indices][LABEL_COL].values.astype(np.int32)
    y_test = df.iloc[test_indices][LABEL_COL].values.astype(np.int32)

    parts_train = []
    parts_test = []
    feature_names = []
    fold_artifacts = {
        "col_medians": None,
        "scaler": None,
        "pca_dict": {},     # comp_key -> fitted PCA
    }

    # ── CSV features ──
    if csv_feature_cols:
        X_csv_train = df.iloc[train_indices][csv_feature_cols].values.astype(np.float64)
        X_csv_test = df.iloc[test_indices][csv_feature_cols].values.astype(np.float64)

        if model_cfg.get("skip_imputation"):
            # Let models that handle NaN natively (e.g. XGBoost) see missing
            # values directly. Only inf -> NaN is needed (XGB treats NaN as missing).
            X_csv_train[np.isinf(X_csv_train)] = np.nan
            X_csv_test[np.isinf(X_csv_test)] = np.nan
            fold_artifacts["col_medians"] = None
            print(f"    CSV features: {X_csv_train.shape[1]} columns "
                  f"(imputation skipped — NaN passed natively)")
        else:
            X_csv_train, X_csv_test, col_medians = impute_nan(
                X_csv_train, X_csv_test, csv_feature_cols, fold_id,
            )
            fold_artifacts["col_medians"] = col_medians

        parts_train.append(X_csv_train)
        parts_test.append(X_csv_test)
        feature_names.extend(csv_feature_cols)

        print(f"    CSV features: {X_csv_train.shape[1]} columns")

    # ── Embedding features ──
    print("Embedding components loaded:", emb_data_dict.keys())
    for comp_key, emb_data in emb_data_dict.items():
        pca_components = emb_data["pca_components"]

        # ── Window-embedding path (flattened-window PCA or raw) ──
        if emb_data.get("kind") == "windows":
            window = emb_data["window"]
            D = emb_data["emb_dim"]
            pca_mode = emb_data.get("pca_mode", "flat")
            pool = emb_data.get("pool")

            if pool is not None:
                # ── Pooled path: reduce to (N, D)/(N, 2D), NO PCA ──
                mask = emb_data.get("pool_mask", "masked")
                print(f"    {comp_key}: pooling '{pool}' ({mask}) "
                      f"(window={window}, D={D}) ...")
                X_emb_train = pool_window_block(
                    emb_data, train_indices, pool, mask)
                X_emb_test = pool_window_block(
                    emb_data, test_indices, pool, mask)
                ipca = None  # pooled — nothing to transform at predict time
                total_components = X_emb_train.shape[1]
                print(f"    {comp_key}: pool-{pool} ({mask}) "
                      f"{window}x{D} → {total_components} features (no PCA)")
                feature_names.extend(
                    pool_feature_names(comp_key, pool, D))
                fold_artifacts["pca_dict"][comp_key] = ipca
                parts_train.append(X_emb_train)
                parts_test.append(X_emb_test)
                continue

            if pca_mode == "flat_raw":
                total_components = int(window * D)   # no PCA — all raw dims
            elif pca_mode == "simple21":
                dims = emb_data.get("dims")
                if dims is None:
                    raise ValueError(
                        "simple21 requires --dims (global top-N dim list)")
                total_components = int(window * len(dims))
            elif pca_mode == "dimselect":
                # placeholder; set after screening inside the branch below
                total_components = int(window * D)
            elif pca_mode == "varfold":
                # placeholder; set after train-only ranking in the branch below
                total_components = int(window * D)
            elif pca_mode == "autoencoder":
                ae_cfg = emb_data.get("autoencoder")
                latent = (ae_cfg["latent"] if ae_cfg else AE_DEFAULT_LATENT)
                total_components = int(window * latent)
            else:
                total_components = int(min(pca_components, window * D))

            if pca_mode == "flat":
                print(f"    {comp_key}: fitting flattened-window PCA "
                      f"(window={window}, D={D}, k={total_components}) ...")
                ipca = fit_window_pca_flat(emb_data, train_indices, total_components)
                X_emb_train = transform_window_block_flat(
                    emb_data, train_indices, ipca, total_components,
                )
                X_emb_test = transform_window_block_flat(
                    emb_data, test_indices, ipca, total_components,
                )
            elif pca_mode == "flat_raw":
                print(f"    {comp_key}: flat-raw (no PCA) "
                      f"(window={window}, D={D} → {total_components} features)")
                X_emb_train = concat_window_slots(emb_data, train_indices)
                X_emb_test = concat_window_slots(emb_data, test_indices)
                ipca = None  # raw features — nothing to transform at predict time
            elif pca_mode == "simple21":
                dims = emb_data.get("dims")
                if dims is None:
                    raise ValueError(
                        "simple21 requires --dims (global top-N dim list)")
                print(f"    {comp_key}: simple21 dim gather (no PCA) "
                      f"({window}x{len(dims)} dims → {total_components} features)")
                X_emb_train = concat_window_slots(
                    emb_data, train_indices, dims=dims)
                X_emb_test = concat_window_slots(
                    emb_data, test_indices, dims=dims)
                ipca = None  # raw gather — nothing to transform at predict time
            elif pca_mode == "dimselect":
                ds_cfg = emb_data.get("dimselect")
                if ds_cfg is None:
                    raise ValueError(
                        "dimselect requires --top-n (wired in main)")
                dimscreen = importlib.import_module("12_dim_screening")
                folds_train = df.iloc[train_indices][FOLD_COL].values
                ranking = dimscreen.screen_dims(
                    emb_data["prepared_path"], train_indices, y_train,
                    folds_train, ds_cfg["screen_model"],
                    cache_dir=ds_cfg["cache_dir"],
                    cache_key=(f"{comp_key}__excl{excl_bucket}"
                               f"__test{fold_id}__{ds_cfg['screen_model']}"),
                    n_jobs=ds_cfg["n_jobs"],
                    tag=f"{comp_key} excl={excl_bucket} test={fold_id}",
                )
                # rank by discriminative AUC (inverted dims count); keep
                # original ascending dim order for the gathered block
                dims = (ranking.nlargest(ds_cfg["top_n"], "auc_disc")
                        ["dim"].sort_values().tolist())
                total_components = int(window * len(dims))
                print(f"    {comp_key}: dimselect top-{len(dims)} dims "
                      f"(screen={ds_cfg['screen_model']}, fold-nested) "
                      f"→ {total_components} raw features")
                X_emb_train = concat_window_slots(
                    emb_data, train_indices, dims=dims)
                X_emb_test = concat_window_slots(
                    emb_data, test_indices, dims=dims)
                ipca = None  # raw screened gather — no transform
                fold_artifacts.setdefault("dimselect_dims", {})[comp_key] = dims
            elif pca_mode == "varfold":
                vf_cfg = emb_data.get("varfold")
                if vf_cfg is None:
                    raise ValueError(
                        "varfold requires --top-n (wired in main)")
                # fold-nested selection: the ranking sees ONLY train rows —
                # leakage-free counterpart of global simple21 --dims
                scores = variance_rank_dims(
                    emb_data["prepared_path"], train_indices)
                order = np.argsort(-scores)
                dims = sorted(int(d) for d in order[:vf_cfg["top_n"]])
                # Jaccard diagnostic vs the GLOBAL top-N (same metric over
                # all rows, computed once per run) — report only; the global
                # set is never used to build features
                if emb_data.get("_varfold_global_rank") is None:
                    emb_data["_varfold_global_rank"] = np.argsort(
                        -variance_rank_dims(
                            emb_data["prepared_path"],
                            np.arange(emb_data["n_samples"], dtype=np.int64)))
                gtop = set(int(d) for d in
                           emb_data["_varfold_global_rank"][:vf_cfg["top_n"]])
                jac = len(gtop & set(dims)) / len(gtop | set(dims))
                total_components = int(window * len(dims))
                print(f"    {comp_key}: varfold top-{len(dims)} "
                      f"(train-only, fold-nested) → {total_components} raw "
                      f"features | Jaccard vs global top-N = {jac:.3f}")
                X_emb_train = concat_window_slots(
                    emb_data, train_indices, dims=dims)
                X_emb_test = concat_window_slots(
                    emb_data, test_indices, dims=dims)
                ipca = None  # raw gather — no transform at predict time
                # fold-selected dims ride the same artifact slot dimselect
                # uses, so the existing predict path (sel_dims) replays
                # exactly these dims on validation/held-out
                fold_artifacts.setdefault("dimselect_dims", {})[comp_key] = dims
            elif pca_mode == "autoencoder":
                ae_cfg = emb_data.get("autoencoder")
                if ae_cfg is None:
                    raise ValueError(
                        "autoencoder requires --ae-latent (wired in main)")
                ae = importlib.import_module("13_autoencoder_windows")
                print(f"    {comp_key}: training 1x1-conv bottleneck AE "
                      f"(D={D} → latent={ae_cfg['latent']}, fold-nested) ...")
                t_ae = time.time()
                encoder, ae_log = ae.train_encoder(
                    emb_data, train_indices, latent=ae_cfg["latent"],
                    epochs=ae_cfg["epochs"], lr=ae_cfg["lr"],
                )
                print(f"    {comp_key}: AE fit in {time.time() - t_ae:.1f}s "
                      f"(epochs_run={ae_log['epochs_run']}, "
                      f"best_epoch={ae_log['best_epoch']}, "
                      f"val masked-MSE={ae_log['best_val_masked_mse']:.4f}, "
                      f"train_rows={ae_log['train_rows']})")
                total_components = int(window * ae_cfg["latent"])
                X_emb_train = ae.encode_block(
                    encoder, emb_data, train_indices
                ).reshape(len(train_indices), total_components)
                X_emb_test = ae.encode_block(
                    encoder, emb_data, test_indices
                ).reshape(len(test_indices), total_components)
                # fitted encoder rides the pca_dict artifact slot;
                # _transform_embeddings duck-types it via encode_indices
                ipca = encoder

            explained = (ipca.explained_variance_ratio_.sum() * 100
                         if hasattr(ipca, "explained_variance_ratio_") else 100.0)
            if pca_mode == "flat":
                print(f"    {comp_key}: window PCA {window * D} → k={total_components} "
                      f"({total_components} features) "
                      f"({explained:.1f}% variance)")
                feature_names.extend(
                    window_feature_names_flat(comp_key, total_components)
                )
            elif pca_mode == "flat_raw":
                print(f"    {comp_key}: flat-raw features "
                      f"{window}x{D} → {total_components} (no PCA, 100% by definition)")
                feature_names.extend(
                    window_feature_names_flat(comp_key, total_components)
                )
            elif pca_mode == "simple21":
                print(f"    {comp_key}: simple21 features "
                      f"{window}x{total_components // window} → {total_components} (no PCA, 100% by definition)")
                feature_names.extend(
                    window_feature_names_flat(comp_key, total_components)
                )
            elif pca_mode == "dimselect":
                print(f"    {comp_key}: dimselect features "
                      f"{window}x{total_components // window} → {total_components} (raw screened dims, no PCA)")
                feature_names.extend(
                    window_feature_names_flat(comp_key, total_components)
                )
            elif pca_mode == "varfold":
                print(f"    {comp_key}: varfold features "
                      f"{window}x{total_components // window} → {total_components} "
                      f"(raw variance-ranked dims, no PCA)")
                # layout matches concat_window_slots: pos-major, dims ascending
                feature_names.extend(
                    f"{comp_key}_pos{p:02d}_dim{d:03d}"
                    for p in range(window) for d in dims
                )
            elif pca_mode == "autoencoder":
                latent = total_components // window
                print(f"    {comp_key}: autoencoder features "
                      f"{window}x{latent} → {total_components} "
                      f"(1x1-conv bottleneck, fold-nested)")
                # layout matches encode_block reshape: pos-major, latent asc
                feature_names.extend(
                    f"{comp_key}_pos{p:02d}_z{i:03d}"
                    for p in range(window) for i in range(latent)
                )

            fold_artifacts["pca_dict"][comp_key] = ipca
            parts_train.append(X_emb_train)
            parts_test.append(X_emb_test)
            continue

        # ── Legacy mean-pooled path ──
        # Single context matrix, clean inf/nan
        X_emb_train = get_context_matrix(emb_data, train_indices)
        X_emb_test = get_context_matrix(emb_data, test_indices)
        X_emb_train[np.isinf(X_emb_train)] = 0.0
        X_emb_train[np.isnan(X_emb_train)] = 0.0
        X_emb_test[np.isinf(X_emb_test)] = 0.0
        X_emb_test[np.isnan(X_emb_test)] = 0.0

        # ── Zero-vector handling ──
        train_norms = np.linalg.norm(X_emb_train, axis=1)
        test_norms = np.linalg.norm(X_emb_test, axis=1)

        train_nonzero_mask = train_norms > 0.0
        test_nonzero_mask = test_norms > 0.0

        n_train_zero = (~train_nonzero_mask).sum()
        n_test_zero = (~test_nonzero_mask).sum()
        pct_train_zero = n_train_zero / len(train_norms) * 100
        pct_test_zero = n_test_zero / len(test_norms) * 100

        print(f"    {comp_key}: zero-vector samples: "
              f"train={n_train_zero}/{len(train_norms)} ({pct_train_zero:.1f}%), "
              f"test={n_test_zero}/{len(test_norms)} ({pct_test_zero:.1f}%)")

        # ── PCA on the single context array ──
        X_emb_train_valid = X_emb_train[train_nonzero_mask]

        total_components = min(
            pca_components,
            X_emb_train_valid.shape[1],
            X_emb_train_valid.shape[0],
        )

        pca = PCA(n_components=total_components, random_state=RANDOM_STATE)
        pca.fit(X_emb_train_valid)

        X_emb_train = pca.transform(X_emb_train)
        X_emb_test = pca.transform(X_emb_test)

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"    {comp_key}: PCA {emb_data['emb_dim']} → "
              f"{total_components} ({explained:.1f}% variance, fitted on "
              f"{train_nonzero_mask.sum()} non-zero samples)")

        fold_artifacts["pca_dict"][comp_key] = pca
        feature_names.extend(
            [f"{comp_key}_PC{i + 1:03d}" for i in range(total_components)]
        )

        parts_train.append(X_emb_train)
        parts_test.append(X_emb_test)

    # ── Concatenate all parts ──
    X_train = np.concatenate(parts_train, axis=1)
    X_test = np.concatenate(parts_test, axis=1)
    print(f"    Total features after concatenation: {X_train.shape[1]}")
    # ── Scaling ──
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        fold_artifacts["scaler"] = scaler

    fold_info = {
        "fold_id": fold_id,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int(len(y_train) - y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int(len(y_test) - y_test.sum()),
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
    }

    return X_train, y_train, X_test, y_test, fold_info, fold_artifacts


def prepare_held_out(df, csv_feature_cols, emb_data_dict, model_cfg, fold_artifacts):
    """
    Prepare the held-out set using artifacts (medians, PCA, scaler)
    fitted during the best fold's training.
    """
    ho_mask = df[FOLD_COL] == HELD_OUT_INDEX
    ho_indices = np.where(ho_mask.values)[0]

    y_ho = df.iloc[ho_indices][LABEL_COL].values.astype(np.int32)

    parts = []

    # ── CSV features ──
    if csv_feature_cols:
        X_csv = df.iloc[ho_indices][csv_feature_cols].values.astype(np.float64)
        if fold_artifacts["col_medians"] is not None:
            X_csv = impute_nan_single(X_csv, fold_artifacts["col_medians"])
        else:
            X_csv[np.isinf(X_csv)] = np.nan
        parts.append(X_csv)

    # ── Embedding features ──
    for comp_key, emb_data in emb_data_dict.items():
        pca_obj = fold_artifacts["pca_dict"][comp_key]
        sel_dims = (fold_artifacts.get("dimselect_dims") or {}).get(comp_key)
        X_emb = _transform_embeddings(emb_data, ho_indices, pca_obj,
                                      sel_dims=sel_dims)
        parts.append(X_emb)

    X_ho = np.concatenate(parts, axis=1)

    if fold_artifacts["scaler"] is not None:
        X_ho = fold_artifacts["scaler"].transform(X_ho)

    return X_ho, y_ho


def prepare_validation(df, csv_feature_cols, emb_data_dict, model_cfg,
                       fold_artifacts, validation_bucket):
    """
    Prepare validation set using artifacts from a specific inner fold.

    Like prepare_held_out but parameterized by validation_bucket
    instead of hardcoding HELD_OUT_INDEX.
    """
    val_mask = df[FOLD_COL] == validation_bucket
    val_indices = np.where(val_mask.values)[0]

    y_val = df.iloc[val_indices][LABEL_COL].values.astype(np.int32)

    parts = []

    # ── CSV features ──
    if csv_feature_cols:
        X_csv = df.iloc[val_indices][csv_feature_cols].values.astype(np.float64)
        if fold_artifacts["col_medians"] is not None:
            X_csv = impute_nan_single(X_csv, fold_artifacts["col_medians"])
        else:
            X_csv[np.isinf(X_csv)] = np.nan
        parts.append(X_csv)

    # ── Embedding features ──
    for comp_key, emb_data in emb_data_dict.items():
        pca_obj = fold_artifacts["pca_dict"][comp_key]
        # dimselect: the fold-specific screened dims live in THIS fold's
        # artifacts (emb_data["dims"] is simple21-only and shared/global)
        sel_dims = (fold_artifacts.get("dimselect_dims") or {}).get(comp_key)
        X_emb = _transform_embeddings(emb_data, val_indices, pca_obj,
                                      sel_dims=sel_dims)
        parts.append(X_emb)

    X_val = np.concatenate(parts, axis=1)

    if fold_artifacts["scaler"] is not None:
        X_val = fold_artifacts["scaler"].transform(X_val)

    return X_val, y_val, val_indices


def _transform_embeddings(emb_data, indices, pca_obj, sel_dims=None):
    """
    Transform the single context embedding through its fitted PCA
    (legacy), or stream window slots through the fitted window PCA.
    Pooled window embeddings (emb_data["pool"] set) bypass PCA entirely.
    """
    if emb_data.get("kind") == "windows":
        if emb_data.get("pool") is not None:
            # pooled — identical reduce at predict time
            return pool_window_block(
                emb_data, indices, emb_data["pool"],
                emb_data.get("pool_mask", "masked"))
        if pca_obj is None:
            # flat_raw / simple21 / dimselect — features were raw at fit
            # time; identical at predict time. simple21 must re-apply the
            # same dims gather used at training (emb_data["dims"] from
            # --dims), otherwise the full window comes back and predict
            # crashes on shape mismatch. dimselect passes its fold-specific
            # screened dims explicitly (sel_dims).
            return concat_window_slots(
                emb_data, indices,
                dims=sel_dims if sel_dims is not None
                else emb_data.get("dims"))
        if hasattr(pca_obj, "encode_indices"):
            # autoencoder replay: fold's fitted encoder (13_autoencoder_
            # windows.TrainedEncoder) — same encode+flatten as fit time
            block = pca_obj.encode_indices(emb_data, indices)
            return block.reshape(block.shape[0], -1)
        k = int(getattr(pca_obj, "n_components_", pca_obj.n_components))
        return transform_window_block_flat(emb_data, indices, pca_obj, k)
    X_emb = get_context_matrix(emb_data, indices)
    X_emb[np.isinf(X_emb)] = 0.0
    X_emb[np.isnan(X_emb)] = 0.0
    return pca_obj.transform(X_emb)


def predict_with_averaging(df, csv_feature_cols, emb_data_dict, model_cfg,
                           inner_models, inner_artifacts, validation_bucket):
    """
    Predict on validation set by averaging all inner models' probabilities.

    Each inner model has its own PCA/scaler/median artifacts.
    We transform the validation data through each model's pipeline,
    predict, then average the probability vectors.
    """
    all_probs = []
    for model, artifacts in zip(inner_models, inner_artifacts):
        X_val, y_val, _ = prepare_validation(
            df, csv_feature_cols, emb_data_dict, model_cfg,
            artifacts, validation_bucket
        )
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            probs = model.decision_function(X_val)
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    return y_val, avg_probs


# ──────────────────────────────────────────────
# 8. MODEL TRAINING
# ──────────────────────────────────────────────

def train_one_fold(model_cfg, X_train, y_train, X_test, y_test, fold_id):
    param_overrides = {}
    if model_cfg.get("scale_pos_weight") == "auto":
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
        param_overrides["scale_pos_weight"] = ratio
        print(f"    scale_pos_weight (auto) = {ratio:.4f} "
              f"(neg={n_neg}, pos={n_pos})")

    model = build_model(model_cfg, param_overrides=param_overrides)
    display = model_cfg["display_name"]

    print(f"    Training {display} ...")
    t_start = time.time()
    # XGB early stopping: rounds live in the constructor (xgboost >= 1.6,
    # server 3.4.1); fit only supplies the eval set. Stop after
    # `early_stopping_rounds` trees without logloss improvement on the
    # inner fold's test bucket; each fold keeps its own tree count
    # (model.best_iteration_).
    es_rounds = model_cfg["params"].get("early_stopping_rounds")
    is_xgb = model_cfg["model_class"].endswith("XGBClassifier")
    if is_xgb and es_rounds:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        best_iter = getattr(model, "best_iteration_",
                            getattr(model, "best_iteration", None))
        print(f"    early stopping: rounds={es_rounds}, "
              f"best_iteration={best_iter}")
    else:
        model.fit(X_train, y_train)
    t_end = time.time()
    print(f"    Training time: {t_end - t_start:.1f}s")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    test_metrics = compute_metrics(y_test, y_prob)
    return model, test_metrics, y_prob


# ──────────────────────────────────────────────
# 9. FEATURE WEIGHTS
# ──────────────────────────────────────────────

def extract_feature_weights(model, model_cfg):
    attr = model_cfg.get("coef_attr")
    if attr is None or not hasattr(model, attr):
        return None
    raw = getattr(model, attr)
    return np.asarray(raw).ravel()


def print_feature_weights(weights, feature_names, model_cfg, top_n=20):
    if weights is None:
        print("\n    (model does not expose feature weights)")
        return weights, None

    abs_weights = np.abs(weights)
    indices = np.argsort(abs_weights)[::-1]

    is_coef = model_cfg["coef_attr"] == "coef_"
    col_label = "Coefficient" if is_coef else "Importance"

    print(f"\n    Top {min(top_n, len(feature_names))} feature {col_label.lower()}s:")
    print(f"    {'Rank':<6} {'Feature':<45} {col_label:>12}")
    print(f"    {'-' * 65}")
    for rank in range(min(top_n, len(feature_names))):
        idx = indices[rank]
        line = f"    {rank + 1:<6} {feature_names[idx]:<45} {weights[idx]:>12.6f}"
        print(line)

    return weights, indices


# ──────────────────────────────────────────────
# 10. AGGREGATE CV RESULTS
# ──────────────────────────────────────────────

def aggregate_cv_results(all_fold_metrics):
    metric_names = all_fold_metrics[0].keys()
    summary = {}
    for m in metric_names:
        values = [fold[m] for fold in all_fold_metrics]
        summary[m] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }
    return summary


def print_cv_summary(summary):
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION SUMMARY (mean +/- std)")
    print("=" * 80)
    for metric, stats in summary.items():
        vals = "  ".join(f"{v:.4f}" for v in stats["values"])
        print(f"  {metric:<15} {stats['mean']:.4f} +/- {stats['std']:.4f}    "
              f"[{vals}]")


# ──────────────────────────────────────────────
# 11. CLI
# ──────────────────────────────────────────────

def parse_pca_value(value):
    """Parse PCA value: int ('13') or per-embedding dict ('esmc=1,esmif=9')."""
    if "=" in value:
        result = {}
        for pair in value.split(","):
            k, v = pair.split("=")
            result[k.strip()] = int(v.strip())
        return result
    return int(value)


def parse_dims_value(value):
    """Parse --dims: '0,1,5' or path to JSON list / {'dims': [...]}."""
    if value.endswith(".json"):
        with open(value) as fh:
            data = json.load(fh)
        dims = data.get("dims") if isinstance(data, dict) else data
        if dims is None:
            raise ValueError(f"JSON must hold a list or a 'dims' key: {value}")
    else:
        dims = [int(t) for t in value.split(",") if t.strip() != ""]
    return sorted({int(s) for s in dims})  # dedup, original order


def parse_args():
    parser = argparse.ArgumentParser(
        description="04_modelling: cross-validation + held-out evaluation",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model key from MODEL_REGISTRY (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--features", type=str, default=DEFAULT_FEATURE_SET,
        help=f"Feature set key from FEATURE_SETS (default: {DEFAULT_FEATURE_SET})",
    )
    parser.add_argument(
        "--pca", type=parse_pca_value, default=None,
        help="Override PCA components: int for all embeddings ('13') "
             "or per-embedding dict ('esmc=1,esmif=9')",
    )
    parser.add_argument(
        "--dims", type=parse_dims_value, default=None,
        help="Dim list for --pca-mode simple21 (29x21 test): '0,1,5' or "
             "path to 10_dim_variance top-N JSON. Keeps all 29 slots.",
    )
    parser.add_argument(
        "--pca-mode", type=str, default="flat",
        choices=["flat", "flat_raw", "simple21", "dimselect", "varfold",
                 "autoencoder"],
        help="Window reduction scheme: 'flat' (flatten window*D, PCA across "
             "the full window, default), 'flat_raw' (flatten window*D, NO PCA "
             "— all raw dims; low mem), 'simple21' (29x21 dim gather, "
             "no PCA — positional test, requires --dims), 'dimselect' "
             "(per-dim supervised screening, top-N dims raw; requires "
             "--top-n; see 12_dim_screening.py), 'varfold' (fold-nested "
             "variance ranking, top-N dims raw; requires --top-n), or "
             "'autoencoder' (1x1-conv bottleneck AE, fold-nested; requires "
             "--ae-latent; see 13_autoencoder_windows.py)",
    )
    parser.add_argument(
        "--ae-latent", type=int, default=None, dest="ae_latent",
        help="Latent dim d for --pca-mode autoencoder (features = 29 x d). "
             "Encoder trained fold-nested on train rows only (no leakage).",
    )
    parser.add_argument(
        "--ae-epochs", type=int, default=15, dest="ae_epochs",
        help="Max AE training epochs (early stop patience 3 on a 10%% "
             "train slice; default 15).",
    )
    parser.add_argument(
        "--ae-lr", type=float, default=1e-3, dest="ae_lr",
        help="AE AdamW learning rate (default 1e-3).",
    )
    parser.add_argument(
        "--pool", type=str, default=None,
        choices=["mean", "median", "max", "min",
                 "mean_max", "mean_median", "mean_std"],
        help="Pool window slots to one (or two) D-dim vectors per sample "
             "BEFORE modelling, with NO PCA (e.g. --pool mean). "
             "Omit = current flat-window behavior. "
             "Incompatible with --pca / non-flat --pca-mode.",
    )
    parser.add_argument(
        "--pool-mask", type=str, default="masked",
        choices=["masked", "naive"],
        help="Pooling mask: 'masked' (default) reduces over real slots "
             "only (pad_mask); 'naive' reduces over all 29 slots "
             "including filler. Requires --pool.",
    )
    parser.add_argument(
        "--top-n", type=int, default=None, dest="top_n",
        help="Keep top-N dims for --pca-mode dimselect (screened, fold-"
             "nested/cached) or varfold (variance-ranked, fold-nested/"
             "leak-free, recomputed per fold). Features = 29 x N.",
    )
    parser.add_argument(
        "--screen-model", type=str, default="xgb",
        choices=["xgb", "lr_l2"], dest="screen_model",
        help="Tiny per-dim screener for --pca-mode dimselect "
             "(small XGB or LR-l2; both rankers comparable post-hoc).",
    )
    parser.add_argument(
        "--screen-cache-dir", type=str, default=None,
        dest="screen_cache_dir",
        help="Cache dir for per-fold dim rankings. Default: "
             "<out-root>/screening. Share one dir across an N-sweep so "
             "rankings are computed once per (embedding, fold, model).",
    )
    parser.add_argument(
        "--screen-jobs", type=int, default=12, dest="screen_jobs",
        help="joblib workers for the per-dim screening loop "
             "(default 12; screening models themselves are single-core).",
    )
    parser.add_argument(
        "--C", type=float, default=None,
        help="Override inverse regularization strength C (default: config.py value, "
             "smaller = stronger reg)",
    )
    parser.add_argument(
        "--xgb-params", type=str, default=None, dest="xgb_params",
        help="Override XGB params as JSON dict, e.g. "
             "'{\"learning_rate\": 0.05, \"n_estimators\": 2500}'. "
             "Merged into MODEL_REGISTRY['xgb']['params'].",
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Run-folder root shared by a whole submission "
             "(e.g. runs/4padmodes_and_PCA_handcrafted_sparse_with_embeddings). "
             "Default: automatic "
             "runs/<run_tag>_<timestamp>/ folder per run (no flags needed).",
    )
    return parser.parse_args()

# ──────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    model_key = args.model
    features_key = args.features
    pca_mode = args.pca_mode
    pool = args.pool
    pool_mask = args.pool_mask

    if pool is not None:
        if args.pca is not None:
            raise ValueError("--pool is incompatible with --pca "
                             "(pooled embeddings use no PCA)")
        if pca_mode != "flat":
            raise ValueError("--pool is incompatible with "
                             f"--pca-mode {pca_mode} (use default flat)")
        if args.dims is not None or args.top_n is not None:
            raise ValueError("--pool is incompatible with "
                             "--dims / --top-n")
        print(f"  [CLI OVERRIDE] pool: {pool} ({pool_mask})")
    elif args.pool_mask != "masked":
        raise ValueError("--pool-mask requires --pool")

    validate_config()
    model_cfg = get_model_config(model_key)
    feat_cfg = get_feature_set_config(features_key)
    validate_feature_set(feat_cfg)

    # ── PCA override ──
    pca_override = args.pca
    if pca_override is not None:
        print(f"  [CLI OVERRIDE] PCA components: {pca_override}")

    # ── C override ──
    if args.C is not None:
        print(f"  [CLI OVERRIDE] C: {args.C}")
        model_cfg["params"]["C"] = args.C

    # ── XGB params override (JSON dict merged into registry params) ──
    # shortcut: constructor params only (lr, n_estimators, depth...).
    # early-stopping (fit-time) not wired yet.
    xgb_params_override = None
    if args.xgb_params is not None:
        xgb_params_override = json.loads(args.xgb_params)
        if not isinstance(xgb_params_override, dict):
            raise ValueError("--xgb-params must be a JSON object (dict)")
        print(f"  [CLI OVERRIDE] xgb-params: {xgb_params_override}")
        model_cfg["params"].update(xgb_params_override)

    display_name = model_cfg["display_name"]
    feat_display = feat_cfg["display_name"]

    if pca_override is not None:
        if isinstance(pca_override, dict):
            pca_part = "_".join(f"{k}{v}" for k, v in sorted(pca_override.items()))
            run_tag = f"{model_key}_{features_key}_pca_{pca_part}"
        else:
            run_tag = f"{model_key}_{features_key}_pca{pca_override}"
    else:
        run_tag = f"{model_key}_{features_key}"
    run_tag += f"_{pca_mode}"
    if pool is not None:
        run_tag += f"_pool{pool}"
        if pool_mask != "masked":
            run_tag += f"_{pool_mask}"
    if args.dims is not None:
        run_tag += f"_var{len(args.dims)}"
    if pca_mode == "dimselect":
        run_tag += f"_top{args.top_n}_{args.screen_model}"
    if pca_mode == "varfold":
        run_tag += f"_top{args.top_n}"
    if pca_mode == "autoencoder":
        run_tag += f"_lat{args.ae_latent}"
    if xgb_params_override:
        lr = xgb_params_override.get("learning_rate")
        n_est = xgb_params_override.get("n_estimators")
        if lr is not None or n_est is not None:
            lr_part = f"lr{str(lr).replace('.', 'p')}" if lr is not None else ""
            n_part = f"n{n_est}" if n_est is not None else ""
            run_tag += f"_xgb{lr_part}{n_part}"
        else:
            run_tag += "_xgbcustom"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Namespace all run outputs by PCA experiment so flat/flat_raw batches
    # never mix (logs, cv_results, pickles). Baselines run with the
    # default pca_mode=flat, so they live under flat/ as well.
    # --out-root redirects the base (campaign folder shared by a whole
    # submission). Default = automatic per-run folder
    # runs/<run_tag>_<timestamp>/ — no flags needed. Legacy repo-root
    # paths receive no new files ("." also means automatic).
    if args.out_root and args.out_root != ".":
        out_base = Path(args.out_root)
    else:
        out_base = LOG_DIR.parent / "runs" / f"{run_tag}_{timestamp}"
    log_dir = out_base / "logs" / pca_mode
    model_dir = out_base / "models" / pca_mode
    LOG_PATH = log_dir / f"04_modelling_{run_tag}_log_{timestamp}.txt"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    # -- Header --
    print("=" * 80)
    print(f"  04_MODELLING -- {display_name.upper()} + {feat_display.upper()}")
    print("=" * 80)
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model key:       {model_key}")
    print(f"  Model class:     {model_cfg['model_class']}")
    print(f"  Feature set:     {features_key} ({feat_display})")
    print(f"  PCA mode:        {pca_mode}")
    print(f"  Pool:            {pool if pool is not None else '-'}"
          + (f" ({pool_mask})" if pool is not None else ""))
    print(f"  Components:      {feat_cfg['components']}")
    print(f"  Data path:       {SPLIT_DATA_PATH}")
    print(f"  Log path:        {LOG_PATH}")
    print(f"  CV folds (k):    {N_CV_FOLDS}")
    print(f"  Held-out bucket: {HELD_OUT_INDEX}")
    print(f"  Random state:    {RANDOM_STATE}")
    print(f"  Needs scaling:   {model_cfg['needs_scaling']}")
    print(f"\n  Model hyperparameters:")
    for k, v in model_cfg["params"].items():
        print(f"    {k:<25} {v}")
    print("=" * 80)

    # -- Load split data --
    print(f"\nLoading split data: {SPLIT_DATA_PATH}")
    df_split = pd.read_csv(SPLIT_DATA_PATH)
    print(f"  Shape: {df_split.shape[0]} rows x {df_split.shape[1]} columns")

    # ── Per-peptide pad-count metadata (always present, sourced from df_all
    # coordinates, NOT from the embedding pad_mask). These are context features
    # that exist for every peptide regardless of which embedding/PCA is used,
    # so they live outside the PCA pipeline entirely.
    if "start" in df_split.columns and "n_pad" not in df_split.columns:
        df_split["n_pad"] = 10 - np.minimum(
            10, df_split["start"].astype(int) - 1)
    if ("protein_length" in df_split.columns and "end" in df_split.columns
            and "c_pad" not in df_split.columns):
        df_split["c_pad"] = 10 - np.minimum(
            10, df_split["protein_length"].astype(int)
            - df_split["end"].astype(int))

    if FOLD_COL not in df_split.columns:
        print(f"FATAL: Column '{FOLD_COL}' not found. Run 01_datasplit.py first.")
        logger.close()
        sys.exit(1)

    actual_folds = sorted(df_split[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))
    if actual_folds != expected_folds:
        print(f"FATAL: Expected folds {expected_folds}, found {actual_folds}")
        logger.close()
        sys.exit(1)
    print(f"  [OK] Fold column validated: {actual_folds}")

    # -- Load all components --
    print(f"\n{'=' * 80}")
    print("LOADING FEATURE COMPONENTS")
    print("=" * 80)

    df, csv_feature_cols, emb_data_dict, component_info = load_all_components(
        df_split, feat_cfg, pca_mode, pool=pool, pool_mask=pool_mask,
    )

    # ── Dims override (29x21: global top-N dim gather, simple21 only) ──
    if pca_mode == "simple21" and args.dims is None:
        raise ValueError("--pca-mode simple21 requires --dims")
    if args.dims is not None:
        if pca_mode != "simple21":
            raise ValueError("--dims requires --pca-mode simple21")
        print(f"  [CLI OVERRIDE] dims: {args.dims}")
        for comp_key, emb_data in emb_data_dict.items():
            if emb_data.get("kind") != "windows":
                continue
            edim = emb_data["emb_dim"]
            if (not args.dims or min(args.dims) < 0
                    or max(args.dims) >= edim):
                raise ValueError(
                    f"--dims out of range for D={edim}: {args.dims}")
            emb_data["dims"] = list(args.dims)
        for info in component_info["emb_components"]:
            if info.get("window"):
                info["pca_total"] = info["window"] * len(args.dims)
                info["raw_dim_desc"] = (
                    f"{info['window']}x{len(args.dims)} "
                    f"(simple21 dim21 {info['window'] * len(args.dims)})")

    # ── Dimselect wiring (per-dim supervised screening, fold-nested) ──
    if pca_mode == "dimselect":
        if args.top_n is None or args.top_n < 1:
            raise ValueError("--pca-mode dimselect requires --top-n N")
        screen_cache = (Path(args.screen_cache_dir)
                        if args.screen_cache_dir
                        else out_base / "screening")
        print(f"  [DIMSELECT] top-n={args.top_n}, "
              f"screen={args.screen_model}, cache={screen_cache}")
        for comp_key, emb_data in emb_data_dict.items():
            if emb_data.get("kind") != "windows":
                raise ValueError(
                    "dimselect supports window embeddings only "
                    f"(component '{comp_key}' is '{emb_data.get('kind')}')")
            if args.top_n > emb_data["emb_dim"]:
                raise ValueError(
                    f"--top-n {args.top_n} > D={emb_data['emb_dim']} "
                    f"({comp_key})")
            emb_data["dimselect"] = {
                "top_n": int(args.top_n),
                "screen_model": args.screen_model,
                "cache_dir": str(screen_cache),
                "n_jobs": int(args.screen_jobs),
            }
        for info in component_info["emb_components"]:
            if info.get("window"):
                info["pca_total"] = info["window"] * args.top_n
                info["raw_dim_desc"] = (
                    f"{info['window']}xD screened (dimselect top"
                    f"{args.top_n} = {info['window'] * args.top_n})")

    # ── Varfold wiring (fold-nested unsupervised variance ranking) ──
    if pca_mode == "varfold":
        if args.top_n is None or args.top_n < 1:
            raise ValueError("--pca-mode varfold requires --top-n N")
        print(f"  [VARFOLD] top-n={args.top_n} "
              f"(train-only variance ranking, recomputed per fold)")
        for comp_key, emb_data in emb_data_dict.items():
            if emb_data.get("kind") != "windows":
                raise ValueError(
                    "varfold supports window embeddings only "
                    f"(component '{comp_key}' is '{emb_data.get('kind')}')")
            if args.top_n > emb_data["emb_dim"]:
                raise ValueError(
                    f"--top-n {args.top_n} > D={emb_data['emb_dim']} "
                    f"({comp_key})")
            emb_data["varfold"] = {"top_n": int(args.top_n)}
        for info in component_info["emb_components"]:
            if info.get("window"):
                info["pca_total"] = info["window"] * args.top_n
                info["raw_dim_desc"] = (
                    f"{info['window']}xD variance-ranked (varfold top"
                    f"{args.top_n} = {info['window'] * args.top_n})")

    # ── Autoencoder wiring (fold-nested 1x1-conv bottleneck) ──
    if pca_mode == "autoencoder":
        if args.ae_latent is None or args.ae_latent < 1:
            raise ValueError("--pca-mode autoencoder requires --ae-latent d")
        print(f"  [AUTOENCODER] latent={args.ae_latent}, "
              f"epochs={args.ae_epochs}, lr={args.ae_lr} "
              f"(trained fold-nested on train rows only)")
        for comp_key, emb_data in emb_data_dict.items():
            if emb_data.get("kind") != "windows":
                raise ValueError(
                    "autoencoder supports window embeddings only "
                    f"(component '{comp_key}' is '{emb_data.get('kind')}')")
            if args.ae_latent >= emb_data["emb_dim"]:
                raise ValueError(
                    f"--ae-latent {args.ae_latent} >= D={emb_data['emb_dim']} "
                    f"({comp_key}) — latent must shrink")
            emb_data["autoencoder"] = {
                "latent": int(args.ae_latent),
                "epochs": int(args.ae_epochs),
                "lr": float(args.ae_lr),
            }
        for info in component_info["emb_components"]:
            if info.get("window"):
                info["pca_total"] = info["window"] * args.ae_latent
                info["raw_dim_desc"] = (
                    f"{info['window']}xD 1x1-conv AE latent"
                    f"{args.ae_latent} (= {info['window'] * args.ae_latent})")

    # Apply PCA override if specified
    if pca_override is not None:
        for comp_key, emb_data in emb_data_dict.items():
            if isinstance(pca_override, dict):
                if comp_key in pca_override:
                    emb_data["pca_components"] = pca_override[comp_key]
            else:
                emb_data["pca_components"] = pca_override
        # Update component_info for logging
        for info in component_info["emb_components"]:
            if isinstance(pca_override, dict):
                key = info.get("key", "")
                if key not in pca_override:
                    continue
                k = pca_override[key]
            else:
                k = pca_override
            info["pca_components"] = k
            if info.get("pca_mode") == "flat_raw":
                info["pca_total"] = (info["window"] * info["emb_dim"]
                                     if info.get("window") else int(k))
            else:  # flat
                info["pca_total"] = int(k)

    del df_split  # free the original copy

    # -- Summary --
    print(f"\n{'=' * 80}")
    print("COMPONENT SUMMARY")
    print("=" * 80)

    total_csv_features = len(csv_feature_cols)
    total_emb_features = 0

    if component_info["csv_components"]:
        print(f"\n  CSV components ({len(component_info['csv_components'])}):")
        for info in component_info["csv_components"]:
            print(f"    [{info['key']}] {info['display_name']}")
            print(f"      Source: {info['source']}")
            if "n_new_cols" in info:
                print(f"      New columns added: {info['n_new_cols']}")
        print(f"  Total CSV features: {total_csv_features}")

        # Feature breakdown (frac_* has no {pos}_ prefix; counted separately)
        n_comp = sum(
            info.get("n_new_cols", 0)
            for info in component_info["csv_components"]
            if info["key"] == "compositional"
        )
        encoded_cols = [
            c for c in csv_feature_cols
            if any(c.startswith(f"{pos}_") for pos in POSITION_AA_COLS)
        ]
        print(f"    Structural: "
              f"{len(csv_feature_cols) - len(encoded_cols) - n_comp}")
        print(f"    AA-encoded: {len(encoded_cols)}")
        if n_comp:
            print(f"    Compositional: {n_comp}")

    if component_info["emb_components"]:
        print(f"\n  Embedding components ({len(component_info['emb_components'])}):")
        for info in component_info["emb_components"]:
            print(f"    [{info['key']}] {info['display_name']}")
            print(f"      Context dim:    {info.get('raw_dim_desc', info['raw_dim'])}")
            print(f"      After PCA: {info['pca_total']}")
            total_emb_features += info["pca_total"]

    estimated_total = total_csv_features + total_emb_features
    print(f"\n  Estimated total features: ~{estimated_total}")
    print(f"    CSV:        {total_csv_features}")
    print(f"    Embeddings: {total_emb_features} (after PCA)")

    print(f"\n{'=' * 80}")
    print(f"PIPELINE READY: ~{estimated_total} features, "
          f"{N_CV_FOLDS} folds, {display_name}")
    print(f"{'=' * 80}")

    # ── Cross-Validation Loop ──
    feature_names = None
    n_features_final = None

    # ══════════════════════════════════════════════════════════════════
    # NESTED CV: 6 outer iterations × 5 inner folds = 30 models
    # Each outer iteration: 1 bucket = validation, 5 remaining = inner CV
    # Inner models' predictions averaged on validation set.
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("NESTED CROSS-VALIDATION")
    print(f"  Outer iterations: {N_CV_FOLDS + 1} (rotate validation bucket)")
    print(f"  Inner folds:      {N_CV_FOLDS} (per outer iteration)")
    print(f"  Total models:     {(N_CV_FOLDS + 1) * N_CV_FOLDS}")
    print(f"{'=' * 80}")

    outer_val_metrics = []
    outer_val_predictions = {}
    all_outer_inner_weights = []
    all_best_iterations = []
    fold_times = []

    t_nested_start = time.time()

    for outer_fold in range(N_CV_FOLDS + 1):
        print(f"\n{'═' * 80}")
        print(f"  OUTER FOLD {outer_fold}/{N_CV_FOLDS} "
              f"(validation bucket = {outer_fold})")
        print(f"{'═' * 80}")

        inner_models = []
        inner_artifacts = []
        inner_fold_metrics = []
        inner_fold_predictions = {}
        inner_fold_weights = []

        inner_folds = [f for f in range(N_CV_FOLDS + 1) if f != outer_fold]

        for inner_idx, inner_fold in enumerate(inner_folds):
            print(f"\n  ── Inner fold {inner_idx + 1}/{N_CV_FOLDS} "
                  f"(test bucket = {inner_fold}) ──")

            t_fold_start = time.time()

            X_train, y_train, X_test, y_test, fold_info, fold_artifacts = \
                prepare_fold(df, csv_feature_cols, emb_data_dict, model_cfg,
                             inner_fold, validation_bucket=outer_fold)

            if feature_names is None:
                feature_names = fold_info["feature_names"]

            train_ratio = (fold_info["n_train_neg"] / fold_info["n_train_pos"]
                           if fold_info["n_train_pos"] > 0 else float("inf"))
            test_ratio = (fold_info["n_test_neg"] / fold_info["n_test_pos"]
                          if fold_info["n_test_pos"] > 0 else float("inf"))

            print(f"    Train: {fold_info['n_train']:>6} samples "
                  f"(pos={fold_info['n_train_pos']}, "
                  f"neg={fold_info['n_train_neg']}, ratio={train_ratio:.3f})")
            print(f"    Test:  {fold_info['n_test']:>6} samples "
                  f"(pos={fold_info['n_test_pos']}, "
                  f"neg={fold_info['n_test_neg']}, ratio={test_ratio:.3f})")
            print(f"    Features: {fold_info['n_features']}")

            model, test_metrics, y_prob = train_one_fold(
                model_cfg, X_train, y_train, X_test, y_test, inner_fold,
            )
            best_iter = getattr(model, "best_iteration_",
                                getattr(model, "best_iteration", None))
            if best_iter is not None:
                try:
                    all_best_iterations.append(int(best_iter))
                except (TypeError, ValueError):
                    pass

            t_fold_end = time.time()
            fold_times.append(round(t_fold_end - t_fold_start, 1))
            print(f"    Inner fold {inner_fold} AUC: "
                  f"{test_metrics['auc_roc']:.4f} ({t_fold_end - t_fold_start:.1f}s)")

            weights = extract_feature_weights(model, model_cfg)
            if weights is not None:
                inner_fold_weights.append(weights)
                all_outer_inner_weights.append(weights)

            inner_models.append(model)
            inner_artifacts.append(fold_artifacts)
            inner_fold_metrics.append(test_metrics)
            inner_fold_predictions[inner_fold] = {
                "y_true": y_test.tolist(),
                "y_prob": y_prob.tolist(),
            }

            # Save inner model
            inner_model_path = (model_dir /
                                f"{run_tag}_outer{outer_fold}_inner{inner_idx}.pkl")
            with open(inner_model_path, "wb") as f:
                pickle.dump(model, f)

        # ── Average predictions on validation set ──
        print(f"\n  Averaging {len(inner_models)} inner models on "
              f"validation bucket {outer_fold}...")

        y_val, avg_probs = predict_with_averaging(
            df, csv_feature_cols, emb_data_dict, model_cfg,
            inner_models, inner_artifacts, validation_bucket=outer_fold,
        )

        val_metrics = compute_metrics(y_val, avg_probs)
        outer_val_metrics.append(val_metrics)
        outer_val_predictions[outer_fold] = {
            "y_true": y_val.tolist(),
            "y_prob": avg_probs.tolist(),
        }

        n_val_pos = int(y_val.sum())
        n_val_neg = int(len(y_val) - n_val_pos)
        print(f"  Outer fold {outer_fold} validation "
              f"(n={len(y_val)}, pos={n_val_pos}, neg={n_val_neg}):")
        print_metrics(val_metrics, prefix="  ")

    t_nested_end = time.time()

    # ── Nested CV Summary ──
    summary = aggregate_cv_results(outer_val_metrics)
    print(f"\n{'=' * 80}")
    print("NESTED CV RESULTS (mean ± std across 6 outer iterations)")
    print(f"{'=' * 80}")
    print_cv_summary(summary)

    # ── Per-outer-fold comparison table ──
    print(f"\n{'=' * 80}")
    print("PER-OUTER-FOLD VALIDATION METRICS")
    print("=" * 80)
    metric_names = list(outer_val_metrics[0].keys())
    header = f"  {'Outer':<8}" + "".join(f"{m:<13}" for m in metric_names)
    print(header)
    print(f"  {'-' * (8 + 13 * len(metric_names))}")
    for outer_id, metrics in enumerate(outer_val_metrics):
        row = (f"  {outer_id:<8}" +
               "".join(f"{metrics[m]:<13.4f}" for m in metric_names))
        print(row)
    print(f"  {'-' * (8 + 13 * len(metric_names))}")
    mean_row = (f"  {'mean':<8}" +
                "".join(f"{summary[m]['mean']:<13.4f}" for m in metric_names))
    std_row = (f"  {'std':<8}" +
               "".join(f"{summary[m]['std']:<13.4f}" for m in metric_names))
    print(mean_row)
    print(std_row)

    # ── Averaged feature weights across ALL inner models ──
    if all_outer_inner_weights:
        print(f"\n{'=' * 80}")
        is_coef = model_cfg["coef_attr"] == "coef_"
        weight_label = "COEFFICIENTS" if is_coef else "FEATURE IMPORTANCE"
        print(f"AVERAGED {weight_label} ACROSS ALL "
              f"{len(all_outer_inner_weights)} INNER MODELS")
        print("=" * 80)

        avg_weights = np.mean(all_outer_inner_weights, axis=0)
        std_weights = np.std(all_outer_inner_weights, axis=0)
        abs_avg = np.abs(avg_weights)
        sorted_indices = np.argsort(abs_avg)[::-1]

        col_label = "Mean Coef" if is_coef else "Mean Imp"
        print(f"  {'Rank':<6} {'Feature':<45} {col_label:>12} {'Std':>12}")
        print(f"  {'-' * 77}")
        for rank in range(min(30, len(feature_names))):
            idx = sorted_indices[rank]
            print(f"  {rank + 1:<6} {feature_names[idx]:<45} "
                  f"{avg_weights[idx]:>12.6f} {std_weights[idx]:>12.6f}")

    # ── Save Results ──
    n_features_final = feature_names is not None and len(feature_names)
    t_results = time.time()

    results = {
        "timestamp": timestamp,
        "runtime_seconds": round(t_results - t_start, 1),
        "nested_cv_seconds": round(t_nested_end - t_nested_start, 1),
        "fold_seconds": fold_times,
        "mode": "nested_cv",
        "model_key": model_key,
        "model_type": display_name,
        "model_class": model_cfg["model_class"],
        "features_key": features_key,
        "features_display": feat_display,
        "components": feat_cfg["components"],
        "config": {
            "n_outer_folds": N_CV_FOLDS + 1,
            "n_inner_folds": N_CV_FOLDS,
            "total_models": (N_CV_FOLDS + 1) * N_CV_FOLDS,
            "random_state": RANDOM_STATE,
            "needs_scaling": model_cfg["needs_scaling"],
            "pca_mode": pca_mode,
            "pool": pool,
            "pool_mask": pool_mask,
            "hyperparameters": {k: str(v)
                                for k, v in model_cfg["params"].items()},
            "n_features": n_features_final,
            "feature_names": feature_names,
            "component_info": component_info,
        },
        "cv_summary": {m: {"mean": s["mean"], "std": s["std"]}
                       for m, s in summary.items()},
        "outer_val_metrics": outer_val_metrics,
        "outer_val_predictions": outer_val_predictions,
        "best_iterations": all_best_iterations,
    }

    if all_outer_inner_weights:
        results["avg_feature_weights"] = {
            feature_names[i]: float(avg_weights[i])
            for i in sorted_indices[:min(50, len(feature_names))]
        }

    results_path = model_dir / f"cv_results_{run_tag}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    feat_names_path = model_dir / f"{run_tag}_feature_names.json"
    with open(feat_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved: {feat_names_path}")

    # ── Footer ──
    t_end = time.time()
    total_minutes = (t_end - t_start) / 60
    nested_minutes = (t_nested_end - t_nested_start) / 60
    print(f"\n{'=' * 80}")
    print("RUN SUMMARY (NESTED CV)")
    print("=" * 80)
    print(f"  Total runtime:      {t_end - t_start:.1f}s ({total_minutes:.1f} min)")
    print(f"  Nested CV time:     {t_nested_end - t_nested_start:.1f}s "
          f"({nested_minutes:.1f} min)")
    print(f"  Model:              {display_name} ({model_key})")
    print(f"  Feature set:        {feat_display} ({features_key})")
    print(f"  Components:         {feat_cfg['components']}")
    print(f"  Scaling:            {'yes' if model_cfg['needs_scaling'] else 'no'}")
    print(f"  Features (final):   {n_features_final}")
    if csv_feature_cols:
        print(f"    CSV features:     {len(csv_feature_cols)}")
    for info in component_info["emb_components"]:
        print(f"    {info['key']} PCA:       {info['pca_total']}")
    print(f"  Outer iterations:   {N_CV_FOLDS + 1}")
    print(f"  Inner folds:        {N_CV_FOLDS}")
    print(f"  Total models:       {(N_CV_FOLDS + 1) * N_CV_FOLDS}")
    print(f"  Val AUC-ROC:        {summary['auc_roc']['mean']:.4f} ± "
          f"{summary['auc_roc']['std']:.4f}")
    print(f"  Val MCC:            {summary['mcc']['mean']:.4f} ± "
          f"{summary['mcc']['std']:.4f}")
    print(f"  Models dir:         {model_dir}/")
    print(f"  Results file:       {results_path}")
    print(f"  Feature names:      {feat_names_path}")
    print(f"  Log file:           {LOG_PATH}")
    print(f"  Completed:          "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()
