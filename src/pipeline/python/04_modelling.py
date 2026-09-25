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
from sklearn.decomposition import PCA, IncrementalPCA
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


def load_all_components(df_split, feat_cfg, pca_mode="slot"):
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

    for comp in csv_components:
        comp_key = comp["_key"]
        csv_path = comp.get("csv_path")

        if csv_path is not None:
            print(f"\n  Loading CSV component: {comp['display_name']}")
            new_cols_df = load_alternate_csv(csv_path, df)
            new_col_frames.append(new_cols_df)

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

    # Resolve CSV feature columns (everything numeric, not in exclude set)
    csv_feature_cols = []
    if csv_components:
        all_feature_cols = get_feature_cols(df.columns)
        csv_feature_cols = [
            c for c in all_feature_cols
            if pd.api.types.is_numeric_dtype(df[c])
            #if np.issubdtype(df[c].dtype, np.number)
        ]
        non_numeric_dropped = len(all_feature_cols) - len(csv_feature_cols)
        if non_numeric_dropped > 0:
            print(f"  Excluded {non_numeric_dropped} non-numeric feature column(s)")

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
        emb_data_dict[comp_key] = emb_data

        emb_dim = emb_data["emb_dim"]
        if emb_data.get("kind") == "windows":
            window = emb_data["window"]
            if pca_mode == "flat":
                pca_total = int(pca_components)
                raw_dim_desc = f"{window}x{emb_dim} (flat {window * emb_dim})"
            else:
                pca_total = window * int(min(pca_components, emb_dim))
                raw_dim_desc = f"{window}x{emb_dim}"
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


def fit_window_pca(emb_data, indices, k):
    """
    Stream real (non-pad, non-zero-norm) training slots and fit an
    IncrementalPCA.  Returns the fitted estimator (k components).
    """
    prepared_path = emb_data["prepared_path"]
    D = emb_data["emb_dim"]
    k = int(min(k, D))
    ipca = IncrementalPCA(n_components=k, batch_size=32768)

    n_slots = 0
    n_kept = 0
    for flat, flatm in _stream_window_slots(prepared_path, indices):
        n_slots += flat.shape[0]
        real = ~flatm
        block = flat[real].astype(np.float64)
        norms = np.linalg.norm(block, axis=1)
        block = block[norms > 0.0]
        if block.shape[0] > 0:
            ipca.partial_fit(block)
            n_kept += block.shape[0]

    print(f"      window PCA fit: {n_slots} slots streamed, "
          f"{n_kept} real non-zero slots kept (k={k})")
    return ipca


def transform_window_block(emb_data, indices, ipca, k):
    """
    Stream window rows for `indices`, project each slot through the shared
    PCA, and flatten to (n, window*k).  Padded slots keep their raw token
    values (zero-pad/pad-token are ~0; impute modes carry meaningful imputed
    vectors).  Returns (n, window*k) float64.
    """
    prepared_path = emb_data["prepared_path"]
    window = emb_data["window"]
    n = len(indices)
    out = np.empty((n, window * k), dtype=np.float64)
    cur = 0
    with h5py.File(prepared_path, "r") as f:
        W = f["windows"]
        for i0 in range(0, n, WINDOW_CHUNK_ROWS):
            idx = indices[i0:i0 + WINDOW_CHUNK_ROWS]
            block = W[idx].astype(np.float64)   # (c, window, D)
            c = block.shape[0]
            proj = ipca.transform(block.reshape(c * window, -1))
            flat = proj.reshape(c, window * k)
            out[cur:cur + c] = flat
            cur += c
    return out


def window_feature_names(comp_key, window, k):
    """Slot-indexed PC names (order matches the block)."""
    return [f"{comp_key}_S{ss:02d}_PC{i + 1:03d}"
            for ss in range(window) for i in range(k)]


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

        # ── Window-embedding path (slot PCA or flattened-window PCA) ──
        if emb_data.get("kind") == "windows":
            window = emb_data["window"]
            D = emb_data["emb_dim"]
            pca_mode = emb_data.get("pca_mode", "slot")

            if pca_mode == "flat":
                total_components = int(min(pca_components, window * D))
            else:
                total_components = int(min(pca_components, D))

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
            else:
                print(f"    {comp_key}: fitting shared slot PCA "
                      f"(window={window}, D={D}, k={total_components}) ...")
                ipca = fit_window_pca(emb_data, train_indices, total_components)
                X_emb_train = transform_window_block(
                    emb_data, train_indices, ipca, total_components,
                )
                X_emb_test = transform_window_block(
                    emb_data, test_indices, ipca, total_components,
                )

            explained = (ipca.explained_variance_ratio_.sum() * 100
                         if hasattr(ipca, "explained_variance_ratio_") else np.nan)
            if pca_mode == "flat":
                print(f"    {comp_key}: window PCA {window * D} → k={total_components} "
                      f"({total_components} features) "
                      f"({explained:.1f}% variance)")
                feature_names.extend(
                    window_feature_names_flat(comp_key, total_components)
                )
            else:
                print(f"    {comp_key}: window PCA D={D} → k={total_components} "
                      f"(slot features {window * total_components}) "
                      f"({explained:.1f}% variance)")
                feature_names.extend(
                    window_feature_names(comp_key, window, total_components)
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
        X_emb = _transform_embeddings(emb_data, ho_indices, pca_obj)
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
        X_emb = _transform_embeddings(emb_data, val_indices, pca_obj)
        parts.append(X_emb)

    X_val = np.concatenate(parts, axis=1)

    if fold_artifacts["scaler"] is not None:
        X_val = fold_artifacts["scaler"].transform(X_val)

    return X_val, y_val, val_indices


def _transform_embeddings(emb_data, indices, pca_obj):
    """
    Transform the single context embedding through its fitted PCA
    (legacy), or stream window slots through the shared PCA (windows).
    """
    if emb_data.get("kind") == "windows":
        k = int(getattr(pca_obj, "n_components_", pca_obj.n_components))
        if emb_data.get("pca_mode", "slot") == "flat":
            return transform_window_block_flat(emb_data, indices, pca_obj, k)
        return transform_window_block(emb_data, indices, pca_obj, k)
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
        "--pca-mode", type=str, default="slot",
        choices=["slot", "flat"],
        help="Window PCA scheme: 'slot' (per-position shared PCA, default) or "
             "'flat' (flatten window*D, PCA across the full window)",
    )
    parser.add_argument(
        "--C", type=float, default=None,
        help="Override inverse regularization strength C (default: config.py value, "
             "smaller = stronger reg)",
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
    if pca_mode == "flat":
        run_tag += "_flat"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Namespace all run outputs by PCA experiment so slot/flat batches
    # never mix (logs, cv_results, pickles). Baselines run with the
    # default pca_mode=slot, so they live under slot/ as well.
    log_dir = LOG_DIR / pca_mode
    model_dir = MODEL_DIR / pca_mode
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
        df_split, feat_cfg, pca_mode,
    )

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
            if info.get("pca_mode") == "flat":
                info["pca_total"] = int(k)
            else:
                info["pca_total"] = (info["window"] * int(min(k, info["emb_dim"]))
                                     if info.get("window") else k)

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

        # Feature breakdown
        structural_cols = [
            c for c in csv_feature_cols
            if not any(c.startswith(f"{pos}_") for pos in POSITION_AA_COLS)
        ]
        encoded_cols = [c for c in csv_feature_cols if c not in structural_cols]
        print(f"    Structural: {len(structural_cols)}")
        print(f"    AA-encoded: {len(encoded_cols)}")

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

            t_fold_end = time.time()
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

    results = {
        "timestamp": timestamp,
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
