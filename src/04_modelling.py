"""
04_modelling.py
Cross-validation pipeline for MHC-I processing prediction.
Model-agnostic and feature-set-agnostic.

Usage:
    python 04_modelling.py --model rf  --features handcrafted
    python 04_modelling.py --model lr  --features esmc
    python 04_modelling.py --model rf  --features esmif
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

from config import (
    SPLIT_DATA_PATH, LOG_DIR, MODEL_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX,
    PEPTIDE_COL, LABEL_COL, FOLD_COL,
    POSITION_AA_COLS, EMBEDDING_REGIONS,
    RANDOM_STATE, DEFAULT_MODEL, DEFAULT_FEATURE_SET,
    get_feature_cols, get_model_config,
    get_feature_set_config, get_embedding_source,
    validate_config,
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

def build_model(model_cfg):
    """Instantiate a sklearn model from a MODEL_REGISTRY entry."""
    class_path = model_cfg["model_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**model_cfg["params"])


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

    Returns dict with keys:
        regions  -> dict of {region_name: np.array (N, D)}
        labels   -> np.array (N,)
        folds    -> np.array (N,)
        emb_dim  -> int
        n_samples -> int
    """
    emb_source = get_embedding_source(embedding_key)
    prepared_path = emb_source["prepared_path"]

    if not prepared_path.exists():
        raise FileNotFoundError(
            f"Prepared embedding not found: {prepared_path}\n"
            f"Run: python 03_prepare_embeddings.py --embedding {embedding_key}"
        )

    print(f"  Loading prepared embeddings: {prepared_path}")

    data = {"regions": {}}
    with h5py.File(prepared_path, "r") as f:
        data["labels"] = f["labels"][:]
        data["folds"] = f["folds"][:]
        data["emb_dim"] = int(f.attrs.get("emb_dim", 0))
        data["n_samples"] = int(f.attrs.get("n_samples", len(data["labels"])))

        for region in EMBEDDING_REGIONS:
            if region in f:
                data["regions"][region] = f[region][:]

    n_regions = len(data["regions"])
    first_region = list(data["regions"].values())[0]
    emb_dim = first_region.shape[1]
    data["emb_dim"] = emb_dim

    print(f"  Loaded: {data['n_samples']} samples, {n_regions} regions, "
          f"dim={emb_dim}")

    return data


# ──────────────────────────────────────────────
# 4. DATA PREPARATION — HAND-CRAFTED
# ──────────────────────────────────────────────

def impute_nan(X_train, X_test, feature_cols, fold_id):
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            print(f"  WARNING: {name}: {n_nan} NaN values")

    col_medians = np.nanmedian(X_train, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    for col_idx in range(X_train.shape[1]):
        train_nan = np.isnan(X_train[:, col_idx])
        test_nan = np.isnan(X_test[:, col_idx])
        X_train[train_nan, col_idx] = col_medians[col_idx]
        X_test[test_nan, col_idx] = col_medians[col_idx]

    return X_train, X_test, col_medians


def impute_nan_single(X, col_medians):
    X = np.where(np.isinf(X), np.nan, X)
    for col_idx in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col_idx])
        X[nan_mask, col_idx] = col_medians[col_idx]
    return X


def resolve_feature_cols(df):
    feature_cols = get_feature_cols(df.columns)
    non_numeric = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        feature_cols = [c for c in feature_cols if c not in non_numeric]
        print(f"  Excluded {len(non_numeric)} non-numeric feature column(s)")
    return feature_cols


def prepare_fold_csv(df, feature_cols, fold_id, needs_scaling):
    """Prepare fold data from CSV-based hand-crafted features."""
    cv_df = df[df[FOLD_COL] != HELD_OUT_INDEX].copy()
    train_df = cv_df[cv_df[FOLD_COL] != fold_id].copy()
    test_df = cv_df[cv_df[FOLD_COL] == fold_id].copy()

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[LABEL_COL].values.astype(np.int32)
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df[LABEL_COL].values.astype(np.int32)

    X_train, X_test, col_medians = impute_nan(X_train, X_test, feature_cols, fold_id)

    scaler = None
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    fold_info = {
        "fold_id": fold_id,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int(len(y_train) - y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int(len(y_test) - y_test.sum()),
        "n_features": X_train.shape[1],
    }

    fold_artifacts = {
        "col_medians": col_medians,
        "scaler": scaler,
        "pca": None,
    }

    return X_train, y_train, X_test, y_test, fold_info, fold_artifacts


def prepare_held_out_csv(df, feature_cols, fold_artifacts):
    """Prepare held-out data from CSV-based hand-crafted features."""
    ho_df = df[df[FOLD_COL] == HELD_OUT_INDEX].copy()
    X_ho = ho_df[feature_cols].values.astype(np.float64)
    y_ho = ho_df[LABEL_COL].values.astype(np.int32)

    X_ho = impute_nan_single(X_ho, fold_artifacts["col_medians"])
    if fold_artifacts["scaler"] is not None:
        X_ho = fold_artifacts["scaler"].transform(X_ho)

    return X_ho, y_ho


# ──────────────────────────────────────────────
# 5. DATA PREPARATION — EMBEDDINGS
# ──────────────────────────────────────────────

def concatenate_regions(emb_data, indices):
    """
    Concatenate all region embeddings for given sample indices.
    E.g. peptide_emb(1152) + n_flank_emb(1152) + c_flank_emb(1152) = 3456 dims.
    """
    parts = []
    for region_name in sorted(emb_data["regions"].keys()):
        parts.append(emb_data["regions"][region_name][indices])
    return np.concatenate(parts, axis=1).astype(np.float64)


def prepare_fold_embedding(emb_data, fold_id, feat_cfg, needs_scaling):
    """
    Prepare fold data from embedding arrays.
    Optionally applies PCA (fit on train, transform both) and scaling.
    """
    labels = emb_data["labels"]
    folds = emb_data["folds"]

    cv_mask = folds != HELD_OUT_INDEX
    cv_indices = np.where(cv_mask)[0]
    cv_folds = folds[cv_mask]

    train_indices = cv_indices[cv_folds != fold_id]
    test_indices = cv_indices[cv_folds == fold_id]

    X_train = concatenate_regions(emb_data, train_indices)
    y_train = labels[train_indices].astype(np.int32)
    X_test = concatenate_regions(emb_data, test_indices)
    y_test = labels[test_indices].astype(np.int32)

    # Replace any NaN/Inf (should be rare in embeddings but be safe)
    for arr in [X_train, X_test]:
        arr[np.isinf(arr)] = 0.0
        arr[np.isnan(arr)] = 0.0

    # PCA reduction (fit on train only)
    pca = None
    if feat_cfg.get("needs_pca", False):
        n_components = feat_cfg.get("pca_components", 50)
        n_regions = len(emb_data["regions"])
        total_components = n_components * n_regions
        total_components = min(total_components, X_train.shape[1], X_train.shape[0])

        print(f"  Fold {fold_id}: PCA {X_train.shape[1]} → {total_components} "
              f"({n_components} per region × {n_regions} regions)")

        pca = PCA(n_components=total_components, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  Fold {fold_id}: PCA explains {explained:.1f}% of variance")

    # Scaling
    scaler = None
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    fold_info = {
        "fold_id": fold_id,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int(len(y_train) - y_train.sum()),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int(len(y_test) - y_test.sum()),
        "n_features": X_train.shape[1],
    }

    fold_artifacts = {
        "col_medians": None,
        "scaler": scaler,
        "pca": pca,
    }

    return X_train, y_train, X_test, y_test, fold_info, fold_artifacts


def prepare_held_out_embedding(emb_data, fold_artifacts):
    """Prepare held-out data from embeddings using fold-fitted PCA and scaler."""
    folds = emb_data["folds"]
    labels = emb_data["labels"]

    ho_indices = np.where(folds == HELD_OUT_INDEX)[0]
    X_ho = concatenate_regions(emb_data, ho_indices)
    y_ho = labels[ho_indices].astype(np.int32)

    X_ho[np.isinf(X_ho)] = 0.0
    X_ho[np.isnan(X_ho)] = 0.0

    if fold_artifacts["pca"] is not None:
        X_ho = fold_artifacts["pca"].transform(X_ho)
    if fold_artifacts["scaler"] is not None:
        X_ho = fold_artifacts["scaler"].transform(X_ho)

    return X_ho, y_ho


# ──────────────────────────────────────────────
# 6. UNIFIED FOLD PREPARATION
# ──────────────────────────────────────────────

def prepare_fold(df, feature_cols, emb_data, feat_cfg, model_cfg, fold_id):
    """
    Dispatch to CSV or embedding preparation based on feature set config.
    Returns unified (X_train, y_train, X_test, y_test, fold_info, fold_artifacts).
    """
    needs_scaling = model_cfg["needs_scaling"]

    if feat_cfg["source"] == "csv":
        return prepare_fold_csv(df, feature_cols, fold_id, needs_scaling)
    elif feat_cfg["source"] == "embedding":
        return prepare_fold_embedding(emb_data, fold_id, feat_cfg, needs_scaling)
    else:
        raise ValueError(f"Unknown feature source: {feat_cfg['source']}")


def prepare_held_out(df, feature_cols, emb_data, feat_cfg, fold_artifacts):
    """Dispatch to CSV or embedding held-out preparation."""
    if feat_cfg["source"] == "csv":
        return prepare_held_out_csv(df, feature_cols, fold_artifacts)
    elif feat_cfg["source"] == "embedding":
        return prepare_held_out_embedding(emb_data, fold_artifacts)
    else:
        raise ValueError(f"Unknown feature source: {feat_cfg['source']}")


# ──────────────────────────────────────────────
# 7. MODEL TRAINING
# ──────────────────────────────────────────────

def train_one_fold(model_cfg, X_train, y_train, X_test, y_test, fold_id):
    model = build_model(model_cfg)
    display = model_cfg["display_name"]

    print(f"  Training {display} ...")
    t_start = time.time()
    model.fit(X_train, y_train)
    t_end = time.time()
    print(f"  Training time: {t_end - t_start:.1f}s")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    test_metrics = compute_metrics(y_test, y_prob)
    return model, test_metrics, y_prob


# ──────────────────────────────────────────────
# 8. FEATURE WEIGHTS
# ──────────────────────────────────────────────

def extract_feature_weights(model, model_cfg):
    attr = model_cfg.get("coef_attr")
    if attr is None or not hasattr(model, attr):
        return None
    raw = getattr(model, attr)
    return np.asarray(raw).ravel()


def make_feature_names(feat_cfg, emb_data, feature_cols, n_features):
    """
    Generate human-readable feature names.
    For CSV: use actual column names.
    For embeddings with PCA: generate PC_001, PC_002, ...
    """
    if feat_cfg["source"] == "csv":
        return list(feature_cols)

    if feat_cfg.get("needs_pca", False):
        return [f"PC_{i + 1:03d}" for i in range(n_features)]

    # Raw embeddings without PCA
    names = []
    for region in sorted(emb_data["regions"].keys()):
        dim = emb_data["regions"][region].shape[1]
        short = region.replace("_emb", "")
        names.extend([f"{short}_d{i}" for i in range(dim)])
    return names[:n_features]


def print_feature_weights(weights, feature_names, model_cfg, top_n=20):
    if weights is None:
        print("\n  (model does not expose feature weights)")
        return weights, None

    abs_weights = np.abs(weights)
    indices = np.argsort(abs_weights)[::-1]

    is_coef = model_cfg["coef_attr"] == "coef_"
    col_label = "Coefficient" if is_coef else "Importance"

    print(f"\n  Top {min(top_n, len(feature_names))} feature {col_label.lower()}s:")
    print(f"  {'Rank':<6} {'Feature':<40} {col_label:>12}")
    print(f"  {'-' * 60}")
    for rank in range(min(top_n, len(feature_names))):
        idx = indices[rank]
        line = f"  {rank + 1:<6} {feature_names[idx]:<40} {weights[idx]:>12.6f}"
        print(line)

    return weights, indices


# ──────────────────────────────────────────────
# 9. AGGREGATE CV RESULTS
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
# 10. CLI
# ──────────────────────────────────────────────

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
    return parser.parse_args()


# ──────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    model_key = args.model
    features_key = args.features

    validate_config()
    model_cfg = get_model_config(model_key)
    feat_cfg = get_feature_set_config(features_key)
    display_name = model_cfg["display_name"]
    feat_display = feat_cfg["display_name"]

    # Run tag used for all output filenames
    run_tag = f"{model_key}_{features_key}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"04_modelling_{run_tag}_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
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
    print(f"  Feature source:  {feat_cfg['source']}")
    print(f"  Data path:       {SPLIT_DATA_PATH}")
    print(f"  Log path:        {LOG_PATH}")
    print(f"  CV folds (k):    {N_CV_FOLDS}")
    print(f"  Held-out bucket: {HELD_OUT_INDEX}")
    print(f"  Random state:    {RANDOM_STATE}")
    print(f"  Needs scaling:   {model_cfg['needs_scaling']}")
    if feat_cfg.get("needs_pca"):
        print(f"  PCA components:  {feat_cfg.get('pca_components', '?')} per region")
    print(f"\n  Model hyperparameters:")
    for k, v in model_cfg["params"].items():
        print(f"    {k:<25} {v}")
    print("=" * 80)

    # -- Load split data (always needed for labels/folds in CSV mode,
    #    and for metadata in embedding mode) --
    print(f"\nLoading split data: {SPLIT_DATA_PATH}")
    df = pd.read_csv(SPLIT_DATA_PATH)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Fold validation
    if FOLD_COL not in df.columns:
        print(f"FATAL: Column '{FOLD_COL}' not found. Run 01_datasplit.py first.")
        logger.close()
        sys.exit(1)

    actual_folds = sorted(df[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))
    if actual_folds != expected_folds:
        print(f"FATAL: Expected folds {expected_folds}, found {actual_folds}")
        logger.close()
        sys.exit(1)
    print(f"  [OK] Fold column validated: {actual_folds}")

    # -- Load feature-set-specific data --
    feature_cols = []
    emb_data = None

    if feat_cfg["source"] == "csv":
        feature_cols = resolve_feature_cols(df)
        n_features_initial = len(feature_cols)
        if n_features_initial == 0:
            print("FATAL: No numeric feature columns found.")
            logger.close()
            sys.exit(1)
        print(f"  Using {n_features_initial} hand-crafted feature columns")

    elif feat_cfg["source"] == "embedding":
        embedding_key = feat_cfg["embedding_key"]
        emb_data = load_embedding_data(embedding_key)

        n_regions = len(emb_data["regions"])
        emb_dim = emb_data["emb_dim"]
        raw_dim = n_regions * emb_dim

        if feat_cfg.get("needs_pca"):
            pca_per = feat_cfg.get("pca_components", 50)
            n_features_initial = pca_per * n_regions
            print(f"  Raw embedding dim: {raw_dim} "
                  f"({n_regions} regions × {emb_dim})")
            print(f"  After PCA: {n_features_initial} "
                  f"({pca_per} per region × {n_regions} regions)")
        else:
            n_features_initial = raw_dim
            print(f"  Embedding dim: {raw_dim} "
                  f"({n_regions} regions × {emb_dim})")

    print(f"\n{'=' * 80}")
    print(f"PIPELINE READY: ~{n_features_initial} features, "
          f"{N_CV_FOLDS} folds, {display_name} + {feat_display}")
    print(f"{'=' * 80}")

    # ── Cross-Validation Loop ──
    all_fold_metrics = []
    all_fold_predictions = {}
    all_fold_weights = []
    best_fold_auc = -1
    best_fold_id = -1
    best_fold_model = None
    best_fold_artifacts = None
    feature_names = None

    for fold_id in range(N_CV_FOLDS):
        print(f"\n{'-' * 80}")
        print(f"  FOLD {fold_id}/{N_CV_FOLDS - 1}")
        print(f"{'-' * 80}")

        t_fold_start = time.time()

        X_train, y_train, X_test, y_test, fold_info, fold_artifacts = prepare_fold(
            df, feature_cols, emb_data, feat_cfg, model_cfg, fold_id,
        )

        # Build feature names on first fold (PCA dims are known now)
        if feature_names is None:
            feature_names = make_feature_names(
                feat_cfg, emb_data, feature_cols, fold_info["n_features"],
            )

        train_ratio = fold_info["n_train_neg"] / fold_info["n_train_pos"] if fold_info["n_train_pos"] > 0 else float("inf")
        test_ratio = fold_info["n_test_neg"] / fold_info["n_test_pos"] if fold_info["n_test_pos"] > 0 else float("inf")

        print(f"  Train: {fold_info['n_train']:>6} samples "
              f"(pos={fold_info['n_train_pos']}, neg={fold_info['n_train_neg']}, "
              f"ratio={train_ratio:.3f})")
        print(f"  Test:  {fold_info['n_test']:>6} samples "
              f"(pos={fold_info['n_test_pos']}, neg={fold_info['n_test_neg']}, "
              f"ratio={test_ratio:.3f})")
        print(f"  Features: {fold_info['n_features']}")

        # Train
        model, test_metrics, y_prob = train_one_fold(
            model_cfg, X_train, y_train, X_test, y_test, fold_id,
        )

        t_fold_end = time.time()

        print(f"\n  Fold {fold_id} results ({t_fold_end - t_fold_start:.1f}s):")
        print_metrics(test_metrics, prefix="  ")

        # Feature weights
        weights = extract_feature_weights(model, model_cfg)
        print_feature_weights(weights, feature_names, model_cfg, top_n=15)
        if weights is not None:
            all_fold_weights.append(weights)

        all_fold_predictions[fold_id] = {
            "y_true": y_test.tolist(),
            "y_prob": y_prob.tolist(),
        }

        all_fold_metrics.append(test_metrics)

        if test_metrics["auc_roc"] > best_fold_auc:
            best_fold_auc = test_metrics["auc_roc"]
            best_fold_id = fold_id
            best_fold_model = model
            best_fold_artifacts = fold_artifacts

        fold_model_path = MODEL_DIR / f"{run_tag}_model_fold{fold_id}.pkl"
        with open(fold_model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model: {fold_model_path}")

    # ── CV Summary ──
    summary = aggregate_cv_results(all_fold_metrics)
    print_cv_summary(summary)
    print(f"\n  Best fold: {best_fold_id} (AUC-ROC = {best_fold_auc:.4f})")

    # ── Per-fold comparison table ──
    print(f"\n{'=' * 80}")
    print("PER-FOLD METRIC COMPARISON")
    print("=" * 80)
    metric_names = list(all_fold_metrics[0].keys())
    header = f"  {'Fold':<8}" + "".join(f"{m:<13}" for m in metric_names)
    print(header)
    print(f"  {'-' * (8 + 13 * len(metric_names))}")
    for fold_id, metrics in enumerate(all_fold_metrics):
        row = f"  {fold_id:<8}" + "".join(f"{metrics[m]:<13.4f}" for m in metric_names)
        print(row)
    print(f"  {'-' * (8 + 13 * len(metric_names))}")
    mean_row = f"  {'mean':<8}" + "".join(f"{summary[m]['mean']:<13.4f}" for m in metric_names)
    std_row = f"  {'std':<8}" + "".join(f"{summary[m]['std']:<13.4f}" for m in metric_names)
    print(mean_row)
    print(std_row)

    # ── Averaged feature weights across folds ──
    if all_fold_weights:
        print(f"\n{'=' * 80}")
        is_coef = model_cfg["coef_attr"] == "coef_"
        weight_label = "COEFFICIENTS" if is_coef else "FEATURE IMPORTANCE"
        print(f"AVERAGED {weight_label} ACROSS ALL FOLDS")
        print("=" * 80)

        avg_weights = np.mean(all_fold_weights, axis=0)
        std_weights = np.std(all_fold_weights, axis=0)
        abs_avg = np.abs(avg_weights)
        sorted_indices = np.argsort(abs_avg)[::-1]

        col_label = "Mean Coef" if is_coef else "Mean Imp"
        print(f"  {'Rank':<6} {'Feature':<40} {col_label:>12} {'Std':>12}")
        print(f"  {'-' * 72}")
        for rank in range(min(30, len(feature_names))):
            idx = sorted_indices[rank]
            print(f"  {rank + 1:<6} {feature_names[idx]:<40} "
                  f"{avg_weights[idx]:>12.6f} {std_weights[idx]:>12.6f}")

    # ── Held-Out Evaluation ──
    print(f"\n{'=' * 80}")
    print("HELD-OUT VALIDATION SET EVALUATION")
    print(f"{'=' * 80}")
    print(f"  Using model from best fold ({best_fold_id})")

    X_ho, y_ho = prepare_held_out(
        df, feature_cols, emb_data, feat_cfg, best_fold_artifacts,
    )

    n_ho_pos = int(y_ho.sum())
    n_ho_neg = int(len(y_ho) - y_ho.sum())
    ho_ratio = n_ho_neg / n_ho_pos if n_ho_pos > 0 else float("inf")
    print(f"  Held-out samples: {len(y_ho)} "
          f"(pos={n_ho_pos}, neg={n_ho_neg}, ratio={ho_ratio:.3f})")

    if hasattr(best_fold_model, "predict_proba"):
        ho_probs = best_fold_model.predict_proba(X_ho)[:, 1]
    else:
        ho_probs = best_fold_model.decision_function(X_ho)

    ho_metrics = compute_metrics(y_ho, ho_probs)

    print(f"\n  Held-out results ({len(y_ho)} samples):")
    print_metrics(ho_metrics, prefix="  ")

    ho_preds = (ho_probs >= 0.5).astype(int)
    print(f"\n  Classification Report:")
    print(classification_report(y_ho, ho_preds,
                                target_names=["Negative", "Positive"], digits=4))

    # ── CV vs Held-out comparison ──
    print(f"{'=' * 80}")
    print("CV AVERAGE vs HELD-OUT COMPARISON")
    print("=" * 80)
    print(f"  {'Metric':<15} {'CV mean':>10} {'CV std':>10} {'Held-out':>10} {'Delta':>10}")
    print(f"  {'-' * 57}")
    for m in metric_names:
        cv_mean = summary[m]["mean"]
        cv_std = summary[m]["std"]
        ho_val = ho_metrics[m]
        delta = ho_val - cv_mean
        print(f"  {m:<15} {cv_mean:>10.4f} {cv_std:>10.4f} {ho_val:>10.4f} {delta:>+10.4f}")

    # ── Save All Results ──
    n_features_final = fold_info["n_features"]

    results = {
        "timestamp": timestamp,
        "model_key": model_key,
        "model_type": display_name,
        "model_class": model_cfg["model_class"],
        "features_key": features_key,
        "features_display": feat_display,
        "features_source": feat_cfg["source"],
        "config": {
            "n_cv_folds": N_CV_FOLDS,
            "held_out_index": HELD_OUT_INDEX,
            "random_state": RANDOM_STATE,
            "needs_scaling": model_cfg["needs_scaling"],
            "needs_pca": feat_cfg.get("needs_pca", False),
            "pca_components_per_region": feat_cfg.get("pca_components"),
            "hyperparameters": {k: str(v) for k, v in model_cfg["params"].items()},
            "n_features": n_features_final,
            "feature_names": feature_names,
        },
        "cv_summary": {m: {"mean": s["mean"], "std": s["std"]}
                       for m, s in summary.items()},
        "fold_metrics": all_fold_metrics,
        "held_out_metrics": ho_metrics,
        "best_fold_id": best_fold_id,
        "best_fold_auc": best_fold_auc,
        "fold_predictions": all_fold_predictions,
    }

    if all_fold_weights:
        results["avg_feature_weights"] = {
            feature_names[i]: float(avg_weights[i])
            for i in sorted_indices[:min(50, len(feature_names))]
        }

    results_path = MODEL_DIR / f"cv_results_{run_tag}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Save best model
    best_model_path = MODEL_DIR / f"best_{run_tag}_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_fold_model, f)
    print(f"Best model saved: {best_model_path}")

    # Save fold artifacts (medians, scaler, PCA)
    artifacts_path = MODEL_DIR / f"best_{run_tag}_artifacts.pkl"
    with open(artifacts_path, "wb") as f:
        pickle.dump(best_fold_artifacts, f)
    print(f"Fold artifacts saved: {artifacts_path}")

    # Save feature names
    feat_names_path = MODEL_DIR / f"{run_tag}_feature_names.json"
    with open(feat_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved: {feat_names_path}")

    # ── Footer ──
    t_end = time.time()
    total_minutes = (t_end - t_start) / 60
    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"  Total runtime:    {t_end - t_start:.1f}s ({total_minutes:.1f} min)")
    print(f"  Model:            {display_name} ({model_key})")
    print(f"  Features:         {feat_display} ({features_key})")
    print(f"  Feature source:   {feat_cfg['source']}")
    print(f"  Scaling:          {'yes' if model_cfg['needs_scaling'] else 'no'}")
    print(f"  PCA:              {'yes (' + str(feat_cfg.get('pca_components', '?')) + '/region)' if feat_cfg.get('needs_pca') else 'no'}")
    print(f"  Features (final): {n_features_final}")
    print(f"  CV folds:         {N_CV_FOLDS}")
    print(f"  Best fold:        {best_fold_id} (AUC = {best_fold_auc:.4f})")
    print(f"  Held-out AUC:     {ho_metrics['auc_roc']:.4f}")
    print(f"  Held-out MCC:     {ho_metrics['mcc']:.4f}")
    print(f"  Models dir:       {MODEL_DIR}/")
    print(f"  Results file:     {results_path}")
    print(f"  Feature names:    {feat_names_path}")
    print(f"  Log file:         {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    logger.close()