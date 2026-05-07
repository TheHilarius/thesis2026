"""
04_modelling.py
Cross-validation pipeline for MHC-I processing prediction.
Model-agnostic with composable feature sets.

Usage:
    python 04_modelling.py --model rf  --features handcrafted
    python 04_modelling.py --model lr  --features handcrafted_sparse
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

    print(f"    Loading prepared embeddings: {prepared_path}")

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

    print(f"    Loaded: {data['n_samples']} samples, {n_regions} regions, "
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
        elif comp["type"] == "embedding":
            emb_components.append(comp)
        else:
            raise ValueError(f"Unknown component type: {comp['type']}")

    return csv_components, emb_components


def load_all_components(df_split, feat_cfg):
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
            if np.issubdtype(df[c].dtype, np.number)
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
        emb_data_dict[comp_key] = emb_data

        n_regions = len(emb_data["regions"])
        emb_dim = emb_data["emb_dim"]
        component_info["emb_components"].append({
            "key": comp_key,
            "display_name": comp["display_name"],
            "embedding_key": emb_key,
            "emb_dim": emb_dim,
            "n_regions": n_regions,
            "raw_dim": n_regions * emb_dim,
            "pca_components": pca_components,
            "pca_total": pca_components * n_regions,
        })

    return df, csv_feature_cols, emb_data_dict, component_info


# ──────────────────────────────────────────────
# 5. DATA PREPARATION — CSV PART
# ──────────────────────────────────────────────

def impute_nan(X_train, X_test, feature_cols, fold_id):
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_test = np.where(np.isinf(X_test), np.nan, X_test)

    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            print(f"    WARNING: {name}: {n_nan} NaN values replaced with medians")

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


# ──────────────────────────────────────────────
# 6. DATA PREPARATION — EMBEDDING PART
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


# ──────────────────────────────────────────────
# 7. UNIFIED FOLD PREPARATION (composable)
# ──────────────────────────────────────────────

def prepare_fold(df, csv_feature_cols, emb_data_dict, model_cfg, fold_id):
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
    cv_mask = df[FOLD_COL] != HELD_OUT_INDEX
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
        n_regions = len(emb_data["regions"])

        X_emb_train = concatenate_regions(emb_data, train_indices)
        X_emb_test = concatenate_regions(emb_data, test_indices)

        # Clean inf values
        for arr in [X_emb_train, X_emb_test]:
            arr[np.isinf(arr)] = 0.0
            arr[np.isnan(arr)] = 0.0

        # ── Zero-vector handling ──
        # Identify samples where ALL embedding dims are zero
        # (unresolved structures produce all-zero rows)
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

        # Fit PCA only on non-zero training samples
        X_emb_train_valid = X_emb_train[train_nonzero_mask]

        total_components = pca_components * n_regions
        print(f"The total number of PCA components requested for this embedding component is {pca_components} per region × {n_regions} regions = {total_components} total. However, the number of non-zero training samples is only {X_emb_train_valid.shape[0]}, which limits the maximum number of PCA components that can be fitted. Therefore, we will use {total_components} PCA components for this component.")

        total_components = min(
            total_components,
            X_emb_train_valid.shape[1],
            X_emb_train_valid.shape[0],
        )
        
        pca = PCA(n_components=total_components, random_state=RANDOM_STATE)
        pca.fit(X_emb_train_valid)

        # Transform ALL samples (non-zero get real projections,
        # zero vectors project to origin in PC space)
        X_emb_train = pca.transform(X_emb_train)
        X_emb_test = pca.transform(X_emb_test)

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"    {comp_key}: PCA {emb_data['emb_dim'] * n_regions} → {total_components} "
              f"({pca_components}/region × {n_regions} regions, "
              f"{explained:.1f}% variance, fitted on {train_nonzero_mask.sum()} non-zero samples)")

        fold_artifacts["pca_dict"][comp_key] = pca

        parts_train.append(X_emb_train)
        parts_test.append(X_emb_test)
        feature_names.extend([f"{comp_key}_PC{i + 1:03d}" for i in range(total_components)])

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
        X_csv = impute_nan_single(X_csv, fold_artifacts["col_medians"])
        parts.append(X_csv)

    # ── Embedding features ──
    for comp_key, emb_data in emb_data_dict.items():
        X_emb = concatenate_regions(emb_data, ho_indices)
        X_emb[np.isinf(X_emb)] = 0.0
        X_emb[np.isnan(X_emb)] = 0.0

        pca = fold_artifacts["pca_dict"][comp_key]
        X_emb = pca.transform(X_emb)
        parts.append(X_emb)

    X_ho = np.concatenate(parts, axis=1)

    if fold_artifacts["scaler"] is not None:
        X_ho = fold_artifacts["scaler"].transform(X_ho)

    return X_ho, y_ho


# ──────────────────────────────────────────────
# 8. MODEL TRAINING
# ──────────────────────────────────────────────

def train_one_fold(model_cfg, X_train, y_train, X_test, y_test, fold_id):
    model = build_model(model_cfg)
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
        "--pca", type=int, default=None,
        help="Override PCA components per region for ALL embedding components "
             "(default: use config.py value)",
    )
    return parser.parse_args()

# ──────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    model_key = args.model
    features_key = args.features

    validate_config()
    model_cfg = get_model_config(model_key)
    feat_cfg = get_feature_set_config(features_key)
    validate_feature_set(feat_cfg)

    # ── PCA override ──
    pca_override = args.pca
    if pca_override is not None:
        print(f"  [CLI OVERRIDE] PCA components per region: {pca_override}")

    display_name = model_cfg["display_name"]
    feat_display = feat_cfg["display_name"]

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
        df_split, feat_cfg,
    )

    # Apply PCA override if specified
    if pca_override is not None:
        for comp_key, emb_data in emb_data_dict.items():
            emb_data["pca_components"] = pca_override
        # Update component_info for logging
        for info in component_info["emb_components"]:
            info["pca_components"] = pca_override
            info["pca_total"] = pca_override * info["n_regions"]
            
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
            print(f"      Raw dim: {info['raw_dim']} "
                  f"({info['n_regions']} regions × {info['emb_dim']})")
            print(f"      After PCA: {info['pca_total']} "
                  f"({info['pca_components']}/region × {info['n_regions']} regions)")
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
            df, csv_feature_cols, emb_data_dict, model_cfg, fold_id,
        )

        if feature_names is None:
            feature_names = fold_info["feature_names"]

        train_ratio = fold_info["n_train_neg"] / fold_info["n_train_pos"] if fold_info["n_train_pos"] > 0 else float("inf")
        test_ratio = fold_info["n_test_neg"] / fold_info["n_test_pos"] if fold_info["n_test_pos"] > 0 else float("inf")

        print(f"    Train: {fold_info['n_train']:>6} samples "
              f"(pos={fold_info['n_train_pos']}, neg={fold_info['n_train_neg']}, "
              f"ratio={train_ratio:.3f})")
        print(f"    Test:  {fold_info['n_test']:>6} samples "
              f"(pos={fold_info['n_test_pos']}, neg={fold_info['n_test_neg']}, "
              f"ratio={test_ratio:.3f})")
        print(f"    Features: {fold_info['n_features']}")

        model, test_metrics, y_prob = train_one_fold(
            model_cfg, X_train, y_train, X_test, y_test, fold_id,
        )

        t_fold_end = time.time()

        print(f"\n    Fold {fold_id} results ({t_fold_end - t_fold_start:.1f}s):")
        print_metrics(test_metrics, prefix="    ")

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
        print(f"    Saved model: {fold_model_path}")

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

    # ── Averaged feature weights ──
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
        print(f"  {'Rank':<6} {'Feature':<45} {col_label:>12} {'Std':>12}")
        print(f"  {'-' * 77}")
        for rank in range(min(30, len(feature_names))):
            idx = sorted_indices[rank]
            print(f"  {rank + 1:<6} {feature_names[idx]:<45} "
                  f"{avg_weights[idx]:>12.6f} {std_weights[idx]:>12.6f}")

    # ── Held-Out Evaluation ──
    print(f"\n{'=' * 80}")
    print("HELD-OUT VALIDATION SET EVALUATION")
    print(f"{'=' * 80}")
    print(f"  Using model from best fold ({best_fold_id})")

    X_ho, y_ho = prepare_held_out(
        df, csv_feature_cols, emb_data_dict, model_cfg, best_fold_artifacts,
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
        "components": feat_cfg["components"],
        "config": {
            "n_cv_folds": N_CV_FOLDS,
            "held_out_index": HELD_OUT_INDEX,
            "random_state": RANDOM_STATE,
            "needs_scaling": model_cfg["needs_scaling"],
            "hyperparameters": {k: str(v) for k, v in model_cfg["params"].items()},
            "n_features": n_features_final,
            "feature_names": feature_names,
            "component_info": component_info,
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

    best_model_path = MODEL_DIR / f"best_{run_tag}_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_fold_model, f)
    print(f"Best model saved: {best_model_path}")

    artifacts_path = MODEL_DIR / f"best_{run_tag}_artifacts.pkl"
    with open(artifacts_path, "wb") as f:
        pickle.dump(best_fold_artifacts, f)
    print(f"Fold artifacts saved: {artifacts_path}")

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
    print(f"  Feature set:      {feat_display} ({features_key})")
    print(f"  Components:       {feat_cfg['components']}")
    print(f"  Scaling:          {'yes' if model_cfg['needs_scaling'] else 'no'}")
    print(f"  Features (final): {n_features_final}")
    if csv_feature_cols:
        print(f"    CSV features:   {len(csv_feature_cols)}")
    for info in component_info["emb_components"]:
        print(f"    {info['key']} PCA:     {info['pca_total']}")
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
