#!/usr/bin/env python3
"""
04b_pca_optimization_cuml.py
Joint PCA component optimization with cuML GPU acceleration.

Evaluates ALL (peptide × n_flank × c_flank) combinations jointly across
6 outer folds. Final PCA is selected by pooling mean inner-CV AUC across
all 6 outer folds for each coarse combination.

Procedure:
  1. Use fixed coarse grid (default: 5,10,15,20,25) → 125 combinations
  2. For each outer fold: hold out 1 bin as validation, inner CV on 5 bins
  3. Evaluate all 125 combos using inner CV, select best per fold
  4. Evaluate selected combo once on untouched outer validation
  5. After all 6 folds: pool inner-CV scores, select best complete tuple
  6. Write best tuple to pca_optimal_settings.json

Final PCA selection uses pooled inner-CV scores across all 6 outer folds.
Outer validation AUCs are reported separately as evaluation of the procedure.

Usage:
    # Standard coarse-only optimization (recommended)
    python3 src/pipeline/python/04b_pca_optimization_cuml.py \
        --features handcrafted_sparse_esmc \
        --joint-pca --use-gpu \
        --grid 5,10,15,20,25 \
        --outer-folds 6 --inner-folds 5 --parallel 6

    # Both feature sets
    python3 src/pipeline/python/04b_pca_optimization_cuml.py \
        --features handcrafted_sparse_esmc handcrafted_sparse_esmif \
        --joint-pca --use-gpu \
        --grid 5,10,15,20,25 \
        --outer-folds 6 --inner-folds 5 --parallel 6
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import json
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR, SPLIT_DATA_PATH, LOG_DIR, MODEL_DIR, FIGURES_DIR,
    N_CV_FOLDS, HELD_OUT_INDEX, RANDOM_STATE,
    PEPTIDE_COL, LABEL_COL, FOLD_COL,
    get_feature_cols, get_feature_set_config, validate_config,
    validate_feature_set,
)

# ──────────────────────────────────────────────
# 0. LOGGER
# ──────────────────────────────────────────────

class Logger:
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
# 1. CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="04b_pca_optimization: joint PCA optimization for embeddings",
    )
    parser.add_argument(
        "--features", nargs="+", required=True,
        help="Feature set key(s) from FEATURE_SETS (e.g. handcrafted_sparse_esmc)",
    )
    parser.add_argument(
        "--joint-pca", action="store_true",
        help="Use joint PCA optimization (combinations of all regions)",
    )
    parser.add_argument(
        "--grid", type=str, default="5,10,15,20,25",
        help="Comma-separated PCA values to search (default: 5,10,15,20,25)",
    )
    parser.add_argument(
        "--outer-folds", type=int, default=6,
        help="Number of outer folds for evaluation (default: 6)",
    )
    parser.add_argument(
        "--inner-folds", type=int, default=5,
        help="Number of inner folds for PCA selection (default: 5)",
    )
    parser.add_argument(
        "--parallel", type=int, default=6,
        help="Number of parallel workers for outer folds (default: 6)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Use cuML for GPU-accelerated PCA fitting (requires rapids-26.08 env)",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help=argparse.SUPPRESS,  # experimental: fine-grid refinement after coarse sweep
    )
    parser.add_argument(
        "--fine-size", type=int, default=5,
        help=argparse.SUPPRESS,  # experimental: values per region in fine grid
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable PCA caching",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 2. DATA LOADING
# ──────────────────────────────────────────────

def load_split_data():
    """Load and validate split data."""
    df = pd.read_csv(SPLIT_DATA_PATH)
    actual_folds = sorted(df[FOLD_COL].unique())
    expected_folds = list(range(N_CV_FOLDS + 1))
    assert actual_folds == expected_folds, \
        f"Expected folds {expected_folds}, found {actual_folds}"
    return df


import h5py

def load_embedding_h5(emb_source):
    """Load prepared embedding HDF5 into dict of regions."""
    prepared_path = emb_source["prepared_path"]
    data = {"regions": {}}
    with h5py.File(prepared_path, "r") as f:
        for region in ["peptide_emb", "n_flank_emb", "c_flank_emb"]:
            if region in f:
                data["regions"][region] = f[region][:]
    return data


def load_all_feature_data(df, feat_cfg):
    """Load all components for a feature set."""
    from config import (
        FEATURE_COMPONENTS, EMBEDDING_SOURCES, EMBEDDING_REGIONS,
    )

    csv_feature_cols = []
    emb_data_dict = {}
    new_col_frames = []

    for comp_key in feat_cfg["components"]:
        comp = FEATURE_COMPONENTS[comp_key]

        if comp["type"] == "csv":
            if comp.get("csv_path") is not None:
                csv_path = comp["csv_path"]
                if csv_path.exists():
                    df_alt = pd.read_csv(csv_path)
                    new_cols = [c for c in df_alt.columns if c not in df.columns]
                    if new_cols:
                        new_col_frames.append(df_alt[new_cols])

            all_feature_cols = get_feature_cols(df.columns)
            csv_feature_cols = [
                c for c in all_feature_cols
                if np.issubdtype(df[c].dtype, np.number)
            ]

        elif comp["type"] == "embedding":
            emb_key = comp["embedding_key"]
            emb_source = EMBEDDING_SOURCES[emb_key]
            emb_data = load_embedding_h5(emb_source)
            emb_data["pca_components"] = comp.get("pca_components", 50)
            emb_data["emb_dim"] = emb_source["emb_dim"]
            emb_data_dict[comp_key] = emb_data

    if new_col_frames:
        df = pd.concat([df] + new_col_frames, axis=1)

    return df, csv_feature_cols, emb_data_dict


# ──────────────────────────────────────────────
# 3. PCA FITTING WITH CACHING + GPU SUPPORT
# ──────────────────────────────────────────────

# Global cache for PCA fits (per worker process)
_pca_cache = {}

def _hash_array(arr):
    """Create hash of numpy array for caching."""
    return hashlib.md5(arr.tobytes()).hexdigest()

def fit_pca_cached(X_train, n_components, use_cache=True, use_gpu=False):
    """
    Fit PCA with optional caching and GPU acceleration.

    Parameters:
    -----------
    X_train : np.ndarray
        Training data
    n_components : int
        Number of PCA components
    use_cache : bool
        Whether to cache PCA fits
    use_gpu : bool
        Whether to use cuML for GPU acceleration

    Returns:
    --------
    pca : fitted PCA object
    X_transformed : np.ndarray
        Transformed data
    """
    # Try GPU first if requested
    if use_gpu:
        try:
            import cupy as cp
            from cuml.decomposition import PCA as cuML_PCA

            # Quick GPU test - create and destroy a small array
            test = cp.array([1.0])
            del test

            gpu_available = True
        except Exception as e:
            gpu_available = False
            if use_gpu:
                print(f"  WARNING: cuML/GPU not available ({e}), falling back to CPU")

        if gpu_available:
            return _fit_pca_gpu(X_train, n_components, use_cache)

    # CPU fallback
    return _fit_pca_cpu(X_train, n_components, use_cache)


def _fit_pca_gpu(X_train, n_components, use_cache=True):
    """Fit PCA on GPU using cuML."""
    import cupy as cp
    from cuml.decomposition import PCA as cuML_PCA

    # Check cache
    if use_cache:
        X_hash = _hash_array(X_train)
        cache_key = (X_hash, n_components, "gpu")
        if cache_key in _pca_cache:
            pca, X_transformed = _pca_cache[cache_key]
            return pca, X_transformed

    # Convert to CuPy array
    X_gpu = cp.asarray(X_train, dtype=cp.float32)

    # Handle zero vectors
    norms = cp.linalg.norm(X_gpu, axis=1)
    nonzero_mask = norms > 0.0
    X_valid = X_gpu[nonzero_mask]

    if X_valid.shape[0] > 0:
        pca_n = min(n_components, X_valid.shape[1], X_valid.shape[0])
        pca = cuML_PCA(n_components=pca_n)  # cuML PCA doesn't support random_state
        pca.fit(X_valid)
    else:
        pca = cuML_PCA(n_components=min(n_components, X_train.shape[1]))
        pca.fit(X_gpu)

    # Transform
    X_transformed_gpu = pca.transform(X_gpu)
    X_transformed = cp.asnumpy(X_transformed_gpu).astype(np.float64)

    # Cache result
    if use_cache:
        _pca_cache[cache_key] = (pca, X_transformed)

    return pca, X_transformed


def _fit_pca_cpu(X_train, n_components, use_cache=True):
    """Fit PCA on CPU using scikit-learn."""
    # Check cache
    if use_cache:
        X_hash = _hash_array(X_train)
        cache_key = (X_hash, n_components, "cpu")
        if cache_key in _pca_cache:
            pca, X_transformed = _pca_cache[cache_key]
            return pca, X_transformed

    # Handle zero vectors
    nonzero = np.linalg.norm(X_train, axis=1) > 0.0
    X_valid = X_train[nonzero]

    if X_valid.shape[0] > 0:
        pca_n = min(n_components, X_valid.shape[1], X_valid.shape[0])
        pca = PCA(n_components=pca_n, random_state=RANDOM_STATE)
        pca.fit(X_valid)
    else:
        pca = PCA(n_components=min(n_components, X_train.shape[1]),
                  random_state=RANDOM_STATE)
        pca.fit(X_train)

    # Transform
    X_transformed = pca.transform(X_train)

    # Cache result
    if use_cache:
        _pca_cache[cache_key] = (pca, X_transformed)

    return pca, X_transformed


# ──────────────────────────────────────────────
# 4. FEATURE BUILDING
# ──────────────────────────────────────────────

def build_pca_features(emb_data_dict, indices, pca_config, use_cache=True, use_gpu=False):
    """
    Build PCA-transformed embedding features for given indices.

    Returns list of arrays (one per embedding component/region).
    """
    parts = []
    for comp_key, emb_data in emb_data_dict.items():
        for region_name in sorted(emb_data["regions"].keys()):
            X_region = emb_data["regions"][region_name][indices].copy()
            X_region[np.isinf(X_region)] = 0.0
            X_region[np.isnan(X_region)] = 0.0

            pca_n = pca_config.get(region_name, 50)

            # Fit PCA (GPU or CPU) and transform
            pca, X_transformed = fit_pca_cached(
                X_region, pca_n, use_cache=use_cache, use_gpu=use_gpu
            )

            parts.append(X_transformed)
    return parts


def build_full_features(df, csv_X, emb_data_dict, indices, pca_config,
                        use_cache=True, use_gpu=False):
    """Build full feature matrix (CSV + PCA embeddings) for given indices."""
    emb_parts = build_pca_features(
        emb_data_dict, indices, pca_config, use_cache=use_cache, use_gpu=use_gpu
    )

    if csv_X is not None:
        X = np.concatenate([csv_X[indices]] + emb_parts, axis=1)
    else:
        X = np.concatenate(emb_parts, axis=1)

    return X


# ──────────────────────────────────────────────
# 5. MODEL DEFINITIONS
# ──────────────────────────────────────────────

def get_models(use_parallel=True, use_gpu=False):
    """
    Return model configurations.

    When use_gpu=True, uses cuML RandomForestClassifier (GPU-accelerated).
    cuML RF is ~10-50x faster than sklearn RF with n_jobs=1.
    """
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestClassifier as cuML_RF
            print("  Using cuML RandomForestClassifier (GPU)")
            rf = {
                "class": cuML_RF,
                "params": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE,
                    "n_streams": 1,
                },
                "needs_scaling": False,
                "is_gpu": True,
            }
        except ImportError:
            print("  WARNING: cuML not available, falling back to sklearn RF")
            use_gpu = False
            rf = _get_sklearn_rf(use_parallel)
    else:
        rf = _get_sklearn_rf(use_parallel)

    lr = {
        "class": LogisticRegression,
        "params": {
            "C": np.inf,
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": True,
        "is_gpu": False,
    }

    return {"rf": rf, "lr": lr}


def _get_sklearn_rf(use_parallel=True):
    """Return sklearn RandomForest configuration."""
    rf_n_jobs = 1 if use_parallel else -1
    return {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": rf_n_jobs,
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": False,
        "is_gpu": False,
    }


# ──────────────────────────────────────────────
# 5b. FINE GRID GENERATION
# ──────────────────────────────────────────────

def generate_fine_grid(best_combo, coarse_grid, fine_size=5):
    """
    Generate fine grid around best coarse combo.

    For each region, fine grid = fine_size values centered on best.
    Steps between values depend on fine_size:
      - fine_size=5: [best-2, best-1, best, best+1, best+2] (step 1)
      - fine_size=4: [best-2, best-1, best+1, best+2] (skip best)
      - fine_size=3: [best-1, best, best+1] (step 1)
      - fine_size=2: [best-1, best+1] (skip best)
      - fine_size=1: [best] (no refinement)
    """
    fine_per_region = []
    coarse_max = max(coarse_grid) + 5

    for val in best_combo:
        if fine_size <= 1:
            vals = [val]
        elif fine_size == 2:
            # skip best (already in coarse)
            vals = [max(1, val-1), min(coarse_max, val+1)]
        elif fine_size == 3:
            vals = [max(1, val-1), val, min(coarse_max, val+1)]
        elif fine_size == 4:
            # skip best
            vals = [max(1, val-2), max(1, val-1),
                    min(coarse_max, val+1), min(coarse_max, val+2)]
        else:
            # fine_size >= 5: centered, step 1
            half = fine_size // 2
            vals = [max(1, val - half + i) for i in range(fine_size)]
            vals = [min(coarse_max, v) for v in vals]
        fine_per_region.append(sorted(set(vals)))

    # All combinations of fine values
    return list(product(*fine_per_region))


# ──────────────────────────────────────────────
# 6. SINGLE COMBINATION EVALUATION
# ──────────────────────────────────────────────

def evaluate_single_combination(df, csv_X, emb_data_dict, pca_config,
                                 train_indices, test_indices, model_cfg,
                                 use_cache=True, use_gpu=False):
    """
    Evaluate a single PCA configuration on a specific train/test split.

    Returns AUC-ROC score.
    """
    y_train = df.iloc[train_indices][LABEL_COL].values.astype(np.int32)
    y_test = df.iloc[test_indices][LABEL_COL].values.astype(np.int32)

    # Build features (with GPU support for PCA)
    X_train = build_full_features(df, csv_X, emb_data_dict, train_indices,
                                   pca_config, use_cache=use_cache, use_gpu=use_gpu)
    X_test = build_full_features(df, csv_X, emb_data_dict, test_indices,
                                  pca_config, use_cache=use_cache, use_gpu=use_gpu)

    # Scale if needed (for LR)
    if model_cfg["needs_scaling"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # cuML needs float32 input
    is_gpu_model = model_cfg.get("is_gpu", False)
    if is_gpu_model:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    # Train and predict
    model = model_cfg["class"](**model_cfg["params"])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    # cuML returns cupy array — convert to numpy
    if hasattr(prob, "get"):
        import cupy as cp
        prob = cp.asnumpy(prob)

    return roc_auc_score(y_test, prob)


# ──────────────────────────────────────────────
# 7. OUTER FOLD EVALUATION (PARALLEL WORKER)
# ──────────────────────────────────────────────

def evaluate_outer_fold(args):
    """
    Evaluate one outer fold with joint PCA optimization.

    Evaluates all coarse combinations using inner CV, selects the best
    complete tuple, then evaluates that tuple once on the untouched
    outer validation fold.

    This function runs in a separate process for parallelization.
    """
    (outer_fold, df_data, csv_X_data, emb_data_dict_data,
     pca_combinations, n_inner, use_cache, use_gpu) = args

    import sys

    # Reconstruct data in worker process
    df = df_data
    csv_X = csv_X_data
    emb_data_dict = emb_data_dict_data

    # Get models — cuML RF when use_gpu=True, sklearn RF otherwise
    models = get_models(use_parallel=True, use_gpu=use_gpu)
    all_folds = sorted(df[FOLD_COL].unique())

    # Split outer train/test
    outer_test_mask = df[FOLD_COL] == outer_fold
    outer_train_mask = df[FOLD_COL] != outer_fold
    outer_test_idx = np.where(outer_test_mask.values)[0]
    outer_train_idx = np.where(outer_train_mask.values)[0]

    outer_result = {"outer_fold": outer_fold}
    total_coarse = len(pca_combinations)
    total_inner = len([f for f in all_folds if f != outer_fold][:n_inner])

    for model_key, model_cfg in models.items():
        is_gpu = model_cfg.get("is_gpu", False)
        print(f"  [Fold {outer_fold}] {model_key.upper()}: starting "
              f"({total_coarse} coarse combos × {total_inner} inner folds, "
              f"GPU={'ON' if is_gpu else 'OFF'})", flush=True)

        # ── Stage 1: Coarse sweep ──
        inner_folds = [f for f in all_folds if f != outer_fold][:n_inner]
        combo_scores = {}
        t_model_start = time.time()

        for combo_idx, combo in enumerate(pca_combinations):
            pca_config = {
                "peptide_emb": combo[0],
                "n_flank_emb": combo[1],
                "c_flank_emb": combo[2],
            }

            inner_aucs = []
            for inner_val_fold in inner_folds:
                inner_train_mask = (df[FOLD_COL] != outer_fold) & \
                                   (df[FOLD_COL] != inner_val_fold)
                inner_test_mask = df[FOLD_COL] == inner_val_fold
                inner_train_idx = np.where(inner_train_mask.values)[0]
                inner_test_idx = np.where(inner_test_mask.values)[0]

                auc = evaluate_single_combination(
                    df, csv_X, emb_data_dict, pca_config,
                    inner_train_idx, inner_test_idx, model_cfg,
                    use_cache=use_cache, use_gpu=use_gpu,
                )
                inner_aucs.append(auc)

            combo_scores[combo] = np.mean(inner_aucs)

            # Progress logging every 25 combos
            if (combo_idx + 1) % 25 == 0 or combo_idx == 0:
                elapsed = time.time() - t_model_start
                rate = (combo_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_coarse - combo_idx - 1) / rate if rate > 0 else 0
                best_so_far = max(combo_scores.values())
                print(f"  [Fold {outer_fold}] {model_key.upper()}: "
                      f"COARSE {combo_idx + 1}/{total_coarse} "
                      f"({(combo_idx + 1) / total_coarse * 100:.0f}%) "
                      f"[{elapsed:.0f}s, ETA {eta:.0f}s] "
                      f"best={best_so_far:.4f}", flush=True)

        # Select best coarse combination
        best_coarse = max(combo_scores, key=combo_scores.get)
        best_coarse_auc = combo_scores[best_coarse]
        coarse_time = time.time() - t_model_start
        print(f"  [Fold {outer_fold}] {model_key.upper()}: COARSE DONE "
              f"(best={best_coarse}, AUC={best_coarse_auc:.4f}, {coarse_time:.0f}s)",
              flush=True)

        # Use coarse-only result (fine refinement removed — experimental, see --refine)
        best_final = best_coarse
        best_final_auc = best_coarse_auc

        # ── Final evaluation on outer test ──
        pca_config_best = {
            "peptide_emb": best_final[0],
            "n_flank_emb": best_final[1],
            "c_flank_emb": best_final[2],
        }

        outer_auc = evaluate_single_combination(
            df, csv_X, emb_data_dict, pca_config_best,
            outer_train_idx, outer_test_idx, model_cfg,
            use_cache=use_cache, use_gpu=use_gpu,
        )

        outer_result[model_key] = {
            "best_combo": best_final,
            "best_inner_auc": best_final_auc,
            "outer_auc": outer_auc,
            "combo_scores": combo_scores,
        }

        model_time = time.time() - t_model_start
        print(f"  [Fold {outer_fold}] {model_key.upper()}: DONE "
              f"(best={best_final}, inner={best_final_auc:.4f}, "
              f"outer={outer_auc:.4f}, {model_time:.0f}s)", flush=True)

    return outer_result


# ──────────────────────────────────────────────
# 8. JOINT PCA OPTIMIZATION
# ──────────────────────────────────────────────

def run_joint_pca_optimization(df, csv_feature_cols, emb_data_dict,
                               pca_grid, n_outer=6, n_inner=5,
                               n_workers=6, use_cache=True, use_gpu=False,
                               do_refine=False, fine_size=5):
    """
    Joint PCA optimization: evaluate all combinations of (peptide, nflank, cflank).

    Coarse-only by default. Optional fine refinement via --refine flag.

    Parameters:
    -----------
    pca_grid : list
        Grid values per region (e.g., [5,10,15,20,25])
    n_outer : int
        Number of outer folds (default: 6)
    n_inner : int
        Number of inner folds for PCA selection (default: 5)
    n_workers : int
        Number of parallel workers (default: 6)
    use_cache : bool
        Whether to cache PCA fits (default: True)
    use_gpu : bool
        Whether to use cuML for GPU acceleration (default: False)
    do_refine : bool
        [EXPERIMENTAL] Run fine refinement around coarse best (default: False)
    fine_size : int
        [EXPERIMENTAL] Values per region in fine grid (default: 5)
    """
    # Generate all combinations
    combinations = list(product(pca_grid, repeat=3))
    n_combos = len(combinations)
    n_evals_per_combo = n_inner * 2  # 2 models (RF + LR)
    total_evals = n_combos * n_evals_per_combo * n_outer

    print(f"\n{'='*70}")
    print(f"  JOINT PCA OPTIMIZATION")
    print(f"{'='*70}")
    print(f"  Grid: {pca_grid}")
    print(f"  Combinations: {n_combos} ({len(pca_grid)}³)")
    if do_refine:
        print(f"  Refinement: ON (experimental)")
    print(f"  Inner folds: {n_inner}")
    print(f"  Outer folds: {n_outer}")
    print(f"  Parallel workers: {n_workers}")
    print(f"  Caching: {'ON' if use_cache else 'OFF'}")
    print(f"  GPU acceleration: {'ON (cuML)' if use_gpu else 'OFF (CPU)'}")
    print(f"  Total evaluations (coarse): {total_evals:,}")
    print(f"{'='*70}")

    # Pre-compute CSV features
    csv_X = None
    if csv_feature_cols:
        csv_X = df[csv_feature_cols].values.astype(np.float64)
        csv_X = np.where(np.isinf(csv_X), np.nan, csv_X)
        col_medians = np.nanmedian(csv_X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        for col_idx in range(csv_X.shape[1]):
            csv_X[np.isnan(csv_X[:, col_idx]), col_idx] = col_medians[col_idx]

    # Prepare arguments for each outer fold
    outer_folds = sorted(df[FOLD_COL].unique())[:n_outer]
    fold_args = [
        (fold, df, csv_X, emb_data_dict, combinations, n_inner,
         use_cache, use_gpu)
        for fold in outer_folds
    ]

    # Parallel execution
    t_start = time.time()
    print(f"\nRunning {n_outer} outer folds in parallel ({n_workers} workers)...")

    with Pool(n_workers) as pool:
        results = pool.map(evaluate_outer_fold, fold_args)

    t_end = time.time()
    print(f"\nCompleted in {t_end - t_start:.1f}s ({(t_end - t_start)/60:.1f} min)")

    return results


# ──────────────────────────────────────────────
# 9. RESULT AGGREGATION
# ──────────────────────────────────────────────

def aggregate_joint_results(results):
    """
    Pool inner-CV scores across all outer folds for each coarse combo.

    For each of the 125 coarse combinations, collect its mean inner-CV AUC
    from all 6 outer folds and average them. Select the complete tuple
    with the highest pooled mean inner-CV AUC.

    Outer validation AUCs are reported separately as evaluation of the
    PCA-selection procedure — they are NOT used for final PCA selection.
    """
    models = ["rf", "lr"]
    aggregated = {}

    for model_key in models:
        outer_aucs = [r[model_key]["outer_auc"] for r in results]
        best_combos = [r[model_key]["best_combo"] for r in results]
        best_inner_aucs = [r[model_key]["best_inner_auc"] for r in results]

        # Pool: collect combo_scores from all folds
        all_combo_scores = {}
        for r in results:
            for combo, score in r[model_key]["combo_scores"].items():
                if combo not in all_combo_scores:
                    all_combo_scores[combo] = []
                all_combo_scores[combo].append(score)

        # Average inner-CV scores across all 6 folds for each combo
        pooled_scores = {}
        for combo, scores in all_combo_scores.items():
            pooled_scores[combo] = np.mean(scores)

        # Select best complete tuple by pooled score
        best_pooled_combo = max(pooled_scores, key=pooled_scores.get)
        best_pooled_auc = pooled_scores[best_pooled_combo]

        aggregated[model_key] = {
            # Outer validation (evaluation of procedure, NOT for selection)
            "mean_outer_auc": float(np.mean(outer_aucs)),
            "std_outer_auc": float(np.std(outer_aucs)),
            "outer_aucs": outer_aucs,
            # Per-fold best combos and inner-CV AUCs
            "best_combos": [list(c) for c in best_combos],
            "best_inner_aucs": best_inner_aucs,
            # Pooled selection (used for final PCA)
            "pooled_scores": {str(k): v for k, v in pooled_scores.items()},
            "best_pooled_combo": list(best_pooled_combo),
            "best_pooled_auc": best_pooled_auc,
        }

    return aggregated


def print_joint_results(aggregated, outer_results):
    """Print detailed results from joint PCA optimization."""
    print(f"\n{'='*70}")
    print(f"  JOINT PCA RESULTS")
    print(f"{'='*70}")

    for model_key in ["rf", "lr"]:
        stats = aggregated[model_key]

        # Final PCA selection (pooled inner-CV)
        print(f"\n  {model_key.upper()}:")
        print(f"\n  FINAL PCA SELECTION (pooled inner-CV):")
        print(f"    Best tuple: {stats['best_pooled_combo']}")
        print(f"    Pooled inner-CV AUC: {stats['best_pooled_auc']:.4f}")

        # Outer validation performance (evaluation of procedure)
        print(f"\n  OUTER VALIDATION PERFORMANCE:")
        print(f"    Per-fold AUCs: {[f'{a:.4f}' for a in stats['outer_aucs']]}")
        print(f"    Mean ± SD: {stats['mean_outer_auc']:.4f} ± {stats['std_outer_auc']:.4f}")
        print(f"    Per-fold best combos: {stats['best_combos']}")
        print(f"    Per-fold inner-CV AUCs: {[f'{a:.4f}' for a in stats['best_inner_aucs']]}")


def _sanitize_for_json(obj):
    """Recursively convert tuples/numpy types to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


def save_joint_results(aggregated, outer_results, features_key, out_dir):
    """Save results to JSON and CSV."""
    # Save detailed JSON
    json_path = out_dir / f"joint_pca_results_{features_key}.json"
    with open(json_path, "w") as f:
        json.dump({
            "feature_set": features_key,
            "aggregated": _sanitize_for_json(aggregated),
            "outer_results": _sanitize_for_json(outer_results),
        }, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Save summary CSV
    csv_rows = []
    for model_key in ["rf", "lr"]:
        stats = aggregated[model_key]
        for i, (combo, auc) in enumerate(zip(stats["best_combos"], stats["outer_aucs"])):
            csv_rows.append({
                "feature_set": features_key,
                "model": model_key,
                "outer_fold": i,
                "peptide_pca": combo[0],
                "nflank_pca": combo[1],
                "cflank_pca": combo[2],
                "inner_cv_auc": stats["best_inner_aucs"][i],
                "outer_auc": auc,
            })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = out_dir / f"joint_pca_sweep_{features_key}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save optimal settings — pooled best tuple (not mode)
    optimal = {}
    for model_key in ["rf", "lr"]:
        stats = aggregated[model_key]
        optimal[model_key] = {
            "peptide_emb": stats["best_pooled_combo"][0],
            "n_flank_emb": stats["best_pooled_combo"][1],
            "c_flank_emb": stats["best_pooled_combo"][2],
        }
    return optimal


# ──────────────────────────────────────────────
# 10. PLOTTING
# ──────────────────────────────────────────────

def plot_joint_pca_stability(aggregated, out_dir, feature_set_name):
    """Plot PCA stability across outer folds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    regions = ["peptide_emb", "n_flank_emb", "c_flank_emb"]
    region_labels = {"peptide_emb": "Peptide", "n_flank_emb": "N-flank",
                     "c_flank_emb": "C-flank"}
    model_colors = {"rf": "#2ecc71", "lr": "#3498db"}

    for ax, region in zip(axes, regions):
        for model_key in ["rf", "lr"]:
            # Extract per-fold values for this region from best_combos
            region_idx = regions.index(region)
            values = [c[region_idx] for c in aggregated[model_key]["best_combos"]]
            x = np.arange(len(values))
            offset = 0.2 if model_key == "rf" else -0.2

            # Mark the pooled best value
            pooled_best = aggregated[model_key]["best_pooled_combo"][region_idx]

            ax.bar(x + offset, values, 0.4,
                   color=model_colors[model_key], alpha=0.7,
                   label=f"{model_key.upper()} (pooled best={pooled_best})")

        ax.set_xlabel("Outer Fold")
        ax.set_ylabel("PCA Components")
        ax.set_title(f"{region_labels[region]} Region")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i}" for i in range(len(values))])
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Joint PCA Stability — {feature_set_name}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = out_dir / f"joint_pca_stability_{feature_set_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_joint_pca_performance(aggregated, out_dir, feature_set_name):
    """Plot RF vs LR AUC across outer folds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    model_colors = {"rf": "#2ecc71", "lr": "#3498db"}

    x = np.arange(len(aggregated["rf"]["outer_aucs"]))
    width = 0.35

    for i, model_key in enumerate(["rf", "lr"]):
        aucs = aggregated[model_key]["outer_aucs"]
        offset = width / 2 if i == 0 else -width / 2
        ax.bar(x + offset, aucs, width,
               color=model_colors[model_key], alpha=0.8,
               label=f"{model_key.upper()} (mean={np.mean(aucs):.4f})")

    ax.set_xlabel("Outer Fold")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Joint PCA Performance — {feature_set_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(len(aucs))])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"joint_pca_performance_{feature_set_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    validate_config()

    # Parse grid
    pca_grid = [int(x) for x in args.grid.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOG_DIR / f"04b_pca_optimization_log_{timestamp}.txt"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger = Logger(str(LOG_PATH))
    sys.stdout = logger

    t_start = time.time()

    print("=" * 70)
    print("  04B_PCA_OPTIMIZATION — Joint PCA Optimization")
    print("=" * 70)
    print(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Features:       {args.features}")
    print(f"  Grid:           {pca_grid}")
    print(f"  Combinations:   {len(pca_grid)**3}")
    print(f"  Refine:         {'ON (experimental)' if args.refine else 'OFF'}")
    print(f"  Outer folds:    {args.outer_folds}")
    print(f"  Inner folds:    {args.inner_folds}")
    print(f"  Parallel:       {args.parallel} workers")
    print(f"  Caching:        {'OFF' if args.no_cache else 'ON'}")
    print(f"  GPU (cuML):     {'ON' if args.use_gpu else 'OFF'}")
    print(f"  Log:            {LOG_PATH}")
    print("=" * 70)

    # Check GPU availability if requested
    if args.use_gpu:
        try:
            import cupy as cp
            # Simple GPU test - try to create an array
            test_array = cp.array([1.0, 2.0, 3.0])
            _ = cp.asnumpy(test_array)
            gpu_name = "GPU detected (cuML available)"
            print(f"\n  {gpu_name}")
            del test_array
        except Exception as e:
            print(f"\n  WARNING: GPU/cuML not available ({e})")
            print(f"  Falling back to CPU mode")
            args.use_gpu = False

    # Load data
    print("\nLoading split data...")
    df = load_split_data()
    print(f"  Shape: {df.shape}")

    all_optimal = {}
    all_results = {}

    for features_key in args.features:
        print(f"\n{'#' * 70}")
        print(f"  FEATURE SET: {features_key}")
        print(f"{'#' * 70}")

        feat_cfg = get_feature_set_config(features_key)
        validate_feature_set(feat_cfg)

        # Load features
        df_feat, csv_feature_cols, emb_data_dict = load_all_feature_data(
            df.copy(), feat_cfg
        )

        if not emb_data_dict:
            print(f"  WARNING: No embedding components in {features_key}. Skipping.")
            continue

        # Run joint optimization
        outer_results = run_joint_pca_optimization(
            df_feat, csv_feature_cols, emb_data_dict,
            pca_grid=pca_grid,
            n_outer=args.outer_folds,
            n_inner=args.inner_folds,
            n_workers=args.parallel,
            use_cache=not args.no_cache,
            use_gpu=args.use_gpu,
            do_refine=args.refine,
            fine_size=args.fine_size,
        )

        # Aggregate results
        aggregated = aggregate_joint_results(outer_results)

        # Print results
        print_joint_results(aggregated, outer_results)

        # Save results
        out_dir = FIGURES_DIR / "pca_optimization"
        os.makedirs(out_dir, exist_ok=True)

        optimal = save_joint_results(aggregated, outer_results, features_key, out_dir)
        all_optimal[features_key] = optimal

        # Generate plots
        plot_joint_pca_stability(aggregated, out_dir, features_key)
        plot_joint_pca_performance(aggregated, out_dir, features_key)

    # Save all optimal settings
    optimal_path = MODEL_DIR / "pca_optimal_settings.json"
    with open(optimal_path, "w") as f:
        json.dump(all_optimal, f, indent=2)
    print(f"\nOptimal PCA settings saved: {optimal_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL PCA SETTINGS (pooled inner-CV)")
    print(f"{'=' * 70}")
    for features_key, config in all_optimal.items():
        print(f"\n  {features_key}:")
        for model_key in ["rf", "lr"]:
            print(f"    {model_key.upper()}: {list(config[model_key].values())}")

    # Footer
    t_end = time.time()
    total_minutes = (t_end - t_start) / 60
    print(f"\n{'=' * 70}")
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"  Total runtime:    {t_end - t_start:.1f}s ({total_minutes:.1f} min)")
    print(f"  Feature sets:     {len(args.features)}")
    print(f"  Grid:             {pca_grid}")
    print(f"  Combinations:     {len(pca_grid)**3}")
    print(f"  Outer folds:      {args.outer_folds}")
    print(f"  Inner folds:      {args.inner_folds}")
    print(f"  Parallel workers: {args.parallel}")
    print(f"  Optimal settings: {optimal_path}")
    print(f"  Log:              {LOG_PATH}")
    print(f"  Completed:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    logger.close()
