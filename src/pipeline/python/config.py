"""
config.py
Shared configuration for the MHC-I processing prediction pipeline.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# PROJECT ROOT
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "models"

RAW_DATA_PATH = DATA_DIR / "df_all.csv"
SPLIT_DATA_PATH = DATA_DIR / "df_all_with_folds.csv"

# Alternate CSVs with encoded positional AA columns
SPARSE_DATA_PATH = DATA_DIR / "df_all_sparse.csv"
BLOSUM_DATA_PATH = DATA_DIR / "df_all_blosum50.csv"

# ──────────────────────────────────────────────
# EMBEDDING PATHS & SCHEMAS
# ──────────────────────────────────────────────
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
PREPARED_EMBEDDING_DIR = DATA_DIR / "embeddings_prepared"

EMBEDDING_SOURCES = {
    "esmc": {
        "display_name": "ESM-C (600M) context window",
        "raw_path": EMBEDDING_DIR / "esmc_context_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_context_prepared.h5",
        "emb_dim": 1152,
        "region_map": {
            "context": "context_emb",
        },
        "peptide_id_col": "peptide_seqs",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    "esmif": {
        "display_name": "ESM-IF1 context window",
        "raw_path": EMBEDDING_DIR / "esmif_context_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_context_prepared.h5",
        "emb_dim": 512,
        "region_map": {
            "context": "context_if_struct",
        },
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": False,
        "has_start_end": False,
    },
    # ── Fixed-29 per-residue window embeddings (B pipeline) ──
    # kind=="windows": raw HDF5 holds window_if_struct (N,29,D) + pad_mask.
    # pad_mask convention: True = PADDED slot, False = real residue slot.
    # emb_dim is EXPECTED value; the authoritative dim is read from the H5
    # attr at load time (so the same code path serves D=512 IF and D=1152 C).
    "esmif_zero": {
        "display_name": "ESM-IF1 windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esm-if_test_zero.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_zero_windows_prepared.h5",
        "window_ds": "window_if_struct",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": "n_pad",
        "c_pad_ds": "c_pad",
        "status_ds": "status",
        "emb_dim": 512,
        "window": 29,
        "pad_mode": "zero",
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    "esmif_padtoken": {
        "display_name": "ESM-IF1 windows (pad-token)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esm-if_test_padtoken.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_padtoken_windows_prepared.h5",
        "window_ds": "window_if_struct",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": "n_pad",
        "c_pad_ds": "c_pad",
        "status_ds": "status",
        "emb_dim": 512,
        "window": 29,
        "pad_mode": "padtoken",
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    "esmif_eosrepeat": {
        "display_name": "ESM-IF1 windows (eos-repeat)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esm-if_test_eosrepeat.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_eosrepeat_windows_prepared.h5",
        "window_ds": "window_if_struct",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": "n_pad",
        "c_pad_ds": "c_pad",
        "status_ds": "status",
        "emb_dim": 512,
        "window": 29,
        "pad_mode": "eosrepeat",
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    # ── ESM-C (600M) fixed-29 window embeddings ──
    # Raw HDF5: window_embeddings (N,29,1152) + pad_mask + peptide/uniprot_id/
    # row_indices/fallback_flag. No status/n_pad/c_pad/start/end datasets.
    "esmc_win_zeropad": {
        "display_name": "ESM-C (600M) windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc_context_embeddings_zeropad.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_win_zeropad_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 1152,
        "window": 29,
        "pad_mode": "zeropad",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "esmc_win_padtoken": {
        "display_name": "ESM-C (600M) windows (pad-token)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc_context_embeddings_padtoken.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_win_padtoken_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 1152,
        "window": 29,
        "pad_mode": "pad_token",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "esmc_win_impute_bos_eos": {
        "display_name": "ESM-C (600M) windows (impute-bos-eos)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc_context_embeddings_impute_bos_eos.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_win_impute_bos_eos_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 1152,
        "window": 29,
        "pad_mode": "impute_bos_eos",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "esmc_win_impute_boundary": {
        "display_name": "ESM-C (600M) windows (impute-boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc_context_embeddings_impute_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_win_impute_boundary_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 1152,
        "window": 29,
        "pad_mode": "impute_boundary",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
}

EMBEDDING_REGIONS = ["context_emb"]

# PCA components for the single context embedding
PCA_COMPONENTS = 50

# PCA optimization sweep values
PCA_COMPONENTS_SWEEP = [10, 25, 50, 75, 100, 150]

# ──────────────────────────────────────────────
# CROSS-VALIDATION SETTINGS
# ──────────────────────────────────────────────
N_CV_FOLDS = 5
HELD_OUT_INDEX = N_CV_FOLDS
N_BUCKETS = N_CV_FOLDS + 1

RANDOM_STATE = 42

# ──────────────────────────────────────────────
# COLUMN DEFINITIONS
# ──────────────────────────────────────────────
PEPTIDE_COL = "peptide"
LABEL_COL = "label"
FOLD_COL = "fold"

METADATA_COLS = [
    "start",
    "end",
    "peptide",
    "pep_length",
    "rank",
    "uniprot_id",
    "source_molecule",
    "molecule_parent",
    "label",
    "sequence",
    "protein_length",
    "position_shift",
    "n_flank",
    "c_flank",
    "full_context",
    "fold",
    "nflank_start",
    "nflank_end",
    "cflank_start",
    "cflank_end",
    "rel_distance_from_n_terminus"
]

POSITION_AA_COLS = [
    "N10", "N9", "N8", "N7", "N6", "N5", "N4", "N3", "N2", "N1",
    "P1",  "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
    "C1",  "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
]

AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

_EXCLUDE_COLS = set(METADATA_COLS) | set(POSITION_AA_COLS)

# ──────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────
HAMMING_CUTOFF = 1

# ──────────────────────────────────────────────
# FEATURE COMPONENTS (atomic building blocks)
# ──────────────────────────────────────────────
FEATURE_COMPONENTS = {
    "handcrafted": {
        "display_name": "Structural features",
        "type": "csv",
        "csv_path": None,       # columns already in split data
    },
    "sparse": {
        "display_name": "One-hot AA encoding",
        "type": "csv",
        "csv_path": SPARSE_DATA_PATH,
    },
    "blosum": {
        "display_name": "BLOSUM50 AA encoding",
        "type": "csv",
        "csv_path": BLOSUM_DATA_PATH,
    },
    "esmc": {
        "display_name": "ESM-C (600M) context window",
        "type": "embedding",
        "embedding_key": "esmc",
        "pca_components": PCA_COMPONENTS,
    },
    "esmif": {
        "display_name": "ESM-IF1 context window",
        "type": "embedding",
        "embedding_key": "esmif",
        "pca_components": PCA_COMPONENTS,
    },
    "esmif_win": {
        "display_name": "ESM-IF1 windows (zero-pad, 29x512)",
        "type": "embedding_windows",
        "embedding_key": "esmif_zero",
        "pca_components": 64,
    },
    "esmc_win_zeropad": {
        "display_name": "ESM-C windows (zero-pad, 29x1152)",
        "type": "embedding_windows",
        "embedding_key": "esmc_win_zeropad",
        "pca_components": 64,
    },
    "esmc_win_padtoken": {
        "display_name": "ESM-C windows (pad-token, 29x1152)",
        "type": "embedding_windows",
        "embedding_key": "esmc_win_padtoken",
        "pca_components": 64,
    },
    "esmc_win_impute_bos_eos": {
        "display_name": "ESM-C windows (impute-bos-eos, 29x1152)",
        "type": "embedding_windows",
        "embedding_key": "esmc_win_impute_bos_eos",
        "pca_components": 64,
    },
    "esmc_win_impute_boundary": {
        "display_name": "ESM-C windows (impute-boundary, 29x1152)",
        "type": "embedding_windows",
        "embedding_key": "esmc_win_impute_boundary",
        "pca_components": 64,
    },
}

# ──────────────────────────────────────────────
# FEATURE SETS (named combinations of components)
# ──────────────────────────────────────────────
FEATURE_SETS = {
    # ── Single-source baselines ──
    "handcrafted": {
        "display_name": "Structural features only",
        "components": ["handcrafted"],
    },
    "esmc": {
        "display_name": "ESM-C context window only",
        "components": ["esmc"],
    },
    "esmif": {
        "display_name": "ESM-IF1 context window only",
        "components": ["esmif"],
    },
    # ── Structural + AA encoding ──
    "handcrafted_sparse": {
        "display_name": "Structural + one-hot AA",
        "components": ["handcrafted", "sparse"],
    },
    "handcrafted_blosum": {
        "display_name": "Structural + BLOSUM50 AA",
        "components": ["handcrafted", "blosum"],
    },
    # ── Structural + AA + single embedding (context window) ──
    "handcrafted_esmc": {
        "display_name": "Structural + ESM-C context",
        "components": ["handcrafted", "esmc"],
    },
    "handcrafted_esmif": {
        "display_name": "Structural + ESM-IF1 context",
        "components": ["handcrafted", "esmif"],
    },
    "handcrafted_sparse_esmc": {
        "display_name": "Structural + one-hot + ESM-C",
        "components": ["handcrafted", "sparse", "esmc"],
    },
    "handcrafted_sparse_esmif": {
        "display_name": "Structural + one-hot + ESM-IF",
        "components": ["handcrafted", "sparse", "esmif"],
    },
    # ── Window-embedding feature sets (B pipeline) ──
    "handcrafted_esmif_win": {
        "display_name": "Structural + ESM-IF windows",
        "components": ["handcrafted", "esmif_win"],
    },
    "handcrafted_sparse_esmif_win": {
        "display_name": "Structural + one-hot + ESM-IF windows",
        "components": ["handcrafted", "sparse", "esmif_win"],
    },
    "handcrafted_blosum_esmif_win": {
        "display_name": "Structural + BLOSUM50 + ESM-IF windows",
        "components": ["handcrafted", "blosum", "esmif_win"],
    },
    # ── ESM-C window-embedding feature sets ──
    "handcrafted_sparse_esmc_win_zeropad": {
        "display_name": "Structural + one-hot + ESM-C windows (zero-pad)",
        "components": ["handcrafted", "sparse", "esmc_win_zeropad"],
    },
    "handcrafted_sparse_esmc_win_padtoken": {
        "display_name": "Structural + one-hot + ESM-C windows (pad-token)",
        "components": ["handcrafted", "sparse", "esmc_win_padtoken"],
    },
    "handcrafted_sparse_esmc_win_impute_bos_eos": {
        "display_name": "Structural + one-hot + ESM-C windows (impute-bos-eos)",
        "components": ["handcrafted", "sparse", "esmc_win_impute_bos_eos"],
    },
    "handcrafted_sparse_esmc_win_impute_boundary": {
        "display_name": "Structural + one-hot + ESM-C windows (impute-boundary)",
        "components": ["handcrafted", "sparse", "esmc_win_impute_boundary"],
    },
    "handcrafted_blosum_esmc": {
        "display_name": "Structural + BLOSUM50 + ESM-C",
        "components": ["handcrafted", "blosum", "esmc"],
    },
    "handcrafted_blosum_esmif": {
        "display_name": "Structural + BLOSUM50 + ESM-IF",
        "components": ["handcrafted", "blosum", "esmif"],
    },
    # ── Combined context window ──
    "handcrafted_sparse_esmc_esmif": {
        "display_name": "Structural + one-hot + ESM-C + ESM-IF",
        "components": ["handcrafted", "sparse", "esmc", "esmif"],
    },
}

DEFAULT_FEATURE_SET = "handcrafted"

# ──────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "rf": {
        "display_name": "Random Forest",
        "model_class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "n_estimators": 1000, #The number of trees in the forrest.
            "max_depth": None, #The maximum depth of the tree.
            "min_samples_split": 2, #The minimum number of samples required to split an internal node.
            "min_samples_leaf": 1, #The minimum number of samples required to be at a leaf node.
            "max_features": "sqrt", #The number of features to consider when looking for the best split.
            "class_weight": "balanced", #We have class imbalance, so use balanced weights to give more importance to the minority class.
            "n_jobs": -1, #Use all available CPU cores for parallel processing.
            "random_state": RANDOM_STATE, #Set random state for reproducibility, in our case we set it to 42.
            "verbose": 0, #
        },
        "needs_scaling": False, #Random forests are tree-based models and do not require feature scaling.
        "coef_attr": "feature_importances_", #The attribute of the fitted model that contains feature importance scores.
    },
    "lr_l2": {
        "display_name": "Logistic Regression (L2)",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "penalty": "l2",
            "C": 1.0,                       # inverse regularization strength; smaller = stronger
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": True,
        "coef_attr": "coef_",
    },
    "lr_elasticnet": {
        "display_name": "Logistic Regression (ElasticNet)",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "penalty": "elasticnet",
            "C": 1.0,                       # inverse regularization strength; smaller = stronger
            "solver": "saga",
            "l1_ratio": 0.5,
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": True,
        "coef_attr": "coef_",
    },
    "xgb": {
        "display_name": "XGBoost",
        "model_class": "xgboost.XGBClassifier",
        "params": {
            "n_estimators": 1000,           # number of boosting rounds
            "max_depth": 6,                 # maximum tree depth
            "learning_rate": 0.1,           # step size shrinkage (eta)
            "subsample": 1.0,               # fraction of rows sampled per tree
            "colsample_bytree": 1.0,        # fraction of features sampled per tree
            "min_child_weight": 1,          # minimum sum of instance weight in a leaf
            "reg_lambda": 1.0,              # L2 regularization
            "reg_alpha": 0.0,               # L1 regularization
            "gamma": 0.0,                   # min loss reduction required for a split
            "eval_metric": "logloss",
            "n_jobs": -1,                   # use all CPU cores
            "verbosity": 0,                 # silent
            "random_state": RANDOM_STATE,   # reproducibility
        },
        "needs_scaling": False,             # tree-based; no scaling required
        "coef_attr": "feature_importances_",  # gain-based importance
        "scale_pos_weight": "auto",         # computed per-fold from y_train
        "skip_imputation": True,            # let XGBoost handle NaN natively
    },
}

DEFAULT_MODEL = "rf"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_feature_cols(df_columns):
    """Return all columns that are not metadata or raw AA letters."""
    return [c for c in df_columns if c not in _EXCLUDE_COLS]


def get_model_config(model_key):
    if model_key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {valid}")
    return MODEL_REGISTRY[model_key]


def get_feature_set_config(features_key):
    if features_key not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS.keys()))
        raise KeyError(f"Unknown feature set '{features_key}'. Valid keys: {valid}")
    return FEATURE_SETS[features_key]


def get_feature_component(comp_key):
    if comp_key not in FEATURE_COMPONENTS:
        valid = ", ".join(sorted(FEATURE_COMPONENTS.keys()))
        raise KeyError(f"Unknown component '{comp_key}'. Valid: {valid}")
    return FEATURE_COMPONENTS[comp_key]


def get_embedding_source(embedding_key):
    if embedding_key not in EMBEDDING_SOURCES:
        valid = ", ".join(sorted(EMBEDDING_SOURCES.keys()))
        raise KeyError(f"Unknown embedding key '{embedding_key}'. Valid keys: {valid}")
    return EMBEDDING_SOURCES[embedding_key]


def validate_feature_set(feat_cfg):
    """Check that a feature set's components are valid and compatible."""
    components = feat_cfg["components"]

    # Each component must exist
    for comp_key in components:
        get_feature_component(comp_key)

    # Cannot combine sparse + blosum (same column names, different values)
    aa_encodings = [c for c in components if c in ("sparse", "blosum")]
    if len(aa_encodings) > 1:
        raise ValueError(
            f"Cannot combine multiple AA encodings in one feature set: "
            f"{aa_encodings}. Use either 'sparse' or 'blosum', not both."
        )

    # No duplicate components
    if len(components) != len(set(components)):
        raise ValueError(f"Duplicate components in feature set: {components}")


def validate_config():
    assert N_CV_FOLDS >= 2, f"Need at least 2 CV folds, got {N_CV_FOLDS}"
    assert HELD_OUT_INDEX == N_CV_FOLDS, "HELD_OUT_INDEX must equal N_CV_FOLDS"
    assert HAMMING_CUTOFF >= 0, "HAMMING_CUTOFF must be non-negative"

    if not RAW_DATA_PATH.exists():
        print(f"WARNING: Data file not found: {RAW_DATA_PATH}")
    else:
        print(f"[OK] Data file found: {RAW_DATA_PATH}")

    print(f"[OK] Config validated: {N_CV_FOLDS}-fold CV + 1 held-out = {N_BUCKETS} buckets")
    print(f"     Project root:  {PROJECT_ROOT}")
    print(f"     Data dir:      {DATA_DIR}")
    print(f"     Embedding dir: {EMBEDDING_DIR}")
    print(f"     Log dir:       {LOG_DIR}")
    print(f"     Model dir:     {MODEL_DIR}")
    print(f"     Models:        {', '.join(sorted(MODEL_REGISTRY.keys()))}")
    print(f"     Feature sets:  {', '.join(sorted(FEATURE_SETS.keys()))}")
    print(f"     Components:    {', '.join(sorted(FEATURE_COMPONENTS.keys()))}")
