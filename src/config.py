"""
config.py
Shared configuration for the MHC-I processing prediction pipeline.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# PROJECT ROOT
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "models"

RAW_DATA_PATH = DATA_DIR / "df_all.csv"
SPLIT_DATA_PATH = DATA_DIR / "df_all_with_folds.csv"

# ──────────────────────────────────────────────
# EMBEDDING PATHS & SCHEMAS
# ──────────────────────────────────────────────
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
PREPARED_EMBEDDING_DIR = DATA_DIR / "embeddings_prepared"

# Each source defines its HDF5 layout so the prepare script
# knows exactly which datasets to read and how to align.
EMBEDDING_SOURCES = {
    "esmc": {
        "display_name": "ESM-C (600M)",
        "raw_path": EMBEDDING_DIR / "esmc_protein_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_prepared.h5",
        "emb_dim": 1152,
        # Dataset names for the three regions inside the HDF5
        "region_map": {
            "peptide": "peptide_emb",
            "n_flank": "n_flank_emb",
            "c_flank": "c_flank_emb",
        },
        # Column names for alignment keys
        "peptide_id_col": "peptide_seqs",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    "esmif": {
        "display_name": "ESM-IF1",
        "raw_path": EMBEDDING_DIR / "esmif_structure_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_prepared.h5",
        "emb_dim": 512,
        "region_map": {
            "peptide": "peptide_if_struct",
            "n_flank": "n_flank_if_struct",
            "c_flank": "c_flank_if_struct",
        },
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": False,
        "has_start_end": False,
    },
}

# Canonical region keys (used in prepared HDF5 — always these names)
EMBEDDING_REGIONS = ["peptide_emb", "n_flank_emb", "c_flank_emb"]

# PCA reduction settings
PCA_COMPONENTS_PER_REGION = 50

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
]

POSITION_AA_COLS = [
    "N4", "N3", "N2", "N1",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
    "C1", "C2", "C3", "C4",
]

_EXCLUDE_COLS = set(METADATA_COLS) | set(POSITION_AA_COLS)

# ──────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────
HAMMING_CUTOFF = 1

# ──────────────────────────────────────────────
# FEATURE SET REGISTRY
# ──────────────────────────────────────────────
FEATURE_SETS = {
    "handcrafted": {
        "display_name": "Hand-crafted features",
        "source": "csv",
        "needs_pca": False,
    },
    "esmc": {
        "display_name": "ESM-C embeddings (PCA-reduced)",
        "source": "embedding",
        "embedding_key": "esmc",
        "needs_pca": True,
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    "esmif": {
        "display_name": "ESM-IF embeddings (PCA-reduced)",
        "source": "embedding",
        "embedding_key": "esmif",
        "needs_pca": True,
        "pca_components": PCA_COMPONENTS_PER_REGION,
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
            "n_estimators": 1000,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": False,
        "coef_attr": "feature_importances_",
    },
    "lr": {
        "display_name": "Logistic Regression",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "penalty": None,
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": None,
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": True,
        "coef_attr": "coef_",
    },
}

DEFAULT_MODEL = "rf"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_feature_cols(df_columns):
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


def get_embedding_source(embedding_key):
    if embedding_key not in EMBEDDING_SOURCES:
        valid = ", ".join(sorted(EMBEDDING_SOURCES.keys()))
        raise KeyError(f"Unknown embedding key '{embedding_key}'. Valid keys: {valid}")
    return EMBEDDING_SOURCES[embedding_key]


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