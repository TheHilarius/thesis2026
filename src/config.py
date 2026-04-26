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
        "display_name": "ESM-C (600M)",
        "raw_path": EMBEDDING_DIR / "esmc_protein_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc_prepared.h5",
        "emb_dim": 1152,
        "region_map": {
            "peptide": "peptide_emb",
            "n_flank": "n_flank_emb",
            "c_flank": "c_flank_emb",
        },
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
    "esmif_hybrid": {
        "display_name": "ESM-IF1 (hybrid PDB+AF2)",
        "raw_path": EMBEDDING_DIR / "esmif_hybrid_structure_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_hybrid_prepared.h5",
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
    # In EMBEDDING_SOURCES:
        "esmif_hybrid_80_fb": {
        "display_name": "ESM-IF1 hybrid 80% + AF2 fallback",
        "raw_path": EMBEDDING_DIR / "esmif_hybrid_80_fb_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_hybrid_80_fb_prepared.h5",
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
    "esmif_hybrid_60_fb": {
        "display_name": "ESM-IF1 hybrid 60% + AF2 fallback",
        "raw_path": EMBEDDING_DIR / "esmif_hybrid_60_fb_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_hybrid_60_fb_prepared.h5",
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
    "esmif_hybrid_40_fb": {
        "display_name": "ESM-IF1 hybrid 40% + AF2 fallback",
        "raw_path": EMBEDDING_DIR / "esmif_hybrid_40_fb_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_hybrid_40_fb_prepared.h5",
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
    "esmif_hybrid_20_fb": {
        "display_name": "ESM-IF1 hybrid 20% + AF2 fallback",
        "raw_path": EMBEDDING_DIR / "esmif_hybrid_20_fb_embeddings.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_hybrid_20_fb_prepared.h5",
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

EMBEDDING_REGIONS = ["peptide_emb", "n_flank_emb", "c_flank_emb"]

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
        "display_name": "ESM-C (600M) embeddings",
        "type": "embedding",
        "embedding_key": "esmc",
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    "esmif": {
        "display_name": "ESM-IF1 embeddings",
        "type": "embedding",
        "embedding_key": "esmif",
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    "esmif_hybrid": {
        "display_name": "ESM-IF1 hybrid (PDB+AF2) embeddings",
        "type": "embedding",
        "embedding_key": "esmif_hybrid",
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    # In FEATURE_COMPONENTS:
    "esmif_hybrid_80_fb": {
        "display_name": "ESM-IF1 hybrid 80% + AF2 fallback",
        "type": "embedding",
        "embedding_key": "esmif_hybrid_80_fb",
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    "esmif_hybrid_60_fb": {
        "display_name": "ESM-IF1 hybrid 60% + AF2 fallback",
        "type": "embedding",
        "embedding_key": "esmif_hybrid_60_fb",
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    "esmif_hybrid_40_fb": {
        "display_name": "ESM-IF1 hybrid 40% + AF2 fallback",
        "type": "embedding",
        "embedding_key": "esmif_hybrid_40_fb",
        "pca_components": PCA_COMPONENTS_PER_REGION,
    },
    "esmif_hybrid_20_fb": {
        "display_name": "ESM-IF1 hybrid 20% + AF2 fallback",
        "type": "embedding",
        "embedding_key": "esmif_hybrid_20_fb",
        "pca_components": PCA_COMPONENTS_PER_REGION,
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
        "display_name": "ESM-C embeddings only",
        "components": ["esmc"],
    },
    "esmif": {
        "display_name": "ESM-IF1 embeddings only",
        "components": ["esmif"],
    },
    "esmif_hybrid": {
        "display_name": "ESM-IF1 hybrid embeddings only",
        "components": ["esmif_hybrid"],
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
    # ── Structural + AA + single embedding ──
    "handcrafted_sparse_esmc": {
        "display_name": "Structural + one-hot + ESM-C",
        "components": ["handcrafted", "sparse", "esmc"],
    },
    "handcrafted_sparse_esmif": {
        "display_name": "Structural + one-hot + ESM-IF",
        "components": ["handcrafted", "sparse", "esmif"],
    },
    "handcrafted_sparse_esmif_hybrid": {
        "display_name": "Structural + one-hot + ESM-IF hybrid",
        "components": ["handcrafted", "sparse", "esmif_hybrid"],
    },
    "handcrafted_blosum_esmc": {
        "display_name": "Structural + BLOSUM50 + ESM-C",
        "components": ["handcrafted", "blosum", "esmc"],
    },
    "handcrafted_blosum_esmif": {
        "display_name": "Structural + BLOSUM50 + ESM-IF",
        "components": ["handcrafted", "blosum", "esmif"],
    },
    # ── Kitchen sink ──
    "all_sparse": {
        "display_name": "Structural + one-hot + ESM-C + ESM-IF",
        "components": ["handcrafted", "sparse", "esmc", "esmif"],
    },
    "all_blosum": {
        "display_name": "Structural + BLOSUM50 + ESM-C + ESM-IF",
        "components": ["handcrafted", "blosum", "esmc", "esmif"],
    },
        # ── Hybrid ESM-IF standalone (A/B comparison) ──
    "esmif_hybrid_80_fb": {
        "display_name": "ESM-IF1 hybrid 80% fb only",
        "components": ["esmif_hybrid_80_fb"],
    },
    "esmif_hybrid_60_fb": {
        "display_name": "ESM-IF1 hybrid 60% fb only",
        "components": ["esmif_hybrid_60_fb"],
    },
    "esmif_hybrid_40_fb": {
        "display_name": "ESM-IF1 hybrid 40% fb only",
        "components": ["esmif_hybrid_40_fb"],
    },
    "esmif_hybrid_20_fb": {
        "display_name": "ESM-IF1 hybrid 20% fb only",
        "components": ["esmif_hybrid_20_fb"],
    },
    # ── Hybrid combined with handcrafted + sparse ──
    "handcrafted_sparse_esmif_h80fb": {
        "display_name": "Structural + one-hot + ESM-IF hybrid 80% fb",
        "components": ["handcrafted", "sparse", "esmif_hybrid_80_fb"],
    },
    "handcrafted_sparse_esmif_h60fb": {
        "display_name": "Structural + one-hot + ESM-IF hybrid 60% fb",
        "components": ["handcrafted", "sparse", "esmif_hybrid_60_fb"],
    },
    "handcrafted_sparse_esmif_h40fb": {
        "display_name": "Structural + one-hot + ESM-IF hybrid 40% fb",
        "components": ["handcrafted", "sparse", "esmif_hybrid_40_fb"],
    },
    "handcrafted_sparse_esmif_h20fb": {
        "display_name": "Structural + one-hot + ESM-IF hybrid 20% fb",
        "components": ["handcrafted", "sparse", "esmif_hybrid_20_fb"],
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
