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
# MODEL REGISTRY
# ──────────────────────────────────────────────
# Each entry defines everything 03_modelling.py needs to
# instantiate, train, and interpret a model.
#
#   model_class:   fully qualified sklearn class name (string)
#   params:        dict passed as **kwargs to the constructor
#   needs_scaling: whether features must be standardised before training
#   coef_attr:     attribute name for feature weights (None if N/A)
#
# To add a new model: add an entry here, then run
#   python 03_modelling.py --model <key>

MODEL_REGISTRY = {

    # ── Random Forest (no-regularisation baseline) ──
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
        "coef_attr": "feature_importances_",   # Gini importance
    },

    # ── Logistic Regression (no-regularisation baseline) ──
    "lr": {
        "display_name": "Logistic Regression",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "penalty": None,           # no regularisation
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": None,
            "random_state": RANDOM_STATE,
            "verbose": 0,
        },
        "needs_scaling": True,
        "coef_attr": "coef_",          # weight vector (shape 1×p)
    },
}

# Convenience: default model key when --model is not supplied
DEFAULT_MODEL = "rf"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_feature_cols(df_columns):
    """
    Return list of feature columns = all columns minus metadata and
    unencoded positional AA columns.
    """
    return [c for c in df_columns if c not in _EXCLUDE_COLS]


def get_model_config(model_key):
    """
    Look up a model configuration by its short key (e.g. 'rf', 'lr').
    Raises KeyError with a helpful message if the key is unknown.
    """
    if model_key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Valid keys: {valid}"
        )
    return MODEL_REGISTRY[model_key]


def validate_config():
    """Sanity-check the configuration."""
    assert N_CV_FOLDS >= 2, f"Need at least 2 CV folds, got {N_CV_FOLDS}"
    assert HELD_OUT_INDEX == N_CV_FOLDS, "HELD_OUT_INDEX must equal N_CV_FOLDS"
    assert HAMMING_CUTOFF >= 0, "HAMMING_CUTOFF must be non-negative"

    if not RAW_DATA_PATH.exists():
        print(f"WARNING: Data file not found: {RAW_DATA_PATH}")
    else:
        print(f"[OK] Data file found: {RAW_DATA_PATH}")

    print(f"[OK] Config validated: {N_CV_FOLDS}-fold CV + 1 held-out = {N_BUCKETS} buckets")
    print(f"     Project root: {PROJECT_ROOT}")
    print(f"     Data dir:     {DATA_DIR}")
    print(f"     Log dir:      {LOG_DIR}")
    print(f"     Model dir:    {MODEL_DIR}")
    print(f"     Models registered: {', '.join(sorted(MODEL_REGISTRY.keys()))}")