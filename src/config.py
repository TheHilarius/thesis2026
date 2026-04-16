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
]

# ──────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────
HAMMING_CUTOFF = 1

# ──────────────────────────────────────────────
# RANDOM FOREST HYPERPARAMETERS (no regularization baseline)
# ──────────────────────────────────────────────
RF_N_ESTIMATORS = 1000       # Number of trees
RF_MAX_DEPTH = None         # No depth limit (no regularization)
RF_MIN_SAMPLES_SPLIT = 2    # Default (minimal constraint)
RF_MIN_SAMPLES_LEAF = 1     # Default (minimal constraint)
RF_MAX_FEATURES = "sqrt"    # Standard for classification
RF_N_JOBS = -1              # Use all CPU cores
RF_CLASS_WEIGHT = None      # No class weighting (change to "balanced" if needed)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_feature_cols(df_columns):
    """Return list of feature columns = all columns minus metadata."""
    return [c for c in df_columns if c not in METADATA_COLS]


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