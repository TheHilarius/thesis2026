"""
Shared configuration for the MHC-I processing prediction pipeline.
Single source of truth — both 01_datasplit.py and 02_cross_validation.py import from here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

DATA_DIR = Path("data/processed")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")

RAW_DATA_PATH = DATA_DIR / "df_all.csv"
SPLIT_DATA_PATH = DATA_DIR / "df_all_with_folds.csv"

# ──────────────────────────────────────────────
# CROSS-VALIDATION SETTINGS
# ──────────────────────────────────────────────

N_CV_FOLDS = 5                  # k in k-fold CV — change this one number
HELD_OUT_INDEX = N_CV_FOLDS     # Always the bucket after the last CV fold
N_BUCKETS = N_CV_FOLDS + 1     # Total buckets (CV folds + 1 held-out)

RANDOM_STATE = 42

# ──────────────────────────────────────────────
# COLUMN DEFINITIONS
# ──────────────────────────────────────────────

PEPTIDE_COL = "peptide"
LABEL_COL = "label"
FOLD_COL = "fold"

# Metadata columns — NOT used as features
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
    "fold",  # Added by 01_datasplit.py
]

# ──────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────

HAMMING_CUTOFF = 1

# ──────────────────────────────────────────────
# TRAINING HYPERPARAMETERS (defaults, tunable)
# ──────────────────────────────────────────────

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4
LR_SCHEDULER_PATIENCE = 7
LR_SCHEDULER_FACTOR = 0.5

# ──────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────

def get_feature_cols(df_columns):
    """Return list of feature columns = all columns minus metadata."""
    return [c for c in df_columns if c not in METADATA_COLS]


def validate_config():
    """Sanity-check the configuration."""
    assert N_CV_FOLDS >= 2, f"Need at least 2 CV folds, got {N_CV_FOLDS}"
    assert HELD_OUT_INDEX == N_CV_FOLDS, "HELD_OUT_INDEX must equal N_CV_FOLDS"
    assert HAMMING_CUTOFF >= 0, "HAMMING_CUTOFF must be non-negative"
    print(f"Config validated: {N_CV_FOLDS}-fold CV + 1 held-out = {N_BUCKETS} buckets")