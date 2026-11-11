"""
config.py — HLA-A*01:01 (a0101) external-validation configuration.

Edition of the legacy src/pipeline/python/config.py. The legacy pipeline is
untouched and documents the previous (HLA-A*02:01) dataset; this module
re-uses ALL of its logic (feature sets, model configs, embedding schemas,
constants) and only redirects the per-dataset paths:

  data/processed/                     (legacy, HLA-A*02:01)   → untouched
  data/processed/hla_a0101/           (this allele)           ← everything here

Shared, allele-independent resources stay at their legacy locations:
  - models/                (trained .pkl models — loaded read-only)
  - logs/                  (run logs)
  - raw embeddings live per-allele: hla_a0101/embeddings/ must be generated
    on the GPU for the a0101 peptides (see README.md), because raw H5s are
    keyed to the dataset they were computed on.

Implementation note: this module executes the legacy config SOURCE in its own
namespace (importlib), mutates the per-dataset path entries inside the legacy
dicts in place (FEATURE_COMPONENTS csv_path, EMBEDDING_SOURCES raw/prepared
paths), and re-exports every public name. Scripts that do
`sys.path.insert(0, <this dir>)` + `from config import ...` (the legacy
import pattern) transparently pick up the a0101 paths, including when the
legacy 04_modelling.py functions are re-used via importlib (they bind this
module through sys.modules["config"]).
"""

import os
import sys
import importlib.util
import types
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parents[2]
LEGACY_CONFIG_PATH = PROJECT_ROOT / "src" / "pipeline" / "python" / "config.py"

# Force the legacy A/B-variant redirect OFF for the a0101 edition: the
# external allele is its own dataset dimension and must never be combined
# with THESIS_VARIANT filename suffixes.
_saved_variant = os.environ.get("THESIS_VARIANT")
os.environ["THESIS_VARIANT"] = ""
try:
    _spec = importlib.util.spec_from_file_location(
        "legacy_config_source", LEGACY_CONFIG_PATH)
    _legacy = importlib.util.module_from_spec(_spec)
    sys.modules["legacy_config_source"] = _legacy
    _spec.loader.exec_module(_legacy)
finally:
    if _saved_variant is None:
        os.environ.pop("THESIS_VARIANT", None)
    else:
        os.environ["THESIS_VARIANT"] = _saved_variant

# ──────────────────────────────────────────────
# A0101 PATH OVERRIDES
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hla_a0101"
EMBEDDING_DIR = DATA_DIR / "embeddings"
PREPARED_EMBEDDING_DIR = DATA_DIR / "embeddings_prepared"

# The external set has NO CV folds — both "raw" and "split" point at the
# same a0101 df_all.csv produced by the a0101 R chain (07_alpha_fold).
RAW_DATA_PATH = DATA_DIR / "df_all.csv"
SPLIT_DATA_PATH = DATA_DIR / "df_all.csv"

# Alternate encoded CSVs (produced by a0101 08_aa_encoding.R)
SPARSE_DATA_PATH = DATA_DIR / "df_sparse.csv"
BLOSUM50_DATA_PATH = DATA_DIR / "df_blosum50.csv"
BLOSUM62_DATA_PATH = DATA_DIR / "df_blosum62.csv"
BLOSUM80_DATA_PATH = DATA_DIR / "df_blosum80.csv"
KIDERA_DATA_PATH = DATA_DIR / "df_kidera.csv"
ATCHLEY_DATA_PATH = DATA_DIR / "df_atchley.csv"
COMPOSITIONAL_DATA_PATH = DATA_DIR / "df_compositional.csv"

# Shared with legacy (read-only resources)
LOG_DIR = _legacy.LOG_DIR
MODEL_DIR = _legacy.MODEL_DIR
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "models_hla_a0101"

# ──────────────────────────────────────────────
# IN-PLACE REDIRECTION OF LEGACY DICTS
# (affects get_feature_component / get_embedding_source lookups made by
#  re-used legacy functions — they operate on these very dicts)
# ──────────────────────────────────────────────
_REDIRECT_CSV_NAMES = {
    "df_all.csv", "df_all_with_folds.csv",
    "df_sparse.csv", "df_blosum50.csv", "df_blosum62.csv",
    "df_blosum80.csv", "df_kidera.csv", "df_atchley.csv",
    "df_compositional.csv",
}

for _comp in _legacy.FEATURE_COMPONENTS.values():
    _p = _comp.get("csv_path")
    if _p is not None and Path(_p).name in _REDIRECT_CSV_NAMES:
        _comp["csv_path"] = DATA_DIR / Path(_p).name

for _src in _legacy.EMBEDDING_SOURCES.values():
    _raw = _src.get("raw_path")
    if _raw is not None:
        # raw H5s are dataset-keyed → per-allele dir (must be generated)
        _src["raw_path"] = EMBEDDING_DIR / Path(_raw).name
    _prep = _src.get("prepared_path")
    if _prep is not None:
        _src["prepared_path"] = PREPARED_EMBEDDING_DIR / Path(_prep).name

# ──────────────────────────────────────────────
# RE-EXPORT every public name from the legacy config
# ──────────────────────────────────────────────
for _name, _val in list(vars(_legacy).items()):
    if _name.startswith("_"):
        continue
    if isinstance(_val, types.ModuleType):
        continue
    if _name in ("VARIANT", "_variant"):
        continue
    if _name in globals():
        continue  # keep the a0101 overrides above
    globals()[_name] = _val

# Marker so downstream tools can assert they got the right config edition
CONFIG_EDITION = "hla_a0101"
