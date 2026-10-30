"""
config.py
Shared configuration for the MHC-I processing prediction pipeline.
"""

from pathlib import Path
import os

# ──────────────────────────────────────────────
# PROJECT ROOT
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ──────────────────────────────────────────────
# DATASET VARIANT (A/B testing; unset = current behavior)
# THESIS_VARIANT=remove715negonly redirects every per-dataset path
# (raw/split/encoding CSVs, prepared embedding H5s) to *_<variant> files.
# Raw embedding H5s are shared — the prepare step aligns them to the split.
# ──────────────────────────────────────────────
VARIANT = os.environ.get("THESIS_VARIANT") or None


def _variant(path: Path) -> Path:
    if not VARIANT:
        return path
    return path.with_name(f"{path.stem}_{VARIANT}{path.suffix}")

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "models"

RAW_DATA_PATH = _variant(DATA_DIR / "df_all.csv")
SPLIT_DATA_PATH = _variant(DATA_DIR / "df_all_with_folds.csv")

# Alternate CSVs with encoded positional AA columns
# df_<enc>.csv: peptide+label keys + encoded cols only (used by modelling)
# df_all_<enc>.csv: encoded + all non-position cols (for R/analyses)
SPARSE_DATA_PATH = _variant(DATA_DIR / "df_sparse.csv")
BLOSUM50_DATA_PATH = _variant(DATA_DIR / "df_blosum50.csv")
BLOSUM62_DATA_PATH = _variant(DATA_DIR / "df_blosum62.csv")
BLOSUM80_DATA_PATH = _variant(DATA_DIR / "df_blosum80.csv")
KIDERA_DATA_PATH = _variant(DATA_DIR / "df_kidera.csv")
ATCHLEY_DATA_PATH = _variant(DATA_DIR / "df_atchley.csv")
# Compositional: frac_<aa> = aa count across the 29 positions / 29
COMPOSITIONAL_DATA_PATH = _variant(DATA_DIR / "df_compositional.csv")

# ──────────────────────────────────────────────
# EMBEDDING PATHS & SCHEMAS
# ──────────────────────────────────────────────
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
PREPARED_EMBEDDING_DIR = DATA_DIR / "embeddings_prepared"

# Default homemade (top-variance) reduction rung per embedding feature
# key — used by --vardims (04_modelling / 04b_run_matrix) when no explicit
# EV is given. Dim lists: results/tables/dim_variance_<key>_ev<N>.json.
# Chosen rungs are DeLong-tied with the unreduced raw embedding ceiling.
DEFAULT_VAR_EV = {
    "esmc300m_win_padtoken": 80,   # 643 dims, 18,647 feats (raw 0.7040, ns)
    "esmif_eosrepeat": 65,         # 260 dims, 7,540 feats  (raw 0.6498, ns)
}

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
    "esmif_pad": {
        "display_name": "ESM-IF1 windows (pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esm-if_test_pad.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_pad_windows_prepared.h5",
        "window_ds": "window_if_struct",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": "n_pad",
        "c_pad_ds": "c_pad",
        "status_ds": "status",
        "emb_dim": 512,
        "window": 29,
        "pad_mode": "pad",
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    "esmif_boundary": {
        "display_name": "ESM-IF1 windows (boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esm-if_test_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_boundary_windows_prepared.h5",
        "window_ds": "window_if_struct",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": "n_pad",
        "c_pad_ds": "c_pad",
        "status_ds": "status",
        "emb_dim": 512,
        "window": 29,
        "pad_mode": "boundary",
        "peptide_id_col": "peptide_ids",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": True,
        "has_start_end": True,
    },
    "esmif_eos_bos_repeat": {
        "display_name": "ESM-IF1 windows (eos/bos-repeat)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esm-if_test_eos_bos_repeat.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_eos_bos_repeat_windows_prepared.h5",
        "window_ds": "window_if_struct",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": "n_pad",
        "c_pad_ds": "c_pad",
        "status_ds": "status",
        "emb_dim": 512,
        "window": 29,
        "pad_mode": "eos_bos_repeat",
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
    # ── Reduced 29x21 prepared windows (dim21 test) ──
    # Built by 11_reduce_prepared_windows.py --first 21 / --dims <top21 JSON>.
    # raw_path = the FULL prepared H5 (the reduce script's input), NOT a raw
    # extraction file — the reducer copies the prepared schema, dim-gathered.
    "esmc300m_win_padtoken_d21_first": {
        "display_name": "ESM-C-300M windows (pad-token, first-21 dims)",
        "kind": "windows",
        "raw_path": None,
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc300m_win_padtoken_prepared_first21.h5",
        "window_ds": "windows",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 21,
        "window": 29,
        "pad_mode": "zeropad",
        "peptide_id_col": "peptide_seqs",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": False,
        "has_start_end": False,
    },
    "esmc300m_win_padtoken_d21_var": {
        "display_name": "ESM-C-300M windows (pad-token, var-21 dims)",
        "kind": "windows",
        "raw_path": None,
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc300m_win_padtoken_prepared_var21.h5",
        "window_ds": "windows",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 21,
        "window": 29,
        "pad_mode": "zeropad",
        "peptide_id_col": "peptide_seqs",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": False,
        "has_start_end": False,
    },
    "esmif_eosrepeat_d21_first": {
        "display_name": "ESM-IF1 windows (eos-repeat, first-21 dims)",
        "kind": "windows",
        "raw_path": None,
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_eosrepeat_windows_prepared_first21.h5",
        "window_ds": "windows",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 21,
        "window": 29,
        "pad_mode": "eosrepeat",
        "peptide_id_col": "peptide_seqs",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": False,
        "has_start_end": False,
    },
    "esmif_eosrepeat_d21_var": {
        "display_name": "ESM-IF1 windows (eos-repeat, var-21 dims)",
        "kind": "windows",
        "raw_path": None,
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmif_eosrepeat_windows_prepared_var21.h5",
        "window_ds": "windows",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 21,
        "window": 29,
        "pad_mode": "eosrepeat",
        "peptide_id_col": "peptide_seqs",
        "uniprot_id_col": "uniprot_ids",
        "has_row_indices": False,
        "has_start_end": False,
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
    # ── Decided pad modes (thesis decision, Oct 2026, for future agents) ──
    # ESM-C  -> padtoken       (learned <pad> vector; filler stays into conv)
    # ESM-IF -> eos_bos_repeat (BOS/EOS copies; filler stays into conv)
    # Rationale: only ~1600/46617 rows padded (~3.5%), so pads are not the
    # decision variable. 05 CNN masks post-conv only (no pre-zeroing) and has
    # no per-mode special-casing. Do not reintroduce pad ablations without
    # beating the current champion first.
    # ── ESM-C (300M) fixed-29 window embeddings ──
    # Same schema as the 600M windows; emb_dim=960 (d_model of esmc_300m).
    "esmc300m_win_zeropad": {
        "display_name": "ESM-C (300M) windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc300m_context_embeddings_zeropad.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc300m_win_zeropad_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 960,
        "window": 29,
        "pad_mode": "zeropad",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "esmc300m_win_padtoken": {
        "display_name": "ESM-C (300M) windows (pad-token)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc300m_context_embeddings_padtoken.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc300m_win_padtoken_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 960,
        "window": 29,
        "pad_mode": "pad_token",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "esmc300m_win_impute_bos_eos": {
        "display_name": "ESM-C (300M) windows (impute-bos-eos)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc300m_context_embeddings_impute_bos_eos.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc300m_win_impute_bos_eos_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 960,
        "window": 29,
        "pad_mode": "impute_bos_eos",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "esmc300m_win_impute_boundary": {
        "display_name": "ESM-C (300M) windows (impute-boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc300m_context_embeddings_impute_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc300m_win_impute_boundary_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 960,
        "window": 29,
        "pad_mode": "impute_boundary",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    # ── ESM-C (6B) fixed-29 window embeddings ──
    # Same schema as the 600M/300M windows; emb_dim=2560 (d_model of esmc_6b).
    "esmc6b_win_padtoken": {
        "display_name": "ESM-C (6B) windows (pad-token)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "esmc6b_context_embeddings_padtoken.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "esmc6b_win_padtoken_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": None,
        "emb_dim": 2560,
        "window": 29,
        "pad_mode": "pad_token",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    # ── ProteinMPNN fixed-29 window embeddings (encoder-only h_V, dim 128) ──
    # Raw HDF5: window_embeddings (N,29,128) + pad_mask + window_plddt + status
    # + peptide/uniprot_id/row_indices/fallback_flag. Hybrid mapping: positional
    # when df sequence == structure sequence (status=0, ~46550 rows); the 13
    # isoform-mismatch proteins are located by sequence search (status=1, 67
    # rows), not zero-filled. No notfound rows.
    # Only zero + boundary pad modes exist — ProteinMPNN has no BOS/EOS or pad token.
    # window_plddt (AFDB CA B-factor, NaN at pads) is available for confidence
    # stratification but is not consumed by the flat/PCA modelling path.
    "mpnn_v48020_zeropad": {
        "display_name": "ProteinMPNN v_48_020 windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_020_context_embeddings_zero.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48020_zeropad_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "zero",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48020_boundary": {
        "display_name": "ProteinMPNN v_48_020 windows (boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_020_context_embeddings_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48020_boundary_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "boundary",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48002_zeropad": {
        "display_name": "ProteinMPNN v_48_002 windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_002_context_embeddings_zero.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48002_zeropad_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "zero",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48002_boundary": {
        "display_name": "ProteinMPNN v_48_002 windows (boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_002_context_embeddings_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48002_boundary_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "boundary",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48010_zeropad": {
        "display_name": "ProteinMPNN v_48_010 windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_010_context_embeddings_zero.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48010_zeropad_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "zero",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48010_boundary": {
        "display_name": "ProteinMPNN v_48_010 windows (boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_010_context_embeddings_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48010_boundary_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "boundary",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48030_zeropad": {
        "display_name": "ProteinMPNN v_48_030 windows (zero-pad)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_030_context_embeddings_zero.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48030_zeropad_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "zero",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
    "mpnn_v48030_boundary": {
        "display_name": "ProteinMPNN v_48_030 windows (boundary)",
        "kind": "windows",
        "raw_path": EMBEDDING_DIR / "mpnn_v_48_030_context_embeddings_boundary.h5",
        "prepared_path": PREPARED_EMBEDDING_DIR / "mpnn_v48030_boundary_prepared.h5",
        "window_ds": "window_embeddings",
        "pad_mask_ds": "pad_mask",
        "n_pad_ds": None,
        "c_pad_ds": None,
        "status_ds": "status",
        "emb_dim": 128,
        "window": 29,
        "pad_mode": "boundary",
        "peptide_id_col": "peptide",
        "uniprot_id_col": "uniprot_id",
        "has_row_indices": True,
        "has_start_end": False,
    },
}

# Variant redirect for prepared H5s only — raw H5s are shared (the prepare
# step aligns raw embeddings to the variant split and bakes labels+folds in).
for _src in EMBEDDING_SOURCES.values():
    if "prepared_path" in _src:
        _src["prepared_path"] = _variant(_src["prepared_path"])

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
    "blosum50": {
        "display_name": "BLOSUM50 AA encoding",
        "type": "csv",
        "csv_path": BLOSUM50_DATA_PATH,
    },
    "blosum62": {
        "display_name": "BLOSUM62 AA encoding",
        "type": "csv",
        "csv_path": BLOSUM62_DATA_PATH,
    },
    "blosum80": {
        "display_name": "BLOSUM80 AA encoding",
        "type": "csv",
        "csv_path": BLOSUM80_DATA_PATH,
    },
    "kidera": {
        "display_name": "Kidera factors AA encoding",
        "type": "csv",
        "csv_path": KIDERA_DATA_PATH,
    },
    "atchley": {
        "display_name": "Atchley factors AA encoding",
        "type": "csv",
        "csv_path": ATCHLEY_DATA_PATH,
    },
    "compositional": {
        "display_name": "AA composition (frac over 29 positions)",
        "type": "csv",
        "csv_path": COMPOSITIONAL_DATA_PATH,
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
    "esmif_pad": {
        "display_name": "ESM-IF1 windows (pad, 29x512)",
        "type": "embedding_windows",
        "embedding_key": "esmif_pad",
        "pca_components": 64,
    },
    "esmif_boundary": {
        "display_name": "ESM-IF1 windows (boundary, 29x512)",
        "type": "embedding_windows",
        "embedding_key": "esmif_boundary",
        "pca_components": 64,
    },
    "esmif_eos_bos_repeat": {
        "display_name": "ESM-IF1 windows (eos/bos-repeat, 29x512)",
        "type": "embedding_windows",
        "embedding_key": "esmif_eos_bos_repeat",
        "pca_components": 64,
    },
    "esmif_eosrepeat": {
        "display_name": "ESM-IF1 windows (eos-repeat, 29x512)",
        "type": "embedding_windows",
        "embedding_key": "esmif_eosrepeat",
        "pca_components": 64,
    },
    # ── Reduced 29x21 windows (dim21 test; built by 11_reduce_prepared) ──
    "esmc300m_win_padtoken_d21_first": {
        "display_name": "ESM-C-300M windows (pad-token, first-21 dims, 29x21)",
        "type": "embedding_windows",
        "embedding_key": "esmc300m_win_padtoken_d21_first",
        "pca_components": 64,
    },
    "esmc300m_win_padtoken_d21_var": {
        "display_name": "ESM-C-300M windows (pad-token, var-21 dims, 29x21)",
        "type": "embedding_windows",
        "embedding_key": "esmc300m_win_padtoken_d21_var",
        "pca_components": 64,
    },
    "esmif_eosrepeat_d21_first": {
        "display_name": "ESM-IF1 windows (eos-repeat, first-21 dims, 29x21)",
        "type": "embedding_windows",
        "embedding_key": "esmif_eosrepeat_d21_first",
        "pca_components": 64,
    },
    "esmif_eosrepeat_d21_var": {
        "display_name": "ESM-IF1 windows (eos-repeat, var-21 dims, 29x21)",
        "type": "embedding_windows",
        "embedding_key": "esmif_eosrepeat_d21_var",
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
    # ── ESM-C 300M windows (D=960) ──
    "esmc300m_win_zeropad": {
        "display_name": "ESM-C-300M windows (zero-pad, 29x960)",
        "type": "embedding_windows",
        "embedding_key": "esmc300m_win_zeropad",
        "pca_components": 64,
    },
    "esmc300m_win_padtoken": {
        "display_name": "ESM-C-300M windows (pad-token, 29x960)",
        "type": "embedding_windows",
        "embedding_key": "esmc300m_win_padtoken",
        "pca_components": 64,
    },
    "esmc300m_win_impute_bos_eos": {
        "display_name": "ESM-C-300M windows (impute-bos-eos, 29x960)",
        "type": "embedding_windows",
        "embedding_key": "esmc300m_win_impute_bos_eos",
        "pca_components": 64,
    },
    "esmc300m_win_impute_boundary": {
        "display_name": "ESM-C-300M windows (impute-boundary, 29x960)",
        "type": "embedding_windows",
        "embedding_key": "esmc300m_win_impute_boundary",
        "pca_components": 64,
    },
    # ── ESM-C 6B windows (D=2560) ──
    "esmc6b_win_padtoken": {
        "display_name": "ESM-C-6B windows (pad-token, 29x2560)",
        "type": "embedding_windows",
        "embedding_key": "esmc6b_win_padtoken",
        "pca_components": 64,
    },
    # ── ProteinMPNN windows (D=128) ──
    # pca_components = default only; ALWAYS overridden by --pca N90.
    # v_48_020=1325 / v_48_002=1478 = 90% variance (08_pca_variance_analysis
    # flat-full). v_48_010 / v_48_030 PROVISIONAL until their spectra exist.
    "mpnn_v48020_zeropad": {
        "display_name": "ProteinMPNN v_48_020 windows (zero-pad, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48020_zeropad",
        "pca_components": 1325,
    },
    "mpnn_v48020_boundary": {
        "display_name": "ProteinMPNN v_48_020 windows (boundary, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48020_boundary",
        "pca_components": 1325,
    },
    "mpnn_v48010_zeropad": {
        "display_name": "ProteinMPNN v_48_010 windows (zero-pad, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48010_zeropad",
        "pca_components": 1325,
    },
    "mpnn_v48010_boundary": {
        "display_name": "ProteinMPNN v_48_010 windows (boundary, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48010_boundary",
        "pca_components": 1325,
    },
    "mpnn_v48002_zeropad": {
        "display_name": "ProteinMPNN v_48_002 windows (zero-pad, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48002_zeropad",
        "pca_components": 1478,
    },
    "mpnn_v48002_boundary": {
        "display_name": "ProteinMPNN v_48_002 windows (boundary, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48002_boundary",
        "pca_components": 1478,
    },
    "mpnn_v48030_zeropad": {
        "display_name": "ProteinMPNN v_48_030 windows (zero-pad, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48030_zeropad",
        "pca_components": 1325,
    },
    "mpnn_v48030_boundary": {
        "display_name": "ProteinMPNN v_48_030 windows (boundary, 29x128)",
        "type": "embedding_windows",
        "embedding_key": "mpnn_v48030_boundary",
        "pca_components": 1325,
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
    "sparse": {
        "display_name": "Sparse one-hot AA only",
        "components": ["sparse"],
    },
    "blosum50": {
        "display_name": "BLOSUM50 AA only",
        "components": ["blosum50"],
    },
    "blosum62": {
        "display_name": "BLOSUM62 AA only",
        "components": ["blosum62"],
    },
    "blosum80": {
        "display_name": "BLOSUM80 AA only",
        "components": ["blosum80"],
    },
    "kidera": {
        "display_name": "Kidera factors AA only",
        "components": ["kidera"],
    },
    "atchley": {
        "display_name": "Atchley factors AA only",
        "components": ["atchley"],
    },
    "compositional": {
        "display_name": "AA composition only (21 frac features)",
        "components": ["compositional"],
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
    "handcrafted_blosum50": {
        "display_name": "Structural + BLOSUM50 AA",
        "components": ["handcrafted", "blosum50"],
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
    "handcrafted_blosum50_esmif_win": {
        "display_name": "Structural + BLOSUM50 + ESM-IF windows",
        "components": ["handcrafted", "blosum50", "esmif_win"],
    },
    "handcrafted_sparse_esmif_pad": {
        "display_name": "Structural + one-hot + ESM-IF windows (pad)",
        "components": ["handcrafted", "sparse", "esmif_pad"],
    },
    "handcrafted_sparse_esmif_boundary": {
        "display_name": "Structural + one-hot + ESM-IF windows (boundary)",
        "components": ["handcrafted", "sparse", "esmif_boundary"],
    },
    "handcrafted_sparse_esmif_eos_bos_repeat": {
        "display_name": "Structural + one-hot + ESM-IF windows (eos/bos-repeat)",
        "components": ["handcrafted", "sparse", "esmif_eos_bos_repeat"],
    },
    # ── Explicit + embedding (no AA encoding) ──
    "handcrafted_esmif_eos_bos_repeat": {
        "display_name": "Structural + ESM-IF windows (eos/bos-repeat)",
        "components": ["handcrafted", "esmif_eos_bos_repeat"],
    },
    "handcrafted_esmc300m_win_padtoken": {
        "display_name": "Structural + ESM-C-300M windows (pad-token)",
        "components": ["handcrafted", "esmc300m_win_padtoken"],
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
    "handcrafted_blosum50_esmc": {
        "display_name": "Structural + BLOSUM50 + ESM-C",
        "components": ["handcrafted", "blosum50", "esmc"],
    },
    "handcrafted_blosum50_esmif": {
        "display_name": "Structural + BLOSUM50 + ESM-IF",
        "components": ["handcrafted", "blosum50", "esmif"],
    },
    # ── Combined context window ──
    "handcrafted_sparse_esmc_esmif": {
        "display_name": "Structural + one-hot + ESM-C + ESM-IF",
        "components": ["handcrafted", "sparse", "esmc", "esmif"],
    },
    # ── Embeddings-only (single component; pure-embedding comparison) ──
    "emb_esmc_win_zeropad": {
        "display_name": "ESM-C windows only (zero-pad)",
        "components": ["esmc_win_zeropad"],
    },
    "emb_esmc_win_padtoken": {
        "display_name": "ESM-C windows only (pad-token)",
        "components": ["esmc_win_padtoken"],
    },
    "emb_esmc_win_impute_bos_eos": {
        "display_name": "ESM-C windows only (impute-bos-eos)",
        "components": ["esmc_win_impute_bos_eos"],
    },
    "emb_esmc_win_impute_boundary": {
        "display_name": "ESM-C windows only (impute-boundary)",
        "components": ["esmc_win_impute_boundary"],
    },
    # ── ESM-C 300M (D=960) ──
    "emb_esmc300m_win_zeropad": {
        "display_name": "ESM-C-300M windows only (zero-pad)",
        "components": ["esmc300m_win_zeropad"],
    },
    "emb_esmc300m_win_padtoken": {
        "display_name": "ESM-C-300M windows only (pad-token)",
        "components": ["esmc300m_win_padtoken"],
    },
    "emb_esmc300m_win_impute_bos_eos": {
        "display_name": "ESM-C-300M windows only (impute-bos-eos)",
        "components": ["esmc300m_win_impute_bos_eos"],
    },
    "emb_esmc300m_win_impute_boundary": {
        "display_name": "ESM-C-300M windows only (impute-boundary)",
        "components": ["esmc300m_win_impute_boundary"],
    },
    # ── ESM-C 6B (D=2560) ──
    "emb_esmc6b_win_padtoken": {
        "display_name": "ESM-C-6B windows only (pad-token)",
        "components": ["esmc6b_win_padtoken"],
    },
    "emb_esmif_win": {
        "display_name": "ESM-IF windows only (zero)",
        "components": ["esmif_win"],
    },
    "emb_esmif_pad": {
        "display_name": "ESM-IF windows only (pad)",
        "components": ["esmif_pad"],
    },
    "emb_esmif_boundary": {
        "display_name": "ESM-IF windows only (boundary)",
        "components": ["esmif_boundary"],
    },
    "emb_esmif_eos_bos_repeat": {
        "display_name": "ESM-IF windows only (eos/bos-repeat)",
        "components": ["esmif_eos_bos_repeat"],
    },
    "emb_esmif_eosrepeat": {
        "display_name": "ESM-IF windows only (eos-repeat)",
        "components": ["esmif_eosrepeat"],
    },
    "emb_esmc300m_win_padtoken_d21_first": {
        "display_name": "ESM-C-300M windows only (pad-token, first-21 dims)",
        "components": ["esmc300m_win_padtoken_d21_first"],
    },
    "emb_esmc300m_win_padtoken_d21_var": {
        "display_name": "ESM-C-300M windows only (pad-token, var-21 dims)",
        "components": ["esmc300m_win_padtoken_d21_var"],
    },
    "emb_esmif_eosrepeat_d21_first": {
        "display_name": "ESM-IF windows only (eos-repeat, first-21 dims)",
        "components": ["esmif_eosrepeat_d21_first"],
    },
    "emb_esmif_eosrepeat_d21_var": {
        "display_name": "ESM-IF windows only (eos-repeat, var-21 dims)",
        "components": ["esmif_eosrepeat_d21_var"],
    },
    # ── ProteinMPNN windows (embedding-only) ──
    "emb_mpnn_v48020_zeropad": {
        "display_name": "ProteinMPNN v_48_020 only (zero-pad)",
        "components": ["mpnn_v48020_zeropad"],
    },
    "emb_mpnn_v48020_boundary": {
        "display_name": "ProteinMPNN v_48_020 only (boundary)",
        "components": ["mpnn_v48020_boundary"],
    },
    "emb_mpnn_v48010_zeropad": {
        "display_name": "ProteinMPNN v_48_010 only (zero-pad)",
        "components": ["mpnn_v48010_zeropad"],
    },
    "emb_mpnn_v48010_boundary": {
        "display_name": "ProteinMPNN v_48_010 only (boundary)",
        "components": ["mpnn_v48010_boundary"],
    },
    "emb_mpnn_v48002_zeropad": {
        "display_name": "ProteinMPNN v_48_002 only (zero-pad)",
        "components": ["mpnn_v48002_zeropad"],
    },
    "emb_mpnn_v48002_boundary": {
        "display_name": "ProteinMPNN v_48_002 only (boundary)",
        "components": ["mpnn_v48002_boundary"],
    },
    "emb_mpnn_v48030_zeropad": {
        "display_name": "ProteinMPNN v_48_030 only (zero-pad)",
        "components": ["mpnn_v48030_zeropad"],
    },
    "emb_mpnn_v48030_boundary": {
        "display_name": "ProteinMPNN v_48_030 only (boundary)",
        "components": ["mpnn_v48030_boundary"],
    },
}

# ── Generated AA-encoding feature sets ──
# <enc> (encoding alone), handcrafted_<enc> (structural + encoding),
# <enc>_<emb> (encoding + embedding, no structural) and
# handcrafted_<enc>_<emb> (all three).
# Existing entries are kept as-is.
_AA_ENCODINGS = ["sparse", "blosum50", "blosum62", "blosum80", "kidera", "atchley",
                 "compositional"]
_AA_DISPLAY = {
    "sparse": "one-hot", "blosum50": "BLOSUM50", "blosum62": "BLOSUM62",
    "blosum80": "BLOSUM80", "kidera": "Kidera factors", "atchley": "Atchley factors",
    "compositional": "AA composition",
}
_AA_EMB_PAIRS = [
    ("esmif_eos_bos_repeat", "ESM-IF eos/bos-repeat"),
    ("esmc300m_win_padtoken", "ESM-C-300M pad-token"),
    ("esmc6b_win_padtoken", "ESM-C-6B pad-token"),
]
for _e in _AA_ENCODINGS:
    _ed = _AA_DISPLAY[_e]
    _generated = {f"handcrafted_{_e}": (f"Structural + {_ed}", ["handcrafted", _e])}
    for _emb, _elab in _AA_EMB_PAIRS:
        _generated[f"{_e}_{_emb}"] = (
            f"{_ed} + {_elab}", [_e, _emb])
        _generated[f"handcrafted_{_e}_{_emb}"] = (
            f"Structural + {_ed} + {_elab}", ["handcrafted", _e, _emb])
    for _k, (_dn, _comps) in _generated.items():
        if _k not in FEATURE_SETS:
            FEATURE_SETS[_k] = {"display_name": _dn, "components": _comps}
del _e, _ed, _generated, _emb, _elab, _k, _dn, _comps

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

# Pre-2026-10-12 runs labelled "blosum" are BLOSUM50 (read-path only).
# New runs must use an explicit blosum50/blosum62/blosum80.
FEATURE_SET_ALIASES = {
    "blosum": "blosum50",
    "handcrafted_blosum": "handcrafted_blosum50",
}


def canonical_feature_set(features_key):
    """Map legacy feature-set names to their post-rename equivalents."""
    return FEATURE_SET_ALIASES.get(features_key, features_key)


def get_feature_cols(df_columns):
    """Return all columns that are not metadata or raw AA letters."""
    return [c for c in df_columns if c not in _EXCLUDE_COLS]


def get_model_config(model_key):
    if model_key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model key '{model_key}'. Valid keys: {valid}")
    return MODEL_REGISTRY[model_key]


def get_feature_set_config(features_key):
    if features_key in ("blosum", "handcrafted_blosum"):
        raise KeyError(
            f"Bare '{features_key}' was retired 2026-10-12; existing runs "
            f"labelled '{features_key}' are {FEATURE_SET_ALIASES[features_key]} "
            f"(canonical_feature_set() maps them on read). New runs must be "
            f"explicit: blosum50 / blosum62 / blosum80."
        )
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

    # Cannot combine multiple AA encodings in one feature set
    # (same column names, different values)
    aa_encodings = [
        c for c in components
        if c in ("sparse", "blosum50", "blosum62", "blosum80", "kidera", "atchley",
                 "compositional")
    ]
    if len(aa_encodings) > 1:
        raise ValueError(
            f"Cannot combine multiple AA encodings in one feature set: "
            f"{aa_encodings}. Use exactly one."
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
