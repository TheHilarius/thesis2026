"""
14_predict_external.py (a0101)
Score the HLA-A*01:01 external-validation set with models trained on the
legacy HLA-A*02:01 pipeline — NO retraining.

Loads the saved nested-CV inner models ({run_tag}_outer{i}_inner{j}.pkl in
models/) together with their preprocessing artifacts
({run_tag}_outer{i}_inner{j}_artifacts.pkl — scaler, fold PCA objects,
column medians, dimselect dims), rebuilds the exact feature matrix from the
a0101 data (encoded CSVs and/or prepared embedding H5s), transforms it
through each fold's pipeline, predicts, and averages the probabilities
across all inner models (same ensemble logic as 04_modelling's
predict_with_averaging).

Prerequisites:
  - a0101 R chain done (data/processed/hla_a0101/df_all.csv + df_<enc>.csv)
  - for embedding feature sets: raw H5s generated (GPU) + this folder's
    03_prepare_embeddings.py run for the a0101 data
  - trained models SAVED WITH ARTIFACTS. Legacy 04_modelling.py now saves
    fold artifacts alongside each model .pkl (additive change); models
    trained before that change have no artifacts and must be re-run once
    with identical settings to regenerate them deterministically.

Usage:
    # encodings-only feature set (no GPU stage needed):
    python 14_predict_external.py --run-tag xgb_blosum50 --model xgb --features blosum50

    # embedding feature set (flat PCA, 4 components):
    python 14_predict_external.py --run-tag xgb_sparse_esmc_win_padtoken_pca4_flat \\
        --model xgb --features sparse_esmc_win_padtoken --pca 4 --pca-mode flat

    # simple21 dim-gather run:
    python 14_predict_external.py --run-tag <tag> --model xgb \\
        --features <set> --pca-mode simple21 --dims 0,1,2,20
"""

import sys
import os
import argparse
import importlib.util
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── A0101 config (this folder's config.py — a0101 paths, legacy logic) ──────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config  # noqa: E402  (a0101 edition)

# ── Re-use the legacy 04_modelling functions (import binds OUR config via
#    sys.modules["config"], so every path lookup inside them is a0101) ────────
LEGACY_MODELLING = (
    config.PROJECT_ROOT / "src" / "pipeline" / "python" / "04_modelling.py"
)
_spec = importlib.util.spec_from_file_location("modelling_legacy",
                                               LEGACY_MODELLING)
M = importlib.util.module_from_spec(_spec)
sys.modules["modelling_legacy"] = M
_spec.loader.exec_module(M)


def parse_args():
    p = argparse.ArgumentParser(
        description="14_predict_external (a0101): score the HLA-A*01:01 "
                    "external set with trained legacy models (no retraining)")
    p.add_argument("--run-tag", required=True,
                   help="Exact run-tag prefix of the saved models, e.g. "
                        "'xgb_handcrafted' — files "
                        "models/<run_tag>_outer<i>_inner<j>.pkl")
    p.add_argument("--model", default=config.DEFAULT_MODEL,
                   help=f"Model key (default: {config.DEFAULT_MODEL})")
    p.add_argument("--features", default=config.DEFAULT_FEATURE_SET,
                   help=f"Feature-set key (default: {config.DEFAULT_FEATURE_SET})")
    p.add_argument("--pca-mode", default="flat",
                   choices=["flat", "flat_raw", "pool", "simple21",
                            "vardims", "dimselect", "varfold",
                            "autoencoder"],
                   help="PCA mode used at TRAINING time (must match; "
                        "default: flat)")
    p.add_argument("--pca", default=None,
                   help="PCA component override used at training time "
                        "(int, or per-component JSON) — must match")
    p.add_argument("--dims", default=None,
                   help="Comma-separated dim gather used at training time "
                        "(simple21/vardims) — must match")
    p.add_argument("--pool", default=None,
                   help="Pooling mode used at training time (pool mode)")
    p.add_argument("--pool-mask", default="masked",
                   help="Pool mask mode (default: masked)")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: "
                        "data/processed/hla_a0101/predictions)")
    return p.parse_args()


def discover_models(model_dir, run_tag):
    """Find (outer, inner) model + artifact file pairs for a run-tag."""
    pairs = []
    outer = 0
    while True:
        found_any = False
        inner = 0
        while True:
            m = model_dir / f"{run_tag}_outer{outer}_inner{inner}.pkl"
            a = model_dir / f"{run_tag}_outer{outer}_inner{inner}_artifacts.pkl"
            if m.exists():
                found_any = True
                if not a.exists():
                    raise FileNotFoundError(
                        f"Model {m.name} has no artifacts file.\n"
                        f"  Expected: {a.name}\n"
                        f"  The legacy 04_modelling.py now saves fold "
                        f"artifacts next to each model .pkl. Models trained "
                        f"before that change must be re-run once with "
                        f"identical settings to regenerate artifacts "
                        f"(deterministic: same data, folds, seed)."
                    )
                pairs.append((outer, inner, m, a))
                inner += 1
            else:
                break
        if not found_any:
            break
        outer += 1
    if not pairs:
        raise FileNotFoundError(
            f"No models found for run-tag '{run_tag}' in {model_dir} "
            f"(pattern: {run_tag}_outer<i>_inner<j>.pkl)"
        )
    return pairs


def prepare_external(df, csv_feature_cols, emb_data_dict, fold_artifacts):
    """
    Build the external-set feature matrix through ONE fold's pipeline.
    Mirrors legacy prepare_validation(), but scores ALL rows (the external
    set is never split).
    """
    indices = np.arange(len(df))
    parts = []

    # ── CSV features ──
    if csv_feature_cols:
        X_csv = df.iloc[indices][csv_feature_cols].values.astype(np.float64)
        if fold_artifacts["col_medians"] is not None:
            X_csv = M.impute_nan_single(X_csv, fold_artifacts["col_medians"])
        else:
            X_csv[np.isinf(X_csv)] = np.nan
        parts.append(X_csv)

    # ── Embedding features ──
    for comp_key, emb_data in emb_data_dict.items():
        pca_obj = fold_artifacts["pca_dict"][comp_key]
        sel_dims = (fold_artifacts.get("dimselect_dims") or {}).get(comp_key)
        X_emb = M._transform_embeddings(emb_data, indices, pca_obj,
                                        sel_dims=sel_dims)
        parts.append(X_emb)

    X = np.concatenate(parts, axis=1)

    if fold_artifacts["scaler"] is not None:
        X = fold_artifacts["scaler"].transform(X)

    return X


def predict_one(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    args = parse_args()
    t0 = time.time()

    config.validate_config()

    model_cfg = config.get_model_config(args.model)
    feat_cfg = config.get_feature_set_config(args.features)
    config.validate_feature_set(feat_cfg)

    print("=" * 80)
    print("  14_PREDICT_EXTERNAL — HLA-A*01:01 EXTERNAL VALIDATION")
    print("=" * 80)
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Run tag:      {args.run_tag}")
    print(f"  Model:        {model_cfg.get('display_name', args.model)}")
    print(f"  Features:     {feat_cfg.get('display_name', args.features)}")
    print(f"  PCA mode:     {args.pca_mode}")
    if args.pca is not None:
        print(f"  PCA override: {args.pca}")
    if args.dims is not None:
        print(f"  Dims:         {args.dims}")
    print(f"  External set: {config.RAW_DATA_PATH}")
    print("=" * 80)

    # ── Load external data ──
    if not config.RAW_DATA_PATH.exists():
        sys.exit(f"FATAL: a0101 dataset not found: {config.RAW_DATA_PATH}\n"
                 f"Run the a0101 R chain first "
                 f"(src/pipeline_a0101/r/run_pipeline.R).")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"\nExternal data: {df.shape[0]} rows x {df.shape[1]} columns")
    n_pos = int((df[config.LABEL_COL] == 1).sum())
    n_neg = int((df[config.LABEL_COL] == 0).sum())
    print(f"  Positives: {n_pos}  Negatives: {n_neg}")

    # Per-peptide pad-count metadata (same as 04_modelling main)
    if "start" in df.columns and "n_pad" not in df.columns:
        df["n_pad"] = 10 - np.minimum(10, df["start"].astype(int) - 1)
    if ("protein_length" in df.columns and "end" in df.columns
            and "c_pad" not in df.columns):
        df["c_pad"] = 10 - np.minimum(
            10, df["protein_length"].astype(int) - df["end"].astype(int))

    # External set has no folds — dummy column for schema compatibility
    if config.FOLD_COL not in df.columns:
        df[config.FOLD_COL] = 0

    # ── Build features exactly as training did ──
    print(f"\nLoading feature components (a0101 paths)...")
    pca_override = M.parse_pca_value(args.pca) if args.pca is not None else None
    dims = M.parse_dims_value(args.dims) if args.dims is not None else None

    df, csv_feature_cols, emb_data_dict, component_info = M.load_all_components(
        df, feat_cfg, args.pca_mode, pool=args.pool, pool_mask=args.pool_mask,
    )

    # Dims override (simple21/vardims) — must match training
    if dims is not None:
        for comp_key, emb_data in emb_data_dict.items():
            if emb_data.get("kind") != "windows":
                continue
            emb_data["dims"] = list(dims)

    # PCA override — must match training
    if pca_override is not None:
        for comp_key, emb_data in emb_data_dict.items():
            if isinstance(pca_override, dict):
                if comp_key in pca_override:
                    emb_data["pca_components"] = pca_override[comp_key]
            else:
                emb_data["pca_components"] = pca_override

    print(f"\n  CSV features:    {len(csv_feature_cols)}")
    print(f"  Embedding comps: {len(emb_data_dict)}")

    # ── Discover models + artifacts ──
    model_dir = config.MODEL_DIR
    pairs = discover_models(model_dir, args.run_tag)
    print(f"\nModels found: {len(pairs)} "
          f"(outer 0..{pairs[-1][0]}, inner 0..{pairs[-1][1]})")

    # Feature-name sanity check against the saved run-level file
    fn_path = model_dir / f"{args.run_tag}_feature_names.json"
    if fn_path.exists():
        with open(fn_path) as f:
            saved_names = json.load(f)
        print(f"Feature names file: {len(saved_names)} features (run-level)")
    else:
        saved_names = None
        print("Feature names file: not found (skipping name check)")

    # ── Predict with each inner model, average ──
    y_true = df[config.LABEL_COL].values.astype(np.int32)
    all_probs = []
    per_model = {}

    for outer, inner, m_path, a_path in pairs:
        with open(m_path, "rb") as f:
            model = pickle.load(f)
        with open(a_path, "rb") as f:
            artifacts = pickle.load(f)

        X = prepare_external(df, csv_feature_cols, emb_data_dict, artifacts)

        if saved_names is not None and len(saved_names) != X.shape[1]:
            raise ValueError(
                f"Feature count mismatch for {m_path.name}: built "
                f"{X.shape[1]} features, training saved "
                f"{len(saved_names)}. Check --pca-mode/--pca/--dims/--pool "
                f"match the training run."
            )

        probs = predict_one(model, X)
        all_probs.append(probs)
        per_model[f"prob_outer{outer}_inner{inner}"] = probs
        print(f"  [{outer}/{inner}] predicted {len(probs)} rows "
              f"(pos_frac={probs.mean():.3f})")

    avg_probs = np.mean(all_probs, axis=0)

    # ── Metrics ──
    metrics = M.compute_metrics(y_true, avg_probs)
    print(f"\n{'=' * 80}")
    print("EXTERNAL VALIDATION (HLA-A*01:01) — ensemble of "
          f"{len(all_probs)} inner models")
    print("=" * 80)
    M.print_metrics(metrics, prefix="  ")

    # ── Save ──
    out_dir = (Path(args.out_dir) if args.out_dir
               else config.DATA_DIR / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_path = out_dir / f"{args.run_tag}_external_predictions.csv"
    out = df[[c for c in ("peptide", "uniprot_id", "start", "end",
                          config.LABEL_COL) if c in df.columns]].copy()
    out["prob_avg"] = avg_probs
    for name, probs in per_model.items():
        out[name] = probs
    out.to_csv(pred_path, index=False)
    print(f"\nPredictions saved: {pred_path}")

    metrics_path = out_dir / f"{args.run_tag}_external_metrics.json"
    payload = {
        "edition": "hla_a0101",
        "run_tag": args.run_tag,
        "model": args.model,
        "features": args.features,
        "pca_mode": args.pca_mode,
        "pca": args.pca,
        "dims": args.dims,
        "pool": args.pool,
        "n_models": len(all_probs),
        "n_rows": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "timestamp": timestamp,
        "metrics": metrics,
        "data": str(config.RAW_DATA_PATH),
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Metrics saved:     {metrics_path}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
