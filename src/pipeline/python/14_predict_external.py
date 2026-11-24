"""
14_predict_external.py
Score an external-validation dataset with models trained on the legacy
HLA-A*02:01 pipeline — NO retraining.

Datasets (--dataset):
  a0101     (default) The HLA-A*01:01 external allele set
            (THESIS_ALLELE=hla_a0101 edition). Reads
            data/processed/hla_a0101/df_all.csv + the per-encoding CSVs
            produced by the external-allele R chain.
  assarsson           The Assarsson external set (06_external_eval's
            data, replayed through fold artifacts). Reads
            data/processed/df_assarsson.csv (meta) +
            data/processed/df_<enc>_assarsson.csv per CSV component.
            NOTE: the existing df_*_assarsson.csv encodings were built
            for the OLD 29-mer campaign (609 features); canonical
            flank5 19-mer models expect 399. Regenerate the encoding at
            FLANK_SIZE=5 before scoring with the h6 canonical models —
            the feature-count guard refuses mismatches.

Loading the models: --run-tag is the training run-tag prefix; ALL saved
nested-CV inner models ({run_tag}_outer{i}_inner{j}.pkl in --model-dir)
are loaded together with their preprocessing artifacts
({run_tag}_outer{i}_inner{j}_artifacts.pkl — scaler, fold PCA objects,
column medians, dimselect dims), the exact feature matrix is rebuilt from
the external data, transformed through each fold's pipeline, predicted,
and the probabilities averaged across all inner models (same ensemble
logic as 04_modelling's predict_with_averaging).

Allele switch (env, a0101 dataset only — same pattern as
THESIS_VARIANT):
  THESIS_ALLELE=hla_a0101 python 14_predict_external.py ...   # external set
  unset / hla0201                                            # refused

Prerequisites (a0101):
  - external-allele R chain done
    (Rscript src/pipeline/r/run_pipeline.R hla_a0101 →
     data/processed/hla_a0101/df_all.csv + df_<enc>.csv)
  - for embedding feature sets: raw H5s generated (GPU, legacy
    src/tools/esm|proteinmpnn/embed_peptides_*.py with --csv/--out-dir
    pointed at the external-allele data) + THESIS_ALLELE=<allele>
    03_prepare_embeddings.py run
  - trained models SAVED WITH ARTIFACTS. 04_modelling.py saves fold
    artifacts alongside each model .pkl (additive change); models
    trained before that change have no artifacts and must be re-run once
    with identical settings to regenerate them deterministically.

Usage:
    # a0101, encodings-only feature set (no GPU stage needed):
    THESIS_ALLELE=hla_a0101 python 14_predict_external.py \\
        --model-dir runs/r19_rmdup/h6/models/flat \\
        --run-tag xgb_blosum50_flat --model xgb --features blosum50

    # a0101, embedding feature set (flat PCA, 4 components):
    THESIS_ALLELE=hla_a0101 python 14_predict_external.py \\
        --model-dir <dir> --run-tag xgb_sparse_esmc_win_padtoken_pca4_flat \\
        --model xgb --features sparse_esmc_win_padtoken \\
        --pca 4 --pca-mode flat

    # Assarsson (CSV-only feature sets; encoding must match training):
    python 14_predict_external.py --dataset assarsson \\
        --model-dir <dir> --run-tag xgb_blosum50_flat \\
        --model xgb --features blosum50
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# ── Config: same folder's config.py (allele-aware via THESIS_ALLELE) ─────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config  # noqa: E402  (allele-aware edition)

# ── Re-use the 04_modelling functions (import binds OUR config via
#    sys.modules["config"], so every path lookup inside them follows the
#    THESIS_ALLELE edition) ───────────────────────────────────────────────────
_MODELLING_PATH = Path(SRC_DIR) / "04_modelling.py"
_spec = importlib.util.spec_from_file_location("modelling_pipeline",
                                               _MODELLING_PATH)
M = importlib.util.module_from_spec(_spec)
sys.modules["modelling_pipeline"] = M
_spec.loader.exec_module(M)


def parse_args():
    p = argparse.ArgumentParser(
        description="14_predict_external: score an external dataset with "
                    "trained legacy models (no retraining)")
    p.add_argument("--dataset", default="a0101",
                   choices=["a0101", "assarsson"],
                   help="External dataset to score (default: a0101, the "
                        "THESIS_ALLELE=hla_a0101 edition)")
    p.add_argument("--run-tag", required=True,
                   help="Exact run-tag prefix of the saved models, e.g. "
                        "'xgb_blosum50_flat' — files "
                        "<model-dir>/<run_tag>_outer<i>_inner<j>.pkl")
    p.add_argument("--model-dir", default=None,
                   help="Directory holding the model pkls + artifacts + "
                        f"feature_names.json (default: {config.MODEL_DIR})")
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
                   help="Output dir (a0101 default: "
                        "data/processed/hla_a0101/predictions; assarsson "
                        "default: results/external)")
    return p.parse_args()


def resolve_model_dir(arg_model_dir):
    model_dir = (Path(arg_model_dir) if arg_model_dir
                 else config.MODEL_DIR)
    if not model_dir.is_absolute():
        model_dir = config.PROJECT_ROOT / model_dir
    return model_dir


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
                        f"  04_modelling.py saves fold artifacts next to "
                        f"each model .pkl. Models trained before that "
                        f"change must be re-run once with identical "
                        f"settings to regenerate artifacts "
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


def exit_with_missing(missing):
    """Print every missing prerequisite + the command that produces it."""
    print("=" * 80)
    print("  MISSING PREREQUISITES — nothing scored")
    print("=" * 80)
    for path, fix in missing:
        print(f"  - {path}")
        print(f"    fix: {fix}")
    sys.exit(1)


def check_prereqs_a0101(feat_cfg, model_dir, run_tag):
    """All a0101-mode inputs: base CSV, component CSVs/H5s, models."""
    missing = []
    if not config.RAW_DATA_PATH.exists():
        missing.append(
            (str(config.RAW_DATA_PATH),
             "Rscript src/pipeline/r/07_alpha_fold_features.R hla_a0101 "
             "(or the full chain: Rscript src/pipeline/r/run_pipeline.R "
             "hla_a0101)"))
    csv_comps, emb_comps = M.resolve_components(feat_cfg)
    for comp in csv_comps:
        csv_path = comp.get("csv_path")
        if csv_path is not None and not Path(csv_path).exists():
            missing.append(
                (str(csv_path),
                 "Rscript src/pipeline/r/08_aa_encoding.R hla_a0101"))
    for comp in emb_comps:
        emb_key = comp["embedding_key"]
        prepared = config.get_embedding_source(emb_key)["prepared_path"]
        if not Path(prepared).exists():
            missing.append(
                (str(prepared),
                 f"THESIS_ALLELE=hla_a0101 python 03_prepare_embeddings.py "
                 f"--embedding {emb_key}"))
    fn_path = model_dir / f"{run_tag}_feature_names.json"
    if not fn_path.exists():
        missing.append(
            (str(fn_path),
             "re-run 04_modelling.py once with the training settings "
             "(deterministic) — it saves feature names next to the models"))
    try:
        discover_models(model_dir, run_tag)
    except FileNotFoundError as e:
        missing.append((str(model_dir), str(e)))
    return missing


def check_prereqs_assarsson(feat_cfg, model_dir, run_tag):
    """All assarsson-mode inputs: meta, per-component encodings, models."""
    missing = []
    meta_path = config.DATA_DIR / "df_assarsson.csv"
    if not meta_path.exists():
        missing.append(
            (str(meta_path),
             "the Assarsson meta table (06_external_eval input) is "
             "missing — check data/processed/"))
    csv_comps, emb_comps = M.resolve_components(feat_cfg)
    for comp in emb_comps:
        missing.append(
            (f"embedding component '{comp['_key']}'",
             "assarsson mode supports CSV-only feature sets (06_external_"
             "eval precedent); embedding artifacts were never built for "
             "the Assarsson chain"))
    for comp in csv_comps:
        enc_path = config.DATA_DIR / f"df_{comp['_key']}_assarsson.csv"
        if enc_path.exists():
            continue
        fix = ("regenerate with the Assarsson chain at the training-time "
               "window size (FLANK_SIZE=5 for the flank5 19-mer models; "
               "the existing file was built for the old 29-mer campaign)")
        missing.append((str(enc_path), fix))
    fn_path = model_dir / f"{run_tag}_feature_names.json"
    if not fn_path.exists():
        missing.append(
            (str(fn_path),
             "re-run 04_modelling.py once with the training settings "
             "(deterministic) — it saves feature names next to the models"))
    try:
        discover_models(model_dir, run_tag)
    except FileNotFoundError as e:
        missing.append((str(model_dir), str(e)))
    return missing


def load_data_a0101(feat_cfg, args):
    """Load + build features for the external-allele dataset (R chain)."""
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
    print(f"\nLoading feature components ({config.ALLELE} paths)...")
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
    return df, csv_feature_cols, emb_data_dict


def load_data_assarsson(feat_cfg, saved_names):
    """
    Load the Assarsson set: per-component encoding CSVs joined 1:1 on
    peptide, merged with the meta table (06_external_eval layout).
    Feature columns are taken verbatim from the training feature names,
    so the matrix matches training column-for-column.
    """
    csv_comps, emb_comps = M.resolve_components(feat_cfg)
    if emb_comps:
        sys.exit("FATAL: assarsson mode supports CSV-only feature sets "
                 "(06_external_eval precedent).")
    if not csv_comps:
        sys.exit("FATAL: feature set has no CSV components.")

    df = None
    for comp in csv_comps:
        enc_path = config.DATA_DIR / f"df_{comp['_key']}_assarsson.csv"
        print(f"\n  Loading encoding CSV: {enc_path}")
        cdf = pd.read_csv(enc_path)
        if df is None:
            df = cdf
        else:
            drop = [c for c in ("peptide", config.LABEL_COL)
                    if c in cdf.columns]
            df = df.merge(cdf.drop(columns=drop), on="peptide",
                          validate="one_to_one")

    meta_path = config.DATA_DIR / "df_assarsson.csv"
    meta_cols = ["peptide", "classification", "pep_length", "true_label",
                 "rank", "ba_rank"]
    meta = pd.read_csv(meta_path)
    absent = [c for c in meta_cols if c not in meta.columns]
    if absent:
        sys.exit(f"FATAL: meta table {meta_path} lacks columns: {absent}")
    df = df.merge(meta[meta_cols], on="peptide", how="left",
                  validate="one_to_one")
    if df["classification"].isna().any():
        sys.exit("FATAL: meta merge failed for some peptides "
                 "(classification is NaN — peptide mismatch).")

    n_pos = int((df[config.LABEL_COL] == 1).sum())
    n_neg = int((df[config.LABEL_COL] == 0).sum())
    print(f"\nAssarsson data: {len(df)} rows x {df.shape[1]} columns")
    print(f"  Positives: {n_pos}  Negatives: {n_neg}")

    csv_feature_cols = list(saved_names)
    absent_feats = [c for c in csv_feature_cols if c not in df.columns]
    if absent_feats:
        sys.exit(f"FATAL: {len(absent_feats)} training feature columns "
                 f"missing from the Assarsson encoding, e.g. "
                 f"{absent_feats[:5]}. Regenerate the encoding at the "
                 f"training-time window size (FLANK_SIZE=5 for the "
                 f"flank5 19-mer models).")
    print(f"  CSV features:    {len(csv_feature_cols)}")
    return df, csv_feature_cols, {}


def prepare_external(df, csv_feature_cols, emb_data_dict, fold_artifacts):
    """
    Build the external-set feature matrix through ONE fold's pipeline.
    Mirrors 04_modelling prepare_validation(), but scores ALL rows (the
    external set is never split).
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


# ── Metrics helpers ───────────────────────────────────────────────────────────

def confusion_block(y_true, y_prob, threshold):
    """tn/fp/fn/tp counts at a threshold (the count matrix)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def youden_threshold(y_true, y_prob):
    """ROC-optimal (max J = tpr - fpr) threshold."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def per_model_auc_stats(y_true, per_model):
    """AUC of every fold model's probability vector — ensemble spread."""
    aucs = {}
    for name, probs in per_model.items():
        try:
            aucs[name] = float(roc_auc_score(y_true, probs))
        except ValueError:
            return None      # single-class guard
    vals = np.array(list(aucs.values()))
    return {
        "n_models": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "values": {k: round(v, 4) for k, v in aucs.items()},
    }


def score_summary(y_true, y_prob):
    y_prob = np.asarray(y_prob)
    pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
    return {
        "pos_mean": float(pos.mean()), "pos_std": float(pos.std()),
        "neg_mean": float(neg.mean()), "neg_std": float(neg.std()),
    }


def print_confusion(cm, prefix="  ", label=""):
    print(f"{prefix}Count matrix{' (' + label + ')' if label else ''} "
          f"(rows=true, cols=pred):")
    print(f"{prefix}           pred=0   pred=1")
    print(f"{prefix}  true=0  {cm['tn']:>7}  {cm['fp']:>7}")
    print(f"{prefix}  true=1  {cm['fn']:>7}  {cm['tp']:>7}")


def classification_block(df, y_true, y_prob, threshold):
    """Assarsson per-classification breakdown (06_external_eval logic)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    work = pd.DataFrame({
        "classification": df["classification"].to_numpy(),
        "label": y_true, "prob": y_prob, "pred": y_pred,
    })
    out = {}
    for cls, sub in work.groupby("classification"):
        out[cls] = {
            "n": int(len(sub)),
            "mean_prob": round(float(sub["prob"].mean()), 4),
            "pos_rate": round(float(sub["label"].mean()), 4),
            "accuracy": round(float((sub["pred"] == sub["label"]).mean()), 4),
        }
    return out


def print_classification(by_class):
    print(f"\n  {'classification':<16} {'n':>4} {'mean_p':>7} "
          f"{'pos_rate':>8} {'acc@0.5':>8}")
    for cls, b in by_class.items():
        print(f"  {cls:<16} {b['n']:>4} {b['mean_prob']:>7} "
              f"{b['pos_rate']:>8} {b['accuracy']:>8}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    if args.dataset == "a0101" and config.ALLELE != "hla_a0101":
        sys.exit(
            "FATAL: THESIS_ALLELE is unset/hla0201 — the a0101 dataset "
            "is scored with THESIS_ALLELE=hla_a0101 (the external set "
            "must never be scored with models trained on it)."
        )

    config.validate_config()

    model_cfg = config.get_model_config(args.model)
    feat_cfg = config.get_feature_set_config(args.features)
    config.validate_feature_set(feat_cfg)
    model_dir = resolve_model_dir(args.model_dir)

    print("=" * 80)
    print(f"  14_PREDICT_EXTERNAL — {args.dataset.upper()} "
          f"EXTERNAL VALIDATION")
    print("=" * 80)
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Run tag:      {args.run_tag}")
    print(f"  Model dir:    {model_dir}")
    print(f"  Model:        {model_cfg.get('display_name', args.model)}")
    print(f"  Features:     {feat_cfg.get('display_name', args.features)}")
    print(f"  PCA mode:     {args.pca_mode}")
    if args.pca is not None:
        print(f"  PCA override: {args.pca}")
    if args.dims is not None:
        print(f"  Dims:         {args.dims}")
    print("=" * 80)

    # ── Prerequisites: list EVERYTHING missing, then exit once ──
    if args.dataset == "a0101":
        missing = check_prereqs_a0101(feat_cfg, model_dir, args.run_tag)
    else:
        missing = check_prereqs_assarsson(feat_cfg, model_dir, args.run_tag)
    if missing:
        exit_with_missing(missing)

    # ── Run-level feature names (training ground truth) ──
    fn_path = model_dir / f"{args.run_tag}_feature_names.json"
    with open(fn_path) as f:
        saved_names = json.load(f)
    print(f"Feature names file: {len(saved_names)} features (run-level)")

    # ── Load external data + build features ──
    if args.dataset == "a0101":
        df, csv_feature_cols, emb_data_dict = load_data_a0101(
            feat_cfg, args)
    else:
        df, csv_feature_cols, emb_data_dict = load_data_assarsson(
            feat_cfg, saved_names)

    # ── Discover models + artifacts ──
    pairs = discover_models(model_dir, args.run_tag)
    print(f"\nModels found: {len(pairs)} "
          f"(outer 0..{pairs[-1][0]}, inner 0..{pairs[-1][1]})")

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

        if len(saved_names) != X.shape[1]:
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

    # ── Metrics: scores + count matrix + Youden + summaries ──
    THR = 0.5
    metrics = M.compute_metrics(y_true, avg_probs, threshold=THR)
    cm = confusion_block(y_true, avg_probs, THR)
    j_thr = youden_threshold(y_true, avg_probs)
    j_metrics = M.compute_metrics(y_true, avg_probs, threshold=j_thr)
    j_cm = confusion_block(y_true, avg_probs, j_thr)
    spread = per_model_auc_stats(y_true, per_model)
    ssum = score_summary(y_true, avg_probs)

    print(f"\n{'=' * 80}")
    print(f"EXTERNAL VALIDATION ({args.dataset}) — ensemble of "
          f"{len(all_probs)} inner models")
    print("=" * 80)
    print(f"  Rows: {len(y_true)}  (pos={int(y_true.sum())}, "
          f"neg={int(len(y_true) - y_true.sum())})")
    print(f"\n  Ensemble metrics (threshold {THR}):")
    M.print_metrics(metrics, prefix="  ")
    print_confusion(cm, prefix="  ", label=f"threshold {THR}")

    print(f"\n  Youden-optimal threshold: {j_thr:.4f}")
    print(f"  AUROC {j_metrics['auc_roc']:.4f}  |  Acc "
          f"{j_metrics['accuracy']:.4f}  |  MCC {j_metrics['mcc']:.4f}  |  "
          f"F1 {j_metrics['f1']:.4f}")
    print_confusion(j_cm, prefix="  ", label="Youden")

    print(f"\n  Score summary (ensemble prob_avg):")
    print(f"    true pos: mean={ssum['pos_mean']:.3f} "
          f"(std {ssum['pos_std']:.3f})")
    print(f"    true neg: mean={ssum['neg_mean']:.3f} "
          f"(std {ssum['neg_std']:.3f})")
    if spread is not None:
        print(f"\n  Per-fold-model AUC: mean={spread['mean']:.4f} "
              f"± {spread['std']:.4f}  (min {spread['min']:.4f}, "
              f"max {spread['max']:.4f}, n={spread['n_models']})")

    # ── Assarsson-only blocks (06_external_eval parity) ──
    niner = None
    by_class = None
    if args.dataset == "assarsson":
        mask9 = df["pep_length"] == 9
        if mask9.any():
            niner = {
                "n": int(mask9.sum()),
                "metrics": M.compute_metrics(
                    y_true[mask9.values], avg_probs[mask9.values],
                    threshold=THR),
                "confusion": confusion_block(
                    y_true[mask9.values], avg_probs[mask9.values], THR),
            }
            print(f"\n  9-mer-only ({niner['n']} rows): "
                  f"AUROC={niner['metrics']['auc_roc']:.4f}  "
                  f"AUPRC={niner['metrics']['auc_pr']:.4f}")
        by_class = classification_block(df, y_true, avg_probs, THR)
        print_classification(by_class)

    # ── Save ──
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = config.PROJECT_ROOT / out_dir
    elif args.dataset == "a0101":
        out_dir = config.ALLELE_DATA_DIR / "predictions"
    else:
        out_dir = config.PROJECT_ROOT / "results" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dataset == "a0101":
        pred_path = out_dir / f"{args.run_tag}_external_predictions.csv"
        out = df[[c for c in ("peptide", "uniprot_id", "start", "end",
                               config.LABEL_COL) if c in df.columns]].copy()
    else:
        # timestamped: never clobbers 06_external_eval's legacy outputs
        pred_path = out_dir / (
            f"assarsson_{args.run_tag}_predictions_{timestamp}.csv")
        out = df[["peptide", "classification", "pep_length",
                  config.LABEL_COL, "true_label", "rank", "ba_rank"]].copy()
    out["prob_avg"] = avg_probs
    for name, probs in per_model.items():
        out[name] = probs
    out.to_csv(pred_path, index=False)
    print(f"\nPredictions saved: {pred_path}")

    metrics_path = (out_dir / f"{args.run_tag}_external_metrics.json"
                    if args.dataset == "a0101" else
                    out_dir / f"assarsson_{args.run_tag}_metrics_"
                              f"{timestamp}.json")
    payload = {
        "dataset": args.dataset,
        "edition": config.ALLELE,
        "run_tag": args.run_tag,
        "model_dir": str(model_dir),
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
        "confusion": cm,
        "youden": {
            "threshold": j_thr,
            "metrics": j_metrics,
            "confusion": j_cm,
        },
        "score_summary": ssum,
        "per_model_auc": spread,
        "data": (str(config.RAW_DATA_PATH)
                 if args.dataset == "a0101"
                 else str(config.DATA_DIR / "df_assarsson.csv")),
    }
    if niner is not None:
        payload["9mer_only"] = niner
    if by_class is not None:
        payload["by_classification"] = by_class
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Metrics saved:     {metrics_path}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
