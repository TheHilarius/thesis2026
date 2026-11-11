#!/usr/bin/env python3
"""Score a trained run against the Assarsson external-validation set.

Loads a run's pickles + feature_names, averages predict_proba across all
outer/inner models (mirrors 04_modelling.predict_with_averaging), and
reports true_label vs pred_label metrics.

Supported for v1: CSV-only, non-scaled feature sets (sparse/blosum50/...,
models rf and xgb). Embedding runs (PCA artifacts) and scaling runs
(StandardScaler) were never persisted by 04_modelling.py and are refused.

Usage:
  python3 src/pipeline/python/06_external_eval.py \
      --run-dir runs/encodings_sweep/models/flat \
      --run-tag xgb_blosum50_flat
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FEATURES_CSV = ROOT / "data/processed/df_blosum50_assarsson.csv"
META_CSV = ROOT / "data/processed/df_assarsson.csv"
THRESHOLD = 0.5


def model_key_for(run_tag: str) -> str:
    """Longest MODEL_REGISTRY key that prefixes the run tag (xgb, lr_l2, ...)."""
    for key in sorted(MODEL_REGISTRY, key=len, reverse=True):
        if run_tag.startswith(key + "_"):
            return key
    return ""


def metrics_block(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "threshold": thr,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="dir with *.pkl + feature_names json")
    ap.add_argument("--run-tag", required=True, help="e.g. xgb_blosum50_flat")
    ap.add_argument("--features", default=str(FEATURES_CSV))
    ap.add_argument("--meta", default=str(META_CSV))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out-dir", default=str(ROOT / "results/external"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    # ── Refusals: artifacts 04_modelling.py never persisted ──
    key = model_key_for(args.run_tag)
    if key and MODEL_REGISTRY[key]["needs_scaling"]:
        sys.exit(
            f"ERROR: '{key}' needs_scaling=True, but StandardScaler fit on "
            "training data was never persisted — external scores would be "
            "invalid. Use an rf/xgb run, or patch 04_modelling.py to dump "
            "fold_artifacts and retrain."
        )
    if not key:
        print(f"WARNING: could not map '{args.run_tag}' to MODEL_REGISTRY; "
              "assuming no scaling (verify this run used a scaled model!).")

    feat_names_path = run_dir / f"{args.run_tag}_feature_names.json"
    if not feat_names_path.exists():
        sys.exit(f"ERROR: feature names not found: {feat_names_path}")
    feature_names = json.loads(feat_names_path.read_text())

    pkls = sorted(glob.glob(str(run_dir / f"{args.run_tag}_outer*_inner*.pkl")))
    if not pkls:
        sys.exit(f"ERROR: no pickles matching {args.run_tag}_outer*_inner*.pkl in {run_dir}")
    print(f"Run:      {args.run_tag}")
    print(f"Model:    {key or 'unknown'}")
    print(f"Pickles:  {len(pkls)} (averaging predict_proba)")
    print(f"Features: {len(feature_names)}")

    # ── Load data ──
    df = pd.read_csv(args.features)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {len(missing)} feature columns missing from "
                 f"{args.features}, e.g. {missing[:5]}")
    extra_emb = [c for c in feature_names if c.startswith(("esmc_", "esmif_", "mpnn_"))]
    if extra_emb:
        sys.exit("ERROR: run uses embedding features — window PCA artifacts "
                 "were never persisted; v1 eval supports CSV-only runs.")

    meta = pd.read_csv(args.meta)
    df = df.merge(
        meta[["peptide", "classification", "pep_length", "true_label", "rank", "ba_rank"]],
        on="peptide", how="left", validate="one_to_one",
    )
    if df["classification"].isna().any():
        sys.exit("ERROR: meta merge failed for some peptides")

    X = df[feature_names].to_numpy(dtype=np.float64)
    y_true = df["label"].to_numpy(dtype=np.int32)
    if np.isnan(X).any():
        print(f"WARNING: {np.isnan(X).sum()} NaN feature values — models see them "
              "as missing (xgb) or they will error (rf).")

    # ── Average predictions across all inner models ──
    probs = []
    for p in pkls:
        with open(p, "rb") as f:
            model = pickle.load(f)
        probs.append(model.predict_proba(X)[:, 1])
    y_prob = np.mean(probs, axis=0)

    # ── Metrics ──
    m_all = metrics_block(y_true, y_prob, args.threshold)
    m_9mer = metrics_block(
        y_true[df["pep_length"] == 9], y_prob[df["pep_length"] == 9], args.threshold
    )

    out = {
        "run_tag": args.run_tag,
        "run_dir": str(run_dir),
        "n_pickles": len(pkls),
        "all_rows": m_all,
        "9mer_only": m_9mer,
        "by_classification": {},
    }
    print("\n=== AUROC / AUPRC ===")
    print(f"  all ({m_all['n']}): AUROC={m_all['auroc']}  AUPRC={m_all['auprc']}  "
          f"acc={m_all['accuracy']}  conf={m_all['confusion']}")
    print(f"  9mer ({m_9mer['n']}): AUROC={m_9mer['auroc']}  AUPRC={m_9mer['auprc']}  "
          f"acc={m_9mer['accuracy']}  conf={m_9mer['confusion']}")

    print("\n=== Per classification (paper group) ===")
    print(f"  {'classification':<16} {'n':>4} {'mean_p':>7} {'pos_rate':>8} {'acc@thr':>8}")
    y_pred = (y_prob >= args.threshold).astype(int)
    work = pd.DataFrame(
        {"classification": df["classification"].to_numpy(),
         "label": y_true, "prob": y_prob, "pred": y_pred}
    )
    for cls, sub in work.groupby("classification"):
        block = {
            "n": int(len(sub)),
            "mean_prob": round(float(sub["prob"].mean()), 4),
            "pos_rate": round(float(sub["label"].mean()), 4),
            "accuracy": round(float((sub["pred"] == sub["label"]).mean()), 4),
        }
        out["by_classification"][cls] = block
        print(f"  {cls:<16} {block['n']:>4} {block['mean_prob']:>7} "
              f"{block['pos_rate']:>8} {block['accuracy']:>8}")

    # ── Write outputs ──
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = df[["peptide", "classification", "pep_length", "label",
                  "true_label", "rank", "ba_rank"]].copy()
    pred_df["prob"] = y_prob
    pred_df["pred_label"] = np.where(y_pred == 1, "positive", "negative")
    pred_path = out_dir / f"assarsson_{args.run_tag}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = out_dir / f"assarsson_{args.run_tag}_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2))

    print(f"\nSaved: {pred_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
