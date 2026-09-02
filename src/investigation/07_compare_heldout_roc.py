#!/usr/bin/env python3
"""
07_compare_heldout_roc.py

Ad hoc ROC comparison across selected cv_results_*.json files.

Use:
  --split heldout  requires held_out_predictions in JSON
  --split cv       uses fold_predictions in JSON

Examples:
  python src/pipeline/python/07_compare_heldout_roc.py \
    --split heldout \
    --split-by-model \
    --out-prefix results/figures/models/heldout_roc_selected \
    --results models/cv_results_lr_handcrafted_blosum_esmc_*.json models/cv_results_rf_handcrafted_blosum_esmc_*.json

  python src/pipeline/python/07_compare_heldout_roc.py \
    --split cv \
    --out results/figures/models/cv_roc_selected.png \
    --results models/cv_results_lr_handcrafted_blosum_esmc_*.json models/cv_results_rf_handcrafted_blosum_esmc_*.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def infer_embedding_mode(results):
    feature_set = results.get("features_key", "unknown")
    n_features = int(results.get("config", {}).get("n_features", -1))

    if feature_set in ("handcrafted_sparse", "handcrafted_blosum"):
        return "no emb"

    if feature_set == "esmc":
        return "ESM-C PCA" if n_features == 150 else "ESM-C full"

    if feature_set == "esmif":
        return "ESM-IF PCA" if n_features == 150 else "ESM-IF full"

    if "esmc" in feature_set and "esmif" not in feature_set:
        return "ESM-C PCA" if n_features == 781 else "ESM-C full"

    if "esmif" in feature_set and "esmc" not in feature_set:
        return "ESM-IF PCA" if n_features == 781 else "ESM-IF full"

    if feature_set in ("all_sparse", "all_blosum"):
        return "all PCA" if n_features == 931 else "all full"

    return f"{n_features} feats"


def get_predictions(results, split):
    if split == "heldout":
        if "held_out_predictions" not in results:
            raise ValueError("missing held_out_predictions")
        pred = results["held_out_predictions"]
        return np.asarray(pred["y_true"]), np.asarray(pred["y_prob"])

    if split == "cv":
        if "fold_predictions" not in results:
            raise ValueError("missing fold_predictions")
        y_true = []
        y_prob = []
        for fold_id, pred in sorted(results["fold_predictions"].items(), key=lambda x: int(x[0])):
            y_true.extend(pred["y_true"])
            y_prob.extend(pred["y_prob"])
        return np.asarray(y_true), np.asarray(y_prob)

    raise ValueError(f"Unknown split: {split}")


def label_for(results, split, curve_auc):
    model_key = results.get("model_key", "model").upper()
    feature_set = results.get("features_key", "features")
    mode = infer_embedding_mode(results)

    ho = results.get("held_out_metrics", {})
    cv = results.get("cv_summary", {})

    if split == "heldout":
        auc_text = f"val AUC={ho.get('auc_roc', curve_auc):.3f}"
        mcc_text = f"MCC={ho.get('mcc', float('nan')):.3f}"
    else:
        cv_auc = cv.get("auc_roc", {}).get("mean", curve_auc)
        cv_std = cv.get("auc_roc", {}).get("std", 0.0)
        auc_text = f"CV AUC={cv_auc:.3f}+/-{cv_std:.3f}"
        mcc_text = f"val AUC={ho.get('auc_roc', float('nan')):.3f}"

    return f"{model_key} {feature_set} [{mode}] ({auc_text}, {mcc_text})"


def expand_paths(patterns):
    paths = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(matches)
        elif Path(pat).exists():
            paths.append(pat)
        else:
            print(f"WARNING: no match for {pat}")
    # de-duplicate while preserving sorted-ish order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(Path(p))
    return out


def select_latest_per_model_feature(paths):
    """
    If multiple timestamps exist for same model_key + feature_set + n_features,
    keep only the newest timestamp.
    """
    best = {}
    for path in paths:
        r = load_json(path)
        key = (
            r.get("model_key", ""),
            r.get("features_key", ""),
            int(r.get("config", {}).get("n_features", -1)),
        )
        ts = r.get("timestamp", "")
        if key not in best or ts > best[key][0]:
            best[key] = (ts, path)
    return [v[1] for v in best.values()]


def plot_group(items, out_path, split, title):
    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    cmap = plt.get_cmap("tab10")
    plotted = 0

    for i, (path, results) in enumerate(items):
        try:
            y_true, y_prob = get_predictions(results, split)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        curve_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            lw=2.0,
            alpha=0.90,
            color=cmap(i % 10),
            label=label_for(results, split, curve_auc),
        )
        plotted += 1

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Plotted {plotted} curve(s).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare held-out or CV ROC curves across selected model JSONs."
    )
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--split", choices=["heldout", "cv"], default="heldout")
    parser.add_argument("--split-by-model", action="store_true")
    parser.add_argument("--model", choices=["lr", "rf"], default=None)
    parser.add_argument("--out", default=None, help="Output PNG for one combined plot.")
    parser.add_argument("--out-prefix", default=None, help="Prefix when using --split-by-model.")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--latest-per-setting",
        action="store_true",
        help="Keep only latest JSON per model_key + feature_set + n_features.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    paths = expand_paths(args.results)

    if args.latest_per_setting:
        paths = select_latest_per_model_feature(paths)

    loaded = []
    for path in paths:
        r = load_json(path)
        if args.model is not None and r.get("model_key") != args.model:
            continue
        loaded.append((path, r))

    loaded.sort(
        key=lambda x: (
            x[1].get("model_key", ""),
            x[1].get("features_key", ""),
            int(x[1].get("config", {}).get("n_features", -1)),
            x[1].get("timestamp", ""),
        )
    )

    if not loaded:
        raise SystemExit("No result files loaded after filtering.")

    if args.split_by_model:
        if args.out_prefix is None:
            raise SystemExit("--out-prefix is required with --split-by-model")

        for model_key in ["lr", "rf"]:
            group = [(p, r) for p, r in loaded if r.get("model_key") == model_key]
            if not group:
                continue

            title = args.title or f"{args.split.upper()} ROC comparison: {model_key.upper()}"
            out = f"{args.out_prefix}_{model_key}.png"
            plot_group(group, out, args.split, title)

    else:
        if args.out is None:
            raise SystemExit("--out is required unless using --split-by-model")

        title = args.title or f"{args.split.upper()} ROC comparison"
        plot_group(loaded, args.out, args.split, title)


if __name__ == "__main__":
    main()
