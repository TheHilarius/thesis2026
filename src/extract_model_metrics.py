#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime


METRIC_NAMES = [
    "auc_roc",
    "auc_pr",
    "accuracy",
    "f1",
    "mcc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
]


def parse_timestamp(ts):
    # Expected formats: 20260506_151424 or 2026-05-06 15:14:00
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass
    return None


def infer_embedding_mode(results):
    feature_set = results.get("features_key", "")
    n_features = int(results.get("config", {}).get("n_features", -1))

    if feature_set in ("handcrafted_sparse", "handcrafted_blosum"):
        return "none"

    if feature_set == "esmc":
        return "esmc_pca" if n_features == 150 else "esmc_full"

    if feature_set == "esmif":
        return "esmif_pca" if n_features == 150 else "esmif_full"

    if "esmc" in feature_set and "esmif" not in feature_set:
        return "esmc_pca" if n_features == 781 else "esmc_full"

    if "esmif" in feature_set and "esmc" not in feature_set:
        return "esmif_pca" if n_features == 781 else "esmif_full"

    if feature_set in ("all_sparse", "all_blosum"):
        return "all_pca" if n_features == 931 else "all_full"

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs="+",
        default=["models/cv_results_*.json"],
        help="Result JSON paths or globs",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include JSONs >= this timestamp (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--out",
        default="model_metrics_current.tsv",
        help="Output TSV file",
    )
    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S")

    # Expand globs
    paths = []
    for pat in args.results:
        matches = sorted(Path(".").glob(pat))
        paths.extend(matches)

    rows = []

    for path in sorted(set(paths)):
        with open(path) as f:
            r = json.load(f)

        ts_raw = r.get("timestamp", "")
        ts_dt = parse_timestamp(ts_raw)

        if since and (ts_dt is None or ts_dt < since):
            continue

        cfg = r.get("config", {})
        cv = r.get("cv_summary", {})
        ho = r.get("held_out_metrics", {})

        row = {
            "result_file": str(path),
            "timestamp": ts_raw,
            "model_key": r.get("model_key", ""),
            "model_class": r.get("model_class", ""),
            "feature_set": r.get("features_key", ""),
            "components": r.get("components", ""),
            "embedding_mode": infer_embedding_mode(r),
            "features": cfg.get("n_features", ""),
            "cv_folds": cfg.get("n_cv_folds", ""),
            "best_fold": r.get("best_fold_id", ""),
            "best_fold_auc": r.get("best_fold_auc", ""),
            "has_heldout_predictions": "held_out_predictions" in r,
        }

        for m in METRIC_NAMES:
            stats = cv.get(m, {})
            row[f"cv_mean_{m}"] = stats.get("mean", "")
            row[f"cv_std_{m}"] = stats.get("std", "")

        for m in METRIC_NAMES:
            row[f"heldout_{m}"] = ho.get(m, "")

        rows.append(row)

    rows.sort(
        key=lambda x: (
            float(x["heldout_auc_roc"]) if x["heldout_auc_roc"] != "" else -999,
            float(x["heldout_mcc"]) if x["heldout_mcc"] != "" else -999,
        ),
        reverse=True,
    )

    fields = [
        "result_file",
        "timestamp",
        "model_key",
        "model_class",
        "feature_set",
        "components",
        "embedding_mode",
        "features",
        "cv_folds",
        "best_fold",
        "best_fold_auc",
        "has_heldout_predictions",
    ]

    for m in METRIC_NAMES:
        fields += [f"cv_mean_{m}", f"cv_std_{m}"]

    for m in METRIC_NAMES:
        fields.append(f"heldout_{m}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
