#!/usr/bin/env python3
"""
Extract PCA sweep results from cv_results_*.json files.

Outputs:
  results/tables/pca_sweep_clean.tsv             — all combos, sorted by model/features/PCA
  results/tables/pca_sweep_best_by_model_feature.tsv — best PCA per model+features
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

MODELS_DIR = Path("models")
OUT_DIR = Path("results/tables")
OUT_ALL = OUT_DIR / "pca_sweep_clean.tsv"
OUT_BEST = OUT_DIR / "pca_sweep_best_by_model_feature.tsv"

PCA_RE = re.compile(r"_pca(\d+)_\d{8}_\d{6}\.json$")


def get_cv_auc(summary: dict) -> tuple[float | None, float | None]:
    auc = summary.get("auc_roc", {})
    return auc.get("mean"), auc.get("std")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for fp in sorted(MODELS_DIR.glob("cv_results_*.json")):
        m = PCA_RE.search(fp.name)
        if not m:
            continue
        pca = int(m.group(1))

        with fp.open() as f:
            d = json.load(f)

        cv_summary = d.get("cv_summary", {})
        cv_auc_mean, cv_auc_std = get_cv_auc(cv_summary)

        features = d.get("features_key", "")
        if "esmc" in features and "esmif" in features:
            embedding = "combined"
        elif "esmc" in features:
            embedding = "esmc"
        elif "esmif" in features:
            embedding = "esmif"
        else:
            embedding = "other"

        rows.append({
            "model": d.get("model_key", ""),
            "features": features,
            "embedding": embedding,
            "pca": pca,
            "cv_auc_roc_mean": cv_auc_mean,
            "cv_auc_roc_std": cv_auc_std,
            "cv_mcc_mean": cv_summary.get("mcc", {}).get("mean"),
            "cv_f1_mean": cv_summary.get("f1", {}).get("mean"),
            "cv_accuracy_mean": cv_summary.get("accuracy", {}).get("mean"),
            "result_file": str(fp),
        })

    rows.sort(key=lambda r: (r["model"], r["features"], r["pca"]))

    # All rows
    if rows:
        with OUT_ALL.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    # Best PCA per (model, features) by CV AUC
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["model"], r["features"])
        auc = r["cv_auc_roc_mean"]
        if auc is None:
            continue
        if k not in best or auc > best[k]["cv_auc_roc_mean"]:
            best[k] = r

    best_rows = sorted(best.values(), key=lambda r: (r["model"], r["features"]))
    if best_rows:
        with OUT_BEST.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(best_rows)

    print(f"Wrote {len(rows)} rows -> {OUT_ALL}")
    print(f"Wrote {len(best_rows)} rows -> {OUT_BEST}")


if __name__ == "__main__":
    main()
