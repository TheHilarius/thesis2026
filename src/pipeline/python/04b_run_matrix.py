#!/usr/bin/env python3
"""
04b_run_matrix.py
Run 04_modelling.py across a matrix of (model, feature_set, pca) combos
and aggregate results into a ranked performance table.

Usage:
    python 04b_run_matrix.py                                         # default combos
    python 04b_run_matrix.py --dry-run                               # print only

    # PCA sweep (scalar values only)
    python 04b_run_matrix.py --pca-sweep \
        rf:handcrafted_sparse_esmc:1,13,26,66,218,718 \
        rf:handcrafted_sparse_esmif:9,64,95,148,248,420

    # Combined set (per-embedding PCA via --combos)
    python 04b_run_matrix.py --combos \
        rf,handcrafted_sparse_esmc_esmif,esmc=1,esmif=9
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
MODELING_SCRIPT = SRC_DIR / "04_modelling.py"
EXTRACT_SCRIPT = SRC_DIR / "extract_model_metrics.py"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

# Default combos: (model, feature_set, pca)
# pca=None means use config.py default (50)
# pca can be int or dict string (e.g. "esmc=1,esmif=9")
DEFAULT_COMBOS = [
    # Baselines
    ("rf", "handcrafted", None),
    ("rf", "handcrafted_sparse", None),
    ("rf", "handcrafted_blosum", None),
    # ESM-C context (50% variance -> 1 PC)
    ("rf", "handcrafted_sparse_esmc", 1),
    # ESM-IF context (50% variance -> 9 PCs)
    ("rf", "handcrafted_sparse_esmif", 9),
    # Combined
    ("rf", "handcrafted_sparse_esmc_esmif", "esmc=1,esmif=9"),
    # Same with L2
    ("lr_l2", "handcrafted", None),
    ("lr_l2", "handcrafted_sparse", None),
    ("lr_l2", "handcrafted_blosum", None),
    ("lr_l2", "handcrafted_sparse_esmc", 1),
    ("lr_l2", "handcrafted_sparse_esmif", 9),
    ("lr_l2", "handcrafted_sparse_esmc_esmif", "esmc=1,esmif=9"),
    # Same with ElasticNet
    ("lr_elasticnet", "handcrafted", None),
    ("lr_elasticnet", "handcrafted_sparse", None),
    ("lr_elasticnet", "handcrafted_blosum", None),
    ("lr_elasticnet", "handcrafted_sparse_esmc", 1),
    ("lr_elasticnet", "handcrafted_sparse_esmif", 9),
    ("lr_elasticnet", "handcrafted_sparse_esmc_esmif", "esmc=1,esmif=9"),
    # XGBoost
    ("xgb", "handcrafted", None),
    ("xgb", "handcrafted_sparse", None),
    ("xgb", "handcrafted_blosum", None),
    ("xgb", "handcrafted_sparse_esmc", 1),
    ("xgb", "handcrafted_sparse_esmif", 9),
    ("xgb", "handcrafted_sparse_esmc_esmif", "esmc=1,esmif=9"),
]


def run_combo(model, features, pca, pca_mode="slot", dry_run=False):
    """Run one 04_modelling.py invocation."""
    cmd = [
        sys.executable, str(MODELING_SCRIPT),
        "--model", model,
        "--features", features,
        "--pca-mode", pca_mode,
    ]
    if pca is not None:
        cmd += ["--pca", str(pca)]

    label = f"{model} x {features}"
    if pca is not None:
        label += f" (pca={pca})"
    if pca_mode != "slot":
        label += f" [{pca_mode}]"

    print(f"\n{'_' * 60}")
    print(f"  RUNNING: {label}")
    print(f"{'_' * 60}")

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  DONE in {elapsed:.1f}s")
    else:
        print(f"  FAILED (exit code {result.returncode})")
    return result.returncode == 0


def expand_pca_sweep(spec):
    """Expand 'model:feature_set:pca1,pca2,...' into combos."""
    model, features, pca_list = spec.split(":")
    model = model.strip()
    features = features.strip()
    combos = []
    for p in pca_list.split(","):
        p = p.strip()
        if "=" in p:
            combos.append((model, features, p))
        else:
            combos.append((model, features, int(p)))
    return combos


def aggregate_results(start_time, combos, pca_mode="slot", dry_run=False):
    """Run extract_model_metrics.py to build performance table."""
    out_dir = RESULTS_DIR / pca_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    models = sorted({m for m, _, _ in combos})
    model_tag = "_".join(models) if models else "all"
    ts_file = start_time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"model_matrix_{model_tag}_{ts_file}.tsv"

    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        "--results", f"models/{pca_mode}/cv_results_*.json",
        "--since", ts_str,
        "--out", str(out_path),
    ]

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATING RESULTS")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0 and out_path.exists():
        print(f"\n  Results: {out_path}")
        _print_table(out_path)
    else:
        print(f"  Aggregation failed or no results found.")


def _print_table(tsv_path):
    """Print a formatted table from the TSV."""
    import csv
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        print("  No results found.")
        return

    # Sort by held-out AUC descending
    rows.sort(
        key=lambda x: float(x.get("heldout_auc_roc", 0) or 0),
        reverse=True,
    )

    print(
        f"\n  {'Model':<6} {'Features':<45} {'CV AUC':>8} "
        f"{'HO AUC':>8} {'HO MCC':>8}"
    )
    print(f"  {'_' * 6} {'_' * 45} {'_' * 8} {'_' * 8} {'_' * 8}")

    for r in rows:
        model = r.get("model_key", "")
        feat = r.get("feature_set", "")
        cv_auc = r.get("cv_mean_auc_roc", "")
        ho_auc = r.get("heldout_auc_roc", "")
        ho_mcc = r.get("heldout_mcc", "")

        # Truncate long feature names
        if len(feat) > 44:
            feat = feat[:41] + "..."

        print(
            f"  {model:<6} {feat:<45} {cv_auc:>8} "
            f"{ho_auc:>8} {ho_mcc:>8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run 04_modelling across a matrix of combos and report performance.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--combos", nargs="*", default=None,
        help="Override combos: model,feature,pca model,feature,pca ... "
             "pca can be int or 'esmc=1,esmif=9'. Use None for no PCA.",
    )
    parser.add_argument(
        "--pca-sweep", nargs="*", default=None, dest="pca_sweep",
        help="PCA sweep: model:feature_set:pca1,pca2,... (repeatable). "
             "E.g. rf:handcrafted_sparse_esmc:1,13,26,66,218,718",
    )
    parser.add_argument(
        "--pca-mode", type=str, default="slot", choices=["slot", "flat"],
        help="Window PCA scheme passed to 04_modelling (default: slot)",
    )
    args = parser.parse_args()

    combos = []
    if args.pca_sweep:
        for spec in args.pca_sweep:
            combos.extend(expand_pca_sweep(spec))
    if args.combos:
        for spec in args.combos:
            parts = spec.split(",")
            model = parts[0].strip()
            features = parts[1].strip()
            # Join remaining parts as pca (handles per-embedding: esmc=1,esmif=9)
            pca = ",".join(parts[2:]).strip() if len(parts) > 2 else None
            if pca in (None, "", "None"):
                pca = None
            elif pca.isdigit():
                pca = int(pca)
            combos.append((model, features, pca))
    if not combos:
        combos = DEFAULT_COMBOS

    print("=" * 60)
    print("  MODEL MATRIX RUNNER")
    print("=" * 60)
    print(f"  Combos: {len(combos)}")
    print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print(f"  MODE:   DRY RUN")
    print("=" * 60)

    start_time = datetime.now()
    t_start = time.time()

    successes = 0
    failures = 0
    for model, features, pca in combos:
        ok = run_combo(model, features, pca, pca_mode=args.pca_mode,
                       dry_run=args.dry_run)
        if ok:
            successes += 1
        else:
            failures += 1

    t_end = time.time()
    total_minutes = (t_end - t_start) / 60

    print(f"\n{'=' * 60}")
    print(f"  MATRIX RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total:    {successes + failures} combos")
    print(f"  Success:  {successes}")
    print(f"  Failed:   {failures}")
    print(f"  Runtime:  {t_end - t_start:.1f}s ({total_minutes:.1f} min)")

    if not args.dry_run:
        aggregate_results(start_time, combos, pca_mode=args.pca_mode)


if __name__ == "__main__":
    main()
