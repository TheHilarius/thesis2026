#!/usr/bin/env python3
"""
13_dim_screening_report.py
Full-data per-dim screening report (DESCRIPTIVE ONLY — these AUCs are not
performance estimates; leak-free numbers come from the fold-nested
dimselect runs in 04_modelling.py).

Ranks every embedding dim with BOTH screeners (small XGB, LR-l2) on all
rows, then reports agreement:
    - Spearman rho over the full ranking
    - top-N Jaccard for N in {25, 50, 100, 200}
    - side-by-side top-20 dims
Outputs (runs/<runtag>/):
    results/dim_screening_full_<key>_<model>.csv   per-dim ranking
    results/dim_screening_overlap.tsv              agreement stats
    plots/dim_screening_auc_per_dim.png            ranked AUC curves

Example (server, env_esmc_server):
    python src/investigation/13_dim_screening_report.py \
        --out-root runs/dimselect_sweep
"""

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import numpy as np
import pandas as pd
from pathlib import Path

from config import SPLIT_DATA_PATH, FOLD_COL, LABEL_COL, \
    get_embedding_source, PROJECT_ROOT
from importlib import import_module

_dimscreen = import_module("12_dim_screening")

DEFAULT_KEYS = ["esmc300m_win_padtoken", "esmif_eos_bos_repeat"]
MODELS = ["xgb", "lr_l2"]
TOPNS = [25, 50, 100, 200]


def spearman_rho(r1, r2):
    """Spearman rho between two rank DataFrames (columns rank, dim)."""
    m = r1[["dim", "rank"]].merge(r2[["dim", "rank"]], on="dim",
                                  suffixes=("_a", "_b"))
    return float(m["rank_a"].corr(m["rank_b"], method="spearman"))


def top_jaccard(r1, r2, n):
    a = set(r1.nsmallest(n, "rank")["dim"])
    b = set(r2.nsmallest(n, "rank")["dim"])
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--out-root", default="runs/dimselect_sweep",
                    help="campaign folder (results/ + plots/ subdirs)")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    out_root = Path(PROJECT_ROOT) / args.out_root
    res_dir = out_root / "results"
    plot_dir = out_root / "plots"
    res_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLIT_DATA_PATH)
    indices = np.arange(len(df))
    y = df[LABEL_COL].values.astype(np.int32)
    folds = df[FOLD_COL].values

    overlap_rows = []
    for key in args.keys:
        src = get_embedding_source(key)
        prepared = src.get("prepared_path")
        if prepared is None or not Path(prepared).exists():
            print(f"SKIP {key}: prepared H5 missing ({prepared})")
            continue
        rankings = {}
        for model in MODELS:
            print(f"\n=== {key} / {model} (full data, descriptive) ===")
            r = _dimscreen.screen_dims(
                str(prepared), indices, y, folds, model,
                cache_dir=res_dir, cache_key=f"dim_screening_full_{key}_{model}",
                n_jobs=args.jobs, tag=f"{key} full",
            )
            rankings[model] = r
            r.to_csv(res_dir / f"dim_screening_full_{key}_{model}.csv",
                     index=False)
            print(r.head(20).to_string(index=False))

        rho = spearman_rho(rankings["xgb"], rankings["lr_l2"])
        row = {"key": key, "n_dims": len(rankings["xgb"]),
               "spearman_rho": round(rho, 4)}
        for n in TOPNS:
            row[f"jaccard_top{n}"] = round(
                top_jaccard(rankings["xgb"], rankings["lr_l2"], n), 4)
        overlap_rows.append(row)
        print(f"\n  {key}: Spearman rho={rho:.4f}, "
              + ", ".join(f"J@{n}={row[f'jaccard_top{n}']}" for n in TOPNS))

        # ranked AUC curves, both screeners
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for model, r in rankings.items():
                rr = r.sort_values("rank")
                ax.plot(rr["rank"], rr["auc_disc"], label=model, lw=1.2)
            ax.set_xlabel("dim rank")
            ax.set_ylabel("discriminative AUC (max(AUC, 1-AUC))")
            ax.set_title(f"Per-dim screening — {key} "
                         "(full data, descriptive only)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"dim_screening_auc_per_dim_{key}.png",
                        dpi=150)
            plt.close(fig)
        except ImportError:
            print("  (matplotlib missing — plot skipped)")

    if overlap_rows:
        ov = pd.DataFrame(overlap_rows)
        ov.to_csv(res_dir / "dim_screening_overlap.tsv", sep="\t",
                  index=False)
        print(f"\n  wrote {res_dir / 'dim_screening_overlap.tsv'}")


if __name__ == "__main__":
    main()
