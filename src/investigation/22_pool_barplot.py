#!/usr/bin/env python3
"""
22_pool_barplot.py — AUC-ROC barplot: pooled ESM-C windows vs BLOSUM50 and
the cumulative-variance anchor (topX 643 dims at 80% EV).

One figure (naive pool mask):
    runs/<runtag>/plots/pool_naive_auc.png

NOTE: masked and naive pooling proved near-identical, so naive is the
standard from now on. Only the naive figure is drawn (filenames keep
the pool_naive_auc names).

Bars (left to right):
    BLOSUM50 (609 feat)                        — encodings_sweep
    ESM-C-300M cumulative variance 80% (643 dims, hatched, no pooling)
                                               — ev_matched_topXvariance_sweep
    ESM-C-300M pooled x7 (mean/median/max/min/mean_max/mean_median/mean_std)

Pool AUCs are the nested-CV means (cv_summary.auc_roc) parsed from the
pool-grid cv_results JSONs (runs/<runtag>/models/flat/cv_results_*.json).

Bar colors follow the PCA-sweep figures (pca_sweep_AUC_delong_
comparison.py): ESM-C-300M #EE6677, BLOSUM50 #BBBBBB.
The cumulative-variance anchor bar is hatched to separate it from pool bars.

Outputs (default runtag=pool_xgb):
    runs/pool_xgb/plots/pool_naive_auc.png (+ .csv)
    runs/pool_xgb/results/pool_auc_all.csv (aggregate table, all pool arms found)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# fallback values if the sweep files are absent
FALLBACK = {
    "blosum50": (0.7633, 0.0044),
    "esmc_var643": (0.701727, 0.004966),
}

COLORS = {
    "encoding": "#BBBBBB",   # BLOSUM50 / AA encoding grey (pca-sweep refs)
    "esmc": "#EE6677",       # ESM-C-300M — pca_sweep_AUC_delong_comparison
}

POOL_ORDER = ["mean", "median", "max", "min",
              "mean_max", "mean_median", "mean_std"]
POOL_LABEL = {
    "mean": "mean",
    "median": "median",
    "max": "max",
    "min": "min",
    "mean_max": "mean+max",
    "mean_median": "mean+med",
    "mean_std": "mean+std",
}
EMB_ORDER = ["esmc"]
EMB_FEATURE = {
    "esmc": "emb_esmc300m_win_padtoken",
}


def load_pool_jsons(models_dir: Path):
    """(mask, features_key, pool) -> (auc, std) from cv_results JSONs."""
    out = {}
    files = sorted(models_dir.glob("cv_results_*.json"))
    for f in files:
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            print(f"  WARN: unreadable {f.name} — skipped")
            continue
        cfg = r.get("config", {})
        pool = cfg.get("pool")
        if not pool:
            continue  # not a pooling run
        mask = cfg.get("pool_mask", "masked")
        feat = r.get("features_key", "")
        summ = r.get("cv_summary", {}).get("auc_roc", {})
        if "mean" not in summ:
            print(f"  WARN: no auc_roc summary in {f.name} — skipped")
            continue
        out[(mask, feat, pool)] = (float(summ["mean"]),
                                   float(summ.get("std", 0.0)))
    return out


def load_delong_pick(path: Path, campaign: str, k: str):
    """(auc_outer_mean, auc_outer_std) from pca_sweep_results.csv."""
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("campaign") == campaign and r.get("k") == k:
                return (float(r["auc_outer_mean"]), float(r["auc_outer_std"]))
    return None


def load_var_anchor():
    """(auc_mean, auc_std) for the 643-dim cumulative-variance anchor.

    Reads cv_summary.auc_roc straight from the newest var643 JSON
    (same source 26_ev_variance_vs_pca.py uses); None if absent.
    """
    pat = (PROJECT_ROOT / "runs" / "ev_matched_topXvariance_sweep"
           / "models" / "simple21")
    files = sorted(pat.glob(
        "cv_results_xgb_emb_esmc300m_win_padtoken_simple21_var643_*.json")) \
        if pat.exists() else []
    if not files:
        return None
    try:
        with open(files[-1]) as fh:
            summ = json.load(fh)["cv_summary"]["auc_roc"]
        return (float(summ["mean"]), float(summ.get("std", 0.0)))
    except (OSError, ValueError, KeyError):
        return None


def load_blosum(path: Path):
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("feature_set") == "blosum50" and r.get("model_key") == "xgb":
                return (float(r["cv_mean_auc_roc"]), float(r["cv_std_auc_roc"]))
    return None


def draw_figure(pool_vals, blosum, esmc_var, png, csvp,
                stats_csv=None):
    bars = [("BLOSUM50", blosum, "encoding", False, "blosum50")]
    feat = EMB_FEATURE["esmc"]
    bars.append(("cumulative variance 80%\n(643 dims)", esmc_var, "esmc",
                 True, "esmc_var643"))
    for pool in POOL_ORDER:
        val = pool_vals.get(("naive", feat, pool))
        bars.append((POOL_LABEL[pool], val, "esmc", False,
                     f"esmc_{pool}"))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    rows = []
    for i, (label, val, color, hatched, _arm) in enumerate(bars):
        if val is None:
            print(f"  WARN: missing result for bar {label!r} — skipped")
            continue
        kw = {}
        if hatched:
            kw = {"hatch": "///"}
        ax.bar(i, val[0], yerr=val[1], capsize=3, width=0.62,
               color=COLORS[color], edgecolor="black", linewidth=0.6, **kw)
        ax.text(i, val[0] + val[1] + 0.006, f"{val[0]:.3f}",
                ha="center", fontsize=8, fontweight="bold")
        rows.append({"bar": label.replace("\n", " "),
                     "auc_roc": f"{val[0]:.6f}", "std": f"{val[1]:.6f}"})

    present = [(i, b) for i, b in enumerate(bars) if b[1] is not None]
    ax.set_xticks([i for i, _ in present])
    ax.set_xticklabels([b[0] for _, b in present], fontsize=8)
    ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")

    # ── significance brackets (--stats-csv from 23_pool_delong.py) ──
    # family is esmc (pool ~ variance anchor); all drawn.
    arm_x = {arm: i for i, (_, v, _, _, arm) in enumerate(bars)
             if v is not None}
    if stats_csv:
        with open(stats_csv) as fh:
            stats_rows = list(csv.DictReader(fh, delimiter="\t"))
        data_top = max(v[0] + v[1] for _, v, _, _, _ in bars if v is not None)
        base, step, fam_lvl = data_top + 0.015, 0.016, {}
        for r in stats_rows:
            a, b = r["arm_a"], r["arm_b"]
            if a not in arm_x or b not in arm_x:
                print(f"  WARN: stats pair {a}~{b} has no bar — skipped")
                continue
            lvl = fam_lvl.get(r["family"], 0)
            fam_lvl[r["family"]] = lvl + 1
            y = base + lvl * step
            xa, xb = sorted((arm_x[a], arm_x[b]))
            ax.plot([xa, xa, xb, xb], [y - 0.004, y, y, y - 0.004],
                    lw=0.8, color="black")
            ax.text((xa + xb) / 2, y + 0.002, r["star"],
                    ha="center", va="bottom", fontsize=9.5)

    ax.legend(handles=[
        Patch(facecolor=COLORS["encoding"], label="BLOSUM50"),
        Patch(facecolor=COLORS["esmc"], label="ESM-C-300M padtoken"),
        Patch(facecolor="white", hatch="///",
              label="topX-variance (no pooling)"),
    ], loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("pooled ESM-C windows (naive) vs BLOSUM50 and cumulative variance",
                 fontsize=12)
    fig.tight_layout()

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bar", "auc_roc", "std"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {png}")
    print(f"  wrote {csvp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runtag", default="pool_xgb",
                    help="run folder under runs/ owning inputs+outputs")
    ap.add_argument("--models-dir", default=None,
                    help="explicit models dir; default runs/<runtag>/models/flat")
    ap.add_argument("--stats-csv", default=None,
                    help="pool_delong_pairs.tsv from 23_pool_delong.py — "
                         "draw significance brackets with stars. Default: "
                         "runs/<runtag>/results/pool_delong_pairs.tsv if it "
                         "exists (brackets drawn automatically)")
    args = ap.parse_args()

    root = PROJECT_ROOT
    run_dir = root / "runs" / args.runtag
    models_dir = Path(args.models_dir) if args.models_dir else (
        run_dir / "models" / "flat")
    stats_csv = args.stats_csv
    if stats_csv is None:
        default_stats = run_dir / "results" / "pool_delong_pairs.tsv"
        stats_csv = str(default_stats) if default_stats.exists() else None
        print(f"  stats: {stats_csv or 'none (no --stats-csv, no brackets)'}")

    print("Loading baselines ...")
    blosum = load_blosum(root / "runs/encodings_sweep/results/metrics.tsv") \
        or FALLBACK["blosum50"]
    esmc_var = load_var_anchor() or FALLBACK["esmc_var643"]
    print(f"  blosum50       {blosum[0]:.4f} +/- {blosum[1]:.4f}")
    print(f"  esmc var643  {esmc_var[0]:.4f} +/- {esmc_var[1]:.4f}  (643 dims, 80% EV)")

    print(f"Loading pool JSONs from {models_dir} ...")
    pool_vals = load_pool_jsons(models_dir)
    print(f"  {len(pool_vals)} pooled results loaded")
    for (mask, feat, pool), v in sorted(pool_vals.items()):
        print(f"  {mask:<7} {feat:<30} {pool:<11} "
              f"{v[0]:.4f} +/- {v[1]:.4f}")

    plots_dir = run_dir / "plots"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    draw_figure(pool_vals, blosum, esmc_var,
                plots_dir / "pool_naive_auc.png",
                plots_dir / "pool_naive_auc.csv",
                stats_csv=stats_csv)

    agg = results_dir / "pool_auc_all.csv"
    with open(agg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mask", "features", "pool",
                                           "auc_roc", "std"])
        w.writeheader()
        for (mask, feat, pool) in sorted(pool_vals):
            v = pool_vals[(mask, feat, pool)]
            w.writerow({"mask": mask, "features": feat, "pool": pool,
                        "auc_roc": f"{v[0]:.6f}", "std": f"{v[1]:.6f}"})
    print(f"  wrote {agg}")

    print("\n" + "=" * 60)
    print("  CAPTION (paste below the figure):")
    print("=" * 60)
    print("Pooled-window XGB AUC-ROC (nested CV, mean +/- std over 6 outer "
          "iterations). Pooling collapses the 29-slot window to one (or two) "
          "D-dim vectors per sample with no PCA (naive pooling over all 29 "
          "slots; masked proved near-identical). Variance bar = topX "
          "cumulative-variance selection (643 raw ESM-C dims at 80% EV, "
          "no rotation, hatched). "
          "BLOSUM50 (0.763) = xgb one-hot baseline (encodings_sweep).")
    print("=" * 60)


if __name__ == "__main__":
    main()
