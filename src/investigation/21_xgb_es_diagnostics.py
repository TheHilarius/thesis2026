#!/usr/bin/env python3
"""
21_xgb_es_diagnostics.py
Early-stopping diagnostic figures for the XGB blosum50 sweep.

Figures (all written to runs/xgb_early_stop_auc/plots/):
  es_sweep_auc_bars.png        padmode-style bar chart: AUC-ROC mean +/- std
                               per arm (baseline, logloss-ES, auc-ES,
                               fixed-budget low-LR if present), baseline
                               dashed line, DeLong Holm stars from
                               delong_vs_baseline.tsv.
  es_sweep_mcc_bars.png        same layout, MCC.
  es_best_iteration_strip.png  "luck" figure: one row per ES arm, one dot per
                               inner fold's chosen tree count (log x), vertical
                               line at baseline's fixed 1000.
  es_fold_auc_deltas.png       per-fold paired deltas (arm - baseline) across
                               the 6 outer buckets; every point < 0 = the
                               selection step hurts on every fold.
  es_learning_curves.png       (--curves; needs xgboost, run on server)
                               one representative split: inner eval-bucket vs
                               outer held-out-bucket AUC/logloss per boosting
                               round, with the simulated ES stop point marked
                               — shows the inner peak is NOT the outer peak.

Usage:
  python src/investigation/21_xgb_es_diagnostics.py            # first 4 figs
  python src/investigation/21_xgb_es_diagnostics.py --curves   # + learning curves
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH

PLOTS = PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "plots"
BASELINE = (PROJECT_ROOT / "runs" / "encodings_sweep" / "models" / "flat" /
            "cv_results_xgb_blosum50_flat_20261014_154653.json")
RUN_DIRS = {
    "es_ll": PROJECT_ROOT / "runs" / "xgb_early_stop_logloss" / "models" / "flat",
    "es_auc": PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "models" / "flat",
    "fixed": PROJECT_ROOT / "runs" / "xgb_fixed_lowlr" / "models" / "flat",
}
DELONG_TSV = (PROJECT_ROOT / "runs" / "xgb_early_stop_auc" / "results" /
              "delong_vs_baseline.tsv")

# Paul Tol bright (colour-blind safe) — same scheme as plot_padmode_data_compare
GROUP_COLORS = {"baseline": "#BBBBBB", "es_ll": "#4477AA",
                "es_auc": "#EE6677", "fixed": "#CCBB44"}
GROUP_LABELS = {"baseline": "Fixed budget (baseline)",
                "es_ll": "Early stop — logloss",
                "es_auc": "Early stop — AUC",
                "fixed": "Fixed budget — low LR"}


# ── loaders ─────────────────────────────────────────────────────────────
def load_arm(path: Path):
    with open(path) as fh:
        res = json.load(fh)
    tag = path.name
    m_lr = re.search(r"xgblr([\dp]+)n(\d+)", tag)
    if m_lr:
        lr = float(m_lr.group(1).replace("p", "."))
        nest = int(m_lr.group(2))
    else:
        hp = res["config"]["hyperparameters"]
        lr = float(hp.get("learning_rate", "0.1"))
        nest = int(hp.get("n_estimators", "1000"))
    folds = [float(m["auc_roc"]) for m in res["outer_val_metrics"]]
    folds_mcc = [float(m["mcc"]) for m in res["outer_val_metrics"]]
    return {
        "tag": tag,
        "lr": lr,
        "n_estimators": nest,
        "auc": float(res["cv_summary"]["auc_roc"]["mean"]),
        "auc_sd": float(res["cv_summary"]["auc_roc"]["std"]),
        "mcc": float(res["cv_summary"]["mcc"]["mean"]),
        "mcc_sd": float(res["cv_summary"]["mcc"]["std"]),
        "folds": folds,
        "folds_mcc": folds_mcc,
        "best_iterations": res.get("best_iterations") or [],
        "group": None,
    }


def load_all():
    arms = []
    base = load_arm(BASELINE)
    base["group"] = "baseline"
    base["es"] = False
    arms.append(base)
    for group, d in RUN_DIRS.items():
        if not Path(d).is_dir():
            print(f"  [skip] {group}: {d} not found")
            continue
        for p in sorted(Path(d).glob("cv_results_*.json")):
            a = load_arm(p)
            a["group"] = group
            a["es"] = bool(a["best_iterations"])
            arms.append(a)
    return arms


def load_stars():
    """arm label -> Holm stars, from the DeLong TSV (if present)."""
    if not DELONG_TSV.exists():
        return {}
    df = pd.read_csv(DELONG_TSV, sep="\t")
    out = {}
    for _, r in df.iterrows():
        out[str(r["arm"])] = r["sig"]
    return out


# ── figure 1+2: padmode-style bars ──────────────────────────────────────
def bar_figure(arms, stars, value, sd, fname, ylabel, fixed1000_line=True):
    """One bar per arm: baseline first, then group, LR ascending within."""
    order = {"baseline": 0, "es_ll": 1, "es_auc": 2, "fixed": 3}
    rows = sorted(arms, key=lambda a: (order[a["group"]], a["lr"],
                                       a["n_estimators"]))
    n = len(rows)
    x = np.arange(n)
    colors = [GROUP_COLORS[a["group"]] for a in rows]
    vals = np.array([a[value] for a in rows])
    errs = np.array([a[sd] for a in rows])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, vals, yerr=errs, color=colors, width=0.62,
           edgecolor="black", linewidth=0.5,
           error_kw=dict(lw=0.8, capsize=2))
    base_val = rows[0][value]
    for xi, v, a in zip(x, vals, rows):
        lab = f"{v:.4f}"
        star = ""
        if a["group"] != "baseline":
            key = (f"es_auc_lr{a['lr']}" if a["group"] == "es_auc"
                   else f"es_ll_lr{a['lr']}" if a["group"] == "es_ll" else None)
            if key and key in stars:
                star = str(stars[key])
        ax.text(xi, v + errs[list(x).index(xi)] + 0.004,
                f"{lab}\n{star}" if star else lab,
                ha="center", va="bottom", fontsize=7.5)

    labels = []
    for a in rows:
        if a["group"] == "baseline":
            labels.append(f"baseline\n0.1 / 1000")
        elif a["group"] == "fixed":
            labels.append(f"fixed\n{a['lr']:g} / {a['n_estimators']}")
        else:
            tag = "logloss" if a["group"] == "es_ll" else "auc"
            labels.append(f"ES-{tag}\n{a['lr']:g}")
    ax.axhline(base_val, color="black", linestyle="--", lw=1.0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"XGB blosum50 — {ylabel} per configuration "
        f"(mean ± std, 6 outer buckets; dashed: baseline {base_val:.4f})",
        fontsize=11)
    ymin = min(vals - errs) - 0.01
    ymax = max(vals + errs) + 0.012
    ax.set_ylim(ymin, ymax)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[g],
                             edgecolor="black", linewidth=0.5)
               for g in order if any(a["group"] == g for a in rows)]
    ax.legend(handles, [GROUP_LABELS[g] for g in order
                        if any(a["group"] == g for a in rows)],
              fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PLOTS / fname
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


# ── figure 3: best_iteration strip ──────────────────────────────────────
def strip_figure(arms):
    es = [a for a in arms if a["es"]]
    order = {"es_ll": 0, "es_auc": 1}
    es = sorted(es, key=lambda a: (order[a["group"]], a["lr"]))
    if not es:
        print("  [skip] strip: no ES arms")
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for yi, a in enumerate(es):
        bi = np.asarray(a["best_iterations"], dtype=float)
        col = GROUP_COLORS[a["group"]]
        jit = np.random.default_rng(42).uniform(-0.13, 0.13, len(bi))
        ax.scatter(bi, yi + jit, s=26, color=col, alpha=0.75,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.scatter([bi.mean()], yi, marker="|", s=260, color="black",
                   linewidth=1.6, zorder=4)
        ax.annotate(f"med={int(np.median(bi))}  [{int(bi.min())}–{int(bi.max())}]",
                    xy=(1.02, yi), xycoords=("axes fraction", "data"),
                    va="center", fontsize=8)
    ax.axvline(1000, color="#333333", linestyle="--", lw=1.2, zorder=2)
    ax.text(1000, len(es) - 0.35, " baseline: fixed 1000", fontsize=8.5,
            color="#333333", va="center")
    ax.set_yticks(range(len(es)))
    ax.set_yticklabels([
        f"{'logloss-ES' if a['group']=='es_ll' else 'auc-ES'}  lr={a['lr']:g} "
        f"(n={len(a['best_iterations'])})"
        for a in es], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Chosen stopping iteration (best_iteration, log scale)")
    ax.set_title("Early stopping picks a different 'optimal' tree count in "
                 "every fold — the choice tracks validation noise, not a "
                 "property of the data", fontsize=11)
    ax.set_xlim(20, 6000)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    out = PLOTS / "es_best_iteration_strip.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


# ── figure 4: per-fold paired deltas ────────────────────────────────────
def delta_figure(arms):
    base = arms[0]
    others = [a for a in arms[1:] if a["group"] != "baseline"]
    order = {"es_ll": 0, "es_auc": 1, "fixed": 2}
    others = sorted(others, key=lambda a: (order[a["group"]], a["lr"],
                                           a["n_estimators"]))
    if not others:
        print("  [skip] deltas: no comparison arms")
        return
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for xi, a in enumerate(others):
        d = np.array(a["folds"]) - np.array(base["folds"])
        col = GROUP_COLORS[a["group"]]
        jit = np.random.default_rng(7).uniform(-0.12, 0.12, len(d))
        ax.scatter(xi + jit, d, s=30, color=col, alpha=0.75,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.plot([xi - 0.28, xi + 0.28], [d.mean(), d.mean()],
                color="black", lw=1.6, zorder=4)
        ax.annotate(f"{d.mean():+.4f}", xy=(xi, d.min() - 0.0022),
                    ha="center", va="top", fontsize=7.5)
    labels = [
        (f"{'logloss-ES' if a['group']=='es_ll' else 'auc-ES' if a['group']=='es_auc' else 'fixed'}"
         f"\n{a['lr']:g}" + (f"/{a['n_estimators']}" if a["group"] == "fixed" else ""))
        for a in others]
    ax.axhline(0, color="black", lw=1.2, zorder=2)
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Δ AUC-ROC (arm − baseline), per outer bucket")
    ax.set_title("Every fold of every arm falls below the fixed-budget "
                 "baseline (6 paired outer buckets; black bar = mean)",
                 fontsize=11)
    ax.set_ylim(min(a["folds"][i] - base["folds"][i]
                    for a in others for i in range(6)) - 0.004,
                max(0.004, max(a["folds"][i] - base["folds"][i]
                               for a in others for i in range(6)) + 0.004))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = PLOTS / "es_fold_auc_deltas.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


# ── figure 5: learning-curve divergence (--curves; server only) ────────
def simulated_stop(curve, metric, patience=50):
    """Mirror xgboost ES: first round with no improvement for `patience`
    rounds; returns (stop_round, best_round). curve aligned to round 1..n."""
    improve = np.less if metric == "auc" else np.greater
    best_r, best_v = 1, curve[0]
    for i, v in enumerate(curve[1:], start=2):
        if improve(v, best_v):
            best_r, best_v = i, v
        elif i - best_r >= patience:
            return i, best_r
    return len(curve), best_r


def curves_figure():
    import h5py  # noqa: F401  (04_modelling import needs it)
    import importlib
    mod = importlib.import_module("04_modelling")
    import xgboost
    from config import get_model_config, get_feature_set_config

    print(f"  xgboost {xgboost.__version__}")
    df = pd.read_csv(SPLIT_DATA_PATH)
    y_all = df["label"].values.astype(np.int32)

    feat_cfg = get_feature_set_config("blosum50")
    model_cfg = dict(get_model_config("xgb"))
    # mimic prepare_fold defaults: outer val bucket = 0, inner test bucket = 1
    OUTER, INNER = 0, 1
    df_csv, csv_cols, emb_dict, _ = mod.load_all_components(df, feat_cfg, "flat")
    X_tr, y_tr, X_te, y_te, _, art = mod.prepare_fold(
        df_csv, csv_cols, emb_dict, model_cfg, INNER, validation_bucket=OUTER)
    X_out, y_out, _ = mod.prepare_validation(
        df_csv, csv_cols, emb_dict, model_cfg, art, validation_bucket=OUTER)
    print(f"  train={len(y_tr)} inner-eval={len(y_te)} outer-eval={len(y_out)}")

    panels = [("auc", 0.1), ("logloss", 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (metric, lr) in zip(axes, panels):
        params = {k: v for k, v in model_cfg["params"].items()
                  if k not in ("n_estimators", "learning_rate",
                               "eval_metric", "early_stopping_rounds")}
        params.update(n_estimators=1000, learning_rate=lr,
                      eval_metric=metric, verbosity=0)
        clf = xgboost.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te), (X_out, y_out)],
                verbose=False)
        er = clf.evals_result_
        key_in = "validation_0"
        key_out = "validation_1"
        cin = np.asarray(er[key_in][metric])
        cout = np.asarray(er[key_out][metric])
        rounds = np.arange(1, len(cin) + 1)
        stop_r, best_r = simulated_stop(cin, metric)

        better = np.less if metric == "auc" else np.greater
        outer_at_best = cout[best_r - 1]
        outer_best_r = int(np.argmin(cout) + 1) if metric == "logloss" \
            else int(np.argmax(cout) + 1)
        inner_label = ("inner eval bucket (ES decides here)"
                       if metric == "auc" else "inner eval bucket (logloss)")
        outer_label = ("outer held-out bucket (reported)"
                       if metric == "auc" else "outer held-out bucket (logloss)")
        ax.plot(rounds, cin, color="#4477AA", lw=1.6, label=inner_label)
        ax.plot(rounds, cout, color="#EE6677", lw=1.6, label=outer_label)
        ax.axvline(best_r, color="black", linestyle="--", lw=1.2)
        ax.scatter([best_r], [cin[best_r - 1]], color="black", zorder=5, s=30)
        ax.annotate(f"ES rolls back here\n(round {best_r})",
                    xy=(best_r, cin[best_r - 1]),
                    xytext=(best_r + 90, cin[best_r - 1]),
                    fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", lw=0.9))
        ax.scatter([outer_best_r], [cout[outer_best_r - 1]],
                   color="#EE6677", zorder=5, s=40,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(f"outer optimum\n(round {outer_best_r})",
                    xy=(outer_best_r, cout[outer_best_r - 1]),
                    xytext=(outer_best_r - 330, cout[outer_best_r - 1]
                            + (0.006 if metric == "auc" else -0.008)),
                    fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", lw=0.9))
        ax.set_xlabel("Boosting round")
        ax.set_ylabel(metric.upper() if metric == "auc" else "logloss")
        sign = "peak" if metric == "auc" else "minimum"
        ax.set_title(f"eval_metric={metric}: inner {sign} at round {best_r}, "
                     f"outer {sign} at round {outer_best_r}", fontsize=10.5)
        ax.legend(fontsize=8.5, loc="lower left" if metric == "auc"
                  else "upper right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Why early stopping underperforms: the bucket ES validates on peaks "
        "far too early — held-out performance keeps improving to ~1000 rounds "
        "(lr 0.1, outer bucket 0, inner bucket 1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "es_learning_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", action="store_true",
                    help="also train single-fold models and draw the "
                         "learning-curve figure (needs xgboost/h5py)")
    args = ap.parse_args()

    PLOTS.mkdir(parents=True, exist_ok=True)
    arms = load_all()
    stars = load_stars()
    print(f"  {len(arms)} arms: baseline + {len(arms) - 1} comparison")

    bar_figure(arms, stars, "auc", "auc_sd", "es_sweep_auc_bars.png",
               "AUC-ROC")
    bar_figure(arms, stars, "mcc", "mcc_sd", "es_sweep_mcc_bars.png", "MCC")
    strip_figure(arms)
    delta_figure(arms)

    if args.curves:
        curves_figure()
    else:
        print("  (learning-curve figure skipped — run with --curves on "
              "the server)")


if __name__ == "__main__":
    main()
