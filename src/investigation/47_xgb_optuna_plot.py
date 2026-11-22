#!/usr/bin/env python3
"""
47_xgb_optuna_plot.py
Optuna tuning figures for the XGB blosum50 campaign (runs/xgb_optuna/).

Figures (runs/xgb_optuna/plots/):
  optuna_history.png        per-trial AUC-ROC dots + best-so-far step line
  optuna_importance.png     Optuna hyperparameter importance (fANOVA)
  optuna_parallel.png       parallel coordinates, lines colored by AUC
  optuna_auc_vs_logloss.png trial-wise AUC vs mean logloss + Spearman rho
                            (objective = AUC; this checks the metrics agree)

NOTE: no winner-vs-baseline bar here on purpose — the local
cv_results_xgb_blosum50_flat_20261014_*.json baseline is the 29-mer-era
split, confounded against this flank5_rmdup study. The proper comparison
is the same-split refit arm (04_modelling --xgb-params) + DeLong, which
follows this script.

Usage:
  python src/investigation/47_xgb_optuna_plot.py
  python src/investigation/47_xgb_optuna_plot.py --study study_smoke.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
for p in (str(SRC_DIR), str(PIPELINE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from config import PROJECT_ROOT  # noqa: E402

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    sys.exit("optuna not installed locally — pip install optuna")

RUN_DIR = PROJECT_ROOT / "runs" / "xgb_optuna"     # blosum50 arm default;
PLOTS = RUN_DIR / "plots"                          # overridden per --study-name
STUDY_NAME = "xgb_blosum50"
LABEL = "XGB blosum50"                             # figure title prefix
PARAMS = ["max_depth", "min_child_weight", "gamma", "subsample",
          "colsample_bytree", "reg_alpha", "reg_lambda"]


def arm_dir(study_name: str) -> Path:
    """Mirror 46_xgb_optuna_tune.arm_paths: xgb_blosum50 stays the original
    runs/xgb_optuna/, any other arm lives in runs/xgb_optuna_<features>/."""
    if study_name == "xgb_blosum50":
        return PROJECT_ROOT / "runs" / "xgb_optuna"
    return PROJECT_ROOT / "runs" / f"xgb_optuna_{study_name.removeprefix('xgb_')}"


def load_study(db_name: str, study_name: str) -> optuna.Study:
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{arm_dir(study_name) / db_name}")
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"Study '{study_name}': {len(study.trials)} trials, "
          f"{len(done)} complete, best AUC={study.best_value:.4f}")
    return study


def completed(study):
    return [t for t in study.trials if t.state.name == "COMPLETE"]


def fig_history(study):
    ts = completed(study)
    xs = [t.number for t in ts]
    ys = [t.value for t in ts]
    best = np.maximum.accumulate(ys)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, s=18, alpha=0.6, color="tab:blue", label="trial AUC")
    ax.step(xs, best, where="post", color="tab:red", lw=2,
            label="best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean CV AUC-ROC (6 buckets)")
    ax.set_title(f"{LABEL} — Optuna optimization history "
                 f"(TPE, {len(ts)} trials)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_history.png", dpi=300)
    plt.close(fig)


def fig_importance(study):
    imps = get_param_importances(study)
    names = list(imps.keys())
    vals = list(imps.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], vals[::-1], color="tab:blue")
    ax.set_xlabel("Importance (fraction)")
    ax.set_title(f"{LABEL} — hyperparameter importance "
                 "(fANOVA on completed trials)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_importance.png", dpi=300)
    plt.close(fig)


def fig_parallel(study):
    ts = completed(study)
    ys = np.array([t.value for t in ts], dtype=float)
    # normalize each param to [0, 1] across trials
    cols = {}
    for p in PARAMS:
        v = np.array([t.params.get(p, np.nan) for t in ts], dtype=float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        cols[p] = (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    order = np.argsort(ys)
    for i in order:
        ax.plot(range(len(PARAMS)), [cols[p][i] for p in PARAMS],
                color=cmap((ys[i] - ys.min()) / (ys.max() - ys.min() + 1e-12)),
                alpha=0.5, lw=1)
    ax.set_xticks(range(len(PARAMS)))
    ax.set_xticklabels(PARAMS, rotation=30, ha="right")
    ax.set_ylabel("normalized value")
    ax.set_title(f"{LABEL} — parallel coordinates "
                 "(line color = trial AUC, dark = low, bright = high)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_parallel.png", dpi=300)
    plt.close(fig)


def fig_auc_vs_logloss(study):
    ts = completed(study)
    auc = np.array([t.value for t in ts])
    ll = np.array([t.user_attrs.get("logloss_mean", np.nan) for t in ts])
    ok = ~np.isnan(ll)
    rho, p = spearmanr(auc[ok], ll[ok])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(auc[ok], ll[ok], c=auc[ok], cmap="viridis", s=22,
                    alpha=0.8)
    fig.colorbar(sc, ax=ax, label="trial AUC-ROC")
    ax.set_xlabel("Mean CV AUC-ROC (objective)")
    ax.set_ylabel("Mean CV logloss")
    ax.set_title(f"Trial-wise AUC vs logloss  "
                 f"(Spearman rho={rho:.2f}, p={p:.1e})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_auc_vs_logloss.png", dpi=300)
    plt.close(fig)
    print(f"  AUC-logloss Spearman: rho={rho:.3f}, p={p:.2e} "
          f"({'metrics agree' if rho < -0.8 else 'check ranking'}; "
          f"logloss should anti-correlate)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--study", default="study.db",
                    help="study db filename under the arm's runs dir "
                         "(default study.db; use study_smoke.db for smoke)")
    ap.add_argument("--study-name", default="xgb_blosum50",
                    help="study name (default xgb_blosum50; e.g. "
                         "xgb_handcrafted_blosum50 for the hc arm — "
                         "figures land in that arm's runs dir)")
    args = ap.parse_args()

    global RUN_DIR, PLOTS, STUDY_NAME, LABEL
    STUDY_NAME = args.study_name
    RUN_DIR = arm_dir(STUDY_NAME)
    PLOTS = RUN_DIR / "plots"
    LABEL = f"XGB {STUDY_NAME.removeprefix('xgb_')}"

    PLOTS.mkdir(parents=True, exist_ok=True)
    study = load_study(args.study, STUDY_NAME)
    fig_history(study)
    fig_importance(study)
    fig_parallel(study)
    fig_auc_vs_logloss(study)
    # sidecar summary for the thesis report
    summary = {
        "study": STUDY_NAME, "n_trials": len(study.trials),
        "best_auc": study.best_value, "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "importance": dict(get_param_importances(study)),
    }
    bp = RUN_DIR / "best_params_tuned.json"
    if bp.exists():  # annotate meta if tuner already wrote it
        meta = json.loads(bp.read_text())
        summary["meta"] = {k: meta.get(k) for k in
                           ("logloss_mean", "mode", "features", "split")}
    (PLOTS.parent / "optuna_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"Summary -> {PLOTS.parent / 'optuna_summary.json'}")
    print(f"Figures -> {PLOTS}/optuna_*.png")


if __name__ == "__main__":
    main()
