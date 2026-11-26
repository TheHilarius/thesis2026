#!/usr/bin/env python3
"""
05f_cnn_optuna_plot.py
Optuna tuning figures for the NN campaigns — SHARED by CNN (05e) and FFNN
(06e) via --model, following the 05d_plotting precedent (shared plotting
lives in the 05 family; the FFNN stage calls it too). Hand-rolled house
style, mirroring 47_xgb_optuna_plot.py.

Scope (locked 2026-11-26 plan): Tier 1 + Tier 2 only.
Tier 1 — thesis core:
  optuna_history.png        per-trial AUC dots + best-so-far line PER OUTER
                           STUDY (nesting stays visible; 6 colored lines)
  optuna_slice.png         per-param 1D scatter (AUC vs value) over the
                           CONCATENATED trials of all outer studies;
                           conditional layer params get their own panels
  optuna_importance.png    fANOVA importance averaged across the 6 outer
                           studies (conditional-param raggedness noted in
                           the caption)
  optuna_contour.png       pairwise interaction scatters: lr x epochs,
                           weight_decay x dropout, + model-specific third
                           (CNN: channels_1 x kernel_1; FFNN:
                           units_1 x n_hidden_layers)
Tier 2 — methods defense:
  optuna_intermediate.png  running-mean inner AUC per fold-step, pruned
                           (red, x) vs completed (blue) trials — pruner
                           behavior evidence
  optuna_auc_vs_logloss.png  trial-wise AUC vs logloss + Spearman rho
  optuna_rank.png          trial AUC-rank vs param value for the
                           CATEGORICAL params (complements slice, which is
                           unreadable for categoricals)
  winner_stability.png     per-outer winner inner AUC vs multi-seed outer
                           AUC bars (nested honesty display) — needs the
                           --refit output winner_stability.csv, else skipped

Output: runs/<model>_optuna_<features>/plots/ + optuna_summary.json and
winner_stability.csv next to plots/ (same layout as the XGB campaign).

Usage:
  python src/pipeline/python/05f_cnn_optuna_plot.py --model ffnn
  python src/pipeline/python/05f_cnn_optuna_plot.py --model cnn
  python src/pipeline/python/05f_cnn_optuna_plot.py --model ffnn --outer 0
  python src/pipeline/python/05f_cnn_optuna_plot.py --model ffnn --study-db study_smoke.db
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from config import PROJECT_ROOT, N_CV_FOLDS  # noqa: E402

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    sys.exit("optuna not installed — pip install optuna (or run on the server)")

MODEL = "cnn"
FEATURES = "emb_esmc300m_win_padtoken"
RUN_DIR = PROJECT_ROOT / "runs"
PLOTS = RUN_DIR / "plots"
LABEL = "CNN"

# categorical/int params per model (rank figure; slice covers everything)
CAT_PARAMS = {
    "cnn": ["epochs", "n_conv_layers", "channels_1", "kernel_1",
            "channels_2", "kernel_2", "channels_3", "kernel_3",
            "dense_units", "batch_size"],
    "ffnn": ["epochs", "n_hidden_layers", "units_1", "units_2", "units_3",
             "units_4", "activation", "batch_size"],
}
# contour pairs per model
PAIRS = {
    "cnn": [("lr", "epochs"), ("weight_decay", "dropout"),
            ("channels_1", "kernel_1")],
    "ffnn": [("lr", "epochs"), ("weight_decay", "dropout"),
             ("units_1", "n_hidden_layers")],
}
LOG_PARAMS = {"lr", "weight_decay"}   # log-scale x-axes in slice


def _codes(vals):
    """Categorical (string) params → positional codes + tick labels,
    numeric params pass through unchanged."""
    if all(isinstance(v, (int, float, np.integer, np.floating))
           and not isinstance(v, bool) for v in vals):
        return np.array(vals, dtype=float), None
    uniq = sorted(set(vals))
    m = {v: i for i, v in enumerate(uniq)}
    return np.array([m[v] for v in vals], dtype=float), uniq


def arm_dir(model: str, features: str) -> Path:
    return PROJECT_ROOT / "runs" / f"{model}_optuna_{features}"


def load_studies(db_name: str, outer: int | None) -> list:
    """Load the 6 per-outer studies (or one with --outer)."""
    studies = []
    outers = [outer] if outer is not None else list(range(N_CV_FOLDS + 1))
    for k in outers:
        name = f"{MODEL}_{FEATURES}_outer{k}"
        try:
            studies.append(optuna.load_study(
                study_name=name,
                storage=f"sqlite:///{RUN_DIR / db_name}"))
        except KeyError:
            print(f"  WARNING: study {name} not in {db_name} — skipped")
    assert studies, f"no studies found in {RUN_DIR / db_name}"
    n_done = sum(len(completed(s)) for s in studies)
    print(f"Loaded {len(studies)} studies, {n_done} completed trials total")
    return studies


def completed(study):
    return [t for t in study.trials if t.state.name == "COMPLETE"]


def all_trials(studies):
    return [t for s in studies for t in s.trials
            if t.state.name in ("COMPLETE", "PRUNED")]


def done_trials(studies):
    return [t for s in studies for t in completed(s)]


def fig_history(studies):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(studies):
        ts = completed(s)
        if not ts:
            continue
        xs = [t.number for t in ts]
        ys = [t.value for t in ts]
        ax.scatter(xs, ys, s=14, alpha=0.35, color=cmap(i))
        ax.step(xs, np.maximum.accumulate(ys), where="post",
                color=cmap(i), lw=1.8,
                label=f"outer {s.study_name.rsplit('outer', 1)[-1]} best")
    ax.set_xlabel("Trial number (within outer study)")
    ax.set_ylabel("Mean inner-val AUC-ROC (5 folds)")
    ax.set_title(f"{LABEL} — Optuna history per outer study (TPE, nested)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_history.png", dpi=300)
    plt.close(fig)


def _panel_grid(params):
    ncol = min(4, len(params))
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                            figsize=(3.1 * ncol, 2.7 * nrow),
                            squeeze=False)
    return fig, axes


def fig_slice(studies):
    ts = done_trials(studies)
    params = sorted({p for t in ts for p in t.params})
    fig, axes = _panel_grid(params)
    for j, p in enumerate(params):
        ax = axes[j // min(4, len(params))][j % min(4, len(params))]
        raw = [t.params[p] for t in ts if p in t.params]
        ys = np.array([t.value for t in ts if p in t.params])
        xs, labels = _codes(raw)
        ax.scatter(xs, ys, s=12, alpha=0.6, c=ys, cmap="viridis")
        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
        elif p in LOG_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(p, fontsize=8)
        ax.set_ylabel("inner AUC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(params), len(axes) * len(axes[0])):
        axes[j // len(axes[0])][j % len(axes[0])].set_visible(False)
    fig.suptitle(f"{LABEL} — per-parameter slice "
                 f"({len(ts)} completed trials, all outer studies pooled)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "optuna_slice.png", dpi=300)
    plt.close(fig)


def fig_importance(studies):
    """fANOVA per outer study, importance averaged over studies where the
    param was sampled (conditional params: ragged by design — caption)."""
    acc = {}
    for s in studies:
        try:
            imps = get_param_importances(s)
        except Exception as e:            # too few trials / dynamic space edge
            print(f"  importance skipped for {s.study_name}: {e}")
            continue
        for k, v in imps.items():
            acc.setdefault(k, []).append(v)
    assert acc, "no study produced importances"
    names = sorted(acc, key=lambda k: np.mean(acc[k]))
    vals = [np.mean(acc[k]) for k in names]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(names) + 1.6))
    ax.barh(names, vals, color="tab:blue")
    ax.set_xlabel("Importance (fraction, fANOVA mean over outer studies)")
    ax.set_title(f"{LABEL} — hyperparameter importance\n"
                 "(conditional layer params: averaged over studies that "
                 "sampled them)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_importance.png", dpi=300)
    plt.close(fig)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fig_contour(studies):
    ts = done_trials(studies)
    pairs = PAIRS[MODEL]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.8),
                            squeeze=False)
    for ax, (px, py) in zip(axes[0], pairs):
        sub = [t for t in ts if px in t.params and py in t.params]
        xs = np.array([t.params[px] for t in sub], dtype=float)
        ys = np.array([t.params[py] for t in sub], dtype=float)
        vs = np.array([t.value for t in sub])
        sc = ax.scatter(xs, ys, c=vs, cmap="viridis", s=22, alpha=0.85)
        fig.colorbar(sc, ax=ax, label="inner AUC")
        if px in LOG_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{LABEL} — parameter interactions (pooled trials, "
                 "color = inner AUC)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOTS / "optuna_contour.png", dpi=300)
    plt.close(fig)


def fig_intermediate(studies):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    n_p, n_c = 0, 0
    for t in all_trials(studies):
        iv = t.intermediate_values
        if not iv:
            continue
        xs = sorted(iv)
        ys = [iv[x] for x in xs]
        pruned = t.state.name == "PRUNED"
        if pruned:
            n_p += 1
            ax.plot(xs, ys, color="tab:red", alpha=0.45, lw=1, ls="--")
            ax.scatter(xs[-1], ys[-1], marker="x", color="tab:red", s=30)
        else:
            n_c += 1
            ax.plot(xs, ys, color="tab:blue", alpha=0.3, lw=1)
    ax.set_xlabel("Inner-fold step (0..4)")
    ax.set_ylabel("Running-mean inner AUC-ROC")
    ax.set_title(f"{LABEL} — intermediate values: {n_c} completed vs "
                 f"{n_p} pruned trials (MedianPruner, fold-step)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_intermediate.png", dpi=300)
    plt.close(fig)
    print(f"  intermediate: {n_c} completed, {n_p} pruned lines")


def fig_auc_vs_logloss(studies):
    ts = [t for t in done_trials(studies)
          if not np.isnan(t.user_attrs.get("logloss_mean", np.nan))]
    auc = np.array([t.value for t in ts])
    ll = np.array([t.user_attrs["logloss_mean"] for t in ts])
    rho, p = spearmanr(auc, ll)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(auc, ll, c=auc, cmap="viridis", s=22, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="trial inner AUC-ROC")
    ax.set_xlabel("Mean inner AUC-ROC (objective)")
    ax.set_ylabel("Mean inner logloss (user_attr)")
    ax.set_title(f"Trial-wise AUC vs logloss  (Spearman rho={rho:.2f}, "
                 f"p={p:.1e})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optuna_auc_vs_logloss.png", dpi=300)
    plt.close(fig)
    print(f"  AUC-logloss Spearman: rho={rho:.3f}, p={p:.2e} "
          f"({'metrics agree' if rho < -0.8 else 'check ranking'}; "
          f"logloss should anti-correlate)")


def fig_rank(studies):
    """Hand-rolled rank plot for categorical params: y = trial AUC rank
    (0 = best trial), x = param value. Readable where slice is not."""
    ts = done_trials(studies)
    vals = np.array([t.value for t in ts])
    order = np.argsort(vals)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(ts)) / max(len(ts) - 1, 1)   # 0 best .. 1 worst
    cats = [p for p in CAT_PARAMS[MODEL]
            if any(p in t.params for t in ts)]
    if not cats:
        return
    fig, axes = _panel_grid(cats)
    ncol = len(axes[0])
    for j, p in enumerate(cats):
        ax = axes[j // ncol][j % ncol]
        sub = [(t.params[p], rank[i]) for i, t in enumerate(ts) if p in t.params]
        xs, labels = _codes([s[0] for s in sub])
        ys = np.array([s[1] for s in sub])
        ax.scatter(xs, ys, s=14, alpha=0.7, c=ys, cmap="viridis_r")
        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
        elif p in LOG_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(p, fontsize=8)
        ax.set_ylabel("trial rank (0=best)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(cats), len(axes) * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(f"{LABEL} — trial rank vs categorical params "
                 "(pooled trials; bottom = better)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS / "optuna_rank.png", dpi=300)
    plt.close(fig)


def fig_stability():
    """Needs winner_stability.csv from the tuner --refit step."""
    csv_path = RUN_DIR / "winner_stability.csv"
    if not csv_path.exists():
        print("  winner_stability.csv not found — run the tuner --refit "
              "step first; figure skipped")
        return None
    rows = list(csv.DictReader(open(csv_path)))
    outers = [int(r["outer"]) for r in rows]
    inner = [float(r["inner_auc"]) for r in rows]
    outer_m = [float(r["outer_auc_mean"]) for r in rows]
    outer_s = [float(r["outer_auc_std"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, inner, 0.38, color="tab:blue", label="inner (tuning) AUC")
    ax.bar(x + 0.2, outer_m, 0.38, yerr=outer_s, capsize=3,
           color="tab:orange", label="outer refit AUC (multi-seed)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"outer {o}" for o in outers])
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"{LABEL} — per-outer winner stability "
                 "(gap inner-vs-outer = selection optimism; spread = noise)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS / "winner_stability.png", dpi=300)
    plt.close(fig)
    return {"outer_auc_mean": float(np.mean(outer_m)),
            "outer_auc_std": float(np.std(outer_m)),
            "inner_auc_mean": float(np.mean(inner))}


def main():
    global MODEL, FEATURES, RUN_DIR, PLOTS, LABEL
    ap = argparse.ArgumentParser(description="05f_cnn_optuna_plot: shared "
                                            "Optuna figures for CNN + FFNN")
    ap.add_argument("--model", choices=["cnn", "ffnn"], default="cnn",
                    help="which tuner's campaign to plot (study prefix + "
                         "runs dir follow the model)")
    ap.add_argument("--features", default="emb_esmc300m_win_padtoken",
                   help="feature-set key; selects runs/<model>_optuna_<features>/")
    ap.add_argument("--study-db", default="study.db",
                    help="study db filename under the arm's runs dir "
                         "(default study.db; smoke studies use their own)")
    ap.add_argument("--outer", type=int, default=None,
                   help="plot a single outer study instead of pooling all 6")
    args = ap.parse_args()

    MODEL, FEATURES = args.model, args.features
    RUN_DIR = arm_dir(MODEL, FEATURES)
    PLOTS = RUN_DIR / "plots"
    LABEL = f"{MODEL.upper()} {FEATURES}"
    if not RUN_DIR.exists():
        sys.exit(f"no run dir {RUN_DIR} — run the tuner first")
    PLOTS.mkdir(parents=True, exist_ok=True)

    studies = load_studies(args.study_db, args.outer)
    fig_history(studies)
    imp = fig_importance(studies)
    fig_slice(studies)
    fig_contour(studies)
    fig_intermediate(studies)
    fig_auc_vs_logloss(studies)
    fig_rank(studies)
    stab = fig_stability()

    best = max((t for s in studies for t in completed(s)),
               key=lambda t: t.value, default=None)
    summary = {
        "model": MODEL, "features": FEATURES, "study_db": args.study_db,
        "n_studies": len(studies),
        "n_trials": sum(len(s.trials) for s in studies),
        "n_completed": sum(len(completed(s)) for s in studies),
        "importance_mean": imp,
        "best_inner_auc": best.value if best else None,
        "best_params": best.params if best else None,
        "best_study": best.study_name() if best else None,
        "nested_refit": stab,
    }
    (RUN_DIR / "optuna_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary -> {RUN_DIR / 'optuna_summary.json'}")
    print(f"Figures -> {PLOTS}/optuna_*.png"
          + ("" if stab else " (winner_stability.png pending --refit)"))


if __name__ == "__main__":
    main()
