#!/usr/bin/env python3
"""
31_model_compare.py — XGBoost vs simpler models (rf, lr_l2, lr_elasticnet)
on blosum50 and handcrafted_blosum50.

Answers: can a simpler model match XGBoost on the same features, and which
features does each model lean on (promoting vs inhibiting presentation)?

Part A  performance
  results/model_compare_metrics.tsv    pooled OOF AUC, outer mean +/- SD, runtime
  results/model_compare_delong.tsv     xgb vs each simple model, DeLong + Holm
  plots/model_compare_cv_auc.png       CV-AUC bars, star-only Holm brackets

Part B  feature importance (source: avg_feature_weights already inside each
        cv_results JSON = top-50 features by |mean weight| over all inner+outer
        fits; xgb uses gain, rf uses Gini impurity, lr uses coef_)
  results/model_compare_importance.tsv arm, feature, weight, direction, rank
  plots/importance_blosum50.png        4 panels, top 15, promote/inhibit colour
  plots/importance_handcrafted_blosum50.png

Direction convention:
  LR   -> sign(coefficient) (model's own signed weight)
  tree -> sign(Pearson r(feature, that arm's OOF probability)); gain/Gini are
          unsigned, so the correlation with the model's own predictions gives
          the direction the model pushes the feature in.

DeLong helpers are imported from 19_handcrafted_sweep_delong.py (repo
convention: one copy of the test per script is avoided by importing it).

Run from repo root:
  python src/investigation/31_model_compare.py
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline" / "python"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (PROJECT_ROOT, SPLIT_DATA_PATH, BLOSUM50_DATA_PATH,
                    PEPTIDE_COL, LABEL_COL)

# ── repo DeLong / Holm helpers ─────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location("d19", HERE / "19_handcrafted_sweep_delong.py")
d19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d19)
delong_full = d19.delong_full          # -> auc_a, auc_b, delta, se, z, p
holm_adjust = d19.holm_adjust
stars = d19.stars
load_model = d19.load_model            # -> dict(key, k, probs, auc, outer, ...)

# ── arms ───────────────────────────────────────────────────────────────────
CAMPAIGN = PROJECT_ROOT / "runs" / "investigate_diff_logo_pattern_w_simpler_models"
RESULTS = CAMPAIGN / "results"
PLOTS = CAMPAIGN / "plots"

FEATURE_SETS = ["blosum50", "handcrafted_blosum50"]
MODELS = ["xgb", "rf", "lr_l2", "lr_elasticnet"]

ARMS = [
    ("xgb", "blosum50",
     PROJECT_ROOT / "runs/encodings_sweep/models/flat"
                    "/cv_results_xgb_blosum50_flat_20261014_154653.json"),
    ("rf", "blosum50",
     CAMPAIGN / "models/flat/cv_results_rf_blosum50_flat_20261107_212947.json"),
    ("lr_l2", "blosum50",
     CAMPAIGN / "models/flat/cv_results_lr_l2_blosum50_flat_20261107_214749.json"),
    ("lr_elasticnet", "blosum50",
     CAMPAIGN / "models/flat/cv_results_lr_elasticnet_blosum50_flat_20261107_215546.json"),
    ("xgb", "handcrafted_blosum50",
     PROJECT_ROOT / "runs/handcrafted_sweep/models/flat"
                    "/cv_results_xgb_handcrafted_blosum50_flat_20261020_111939.json"),
    ("rf", "handcrafted_blosum50",
     CAMPAIGN / "models/flat/cv_results_rf_handcrafted_blosum50_flat_20261107_220546.json"),
    ("lr_l2", "handcrafted_blosum50",
     CAMPAIGN / "models/flat/cv_results_lr_l2_handcrafted_blosum50_flat_20261107_223447.json"),
    ("lr_elasticnet", "handcrafted_blosum50",
     CAMPAIGN / "models/flat"
                 "/cv_results_lr_elasticnet_handcrafted_blosum50_flat_20261108_210326.json"),
]

# palette reused from 05_model_analysis.py
COLORS = {"rf": "#2ecc71", "lr_l2": "#3498db",
          "lr_elasticnet": "#9b59b6", "xgb": "#e74c3c"}
PROMOTE, INHIBIT = "#D62839", "#255C99"   # chemistry palette red / blue
DISPLAY = {"xgb": "XGBoost", "rf": "Random Forest",
           "lr_l2": "LogReg (L2)", "lr_elasticnet": "LogReg (ElasticNet)"}
TOP_N = 15


def stars_label(p):
    try:
        return stars(p)
    except Exception:
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── load split + arms ──────────────────────────────────────────────────────
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(SPLIT_DATA_PATH)
    fold_vec = split["fold"].to_numpy(dtype=int)
    y = split["label"].to_numpy(dtype=int)
    print(f"split: {len(y)} rows, {int(y.sum())} positives, "
          f"{len(set(fold_vec))} folds")

    arms = {}
    for model, feat, path in ARMS:
        key = f"{model}/{feat}"
        if not path.exists():
            raise SystemExit(f"missing result JSON: {path}")
        m = load_model(str(path), fold_vec, y, key)
        m["model"], m["features"] = model, feat
        m["raw"] = json.loads(Path(path).read_text())
        arms[key] = m
        print(f"  {key:34s} AUC={m['auc']:.4f} "
              f"outer={np.mean(m['outer']):.4f}+-{np.std(m['outer'], ddof=1):.4f} "
              f"k={m['k']}")

    # ── A1: metrics table ──────────────────────────────────────────────────
    met = pd.DataFrame([
        dict(arm=key, model=m["model"], features=m["features"], n_features=m["k"],
             pooled_oof_auc=round(float(m["auc"]), 6),
             outer_mean=round(float(np.mean(m["outer"])), 6),
             outer_std=round(float(np.std(m["outer"], ddof=1)), 6),
             nested_cv_seconds=m["raw"].get("nested_cv_seconds"),
             n_weights_stored=len(m["raw"].get("avg_feature_weights") or {}))
        for key, m in arms.items()
    ])
    met_path = RESULTS / "model_compare_metrics.tsv"
    met.to_csv(met_path, sep="\t", index=False)
    print(f"\nwrote {met_path.relative_to(PROJECT_ROOT)}")

    # ── A2: DeLong xgb vs each simple model, Holm within feature set ───────
    dl_rows = []
    for feat in FEATURE_SETS:
        xkey = f"xgb/{feat}"
        others = [(k, m) for k, m in arms.items()
                  if k.endswith(f"/{feat}") and k != xkey]
        res = [(k, delong_full(y, arms[xkey]["probs"], m["probs"]))
               for k, m in others]
        adj = holm_adjust([r[5] for _, r in res])
        for (k, (a, b, delta, se, z, p)), pa in zip(res, adj):
            dl_rows.append(dict(
                features=feat, reference=xkey, comparison=k,
                auc_xgb=round(a, 6), auc_other=round(b, 6),
                delta=round(delta, 6), se=round(se, 6), z=round(z, 4),
                p_value=float(f"{p:.4e}"), p_holm=float(f"{pa:.4e}"),
                significance=stars_label(pa)))
            print(f"  delong {feat:22s} {k:32s} "
                  f"delta={delta:+.4f} pHolm={pa:.2e} {stars_label(pa)}")
    dl = pd.DataFrame(dl_rows)
    dl_path = RESULTS / "model_compare_delong.tsv"
    dl.to_csv(dl_path, sep="\t", index=False)
    print(f"wrote {dl_path.relative_to(PROJECT_ROOT)}")

    # ── A3: CV-AUC bar figure ──────────────────────────────────────────────
    plot_cv_auc(arms, dl)

    # ── B: feature importance ──────────────────────────────────────────────
    plot_importance(arms, split)

    print("\nDONE")


# ───────────────────────────────────────────────────────────────────────────
def plot_cv_auc(arms, dl):
    """CV-AUC bars (thesis convention) with star-only Holm brackets vs xgb."""
    order = [(m, f) for f in FEATURE_SETS for m in MODELS]
    xs, x_of = [], {}
    x = 0.0
    for m, f in order:
        x_of[(m, f)] = x
        xs.append(x)
        x += 1.0
        if m == MODELS[-1]:
            x += 0.9                      # gap between feature-set groups

    vals = [np.mean(arms[f"{m}/{f}"]["outer"]) for m, f in order]
    errs = [np.std(arms[f"{m}/{f}"]["outer"], ddof=1) for m, f in order]

    fig, ax = plt.subplots(figsize=(13, 6.0))
    for i, ((m, f), v, e) in enumerate(zip(order, vals, errs)):
        ax.bar(xs[i], v, yerr=e, width=0.72, color=COLORS[m],
               edgecolor="black", linewidth=0.6, capsize=3,
               error_kw=dict(lw=1.0, ecolor="black"))

    # brackets: level 0 = rf (shortest span), level 2 = elasticnet (longest)
    tops = [v + e for v, e in zip(vals, errs)]
    bots = [v - e for v, e in zip(vals, errs)]
    top, bot = max(tops), min(bots)
    span = (top - bot) or 0.05
    for feat in FEATURE_SETS:
        x0 = x_of[("xgb", feat)]
        for lvl, model in enumerate(["rf", "lr_l2", "lr_elasticnet"]):
            row = dl[(dl["features"] == feat) & (dl["comparison"] == f"{model}/{feat}")]
            if row.empty:
                continue
            p = float(row["p_holm"].iloc[0])
            yb = top + span * (0.14 + 0.13 * lvl)
            x1 = x_of[(model, feat)]
            ax.plot([x0, x0, x1, x1], [yb - span * 0.05, yb, yb, yb - span * 0.05],
                    color="black", lw=0.8)
            ax.text((x0 + x1) / 2, yb + span * 0.03, stars_label(p),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[m] for m, _ in order], fontsize=9.5,
                       rotation=18, ha="right")
    ax.set_ylabel("CV-AUC", fontsize=12)
    lo = bot
    ax.set_ylim(lo - span * 0.10, top + span * 0.75)

    # feature-set group labels below the (rotated) tick labels
    for f in FEATURE_SETS:
        centre = np.mean([x_of[(m, f)] for m in MODELS])
        ax.text(centre, -0.22, f, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10.5, style="italic",
                color="0.30", fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODELS]
    ax.legend(handles, [DISPLAY[m] for m in MODELS], loc="upper left",
              frameon=False, ncol=4, fontsize=9)
    ax.set_title("XGBoost vs simpler models — nested-CV AUC per feature set",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", lw=0.7)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.33,
            "Error bars = SD across 6 outer folds. Brackets = DeLong xgb vs each "
            "model, Holm-corrected within feature set.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="0.35")

    out = PLOTS / "model_compare_cv_auc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


# ───────────────────────────────────────────────────────────────────────────
def load_feature_matrix(split, needed):
    """BLOSUM columns live in df_blosum50.csv; handcrafted columns are already
    in the split CSV. Validates row alignment like 04_modelling does."""
    alt_cols = set()
    alt = None
    need_alt = {c for c in needed if c not in split.columns}
    if need_alt:
        alt = pd.read_csv(
            BLOSUM50_DATA_PATH,
            usecols=lambda c: c in need_alt or c in (PEPTIDE_COL, LABEL_COL))
        if len(alt) != len(split):
            raise SystemExit(f"row mismatch: split={len(split)} "
                             f"blosum csv={len(alt)}")
        if not (alt[PEPTIDE_COL].fillna("").to_numpy()
                == split[PEPTIDE_COL].fillna("").to_numpy()).all():
            raise SystemExit("peptide alignment check failed vs df_blosum50.csv")
        alt_cols = set(alt.columns) - {PEPTIDE_COL, LABEL_COL}
        print(f"  blosum csv: {len(need_alt)} columns needed, "
              f"{len(alt_cols)} loaded, alignment OK")
    return alt, alt_cols


def direction_sign(model, feat, probs, split, alt, alt_cols):
    """LR direction comes from its own signed coefficient; tree models are
    unsigned, so the sign of corr(feature, OOF probability) gives it."""
    if model.startswith("lr"):
        return 0                                    # handled by the caller
    if feat in alt_cols:
        x = alt[feat].to_numpy(dtype=float)
    elif feat in split.columns:
        x = split[feat].to_numpy(dtype=float)
    else:
        return 0
    mask = np.isfinite(x) & np.isfinite(probs)
    if mask.sum() < 2:
        return 0
    xs, ps = x[mask], probs[mask]
    if xs.std() == 0 or ps.std() == 0:
        return 0
    r = float(np.corrcoef(xs, ps)[0, 1])
    if not np.isfinite(r):
        return 0
    return 1 if r >= 0 else -1


def plot_importance(arms, split):
    # features needed for the tree-direction correlation
    needed = set()
    for m in arms.values():
        for f in (m["raw"].get("avg_feature_weights") or {}):
            needed.add(f)
    alt, alt_cols = load_feature_matrix(split, needed)

    rows = []
    for key, m in arms.items():
        weights = m["raw"].get("avg_feature_weights") or {}
        if not weights:
            print(f"  [WARN] {key}: no avg_feature_weights in JSON")
            continue
        signed_model = m["model"].startswith("lr")
        ranked = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
        for rank, (feat, w) in enumerate(ranked, 1):
            if signed_model:
                direction = 1 if w >= 0 else -1
                w_signed = float(w)                 # coef already carries sign
            else:
                direction = direction_sign(m["model"], feat, m["probs"],
                                           split, alt, alt_cols)
                w_signed = float(w) * direction
            rows.append(dict(arm=key, model=m["model"], features=m["features"],
                             rank=rank, feature=feat, weight_raw=float(w),
                             direction=direction, weight_signed=w_signed))
    imp = pd.DataFrame(rows)
    imp_path = RESULTS / "model_compare_importance.tsv"
    imp.to_csv(imp_path, sep="\t", index=False)
    print(f"wrote {imp_path.relative_to(PROJECT_ROOT)} ({len(imp)} rows)")

    for feat_set in FEATURE_SETS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ax, model in zip(axes.flat, MODELS):
            sub = imp[(imp["features"] == feat_set) & (imp["model"] == model)]
            sub = sub.nsmallest(TOP_N, "rank")
            if sub.empty:
                ax.set_visible(False)
                continue
            sub = sub.sort_values("weight_signed")   # ascending -> biggest on top
            cols = [PROMOTE if v >= 0 else INHIBIT for v in sub["weight_signed"]]
            ax.barh(sub["feature"], sub["weight_signed"], color=cols,
                    edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="black", lw=0.9)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=9)
            auc = np.mean(arms[f"{model}/{feat_set}"]["outer"])
            kind = ("coefficient" if model.startswith("lr")
                    else "gain x direction" if model == "xgb"
                    else "Gini x direction")
            ax.set_title(f"{DISPLAY[model]} — {kind}   (CV-AUC {auc:.4f})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("signed importance", fontsize=9)
            ax.grid(axis="x", color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Feature importance — {feat_set} "
                     f"(top {TOP_N} by |mean weight|, 36 inner+outer fits)",
                     fontsize=14, fontweight="bold", y=0.995)
        fig.text(0.5, 0.005,
                 "Red = promotes presentation (positive direction), "
                 "blue = inhibits. Direction: LR = sign(coefficient); "
                 "tree models = sign(Pearson r(feature, OOF probability)) — "
                 "gain/Gini are unsigned. Source: avg_feature_weights (top 50).",
                 ha="center", fontsize=9, color="0.30")
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))
        out = PLOTS / f"importance_{feat_set}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
