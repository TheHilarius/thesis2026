#!/usr/bin/env python3
"""
06_feature_importance.py
Extract and visualize feature weights/importances from any model
with saved per-fold artifacts.

Visual encoding:
    Color   = feature type (handcrafted / AA encoding / embedding PCA)
    Hatching = direction (only for signed models like LR)
                /// = promotes presentation (positive coefficient)
                \\\ = inhibits presentation (negative coefficient)
                none = unsigned importance (RF, XGB, etc.)

Usage:
    python src/06_feature_importance.py --features handcrafted_sparse
    python src/06_feature_importance.py --features handcrafted_sparse --top 50
    python src/06_feature_importance.py --features handcrafted_sparse_esmc --top 100
    python src/06_feature_importance.py --features handcrafted_sparse --models lr rf xgb
"""

import argparse
import json
import pickle
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from matplotlib.patches import Patch
from config import POSITION_AA_COLS

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Feature importance extraction & visualization")
parser.add_argument('--features', required=True,
                    help="Feature set tag (e.g. handcrafted_sparse, handcrafted_blosum)")
parser.add_argument('--models', nargs='+', default=['lr', 'rf'],
                    help="Which models to analyze (default: lr rf)")
parser.add_argument('--top', type=int, default=30,
                    help="Top features in the per-model plots (default: 30)")
parser.add_argument('--top_compare', type=int, default=None,
                    help="Top features in the comparison plot (default: --top value)")
parser.add_argument('--top_big', type=int, default=70,
                    help="Top features in the BIG comparison plot (default: 70)")
parser.add_argument('--n_folds', type=int, default=5,
                    help="Number of CV folds (default: 5)")
parser.add_argument('--xgb_importance', default='gain',
                    choices=['gain', 'weight', 'cover', 'total_gain', 'total_cover'],
                    help="XGBoost importance type (default: gain)")
parser.add_argument('--out_root', default='results/figures/models',
                    help="Root folder for figures (default: results/figures/models)")
args = parser.parse_args()

TAG = args.features
TOP_N = args.top
TOP_COMPARE = args.top_compare if args.top_compare is not None else args.top
TOP_BIG = args.top_big
N_FOLDS = args.n_folds
MODELS = args.models
XGB_IMP = args.xgb_importance

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
OUT_FIG = Path(args.out_root) / TAG
OUT_TAB = Path("results/tables") / TAG
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    'lr':   'Logistic Regression',
    'rf':   'Random Forest',
    'xgb':  'XGBoost',
    'lgbm': 'LightGBM',
    'svc':  'Linear SVM',
    'enet': 'ElasticNet',
}

print(f"Feature set:    {TAG}")
print(f"Models:         {MODELS}")
print(f"Top per-model:  {TOP_N}")
print(f"Top comparison: {TOP_COMPARE}")
print(f"Top big plot:   {TOP_BIG}")
print(f"Output dir:     {OUT_FIG}")

# ── Feature classification ────────────────────────────────────────────────────
POSITION_PREFIXES = {c.rstrip("0123456789") for c in POSITION_AA_COLS}


def classify_feature(name):
    parts = name.split('_')

    # AA positional encoding (sparse or BLOSUM)
    if len(parts) == 2:
        pos, aa = parts
        prefix = pos.rstrip("0123456789")
        if prefix in POSITION_PREFIXES:
            return 'aa_encoding'

    # Embedding PCA features
    if name.startswith(('esmc_', 'esmif_')) or 'PC' in name:
        return 'embedding_pca'

    # Default: handcrafted structural features
    return 'handcrafted'


TYPE_COLORS = {
    'handcrafted':   '#2ecc71',
    'aa_encoding':   '#9b59b6',
    'embedding_pca': '#e67e22',
}
TYPE_LABELS = {
    'handcrafted':   'Handcrafted (structural)',
    'aa_encoding':   'AA encoding (sparse/BLOSUM)',
    'embedding_pca': 'Embedding (PCA)',
}

# Hatching encodes direction for signed models
# HATCH_PROMOTE = '///'   # positive coefficient → promotes presentation
HATCH_INHIBIT = '\\\\'  # negative coefficient → inhibits presentation

# ── Tick helpers ──────────────────────────────────────────────────────────────
def set_integer_xticks(ax, values, min_step=1):
    if len(values) == 0:
        return
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = 0.05 * max(abs(vmin), abs(vmax), 1e-9)
    lo = math.floor(min(vmin, 0) - pad)
    hi = math.ceil(max(vmax, 0) + pad)
    span = hi - lo
    if span <= 12:
        step = 1
    elif span <= 25:
        step = 2
    elif span <= 60:
        step = 5
    else:
        step = 10
    step = max(step, min_step)
    ticks = np.arange(lo, hi + step, step)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.set_xlim(lo, hi)


def set_fractional_xticks(ax, values, n_ticks=8):
    if len(values) == 0:
        return
    vmax = float(np.max(values))
    if vmax <= 0:
        return
    raw_step = vmax / n_ticks
    exp = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    base = 10 ** exp
    for mult in (1, 2, 5, 10):
        if mult * base >= raw_step:
            step = mult * base
            break
    else:
        step = 10 * base
    hi = math.ceil(vmax / step) * step
    ticks = np.arange(0, hi + step, step)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.set_xlim(0, hi)

def set_asymmetric_xticks(ax, values, outlier_ratio=3.0, pad=0.05):
    """
    If the largest value is much bigger than the rest,
    compress the x-axis to reveal structure among non-dominant features.

    outlier_ratio : how many times larger the top value must be compared
                    to the second-largest to trigger compression.
    """
    if len(values) < 3:
        return

    values = np.asarray(values)
    vmax = float(np.max(values))
    sorted_vals = np.sort(values)
    v2 = sorted_vals[-2]  # second-largest value

    if v2 > 0 and vmax / v2 >= outlier_ratio:
        hi = v2 * (1 + pad)
    else:
        hi = vmax * (1 + pad)

    ax.set_xlim(0, hi)


# ── Model dispatcher ──────────────────────────────────────────────────────────
def unwrap_pipeline(model):
    if hasattr(model, 'named_steps'):
        return list(model.named_steps.values())[-1]
    if hasattr(model, 'steps'):
        return model.steps[-1][1]
    return model

def extract_weights(model, model_key, n_features, xgb_importance='gain'):
    est = unwrap_pipeline(model)
    if model_key == 'xgb' or est.__class__.__name__ in ('XGBClassifier', 'XGBRegressor'):
        try:
            booster = est.get_booster()
            score_dict = booster.get_score(importance_type=xgb_importance)
            w = np.zeros(n_features)
            for k, v in score_dict.items():
                idx = int(k.lstrip('f'))
                if idx < n_features:
                    w[idx] = v
            return w, False
        except Exception as e:
            print(f"    [WARN] XGBoost booster extraction failed: {e}; falling back")
    if hasattr(est, 'coef_'):
        coef = est.coef_
        if coef.ndim == 1:
            return coef, True
        if coef.shape[0] == 1:
            return coef.flatten(), True
        print(f"    [INFO] multiclass coef_ shape {coef.shape} — aggregating |.| per feature")
        return np.linalg.norm(coef, axis=0), False
    if hasattr(est, 'feature_importances_'):
        return est.feature_importances_, False
    raise ValueError(f"Model {type(est).__name__} has no coef_ or feature_importances_")

# ── Plotting helper ───────────────────────────────────────────────────────────
def plot_panel(ax, df_top, signed, label, model_key, fontsize=8):
    """
    Draw a horizontal bar chart on `ax`.
    Color   = feature type
    Hatching = direction (signed models only)
    For signed models: bars are drawn at |value| but ordered/labeled with sign info.
    For unsigned models: bars are drawn at the raw importance.
    """
    # We always plot magnitude on x-axis so the two panels look comparable.
    # Direction is encoded via hatching for signed models.
    if signed:
        df_top = df_top.sort_values('weight_abs_mean', ascending=True)
        values = df_top['weight_abs_mean'].values
        stds = df_top['weight_std'].values
        signs = df_top['weight_mean'].values
        colors = [TYPE_COLORS.get(t, '#999999') for t in df_top['feature_type']]
        # Only inhibits gets a hatch; promotes is plain fill
        hatches = [HATCH_INHIBIT if s < 0 else None for s in signs]
    else:
        df_top = df_top.sort_values('weight_abs_mean', ascending=True)
        values = df_top['weight_mean'].values
        stds = df_top['weight_std'].values
        colors = [TYPE_COLORS.get(t, '#999999') for t in df_top['feature_type']]
        hatches = [None] * len(df_top)

    bars = ax.barh(range(len(df_top)), values,
                   xerr=stds, color=colors, alpha=0.85,
                   capsize=2, ecolor='gray', edgecolor='black', linewidth=0.4)

    # Apply hatches per-bar (matplotlib doesn't accept a list to barh's hatch arg)
    for bar, h in zip(bars, hatches):
        if h is not None:
            bar.set_hatch(h)

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'], fontsize=fontsize)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # Bold handcrafted feature labels
    for i, (_, row) in enumerate(df_top.iterrows()):
        if row['feature_type'] == 'handcrafted':
            ax.get_yticklabels()[i].set_fontweight('bold')

    # X-axis ticks
    if signed:
        # |coef| values are non-negative, but the spread is large (0–10)
        # Use integer-like ticks
        set_integer_xticks(ax, values)
        xlabel = f'{label} |Coefficient| (mean ± std across folds)'
    else:
        set_fractional_xticks(ax, values)
        set_asymmetric_xticks(ax, values)
        imp_label = f' ({XGB_IMP})' if model_key == 'xgb' else ''
        xlabel = f'{label} Importance{imp_label} (mean ± std across folds)'

    ax.set_xlabel(xlabel)
    return df_top

def build_legend(used_types, has_signed):
    """Build a unified legend: feature-type swatches + inhibits hatch."""
    elements = []
    for t in ('handcrafted', 'aa_encoding', 'embedding_pca'):
        if t in used_types:
            elements.append(Patch(facecolor=TYPE_COLORS[t], alpha=0.85,
                                  edgecolor='black', linewidth=0.4,
                                  label=TYPE_LABELS[t]))
    if has_signed:
        elements.append(Patch(facecolor='white', edgecolor='black', linewidth=0.4,
                              hatch=HATCH_INHIBIT,
                              label='Inhibits presentation (− coef)'))
        # Plain = promotes (implied by absence of hatch — no legend entry needed,
        # but we add a subtle one so it's explicit)
        elements.append(Patch(facecolor='white', edgecolor='black', linewidth=0.4,
                              label='Promotes presentation (+ coef)'))
    return elements

# ── Process each model ────────────────────────────────────────────────────────
results = {}

for model_key in MODELS:
    label = MODEL_LABELS.get(model_key, model_key.upper())
    print(f"\n{'='*60}")
    print(f"  {label} — {TAG}")
    print(f"{'='*60}")

    fname_path = MODEL_DIR / f"{model_key}_{TAG}_feature_names.json"
    if not fname_path.exists():
        print(f"  SKIP: {fname_path} not found")
        continue
    with open(fname_path) as f:
        feature_names = json.load(f)
    n_features = len(feature_names)
    print(f"  Features: {n_features}")

    weights_all = []
    is_signed = None
    for fold in range(N_FOLDS):
        path = MODEL_DIR / f"{model_key}_{TAG}_model_fold{fold}.pkl"
        if not path.exists():
            print(f"  SKIP fold {fold}: {path} not found")
            continue
        with open(path, 'rb') as f:
            model = pickle.load(f)
        try:
            w, signed = extract_weights(model, model_key, n_features, XGB_IMP)
        except Exception as e:
            print(f"  SKIP fold {fold}: {e}")
            continue
        if len(w) != n_features:
            print(f"  SKIP fold {fold}: weight length {len(w)} != {n_features}")
            continue
        if is_signed is None:
            is_signed = signed
        elif is_signed != signed:
            print(f"  [WARN] fold {fold} signedness mismatch — using first fold's choice")
        weights_all.append(w)
        print(f"  Fold {fold}: {len(w)} weights loaded "
              f"({'signed' if signed else 'unsigned'})")

    if not weights_all:
        print(f"  No folds loaded for {model_key}")
        continue

    weights = np.array(weights_all)
    w_mean = weights.mean(axis=0)
    w_std = weights.std(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'weight_mean': w_mean,
        'weight_std': w_std,
        'weight_abs_mean': np.abs(w_mean),
        'feature_type': [classify_feature(n) for n in feature_names],
    })
    df.attrs['is_signed'] = is_signed
    df.attrs['model_label'] = label
    df.attrs['model_key'] = model_key
    results[model_key] = df

    csv_path = OUT_TAB / f"feature_importance_{model_key}_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    total_abs = df['weight_abs_mean'].sum()
    print(f"\n  {'Feature Type':<25s} {'Count':>6s} {'Total |w|':>12s} {'Share':>8s}")
    print(f"  {'-'*55}")
    for ftype in sorted(df['feature_type'].unique()):
        subset = df[df['feature_type'] == ftype]
        s = subset['weight_abs_mean'].sum()
        print(f"  {ftype:<25s} {len(subset):>6d} {s:>12.4f} {s/total_abs*100:>7.1f}%")

    # ── Top features ──────────────────────────────────────────────────────
    print(f"\n  Top 20 features:")
    for _, row in df.nlargest(20, 'weight_abs_mean').iterrows():
        if is_signed:
            sign = "+" if row['weight_mean'] > 0 else "-"
            print(f"    {sign} {row['feature']:40s}  {row['weight_mean']:+.4f}  "
                  f"({row['feature_type']})")
        else:
            print(f"      {row['feature']:40s}  {row['weight_mean']:.4f}  "
                  f"({row['feature_type']})")

    # ── Individual plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(8, TOP_N * 0.28)))
    df_top = df.nlargest(TOP_N, 'weight_abs_mean').copy()
    plot_panel(ax, df_top, is_signed, label, model_key)

    used_types = set(df_top['feature_type'].unique())
    legend_elements = build_legend(used_types, has_signed=is_signed)
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

    direction_note = '\n(Hatching: /// = promotes, \\\\\\ = inhibits)' if is_signed else ''
    ax.set_title(f'Top {TOP_N} {label} — {TAG}{direction_note}')

    plt.tight_layout()
    fig_path = OUT_FIG / f"feature_importance_{model_key}_{TAG}_top{TOP_N}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  Saved: {fig_path}")
    plt.close()

# ── Pairwise comparison plots ─────────────────────────────────────────────────
def make_comparison(top_n, suffix=''):
    """Generate pairwise comparison plots showing `top_n` features."""
    if len(results) < 2:
        return
    for m1, m2 in itertools.combinations(results.keys(), 2):
        fig, axes = plt.subplots(1, 2, figsize=(20, max(8, top_n * 0.3)))

        used_types = set()
        has_signed = False

        for ax, mk in zip(axes, [m1, m2]):
            df_m = results[mk]
            signed = df_m.attrs.get('is_signed', False)
            label = df_m.attrs.get('model_label', mk.upper())

            df_top = df_m.nlargest(top_n, 'weight_abs_mean').copy()
            df_top_sorted = plot_panel(ax, df_top, signed, label, mk,
                                        fontsize=7 if top_n > 50 else 8)

            used_types.update(df_top['feature_type'].unique())
            has_signed = has_signed or signed
            ax.set_title(f'Top {top_n} — {label}')

        legend_elements = build_legend(used_types, has_signed)
        fig.legend(handles=legend_elements, loc='upper center',
                   ncol=len(legend_elements), fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = OUT_FIG / f"feature_importance_comparison_{m1}_vs_{m2}_{TAG}_top{top_n}{suffix}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"  Saved: {fig_path}")
        plt.close()

print(f"\n{'='*60}")
print(f"  COMPARISON PLOTS (top {TOP_COMPARE})")
print(f"{'='*60}")
make_comparison(TOP_COMPARE)

print(f"\n{'='*60}")
print(f"  BIG COMPARISON PLOT (top {TOP_BIG})")
print(f"{'='*60}")
make_comparison(TOP_BIG, suffix='_BIG')

print(f"\nDone. All outputs in: {OUT_FIG}")
