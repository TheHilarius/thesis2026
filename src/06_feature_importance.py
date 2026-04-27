#!/usr/bin/env python3
"""
06_feature_importance.py
Extract and visualize LR coefficients and RF feature importances
from any feature set with saved per-fold models.

Usage:
    python src/06_feature_importance.py --features handcrafted_sparse
    python src/06_feature_importance.py --features handcrafted_blosum
    python src/06_feature_importance.py --features handcrafted_sparse_esmc --top 50
    python src/06_feature_importance.py --features handcrafted_sparse --models lr
    python src/06_feature_importance.py --features handcrafted_sparse --models rf
    python src/06_feature_importance.py --features handcrafted_sparse --models lr rf
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from matplotlib.patches import Patch

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Feature importance extraction & visualization")
parser.add_argument('--features', required=True,
                    help="Feature set tag (e.g. handcrafted_sparse, handcrafted_blosum)")
parser.add_argument('--models', nargs='+', default=['lr', 'rf'],
                    help="Which models to analyze (default: lr rf)")
parser.add_argument('--top', type=int, default=40,
                    help="Number of top features to show in plots (default: 40)")
parser.add_argument('--n_folds', type=int, default=5,
                    help="Number of CV folds (default: 5)")
args = parser.parse_args()

TAG = args.features
TOP_N = args.top
N_FOLDS = args.n_folds
MODELS = args.models

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
OUT_FIG = Path("results/figures/models")
OUT_TAB = Path("results/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

print(f"Feature set: {TAG}")
print(f"Models:      {MODELS}")
print(f"Top N:       {TOP_N}")

# ── Classify features ─────────────────────────────────────────────────────────
def classify_feature(name):
    """Classify a feature as handcrafted, sparse, blosum, or embedding."""
    # Sparse: {position}_{AA} where position is N1-N4, P1-P9, C1-C4
    # and AA is a single uppercase letter
    parts = name.split('_')
    if len(parts) == 2:
        pos, aa = parts
        if pos in ('N1','N2','N3','N4','P1','P2','P3','P4','P5',
                    'P6','P7','P8','P9','C1','C2','C3','C4'):
            if len(aa) == 1 and aa.isupper():
                return 'sparse_encoding'
    # BLOSUM: {position}_{AA} same pattern but values are continuous
    # Check if it matches the blosum naming convention
    if len(parts) == 2:
        pos, rest = parts
        if pos in ('N1','N2','N3','N4','P1','P2','P3','P4','P5',
                    'P6','P7','P8','P9','C1','C2','C3','C4'):
            return 'aa_encoding'
    # Embedding PCA components
    if name.startswith(('esmc_', 'esmif_')):
        return 'embedding_pca'
    if 'PC' in name:
        return 'embedding_pca'
    return 'handcrafted'

# ── Color scheme ──────────────────────────────────────────────────────────────
TYPE_COLORS = {
    'handcrafted':    '#2ecc71',
    'sparse_encoding': '#9b59b6',
    'aa_encoding':    '#9b59b6',
    'embedding_pca':  '#e67e22',
}
TYPE_LABELS = {
    'handcrafted':    'Handcrafted (structural)',
    'sparse_encoding': 'Sparse (AA one-hot)',
    'aa_encoding':    'AA encoding (sparse/BLOSUM)',
    'embedding_pca':  'Embedding (PCA)',
}

# ── Process each model ────────────────────────────────────────────────────────
results = {}

for model_key in MODELS:
    print(f"\n{'='*60}")
    print(f"  {model_key.upper()} — {TAG}")
    print(f"{'='*60}")

    # Load feature names
    fname_path = MODEL_DIR / f"{model_key}_{TAG}_feature_names.json"
    if not fname_path.exists():
        print(f"  SKIP: {fname_path} not found")
        continue

    with open(fname_path) as f:
        feature_names = json.load(f)
    print(f"  Features: {len(feature_names)}")

    # Load per-fold models and extract weights
    weights_all = []
    for fold in range(N_FOLDS):
        path = MODEL_DIR / f"{model_key}_{TAG}_model_fold{fold}.pkl"
        if not path.exists():
            print(f"  SKIP fold {fold}: {path} not found")
            continue
        with open(path, 'rb') as f:
            model = pickle.load(f)

        if model_key == 'lr':
            w = model.coef_.flatten()
        elif model_key == 'rf':
            w = model.feature_importances_
        else:
            # Try both
            if hasattr(model, 'coef_'):
                w = model.coef_.flatten()
            elif hasattr(model, 'feature_importances_'):
                w = model.feature_importances_
            else:
                print(f"  SKIP fold {fold}: no coef_ or feature_importances_")
                continue

        weights_all.append(w)
        print(f"  Fold {fold}: {len(w)} weights loaded")

    if not weights_all:
        print(f"  No folds loaded for {model_key}")
        continue

    weights = np.array(weights_all)
    w_mean = weights.mean(axis=0)
    w_std = weights.std(axis=0)

    # Build DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'weight_mean': w_mean,
        'weight_std': w_std,
        'weight_abs_mean': np.abs(w_mean),
    })
    df['feature_type'] = df['feature'].apply(classify_feature)

    results[model_key] = df

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = OUT_TAB / f"feature_importance_{model_key}_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\n  {'Feature Type':<25s} {'Count':>6s} {'Total |weight|':>15s} {'Share':>8s}")
    print(f"  {'-'*55}")
    total_abs = df['weight_abs_mean'].sum()
    for ftype in sorted(df['feature_type'].unique()):
        subset = df[df['feature_type'] == ftype]
        type_sum = subset['weight_abs_mean'].sum()
        pct = type_sum / total_abs * 100
        print(f"  {ftype:<25s} {len(subset):>6d} {type_sum:>15.4f} {pct:>7.1f}%")

    # ── Top features ──────────────────────────────────────────────────────
    if model_key == 'lr':
        sort_col = 'weight_abs_mean'
        xlabel = 'LR Coefficient (mean ± std across folds)'
        plot_signed = True
    else:
        sort_col = 'weight_abs_mean'
        xlabel = 'RF Feature Importance (mean ± std across folds)'
        plot_signed = False

    print(f"\n  Top 20 features:")
    for _, row in df.nlargest(20, sort_col).iterrows():
        if plot_signed:
            direction = "+" if row['weight_mean'] > 0 else "-"
            print(f"    {direction} {row['feature']:40s}  {row['weight_mean']:+.4f}  "
                  f"({row['feature_type']})")
        else:
            print(f"      {row['feature']:40s}  {row['weight_mean']:.4f}  "
                  f"({row['feature_type']})")

    # ── Individual plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(8, TOP_N * 0.28)))
    df_top = df.nlargest(TOP_N, sort_col)

    if plot_signed:
        df_top = df_top.sort_values('weight_mean')
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in df_top['weight_mean']]
        ax.barh(range(len(df_top)), df_top['weight_mean'],
                xerr=df_top['weight_std'], color=colors, alpha=0.8,
                capsize=2, ecolor='gray')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(f'Top {TOP_N} LR Coefficients — {TAG}\n'
                     f'Red = promotes presentation, Blue = inhibits')
    else:
        df_top = df_top.sort_values('weight_mean')
        colors = [TYPE_COLORS.get(t, '#999999') for t in df_top['feature_type']]
        ax.barh(range(len(df_top)), df_top['weight_mean'],
                xerr=df_top['weight_std'], color=colors, alpha=0.8,
                capsize=2, ecolor='gray')
        ax.set_xlabel(xlabel)
        ax.set_title(f'Top {TOP_N} RF Feature Importances — {TAG}')

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(df_top['feature'], fontsize=8)

    # Bold handcrafted features
    for i, (_, row) in enumerate(df_top.iterrows()):
        if row['feature_type'] == 'handcrafted':
            ax.get_yticklabels()[i].set_fontweight('bold')

    plt.tight_layout()
    fig_path = OUT_FIG / f"feature_importance_{model_key}_{TAG}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  Saved: {fig_path}")
    plt.close()

# ── Side-by-side comparison (if both LR and RF available) ─────────────────────
if 'lr' in results and 'rf' in results:
    print(f"\n{'='*60}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, 25 * 0.3)))
    n_side = min(25, TOP_N)

    for ax, (model_key, title_label) in zip(axes, [('lr', '|LR Coefficient|'),
                                                      ('rf', 'RF Importance')]):
        df_m = results[model_key]
        df_top = df_m.nlargest(n_side, 'weight_abs_mean').sort_values('weight_abs_mean')
        colors = [TYPE_COLORS.get(t, '#999999') for t in df_top['feature_type']]
        ax.barh(range(len(df_top)), df_top['weight_abs_mean'],
                color=colors, alpha=0.8)
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['feature'], fontsize=8)
        ax.set_xlabel(title_label)
        ax.set_title(f'Top {n_side} by {title_label}')

    # Legend
    used_types = set()
    for df_m in results.values():
        used_types.update(df_m['feature_type'].unique())
    legend_elements = [Patch(facecolor=TYPE_COLORS.get(t, '#999999'), alpha=0.8,
                             label=TYPE_LABELS.get(t, t))
                       for t in sorted(used_types) if t in TYPE_COLORS]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
               fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = OUT_FIG / f"feature_importance_comparison_{TAG}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  Saved: {fig_path}")
    plt.close()

print(f"\nDone.")
