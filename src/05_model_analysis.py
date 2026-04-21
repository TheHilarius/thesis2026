"""
04_model_analysis.py
Post-modelling analysis and visualisation.

Reads the JSON results and pickled models produced by 03_modelling.py,
generates publication-ready figures and a detailed stats report.

Usage:
    python 04_model_analysis.py --results models/cv_results_rf_20260420_135549.json
    python 04_model_analysis.py --results models/cv_results_lr_20260420_140717.json
    python 04_model_analysis.py --results models/cv_results_rf_*.json models/cv_results_lr_*.json
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve as sk_calibration_curve
from datetime import datetime
from pathlib import Path

from config import (
    FIGURES_DIR, MODEL_DIR, LOG_DIR,
    N_CV_FOLDS,
)


# ──────────────────────────────────────────────
# 0. STYLE
# ──────────────────────────────────────────────

COLORS = {
    "rf": "#2ecc71",
    "lr": "#3498db",
    "xgb": "#e74c3c",
    "svm": "#9b59b6",
    "default": "#7f8c8d",
}

FOLD_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})


# ──────────────────────────────────────────────
# 1. LOADERS
# ──────────────────────────────────────────────

def load_results(results_path):
    """Load a cv_results JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def get_model_color(model_key):
    return COLORS.get(model_key, COLORS["default"])


# ──────────────────────────────────────────────
# 2. ROC CURVE (per-fold + mean)
# ──────────────────────────────────────────────

def plot_roc_curves(results, out_dir):
    """
    Plot per-fold ROC curves, the mean ROC, and the chance line.
    """
    model_key = results["model_key"]
    display = results["model_type"]
    fold_preds = results["fold_predictions"]

    fig, ax = plt.subplots(figsize=(7, 6))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 200)

    for fold_id_str, data in sorted(fold_preds.items(), key=lambda x: int(x[0])):
        fold_id = int(fold_id_str)
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, color=FOLD_COLORS[fold_id], alpha=0.4, lw=1,
                label=f"Fold {fold_id} (AUC = {fold_auc:.3f})")

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color=get_model_color(model_key), lw=2.5,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

    # Confidence band
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr,
                     np.clip(mean_tpr - std_tpr, 0, 1),
                     np.clip(mean_tpr + std_tpr, 0, 1),
                     color=get_model_color(model_key), alpha=0.15,
                     label="± 1 std")

    # Chance line
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {display} ({N_CV_FOLDS}-fold CV)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    path = out_dir / f"roc_curve_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    return mean_auc, std_auc


# ──────────────────────────────────────────────
# 3. PRECISION-RECALL CURVE (per-fold + mean)
# ──────────────────────────────────────────────

def plot_pr_curves(results, out_dir):
    """
    Plot per-fold Precision-Recall curves and the mean PR.
    """
    model_key = results["model_key"]
    display = results["model_type"]
    fold_preds = results["fold_predictions"]

    fig, ax = plt.subplots(figsize=(7, 6))

    precisions_interp = []
    aucs = []
    mean_recall = np.linspace(0, 1, 200)

    # Baseline = positive class prevalence
    all_y = []
    for data in fold_preds.values():
        all_y.extend(data["y_true"])
    prevalence = np.mean(all_y)

    for fold_id_str, data in sorted(fold_preds.items(), key=lambda x: int(x[0])):
        fold_id = int(fold_id_str)
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)

        # Interpolate (precision-recall is non-monotonic, flip for interp)
        interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions_interp.append(interp_prec)

        ax.plot(recall, precision, color=FOLD_COLORS[fold_id], alpha=0.4, lw=1,
                label=f"Fold {fold_id} (AUC = {pr_auc:.3f})")

    # Mean PR
    mean_prec = np.mean(precisions_interp, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_recall, mean_prec, color=get_model_color(model_key), lw=2.5,
            label=f"Mean PR (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

    std_prec = np.std(precisions_interp, axis=0)
    ax.fill_between(mean_recall,
                     np.clip(mean_prec - std_prec, 0, 1),
                     np.clip(mean_prec + std_prec, 0, 1),
                     color=get_model_color(model_key), alpha=0.15,
                     label="± 1 std")

    # Baseline
    ax.axhline(y=prevalence, color="k", ls="--", lw=1, alpha=0.5,
               label=f"Baseline (prevalence = {prevalence:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {display} ({N_CV_FOLDS}-fold CV)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    path = out_dir / f"pr_curve_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    return mean_auc, std_auc


# ──────────────────────────────────────────────
# 4. CONFUSION MATRICES
# ──────────────────────────────────────────────

def plot_confusion_matrices(results, out_dir):
    """
    Plot two confusion matrices side by side:
      - Aggregated CV (all fold predictions combined)
      - Held-out set (from held-out metrics)
    """
    model_key = results["model_key"]
    display = results["model_type"]
    fold_preds = results["fold_predictions"]
    ho_metrics = results["held_out_metrics"]

    # Aggregate CV predictions
    all_y_true = []
    all_y_prob = []
    for data in fold_preds.values():
        all_y_true.extend(data["y_true"])
        all_y_prob.extend(data["y_prob"])
    all_y_true = np.array(all_y_true)
    all_y_pred = (np.array(all_y_prob) >= 0.5).astype(int)

    cv_cm = confusion_matrix(all_y_true, all_y_pred)

    # Held-out CM from metrics
    sens = ho_metrics["sensitivity"]
    spec = ho_metrics["specificity"]
    ppv = ho_metrics["ppv"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Negative", "Positive"]

    for ax, cm, title in [
        (axes[0], cv_cm, f"CV Aggregated ({len(all_y_true)} samples)"),
    ]:
        _plot_single_cm(ax, cm, labels, title, get_model_color(model_key))

    # For held-out, we only have metrics, not raw predictions
    # Reconstruct CM from sensitivity/specificity if possible
    # But we also have fold predictions — check if held-out preds are saved
    # They aren't in the current results, so use the aggregated CV only
    # and put a metrics summary in the second panel
    ax2 = axes[1]
    _plot_held_out_metrics_panel(ax2, ho_metrics, display)

    fig.suptitle(f"Confusion Matrix & Held-Out Metrics — {display}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = out_dir / f"confusion_matrix_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_single_cm(ax, cm, labels, title, color):
    """Plot a single annotated confusion matrix heatmap."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = len(labels)
    ax.set(xticks=range(n_classes), yticks=range(n_classes),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label",
           title=title)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            pct = cm[i, j] / cm.sum() * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)


def _plot_held_out_metrics_panel(ax, ho_metrics, display):
    """Render held-out metrics as a clean text panel."""
    ax.axis("off")
    ax.set_title("Held-Out Set Metrics", fontsize=12)

    metrics_order = [
        ("AUC-ROC", "auc_roc"),
        ("AUC-PR", "auc_pr"),
        ("Accuracy", "accuracy"),
        ("F1", "f1"),
        ("MCC", "mcc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("PPV", "ppv"),
        ("NPV", "npv"),
    ]

    y_start = 0.90
    y_step = 0.085
    for i, (label, key) in enumerate(metrics_order):
        y = y_start - i * y_step
        val = ho_metrics[key]
        # Color code: green if good, red if poor
        if key in ("auc_roc", "auc_pr"):
            fc = "#27ae60" if val > 0.7 else "#e67e22" if val > 0.5 else "#e74c3c"
        elif key == "mcc":
            fc = "#27ae60" if val > 0.3 else "#e67e22" if val > 0.1 else "#e74c3c"
        else:
            fc = "black"

        ax.text(0.05, y, f"{label}:", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="center")
        ax.text(0.55, y, f"{val:.4f}", transform=ax.transAxes,
                fontsize=11, va="center", color=fc, fontfamily="monospace")


# ──────────────────────────────────────────────
# 5. PER-FOLD METRIC BAR CHART
# ──────────────────────────────────────────────

def plot_fold_metrics_bars(results, out_dir):
    """
    Grouped bar chart showing each metric across folds,
    with mean ± std overlay.
    """
    model_key = results["model_key"]
    display = results["model_type"]
    fold_metrics = results["fold_metrics"]

    metrics_to_plot = ["auc_roc", "auc_pr", "f1", "mcc", "sensitivity", "specificity"]
    n_folds = len(fold_metrics)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n_metrics)
    width = 0.12
    offsets = np.linspace(-(n_folds - 1) / 2 * width, (n_folds - 1) / 2 * width, n_folds)

    for fold_id, fold_m in enumerate(fold_metrics):
        vals = [fold_m[m] for m in metrics_to_plot]
        ax.bar(x + offsets[fold_id], vals, width, color=FOLD_COLORS[fold_id],
               alpha=0.7, label=f"Fold {fold_id}", edgecolor="white", lw=0.5)

    # Mean + std overlay
    for i, m in enumerate(metrics_to_plot):
        vals = [fm[m] for fm in fold_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ax.errorbar(x[i], mean_val, yerr=std_val, fmt="D", color="black",
                     markersize=6, capsize=4, capthick=1.5, zorder=5,
                     label="Mean ± std" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics_to_plot])
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Fold Metrics — {display}")
    ax.legend(loc="lower left", ncol=3, framealpha=0.9)
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / f"fold_metrics_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 6. FEATURE WEIGHTS
# ──────────────────────────────────────────────

def plot_feature_weights(results, out_dir, top_n=25):
    """
    Horizontal bar chart of top features by importance/coefficient.
    """
    model_key = results["model_key"]
    display = results["model_type"]

    weights = results.get("avg_feature_weights")
    if not weights:
        print("  (no feature weights to plot)")
        return

    # Sort by absolute value
    sorted_feats = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_feats = sorted_feats[:top_n]
    sorted_feats.reverse()  # bottom-to-top for horizontal bar

    names = [f[0] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))

    colors = [get_model_color(model_key) if v >= 0 else "#e74c3c" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    is_coef = any(v < 0 for v in vals)
    xlabel = "Coefficient (log-odds)" if is_coef else "Gini Importance"
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {len(names)} Features — {display}")
    ax.axvline(x=0, color="black", lw=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    path = out_dir / f"feature_weights_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 7. SCORE DISTRIBUTION (HISTOGRAM)
# ──────────────────────────────────────────────

def plot_score_distributions(results, out_dir):
    """
    Histogram of predicted probabilities, split by true label.
    Aggregated across all CV folds.
    """
    model_key = results["model_key"]
    display = results["model_type"]
    fold_preds = results["fold_predictions"]

    all_y_true = []
    all_y_prob = []
    for data in fold_preds.values():
        all_y_true.extend(data["y_true"])
        all_y_prob.extend(data["y_prob"])
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, 1, 51)
    ax.hist(all_y_prob[all_y_true == 0], bins=bins, alpha=0.6,
            color="#3498db", label="Negative", density=True, edgecolor="white", lw=0.3)
    ax.hist(all_y_prob[all_y_true == 1], bins=bins, alpha=0.6,
            color="#e74c3c", label="Positive", density=True, edgecolor="white", lw=0.3)

    ax.axvline(x=0.5, color="black", ls="--", lw=1, alpha=0.7, label="Threshold (0.5)")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {display} (CV aggregated)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = out_dir / f"score_distribution_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 8. CALIBRATION PLOT
# ──────────────────────────────────────────────

def plot_calibration(results, out_dir, n_bins=10):
    """
    Reliability diagram: predicted probability vs observed frequency.
    """
    model_key = results["model_key"]
    display = results["model_type"]
    fold_preds = results["fold_predictions"]

    all_y_true = []
    all_y_prob = []
    for data in fold_preds.values():
        all_y_true.extend(data["y_true"])
        all_y_prob.extend(data["y_prob"])
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                     gridspec_kw={"height_ratios": [3, 1]})

    prob_true, prob_pred = sk_calibration_curve(
        all_y_true, all_y_prob, n_bins=n_bins, strategy="uniform",
    )

    ax1.plot(prob_pred, prob_true, "s-", color=get_model_color(model_key),
             lw=2, markersize=6, label=display)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfectly calibrated")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"Calibration Plot — {display}")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Histogram of predictions below
    ax2.hist(all_y_prob, bins=50, color=get_model_color(model_key),
             alpha=0.6, edgecolor="white", lw=0.3)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"calibration_{model_key}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 9. SUMMARY REPORT (text)
# ──────────────────────────────────────────────

def write_summary_report(results, out_dir):
    """Write a plain-text stats summary alongside the figures."""
    model_key = results["model_key"]
    display = results["model_type"]
    cv_summary = results["cv_summary"]
    ho_metrics = results["held_out_metrics"]
    fold_metrics = results["fold_metrics"]

    path = out_dir / f"summary_report_{model_key}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append(f"  MODEL ANALYSIS REPORT — {display.upper()}")
    lines.append("=" * 70)
    lines.append(f"  Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model key:    {model_key}")
    lines.append(f"  Model class:  {results['model_class']}")
    lines.append(f"  Features:     {results['config']['n_features']}")
    lines.append(f"  CV folds:     {len(fold_metrics)}")
    lines.append(f"  Best fold:    {results['best_fold_id']} "
                 f"(AUC = {results['best_fold_auc']:.4f})")
    lines.append("")

    # CV summary
    lines.append("-" * 70)
    lines.append("CROSS-VALIDATION RESULTS (mean ± std)")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<15} {'Mean':>10} {'Std':>10}")
    lines.append(f"  {'-' * 37}")
    for m, stats in cv_summary.items():
        lines.append(f"  {m:<15} {stats['mean']:>10.4f} {stats['std']:>10.4f}")

    lines.append("")

    # Held-out
    lines.append("-" * 70)
    lines.append("HELD-OUT SET RESULTS")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<15} {'Value':>10}")
    lines.append(f"  {'-' * 27}")
    for m, v in ho_metrics.items():
        lines.append(f"  {m:<15} {v:>10.4f}")

    lines.append("")

    # CV vs held-out delta
    lines.append("-" * 70)
    lines.append("CV vs HELD-OUT DELTA")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<15} {'CV mean':>10} {'Held-out':>10} {'Delta':>10}")
    lines.append(f"  {'-' * 47}")
    for m, stats in cv_summary.items():
        ho_val = ho_metrics.get(m, float("nan"))
        delta = ho_val - stats["mean"]
        lines.append(f"  {m:<15} {stats['mean']:>10.4f} {ho_val:>10.4f} {delta:>+10.4f}")

    lines.append("")

    # Per-fold table
    lines.append("-" * 70)
    lines.append("PER-FOLD METRICS")
    lines.append("-" * 70)
    metric_names = list(fold_metrics[0].keys())
    header = f"  {'Fold':<6}" + "".join(f"{m:<13}" for m in metric_names)
    lines.append(header)
    lines.append(f"  {'-' * (6 + 13 * len(metric_names))}")
    for i, fm in enumerate(fold_metrics):
        row = f"  {i:<6}" + "".join(f"{fm[m]:<13.4f}" for m in metric_names)
        lines.append(row)

    lines.append("")

    # Hyperparameters
    lines.append("-" * 70)
    lines.append("HYPERPARAMETERS")
    lines.append("-" * 70)
    for k, v in results["config"].get("hyperparameters", {}).items():
        lines.append(f"  {k:<25} {v}")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Saved: {path}")

    return report_text


# ──────────────────────────────────────────────
# 10. MULTI-MODEL COMPARISON (if multiple results)
# ──────────────────────────────────────────────

def plot_model_comparison(all_results, out_dir):
    """
    Side-by-side bar chart comparing CV metrics across models.
    Only generated when >1 result file is provided.
    """
    if len(all_results) < 2:
        return

    metrics_to_compare = ["auc_roc", "auc_pr", "f1", "mcc"]
    n_models = len(all_results)
    n_metrics = len(metrics_to_compare)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_metrics)
    width = 0.7 / n_models

    for i, res in enumerate(all_results):
        model_key = res["model_key"]
        display = res["model_type"]
        cv_summary = res["cv_summary"]

        means = [cv_summary[m]["mean"] for m in metrics_to_compare]
        stds = [cv_summary[m]["std"] for m in metrics_to_compare]

        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               color=get_model_color(model_key), alpha=0.8,
               capsize=3, edgecolor="white", lw=0.5,
               label=f"{display} (CV)")

        # Held-out as markers
        ho_vals = [res["held_out_metrics"][m] for m in metrics_to_compare]
        ax.scatter(x + offset, ho_vals, marker="D", s=40, color="black",
                   zorder=5, label=f"{display} (held-out)" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics_to_compare])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — CV Mean (bars) vs Held-Out (diamonds)")
    ax.legend(framealpha=0.9)
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / "model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# 11. CLI + MAIN
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="04_model_analysis: generate figures and reports from model results",
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="Path(s) to cv_results_*.json file(s)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("  04_MODEL_ANALYSIS — POST-MODELLING FIGURES & STATS")
    print("=" * 70)
    print(f"  Output dir: {FIGURES_DIR}")
    print(f"  Results files: {len(args.results)}")

    all_results = []

    for results_path in args.results:
        results_path = Path(results_path)
        if not results_path.exists():
            print(f"\n  WARNING: File not found, skipping: {results_path}")
            continue

        results = load_results(results_path)
        all_results.append(results)

        model_key = results["model_key"]
        display = results["model_type"]

        print(f"\n{'=' * 70}")
        print(f"  Analysing: {display} ({model_key})")
        print(f"  Source: {results_path}")
        print(f"{'=' * 70}")

        # Generate all plots for this model
        plot_roc_curves(results, FIGURES_DIR)
        plot_pr_curves(results, FIGURES_DIR)
        plot_confusion_matrices(results, FIGURES_DIR)
        plot_fold_metrics_bars(results, FIGURES_DIR)
        plot_feature_weights(results, FIGURES_DIR)
        plot_score_distributions(results, FIGURES_DIR)
        plot_calibration(results, FIGURES_DIR)
        write_summary_report(results, FIGURES_DIR)

    # Multi-model comparison
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  Generating multi-model comparison ({len(all_results)} models)")
        print(f"{'=' * 70}")
        plot_model_comparison(all_results, FIGURES_DIR)

    # Footer
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {len(all_results)} model(s) analysed")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 70}")