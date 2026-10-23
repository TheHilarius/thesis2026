#!/usr/bin/env python3
"""
pca_sweep_AUC_delong_comparison.py
AUC-vs-PC-count comparison for the two PCA k-sweeps, with paired DeLong.

Campaigns (both nested-CV XGB, flat PCA, one padmode each):
    esmif_eos_sweep     ESM-IF EOS/BOS-repeat   k = 47..11243 (30-99% EV)
    esmc300m_pca_sweep  ESM-C-300M pad-token    k = 7..9854 (30-95% EV);
                        k=17909 (99% EV) still running — drawn as a
                        pending ghost slot until its JSON lands

Per campaign emits (into <campaign>/plots/), --refs both (default):
    pca_sweep_auc_full.png              bars; DeLong brackets (stars only,
                                        only for k significantly different
                                        from the best-AUC reference k),
                                        BLOSUM50 + Sparse baseline lines
    pca_sweep_auc_full_nobaselines.png  same without the baseline lines
    pca_sweep_auc_line.png              clean log-x line + markers,
                                        baselines, no p-values
    pca_sweep_auc_line_nobaselines.png  same without baselines
    pca_sweep_runtime.png               core-hours per k (log-y),
                                        fold-seconds paired t-test vs the
                                        best-AUC reference k; peak memory
                                        was NOT recorded (JobAcctGather=none)
    pca_sweep_results.csv               per-k metrics + runtime
                                        (maxrss=NA except live-measured
                                        points)
    pca_sweep_delong.csv                every pairwise DeLong test in the
                                        campaign (all_pairs, vs_anchor,
                                        vs_best), Holm-adjusted per family;
                                        columns campaign, family, family_m,
                                        k1, ev1, k2, ev2, auc1, auc2,
                                        delta, z, p, p_adj

All figures: title "PCA dimension sweep: <campaign>" (runtime: "Compute
cost by PCA dimensionality: ..."), x = "Number of PCs" with ticks
"k (EV%)", y = "Held-out AUC".  No in-figure captions — thesis figure
captions live outside the plots.

Reference k (hatched bar / open marker / bold tick / t-test) = best-AUC
ladder k.  The 90%-EV anchor (esmif 4308 external JSON, esmc300 5883)
stays a regular data point + its own DeLong family in the CSVs.

Optimum-PC rule (printed + CSV): smallest ladder k statistically
indistinguishable (paired DeLong, Holm) from the best-AUC ladder k.
"""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
for p in (str(SRC_DIR), str(PIPELINE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.stats import norm, t as t_dist, ttest_rel
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH, EMBEDDING_DIR

# Nadeau-Bengio corrected resampled t for 6-fold CV (n_test/n_train=1/5)
NB_FACTOR = np.sqrt(2.2)
NB_DF = 5

PADMASK_FILES = {
    "ESM-C": EMBEDDING_DIR / "esmc_context_embeddings_zeropad.h5",
    "ESM-IF": EMBEDDING_DIR / "esm-if_test_zero.h5",
}

# Baseline XGB runs (flat, same nested-CV protocol) drawn as reference lines.
REF_JSONS = {
    "BLOSUM50": (PROJECT_ROOT / "runs" / "padmode_choice" / "models" / "flat" /
                 "cv_results_xgb_blosum_flat_20261007_143220.json"),
    "Sparse": (PROJECT_ROOT / "runs" / "padmode_choice" / "models" / "flat" /
               "cv_results_xgb_sparse_flat_20261007_142819.json"),
}

CAMPAIGNS = {
    "esmif_eos_sweep": {
        "models": PROJECT_ROOT / "runs" / "esmif_eos_sweep" / "models" / "flat",
        "out": PROJECT_ROOT / "runs" / "esmif_eos_sweep" / "plots",
        "ladder": [47, 176, 345, 632, 850, 1143, 1548,
                   2125, 2974, 6700, 11243],
        "ev": {47: 30, 176: 40, 345: 50, 632: 60, 850: 65, 1143: 70,
               1548: 75, 2125: 80, 2974: 85, 6700: 95, 11243: 99,
               4308: 90},
        "mask": "ESM-IF",
        "color": "#4477AA",
        "title": "ESM-IF EOS/BOS-repeat",
        "anchor": 4308,
        "anchor_json": (PROJECT_ROOT / "runs" / "padmode_choice" /
                        "models" / "flat" /
                        "cv_results_xgb_emb_esmif_eos_bos_repeat_pca4308_flat_20261008_032609.json"),
        "wall_h": {47: 1.805, 176: 1.738, 345: 1.925, 632: 3.885,
                   850: 3.157, 1143: 3.729, 1548: 6.526,
                   2125: 5.936, 2974: 11.353, 6700: 22.672,
                   11243: 34.325, 4308: 10.507},
        "cores": {47: 8, 176: 8, 345: 8, 632: 8, 850: 8, 1143: 8,
                  1548: 12, 2125: 12, 2974: 12, 6700: 12,
                  11243: 12, 4308: 12},
    },
    "esmc300m_pca_sweep": {
        "models": PROJECT_ROOT / "runs" / "esmc300m_pca_sweep" / "models" / "flat",
        "out": PROJECT_ROOT / "runs" / "esmc300m_pca_sweep" / "plots",
        "ladder": [7, 29, 96, 291, 507, 856, 1403,
                   2278, 3663, 5883, 9854, 17909],
        "ev": {7: 30, 29: 40, 96: 50, 291: 60, 507: 65, 856: 70,
               1403: 75, 2278: 80, 3663: 85, 5883: 90, 9854: 95,
               17909: 99},
        "mask": "ESM-C",
        "color": "#EE6677",
        "title": "ESM-C-300M pad-token",
        "anchor": 5883,
        "anchor_json": None,  # in-ladder
        "wall_h": {7: 2.584, 29: 2.914, 96: 3.051, 291: 3.351,
                   507: 4.124, 856: 4.769, 1403: 7.711,
                   2278: 14.484, 3663: 9.557, 5883: 13.513,
                   9854: 27.545, 17909: 61.434},
        "cores": {7: 8, 29: 8, 96: 8, 291: 8, 507: 8, 856: 8,
                  1403: 12, 2278: 12, 3663: 12, 5883: 12,
                  9854: 12, 17909: 12},
    },
}


# ── stats (inlined from 15_padmode_significance — its import chain is
#    broken upstream; proven code, copied verbatim) ──────────────────
def delong_structure(y, p):
    pos = p[y == 1]
    neg = p[y == 0]
    m, n = len(pos), len(neg)
    neg_sorted = np.sort(neg)
    left = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (left + 0.5 * (right - left)) / n
    pos_sorted = np.sort(pos)
    leftp = np.searchsorted(pos_sorted, neg, side="left")
    rightp = np.searchsorted(pos_sorted, neg, side="right")
    v01 = 1.0 - (leftp + 0.5 * (rightp - leftp)) / m
    return v10, v01


def delong_test(y, p_a, p_b):
    v10a, v01a = delong_structure(y, p_a)
    v10b, v01b = delong_structure(y, p_b)
    auc_a, auc_b = float(v10a.mean()), float(v10b.mean())
    m, n = len(v10a), len(v01a)
    s10a = np.var(v10a, ddof=1) / m
    s10b = np.var(v10b, ddof=1) / m
    s01a = np.var(v01a, ddof=1) / n
    s01b = np.var(v01b, ddof=1) / n
    cov10 = np.cov(v10a, v10b, ddof=1)[0, 1] / m
    cov01 = np.cov(v01a, v01b, ddof=1)[0, 1] / n
    var_d = s10a + s01a + s10b + s01b - 2.0 * (cov10 + cov01)
    return auc_a, auc_b, float(var_d)


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full(len(pvals), np.nan)
    valid = ~np.isnan(pvals)
    pv = pvals[valid]
    m = len(pv)
    order = np.argsort(pv)
    running = 0.0
    out = np.empty(m)
    for rank, i in enumerate(order):
        val = (m - rank) * pv[i]
        running = max(running, val)
        out[i] = min(1.0, running)
    adj[valid] = out
    return adj


def delong_p(y, pa, pb):
    auc_a, auc_b, var_d = delong_test(y, pa, pb)
    d = auc_a - auc_b
    if var_d <= 0:
        z, p = 0.0, 1.0
    else:
        z = d / np.sqrt(var_d)
        p = float(2 * norm.sf(abs(z)))
    return auc_a, auc_b, d, z, p


def fold_t_nb(fa, fb):
    """Paired t on 30 fold-seconds, Nadeau-Bengio corrected."""
    fa, fb = np.asarray(fa, float), np.asarray(fb, float)
    t_stat, p = ttest_rel(fa, fb)
    t_adj = float(t_stat) / NB_FACTOR
    p_corr = float(2 * t_dist.sf(abs(t_adj), df=NB_DF))
    return float(np.mean(fa)), float(np.mean(fb)), float(t_stat), p_corr


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── data ────────────────────────────────────────────────────────────
def load_padded_mask(toolkit):
    """Informative padded rows (>=1 pad AND >=1 real slot), df order.
    Same construction as 14_plot_grid_auc.load_padded_masks."""
    path = PADMASK_FILES[toolkit]
    with h5py.File(path, "r") as f:
        pm = f["pad_mask"][:].astype(bool)
        ri = f["row_indices"][:]
    assert np.array_equal(np.sort(ri), np.arange(len(ri))), path.name
    informative = pm.any(axis=1) & ~pm.all(axis=1)
    by_df = np.zeros(len(ri), dtype=bool)
    by_df[ri] = informative
    return by_df


def reconstruct(res, fold_vec, y, tag):
    n = len(fold_vec)
    probs = np.full(n, np.nan)
    for fk, entry in res["outer_val_predictions"].items():
        mask = fold_vec == int(fk)
        yt = np.asarray(entry["y_true"], dtype=int)
        if mask.sum() != len(entry["y_prob"]):
            raise ValueError(f"{tag}: fold {fk} size mismatch")
        if not (yt == y[mask]).all():
            raise ValueError(f"{tag}: fold {fk} labels mismatch df")
        probs[mask] = np.asarray(entry["y_prob"], dtype=float)
    if np.isnan(probs).any():
        raise ValueError(f"{tag}: rows never predicted")
    return probs


def load_model(path, fold_vec, y, tag):
    with open(path) as fh:
        res = json.load(fh)
    k = int(res["config"]["n_features"])
    probs = reconstruct(res, fold_vec, y, tag)
    outer = [float(m["auc_roc"]) for m in res["outer_val_metrics"]]
    assert len(outer) == 6 and len(res["fold_seconds"]) == 30, path.name
    return {
        "k": k,
        "probs": probs,
        "outer": outer,
        "fold_seconds": [float(s) for s in res["fold_seconds"]],
        "file": path.name,
    }


def holm_family(pairs, key="p"):
    adj = holm_adjust([pr[key] for pr in pairs])
    for pr, a in zip(pairs, adj):
        pr["p_adj"] = float(a)
    return pairs


# ── plotting helpers ────────────────────────────────────────────────
# Baseline line styles. Both are AA encodings — same Tol-grey as the
# "AA encoding" bars in plot_padmode_data_compare (#BBBBBB); the two are
# told apart by linestyle (dashed vs dotted).
REF_STYLE = {
    "BLOSUM50": dict(color="#BBBBBB", ls="--"),
    "Sparse": dict(color="#BBBBBB", ls=":"),
}


def draw_reference_lines(ax, refs, min_sep=0.035):
    """Horizontal baseline lines + direct labels in the right margin."""
    if not refs:
        return
    items = sorted(refs.items(), key=lambda kv: kv[1])
    label_y = []
    for _, v in items:  # nudge labels apart; the line itself stays at v
        label_y.append(v if not label_y else max(v, label_y[-1] + min_sep))
    for (name, v), ly in zip(items, label_y):
        st = REF_STYLE.get(name, dict(color="0.4", ls="-."))
        ax.axhline(v, color=st["color"], ls=st["ls"], lw=1.1, zorder=3.5)
        ax.text(1.008, ly, f"{name} {v:.4f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=st["color"])


def draw_bracket(ax, x0, x1, y, text, h, fontsize=9, ha="center", x=None):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h],
            color="0.35", lw=0.8, zorder=3, solid_joinstyle="miter")
    if text:
        ax.text(x if x is not None else (x0 + x1) / 2, y, text,
                ha=ha, va="bottom", fontsize=fontsize, color="0.2", zorder=3)


def _sig_label(pr):
    """Bracket label: stars only (thresholds live in the thesis caption);
    brackets are only drawn for significant pairs anyway."""
    return stars(pr["p_adj"])


def _check_label_overlaps(fig, tag=""):
    """Print any overlapping rendered text bboxes (cheap layout self-check)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.texts
             + [t for ax in fig.axes for t in ax.texts + [ax.title]]]
    boxes = []
    for t in texts:
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=r)
        boxes.append((s, bb))
    overlaps = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                overlaps += 1
                if overlaps <= 8:
                    print(f"    overlap{'[' + tag + ']' if tag else ''}: "
                          f"{boxes[i][0]!r} <-> {boxes[j][0]!r}")
    if overlaps:
        print(f"    ({overlaps} overlapping label pair(s))")
    else:
        print("    label layout clean (no overlaps)")


def _tick_label(cfg, k, pending):
    tag = " pending" if k in pending else ""
    return f"{k} ({cfg['ev'][k]}%{tag})"


def _finish(fig, ax, title, out_png):
    """Title sits directly above the axes (6 pt pad) — no in-figure caption
    text (thesis captions live outside the plots)."""
    ax.set_title(title, fontsize=12, pad=6)
    fig.tight_layout()
    _check_label_overlaps(fig, tag=out_png.stem)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def plot_auc(cfg, models, refs, pairs_vs_ref, out_png, ref_k,
             alpha=0.05, min_delta=0.0):
    """Pooled AUC ± outer-fold std bars, brackets to the reference ONLY for
    k's significantly different from it (pairs_vs_ref = vs_best family).
    cfg['pending'] k's (results not in yet) get an empty ghost slot."""
    ks = sorted(models)
    pending = sorted(cfg.get("pending", []))
    ref_i = ks.index(ref_k)
    pos = np.arange(len(ks))
    pooled = np.array([models[k]["auc"] for k in ks])
    ostd = np.array([np.std(models[k]["outer"], ddof=1) for k in ks])

    # ---- which k's get a bracket -------------------------------------
    sig = [pr for pr in pairs_vs_ref
           if pr["p_adj"] < alpha and abs(pr["delta"]) >= min_delta]
    dist = lambda pr: abs(ks.index(pr["k1"]) - ref_i)
    left = sorted((pr for pr in sig if ks.index(pr["k1"]) < ref_i), key=dist)
    right = sorted((pr for pr in sig if ks.index(pr["k1"]) > ref_i), key=dist)
    n_levels = max(len(left), len(right), 1)

    # ---- vertical layout: brackets sit above bars AND baselines -------
    h, step, gap = 0.012, 0.045, 0.06
    ref_top = max(refs.values(), default=0.0)
    base = max((pooled + ostd).max() + 0.05, ref_top + 0.04) + h
    y_top = max(1.0, base + (n_levels - 1) * step + 0.5 * step + 0.045)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(pos, pooled, yerr=ostd, color=cfg["color"], alpha=0.85,
           capsize=3, edgecolor="white", lw=0.5, zorder=3)
    for i in range(len(ks)):
        ax.text(i, pooled[i] + ostd[i] + 0.015, f"{pooled[i]:.4f}",
                ha="center", va="bottom", fontsize=8, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))

    # pending ghost slots (results still cooking)
    for j, pk in enumerate(pending):
        xi = len(ks) + j
        ax.bar(xi, 0.35, facecolor="none", edgecolor="0.55",
               ls="--", lw=1.0, zorder=3)
        ax.text(xi, 0.175, "pending", rotation=90, ha="center", va="center",
                fontsize=7.5, color="0.5")

    draw_reference_lines(ax, refs, min_sep=0.06)

    for side, sign in ((left, -1), (right, +1)):
        for lvl, pr in enumerate(side):
            # right-side brackets sit half a step higher so same-level
            # left/right labels can never collide near the reference
            y = base + lvl * step + (0.5 * step if sign > 0 else 0.0)
            draw_bracket(ax, float(ks.index(pr["k1"])),
                         ref_i + sign * gap,
                         y, _sig_label(pr), h, fontsize=7)

    ax.set_ylim(0, y_top)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    tick_ks = ks + pending
    ax.set_xticks(np.arange(len(tick_ks)))
    ax.set_xticklabels([_tick_label(cfg, k, set(pending)) for k in tick_ks],
                       fontsize=7.5, rotation=35, ha="right",
                       rotation_mode="anchor")
    for lab, k in zip(ax.get_xticklabels(), tick_ks):
        if k in pending:
            lab.set_color("0.5")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Held-out AUC")
    ax.grid(True, axis="y", alpha=0.3)

    _finish(fig, ax, f"PCA dimension sweep: {cfg['title']}", out_png)
    print(f"  ({len(sig)} bracket(s) vs k={ref_k})")


def plot_line(cfg, models, refs, out_png, ref_k, rot=35):
    """Clean trend companion: log-x line + markers, NO brackets/p-values.
    Tick labels are single-line 'k (EV%)', rotated by `rot` degrees.
    The best-AUC reference k gets an open marker + bold tick.
    cfg['pending'] k's get a grey dotted vertical marker + ghost tick."""
    ks = sorted(models)
    pending = sorted(cfg.get("pending", []))
    xs = np.array(ks, float)
    pooled = np.array([models[k]["auc"] for k in ks])
    ostd = np.array([np.std(models[k]["outer"], ddof=1) for k in ks])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, pooled, yerr=ostd, fmt="o-", color=cfg["color"],
                lw=1.6, ms=6, capsize=3, zorder=4, label=cfg["title"])
    ax.plot([ref_k], [models[ref_k]["auc"]], marker="o", ms=11,
            mfc="none", mec="0.1", mew=1.4, ls="none", zorder=5)
    draw_reference_lines(ax, refs)

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    tick_ks = ks + pending
    ax.set_xticks(tick_ks)
    ax.set_xticklabels([_tick_label(cfg, k, set(pending)) for k in tick_ks],
                       fontsize=8, rotation=rot, ha="right",
                       rotation_mode="anchor")
    for lab, k in zip(ax.get_xticklabels(), tick_ks):
        if k == ref_k:
            lab.set_fontweight("bold")
        if k in pending:
            lab.set_color("0.5")
    for pk in pending:
        ax.axvline(pk, color="0.6", ls=":", lw=1.0, zorder=2)
    ax.set_xlim(xs.min() * 0.7,
                (pending[-1] if pending else ks[-1]) * 1.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Held-out AUC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    _finish(fig, ax, f"PCA dimension sweep: {cfg['title']}", out_png)


def plot_runtime(cfg, models, t_rows, out_png, ref_k):
    """Core-hours per k (log-y); fold-seconds t-test annotation and hatched
    bar reference the same k the AUC figure uses (best-AUC k)."""
    ks = sorted(models)
    pending = sorted(cfg.get("pending", []))
    core_h = [cfg["wall_h"][k] * cfg["cores"][k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(k) for k in ks], core_h, color=cfg["color"],
                  alpha=0.85)
    for k, bar in zip(ks, bars):
        if k == ref_k:
            bar.set_hatch("//")
            bar.set_edgecolor("0.2")
    ax.set_yscale("log")
    ax.set_ylabel("Core-hours (nested CV, full data)")
    ax.set_xlabel("Number of PCs")
    for i, k in enumerate(ks):
        tr = next((t for t in t_rows if t["k"] == k), None)
        txt = f"{core_h[i]:.1f}h"
        if tr is not None and k != ref_k:
            txt += f"\np={tr['p']:.2g}"
        ax.text(i, core_h[i], txt, ha="center", va="bottom", fontsize=7)
    tick_ks = ks + pending
    ax.set_xticks(range(len(tick_ks)))
    ax.set_xticklabels([_tick_label(cfg, k, set(pending)) for k in tick_ks],
                       fontsize=7.5, rotation=35, ha="right",
                       rotation_mode="anchor")
    for lab, k in zip(ax.get_xticklabels(), tick_ks):
        if k == ref_k:
            lab.set_fontweight("bold")
        if k in pending:
            lab.set_color("0.5")
    for j, pk in enumerate(pending):
        ax.bar(len(ks) + j, max(core_h) * 0.4, facecolor="none",
               edgecolor="0.55", ls="--", lw=1.0)
        ax.text(len(ks) + j, max(core_h) * 0.2, "pending", rotation=90,
                ha="center", va="center", fontsize=7.5, color="0.5")
    ax.grid(True, axis="y", alpha=0.3)
    _finish(fig, ax, f"Compute cost by PCA dimensionality: {cfg['title']}",
            out_png)


# ── main ────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", choices=["both", "with", "none"],
                    default="both",
                    help="baseline reference lines in AUC figures: 'with' "
                         "(only), 'none' (only), or 'both' (default, "
                         "emits _nobaselines variants)")
    refs_mode = ap.parse_args().refs

    df = pd.read_csv(SPLIT_DATA_PATH)
    fold_vec = df["fold"].to_numpy(dtype=int)
    y = df["label"].to_numpy(dtype=int)
    n = len(df)
    masks = {tk: load_padded_mask(tk) for tk in PADMASK_FILES}
    print(f"df rows={n}, folds={sorted(set(fold_vec))}, "
          f"padded: " + ", ".join(f"{tk}={m.sum()}" for tk, m in masks.items()))

    refs = {}
    for ref_name, ref_path in REF_JSONS.items():
        r = load_model(ref_path, fold_vec, y, ref_path.name)
        r["auc"] = roc_auc_score(y, r["probs"])
        refs[ref_name] = r
    ref_aucs = {n_: r["auc"] for n_, r in refs.items()}

    all_results, all_pairs = [], []

    for name, cfg in CAMPAIGNS.items():
        print(f"\n=== {name}: {cfg['title']} ===")
        mask = masks[cfg["mask"]]
        found = {}
        for path in sorted(cfg["models"].glob("cv_results_xgb_*.json")):
            try:
                m = load_model(path, fold_vec, y, path.name)
            except Exception as e:  # half-written JSON from a live job
                print(f"  skip unreadable {path.name}: {e}")
                continue
            found[m["k"]] = m
        if cfg["anchor_json"] is not None:
            found[cfg["anchor"]] = load_model(
                cfg["anchor_json"], fold_vec, y, cfg["anchor_json"].name)
            assert cfg["anchor"] not in cfg["ladder"]
        # round-2 jobs drop extra k's into the folder as they finish —
        # analyze the declared ladder only; extras warn and wait
        known = set(cfg["ladder"])
        if cfg["anchor_json"] is not None:
            known.add(cfg["anchor"])
        extra = set(found) - known
        if extra:
            print(f"  note: {len(extra)} k(s) present but not in ladder "
                  f"(round-2 pending extension): {sorted(extra)}")
        missing = known - set(found)
        assert not missing, f"{name}: missing k {sorted(missing)}"
        found = {k: v for k, v in found.items() if k in known}

        # pooled AUC only; per-k fold SD is computed from outer below
        for k, m in found.items():
            m["auc"] = roc_auc_score(y, m["probs"])

        # DeLong families
        ladder = cfg["ladder"]
        anchor = cfg["anchor"]
        fam_all, fam_anchor, fam_best = [], [], []
        best_k = max(ladder, key=lambda k: found[k]["auc"])
        for i, a in enumerate(ladder):
            for b in ladder[i + 1:]:
                ya, yb, d, z, p = delong_p(y, found[a]["probs"], found[b]["probs"])
                fam_all.append(dict(family="all_pairs", k1=a, k2=b,
                                    auc1=ya, auc2=yb, delta=d, z=z, p=p))
        for k in ladder:
            if k != anchor:
                ya, yb, d, z, p = delong_p(y, found[k]["probs"],
                                            found[anchor]["probs"])
                fam_anchor.append(dict(family="vs_anchor", k1=k, k2=anchor,
                                       auc1=ya, auc2=yb, delta=d, z=z, p=p))
            if k != best_k:  # every non-best ladder k, incl. the 90% anchor
                ya, yb, d, z, p = delong_p(y, found[k]["probs"],
                                            found[best_k]["probs"])
                fam_best.append(dict(family="vs_best", k1=k, k2=best_k,
                                 auc1=ya, auc2=yb, delta=d, z=z, p=p))

        fam_all = holm_family(fam_all)
        fam_anchor = holm_family(fam_anchor)
        fam_best = holm_family(fam_best)
        # stamp campaign + family size + EV% so the CSV is self-describing
        for fam in (fam_all, fam_anchor, fam_best):
            for pr in fam:
                pr["campaign"] = name
                pr["family_m"] = len(fam)
                pr["ev1"] = cfg["ev"][pr["k1"]]
                pr["ev2"] = cfg["ev"][pr["k2"]]
        all_pairs.extend(fam_all + fam_anchor + fam_best)

        # runtime NB t-tests vs the same reference the AUC figure uses
        # (best-AUC k, per user: t-test anchor = AUC anchor)
        t_rows = []
        for k in ladder:
            if k == best_k:
                continue
            ma, mb, t_stat, p = fold_t_nb(found[k]["fold_seconds"],
                                          found[best_k]["fold_seconds"])
            t_rows.append(dict(k=k, mean_k=ma, mean_ref=mb, t=t_stat, p=p))

        # verdict: smallest ladder k Holm-ns vs best
        verdict_k = best_k
        ns_vs_best = [pr["k1"] for pr in fam_best if pr["p_adj"] > 0.05]
        if ns_vs_best:
            verdict_k = min(ns_vs_best)
        print(f"  best AUC k={best_k} ({found[best_k]['auc']:.4f}); "
              f"smallest k ns-vs-best (Holm) = {verdict_k}"
              + ("  [= 90% EV anchor → EV anchor validated]"
                 if verdict_k == anchor else ""))
        for k in ladder:
            m = found[k]
            all_results.append(dict(
                campaign=name, k=k, ev_pct=cfg["ev"][k],
                is_anchor=(k == anchor), auc_pooled=m["auc"],
                auc_outer_mean=float(np.mean(m["outer"])),
                auc_outer_std=float(np.std(m["outer"], ddof=1)),
                fold_seconds_mean=float(np.mean(m["fold_seconds"])),
                wall_h=cfg["wall_h"][k], cores=cfg["cores"][k],
                core_h=round(cfg["wall_h"][k] * cfg["cores"][k], 2),
                maxrss="NA",
                optimal=(k == verdict_k), best=(k == best_k)))
        if anchor not in ladder:
            m = found[anchor]
            all_results.append(dict(
                campaign=name, k=anchor, ev_pct=cfg["ev"][anchor],
                is_anchor=True, auc_pooled=m["auc"],
                auc_outer_mean=float(np.mean(m["outer"])),
                auc_outer_std=float(np.std(m["outer"], ddof=1)),
                fold_seconds_mean=float(np.mean(m["fold_seconds"])),
                wall_h=cfg["wall_h"][anchor], cores=cfg["cores"][anchor],
                core_h=round(cfg["wall_h"][anchor] * cfg["cores"][anchor], 2),
                maxrss="NA", optimal=False, best=False))

        # figures: bars (with brackets vs best-AUC k) + clean line,
        # with and without baseline reference lines (--refs)
        cfg["out"].mkdir(parents=True, exist_ok=True)
        variants = []
        if refs_mode in ("both", "with"):
            variants.append(("", ref_aucs))
        if refs_mode in ("both", "none"):
            variants.append(("_nobaselines", {}))
        for suffix, rmap in variants:
            plot_auc(cfg, found, rmap, fam_best,
                     cfg["out"] / f"pca_sweep_auc_full{suffix}.png",
                     ref_k=best_k)
            plot_line(cfg, found, rmap,
                      cfg["out"] / f"pca_sweep_auc_line{suffix}.png",
                      ref_k=best_k)
        plot_runtime(cfg, {k: found[k] for k in ladder}, t_rows,
                     cfg["out"] / "pca_sweep_runtime.png", ref_k=best_k)
        # figures from retired code paths — remove on every run
        for stale_name in ("pca_sweep_auc_subset.png",
                           "pca_sweep_auc_roi.png",
                           "pca_sweep_auc_roi_nobaselines.png"):
            stale = cfg["out"] / stale_name
            if stale.exists():
                stale.unlink()

        with open(cfg["out"] / "pca_sweep_runtime_t.csv", "w") as fh:
            fh.write("k,mean_fold_seconds_k,mean_fold_seconds_ref,t,p_nb\n")
            for tr in t_rows:
                fh.write(f"{tr['k']},{tr['mean_k']:.1f},{tr['mean_ref']:.1f},"
                         f"{tr['t']:.3f},{tr['p']:.3g}\n")
        for ref_name, r in refs.items():
            sub = roc_auc_score(y[mask], r["probs"][mask])
            print(f"  {ref_name} baseline: full={r['auc']:.4f} "
                  f"subset[{cfg['mask']}]={sub:.4f}")

    # combined CSVs at repo level
    res_dir = PROJECT_ROOT / "runs" / "pca_sweep_delong"
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(res_dir / "pca_sweep_results.csv", index=False)
    DELONG_COLS = ["campaign", "family", "family_m", "k1", "ev1",
                   "k2", "ev2", "auc1", "auc2", "delta", "z", "p", "p_adj"]
    ddf = pd.DataFrame(all_pairs)[DELONG_COLS]
    ddf = ddf.sort_values(["campaign", "family", "k1", "k2"])
    ddf.to_csv(res_dir / "pca_sweep_delong.csv", index=False)
    print(f"\nCombined CSVs -> {res_dir}/pca_sweep_results.csv, "
          f"{res_dir}/pca_sweep_delong.csv")
    print("\nOPTIMUM-PC VERDICTS:")
    rdf = pd.DataFrame(all_results)
    for name in CAMPAIGNS:
        sub = rdf[rdf["campaign"] == name]
        opt = sub[sub["optimal"]].iloc[0]
        best = sub[sub["best"]].iloc[0]
        print(f"  {name}: best k={int(best['k'])} "
              f"(AUC {best['auc_pooled']:.4f}), optimum k={int(opt['k'])} "
              f"({opt['ev_pct']}% EV)")


if __name__ == "__main__":
    main()
