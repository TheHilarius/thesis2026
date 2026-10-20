#!/usr/bin/env python3
"""18_encodings_sweep_delong.py
Significance + figure for the 18-arm AA-encoding × embedding sweep
(runs/encodings_sweep). Answers: do encodings beat embeddings alone, and
does combining an encoding with an embedding help or hurt?

Statistical protocol (pooled out-of-fold DeLong, same as
pca_sweep_AUC_delong_comparison.py / 15_padmode_significance.py):
  * bar heights = pooled OOF AUC over all 46,617 rows (roc_auc_score on the
    concatenated outer-fold predictions) — the same vector DeLong tests.
  * error bars  = std of the 6 outer-fold AUCs.
  * DeLong two-sided z-test on the pooled OOF probability vectors; Holm
    correction within each comparison family; brackets drawn ONLY for
    significant pairs.

Two DeLong families (Holm within family):
  A  vs_encoding_only     each (encoding+embedding) vs its encoding alone
  B  vs_embedding_only    each (encoding+embedding) vs its embedding alone

Emits (into runs/encodings_sweep/):
  plots/encodings_sweep_auc_delong.png (+.pdf)   two panels
  results/encodings_sweep_deltas.tsv             both delta directions + memory
  results/encodings_sweep_delong_stats.tsv       every tested pair

Caption caveat (lives in the thesis, not the figure): the ~0.0024-AUC
detection threshold is smaller than the 0.003-0.009 outer-fold SD, so a
significant delta may still be immaterial.

Stats code is copied verbatim from pca_sweep_AUC_delong_comparison.py (which
itself inlined it from 15_padmode_significance.py — its import chain is
broken upstream).
"""

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH

ALPHA = 0.05

# ── stats (inlined from 15_padmode_significance — proven code) ──────────
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


def delong_full(y, pa, pb):
    """Two-sided DeLong test; returns auc_a, auc_b, delta, SE(delta), z, p."""
    auc_a, auc_b, var_d = delong_test(y, pa, pb)
    d = auc_a - auc_b
    if var_d <= 0:
        return auc_a, auc_b, d, 0.0, 0.0, 1.0
    se = float(np.sqrt(var_d))
    z = d / se
    p = float(2 * norm.sf(abs(z)))
    return auc_a, auc_b, d, se, z, p


def holm_family(pairs, key="p"):
    adj = holm_adjust([pr[key] for pr in pairs])
    for pr, a in zip(pairs, adj):
        pr["p_adj"] = float(a)
    return pairs


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_p(p):
    """Bracket p-text; guards underflow/zero from norm.sf at large |z|."""
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


def fmt_p_pair(p, p_adj):
    """'p=0.51, p_adj=1.0' bracket text; stars come from p_adj."""
    if p_adj is None:
        return fmt_p(p)
    a = "p_adj<0.001" if p_adj < 0.001 else f"p_adj={p_adj:.2g}"
    return f"{fmt_p(p)}, {a}"



# ── data ────────────────────────────────────────────────────────────────
def reconstruct(res, fold_vec, y, tag):
    """Full-length OOF probability vector; validates against the fold CSV."""
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
    probs = reconstruct(res, fold_vec, y, tag)
    outer = [float(m["auc_roc"]) for m in res["outer_val_metrics"]]
    assert len(outer) == 6, path
    return {
        "key": res["features_key"],
        "k": int(res["config"]["n_features"]),
        "probs": probs,
        "auc": float(roc_auc_score(y, probs)),   # pooled OOF (matches DeLong)
        "outer": outer,
        "outer_std": float(np.std(outer, ddof=1)),
        "runtime_s": float(res.get("nested_cv_seconds", np.nan)),
        "file": os.path.basename(path),
    }


def read_mem(features_key):
    """peak_rss_kb, cgroup_peak_bytes, wall_s, cpu_pct from the sidecar."""
    path = PROJECT_ROOT / "runs" / "encodings_sweep" / "results" / \
        f"mem_{features_key}.tsv"
    if not path.exists():
        return dict(peak_rss_kb="", cgroup_peak_bytes="", wall_s="", cpu_pct="")
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        return dict(peak_rss_kb="", cgroup_peak_bytes="", wall_s="", cpu_pct="")
    r = rows[0]
    return dict(peak_rss_kb=r.get("peak_rss_kb", ""),
                cgroup_peak_bytes=r.get("cgroup_peak_bytes", ""),
                wall_s=r.get("wall_s", ""),
                cpu_pct=r.get("cpu_pct", ""))


# ── arms ────────────────────────────────────────────────────────────────
ENCODINGS = ["sparse", "blosum50", "blosum62", "blosum80", "kidera", "atchley"]
ENC_LABEL = {"sparse": "One-hot", "blosum50": "BLOSUM50", "blosum62": "BLOSUM62",
             "blosum80": "BLOSUM80", "kidera": "Kidera", "atchley": "Atchley"}
# padmode_choice palette (plot_padmode_data_compare.GROUP_COLORS)
C_SOLO = "#BBBBBB"          # AA encoding (Tol grey)
C_C    = "#EE6677"          # ESM-C  (Tol red)
C_IF   = "#4477AA"          # ESM-IF (Tol blue)


def _tint(hex_color, mix=0.5):
    """Blend hex_color toward white by `mix` (reproducible combo tints)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    f = lambda c: int(round(c + (255 - c) * mix))
    return f"#{f(r):02X}{f(g):02X}{f(b):02X}"


# 50%-white tints for the "encoding + embedding" combo bars
C_C_COMBO  = _tint(C_C)     # light red
C_IF_COMBO = _tint(C_IF)    # light blue


# (emb component name used in combined keys, emb-ONLY baseline features_key,
#  label, full colour, combo tint colour). Combined keys use the bare
#  component; the emb-only baseline's features_key carries the "emb_" prefix.
EMB_ONLY = [
    ("esmif_eos_bos_repeat", "emb_esmif_eos_bos_repeat", "ESM-IF", C_IF, C_IF_COMBO),
    ("esmc300m_win_padtoken", "emb_esmc300m_win_padtoken", "ESM-C", C_C, C_C_COMBO),
]


def arm_key(enc, emb):
    return f"{enc}_{emb}"



# ── plotting helpers ────────────────────────────────────────────────────
def _check_label_overlaps(fig, tag=""):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.texts + [t for ax in fig.axes for t in ax.texts]]
    boxes = []
    for t in texts:
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        boxes.append((s, t.get_window_extent(renderer=r)))
    overlaps = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                overlaps += 1
                if overlaps <= 6:
                    print(f"    overlap[{tag}]: {boxes[i][0]!r} <-> {boxes[j][0]!r}")
    print(f"    ({overlaps} overlapping label pair(s))" if overlaps
          else "    label layout clean")


# ── figure (padmode_choice style: plot_padmode_data_compare.py) ─────────
def _padmode_bars(ax, xs, values, errors, colors, width=0.62):
    ax.bar(xs, values, yerr=errors, color=colors, width=width,
           edgecolor="black", linewidth=0.5, capsize=3)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)


def _padmode_brackets(ax, pairs, values, fontsize=7.5, stars_only=False):
    """pairs: [(x_a, x_b, p, p_adj), ...]. Widest span on top; text shows
    raw p + Holm p_adj, stars from p_adj (or stars-only for dense strips,
    per plot_padmode_data_compare 'baselines' layout — full p's live in the
    stats TSV). Fits under y=0.97 on the 0-1 axis."""
    if not pairs:
        return
    pairs = sorted(pairs, key=lambda t: -abs(t[1] - t[0]))
    y0 = max(values) * 1.08
    max_lvl = len(pairs)
    headroom = 0.97 - y0 - max(values) * 0.03
    step = (min(max(values) * 0.085, headroom / (max_lvl - 1 + 0.35))
            if max_lvl > 1 else max(values) * 0.085)
    if step < max(values) * 0.05:
        fontsize = 6.5
    for lvl, (xa, xb, p, p_adj) in enumerate(pairs):
        y_br = y0 + lvl * step
        ax.plot([xa, xa, xb, xb], [y_br * 0.985, y_br, y_br, y_br * 0.985],
                color="black", lw=0.9)
        txt = (stars(p_adj) if stars_only
               else f"{fmt_p_pair(p, p_adj)} ({stars(p_adj)})")
        ax.text((xa + xb) / 2, y_br + step * 0.12, txt,
                ha="center", va="bottom", fontsize=fontsize)



def _finish_padmode(fig, ax, title, out_png, legend_entries, footnote=None):
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if footnote:
        fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=7)
    if len(legend_entries) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black")
                   for _lb, c in legend_entries]
        ax.legend(handles, [lb for lb, _c in legend_entries],
                  fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False)
    fig.tight_layout()
    _check_label_overlaps(fig, tag=out_png.stem)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def plot_triple(models, famA, famB, famC, emb_comp, emb_label, emb_full,
                emb_combo, out_png, alpha=ALPHA):
    """6 encoding groups x 3 bars: encoding / encoding+emb / emb-alone.
    padmode style; three sig-only brackets per group: famA enc vs enc+emb,
    famB enc+emb vs emb, famC enc vs emb (stars-only, dense strip)."""
    step_g = 3.0                       # bar centres within a group: +0,+1,+2
    xs, vals, errs, colors = [], [], [], []
    for gi, enc in enumerate(ENCODINGS):
        trio = [(enc, C_SOLO),
                (arm_key(enc, emb_comp), emb_combo),
                (f"emb_{emb_comp}", emb_full)]
        for bi, (k, c) in enumerate(trio):
            xs.append(gi * step_g + bi)
            vals.append(models[k]["auc"]); errs.append(models[k]["outer_std"])
            colors.append(c)

    # brackets from group indices (emb-alone key repeats per group, so a
    # key->x map would collide)
    famA_by = {(pr["enc"], pr["emb"]): pr for pr in famA}
    famB_by = {(pr["enc"], pr["emb"]): pr for pr in famB}
    famC_by = {(pr["enc"], pr["emb"]): pr for pr in famC}
    pairs = []
    for gi, enc in enumerate(ENCODINGS):
        pa = famA_by.get((enc, emb_comp))
        if pa and pa["p_adj"] < alpha:
            pairs.append((gi * step_g + 0, gi * step_g + 1, pa["p"], pa["p_adj"]))
        pb = famB_by.get((enc, emb_comp))
        if pb and pb["p_adj"] < alpha:
            pairs.append((gi * step_g + 1, gi * step_g + 2, pb["p"], pb["p_adj"]))
        pc = famC_by.get((enc, emb_comp))
        if pc and pc["p_adj"] < alpha:
            pairs.append((gi * step_g + 0, gi * step_g + 2, pc["p"], pc["p_adj"]))

    fig, ax = plt.subplots(figsize=(13, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    for gi in range(len(ENCODINGS) - 1):        # dividers between groups
        ax.axvline(gi * step_g + 2.5, color="grey", linestyle="--",
                   linewidth=0.8)
    _padmode_brackets(ax, pairs, vals, stars_only=True)
    ax.set_xticks([gi * step_g + 1 for gi in range(len(ENCODINGS))])
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODINGS],
                       rotation=30, ha="right", fontsize=9)
    _finish_padmode(fig, ax,
                    f"AA encoding vs encoding + {emb_label} vs {emb_label} alone",
                    out_png,
                    [("AA encoding", C_SOLO),
                     (f"encoding + {emb_label}", emb_combo),
                     (f"{emb_label} alone", emb_full)])


def plot_encodings_only(models, famEnc, out_png, alpha=ALPHA):
    """6 encoding bars; BLOSUM50-centred significant contrasts only.

    Non-significant comparisons are not bracketed; the BLOSUM50 vs BLOSUM80
    non-result is reported in the footnote instead.
    """
    ref = "blosum50"
    xs = list(range(len(ENCODINGS)))
    key_to_x = {enc: i for i, enc in enumerate(ENCODINGS)}
    vals = [models[enc]["auc"] for enc in ENCODINGS]
    errs = [models[enc]["outer_std"] for enc in ENCODINGS]
    colors = [C_SOLO] * len(ENCODINGS)

    pairs = []
    for pr in famEnc:
        if pr["p_adj"] >= alpha:
            continue
        if ref not in {pr["name_a"], pr["name_b"]}:
            continue
        xa, xb = sorted((key_to_x[pr["name_a"]], key_to_x[pr["name_b"]]))
        pairs.append((xa, xb, pr["p"], pr["p_adj"]))

    ns = next(pr for pr in famEnc
              if {pr["name_a"], pr["name_b"]} == {"blosum50", "blosum80"})
    borderline = next(pr for pr in famEnc
                      if {pr["name_a"], pr["name_b"]} == {"blosum62", "blosum80"})
    footnote = (
        f"BLOSUM50 vs BLOSUM80: delta={ns['delta']:+.4f}, "
        f"adjusted p={ns['p_adj']:.2g} (ns). "
        f"BLOSUM80 vs BLOSUM62: adjusted p={borderline['p_adj']:.2g} (ns). "
        "Full pairwise tests: encodings_sweep_delong_stats.tsv."
    )

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    _padmode_brackets(ax, pairs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODINGS],
                       rotation=30, ha="right", fontsize=9)
    _finish_padmode(fig, ax,
                    "Held-out AUC: AA encodings alone (BLOSUM50 contrasts)",
                    out_png, [("AA encoding", C_SOLO)], footnote=footnote)


def plot_flat(models, famA, out_png, alpha=ALPHA):
    """All 20 bars, padmode sectioning: encodings | ESM-C | ESM-IF.
    Family-A brackets only, significant pairs only."""
    sec1 = [(enc, C_SOLO, ENC_LABEL[enc]) for enc in ENCODINGS]
    c_if, c_if_combo, c_c, c_c_combo = C_IF, C_IF_COMBO, C_C, C_C_COMBO
    sec2 = [(arm_key(e, "esmc300m_win_padtoken"), c_c_combo,
             f"{ENC_LABEL[e]} +ESM-C") for e in ENCODINGS] + \
           [("emb_esmc300m_win_padtoken", c_c, "ESM-C")]
    sec3 = [(arm_key(e, "esmif_eos_bos_repeat"), c_if_combo,
             f"{ENC_LABEL[e]} +ESM-IF") for e in ENCODINGS] + \
           [("emb_esmif_eos_bos_repeat", c_if, "ESM-IF")]
    bars = sec1 + sec2 + sec3                       # 6 + 7 + 7 = 20

    xs = list(range(len(bars)))
    key_to_x = {k: x for x, (k, _c, _l) in enumerate(bars)}
    vals = [models[k]["auc"] for k, _c, _l in bars]
    errs = [models[k]["outer_std"] for k, _c, _l in bars]
    colors = [c for _k, c, _l in bars]
    labels = [l for _k, _c, l in bars]

    pairs = [(key_to_x[pr["name_b"]], key_to_x[pr["name_a"]],
              pr["p"], pr["p_adj"])
             for pr in famA if pr["p_adj"] < alpha]

    fig, ax = plt.subplots(figsize=(16, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    ax.axvline(5.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(12.5, color="grey", linestyle="--", linewidth=0.8)
    _padmode_brackets(ax, pairs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    _finish_padmode(fig, ax,
                    "Held-out AUC: AA encodings, encodings + embeddings, embeddings alone",
                    out_png,
                    [("AA encoding", C_SOLO),
                     ("encoding + ESM-C", c_c_combo), ("ESM-C", c_c),
                     ("encoding + ESM-IF", c_if_combo), ("ESM-IF", c_if)])



# ── main ────────────────────────────────────────────────────────────────
def main():
    split = pd.read_csv(SPLIT_DATA_PATH)
    fold_vec = split["fold"].to_numpy(dtype=int)
    y = split["label"].to_numpy(dtype=int)
    print(f"Loaded split: {len(y)} rows, {int(y.sum())} positive, "
          f"{len(set(fold_vec))} folds")

    glob_pat = str(PROJECT_ROOT / "runs" / "encodings_sweep" / "models" /
                   "flat" / "cv_results_*.json")
    paths = sorted(glob.glob(glob_pat))
    print(f"Found {len(paths)} result JSONs")
    models = {}
    for p in paths:
        m = load_model(p, fold_vec, y, os.path.basename(p))
        models[m["key"]] = m
        print(f"  {m['key']:<36} k={m['k']:<5} pooled AUC={m['auc']:.4f} "
              f"±{m['outer_std']:.4f}")

    # sanity: every arm present
    needed = [e for e in ENCODINGS] + \
             [arm_key(e, emb) for e in ENCODINGS
              for emb in ("esmif_eos_bos_repeat", "esmc300m_win_padtoken")] + \
             [base for _c, base, _n, _cc, _ct in EMB_ONLY]
    missing = [k for k in needed if k not in models]
    if missing:
        raise SystemExit(f"Missing arms: {missing}")

    # ── DeLong families ──
    famA, famB = [], []
    for enc in ENCODINGS:
        for emb, emb_base, _nm, _c, _ct in EMB_ONLY:
            k_comb = arm_key(enc, emb)
            k_emb = emb_base            # emb-only baseline key
            auc_a, auc_b, d, se, z, p = delong_full(
                y, models[k_comb]["probs"], models[enc]["probs"])
            famA.append(dict(enc=enc, emb=emb, name_a=k_comb, name_b=enc,
                             auc_a=auc_a, auc_b=auc_b, delta=d, se=se,
                             z=z, p=p))
            auc_a2, auc_b2, d2, se2, z2, p2 = delong_full(
                y, models[k_comb]["probs"], models[k_emb]["probs"])
            famB.append(dict(enc=enc, emb=emb, name_a=k_comb, name_b=k_emb,
                             auc_a=auc_a2, auc_b=auc_b2, delta=d2, se=se2,
                             z=z2, p=p2))
    holm_family(famA)
    holm_family(famB)
    print(f"\nFamily A (vs encoding-only): {len(famA)} pairs, "
          f"{sum(1 for p in famA if p['p_adj'] < ALPHA)} significant")
    print(f"Family B (vs embedding-only): {len(famB)} pairs, "
          f"{sum(1 for p in famB if p['p_adj'] < ALPHA)} significant")

    # ── DeLong family C: encoding vs embedding-alone (the outer triple pair)
    famC = []
    for enc in ENCODINGS:
        for emb, emb_base, _nm, _c, _ct in EMB_ONLY:
            auc_a, auc_b, d, se, z, p = delong_full(
                y, models[enc]["probs"], models[emb_base]["probs"])
            famC.append(dict(enc=enc, emb=emb, name_a=enc, name_b=emb_base,
                             auc_a=auc_a, auc_b=auc_b, delta=d, se=se,
                             z=z, p=p))
    holm_family(famC)
    print(f"Family C (encoding vs embedding-alone): {len(famC)} pairs, "
          f"{sum(1 for p in famC if p['p_adj'] < ALPHA)} significant")

    # ── DeLong family: all pairwise AA encodings (encodings-only plot)
    famEnc = []
    for i, a in enumerate(ENCODINGS):
        for b in ENCODINGS[i + 1:]:
            auc_a, auc_b, d, se, z, p = delong_full(
                y, models[a]["probs"], models[b]["probs"])
            famEnc.append(dict(enc_a=a, enc_b=b, name_a=a, name_b=b,
                               auc_a=auc_a, auc_b=auc_b, delta=d, se=se,
                               z=z, p=p))
    holm_family(famEnc)
    print(f"Family Enc (encodings pairwise): {len(famEnc)} pairs, "
          f"{sum(1 for p in famEnc if p['p_adj'] < ALPHA)} significant")

    # ── figures (padmode_choice style) ──
    out_dir = PROJECT_ROOT / "runs" / "encodings_sweep" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for comp, base, label, full, combo in EMB_ONLY:
        plot_triple(models, famA, famB, famC, comp, label, full, combo,
                    out_dir / f"encodings_sweep_triple_{comp.split('_')[0]}.png")
    plot_flat(models, famA, out_dir / "encodings_sweep_flat_padmode.png")
    plot_encodings_only(models, famEnc,
                        out_dir / "encodings_sweep_encodings_only.png")

    # ── TSV: deltas + memory ──
    res_dir = PROJECT_ROOT / "runs" / "encodings_sweep" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    famB_by = {(pr["enc"], pr["emb"]): pr for pr in famB}
    delta_rows = []
    for enc in ENCODINGS:
        for emb, emb_base, nm, _c, _ct in EMB_ONLY:
            k_comb = arm_key(enc, emb)
            pa = next(p for p in famA if p["enc"] == enc and p["emb"] == emb)
            pb = famB_by[(enc, emb)]
            mem_comb = read_mem(k_comb)
            mem_enc = read_mem(enc)
            delta_rows.append(dict(
                encoding=enc, embedding=nm, combined_key=k_comb,
                enc_only_auc=f"{models[enc]['auc']:.6f}",
                combined_auc=f"{models[k_comb]['auc']:.6f}",
                delta=f"{pa['delta']:.6f}", se=f"{pa['se']:.6f}",
                z=f"{pa['z']:.4f}", p=f"{pa['p']:.3e}",
                p_adj=f"{pa['p_adj']:.3e}",
                emb_only_auc=f"{models[pb['name_b']]['auc']:.6f}",
                delta_vs_emb=f"{pb['delta']:.6f}",
                p_vs_emb=f"{pb['p']:.3e}", p_adj_vs_emb=f"{pb['p_adj']:.3e}",
                enc_only_peak_rss_kb=mem_enc["peak_rss_kb"],
                enc_only_cgroup_peak_bytes=mem_enc["cgroup_peak_bytes"],
                enc_only_wall_s=mem_enc["wall_s"],
                combined_peak_rss_kb=mem_comb["peak_rss_kb"],
                combined_cgroup_peak_bytes=mem_comb["cgroup_peak_bytes"],
                combined_wall_s=mem_comb["wall_s"],
                combined_cpu_pct=mem_comb["cpu_pct"],
            ))
    with open(res_dir / "encodings_sweep_deltas.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(delta_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(delta_rows)
    print(f"  Wrote {res_dir/'encodings_sweep_deltas.tsv'}")

    # ── TSV: every tested pair ──
    stat_rows = []
    for fam, pairs in (("vs_encoding_only", famA),
                       ("vs_embedding_only", famB),
                       ("vs_embedding_alone", famC)):
        for pr in pairs:
            stat_rows.append(dict(
                family=fam, encoding=pr["enc"], embedding=pr["emb"],
                name_a=pr["name_a"], name_b=pr["name_b"],
                auc_a=f"{pr['auc_a']:.6f}", auc_b=f"{pr['auc_b']:.6f}",
                delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
                z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
                p_adj=f"{pr['p_adj']:.3e}",
                significant=str(pr["p_adj"] < ALPHA)))
    for pr in famEnc:
        stat_rows.append(dict(
            family="encodings_pairwise", encoding=pr["enc_a"],
            embedding=pr["enc_b"], name_a=pr["name_a"], name_b=pr["name_b"],
            auc_a=f"{pr['auc_a']:.6f}", auc_b=f"{pr['auc_b']:.6f}",
            delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
            z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
            p_adj=f"{pr['p_adj']:.3e}",
            significant=str(pr["p_adj"] < ALPHA)))

    with open(res_dir / "encodings_sweep_delong_stats.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(stat_rows)
    print(f"  Wrote {res_dir/'encodings_sweep_delong_stats.tsv'}")

    print("\nCaption caveat: detection threshold ~0.0024 AUC vs outer-fold "
          "SD 0.003-0.009; significant != material.")


if __name__ == "__main__":
    main()
