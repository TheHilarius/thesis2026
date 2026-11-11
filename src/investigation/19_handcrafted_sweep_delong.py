#!/usr/bin/env python3
"""19_handcrafted_sweep_delong.py
Significance + figures for the explicit-feature sweep (runs/handcrafted_sweep).
Answers the handcrafted questions only — the encoding-vs-embedding contrasts
are already settled by 18_encodings_sweep_delong.py and are NOT recomputed:

  E1  vs_encoding_only   explicit+encoding vs encoding alone      (6)
  E2  vs_explicit_alone  explicit+encoding vs explicit alone      (6)
  E5  explicit_pairwise  explicit+encoding vs each other         (15)
  T1  three_vs_enc_emb   three-way vs encoding+embedding         (12)
  T2  three_vs_exp_enc   three-way vs explicit+encoding          (12)
  T3  three_vs_explicit  three-way vs explicit alone             (12)
  E6  explicit_emb_vs_emb       explicit+embedding vs embedding   (2, when landed)
  E7  explicit_emb_vs_explicit  explicit+embedding vs explicit    (2, when landed)
  E8  explicit_embedding_pairwise  4 embedding arms all-vs-all     (6, when landed)

Statistical protocol identical to 18_ (pooled out-of-fold DeLong, Holm
correction within each family, brackets only for significant pairs) —
stats code copied verbatim. Every arm's OOF labels are md5-checked against
the split CSV at load time.

Plot display terminology: "handcrafted features" (reverted from "explicit
features", user decision Oct 22); code keys keep the handcrafted_* names.

Emits (into runs/<campaign>/, default campaign=handcrafted_sweep):
  plots/handcrafted_effect_on_encodings.png   enc / hc+enc + hc anchor bar
  plots/handcrafted_encodings_ranking.png     hc+enc arms all-vs-all (best pick)
  plots/handcrafted_effect_on_esmif.png       ESM-IF / hc+ESM-IF / hc
  plots/handcrafted_effect_on_esmc.png        ESM-C / hc+ESM-C / hc
  plots/handcrafted_embeddings_ranking.png    4 embedding arms, all-vs-all
  results/handcrafted_sweep_delong_stats.tsv    every tested pair
  results/handcrafted_sweep_deltas.tsv          explicit contrasts + memory

Campaign flags for the vardims-defaults redo:
  --campaign is_handcrafted_helping_v2 --ref-campaign is_encodings_helping_v2
The ref campaign supplies the emb_* and encoding+embedding reference arms
(mirrors the old symlinked refs). Family gating is two-level:
  level A  handcrafted×encoding singles -> E1/E2/E5 + encodings figures
           (embedding-free pure-CSV arms; the helping-v2 campaign has
           these copied from handcrafted_sweep — reduction-agnostic)
  level B  three-way handcrafted×encoding×embedding arms -> T1-T3 +
           with_embedding delta rows (absent on helping-v2 by design:
           old triples are PCA-era and would mix reductions)
Missing levels skip with a warning instead of aborting; E6/E7/E8 and
their figures always run when their arms exist.

Caption caveat: the ~0.0024-AUC detection threshold is smaller than the
0.003-0.009 outer-fold SD, so a significant delta may still be immaterial.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC.parent / "pipeline" / "python"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH

ALPHA = 0.05
EXP = "handcrafted"          # feature-set key prefix (display: "explicit")

# ── stats (verbatim copy from 18_encodings_sweep_delong.py) ──────────────
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
    if p_adj is None:
        return fmt_p(p)
    a = "p_adj<0.001" if p_adj < 0.001 else f"p_adj={p_adj:.2g}"
    return f"{fmt_p(p)}, {a}"


# ── data ────────────────────────────────────────────────────────────────
def reconstruct(res, fold_vec, y, tag):
    """Full-length OOF probability vector; validates against the fold CSV."""
    n = len(fold_vec)
    probs = np.full(n, np.nan)
    yt_all = np.full(n, -1)
    for fk, entry in res["outer_val_predictions"].items():
        mask = fold_vec == int(fk)
        yt = np.asarray(entry["y_true"], dtype=int)
        if mask.sum() != len(entry["y_prob"]):
            raise ValueError(f"{tag}: fold {fk} size mismatch")
        if not (yt == y[mask]).all():
            raise ValueError(f"{tag}: fold {fk} labels mismatch df")
        probs[mask] = np.asarray(entry["y_prob"], dtype=float)
        yt_all[mask] = yt
    if np.isnan(probs).any():
        raise ValueError(f"{tag}: rows never predicted")
    return probs, hashlib.md5(yt_all.tobytes()).hexdigest()


def load_model(path, fold_vec, y, tag):
    with open(path) as fh:
        res = json.load(fh)
    probs, label_md5 = reconstruct(res, fold_vec, y, tag)
    outer = [float(m["auc_roc"]) for m in res["outer_val_metrics"]]
    assert len(outer) == 6, path
    return {
        "key": res["features_key"],
        "k": int(res["config"]["n_features"]),
        "probs": probs,
        "label_md5": label_md5,
        "auc": float(roc_auc_score(y, probs)),   # pooled OOF (matches DeLong)
        "outer": outer,
        "outer_std": float(np.std(outer, ddof=1)),
        "file": os.path.basename(path),
    }


def read_mem(features_key, campaign, ref_campaign):
    """peak_rss_kb, cgroup_peak_bytes, wall_s, cpu_pct; explicit arms live
    in the handcrafted campaign, reference arms in the encodings campaign."""
    for run_root in (campaign, ref_campaign):
        path = PROJECT_ROOT / "runs" / run_root / "results" / \
            f"mem_{features_key}.tsv"
        if path.exists():
            with open(path) as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
            if rows:
                r = rows[0]
                return dict(peak_rss_kb=r.get("peak_rss_kb", ""),
                            cgroup_peak_bytes=r.get("cgroup_peak_bytes", ""),
                            wall_s=r.get("wall_s", ""),
                            cpu_pct=r.get("cpu_pct", ""))
    return dict(peak_rss_kb="", cgroup_peak_bytes="", wall_s="", cpu_pct="")


# ── arms ────────────────────────────────────────────────────────────────
ENCODINGS = ["sparse", "blosum50", "blosum62", "blosum80", "kidera", "atchley"]
ENC_LABEL = {"sparse": "One-hot", "blosum50": "BLOSUM50", "blosum62": "BLOSUM62",
             "blosum80": "BLOSUM80", "kidera": "Kidera", "atchley": "Atchley"}

# padmode_choice palette (plot_padmode_data_compare.GROUP_COLORS) + Tol sand
# for the explicit feature set.
C_SOLO = "#BBBBBB"          # AA encoding (Tol grey)
C_C    = "#EE6677"          # ESM-C  (Tol red)
C_IF   = "#4477AA"          # ESM-IF (Tol blue)
C_EXP  = "#DDCC77"          # explicit features (Tol sand)


def _tint(hex_color, mix=0.5):
    """Blend hex_color toward white by `mix` (reproducible combo tints)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    f = lambda c: int(round(c + (255 - c) * mix))
    return f"#{f(r):02X}{f(g):02X}{f(b):02X}"


C_C_COMBO  = _tint(C_C)     # light red
C_IF_COMBO = _tint(C_IF)    # light blue
C_EXP_FULL = _tint(C_EXP, mix=-0.5)   # darker sand (explicit-alone bars)

# how each embedding-only arm was reduced: n_features -> legend note
REDUCTION_NOTES = {
    18647: "80% EV",               # ESM-C top-variance (vardims default)
    7540: "65% EV",                # ESM-IF top-variance (vardims default)
    2278: "PCA 80% EV (k=2278)",   # old campaign reduction
    632: "PCA 60% EV (k=632)",
}


def reduction_note(models, emb_only_key):
    """Legend note derived from the arm's actual feature count, so the old
    PCA campaign and the vardims redo each label themselves truthfully."""
    k = models[emb_only_key]["k"]
    return REDUCTION_NOTES.get(k, f"k={k}")

EMB_ONLY = [
    ("esmif_eos_bos_repeat", "ESM-IF", C_IF, C_IF_COMBO),
    ("esmc300m_win_padtoken", "ESM-C", C_C, C_C_COMBO),
]


def exp_key(enc):
    return f"{EXP}_{enc}"


def three_key(enc, emb):
    return f"{EXP}_{enc}_{emb}"


def enc_emb_key(enc, emb):
    return f"{enc}_{emb}"


def exp_label(enc=None):
    """Display label for the handcrafted feature set."""
    return f"handcrafted+{ENC_LABEL[enc]}" if enc else "handcrafted features"


# ── plotting helpers (padmode_choice style, copied from 18_) ─────────────
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


def _padmode_bars(ax, xs, values, errors, colors, width=0.62):
    ax.bar(xs, values, yerr=errors, color=colors, width=width,
           edgecolor="black", linewidth=0.5, capsize=3)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)


def _padmode_brackets(ax, pairs, values, fontsize=7.5, stars_only=False):
    """pairs: [(x_a, x_b, p, p_adj), ...]. Widest span on top; fits under
    y=0.97 on the 0-1 axis."""
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
    ax.set_ylabel("CV-AUC")
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


# ── figures ─────────────────────────────────────────────────────────────
def plot_effect_on_encodings(models, famExpEnc, out_png,
                             alpha=ALPHA):
    """6 interleaved groups [encoding, encoding+handcrafted] plus ONE
    standalone handcrafted anchor bar — it is a single constant arm, no
    need to repeat it per group. Brackets: E1 only (encoding vs its
    combo), stars-only; E2 stats stay in the TSV (no brackets to the
    anchor)."""
    step_g = 2
    n_g = len(ENCODINGS)
    xs, vals, errs, colors = [], [], [], []
    key_to_x = {}
    for gi, enc in enumerate(ENCODINGS):
        key_to_x[enc] = gi * step_g
        key_to_x[exp_key(enc)] = gi * step_g + 1
        for x, k, c in ((key_to_x[enc], enc, C_SOLO),
                        (key_to_x[exp_key(enc)], exp_key(enc), C_EXP)):
            xs.append(x)
            vals.append(models[k]["auc"])
            errs.append(models[k]["outer_std"])
            colors.append(c)
    anchor_x = n_g * step_g + 1                      # gap before the anchor
    xs.append(anchor_x)
    vals.append(models[EXP]["auc"])
    errs.append(models[EXP]["outer_std"])
    colors.append(C_EXP_FULL)

    famE1 = {pr["enc"]: pr for pr in famExpEnc}
    pairs = []
    for enc in ENCODINGS:
        p1 = famE1.get(enc)
        if p1 and p1["p_adj"] < alpha:
            pairs.append((key_to_x[enc], key_to_x[exp_key(enc)],
                          p1["p"], p1["p_adj"]))

    fig, ax = plt.subplots(figsize=(13, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    for gi in range(n_g - 1):
        ax.axvline(gi * step_g + 1.5, color="grey", linestyle="--",
                   linewidth=0.8)
    ax.axvline(anchor_x - 0.75, color="grey", linestyle="--", linewidth=0.8)
    _padmode_brackets(ax, pairs, vals, stars_only=True)
    ax.set_xticks([gi * step_g + 0.5 for gi in range(n_g)] + [anchor_x])
    ax.set_xticklabels([ENC_LABEL[e] for e in ENCODINGS] + ["handcrafted"],
                       rotation=30, ha="right", fontsize=9)
    _finish_padmode(fig, ax,
                    "Do handcrafted features improve the AA encodings?",
                    out_png,
                    [("AA encoding", C_SOLO),
                     ("handcrafted + encoding", C_EXP),
                     ("handcrafted features", C_EXP_FULL)])


def _hub_entries(sig, keys, key_to_x):
    """Hub brackets: per left key, ONE bracket to its rightmost significant
    partner; down-ticks at intermediate significant partners; text shows the
    best (lowest p_adj) partner pair."""
    hubs = []
    for i, ka in enumerate(keys):
        rightmost, best, partners = None, None, []
        for kb in keys[i + 1:]:
            hit = next((pr for pr in sig
                        if {pr["name_a"], pr["name_b"]} == {ka, kb}), None)
            if hit is not None:
                rightmost = kb
                partners.append(kb)
                if best is None or hit["p_adj"] < best["p_adj"]:
                    best = hit
        if rightmost is not None:
            hubs.append((key_to_x[ka], key_to_x[rightmost],
                         best["p"], best["p_adj"],
                         [key_to_x[ka]] + [key_to_x[k] for k in partners]))
    return hubs


def _draw_hubs(ax, hubs, vals):
    if not hubs:
        return
    y0 = max(vals) * 1.08
    max_lvl = len(hubs)
    headroom = 0.97 - y0 - max(vals) * 0.03
    step = min(max(vals) * 0.085, headroom / (max_lvl - 1 + 0.35))
    fontsize = 7.5 if step >= max(vals) * 0.05 else 6.5
    for lvl, (xa, xb, p, p_adj, tick_keys) in enumerate(sorted(
            hubs, key=lambda h: -(h[1] - h[0]))):
        y_br = y0 + lvl * step
        ax.plot([xa, xa, xb, xb], [y_br * 0.985, y_br, y_br, y_br * 0.985],
                color="black", lw=0.9)
        for xt in tick_keys:            # "differs from each" down-ticks
            if xt not in (xa, xb):
                ax.plot([xt, xt], [y_br * 0.985, y_br], color="black", lw=0.9)
        ax.text((xa + xb) / 2, y_br + step * 0.12,
                stars(p_adj) if p is None
                else f"{fmt_p_pair(p, p_adj)} ({stars(p_adj)})",
                ha="center", va="bottom", fontsize=fontsize)


def plot_exp_enc_ranking(models, famExpPair, out_png, alpha=ALPHA):
    """Flat 6 bars, handcrafted+encoding arms all-vs-all (pairwise DeLong,
    Holm within family) — the best-pick figure: E1 already established
    that encodings gain slightly from handcrafted, so only the combos
    compete here. Hub brackets (padmode 'baselines' layout): ONE bracket
    per left bar to its rightmost significant partner, down-ticks at
    intermediate significant bars; ns contrasts live in the stats TSV."""
    xs = list(range(len(ENCODINGS)))
    key_to_x = {exp_key(enc): i for i, enc in enumerate(ENCODINGS)}
    vals = [models[exp_key(enc)]["auc"] for enc in ENCODINGS]
    errs = [models[exp_key(enc)]["outer_std"] for enc in ENCODINGS]
    colors = [C_EXP] * len(ENCODINGS)

    sig = [pr for pr in famExpPair if pr["p_adj"] < alpha]
    hubs = _hub_entries(sig, [exp_key(e) for e in ENCODINGS], key_to_x)

    best_enc = max(ENCODINGS, key=lambda e: models[exp_key(e)]["auc"])
    ns_n = len(famExpPair) - len(sig)
    footnote = (
        f"best arm: handcrafted+{ENC_LABEL[best_enc]} "
        f"{models[exp_key(best_enc)]['auc']:.4f}; "
        f"{len(famExpPair) - ns_n}/{len(famExpPair)} pairwise contrasts "
        "significant (delong_stats.tsv, family explicit_pairwise)."
    )

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    _draw_hubs(ax, hubs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{ENC_LABEL[e]}\nk={models[exp_key(e)]['k']}"
                        for e in ENCODINGS],
                       rotation=30, ha="right", fontsize=9)
    _finish_padmode(fig, ax,
                    "Which handcrafted+encoding combination is best?",
                    out_png, [("handcrafted + encoding", C_EXP)],
                    footnote=footnote)


def plot_exp_emb_ranking(models, famEmbPair, out_png, alpha=ALPHA):
    """Flat 4 bars, embedding arms compared against each other (pairwise
    DeLong, Holm within family): ESM-IF / ESM-C / handcrafted+ESM-IF /
    handcrafted+ESM-C. Hub brackets, same layout as the encodings ranking."""
    bars = [("emb_esmif_eos_bos_repeat", C_IF, "ESM-IF"),
            ("emb_esmc300m_win_padtoken", C_C, "ESM-C"),
            (f"{EXP}_esmif_eos_bos_repeat", C_EXP, "handcrafted + ESM-IF"),
            (f"{EXP}_esmc300m_win_padtoken", C_EXP, "handcrafted + ESM-C")]
    xs = list(range(len(bars)))
    key_to_x = {k: x for x, (k, _c, _l) in enumerate(bars)}
    vals = [models[k]["auc"] for k, _c, _l in bars]
    errs = [models[k]["outer_std"] for k, _c, _l in bars]
    colors = [c for _k, c, _l in bars]

    sig = [pr for pr in famEmbPair if pr["p_adj"] < alpha]
    hubs = _hub_entries(sig, [k for k, _c, _l in bars], key_to_x)

    best_k = max((k for k, _c, _l in bars), key=lambda k: models[k]["auc"])
    best_lb = next(l for k, _c, l in bars if k == best_k)
    ns_n = len(famEmbPair) - len(sig)
    footnote = (
        f"best arm: {best_lb} {models[best_k]['auc']:.4f}; "
        f"{ns_n}/{len(famEmbPair)} pairwise contrasts ns "
        "(handcrafted_sweep_delong_stats.tsv, family "
        "explicit_embedding_pairwise)."
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    _draw_hubs(ax, hubs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([lb for _k, _c, lb in bars], rotation=20, ha="right",
                       fontsize=9)
    _finish_padmode(fig, ax,
                    "Embedding arms compared: does handcrafted "
                    "add anything?",
                    out_png,
                    [(f"ESM-IF {reduction_note(models, 'emb_esmif_eos_bos_repeat')}",
                      C_IF),
                     (f"ESM-C {reduction_note(models, 'emb_esmc300m_win_padtoken')}",
                      C_C),
                     ("handcrafted + embedding", C_EXP)],
                    footnote=footnote)


def plot_emb_explicit(models, famEmbExp, famEmbExpVsExp, emb_comp, emb_label,
                      emb_full, out_png, alpha=ALPHA):
    """Flat 3 bars, NO encoding: embedding alone / explicit+embedding /
    explicit features alone. Brackets: explicit+emb vs emb (the explicit
    effect) and explicit+emb vs explicit, significant pairs only."""
    keys = [f"emb_{emb_comp}", f"{EXP}_{emb_comp}", EXP]
    xs = list(range(3))
    vals = [models[k]["auc"] for k in keys]
    errs = [models[k]["outer_std"] for k in keys]
    colors = [emb_full, C_EXP, C_EXP_FULL]

    pairs = []
    for pr in famEmbExp:
        if pr["p_adj"] < alpha:
            pairs.append((0, 1, pr["p"], pr["p_adj"]))
    for pr in famEmbExpVsExp:
        if pr["p_adj"] < alpha:
            pairs.append((1, 2, pr["p"], pr["p_adj"]))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    _padmode_bars(ax, xs, vals, errs, colors)
    _padmode_brackets(ax, pairs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([emb_label, f"handcrafted + {emb_label}",
                        "handcrafted features"], fontsize=10)
    _note = reduction_note(models, f"emb_{emb_comp}")
    _finish_padmode(fig, ax,
                    "Do handcrafted features improve "
                    f"{emb_label}?",
                    out_png,
                    [(f"{emb_label} {_note}", emb_full),
                     (f"handcrafted + {emb_label} ({_note})", C_EXP),
                     ("handcrafted features", C_EXP_FULL)])


# ── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--campaign", default="handcrafted_sweep",
                    help="primary run folder under runs/ (default: "
                         "handcrafted_sweep)")
    ap.add_argument("--ref-campaign", default="encodings_sweep",
                    help="campaign supplying emb_* and encoding+embedding "
                         "reference arms (default: encodings_sweep)")
    args = ap.parse_args()
    campaign, ref_campaign = args.campaign, args.ref_campaign

    split = pd.read_csv(SPLIT_DATA_PATH)
    fold_vec = split["fold"].to_numpy(dtype=int)
    y = split["label"].to_numpy(dtype=int)
    y_md5 = hashlib.md5(y.tobytes()).hexdigest()
    print(f"Loaded split: {len(y)} rows, {int(y.sum())} positive, "
          f"{len(set(fold_vec))} folds, labels md5 {y_md5}")

    # primary campaign + reference campaign (emb_*/enc+emb arms); scan every
    # models/ subdir (flat for CSV arms, vardims for --vardims arms)
    paths = []
    for root in dict.fromkeys((campaign, ref_campaign)):
        hits = sorted(glob.glob(str(
            PROJECT_ROOT / "runs" / root / "models" / "*" /
            "cv_results_*.json")))
        print(f"Found {len(hits)} result JSONs in runs/{root}")
        paths.extend(hits)
    models = {}
    for p in paths:
        m = load_model(p, fold_vec, y, os.path.basename(p))
        models[m["key"]] = m
        print(f"  {m['key']:<48} k={m['k']:<5} pooled AUC={m['auc']:.4f} "
              f"±{m['outer_std']:.4f}")

    # sanity: fold/label identity across every arm, incl. symlinked refs
    bad = [k for k, m in models.items() if m["label_md5"] != y_md5]
    if bad:
        raise SystemExit(f"Fold/label md5 mismatch in arms: {bad}")
    print("Fold/label md5 identical across all arms "
          f"({len(models)} arms)")

    # core arms (always required): embeddings + handcrafted + handcrafted×emb
    core = [EXP] + [f"{EXP}_{emb}" for emb, _nm, _c, _ct in EMB_ONLY] + \
           [f"emb_{emb}" for emb, _nm, _c, _ct in EMB_ONLY]
    missing_core = [k for k in core if k not in models]
    if missing_core:
        raise SystemExit(f"Missing core arms: {missing_core}")

    # level A: handcrafted×encoding singles (embedding-free, pure CSV) —
    # unlock E1/E2/E5 + the two encodings figures + encoding_only rows
    enc_needed = [e for e in ENCODINGS] + [exp_key(e) for e in ENCODINGS]
    missing_enc = [k for k in enc_needed if k not in models]
    has_enc_arms = not missing_enc
    if missing_enc:
        print(f"  WARN: {len(missing_enc)} encoding arms missing "
              f"(e.g. {missing_enc[0]}) — E1/E2/E5 + encodings figures "
              "will be skipped")

    # level B: three-way handcrafted×encoding×embedding arms (PCA-era in
    # the old campaign; deliberately absent on helping-v2) — unlock
    # T1-T3 + the with_embedding delta rows
    three_needed = [three_key(e, emb) for e in ENCODINGS
                    for emb, _nm, _c, _ct in EMB_ONLY]
    missing_three = [k for k in three_needed if k not in models]
    has_three = not missing_three
    if not has_three:
        print(f"  WARN: {len(missing_three)} three-way arms missing "
              f"(e.g. {missing_three[0]}) — T1-T3 + with_embedding delta "
              "rows will be skipped")

    # ── DeLong families (Holm within family) ──
    # E1/E2/E5/T1-T3 need handcrafted×encoding (+three-way) arms, which
    # the helping-v2 campaign does not contain — skip instead of aborting.
    famExpEnc, famExpAlone, famExpPair = [], [], []
    famT1, famT2, famT3 = [], [], []
    if has_enc_arms:
        famExpEnc = []      # E1: explicit+enc vs enc alone
        famExpAlone = []    # E2: explicit+enc vs explicit alone
        for enc in ENCODINGS:
            a, b, d, se, z, p = delong_full(y, models[exp_key(enc)]["probs"],
                                            models[enc]["probs"])
            famExpEnc.append(dict(enc=enc, name_a=exp_key(enc), name_b=enc,
                                  auc_a=a, auc_b=b, delta=d, se=se, z=z, p=p))
            a2, b2, d2, se2, z2, p2 = delong_full(y, models[exp_key(enc)]["probs"],
                                                  models[EXP]["probs"])
            famExpAlone.append(dict(enc=enc, name_a=exp_key(enc), name_b=EXP,
                                    auc_a=a2, auc_b=b2, delta=d2, se=se2,
                                    z=z2, p=p2))
        holm_family(famExpEnc)
        holm_family(famExpAlone)
        print(f"\nFamily E1 (explicit+enc vs enc): {len(famExpEnc)} pairs, "
              f"{sum(1 for p in famExpEnc if p['p_adj'] < ALPHA)} significant")
        print(f"Family E2 (explicit+enc vs explicit): {len(famExpAlone)} pairs, "
              f"{sum(1 for p in famExpAlone if p['p_adj'] < ALPHA)} significant")

        # E5: all pairwise explicit+encoding vs each other (ranking figure)
        famExpPair = []
        for i, a in enumerate(ENCODINGS):
            for b in ENCODINGS[i + 1:]:
                auc_a, auc_b, d, se, z, p = delong_full(
                    y, models[exp_key(a)]["probs"], models[exp_key(b)]["probs"])
                famExpPair.append(dict(enc_a=a, enc_b=b, name_a=exp_key(a),
                                       name_b=exp_key(b), auc_a=auc_a,
                                       auc_b=auc_b, delta=d, se=se, z=z, p=p))
        holm_family(famExpPair)
        print(f"Family E5 (explicit+enc pairwise): {len(famExpPair)} pairs, "
              f"{sum(1 for p in famExpPair if p['p_adj'] < ALPHA)} significant")
    else:
        print("\nSKIP: families E1/E2/E5 — no handcrafted×encoding arms "
              "(need runs/<campaign>/handcrafted_<enc>)")

    if has_three:
        famT1, famT2, famT3 = [], [], []    # three-way vs enc+emb / exp+enc / exp
        for enc in ENCODINGS:
            for emb, _nm, _c, _ct in EMB_ONLY:
                k3 = three_key(enc, emb)
                a, b, d, se, z, p = delong_full(
                    y, models[k3]["probs"], models[enc_emb_key(enc, emb)]["probs"])
                famT1.append(dict(enc=enc, emb=emb, name_a=k3,
                                  name_b=enc_emb_key(enc, emb), auc_a=a,
                                  auc_b=b, delta=d, se=se, z=z, p=p))
                a2, b2, d2, se2, z2, p2 = delong_full(
                    y, models[k3]["probs"], models[exp_key(enc)]["probs"])
                famT2.append(dict(enc=enc, emb=emb, name_a=k3, name_b=exp_key(enc),
                                  auc_a=a2, auc_b=b2, delta=d2, se=se2,
                                  z=z2, p=p2))
                a3, b3, d3, se3, z3, p3 = delong_full(
                    y, models[k3]["probs"], models[EXP]["probs"])
                famT3.append(dict(enc=enc, emb=emb, name_a=k3, name_b=EXP,
                                  auc_a=a3, auc_b=b3, delta=d3, se=se3,
                                  z=z3, p=p3))
        holm_family(famT1)
        holm_family(famT2)
        holm_family(famT3)
        print(f"Family T1 (three-way vs enc+emb): {len(famT1)} pairs, "
              f"{sum(1 for p in famT1 if p['p_adj'] < ALPHA)} significant")
        print(f"Family T2 (three-way vs explicit+enc): {len(famT2)} pairs, "
              f"{sum(1 for p in famT2 if p['p_adj'] < ALPHA)} significant")
        print(f"Family T3 (three-way vs explicit alone): {len(famT3)} pairs, "
              f"{sum(1 for p in famT3 if p['p_adj'] < ALPHA)} significant")
    else:
        print("SKIP: families T1-T3 — no three-way arms (by design on "
              "helping-v2; PCA-era triples would mix reductions)")

    # ── E6/E7/E8: explicit+embedding arms (only once the 2 new cluster jobs
    #    handcrafted_<emb> have landed; otherwise figures are skipped) ──
    famEmbExp, famEmbExpVsExp, famEmbPair = [], [], []
    embs_with_exp = [emb for emb, _nm, _c, _ct in EMB_ONLY
                     if f"{EXP}_{emb}" in models]
    for emb in embs_with_exp:
        a, b, d, se, z, p = delong_full(
            y, models[f"{EXP}_{emb}"]["probs"], models[f"emb_{emb}"]["probs"])
        famEmbExp.append(dict(emb=emb, name_a=f"{EXP}_{emb}",
                              name_b=f"emb_{emb}", auc_a=a, auc_b=b,
                              delta=d, se=se, z=z, p=p))
        a2, b2, d2, se2, z2, p2 = delong_full(
            y, models[f"{EXP}_{emb}"]["probs"], models[EXP]["probs"])
        famEmbExpVsExp.append(dict(emb=emb, name_a=f"{EXP}_{emb}", name_b=EXP,
                                   auc_a=a2, auc_b=b2, delta=d2, se=se2,
                                   z=z2, p=p2))
    holm_family(famEmbExp)
    holm_family(famEmbExpVsExp)
    if embs_with_exp:
        print(f"Family E6 (explicit+emb vs emb): {len(famEmbExp)} pairs, "
              f"{sum(1 for p in famEmbExp if p['p_adj'] < ALPHA)} significant")
        print(f"Family E7 (explicit+emb vs explicit): {len(famEmbExpVsExp)} "
              f"pairs, {sum(1 for p in famEmbExpVsExp if p['p_adj'] < ALPHA)} "
              "significant")
    else:
        print("Families E6/E7 skipped: no explicit+embedding arms found yet "
              "(handcrafted_esmif_eos_bos_repeat / "
              "handcrafted_esmc300m_win_padtoken)")

    # E8: all 4 embedding arms pairwise (ranking figure)
    if len(embs_with_exp) == len(EMB_ONLY):
        emb4 = [f"emb_{emb}" for emb, _nm, _c, _ct in EMB_ONLY] + \
               [f"{EXP}_{emb}" for emb, _nm, _c, _ct in EMB_ONLY]
        for i, a in enumerate(emb4):
            for b in emb4[i + 1:]:
                auc_a, auc_b, d, se, z, p = delong_full(
                    y, models[a]["probs"], models[b]["probs"])
                famEmbPair.append(dict(name_a=a, name_b=b, auc_a=auc_a,
                                       auc_b=auc_b, delta=d, se=se, z=z, p=p))
        holm_family(famEmbPair)
        print(f"Family E8 (embedding-arm pairwise): {len(famEmbPair)} pairs, "
              f"{sum(1 for p in famEmbPair if p['p_adj'] < ALPHA)} significant")

    # ── figures (padmode_choice style) ──
    out_dir = PROJECT_ROOT / "runs" / campaign / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if has_enc_arms:
        plot_effect_on_encodings(models, famExpEnc,
                                 out_dir / "handcrafted_effect_on_encodings.png")
        plot_exp_enc_ranking(models, famExpPair,
                             out_dir / "handcrafted_encodings_ranking.png")
    else:
        print("  (skipping handcrafted_effect_on_encodings.png / "
              "handcrafted_encodings_ranking.png — no handcrafted×encoding "
              "arms)")
    if len(embs_with_exp) == len(EMB_ONLY):
        plot_exp_emb_ranking(models, famEmbPair,
                             out_dir / "handcrafted_embeddings_ranking.png")
    else:
        print("  (skipping handcrafted_embeddings_ranking.png — "
              "handcrafted+embedding arms not landed yet)")
    for emb, nm, full, _combo in EMB_ONLY:
        if emb in embs_with_exp:
            fe = next(p for p in famEmbExp if p["emb"] == emb)
            fv = next(p for p in famEmbExpVsExp if p["emb"] == emb)
            plot_emb_explicit(models, [fe], [fv], emb, nm, full,
                              out_dir / f"handcrafted_effect_on_{nm.lower().replace('-', '')}.png")
        else:
            print(f"  (skipping handcrafted_effect_on_{nm.lower().replace('-', '')}"
                  ".png — handcrafted+embedding arm not landed yet)")

    # ── TSV: every tested pair ──
    res_dir = PROJECT_ROOT / "runs" / campaign / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    stat_rows = []
    for fam, pairs in (("explicit_vs_encoding", famExpEnc),
                       ("explicit_vs_explicit_alone", famExpAlone),
                       ("three_way_vs_enc_emb", famT1),
                       ("three_way_vs_explicit_enc", famT2),
                       ("three_way_vs_explicit", famT3)):
        for pr in pairs:
            stat_rows.append(dict(
                family=fam, encoding=pr["enc"], embedding=pr.get("emb", ""),
                name_a=pr["name_a"], name_b=pr["name_b"],
                auc_a=f"{pr['auc_a']:.6f}", auc_b=f"{pr['auc_b']:.6f}",
                delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
                z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
                p_adj=f"{pr['p_adj']:.3e}",
                significant=str(pr["p_adj"] < ALPHA)))
    for pr in famExpPair:
        stat_rows.append(dict(
            family="explicit_pairwise", encoding=pr["enc_a"],
            embedding=pr["enc_b"], name_a=pr["name_a"], name_b=pr["name_b"],
            auc_a=f"{pr['auc_a']:.6f}", auc_b=f"{pr['auc_b']:.6f}",
            delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
            z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
            p_adj=f"{pr['p_adj']:.3e}",
            significant=str(pr["p_adj"] < ALPHA)))
    for fam, pairs in (("explicit_emb_vs_emb", famEmbExp),
                       ("explicit_emb_vs_explicit", famEmbExpVsExp)):
        for pr in pairs:
            stat_rows.append(dict(
                family=fam, encoding="", embedding=pr["emb"],
                name_a=pr["name_a"], name_b=pr["name_b"],
                auc_a=f"{pr['auc_a']:.6f}", auc_b=f"{pr['auc_b']:.6f}",
                delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
                z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
                p_adj=f"{pr['p_adj']:.3e}",
                significant=str(pr["p_adj"] < ALPHA)))
    for pr in famEmbPair:
        stat_rows.append(dict(
            family="explicit_embedding_pairwise", encoding="",
            embedding="", name_a=pr["name_a"], name_b=pr["name_b"],
            auc_a=f"{pr['auc_a']:.6f}", auc_b=f"{pr['auc_b']:.6f}",
            delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
            z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
            p_adj=f"{pr['p_adj']:.3e}",
            significant=str(pr["p_adj"] < ALPHA)))
    stats_tsv = res_dir / f"{campaign}_delong_stats.tsv"
    with open(stats_tsv, "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(stat_rows)
    print(f"  Wrote {stats_tsv}")

    # ── TSV: explicit-feature contrasts + memory columns ──
    # (needs the handcrafted×encoding families — absent on helping-v2)
    if not has_enc_arms:
        print("  (skipping handcrafted_deltas TSV — no handcrafted×"
              "encoding arms)")
        return
    famT1_by = {(pr["enc"], pr["emb"]): pr for pr in famT1}
    famT2_by = {(pr["enc"], pr["emb"]): pr for pr in famT2}
    delta_rows = []
    for enc in ENCODINGS:
        pr = next(p for p in famExpEnc if p["enc"] == enc)
        mem_exp = read_mem(exp_key(enc), campaign, ref_campaign)
        mem_base = read_mem(enc, campaign, ref_campaign)
        delta_rows.append(dict(
            context="encoding_only", embedding="", encoding=enc,
            explicit_arm=exp_key(enc), baseline_arm=enc,
            explicit_auc=f"{models[exp_key(enc)]['auc']:.6f}",
            baseline_auc=f"{models[enc]['auc']:.6f}",
            delta=f"{pr['delta']:.6f}", se=f"{pr['se']:.6f}",
            z=f"{pr['z']:.4f}", p=f"{pr['p']:.3e}",
            p_adj=f"{pr['p_adj']:.3e}",
            explicit_peak_rss_kb=mem_exp["peak_rss_kb"],
            explicit_cgroup_peak_bytes=mem_exp["cgroup_peak_bytes"],
            explicit_wall_s=mem_exp["wall_s"],
            explicit_cpu_pct=mem_exp["cpu_pct"],
            baseline_peak_rss_kb=mem_base["peak_rss_kb"],
            baseline_cgroup_peak_bytes=mem_base["cgroup_peak_bytes"],
            baseline_wall_s=mem_base["wall_s"],
            baseline_cpu_pct=mem_base["cpu_pct"],
        ))
        for emb, nm, _c, _ct in (EMB_ONLY if has_three else []):
            p1 = famT1_by[(enc, emb)]
            p2 = famT2_by[(enc, emb)]
            k3 = three_key(enc, emb)
            mem_exp = read_mem(k3, campaign, ref_campaign)
            mem_base = read_mem(enc_emb_key(enc, emb), campaign, ref_campaign)
            delta_rows.append(dict(
                context="with_embedding", embedding=nm, encoding=enc,
                explicit_arm=k3, baseline_arm=enc_emb_key(enc, emb),
                explicit_auc=f"{models[k3]['auc']:.6f}",
                baseline_auc=f"{models[enc_emb_key(enc, emb)]['auc']:.6f}",
                delta=f"{p1['delta']:.6f}", se=f"{p1['se']:.6f}",
                z=f"{p1['z']:.4f}", p=f"{p1['p']:.3e}",
                p_adj=f"{p1['p_adj']:.3e}",
                explicit_peak_rss_kb=mem_exp["peak_rss_kb"],
                explicit_cgroup_peak_bytes=mem_exp["cgroup_peak_bytes"],
                explicit_wall_s=mem_exp["wall_s"],
                explicit_cpu_pct=mem_exp["cpu_pct"],
                baseline_peak_rss_kb=mem_base["peak_rss_kb"],
                baseline_cgroup_peak_bytes=mem_base["cgroup_peak_bytes"],
                baseline_wall_s=mem_base["wall_s"],
                baseline_cpu_pct=mem_base["cpu_pct"],
            ))
    deltas_tsv = res_dir / f"{campaign}_deltas.tsv"
    with open(deltas_tsv, "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(delta_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(delta_rows)
    print(f"  Wrote {deltas_tsv}")

    print("\nCaption caveat: detection threshold ~0.0024 AUC vs outer-fold "
          "SD 0.003-0.009; significant != material.")


if __name__ == "__main__":
    main()
