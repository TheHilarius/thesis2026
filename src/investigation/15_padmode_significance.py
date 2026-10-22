#!/usr/bin/env python3
"""
15_padmode_significance.py
Paired significance tests for the padmode/data comparison batch.

Reuses saved nested-CV out-of-fold (OOF) predictions (cv_results_*.json) —
no retraining. All comparisons are PAIRED: both models score the identical
rows under the identical CV partition (random_state=42).

Primary row-level test — paired DeLong (correlated ROC curves, midrank
structure terms à la Sun & Xu 2014) on the POOLED OOF probability vectors.
Deterministic; p = 2*norm.sf(|z|). Holm-corrected within test family
(p_delong_holm). Pooled AUC is the estimand (mean-of-fold AUC is a
different, slightly noisier estimand — that lives in the fold t-test).

Cross-checks per pair:
  - paired row-level bootstrap (B=10000, per-pair RNG seeded from
    sha1(f"{seed}|{group}|{a}|{b}") so adding test families never shifts
    existing CIs): percentile 95% CI on ΔAUC; p = 2*min(P(Δ*<=0),
    P(Δ*>=0)) with +1 smoothing (B-resolution floor ~1e-4).
  - fold-paired Student's t-test on the 6 outer-fold AUCs (subset fold
    AUCs for padmode/baseline-subset families; overall fold AUCs for
    *_full/baseline_full). Approximate: folds share training data.
    Nadeau-Bengio sensitivity: t_nb = t / sqrt(1 + k*n_test/n_train)
    = t/sqrt(2.2) for 6-fold (df=5 kept as heuristic).

Test groups (Holm within each):
  padmode_<tk>          pairwise, padded-subset rows (~1443)
  padmode_<tk>_full     pairwise, all 46617 rows (DeLong + t only)
  baseline_full         sparse/blosum/handcrafted_blosum pairs, full rows
                        (bootstrap only for sparse-vs-blosum)
  baseline_subset_esmc  same pairs, ESM-C-mask padded-subset rows

Checkpoint campaign (--arm-key checkpoint): arms are the ProteinMPNN
train-noise checkpoints (v48002/v48010/v48020/v48030, all zeropad) instead
of pad-fill modes — they all share padtype "zeropad" in PADTYPE_MAP, so
the arm label is parsed from the component name. Groups:
  ckpt_<tk>             pairwise, padded-subset rows
  ckpt_<tk>_full        pairwise, all rows (DeLong + t only)
  ckpt_vs_<tk2>         every checkpoint vs each loaded <tk2> arm
                        (e.g. ESM-IF EOS/BOS-repeat), padded-subset rows;
                        subset block only when the two toolkits' padded
                        masks are row-identical (asserted, MPNN==ESM-IF)
  ckpt_vs_<tk2>_full    same pairs, all rows

No cross-toolkit subset comparisons (masks differ on notfound rows) —
asserted below. Equivalence is NOT formally tested (TOST dropped: a
+/-0.03 margin would exceed the sparse-blosum full-data effect |d|=0.022);
report bootstrap CIs instead.

Inputs (defaults target runs/padmode_choice/):
    --json-dir  cv_results_*.json files (xgb, pca_mode flat)
    pad masks from raw embedding HDF5s (same sources as
        09_padmode_subset_auc.py) + folds/labels from df_all_with_folds.csv

Output:
    runs/padmode_choice/results/paired_tests.csv + stdout table.
    CSV columns per pair: auc_a, auc_b, delta_auc,
    se_delta_delong, z, ci95_lo, ci95_hi (= Δ ± 1.96·SE, primary),
    p_delong, p_delong_holm,
    ci_lo, ci_hi, p_value, p_holm (bootstrap, secondary),
    t_stat, p_ttest, t_stat_nb, p_ttest_nb, p_ttest_nb_holm (fold-level).

Usage:
    env_esmc/bin/python src/investigation/15_padmode_significance.py
    env_esmc/bin/python src/investigation/15_padmode_significance.py --smoke
    env_esmc/bin/python src/investigation/15_padmode_significance.py --skip-bootstrap
    # ProteinMPNN checkpoint campaign (all zeropad):
    env_esmc/bin/python src/investigation/15_padmode_significance.py \
        --arm-key checkpoint \
        --json-dir runs/mpnn_ckpt_zeropad/models/flat \
        --out runs/mpnn_ckpt_zeropad/results/paired_tests.csv
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist, ttest_rel
from sklearn.metrics import roc_auc_score

# 14_ has a leading digit — import via importlib; it sets up pipeline/python
# on sys.path for `config` and owns the pad-mask registry. PADTYPE_MAP here is
# 14_'s FEATURE_MAP (feature_key -> (toolkit, padtype)) — same shape, old name.
import importlib
_mod = importlib.import_module("14_plot_grid_auc")
PADMASK_FILES = _mod.PADMASK_FILES
PADTYPE_MAP = _mod.FEATURE_MAP
load_padded_masks = _mod.load_padded_masks

from config import PROJECT_ROOT, SPLIT_DATA_PATH  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "runs" / "padmode_choice" / "models" / "flat"
OUT_CSV = PROJECT_ROOT / "runs" / "padmode_choice" / "results" / "paired_tests.csv"

# Nadeau-Bengio correction factor sqrt(1 + k * n_test/n_train) for
# 6-fold CV (n_test/n_train = 1/5) -> sqrt(2.2).
NB_FACTOR = np.sqrt(2.2)
NB_DF = 5

EMB_JSON_RE = re.compile(
    r"cv_results_xgb_emb_"
    r"(esmc_win_(?:zeropad|impute_boundary|impute_bos_eos|padtoken)"
    r"|esmif_(?:win|pad|boundary|eos_bos_repeat)"
    r"|mpnn_v48\d{3}_(?:zeropad|boundary))"
    r"_pca(\d+)_flat_"
)
BASELINE_JSON_RE = re.compile(
    r"cv_results_xgb_(sparse|blosum|handcrafted_blosum|"
    r"handcrafted_sparse|handcrafted)_flat_"
)
# component name -> train-noise checkpoint id (v48002 ... v48030)
CKPT_RE = re.compile(r"^mpnn_(v48\d{3})_")


def reconstruct_oof(res: dict, n: int) -> np.ndarray:
    """Full-length OOF probability vector from outer_val_predictions."""
    probs = np.full(n, np.nan)
    for fk, entry in res["outer_val_predictions"].items():
        mask = df_folds == int(fk)
        if int(mask.sum()) != len(entry["y_prob"]):
            raise ValueError(f"fold {fk}: size mismatch")
        probs[mask] = np.asarray(entry["y_prob"], dtype=float)
    if np.isnan(probs).any():
        raise ValueError("rows never predicted")
    return probs


def delong_structure(y: np.ndarray, p: np.ndarray):
    """DeLong V10 (positives) and V01 (negatives) structure terms."""
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


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    """Paired DeLong test for two correlated AUCs on the same rows.

    Returns (auc_a, auc_b, var_delta). Conditional (scores fixed)
    variance; z = delta / sqrt(var) in caller.
    """
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


def paired_bootstrap_delta(y, p_a, p_b, B, rng):
    """Paired bootstrap of AUC(p_a) - AUC(p_b); same rows resampled."""
    n = len(y)
    deltas = np.empty(B)
    auc_a = roc_auc_score(y, p_a)
    auc_b = roc_auc_score(y, p_b)
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            deltas[b] = np.nan
            continue
        deltas[b] = roc_auc_score(yb, p_a[idx]) - roc_auc_score(yb, p_b[idx])
    deltas = deltas[~np.isnan(deltas)]
    delta = auc_a - auc_b
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = min(1.0, (p * len(deltas) + 1) / (len(deltas) + 1))  # +1 smoothing
    return delta, lo, hi, p, auc_a, auc_b, float(np.std(deltas, ddof=1))


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values (NaN-safe)."""
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


def per_pair_rng(master_seed, group, ma, mb):
    """Deterministic RNG per (group, pair) — draw order immune to
    adding new test families."""
    key = f"{master_seed}|{group}|{ma}|{mb}".encode()
    h = int.from_bytes(hashlib.sha1(key).digest()[:8], "little")
    return np.random.default_rng(h)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-dir", default=str(MODELS_DIR))
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="Skip the secondary row-level bootstrap (fast "
                         "dev runs; primary DeLong + fold tests still run)")
    ap.add_argument("--arm-key", default="padtype",
                    choices=["padtype", "checkpoint"],
                    help="Arm axis: padtype (pad-fill modes per toolkit) or "
                         "checkpoint (ProteinMPNN train-noise checkpoints, "
                         "all zeropad -> groups ckpt_<tk>[_full])")
    ap.add_argument("--smoke", action="store_true",
                    help="DeLong asserts on one pair, then exit")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    out_csv = Path(args.out)

    global df_folds
    df = pd.read_csv(SPLIT_DATA_PATH, usecols=["label", "fold"])
    df_folds = df["fold"].to_numpy()
    y_full = df["label"].to_numpy()
    n = len(df)

    masks = load_padded_masks()

    # ── Load OOF vectors + fold AUCs ──
    pad_probs = {"ESM-C": {}, "ESM-IF": {}, "ProteinMPNN": {}}
    pad_fold_full = {"ESM-C": {}, "ESM-IF": {}, "ProteinMPNN": {}}
    ckpt_probs = {"ProteinMPNN": {}}       # --arm-key checkpoint
    ckpt_fold_full = {"ProteinMPNN": {}}
    baseline_probs = {}
    baseline_fold_auc = {}
    for pth in sorted(json_dir.glob("cv_results_xgb_*_flat_*.json")):
        with open(pth) as fh:
            res = json.load(fh)
        m_emb = EMB_JSON_RE.search(pth.name)
        m_base = BASELINE_JSON_RE.search(pth.name)
        if m_emb:
            key = m_emb.group(1)
            toolkit, padtype = PADTYPE_MAP[key]
            probs = reconstruct_oof(res, n)
            folds = [m["auc_roc"] for m in res["outer_val_metrics"]]
            if args.arm_key == "checkpoint" and toolkit == "ProteinMPNN":
                ckpt = CKPT_RE.match(key).group(1)
                ckpt_probs[toolkit][ckpt] = probs
                ckpt_fold_full[toolkit][ckpt] = folds
                print(f"  loaded {toolkit}/{ckpt:<14} {pth.name}")
            else:
                if args.arm_key == "checkpoint":
                    # cross-toolkit arms may coexist at several PCA sizes
                    # (e.g. esmif eos_bos_repeat @60% and @90% EV) — the
                    # padtype alone would collide, so tag with k
                    padtype = f"{padtype}@{m_emb.group(2)}"
                pad_probs[toolkit][padtype] = probs
                pad_fold_full[toolkit][padtype] = folds
                print(f"  loaded {toolkit}/{padtype:<14} {pth.name}")
        elif m_base:
            baseline_probs[m_base.group(1)] = reconstruct_oof(res, n)
            baseline_fold_auc[m_base.group(1)] = [
                m["auc_roc"] for m in res["outer_val_metrics"]]
            print(f"  loaded baseline/{m_base.group(1):<13} {pth.name}")

    # ── Build-time asserts (before any test family) ──
    def run_build_asserts():
        """HARD: row-identity + single-mask invariant. WARN: DeLong SE vs
        bootstrap SD ratio (checked per pair inside families)."""
        # row-identity: every loaded OOF vector is same length, all finite
        for tk in pad_probs:
            for padtype, probs in pad_probs[tk].items():
                assert len(probs) == n, (tk, padtype, len(probs), n)
                assert np.isfinite(probs).all(), (tk, padtype, "nonfinite")
        for tk in ckpt_probs:
            for ckpt, probs in ckpt_probs[tk].items():
                assert len(probs) == n, (tk, ckpt, len(probs), n)
                assert np.isfinite(probs).all(), (tk, ckpt, "nonfinite")
        for feat, probs in baseline_probs.items():
            assert len(probs) == n, (feat, len(probs), n)
            assert np.isfinite(probs).all(), (feat, "nonfinite")
        # single-mask invariant: each subset family uses exactly one
        # toolkit's mask (ESM-C=1443, ESM-IF=1442 differ on notfound rows;
        # cross-toolkit subset pairs would not be matched)
        assert int(masks["ESM-C"].sum()) == 1443
        assert int(masks["ESM-IF"].sum()) == 1442
        # MPNN mask: informative (partially padded) row count is not fixed
        # by construction here, only that the mask is non-empty and in range.
        assert 0 < int(masks["ProteinMPNN"].sum()) <= n, "ProteinMPNN mask"
        _n_emb = (sum(len(pad_probs[tk]) for tk in pad_probs)
                  + sum(len(ckpt_probs[tk]) for tk in ckpt_probs))
        print(f"  build asserts OK: {_n_emb} "
              f"emb + {len(baseline_probs)} baseline OOF vectors; "
              f"masks ESM-C=1443 / ESM-IF=1442 / "
              f"ProteinMPNN={int(masks['ProteinMPNN'].sum())} "
              f"(no cross-toolkit pairs)")

    run_build_asserts()

    def fold_subset_aucs(probs, padded):
        out = []
        for f in range(6):
            sel = (df_folds == f) & padded
            if y_full[sel].min() == y_full[sel].max():
                out.append(np.nan)
            else:
                out.append(roc_auc_score(y_full[sel], probs[sel]))
        return np.asarray(out)

    def holm_pair_block(pairs, compute):
        """compute(ma, mb) -> (row, p_boot, p_t, p_nb, p_delong)."""
        tmp, p_boot_l, p_t_l, p_nb_l, p_d_l = [], [], [], [], []
        for ma, mb in pairs:
            row, pb, pt, pnb, pd = compute(ma, mb)
            tmp.append(row)
            p_boot_l.append(pb)
            p_t_l.append(pt)
            p_nb_l.append(pnb)
            p_d_l.append(pd)
        adj_b = holm_adjust(p_boot_l)
        adj_t = holm_adjust(p_t_l)
        adj_nb = holm_adjust(p_nb_l)
        adj_d = holm_adjust(p_d_l)
        for row, ab, at, anb, ad in zip(tmp, adj_b, adj_t, adj_nb, adj_d):
            row["p_holm"] = ab
            row["p_ttest_holm"] = at
            row["p_ttest_nb_holm"] = anb
            row["p_delong_holm"] = ad
        return tmp

    def fold_t_nb(fa, fb):
        t, p_t = ttest_rel(fa, fb)
        t_nb = t / NB_FACTOR
        # df=5 kept per Nadeau-Bengio sensitivity convention
        p_nb = 2 * t_dist.sf(abs(t_nb), df=NB_DF)
        return t, p_t, t_nb, p_nb

    def delong_p(y_, pa, pb):
        """Primary row-level test: paired DeLong on pooled OOF.

        Returns (p, auc_a, auc_b, se, z). se==0 (identical predictions)
        -> p=1.0, z=0. AUC hard-asserted against roc_auc_score to 1e-12.
        """
        auc_a, auc_b, var_d = delong_test(y_, pa, pb)
        assert abs(auc_a - roc_auc_score(y_, pa)) < 1e-12
        assert abs(auc_b - roc_auc_score(y_, pb)) < 1e-12
        delta = auc_a - auc_b
        if not np.isfinite(var_d) or var_d <= 0:
            if delta == 0:
                return 1.0, auc_a, auc_b, 0.0, 0.0  # identical predictions
            return np.nan, auc_a, auc_b, np.nan, np.nan
        se = np.sqrt(var_d)
        z = delta / se
        return 2 * norm.sf(abs(z)), auc_a, auc_b, se, z

    def delong_fields(d_a, d_b, se, z, p_d):
        """CSV fields for the primary DeLong test (incl. 95% CI on Δ)."""
        delta = d_a - d_b
        return {
            "auc_a": d_a, "auc_b": d_b, "delta_auc": delta,
            "se_delta_delong": se, "z": z,
            "ci95_lo": delta - 1.96 * se if np.isfinite(se) else np.nan,
            "ci95_hi": delta + 1.96 * se if np.isfinite(se) else np.nan,
            "p_delong": p_d,
        }

    def maybe_bootstrap(y_, pa, pb, group, ma, mb):
        """Secondary row-level bootstrap (skippable via --skip-bootstrap).

        Returns (ci_lo, ci_hi, p_boot, sd_boot). Per-pair RNG so adding
        families never shifts other pairs' draws.
        """
        if args.skip_bootstrap:
            return np.nan, np.nan, np.nan, np.nan
        rng = per_pair_rng(args.seed, group, ma, mb)
        _, lo, hi, p, _, _, sd = paired_bootstrap_delta(
            y_, pa, pb, args.B, rng)
        return lo, hi, p, sd

    # ── Campaigns: {prefix, probs, folds, padded, pairs, subset} ──
    # padmode families per toolkit; checkpoint family adds ckpt_<tk>[_full]
    # with arms = train-noise checkpoints on the same toolkit mask; plus
    # cross-toolkit ckpt_vs_<tk2> pairs (checkpoints vs e.g. ESM-IF's
    # chosen pad mode) when non-MPNN emb arms are loaded alongside.
    campaigns = [
        {"prefix": f"padmode_{tk.replace('-', '').lower()}",
         "probs": pad_probs[tk], "folds": pad_fold_full[tk],
         "padded": masks[tk], "pairs": None, "subset": True}
        for tk in sorted(pad_probs) if len(pad_probs[tk]) >= 2]
    if args.arm_key == "checkpoint":
        campaigns += [
            {"prefix": f"ckpt_{tk.replace('-', '').lower()}",
             "probs": ckpt_probs[tk], "folds": ckpt_fold_full[tk],
             "padded": masks[tk], "pairs": None, "subset": True}
            for tk in sorted(ckpt_probs) if len(ckpt_probs[tk]) >= 2]
        # cross-toolkit: every checkpoint vs every loaded non-MPNN arm
        for tk in sorted(pad_probs):
            if not pad_probs[tk] or not any(ckpt_probs.values()):
                continue
            for ctk in sorted(ckpt_probs):
                if not ckpt_probs[ctk]:
                    continue
                same_mask = np.array_equal(masks[ctk], masks[tk])
                if not same_mask:
                    print(f"  WARN ckpt_vs_{tk}: padded masks differ "
                          f"({ctk} vs {tk}) — subset block skipped, "
                          f"full-data only")
                cross_pairs = [(c, p)
                               for c in sorted(ckpt_probs[ctk])
                               for p in sorted(pad_probs[tk])]
                campaigns.append({
                    "prefix": f"ckpt_vs_{tk.replace('-', '').lower()}",
                    "probs": {**ckpt_probs[ctk], **pad_probs[tk]},
                    "folds": {**ckpt_fold_full[ctk], **pad_fold_full[tk]},
                    "padded": masks[ctk], "pairs": cross_pairs,
                    "subset": same_mask})

    # ── Smoke: DeLong asserts on one pair, then exit ──
    if args.smoke:
        camp = next(c for c in campaigns if len(c["probs"]) >= 2)
        prefix, padded = camp["prefix"], camp["padded"]
        probs_by_arm = camp["probs"]
        ma, mb = (camp["pairs"] or sorted(
            (a for a in probs_by_arm), key=str))[:2]
        sub = probs_by_arm[ma][padded], probs_by_arm[mb][padded]
        y_sub = y_full[padded]
        p_d, a_d, b_d, se_d, z_d = delong_p(y_sub, *sub)
        a_r, b_r = roc_auc_score(y_sub, sub[0]), roc_auc_score(y_sub, sub[1])
        assert abs(a_d - a_r) < 1e-12, (a_d, a_r)
        assert abs(b_d - b_r) < 1e-12, (b_d, b_r)
        lo, hi, _, sd_boot = maybe_bootstrap(
            y_sub, sub[0], sub[1], "smoke", ma, mb)
        rel = abs(se_d - sd_boot) / sd_boot if sd_boot else float("nan")
        print(f"  smoke OK ({prefix}): AUC match 1e-12; "
              f"DeLong SE={se_d:.5f} z={z_d:.3f} p={p_d:.3g}; "
              f"bootstrap SD={sd_boot:.5f} rel_diff={rel:.3%}")
        # --skip-bootstrap leaves sd_boot nan -> no cross-check to assert on
        if not np.isnan(rel):
            assert rel < 0.05, f"DeLong SE vs bootstrap SD off by {rel:.1%}"
        return

    rows = []

    # ── 1. Pairwise families (padmode_<tk>, ckpt_<tk>, ckpt_vs_<tk2>) ──
    def run_pair_family(prefix, probs_by_arm, fold_full_by_arm, padded,
                        pairs=None, subset=True):
        """One family: pairwise DeLong (+bootstrap) on the padded subset
        and DeLong (+fold t/NB) on all rows; Holm within each block.
        pairs=None -> all combinations of the arm dict."""
        arms = sorted(probs_by_arm)
        pairs = pairs or list(itertools.combinations(arms, 2))
        fold_sub = {m: fold_subset_aucs(probs_by_arm[m], padded)
                    for m in arms}
        y_sub = y_full[padded]
        if subset:
            assert len(y_sub) == int(padded.sum())  # within-family matched

        # 1a. padded-subset: DeLong (primary) + bootstrap + fold t / NB
        if subset:
            def comp_sub(ma, mb):
                pa = probs_by_arm[ma][padded]
                pb = probs_by_arm[mb][padded]
                p_d, d_a, d_b, se_d, z_d = delong_p(y_sub, pa, pb)
                lo, hi, p, sd_b = maybe_bootstrap(
                    y_sub, pa, pb, prefix, ma, mb)
                t, p_t, t_nb, p_nb = fold_t_nb(fold_sub[ma], fold_sub[mb])
                if (np.isfinite(se_d) and sd_b
                        and abs(se_d - sd_b) / sd_b > 0.05):
                    print(f"  WARN {prefix} {ma}/{mb}: DeLong SE {se_d:.5f} vs "
                          f"bootstrap SD {sd_b:.5f} "
                          f"(ratio {se_d / sd_b:.3f})")
                row = {
                    "test_group": prefix,
                    "model_a": ma, "model_b": mb,
                    "ci_lo": lo, "ci_hi": hi,
                    "p_value": p, "n_rows": int(padded.sum()),
                    "t_stat": t, "p_ttest": p_t,
                    "t_stat_nb": t_nb, "p_ttest_nb": p_nb,
                }
                row.update(delong_fields(d_a, d_b, se_d, z_d, p_d))
                return row, p, p_t, p_nb, p_d
            rows.extend(holm_pair_block(pairs, comp_sub))

        # 1b. full-data: DeLong on pooled OOF + fold t / NB
        def comp_full(ma, mb):
            pa, pb = probs_by_arm[ma], probs_by_arm[mb]
            p_d, d_a, d_b, se_d, z_d = delong_p(y_full, pa, pb)
            fa = np.asarray(fold_full_by_arm[ma])
            fb = np.asarray(fold_full_by_arm[mb])
            t, p_t, t_nb, p_nb = fold_t_nb(fa, fb)
            row = {
                "test_group": f"{prefix}_full",
                "model_a": ma, "model_b": mb,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "n_rows": n,
                "t_stat": t, "p_ttest": p_t,
                "t_stat_nb": t_nb, "p_ttest_nb": p_nb,
            }
            row.update(delong_fields(d_a, d_b, se_d, z_d, p_d))
            return row, np.nan, p_t, p_nb, p_d
        rows.extend(holm_pair_block(pairs, comp_full))

    for camp in campaigns:
        run_pair_family(camp["prefix"], camp["probs"], camp["folds"],
                        camp["padded"], pairs=camp["pairs"],
                        subset=camp["subset"])

    # ── 2. Baseline pairs, full-data ──
    base_modes = [m for m in ("sparse", "blosum", "handcrafted_blosum",
                              "handcrafted_sparse", "handcrafted")
                  if m in baseline_probs]
    if len(base_modes) >= 2:
        pairs = list(itertools.combinations(base_modes, 2))

        def comp_base_full(ma, mb):
            p_d, d_a, d_b, se_d, z_d = delong_p(
                y_full, baseline_probs[ma], baseline_probs[mb])
            fa = np.asarray(baseline_fold_auc[ma])
            fb = np.asarray(baseline_fold_auc[mb])
            t, p_t, t_nb, p_nb = fold_t_nb(fa, fb)
            if {ma, mb} == {"sparse", "blosum"}:
                lo, hi, p, sd_b = maybe_bootstrap(
                    y_full, baseline_probs[ma], baseline_probs[mb],
                    "baseline_full", ma, mb)
            else:
                lo, hi, p, sd_b = np.nan, np.nan, np.nan, np.nan
            row = {
                "test_group": "baseline_full",
                "model_a": ma, "model_b": mb,
                "ci_lo": lo, "ci_hi": hi,
                "p_value": p, "n_rows": n,
                "t_stat": t, "p_ttest": p_t,
                "t_stat_nb": t_nb, "p_ttest_nb": p_nb,
            }
            row.update(delong_fields(d_a, d_b, se_d, z_d, p_d))
            return row, p, p_t, p_nb, p_d
        rows.extend(holm_pair_block(pairs, comp_base_full))
    else:
        print(f"  WARNING: baseline pair incomplete: {sorted(baseline_probs)}")

    # ── 3. Baseline pairs on ESM-C padded subset ──
    if len(base_modes) >= 2:
        padded_c = masks["ESM-C"]
        y_sub = y_full[padded_c]
        fold_sub = {m: fold_subset_aucs(baseline_probs[m], padded_c)
                    for m in base_modes}
        pairs = list(itertools.combinations(base_modes, 2))

        def comp_base_sub(ma, mb):
            pa = baseline_probs[ma][padded_c]
            pb = baseline_probs[mb][padded_c]
            p_d, d_a, d_b, se_d, z_d = delong_p(y_sub, pa, pb)
            lo, hi, p, sd_b = maybe_bootstrap(
                y_sub, pa, pb, "baseline_subset_esmc", ma, mb)
            t, p_t, t_nb, p_nb = fold_t_nb(fold_sub[ma], fold_sub[mb])
            if (np.isfinite(se_d) and sd_b
                    and abs(se_d - sd_b) / sd_b > 0.05):
                print(f"  WARN baseline {ma}/{mb}: DeLong SE {se_d:.5f} vs "
                      f"bootstrap SD {sd_b:.5f} "
                      f"(ratio {se_d / sd_b:.3f})")
            row = {
                "test_group": "baseline_subset_esmc",
                "model_a": ma, "model_b": mb,
                "ci_lo": lo, "ci_hi": hi,
                "p_value": p, "n_rows": int(padded_c.sum()),
                "t_stat": t, "p_ttest": p_t,
                "t_stat_nb": t_nb, "p_ttest_nb": p_nb,
            }
            row.update(delong_fields(d_a, d_b, se_d, z_d, p_d))
            return row, p, p_t, p_nb, p_d
        rows.extend(holm_pair_block(pairs, comp_base_sub))

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # CI extent for Methods wording (padmode subset families; DeLong CIs
    # are primary, bootstrap CIs cross-check)
    for grp in sorted(g for g in out["test_group"].unique()
                      if (g.startswith("padmode_") or g.startswith("ckpt_"))
                      and not g.endswith("_full")):
        sub = out[out["test_group"] == grp]
        d_bound = max(sub["ci95_lo"].abs().max(), sub["ci95_hi"].abs().max())
        b_bound = max(sub["ci_lo"].abs().max(), sub["ci_hi"].abs().max())
        print(f"  {grp}: DeLong 95% CIs within +/-{d_bound:.3f}; "
              f"bootstrap 95% CIs within +/-{b_bound:.3f}")

    pd.set_option("display.width", 250)
    show = out[["test_group", "model_a", "model_b", "delta_auc",
                "ci95_lo", "ci95_hi", "se_delta_delong", "z",
                "p_delong", "p_delong_holm",
                "t_stat", "p_ttest", "p_ttest_nb", "p_ttest_nb_holm",
                "p_value", "p_holm", "n_rows"]].copy()
    for c in ["delta_auc", "ci95_lo", "ci95_hi", "se_delta_delong", "z",
              "t_stat"]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.4f}")
    for c in ["p_delong", "p_delong_holm", "p_ttest", "p_ttest_nb",
              "p_ttest_nb_holm", "p_value", "p_holm"]:
        show[c] = show[c].map(
            lambda v: "" if pd.isna(v) else (f"{v:.2e}" if v < 0.01 else f"{v:.4f}"))
    print(f"\n{show.to_string(index=False)}")
    print(f"\n  Saved: {out_csv}")


if __name__ == "__main__":
    main()
