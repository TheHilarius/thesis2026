#!/usr/bin/env python3
"""38_cnn_2layer_chan_performance.py — 2-layer CNN channel sweep at fixed k2 on flank5-19mer.

Run once per k2 (k19 winner folder, k3 folder): bars left to right C8, C16,
C32, C64, C128 (k1=17 fixed, k2 fixed per run, 2 hidden layers), plus ONE
leading grey reference bar: 1-layer k17 C32 (current best, pooled 0.6847).
Bar height = pooled OOF AUC over the 6 outer folds (roc_auc_score on
concatenated outer-fold predictions — the same vector DeLong tests);
error bars = std of the 6 outer-fold AUCs.

Statistical protocol (pooled OOF DeLong, same as 37_cnn_2layer_k2_performance.py
/ 36_cnn_channel_performance.py / 35_cnn_kernel_performance.py):
  * DeLong two-sided z-test on the pooled OOF probability vectors; Holm
    correction within the vs-anchor family; brackets drawn ONLY for
    significant pairs.
  * Full all-pairs DeLong table written for later digging (10 pairs over 5 channels).
  * The 1-layer reference bar is BARS ONLY (different depth = unpaired
    comparison, excluded from both DeLong families).

Inputs: cv_results_*.json files from one channel-sweep run dir. Group key is
config.hyperparameters.channels (= channels_override), filtered to
config.hyperparameters.kernel2 == --k2; channels-aware outputs also record
kernel1/kernel2. Filename fallbacks: _c{N} for channels, _k2{N} (else last
_k{N}) for the k2 filter. Latest timestamp wins per channel.

Outputs (default --runtag cnn_2layer_chan_k<K>_curve):
    runs/cnn_2layer_chan_k<K>_curve/plots/cnn_2layer_chan_k<K>_auc.png (+ .csv bar values)
    runs/cnn_2layer_chan_k<K>_curve/plots/cnn_2layer_chan_k<K>_overfit_curves.png (per-channel train-loss / val-AUC curves)
    runs/cnn_2layer_chan_k<K>_curve/plots/cnn_2layer_chan_k<K>_val_vs_test.png (best vs final-epoch val AUC: post-peak degradation)
    runs/cnn_2layer_chan_k<K>_curve/results/cnn_2layer_chan_k<K>_summary.csv (kernel1, kernel2, channels, n_params)
    runs/cnn_2layer_chan_k<K>_curve/results/cnn_2layer_chan_k<K>_delong_stats.tsv  (vs-anchor, Holm)
    runs/cnn_2layer_chan_k<K>_curve/results/cnn_2layer_chan_k<K>_delong_allpairs.tsv (all pairs)
    runs/cnn_2layer_chan_k<K>_curve/results/cnn_2layer_chan_k<K>_overfit.tsv (per-fold epochs_run, best/final val AUC)

Note on overfitting: this pipeline runs flat 6-fold CV (no nested split —
`05_cnn_modelling.py` evaluates the early-stopped model on the same held-out
bucket used for early stopping), so inner-val best and outer-test AUC are the
same numbers by construction. The honest overfit signals are (1) train-loss
vs val-AUC trajectories per epoch and (2) best-vs-final-epoch val-AUC
degradation (post-peak overfitting inside the patience window).

Usage:
    python src/investigation/38_cnn_2layer_chan_performance.py --k2 19 --anchor 32
    python src/investigation/38_cnn_2layer_chan_performance.py --k2 k19 --anchor c32 \
        --run-dirs runs/cnn_2layer_k17_k19_channels_20261111/models/cnn --runtag cnn_2layer_chan_k19_curve

Stats code copied verbatim from 37_cnn_2layer_k2_performance.py (which copied it
from 36_cnn_channel_performance.py — which inlined it from
35_cnn_kernel_performance.py / 18_encodings_sweep_delong.py /
15_padmode_significance.py, whose import chain is broken upstream).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ALPHA = 0.05
C_ESMC = "#EE6677"  # ESM-C padtoken red (thesis convention)

K2_FILE_RE = re.compile(r"_k2(\d+)")
K_FILE_RE = re.compile(r"_k(\d+)")
CHANNEL_FILE_RE = re.compile(r"_c(\d+)_")
DEFAULT_CHANNELS = [8, 16, 32, 64, 128]


# ── stats (inlined from 35_cnn_kernel_performance — proven code) ──────────
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


# ── loading ───────────────────────────────────────────────────────────────
def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def channel_of_record(r, path):
    """Authoritative C: config.hyperparameters.channels (= channels_override);
    fallback: _c{N}_ tag in filename. Returns int or None."""
    cfg = r.get("config") or {}
    hyp = cfg.get("hyperparameters") or {}
    for key in ("channels", "channels_override"):
        c = _int_or_none(hyp.get(key, cfg.get(key)))
        if c is not None:
            return c
    m = CHANNEL_FILE_RE.search(str(path))
    if m:
        return int(m.group(1))
    return None


def k2_of_record(r, path):
    """Authoritative k2: config.hyperparameters.kernel2; filename fallback:
    _k2{N}, else last _k{N} in the model-key portion (k17_k7 -> 7).
    Returns int or None."""
    cfg = r.get("config") or {}
    hyp = cfg.get("hyperparameters") or {}
    for key in ("kernel2",):
        v = hyp.get(key, cfg.get(key))
        k = _int_or_none(v)
        if k is not None:
            return k
    m = K2_FILE_RE.search(str(path))
    if m:
        return int(m.group(1))
    ms = K_FILE_RE.findall(str(path))
    if ms:
        return int(ms[-1])
    return None


def kernel1_of_record(r):
    """Layer-1 kernel for the summary table; default 17 (this sweep fixes k1)."""
    cfg = r.get("config") or {}
    hyp = cfg.get("hyperparameters") or {}
    for key in ("kernel1", "kernel"):
        k = _int_or_none(hyp.get(key, cfg.get(key)))
        if k is not None:
            return k
    return 17


def channels_of_record(r):
    """Channel count for the summary table (= channels_override when set)."""
    cfg = r.get("config") or {}
    hyp = cfg.get("hyperparameters") or {}
    for key in ("channels", "channels_override"):
        c = _int_or_none(hyp.get(key, cfg.get(key)))
        if c is not None:
            return c
    return None


def normalize_anchor(anchor):
    """Accept '32', 'c32', 'C32' -> channel int."""
    s = str(anchor).strip()
    m = re.fullmatch(r"[cC]?(\d+)", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse --anchor {anchor!r}: use 32 or c32")


def normalize_k2(k2):
    """Accept '19', 'k19', 'K19', 'k2=19' -> k2 int."""
    s = str(k2).strip()
    m = re.fullmatch(r"(?:[kK](?:2\s*[=_]?)?)?(\d+)", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse --k2 {k2!r}: use 19, k19 or k2=19")


def default_run_dirs(k2):
    """Auto-find channel-sweep dirs for one k2; fall back to bare models/cnn."""
    out = sorted((PROJECT_ROOT / "runs").glob(f"cnn_2layer_k17_k{k2}_channels_*/models/cnn"))
    if out:
        return [str(p) for p in out]
    out = sorted((PROJECT_ROOT / "runs").glob(f"cnn_2layer_k17_k{k2}_channels_*"))
    return [str(p) for p in out]


def default_ref_json():
    """Auto-find the 1-layer k17 C32 reference JSON (current best, pooled 0.6847).

    Prefers the channel-sweep c32 file, else the latest k17 file from the
    kernel curve. Returns a Path or None."""
    cands = sorted((PROJECT_ROOT / "runs").glob(
        "cnn_channels_k17_20261110/models/cnn/*c32*.json"))
    if cands:
        return cands[-1]
    cands = sorted((PROJECT_ROOT / "runs").glob(
        "cnn_kernel3_19_flank5_20261109/models/cnn/*k17*.json"))
    if cands:
        return cands[-1]
    return None


def find_jsons(run_dirs):
    files = []
    for d in run_dirs:
        base = Path(d) if Path(d).is_absolute() else PROJECT_ROOT / d
        for pat in ("models/cnn/cv_results_*.json", "cv_results_*.json"):
            files += sorted(base.glob(pat))
    # dedupe, keep order
    seen, uniq = set(), []
    for f in files:
        if str(f) not in seen:
            seen.add(str(f))
            uniq.append(f)
    return uniq


def _pool_vectors(r):
    """Shared pooled-OOF loader: y, probs, fold_aucs, curves from one JSON."""
    ovp = r.get("outer_val_predictions", {})
    folds = sorted(ovp.keys(), key=lambda x: int(x)) if isinstance(ovp, dict) \
        else list(range(len(ovp)))
    get = (lambda b: ovp[str(b)] if isinstance(ovp, dict) else ovp[b])
    y_parts, p_parts, fold_aucs = [], [], []
    ovm = r.get("outer_val_metrics", [])
    for b in folds:
        e = get(b)
        y_parts.append(np.asarray(e["y_true"], dtype=int))
        p_parts.append(np.asarray(e["y_prob"], dtype=float))
    if isinstance(ovm, list) and len(ovm) == len(folds):
        fold_aucs = [float(m["auc_roc"]) for m in ovm]
    else:  # fallback: per-fold AUC from pooled vectors
        from sklearn.metrics import roc_auc_score as _auc
        fold_aucs = [float(_auc(y_parts[i], p_parts[i])) for i in range(len(folds))]
    # overfit diagnostics: per-fold train-loss / val-AUC curves (may be absent in old JSONs)
    curves = None
    lc = r.get("learning_curves") or {}
    if isinstance(lc, dict) and len(lc) == len(folds):
        try:
            curves = {}
            for b in folds:
                e = lc[str(b)] if str(b) in lc else lc[b]
                curves[int(b)] = {
                    "train_loss": [float(v) for v in e["train_loss"]],
                    "val_auc": [float(v) for v in e["val_auc"]],
                    "best_val_auc": float(e["best_val_auc"]),
                    "epochs_run": int(e["epochs_run"]),
                }
        except (KeyError, TypeError, ValueError):
            curves = None
    return (np.concatenate(y_parts), np.concatenate(p_parts),
            fold_aucs, curves)


def load_chan_json(path):
    """One channel-sweep JSON -> record keyed by channels (None if no C found)."""
    with open(path) as fh:
        r = json.load(fh)
    c = channel_of_record(r, path)
    if c is None:
        return None
    y, probs, fold_aucs, curves = _pool_vectors(r)
    cfg = r.get("config") or {}
    hyp = cfg.get("hyperparameters") or {}
    n_params = cfg.get("n_params", hyp.get("n_params", ""))
    return {
        "channels": c,
        "k2": k2_of_record(r, path),
        "kernel1": kernel1_of_record(r),
        "model_key": r.get("model_key", ""),
        "n_params": n_params,
        "y": y,
        "probs": probs,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "timestamp": str(r.get("timestamp", "")),
        "file": str(path),
        "curves": curves,
    }


def load_ref_json(path):
    """One 1-layer k17 C32 reference JSON -> bars-only record (no k2 key)."""
    with open(path) as fh:
        r = json.load(fh)
    y, probs, fold_aucs, curves = _pool_vectors(r)
    cfg = r.get("config") or {}
    hyp = cfg.get("hyperparameters") or {}
    n_params = cfg.get("n_params", hyp.get("n_params", ""))
    return {
        "k2": "ref",
        "kernel1": None,
        "kernel2": None,
        "channels": channels_of_record(r),
        "model_key": r.get("model_key", ""),
        "n_params": n_params,
        "y": y,
        "probs": probs,
        "fold_aucs": fold_aucs,
        "fold_std": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "timestamp": str(r.get("timestamp", "")),
        "file": str(path),
        "curves": curves,
    }


# ── plotting helpers (padmode style, from 18_encodings_sweep_delong) ──────
def _padmode_brackets(ax, pairs, values, fontsize=7.5, stars_only=False):
    """pairs: [(x_a, x_b, p, p_adj), ...]. Widest span on top; text shows
    raw p + Holm p_adj, stars from p_adj. Fits under y=0.97 on the 0-1 axis."""
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


def main():
    ap = argparse.ArgumentParser(description="2-layer CNN channel sweep at fixed k2: pooled-OOF AUC bars + DeLong vs anchor")
    ap.add_argument("--k2", required=True,
                    help="REQUIRED layer-2 kernel to analyse, e.g. 19, k19, k2=19. Run once per k2 folder.")
    ap.add_argument("--anchor", required=True,
                    help="REQUIRED anchor channels, e.g. 32, c32. No default — decided per run.")
    ap.add_argument("--run-dirs", nargs="*", default=None,
                    help="run dirs holding cv_results JSONs (default: auto-find runs/cnn_2layer_k17_k<K>_channels_*/models/cnn)")
    ap.add_argument("--runtag", default=None,
                    help="output folder under runs/ (default: cnn_2layer_chan_k<K>_curve)")
    ap.add_argument("--channels", nargs="*", type=int, default=list(DEFAULT_CHANNELS),
                    help="channels to plot in order (default: 8 16 32 64 128)")
    ap.add_argument("--ref-json", default=None,
                    help="1-layer k17 C32 reference JSON (default: auto-find channel/kernel runs)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--data-label", default="flank5-19mer k17xX",
                    help="data label for title/caption (default: flank5-19mer k17xX)")
    args = ap.parse_args()

    want_k2 = normalize_k2(args.k2)
    anchor_c = normalize_anchor(args.anchor)
    runtag = args.runtag or f"cnn_2layer_chan_k{want_k2}_curve"
    prefix = f"cnn_2layer_chan_k{want_k2}"
    run_dirs = args.run_dirs if args.run_dirs else default_run_dirs(want_k2)
    if not run_dirs:
        raise SystemExit("No run dirs found: pass --run-dirs explicitly "
                         f"(channel-sweep runs for k2={want_k2} not present locally yet?)")
    files = find_jsons(run_dirs)
    print(f"k2={want_k2} run dirs: {run_dirs}\nFound {len(files)} cv_results JSONs")
    if not files:
        raise SystemExit("No cv_results_*.json found under given run dirs.")

    # load + dedupe per channel, latest timestamp wins; skip other-k2 files
    by_chan = {}
    for f in files:
        try:
            rec = load_chan_json(f)
        except (OSError, ValueError, KeyError) as e:
            print(f"  WARN: unreadable {f} ({e}) — skipped")
            continue
        if rec is None:
            print(f"  WARN: no channel tag in {f} — skipped")
            continue
        if rec["k2"] != want_k2:
            print(f"  WARN: k2={rec['k2']} != --k2 {want_k2} in {f} — skipped")
            continue
        prev = by_chan.get(rec["channels"])
        if prev is None or rec["timestamp"] >= prev["timestamp"]:
            by_chan[rec["channels"]] = rec

    want = list(args.channels)
    missing = [c for c in want if c not in by_chan]
    present = [c for c in want if c in by_chan]
    if missing:
        print(f"  WARN: missing channels (plotted without them): {missing}")
    if not present:
        raise SystemExit(f"None of the requested channels {want} found. "
                         f"Available: {sorted(by_chan)}")
    if anchor_c not in by_chan:
        raise SystemExit(f"--anchor c{anchor_c} not among loaded channels "
                         f"{sorted(by_chan)} — check anchor spelling / run completeness.")

    # row-identity check: all channels must share the same pooled y_true
    ref_y = by_chan[present[0]]["y"]
    for c in present[1:]:
        yc = by_chan[c]["y"]
        if len(yc) != len(ref_y) or not (yc == ref_y).all():
            raise SystemExit(
                f"Row-identity FAILED: channel c{c} y_true differs from c{present[0]} "
                "(folds/labels differ — DeLong on pooled OOF invalid)")
    print(f"  row-identity OK: {len(ref_y)} pooled rows shared across {len(present)} channels")
    y = ref_y

    # ONE 1-layer reference bar (bars only, excluded from DeLong families:
    # different depth = unpaired comparison)
    ref = None
    if args.ref_json:
        rp = Path(args.ref_json) if Path(args.ref_json).is_absolute() else PROJECT_ROOT / args.ref_json
        if not rp.exists():
            raise SystemExit(f"--ref-json not found: {rp}")
        try:
            ref = load_ref_json(rp)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f"--ref-json unreadable {rp} ({e})")
    else:
        auto = default_ref_json()
        if auto is not None:
            try:
                ref = load_ref_json(auto)
            except (OSError, ValueError, KeyError) as e:
                print(f"  WARN: auto-found ref {auto} unreadable ({e}) — plotting channel bars only")
                ref = None
        else:
            print("  WARN: no 1-layer k17 C32 reference JSON auto-found — plotting channel bars only")
    if ref is not None:
        if len(ref["y"]) != len(y) or not (ref["y"] == y).all():
            print("  WARN: reference y_true differs from channel rows "
                  "(different depth = unpaired — bars only, kept)")

    # pooled OOF AUC per channel (+ assert vs DeLong AUC)
    summary = []
    for c in present:
        rec = by_chan[c]
        pooled = float(roc_auc_score(y, rec["probs"]))
        auc_d, _, _ = delong_test(y, rec["probs"], rec["probs"])[:3]
        assert abs(pooled - auc_d) < 1e-9, f"c{c}: sklearn AUC != DeLong AUC"
        rec["pooled_auc"] = pooled
        summary.append(rec)
        print(f"  c{c:<4} {rec['model_key']:<22} pooled AUC={pooled:.4f} "
              f"fold-std={rec['fold_std']:.4f} n_params={rec['n_params']}  {Path(rec['file']).name}")
    if ref is not None:
        pooled = float(roc_auc_score(ref["y"], ref["probs"]))
        ref["pooled_auc"] = pooled
        print(f"  ref    {ref['model_key']:<22} pooled AUC={pooled:.4f} "
              f"fold-std={ref['fold_std']:.4f} n_params={ref['n_params']}  "
              f"{Path(ref['file']).name} (bars only)")

    # ── DeLong family: each channel vs anchor (delta = channel − anchor; ref excluded) ──
    pa = by_chan[anchor_c]["probs"]
    fam = []
    for c in present:
        if c == anchor_c:
            continue
        auc_c, auc_a, d, se, z, p = delong_full(y, by_chan[c]["probs"], pa)
        fam.append(dict(channel=c, anchor=anchor_c, name_a=f"cnn_2layer_k17_k{want_k2}_c{c}",
                        name_b=f"cnn_2layer_k17_k{want_k2}_c{anchor_c}",
                        auc_a=auc_c, auc_b=auc_a, delta=d, se=se, z=z, p=p))
    holm_family(fam)
    nsig = sum(1 for pr in fam if pr["p_adj"] < args.alpha)
    print(f"\nvs-anchor family (k2={want_k2} anchor c{anchor_c}): {len(fam)} pairs, {nsig} significant (Holm, alpha={args.alpha})")
    for pr in sorted(fam, key=lambda r: r["channel"]):
        print(f"  c{pr['channel']:>3} vs c{anchor_c}: delta={pr['delta']:+.4f} "
              f"se={pr['se']:.5f} p={pr['p']:.3g} p_adj={pr['p_adj']:.3g} ({stars(pr['p_adj'])})")

    # ── all-pairs DeLong (delta = larger-C − smaller-C; ref excluded) ──
    allpairs = []
    for i, ca in enumerate(sorted(present)):
        for cb in sorted(present)[i + 1:]:
            auc_a, auc_b, _d, se, _z, p = delong_full(y, by_chan[ca]["probs"], by_chan[cb]["probs"])
            delta = auc_b - auc_a  # reported as larger-C minus smaller-C
            z = float(delta / se) if se > 0 else 0.0
            allpairs.append(dict(channel_a=ca, channel_b=cb,
                                 name_a=f"cnn_2layer_k17_k{want_k2}_c{ca}",
                                 name_b=f"cnn_2layer_k17_k{want_k2}_c{cb}",
                                 auc_a=auc_a, auc_b=auc_b, delta=delta,
                                 se=se, z=z, p=p))
    holm_family(allpairs)
    print(f"all-pairs family: {len(allpairs)} pairs, "
          f"{sum(1 for pr in allpairs if pr['p_adj'] < args.alpha)} significant (Holm)")

    # ── figure (reference bar first, then channels in --channels order) ──
    run_dir = PROJECT_ROOT / "runs" / runtag
    plots_dir = run_dir / "plots"
    results_dir = run_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cols = ([ref] if ref is not None else []) + [by_chan[c] for c in present]
    off = 1 if ref is not None else 0
    xs = list(range(len(cols)))
    c_to_x = {c: i + off for i, c in enumerate(present)}
    vals = [c["pooled_auc"] for c in cols]
    errs = [c["fold_std"] for c in cols]
    if ref is not None:
        labels = ["1-layer\nk17 C32\n(ref)"] + [
            f"c{c}\n(anchor)" if c == anchor_c else f"c{c}" for c in present]
    else:
        labels = [f"c{c}\n(anchor)" if c == anchor_c else f"c{c}" for c in present]

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.1), 5.8))
    for x, c in zip(xs, cols):
        v, e = c["pooled_auc"], c["fold_std"]
        if c is ref:
            ax.bar(x, v, yerr=e, capsize=3, width=0.62, color="#B0B0B0",
                   hatch="///", edgecolor="black", linewidth=0.8)
        else:
            ax.bar(x, v, yerr=e, capsize=3, width=0.62, color=C_ESMC,
                   edgecolor="black", linewidth=2.0 if c["channels"] == anchor_c else 0.6)
    for x, v, e in zip(xs, vals, errs):
        ax.text(x, v + e + 0.006, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    pairs = []
    for pr in fam:
        if pr["p_adj"] >= args.alpha:
            continue
        xa, xb = sorted((c_to_x[pr["channel"]], c_to_x[anchor_c]))
        pairs.append((xa, xb, pr["p"], pr["p_adj"]))
    _padmode_brackets(ax, pairs, vals)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC-ROC (pooled OOF; error bars = outer-fold std)")
    ax.set_ylim(0, 1.0)  # user spec: y-axis starts at 0
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"2-layer CNN channel sweep at k1=17, k2={want_k2} on {args.data_label} windows (anchor c{anchor_c})",
                 fontsize=12)
    fig.tight_layout()
    png = plots_dir / f"{prefix}_auc.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")

    # ── CSV/TSV outputs ──
    bar_csv = plots_dir / f"{prefix}_auc.csv"
    with open(bar_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k2", "channels", "model_key", "kernel1",
                                           "pooled_auc", "fold_std", "n_params", "anchor", "file"])
        w.writeheader()
        if ref is not None:
            w.writerow({"k2": want_k2, "channels": "ref", "model_key": ref["model_key"], "kernel1": "",
                        "pooled_auc": f"{ref['pooled_auc']:.6f}",
                        "fold_std": f"{ref['fold_std']:.6f}",
                        "n_params": ref["n_params"],
                        "anchor": False, "file": Path(ref["file"]).name})
        for c in present:
            rec = by_chan[c]
            w.writerow({"k2": want_k2, "channels": c, "model_key": rec["model_key"],
                        "kernel1": rec["kernel1"],
                        "pooled_auc": f"{rec['pooled_auc']:.6f}",
                        "fold_std": f"{rec['fold_std']:.6f}",
                        "n_params": rec["n_params"],
                        "anchor": str(c == anchor_c), "file": Path(rec["file"]).name})
    print(f"  wrote {bar_csv}")

    summ_csv = results_dir / f"{prefix}_summary.csv"
    with open(summ_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["k2", "channels", "model_key", "kernel1",
                                           "pooled_auc", "fold_std",
                                           "fold_aucs", "n_rows", "n_params", "file"])
        w.writeheader()
        rows = ([("ref", ref)] if ref is not None else []) + [(c, by_chan[c]) for c in present]
        for c, rec in rows:
            w.writerow({"k2": want_k2, "channels": rec["channels"] if c != "ref" else "ref",
                        "model_key": rec["model_key"],
                        "kernel1": rec["kernel1"] if c != "ref" else "",
                        "pooled_auc": f"{rec['pooled_auc']:.6f}",
                        "fold_std": f"{rec['fold_std']:.6f}",
                        "fold_aucs": ";".join(f"{a:.6f}" for a in rec["fold_aucs"]),
                        "n_rows": len(y), "n_params": rec["n_params"],
                        "file": Path(rec["file"]).name})
    print(f"  wrote {summ_csv}")

    stats_tsv = results_dir / f"{prefix}_delong_stats.tsv"
    with open(stats_tsv, "w", newline="") as fh:
        fields = ["family", "k2", "channel", "anchor", "name_a", "name_b",
                  "auc_a", "auc_b", "delta", "se", "z", "p", "p_adj", "significant"]
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for pr in sorted(fam, key=lambda r: r["channel"]):
            w.writerow({"family": "vs_anchor", "k2": want_k2, "channel": pr["channel"], "anchor": pr["anchor"],
                        "name_a": pr["name_a"], "name_b": pr["name_b"],
                        "auc_a": f"{pr['auc_a']:.6f}", "auc_b": f"{pr['auc_b']:.6f}",
                        "delta": f"{pr['delta']:+.6f}", "se": f"{pr['se']:.6f}",
                        "z": f"{pr['z']:.4f}", "p": f"{pr['p']:.3e}",
                        "p_adj": f"{pr['p_adj']:.3e}",
                        "significant": str(pr["p_adj"] < args.alpha)})
    print(f"  wrote {stats_tsv}")

    ap_tsv = results_dir / f"{prefix}_delong_allpairs.tsv"
    with open(ap_tsv, "w", newline="") as fh:
        fields = ["family", "k2", "channel_a", "channel_b", "name_a", "name_b",
                  "auc_a", "auc_b", "delta", "se", "z", "p", "p_adj", "significant"]
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for pr in sorted(allpairs, key=lambda r: (r["channel_a"], r["channel_b"])):
            w.writerow({"family": "allpairs", "k2": want_k2, "channel_a": pr["channel_a"], "channel_b": pr["channel_b"],
                        "name_a": pr["name_a"], "name_b": pr["name_b"],
                        "auc_a": f"{pr['auc_a']:.6f}", "auc_b": f"{pr['auc_b']:.6f}",
                        "delta": f"{pr['delta']:+.6f}", "se": f"{pr['se']:.6f}",
                        "z": f"{pr['z']:.4f}", "p": f"{pr['p']:.3e}",
                        "p_adj": f"{pr['p_adj']:.3e}",
                        "significant": str(pr["p_adj"] < args.alpha)})
    print(f"  wrote {ap_tsv}")

    # ── overfitting diagnostics (standard): train vs val curves + val-vs-test gap ──
    of_chans = [c for c in present if by_chan[c]["curves"] is not None]
    if len(of_chans) < len(present):
        print(f"  WARN: no learning_curves for channels "
              f"{sorted(set(present) - set(of_chans))} — overfit plots cover {of_chans}")
    if of_chans:
        # mean curve per channel: epoch i averaged over folds that ran that long
        def mean_curve(key):
            out = {}
            for c in of_chans:
                per_fold = [ch[key] for ch in by_chan[c]["curves"].values()]
                n_ep = max(len(v) for v in per_fold)
                means = [float(np.mean([v[i] for v in per_fold if len(v) > i]))
                         for i in range(n_ep)]
                out[c] = means
            return out
        mean_train = mean_curve("train_loss")
        mean_val = mean_curve("val_auc")

        n = len(of_chans)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig2, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                                  sharex=False, squeeze=False)
        for ax, c in zip(axes.flat, of_chans):
            ep = np.arange(1, len(mean_train[c]) + 1)
            l1, = ax.plot(ep, mean_train[c], color="#4477AA", lw=1.5, label="train loss")
            ax.set_ylabel("train loss", fontsize=8, color="#4477AA")
            ax.tick_params(axis="y", labelcolor="#4477AA", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax2 = ax.twinx()
            l2, = ax2.plot(ep, mean_val[c], color=C_ESMC, lw=1.5, label="val AUC")
            ax2.set_ylim(0.5, 1.0)
            ax2.tick_params(labelsize=7)
            ax.legend([l1, l2], ["train loss", "val AUC"], fontsize=7, loc="best")
            ax.set_title(f"c{c}", fontsize=10)
            ax.set_xlabel("epoch", fontsize=8)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        fig2.suptitle(f"Train loss vs inner-val AUC per epoch (mean over folds, {args.data_label}, k2={want_k2})",
                      fontsize=11)
        # single shared right-axis label + legend via last visible panel
        fig2.text(0.995, 0.5, "val AUC", va="center", rotation="vertical", fontsize=9)
        fig2.tight_layout(rect=[0, 0, 0.96, 0.95])
        of_png = plots_dir / f"{prefix}_overfit_curves.png"
        fig2.savefig(of_png, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  wrote {of_png}")

        # per-fold best-vs-final val AUC (post-peak degradation) + epochs_run.
        # NOTE: flat 6-fold CV here (no nested split): the early-stopped model
        # is evaluated on the same bucket used for early stopping, so
        # inner-val best == outer-test by construction — no val-vs-test gap
        # to plot. Degradation = best_val_auc − final-epoch val_auc instead.
        gap_rows = []
        for c in of_chans:
            rec = by_chan[c]
            for b in sorted(rec["curves"]):
                ch = rec["curves"][b]
                final_auc = ch["val_auc"][-1]
                gap_rows.append({"channel": c, "fold": b,
                                 "epochs_run": ch["epochs_run"],
                                 "best_val_auc": ch["best_val_auc"],
                                 "final_val_auc": final_auc,
                                 "degradation": ch["best_val_auc"] - final_auc})
        mean_deg = {c: float(np.mean([r["degradation"] for r in gap_rows if r["channel"] == c]))
                    for c in of_chans}
        mean_best = {c: float(np.mean([r["best_val_auc"] for r in gap_rows if r["channel"] == c]))
                     for c in of_chans}
        mean_final = {c: float(np.mean([r["final_val_auc"] for r in gap_rows if r["channel"] == c]))
                      for c in of_chans}
        mean_ep = {c: float(np.mean([r["epochs_run"] for r in gap_rows if r["channel"] == c]))
                   for c in of_chans}
        for c in of_chans:
            print(f"  c{c:<4} val best={mean_best[c]:.4f} "
                  f"final-ep={mean_final[c]:.4f} degradation={mean_deg[c]:+.4f} "
                  f"epochs_run~{mean_ep[c]:.1f}")

        xs2 = list(range(len(of_chans)))
        v_best = [mean_best[c] for c in of_chans]
        v_final = [mean_final[c] for c in of_chans]
        fig3, ax3 = plt.subplots(figsize=(max(10, len(of_chans) * 1.1), 5.8))
        w = 0.36
        ax3.bar([x - w / 2 for x in xs2], v_best, width=w, color="#8A8A8A",
                edgecolor="black", linewidth=0.6, label="val AUC at best epoch")
        ax3.bar([x + w / 2 for x in xs2], v_final, width=w, color=C_ESMC,
                edgecolor="black", linewidth=0.6, label="val AUC at final epoch")
        for x, c in zip(xs2, of_chans):
            ax3.text(x, max(v_best[x], v_final[x]) + 0.004, f"{mean_deg[c]:+.3f}",
                     ha="center", fontsize=8, fontweight="bold")
        ax3.set_xticks(xs2)
        ax3.set_xticklabels([f"c{c}" for c in of_chans], fontsize=9)
        ax3.set_ylabel("val AUC (mean over 6 folds; labels = best − final degradation)")
        ax3.set_ylim(0, 1.0)
        ax3.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax3.grid(axis="y", alpha=0.3)
        ax3.legend(fontsize=9, loc="lower right")
        ax3.set_title(f"Best vs final-epoch val AUC per channel ({args.data_label}, k2={want_k2})",
                      fontsize=12)
        fig3.tight_layout()
        vt_png = plots_dir / f"{prefix}_val_vs_test.png"
        fig3.savefig(vt_png, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  wrote {vt_png}")

        of_tsv = results_dir / f"{prefix}_overfit.tsv"
        with open(of_tsv, "w", newline="") as fh:
            fields = ["k2", "channel", "model_key", "fold", "epochs_run",
                      "best_val_auc", "final_val_auc", "degradation_best_minus_final"]
            wr = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
            wr.writeheader()
            for r_ in sorted(gap_rows, key=lambda d: (d["channel"], d["fold"])):
                wr.writerow({"k2": want_k2, "channel": r_["channel"],
                             "model_key": by_chan[r_["channel"]]["model_key"],
                             "fold": r_["fold"], "epochs_run": r_["epochs_run"],
                             "best_val_auc": f"{r_['best_val_auc']:.6f}",
                             "final_val_auc": f"{r_['final_val_auc']:.6f}",
                             "degradation_best_minus_final": f"{r_['degradation']:+.6f}"})
        print(f"  wrote {of_tsv}")
    else:
        print("  SKIP overfit diagnostics: no learning_curves in any JSON")

    cap = (f"2-layer CNN channel sweep at k1=17, k2={want_k2} on {args.data_label} windows "
           "(ESM-C-300M, naive GAP). k1=17 fixed, k2 fixed. Bars: pooled out-of-fold AUC over "
           "6 outer buckets (same vector DeLong tests); error bars: std of "
           "the 6 outer-fold AUCs. Brackets: paired DeLong vs anchor "
           f"c{anchor_c} (Holm-adjusted, significant pairs only). "
           "Grey hatched bar: 1-layer k17 C32 reference (different depth = "
           "unpaired comparison, bars only, not DeLong-tested). "
           "y-axis 0-1; chance = 0.5. Ignores F1/MCC (fixed 0.5 threshold).")
    if of_chans:
        cap += (" Overfit panels: per-channel mean train-loss (left axis) vs val "
                "AUC (right axis) per epoch; best vs final-epoch val-AUC bars "
                "labelled with post-peak degradation (flat 6-fold CV: val bucket "
                "doubles as test, so no separate val-vs-test gap exists).")
    else:
        cap += " Overfit panels skipped (no learning_curves in JSONs)."
    print("\n" + "=" * 60 + "\n  CAPTION (paste below the figure):\n" + "=" * 60)
    print(cap)
    print("=" * 60)


if __name__ == "__main__":
    main()
