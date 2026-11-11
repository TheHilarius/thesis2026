#!/usr/bin/env python3
"""
40_roc_crossing_check.py — screening: do any DeLong-compared ROC curves cross?

For every run that produced a DeLong/paired-tests output, recompute both ROC
curves from the pooled OOF predictions and look for sign changes of
TPR_a(fpr) - TPR_b(fpr) on a fine shared FPR grid.

Interpretation caveat (deliberate): curve crossings do NOT invalidate the
paired DeLong test itself (its null is delta-AUC = 0, asymptotically valid
either way).  They only weaken the "uniformly better" reading of a
significant delta-AUC — the ranking can flip inside the FPR range.

Two sources of arm definitions:
  * ARMS_SCRIPTS: import the original DeLong script, reuse its ARMS/FAMILIES
    (pairs are exactly the ones DeLong tested).
  * GLOB_RUNS:    generic — every matching cv_results JSON in the run (after
    timestamp-dedup), all pairs. Superset of the DeLong pairs, so no
    comparison is missed; rows flag whether y is identical (aligned).

padmode runs additionally get padded-subset variants (mask from 14_'s
PADMASK_FILES, via 15_'s reconstruct_oof in df order) because that is the
row set the primary DeLong used there.

Output: runs/<tag>/results/roc_crossings.tsv
  columns include paired DeLong p (recomputed via 15's delong_test),
  disagree_range/max_flip_dtp (FPR mass where the dAUC loser wins) and
  tier = critical|weak|noise|clean (see crossing_stats docstring).
  --plot: per-run ROC overlay roc_all_<variant>.png + detail panels
  (ROC + delta-TPR) for tier=critical pairs only.

Usage:
  python src/investigation/40_roc_crossing_check.py            # all runs
  python src/investigation/40_roc_crossing_check.py --run padmode_choice --plot
"""

import argparse
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PIPE_PY = SRC_DIR.parent / "pipeline" / "python"
if str(PIPE_PY) not in sys.path:
    sys.path.insert(0, str(PIPE_PY))
from config import PROJECT_ROOT, SPLIT_DATA_PATH  # noqa: E402

R = PROJECT_ROOT / "runs"

# ── registry ────────────────────────────────────────────────────────────────
# runtag -> (script filename, family filter or None)
ARMS_SCRIPTS = {
    "dim21_eosbos": ("21_dim21_delong.py", None),
    "try_out_compositional": ("23_compositional_delong.py", None),
    "flank5_compare": ("32_flank5_significance.py", None),
    "varfold_pilot": ("24_varfold_delong.py", ("esmc",)),
    "varfold_pilot_eosbos": ("24_varfold_delong.py", ("esmif",)),
    "investigate_diff_logo_pattern_w_simpler_models":
        ("31_model_compare.py", None),
}

# runtag -> spec.  loader: "concat" (fold-ordered, like 21/23/24/32) or
# "df" (df_all_with_folds order via 15.reconstruct_oof — needed for pad masks).
GLOB_RUNS = {
    "encodings_sweep": dict(
        globs=["runs/encodings_sweep/models/*/cv_results_*.json"]),
    "is_encodings_helping_v2": dict(
        globs=["runs/is_encodings_helping_v2/models/*/cv_results_*.json"]),
    "handcrafted_sweep": dict(  # 19: campaign + ref-campaign (encodings_sweep)
        globs=["runs/handcrafted_sweep/models/*/cv_results_*.json",
               "runs/encodings_sweep/models/*/cv_results_*.json"]),
    "is_handcrafted_helping_v2": dict(  # 19 --ref-campaign is_encodings_helping_v2
        globs=["runs/is_handcrafted_helping_v2/models/*/cv_results_*.json",
               "runs/is_encodings_helping_v2/models/*/cv_results_*.json"]),
    "dimselect_matched": dict(  # 24_dimselect + 30_dimselect_ev pairs
        globs=["runs/dimselect_matched/models/*/cv_results_*.json",
               "runs/ev_matched_topXvariance_sweep/models/*/cv_results_*.json",
               "runs/esmc300m_pca_sweep/models/flat/"
               "cv_results_xgb_emb_esmc300m_win_padtoken_pca2278_flat_*.json",
               "runs/esmif_eos_sweep/models/flat/"
               "cv_results_xgb_emb_esmif_eos_bos_repeat_pca632_flat_*.json",
               "runs/encodings_sweep/models/flat/"
               "cv_results_xgb_blosum50_flat_*.json"]),
    "ev_matched_topXvariance_sweep": dict(  # 26 (var~pca) + 27 (var~raw)
        globs=["runs/ev_matched_topXvariance_sweep/models/"
               "*/cv_results_*.json",
               "runs/esmc300m_pca_sweep/models/flat/"
               "cv_results_xgb_emb_esmc300m_win_padtoken_pca*_flat_*.json",
               "runs/esmif_eos_sweep/models/flat/"
               "cv_results_xgb_emb_esmif_eos_bos_repeat_pca*_flat_*.json",
               "runs/esmc300m_pca_sweep/models/flat_raw/"
               "cv_results_xgb_emb_esmc300m_win_padtoken_flat_raw_*.json",
               "runs/esmif_eos_sweep/models/flat_raw/"
               "cv_results_xgb_emb_esmif_eos_bos_repeat_flat_raw_*.json"]),
    "ab_dataset_test": dict(  # 4 cv files: 2 tags x (full, remove715-reduced)
        globs=["runs/ab_dataset_test/models/*/cv_results_*.json"]),
    "esmc300vs600": dict(  # 16: esmc300m run + esmc600m arm in padmode_choice
        globs=["runs/esmc300vs600/models/flat/cv_results_*.json",
               "runs/padmode_choice/models/flat/"
               "cv_results_xgb_emb_esmc_win_padtoken_pca6389_flat_*.json"]),
    "padmode_choice": dict(
        globs=["runs/padmode_choice/models/flat/cv_results_*.json"],
        loader="df", variants="padmode"),
    "mpnn_padmode_v48020": dict(
        globs=["runs/mpnn_padmode_v48020/models/flat/cv_results_*.json"],
        loader="df", variants="padmode"),
}

# ── module / vector loading ────────────────────────────────────────────────
_mod_cache = {}


def load_module(fname: str):
    if fname not in _mod_cache:
        spec = importlib.util.spec_from_file_location(
            fname[:-3], SRC_DIR / fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _mod_cache[fname] = mod
    return _mod_cache[fname]


def load_concat(path: Path):
    """Pooled OOF (y, p), fold keys concatenated in order — like 21/23/24/32."""
    res = json.loads(path.read_text())
    key = "outer_val_predictions" if "outer_val_predictions" in res \
        else "fold_predictions"
    folds = sorted(int(k) for k in res[key])
    y = np.concatenate([np.asarray(res[key][str(f)]["y_true"]) for f in folds])
    p = np.concatenate([np.asarray(res[key][str(f)]["y_prob"], dtype=float)
                        for f in folds])
    return y, p


_df_state = {}


def load_df(path: Path):
    """OOF in df_all_with_folds row order — 15's reconstruct_oof, so that
    PADMASK_FILES masks (also df-ordered) line up."""
    if not _df_state:
        p15 = load_module("15_padmode_significance.py")
        df = pd.read_csv(SPLIT_DATA_PATH, usecols=["label", "fold"])
        p15.df_folds = df["fold"].to_numpy()
        _df_state["p15"] = p15
        _df_state["y"] = df["label"].to_numpy()
    p15 = _df_state["p15"]
    res = json.loads(path.read_text())
    p = p15.reconstruct_oof(res, len(_df_state["y"]))
    return _df_state["y"].copy(), p


# ── arm resolution for ARMS_SCRIPTS ────────────────────────────────────────
def resolve_pattern(pat):
    """Newest file matching pat. Absolute glob -> direct; relative list ->
    24_'s find_json semantics (first pattern with a hit, newest match)."""
    if isinstance(pat, (list, tuple)):
        for p in pat:
            hit = resolve_pattern(p)
            if hit is not None:
                return hit
        return None
    s = str(pat)
    if s.startswith("**/"):  # recursive from PROJECT_ROOT (24_'s find_json)
        hits = sorted(PROJECT_ROOT.glob(s))
        return hits[-1] if hits else None
    pat = Path(s)
    if not pat.is_absolute():
        pat = R / s
    files = sorted(pat.parent.glob(pat.name)) if pat.parent.exists() else []
    return files[-1] if files else None


def arms_from_script(fname, family_filter):
    mod = load_module(fname)
    arms_spec = getattr(mod, "ARMS")
    if isinstance(arms_spec, list):  # 31_model_compare: [(model, feat, path)]
        arms = {f"{m}_{f}": p for m, f, p in arms_spec}
    else:
        arms = dict(arms_spec)
    fams = getattr(mod, "FAMILIES", None)
    if fams is None:
        names = sorted(arms)
        pairs = [(a, b) for a, b in itertools.combinations(names, 2)]
        fams = {"all": pairs}
    if family_filter:
        fams = {k: v for k, v in fams.items() if k in family_filter}
    return arms, fams


# ── padmode variants (subset row sets of 15_padmode_significance) ─────────
def padmode_variant(stem, y, p, masks):
    """{'full': (y,p), 'subset_ESM-C': (y_sub,p_sub), ...}"""
    p15 = _df_state["p15"]
    out = {"full": (y, p)}
    # stems are timestamp-stripped ("..._flat"); both regexes end in "_flat_"
    probe = stem + "_"
    m = p15.EMB_JSON_RE.search(probe)
    if m and m.group(1) in p15.PADTYPE_MAP:
        tk = p15.PADTYPE_MAP[m.group(1)][0]
        mk = masks[tk]
        out[f"subset_{tk}"] = (y[mk], p[mk])
    elif p15.BASELINE_JSON_RE.search(probe):
        # 15's baseline_subset_esmc: baselines on the ESM-C padded mask
        mk = masks["ESM-C"]
        out["subset_ESM-C"] = (y[mk], p[mk])
    return out


# ── crossing detection ─────────────────────────────────────────────────────
def crossing_stats(y_a, p_a, y_b, p_b, min_amp, aligned, crit,
                   lo=0.01, hi=0.99, n=1001):
    """Crossings of TPR_a - TPR_b on a shared FPR grid.

    A sign change counts only if both flanking points reach |dTPR| >= min_amp
    (endpoint convergence and numeric noise are ignored); FPR outside
    (lo, hi) is ignored.  Crossing does NOT invalidate DeLong's p-value —
    see module docstring.  It weakens the "uniformly better" reading, which
    is what `tier` grades:

      critical  significant DeLong (p < crit['p']) AND the ΔAUC loser wins
                back a meaningful FPR mass (disagree_range >= crit['range'])
                by a meaningful margin (max_flip_dtp >= crit['amp'])
      noise     crossings only in the wiggle zone (all FPR < 0.05)
      weak      crossing but not critical (p >= crit['p'] or small flips)
      clean     no crossing

    disagree_range: FPR mass where sign(dTPR) opposes sign(ΔAUC) with
    |dTPR| >= min_amp.  p_delong: paired DeLong (15's delong_test, midrank),
    only when rows are aligned.
    """
    fpr_a, tpr_a, _ = roc_curve(y_a, p_a)
    fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
    grid = np.linspace(0.0, 1.0, n)
    d = np.interp(grid, fpr_a, tpr_a) - np.interp(grid, fpr_b, tpr_b)
    interior = (grid > lo) & (grid < hi)
    g, dd = grid[interior], d[interior]
    sig = np.flatnonzero(np.abs(dd) >= min_amp)
    cross = []
    for i, j in zip(sig[:-1], sig[1:]):
        if np.sign(dd[i]) != np.sign(dd[j]):
            t = abs(dd[i]) / (abs(dd[i]) + abs(dd[j]))
            cross.append(round(float(g[i] + t * (g[j] - g[i])), 4))
    k = int(np.argmax(np.abs(dd)))
    max_abs = float(abs(dd[k]))
    if cross:
        verdict = "crossing"
    elif max_abs < min_amp:
        verdict = "near_identical"
    else:
        verdict = "clean"
    auc_a = float(roc_auc_score(y_a, p_a))
    auc_b = float(roc_auc_score(y_b, p_b))
    delta = auc_a - auc_b

    # disagreement mass: ΔTPR sign opposes the ΔAUC winner
    step = float(g[1] - g[0])
    dis = np.sign(dd) != np.sign(delta) if delta != 0 else np.zeros_like(dd, bool)
    dis &= np.abs(dd) >= min_amp
    disagree_range = round(float(dis.sum() * step), 4)
    max_flip_dtp = round(float(np.abs(dd[dis]).max()) if dis.any() else 0.0, 4)

    p_delong = float("nan")
    if aligned:
        p15 = load_module("15_padmode_significance.py")
        _, _, var_d = p15.delong_test(y_a, p_a, p_b)
        if var_d > 0:
            from scipy.stats import norm
            p_delong = float(2.0 * norm.sf(abs(delta) / np.sqrt(var_d)))

    if verdict != "crossing":
        tier = "clean"
    elif cross and all(c < 0.05 for c in cross):
        tier = "noise"
    elif (not np.isnan(p_delong) and p_delong < crit["p"]
          and disagree_range >= crit["range"]
          and max_flip_dtp >= crit["amp"]):
        tier = "critical"
    else:
        tier = "weak"

    return dict(
        n_cross=len(cross),
        cross_fprs=";".join(f"{c:.3f}" for c in cross),
        max_abs_dtp=round(max_abs, 4),
        fpr_at_max_dtp=round(float(g[k]), 3),
        verdict=verdict,
        tier=tier,
        auc_a=auc_a,
        auc_b=auc_b,
        delta_auc=round(delta, 6),
        p_delong=p_delong,
        disagree_range=disagree_range,
        max_flip_dtp=max_flip_dtp,
    )


# ── per-run drivers ────────────────────────────────────────────────────────
def run_script_run(run_tag, fname, fam_filter, min_amp, crit, rows,
                   plot=False, plot_max=50):
    arms_spec, fams = arms_from_script(fname, fam_filter)
    data, missing = {}, []
    for name, pat in arms_spec.items():
        hit = resolve_pattern(pat)
        if hit is None:
            missing.append(name)
            continue
        data[name] = load_concat(hit)
    if missing:
        print(f"  WARN missing arms: {missing}")
    for fam, pairs in fams.items():
        for a, b in pairs:
            if a not in data or b not in data:
                continue
            (ya, pa), (yb, pb) = data[a], data[b]
            aligned = bool(len(ya) == len(yb) and np.array_equal(ya, yb))
            st = crossing_stats(ya, pa, yb, pb, min_amp, aligned, crit)
            rows.append(dict(run=run_tag, family=fam, arm_a=a, arm_b=b,
                             variant="full", n_a=len(ya), n_b=len(yb),
                             aligned=aligned, **st))
    if plot:
        _plot_run_overlay(run_tag, "full",
                          [(n, y, p) for n, (y, p) in data.items()])
        _plot_crossings(run_tag, "full",
                        [(n, y, p) for n, (y, p) in data.items()],
                        rows, plot_max)


def run_glob_run(run_tag, spec, min_amp, crit, rows, plot=False, plot_max=50):
    loader = load_df if spec.get("loader") == "df" else load_concat
    files = []
    for g in spec["globs"]:
        files += sorted(PROJECT_ROOT.glob(g))
    # dedup: same dir + timestamp-stripped stem -> newest; differing n kept
    # (ab_dataset: full vs remove715-reduced share a stem)
    groups = {}
    for f in files:
        stem = re.sub(r"_\d{8}_\d{6}$", "", f.stem)
        groups.setdefault((f.parent, stem), []).append(f)
    arms = {}
    for (parent, stem), fs in groups.items():
        fs = sorted(fs, key=lambda p: p.name)
        loaded = [(f, *loader(f)) for f in fs]
        if len({len(y) for _, y, _ in loaded}) == 1:
            f, y, p = loaded[-1]
            name = stem
            if name in arms:  # same stem in two dirs
                name = f"{parent.name}/{stem}"
            arms[name] = (y, p)
        else:
            for f, y, p in loaded:
                arms[f"{stem}__n{len(y)}"] = (y, p)
    print(f"  {run_tag}: {len(arms)} arms from {len(files)} files")

    # variant -> [(name, y, p)]
    variants = {}
    if spec.get("variants") == "padmode":
        masks = _df_state["p15"].__dict__.get("_cross_masks")
        if masks is None:
            masks = load_module("14_plot_grid_auc.py").load_padded_masks()
            _df_state["p15"]._cross_masks = masks
        for name, (y, p) in arms.items():
            for v, (vy, vp) in padmode_variant(Path(name).stem, y, p,
                                               masks).items():
                variants.setdefault(v, []).append((name, vy, vp))
    else:
        variants["full"] = [(n, y, p) for n, (y, p) in arms.items()]

    for v, items in variants.items():
        for (na, ya, pa), (nb, yb, pb) in itertools.combinations(items, 2):
            aligned = bool(len(ya) == len(yb) and np.array_equal(ya, yb))
            st = crossing_stats(ya, pa, yb, pb, min_amp, aligned, crit)
            rows.append(dict(run=run_tag, family="glob", arm_a=na, arm_b=nb,
                             variant=v, n_a=len(ya), n_b=len(yb),
                             aligned=aligned, **st))
        if plot:
            _plot_run_overlay(run_tag, v, items)
            _plot_crossings(run_tag, v, items, rows, plot_max)


def _short_arm(name):
    for pre in ("cv_results_xgb_", "cv_results_"):
        if name.startswith(pre):
            name = name[len(pre):]
    return name.replace("_flat", "").replace("__n", " n")


def _plot_run_overlay(run_tag, variant, items):
    """One figure, every arm of this run/variant on a single ROC axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not items:
        return
    scored = sorted(
        ((n, y, p, roc_auc_score(y, p)) for n, y, p in items),
        key=lambda t: -t[3])
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.32 * len(scored) + 2.0)))
    for n, y, p, auc in scored:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.0,
                label=f"{_short_arm(n)}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{run_tag} — {variant} (n={len(scored)} arms)")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    fig.tight_layout()
    out_dir = R / run_tag / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"roc_all_{variant}.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def _plot_crossings(run_tag, variant, items, rows, plot_max):
    """Detail panels (ROC + dTPR) for critical pairs only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hits = [r for r in rows if r["run"] == run_tag
            and r["variant"] == variant and r["tier"] == "critical"]
    by_arm = {n: (y, p) for n, y, p in items}
    out_dir = R / run_tag / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in hits[:plot_max]:
        if r["arm_a"] not in by_arm or r["arm_b"] not in by_arm:
            continue
        (ya, pa), (yb, pb) = by_arm[r["arm_a"]], by_arm[r["arm_b"]]
        fpr_a, tpr_a, _ = roc_curve(ya, pa)
        fpr_b, tpr_b, _ = roc_curve(yb, pb)
        grid = np.linspace(0, 1, 1001)
        d = np.interp(grid, fpr_a, tpr_a) - np.interp(grid, fpr_b, tpr_b)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].plot(fpr_a, tpr_a, label=r["arm_a"])
        ax[0].plot(fpr_b, tpr_b, label=r["arm_b"])
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.5)
        ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
        ax[0].legend(fontsize=7); ax[0].set_title(variant)
        ax[1].plot(grid, d)
        ax[1].axhline(0, color="k", lw=0.5)
        for c in str(r["cross_fprs"]).split(";"):
            if c:
                ax[1].axvline(float(c), color="r", ls=":", lw=0.8)
        ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR_a - TPR_b")
        ax[1].set_title(f"n_cross={r['n_cross']}")
        fig.suptitle(f"{run_tag}: {r['arm_a']} vs {r['arm_b']}")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9._-]", "_",
                      f"cross_{variant}_{r['arm_a']}_vs_{r['arm_b']}")
        fig.savefig(out_dir / f"{safe}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run", action="append", default=None,
                    help="runtag to check (repeatable; default: all)")
    ap.add_argument("--min-amp", type=float, default=0.002,
                    help="min |dTPR| for a sign change to count "
                         "(default 0.002)")
    ap.add_argument("--crit-range", type=float, default=0.05,
                    help="min FPR mass where the dAUC loser wins "
                         "for tier=critical (default 0.05)")
    ap.add_argument("--crit-amp", type=float, default=0.02,
                    help="min |dTPR| in the disagreeing region for "
                         "tier=critical (default 0.02)")
    ap.add_argument("--crit-p", type=float, default=0.05,
                    help="max paired DeLong p for tier=critical "
                         "(default 0.05)")
    ap.add_argument("--plot", action="store_true",
                    help="per-run ROC overlay (roc_all_<variant>.png) + "
                         "detail panels for critical pairs")
    ap.add_argument("--plot-max", type=int, default=50,
                    help="max critical-pair detail plots per variant")
    args = ap.parse_args()
    crit = dict(range=args.crit_range, amp=args.crit_amp, p=args.crit_p)

    all_rows = []
    tags = args.run or (list(ARMS_SCRIPTS) + list(GLOB_RUNS))
    for tag in tags:
        print(f"== {tag}")
        rows = []
        if tag in ARMS_SCRIPTS:
            fname, fam_filter = ARMS_SCRIPTS[tag]
            run_script_run(tag, fname, fam_filter, args.min_amp, crit, rows,
                           plot=args.plot, plot_max=args.plot_max)
        elif tag in GLOB_RUNS:
            run_glob_run(tag, GLOB_RUNS[tag], args.min_amp, crit, rows,
                         plot=args.plot, plot_max=args.plot_max)
        else:
            print("  WARN unknown runtag — skipped")
            continue
        out = R / tag / "results" / "roc_crossings.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["run", "family", "variant", "arm_a", "arm_b", "n_a", "n_b",
                "aligned", "auc_a", "auc_b", "delta_auc", "p_delong",
                "tier", "verdict", "n_cross", "cross_fprs", "max_abs_dtp",
                "fpr_at_max_dtp", "disagree_range", "max_flip_dtp"]
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False, columns=cols)
        n_cross = sum(r["verdict"] == "crossing" for r in rows)
        n_crit = sum(r["tier"] == "critical" for r in rows)
        n_align = sum(r["verdict"] == "crossing" and r["aligned"]
                      for r in rows)
        print(f"  -> {out.relative_to(PROJECT_ROOT)}  "
              f"({len(rows)} pairs, {n_cross} crossing, "
              f"{n_align} aligned, {n_crit} CRITICAL)")
        all_rows += rows

    df = pd.DataFrame(all_rows)
    print("\n== CRITICAL pairs (significant DeLong + meaningful ranking "
          "flip) — these undermine the 'uniformly better' claim")
    if len(df):
        crit_df = df[df["tier"] == "critical"].sort_values(
            "disagree_range", ascending=False)
        if crit_df.empty:
            print("  none")
        for _, r in crit_df.iterrows():
            print(f"  {r['run']:<48} {str(r['variant'])[:18]:<18} "
                  f"{_short_arm(str(r['arm_a']))[:30]:<30} vs "
                  f"{_short_arm(str(r['arm_b']))[:30]:<30} "
                  f"dAUC={r['delta_auc']:+.4f} p={r['p_delong']:.2e} "
                  f"flip@{r['disagree_range']:.0%}FPR max|dTPR|="
                  f"{r['max_flip_dtp']:.3f} n_x={r['n_cross']}")
    print("\n== per-run tier counts")
    for tag, grp in df.groupby("run"):
        vc = grp["tier"].value_counts()
        print(f"  {tag:<50} critical={vc.get('critical', 0):>3} "
              f"weak={vc.get('weak', 0):>4} noise={vc.get('noise', 0):>4} "
              f"clean={vc.get('clean', 0):>4}")


if __name__ == "__main__":
    main()
