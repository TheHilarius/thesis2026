#!/usr/bin/env python3
"""
24_dimselect_delong.py — benchmark the dimselect (supervised per-dim
screening) campaign against topXvariance, PCA and blosum50.

Figures (runs/<out-root>/plots/):
    dimselect_benchmark_curve.png/.pdf
        2 panels (ESM-C, ESM-IF): Val AUC vs N for
        - dimselect (xgb screener) + lr anchors
        - topXvariance (unsupervised variance ranking)
        - horizontal reference lines: PCA (k=2278 / k=632) and blosum50

Tables:
    results/dimselect_curve_values.tsv   all curve points (thesis table)
    results/dimselect_delong_pairs.tsv   paired DeLong + Holm
        arm_a arm_b auc_a auc_b delta se z p_delong p_holm family star

DeLong families (Holm within family):
    matched_esmc   dimselect{434,643,793} ~ variance{434,643,793}
    matched_esmif  dimselect{122,260,429} ~ variance{122,260,429}
    vs_pca         best dimselect ~ PCA, best variance ~ PCA (per embedding)
    vs_blosum      best dimselect ~ blosum50 (per embedding)

Reuse: DeLong + Holm from 15_padmode_significance.py (same estimand as
23_compositional_delong.py: paired DeLong on pooled outer-fold OOF).
Curve y-values are the cv_summary mean over outer folds (matches the
"Val AUC-ROC" lines in the slurm logs); the TSV reports pooled-OOF AUCs
from the DeLong test itself.

dimselect arms are collected from runs/dimselect_sweep, its _legacy
relocation, and runs/dimselect_matched — whichever exists. Missing arms
(matched-N jobs still running) are skipped with a warning, so re-run
once jobs land.
"""

import argparse
import importlib.util
import json
import re
import sys
from math import erfc, sqrt
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import numpy as np  # noqa: E402

# reuse DeLong + Holm from the padmode significance script
_spec = importlib.util.spec_from_file_location(
    "p15", SRC_DIR / "15_padmode_significance.py")
_p15 = importlib.util.module_from_spec(_spec)
sys.modules["p15"] = _p15
_spec.loader.exec_module(_p15)

R = PROJECT_ROOT / "runs"

# dimselect JSONs may live in the original sweep, its _legacy move, or
# the matched-N campaign — scan whichever exists.
DIMSELECT_DIRS = [
    R / "dimselect_sweep/models/dimselect",
    R / "_legacy/dimselect_sweep/models/dimselect",
    R / "dimselect_matched/models/dimselect",
]

# per embedding: dimselect feature key, topXvariance feature key,
# matched-N points (exact pairs vs the variance sweep), PCA reference
EMBS = {
    "esmc": {
        "label": "ESM-C-300M",
        "ds": "emb_esmc300m_win_padtoken",
        "var": "emb_esmc300m_win_padtoken",
        "var_dir": R / "ev_matched_topXvariance_sweep/models/simple21",
        "match_ns": [434, 643, 793],
        "pca": R / "esmc300m_pca_sweep/models/flat"
                 "/cv_results_xgb_emb_esmc300m_win_padtoken_pca2278_flat_*.json",
        "pca_k": 2278,
    },
    "esmif": {
        "label": "ESM-IF",
        "ds": "emb_esmif_eos_bos_repeat",
        "var": "emb_esmif_eos_bos_repeat",
        "var_dir": R / "ev_matched_topXvariance_sweep/models/vardims",
        "match_ns": [122, 260, 429],
        "pca": R / "esmif_eos_sweep/models/flat"
                 "/cv_results_xgb_emb_esmif_eos_bos_repeat_pca632_flat_*.json",
        "pca_k": 632,
    },
}
BLOSUM = R / "encodings_sweep/models/flat/cv_results_xgb_blosum50_flat_*.json"


def newest(pattern: Path):
    """Newest file matching a Path-with-wildcards (None if none exist)."""
    files = sorted(pattern.parent.glob(pattern.name)) if pattern.parent.exists() else []
    return files[-1] if files else None


def load_json(path):
    return json.loads(Path(path).read_text())


def mean_std(res):
    s = res["cv_summary"]["auc_roc"]
    return float(s["mean"]), float(s["std"])


def load_oof(path):
    """Pooled OOF (y, p) over sorted outer folds."""
    res = load_json(path)
    folds = sorted((int(k) for k in res["outer_val_predictions"]))
    y = np.concatenate([np.asarray(res["outer_val_predictions"][str(f)]["y_true"])
                        for f in folds])
    p = np.concatenate([np.asarray(res["outer_val_predictions"][str(f)]["y_prob"],
                                   dtype=float) for f in folds])
    return y, p


def dimselect_points(ds_key):
    """(N, screen) -> (path, mean, std), latest timestamp wins."""
    pts = {}
    for d in DIMSELECT_DIRS:
        if not d.exists():
            continue
        for f in d.glob(f"cv_results_*_{ds_key}_dimselect_top*_*.json"):
            m = re.search(rf"_{re.escape(ds_key)}_dimselect_top(\d+)_(xgb|lr_l2)"
                          r"_\d{8}_\d{6}\.json$", f.name)
            if not m:
                continue
            key = (int(m.group(1)), m.group(2))
            if key not in pts or f.name > pts[key].name:
                pts[key] = f
    return {k: (v,) + mean_std(load_json(v)) for k, v in pts.items()}


def variance_points(var_key, var_dir):
    """N -> (path, mean, std), latest timestamp wins.

    Tag differs per family: esmc arms are models/simple21/*_simple21_varN_*,
    esmif arms are models/vardims/*_vardims_ev<E>_varN_* (eos/bos-repeat) —
    the regex accepts any '<tag>var<N>' suffix before the timestamp.
    """
    pts = {}
    if not var_dir.exists():
        return pts
    for f in var_dir.glob(f"cv_results_*_{var_key}_*.json"):
        m = re.search(rf"_{re.escape(var_key)}_[a-z0-9_]*var(\d+)_"
                      r"\d{8}_\d{6}\.json$", f.name)
        if not m:
            continue
        n = int(m.group(1))
        if n not in pts or f.name > pts[n].name:
            pts[n] = f
    return {n: (v,) + mean_std(load_json(v)) for n, v in pts.items()}


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def arm_name(emb, scheme, n):
    return {"dimselect_xgb": f"dimselect_{emb}_n{n}",
            "dimselect_lr": f"dimselectlr_{emb}_n{n}",
            "variance": f"variance_{emb}_n{n}"}[scheme]


def delong_pair(name_a, pa, name_b, pb, family):
    """Paired DeLong row, or None if the two OOF vectors misalign."""
    if not np.array_equal(pa[0], pb[0]):
        print(f"  WARN [{family}]: skip {name_a}~{name_b} (OOF rows misaligned)")
        return None
    auc_a, auc_b, var_d = _p15.delong_test(pa[0], pa[1], pb[1])
    d = auc_a - auc_b
    se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
    p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
    return (name_a, name_b, auc_a, auc_b, d, se, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-root", default="dimselect_matched",
                    help="run folder under runs/ for outputs "
                         "(default: dimselect_matched)")
    args = ap.parse_args()
    out_root = R / args.out_root
    results_dir = out_root / "results"
    plots_dir = out_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect all arms ------------------------------------------------
    pools = {emb: {"dimselect_xgb": {}, "dimselect_lr": {}, "variance": {}}
             for emb in EMBS}
    refs = {}       # arm name -> (path, mean, std) for pca / blosum50
    pooled = {}     # arm name -> (y, p)
    curve_rows = []

    for emb, cfg in EMBS.items():
        for scheme in ("dimselect_xgb", "dimselect_lr"):
            for n, (f, mean, std) in dimselect_points(cfg["ds"]).items():
                if (scheme == "dimselect_xgb") != (n[1] == "xgb"):
                    continue
                nn = n[0] if isinstance(n, tuple) else n
                pools[emb][scheme][nn] = (f, mean, std)
                curve_rows.append((emb, scheme, nn, mean, std, 29 * nn))
        for n, (f, mean, std) in variance_points(cfg["var"],
                                                 cfg["var_dir"]).items():
            pools[emb]["variance"][n] = (f, mean, std)
            curve_rows.append((emb, "variance", n, mean, std, 29 * n))

        f = newest(cfg["pca"])
        if f is None:
            print(f"  WARN: missing PCA reference for {emb}")
        else:
            res = load_json(f)
            mean, std = mean_std(res)
            refs[f"pca_{emb}"] = (f, mean, std)
            curve_rows.append((emb, "pca", cfg["pca_k"], mean, std, cfg["pca_k"]))

        if "blosum50" not in refs:
            f = newest(BLOSUM)
            if f is None:
                print("  WARN: missing blosum50 reference")
            else:
                res = load_json(f)
                mean, std = mean_std(res)
                refs["blosum50"] = (f, mean, std)
                nfeat = res.get("n_features", "")
                curve_rows.append((emb, "blosum50", nfeat, mean, std, nfeat))

        print(f"  {emb}: dimselect_xgb N={sorted(pools[emb]['dimselect_xgb'])}, "
              f"dimselect_lr N={sorted(pools[emb]['dimselect_lr'])}, "
              f"variance N={sorted(pools[emb]['variance'])}")

    for emb, schemes in pools.items():
        for scheme, pts in schemes.items():
            for n, (f, _, _) in pts.items():
                pooled[arm_name(emb, scheme, n)] = load_oof(f)
    for name, (f, _, _) in refs.items():
        pooled[name] = load_oof(f)

    # ---- DeLong pairs ----------------------------------------------------
    def best_n(emb, scheme):
        pts = pools[emb][scheme]
        return max(pts, key=lambda n: pts[n][1]) if pts else None

    families = {}
    for emb, cfg in EMBS.items():
        matched = []
        for n in cfg["match_ns"]:
            a, b = f"dimselect_{emb}_n{n}", f"variance_{emb}_n{n}"
            if a in pooled and b in pooled:
                matched.append((a, b))
            else:
                print(f"  WARN [matched_{emb}]: skip N={n} (arm not finished)")
        families[f"matched_{emb}"] = matched

    vs_pca, vs_blosum = [], []
    for emb in EMBS:
        bn_ds, bn_var = best_n(emb, "dimselect_xgb"), best_n(emb, "variance")
        if bn_ds is not None and f"pca_{emb}" in pooled:
            vs_pca.append((f"dimselect_{emb}_n{bn_ds}", f"pca_{emb}"))
        if bn_var is not None and f"pca_{emb}" in pooled:
            vs_pca.append((f"variance_{emb}_n{bn_var}", f"pca_{emb}"))
        if bn_ds is not None and "blosum50" in pooled:
            vs_blosum.append((f"dimselect_{emb}_n{bn_ds}", "blosum50"))
    families["vs_pca"] = vs_pca
    families["vs_blosum"] = vs_blosum

    rows = []
    for family, pairs in families.items():
        if not pairs:
            print(f"  WARN [{family}]: no pairs available yet")
            continue
        computed = [g for g in (
            delong_pair(a, pooled[a], b, pooled[b], family) for a, b in pairs)
            if g is not None]
        if not computed:
            continue
        adjs = _p15.holm_adjust([c[6] for c in computed])
        for (a, b, auc_a, auc_b, d, se, p), ph in zip(computed, adjs):
            z = d / se if se > 0 else 0.0
            rows.append({
                "arm_a": a, "arm_b": b,
                "auc_a": f"{auc_a:.6f}", "auc_b": f"{auc_b:.6f}",
                "delta": f"{d:.6f}", "se": f"{se:.6f}", "z": f"{z:.4f}",
                "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                "family": family, "star": star(ph),
            })
            print(f"  {family:<14} {a} ~ {b}: d={d:+.4f} "
                  f"p={p:.2e} holm={ph:.2e} [{star(ph)}]")

    # ---- write TSVs ------------------------------------------------------
    curve_out = results_dir / "dimselect_curve_values.tsv"
    with open(curve_out, "w") as fh:
        fh.write("embedding\tscheme\tn_dims\tmean_auc\tstd_auc\tn_features\n")
        for r in sorted(curve_rows, key=lambda r: (r[0], r[1], str(r[2]))):
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"  wrote {curve_out}  ({len(curve_rows)} rows)")

    if rows:
        pairs_out = results_dir / "dimselect_delong_pairs.tsv"
        with open(pairs_out, "w") as fh:
            hdr = list(rows[0])
            fh.write("\t".join(hdr) + "\n")
            for r in rows:
                fh.write("\t".join(r[c] for c in hdr) + "\n")
        print(f"  wrote {pairs_out}  ({len(rows)} pairs)")
    else:
        print("  WARN: no DeLong pairs computed — matched-N jobs not finished?")

    # ---- figure: AUC vs N, 2 panels --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (emb, cfg) in zip(axes, EMBS.items()):
        ds = pools[emb]["dimselect_xgb"]
        lr = pools[emb]["dimselect_lr"]
        var = pools[emb]["variance"]

        if var:
            ns = sorted(var)
            ax.plot(ns, [var[n][1] for n in ns], "s-", color="tab:orange",
                    label="topXvariance (unsupervised)")
            ax.fill_between(ns,
                            [var[n][1] - var[n][2] for n in ns],
                            [var[n][1] + var[n][2] for n in ns],
                            color="tab:orange", alpha=0.15)
        if ds:
            ns = sorted(ds)
            ax.plot(ns, [ds[n][1] for n in ns], "o-", color="tab:blue",
                    label="dimselect (supervised)")
            ax.fill_between(ns,
                            [ds[n][1] - ds[n][2] for n in ns],
                            [ds[n][1] + ds[n][2] for n in ns],
                            color="tab:blue", alpha=0.15)
            ax.annotate(f"{ds[ns[-1]][1]:.4f}", (ns[-1], ds[ns[-1]][1]),
                        textcoords="offset points", xytext=(-4, 8),
                        ha="right", fontsize=8, color="tab:blue")
        if lr:
            ns = sorted(lr)
            ax.plot(ns, [lr[n][1] for n in ns], "x", color="tab:blue",
                    markersize=7, label="dimselect (LR screener)")

        if f"pca_{emb}" in refs:
            m = refs[f"pca_{emb}"][1]
            ax.axhline(m, ls="--", color="gray",
                       label=f"PCA k={cfg['pca_k']} ({m:.4f})")
        if "blosum50" in refs:
            m = refs["blosum50"][1]
            ax.axhline(m, ls="--", color="tab:green",
                       label=f"blosum50 ({m:.4f})")

        ax.set_xlabel("dims kept (N, raw 29×N features)")
        ax.set_ylabel("Val AUC-ROC (mean ± sd, outer folds)")
        ax.set_title(cfg["label"])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=8, frameon=False)
        ax.set_xticks(sorted(set(list(ds) + list(lr) + list(var))))
        ax.tick_params(axis="x", labelsize=8)
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = plots_dir / f"dimselect_benchmark_curve.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
