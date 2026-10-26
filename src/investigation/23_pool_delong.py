#!/usr/bin/env python3
"""
23_pool_delong.py — paired DeLong tests for the pool barplot.

Compares pooled nested-CV OOF predictions (cv_results_*.json,
outer_val_predictions) between the pool bars of 22_pool_barplot and
their flat-PCA anchor, separately per embedding:

    families (Holm within family):
      esmc     esmc_<pool> ~ esmc_pca80  (7 pairs, one per pool method)
      esmif    esmif_<pool> ~ esmif_pca60 (7 pairs, one per pool method)

DeLong implementation reused from 15_padmode_significance.py (same
estimand: paired DeLong on pooled OOF, Holm correction per family).

Only naive-mask pool runs are used (naive is the standard; filenames
carry _naive). Missing arms (jobs still running) -> their pairs are
skipped with a warning; rerun after all pool jobs finish.

Output: runs/<runtag>/results/pool_delong_pairs.tsv  (default runtag
    pool_xgb; --runtag to override)
    arm_a arm_b auc_a auc_b delta se z p_delong p_holm family star

Then (brackets picked up automatically):
    python src/investigation/22_pool_barplot.py
"""

import argparse
import importlib.util
import json
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
POOLS = ["mean", "median", "max", "min",
         "mean_max", "mean_median", "mean_std"]

ARMS = {
    **{f"esmc_{p}": R / "pool_xgb/models/flat" /
        f"cv_results_xgb_emb_esmc300m_win_padtoken_flat_pool{p}_naive_*.json"
        for p in POOLS},
    **{f"esmif_{p}": R / "pool_xgb/models/flat" /
        f"cv_results_xgb_emb_esmif_eos_bos_repeat_flat_pool{p}_naive_*.json"
        for p in POOLS},
    "esmc_pca80": R / "esmc300m_pca_sweep/models/flat/cv_results_xgb_emb_esmc300m_win_padtoken_pca2278_flat_*.json",
    "esmif_pca60": R / "esmif_eos_sweep/models/flat/cv_results_xgb_emb_esmif_eos_bos_repeat_pca632_flat_*.json",
}
FAMILIES = {
    "esmc": [(f"esmc_{p}", "esmc_pca80") for p in POOLS],
    "esmif": [(f"esmif_{p}", "esmif_pca60") for p in POOLS],
}


def load_oof(pattern: Path):
    """Pooled OOF (y, p) from the newest matching cv_results JSON."""
    files = sorted(pattern.parent.glob(pattern.name)) if pattern.parent.exists() else []
    if not files:
        return None
    res = json.loads(files[-1].read_text())
    folds = sorted((int(k) for k in res["outer_val_predictions"]))
    y_parts, p_parts = [], []
    for fk in folds:
        e = res["outer_val_predictions"][str(fk)]
        y_parts.append(np.asarray(e["y_true"]))
        p_parts.append(np.asarray(e["y_prob"], dtype=float))
    return np.concatenate(y_parts), np.concatenate(p_parts)


def star(p):
    """Thesis convention (same as 15_/pca_sweep figures)."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runtag", default="pool_xgb",
                    help="run folder under runs/ that owns the outputs "
                         "(default: pool_xgb)")
    ap.add_argument("--out", default=None,
                    help="default: runs/<runtag>/results/pool_delong_pairs.tsv")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "runs" / args.runtag / "results" / "pool_delong_pairs.tsv")

    # pool arms live under the run folder; anchors are cross-campaign
    arms = dict(ARMS)
    pool_dir = PROJECT_ROOT / "runs" / args.runtag / "models" / "flat"
    for name, pat in list(arms.items()):
        if name.startswith(("esmc_", "esmif_")) and name not in (
                "esmc_pca80", "esmif_pca60"):
            arms[name] = pool_dir / pat.name

    print("Loading OOF vectors ...")
    data, y_ref = {}, None
    for name, pat in arms.items():
        got = load_oof(pat)
        if got is None:
            print(f"  WARN: no results for {name} yet ({pat.name})")
            continue
        y, p = got
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise SystemExit(f"{name}: OOF rows misaligned with other arms")
        data[name] = p
        print(f"  {name:<14} n={len(y)} loaded ({pat.parent.name})")

    rows = []
    for family, pairs in FAMILIES.items():
        avail = [(a, b) for a, b in pairs if a in data and b in data]
        for a, b in pairs:
            if not (a in data and b in data):
                print(f"  WARN [{family}]: skip {a}~{b} (missing arm)")
        if not avail:
            continue
        computed = []
        for a, b in avail:
            auc_a, auc_b, var_d = _p15.delong_test(y_ref, data[a], data[b])
            d = auc_a - auc_b
            se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
            # two-sided normal tail via erfc (exact, no scipy needed)
            p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
            computed.append((a, b, auc_a, auc_b, d, se, p))
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
            print(f"  {family:<9} {a} ~ {b}: d={d:+.4f} p={p:.2e} "
                  f"holm={ph:.2e} [{star(ph)}]")

    if not rows:
        raise SystemExit("no pairs computed — are the jobs finished?")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        hdr = list(rows[0])
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r[c] for c in hdr) + "\n")
    print(f"  wrote {out}  ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
