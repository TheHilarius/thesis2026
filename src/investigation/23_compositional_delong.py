#!/usr/bin/env python3
"""
23_compositional_delong.py — paired DeLong tests for the compositional
barplots (22_compositional_barplot.py).

Compares pooled nested-CV OOF predictions (cv_results_*.json,
outer_val_predictions) between the three bars:

    family compositional (Holm over all 3 pairs):
      compositional ~ blosum50
      handcrafted_compositional ~ blosum50
      compositional ~ handcrafted_compositional

DeLong implementation reused from 15_padmode_significance.py (same
estimand: paired DeLong on pooled OOF, Holm correction per family).
blosum50 comes from runs/encodings_sweep — the alignment guard aborts
if the campaigns' OOF rows do not line up.

Output: runs/<runtag>/results/compositional_delong_pairs.tsv
(default runtag try_out_compositional; --runtag to override)
    arm_a arm_b auc_a auc_b delta se z p_delong p_holm family star

Then (brackets picked up automatically by 22_):
    python src/investigation/22_compositional_barplot.py
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
ARMS = {
    "compositional": R / "try_out_compositional/models/flat/"
                        "cv_results_xgb_compositional_flat_*.json",
    "handcrafted_compositional": R / "try_out_compositional/models/flat/"
                                     "cv_results_xgb_handcrafted_compositional_flat_*.json",
    "blosum50": R / "encodings_sweep/models/flat/"
                    "cv_results_xgb_blosum50_flat_*.json",
}
FAMILIES = {
    "compositional": [
        ("compositional", "blosum50"),
        ("handcrafted_compositional", "blosum50"),
        ("compositional", "handcrafted_compositional"),
    ],
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
    ap.add_argument("--runtag", default="try_out_compositional",
                    help="run folder under runs/ that owns the outputs "
                         "(default: try_out_compositional)")
    ap.add_argument("--out", default=None,
                    help="default: runs/<runtag>/results/compositional_delong_pairs.tsv")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "runs" / args.runtag / "results" / "compositional_delong_pairs.tsv")

    print("Loading OOF vectors ...")
    data, y_ref = {}, None
    for name, pat in ARMS.items():
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
        print(f"  {name:<26} n={len(y)} loaded ({pat.parent.name})")

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
            print(f"  {family:<13} {a} ~ {b}: d={d:+.4f} p={p:.2e} "
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
