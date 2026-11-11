#!/usr/bin/env python3
"""
32_flank5_significance.py — paired significance tests: flank5 (19mer,
5-aa flanks) vs the 29mer (10-aa flanks) originals.

Rationale: the differential logo showed weak signal at the terminal
positions furthest from the peptide, so the flank5 dataset trims those
positions. Question: does the trim help, hurt, or do nothing?

Four pairs — one Holm family ("flank5_vs_29mer"), each 29mer vs flank5
on IDENTICAL OOF rows (the flank5 CSVs were derived from the 29mer split
with peptide/label/fold byte-identical — asserted below against
df_all_with_folds.csv):

    blosum50              encodings_sweep          vs flank5_compare
    handcrafted_blosum50  handcrafted_sweep        vs flank5_compare
    emb_esmc300m_win_padtoken dimselect top643 (supervised per-dim
                          screening — replaced the EV80 vardims pair
                          2026-11-11; pure embedding, no handcrafted)
                          dimselect_matched      vs dimselect_flank5
    cnn_simple_k5 (ESM-C windows, mode=cnn_cv)
                          flank5_compare/cnn/full  vs cnn/flank5
                          (rerun batch — both arms from the same batch)

Tests per pair (protocol identical to 15_padmode_significance.py):
  - paired DeLong on pooled OOF (primary; Holm within the family)
  - paired row-level bootstrap B=10000 (per-pair seeded RNG)
  - fold-paired t-test on the 6 outer-fold AUCs + Nadeau-Bengio variant

DeLong / bootstrap / Holm are reused from 15_padmode_significance.py.

Output: runs/flank5_compare/results/paired_tests.csv (+ stdout table).

Usage:
    python src/investigation/32_flank5_significance.py
    python src/investigation/32_flank5_significance.py --skip-bootstrap
"""

import argparse
import csv
import importlib.util
import json
import sys
from math import erfc, sqrt
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT, SPLIT_DATA_PATH  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import t as t_dist, ttest_rel  # noqa: E402

# reuse DeLong / bootstrap / Holm from the padmode significance script
_spec = importlib.util.spec_from_file_location(
    "p15", SRC_DIR / "15_padmode_significance.py")
_p15 = importlib.util.module_from_spec(_spec)
sys.modules["p15"] = _p15
_spec.loader.exec_module(_p15)

R = PROJECT_ROOT / "runs"
# Nadeau-Bengio factor sqrt(1 + k*n_test/n_train) for 6-fold (1/5) -> sqrt(2.2)
NB_FACTOR = np.sqrt(2.2)
NB_DF = 5

# arm name -> cv_results JSON glob (newest match wins)
ARMS = {
    "blosum50_29":
        R / "encodings_sweep/models/flat/"
            "cv_results_xgb_blosum50_flat_*.json",
    "blosum50_19":
        R / "flank5_compare/models/flat/"
            "cv_results_xgb_blosum50_flat_*.json",
    "hc_blosum50_29":
        R / "handcrafted_sweep/models/flat/"
            "cv_results_xgb_handcrafted_blosum50_flat_*.json",
    "hc_blosum50_19":
        R / "flank5_compare/models/flat/"
            "cv_results_xgb_handcrafted_blosum50_flat_*.json",
    "esmc_dimselect_29":
        R / "dimselect_matched/models/dimselect/"
            "cv_results_xgb_emb_esmc300m_win_padtoken_"
            "dimselect_top643_xgb_*.json",
    "esmc_dimselect_19":
        R / "dimselect_flank5/models/dimselect/"
            "cv_results_xgb_emb_esmc300m_win_padtoken_"
            "dimselect_top643_xgb_*.json",
    # CNN: rerun batch, BOTH arms from the same job pair (cleanest pair —
    # no cross-campaign concern). mode=cnn_cv, model_key=cnn_simple_k5.
    "cnn_k5_29":
        R / "flank5_compare/cnn/full/models/cnn/"
            "cv_results_cnn_simple_k5_*.json",
    "cnn_k5_19":
        R / "flank5_compare/cnn/flank5/models/cnn/"
            "cv_results_cnn_simple_k5_*.json",
}
# one Holm family: flank5 vs its 29mer counterpart, per feature set
FAMILIES = {
    "flank5_vs_29mer": [
        ("blosum50_29", "blosum50_19"),
        ("hc_blosum50_29", "hc_blosum50_19"),
        ("esmc_dimselect_29", "esmc_dimselect_19"),
        ("cnn_k5_29", "cnn_k5_19"),
    ],
}


def load_oof(pattern: Path):
    """(y, p, fold_aucs) pooled from the newest matching cv_results JSON.

    y is concatenated in fold order 0..5 — the same order for every arm,
    which is what makes the DeLong pairing valid.
    """
    files = sorted(pattern.parent.glob(pattern.name)) if pattern.parent.exists() else []
    if not files:
        return None
    res = json.loads(files[-1].read_text())
    if "outer_val_predictions" not in res:
        raise SystemExit(
            f"{files[-1].name}: no outer_val_predictions — cannot pair")
    folds = sorted(int(k) for k in res["outer_val_predictions"])
    y_parts, p_parts = [], []
    for fk in folds:
        e = res["outer_val_predictions"][str(fk)]
        y_parts.append(np.asarray(e["y_true"]))
        p_parts.append(np.asarray(e["y_prob"], dtype=float))
    fold_aucs = [m["auc_roc"] for m in res["outer_val_metrics"]]
    return np.concatenate(y_parts), np.concatenate(p_parts), fold_aucs


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "runs/flank5_compare/results/paired_tests.csv"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-bootstrap", action="store_true",
                    help="faster dev run (DeLong + fold-t still run)")
    args = ap.parse_args()
    out = Path(args.out)

    # ── split labels/folds: ground truth for the alignment assert ──
    split = pd_read_split()
    y_split, fold_vec = split

    # ── load arms ──
    print("Loading OOF vectors ...")
    data, fold_aucs, y_ref = {}, {}, None
    for name, pat in ARMS.items():
        got = load_oof(pat)
        if got is None:
            print(f"  WARN: no results for {name} ({pat.name}) — skipped")
            continue
        y, p, fa = got
        # alignment: concatenated-in-fold-order y_true must equal the
        # split labels in the same order (labels AND fold sizes AND order)
        expect = np.concatenate([y_split[fold_vec == fk]
                                 for fk in sorted(set(fold_vec))])
        if not np.array_equal(y, expect):
            raise SystemExit(f"{name}: OOF rows misaligned with the split")
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise SystemExit(f"{name}: OOF rows misaligned with other arms")
        data[name] = p
        fold_aucs[name] = fa
        print(f"  {name:<16} n={len(y)} loaded ({pat.parent.parent.parent.name})")

    rows = []
    for family, pairs in FAMILIES.items():
        computed = []
        for a, b in pairs:
            if a not in data or b not in data:
                print(f"  WARN [{family}]: skip {a} ~ {b} (missing arm)")
                continue
            # DeLong: delta = auc(a) - auc(b) with a = 29mer, b = flank5;
            # report delta as flank5 - 29mer (positive = trim helps)
            auc_29, auc_19, var_d = _p15.delong_test(y_ref, data[a], data[b])
            d = auc_19 - auc_29
            se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
            p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
            z = d / se if se > 0 else 0.0
            ci_lo, ci_hi = d - 1.96 * se, d + 1.96 * se

            # paired row bootstrap (per-pair RNG: family+pair keyed)
            b_lo = b_hi = p_boot = float("nan")
            if not args.skip_bootstrap:
                rng = _p15.per_pair_rng(args.seed, family, a, b)
                _d, b_lo, b_hi, p_boot, _, _, _bs = _p15.paired_bootstrap_delta(
                    y_ref, data[b], data[a], args.B, rng)  # b-a = flank5-29
                assert abs(_d - d) < 1e-9, (_d, d)

            # fold-paired t on the 6 outer-fold AUCs (+ Nadeau-Bengio)
            fa29, fa19 = fold_aucs[a], fold_aucs[b]
            t_stat, p_t = ttest_rel(fa19, fa29)
            t_nb = t_stat / NB_FACTOR
            p_nb = float(2 * t_dist.sf(abs(t_nb), df=NB_DF))

            computed.append(dict(
                arm_a=a, arm_b=b,
                auc_29=auc_29, auc_19=auc_19, delta=d, se=se, z=z,
                ci95_lo=ci_lo, ci95_hi=ci_hi, p_delong=p,
                ci_lo=b_lo, ci_hi=b_hi, p_boot=p_boot,
                t_stat=float(t_stat), p_ttest=float(p_t),
                t_stat_nb=float(t_nb), p_ttest_nb=p_nb))
            print(f"  {family:<18} {a} ~ {b}: Δ={d:+.4f} "
                  f"DeLong p={p:.2e} boot p={p_boot:.2e} t p={p_t:.3f}")

        if not computed:
            continue
        holm_d = _p15.holm_adjust([c["p_delong"] for c in computed])
        holm_b = _p15.holm_adjust([c["p_boot"] for c in computed])
        holm_t = _p15.holm_adjust([c["p_ttest_nb"] for c in computed])
        for c, hd, hb, ht in zip(computed, holm_d, holm_b, holm_t):
            c["p_delong_holm"] = hd
            c["p_boot_holm"] = hb
            c["p_ttest_nb_holm"] = ht
            c["star"] = star(hd)
            c["family"] = family
            rows.append(c)

    if not rows:
        raise SystemExit("no pairs computed — are the flank5 jobs finished?")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["family", "arm_a", "arm_b",
            "auc_29", "auc_19", "delta", "se", "z", "ci95_lo", "ci95_hi",
            "p_delong", "p_delong_holm",
            "ci_lo", "ci_hi", "p_boot", "p_boot_holm",
            "t_stat", "p_ttest", "t_stat_nb", "p_ttest_nb",
            "p_ttest_nb_holm", "star"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])
                        for k in cols})
    print(f"\n  wrote {out}  ({len(rows)} pairs)")

    # ── headline table ──
    print("\n  flank5 vs 29mer (Δ = flank5 − 29mer):")
    for r in rows:
        print(f"    {r['arm_a'].rsplit('_', 1)[0]:<16} "
              f"ΔAUC={r['delta']:+.4f}  DeLong p={r['p_delong']:.3g} "
              f"(Holm {r['p_delong_holm']:.3g}) [{r['star']}]")


def pd_read_split():
    """(labels, fold vector) from the split CSV — the alignment ground truth."""
    import pandas as pd
    df = pd.read_csv(SPLIT_DATA_PATH, usecols=["label", "fold"])
    return df["label"].to_numpy(), df["fold"].to_numpy()


if __name__ == "__main__":
    main()
