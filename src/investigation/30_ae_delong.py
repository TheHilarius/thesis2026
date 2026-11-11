#!/usr/bin/env python3
"""
30_ae_delong.py — pooled-OOF paired DeLong for the autoencoder capacity sweep.

Arms (ESM-C esmc300m_win_padtoken only; exact filenames because several
latents have e15/e30 twins with the same tag prefix):

    AE curve   runs/autoencoder_pilot/models/autoencoder/cv_results_xgb_*_autoencoder_lat*.json
               lat32/64/64e30/128/256/256e30/512/512e30/768/768e30/960
    var643     is_encodings_helping_v2/models/vardims/ ev80 var643 (0.7018,
               the benchmark — EV80 top-variance selection)

Families (Holm within family):
    ae_vs_ev80 every AE arm ~ var643          (the EV80 comparison)

Alignment: byte-equal y_true across all arms on pooled OOF fold-concat
(same checks as 21_dim21_delong.py); mismatch aborts, never forced.

Output: runs/autoencoder_pilot/results/ae_delong_pairs.tsv
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_p15 = _load("p15", SRC_DIR / "15_padmode_significance.py")
_d21 = _load("d21", SRC_DIR / "21_dim21_delong.py")
load_oof, star = _d21.load_oof, _d21.star

R = PROJECT_ROOT / "runs"
AE_DIR = R / "autoencoder_pilot" / "models" / "autoencoder"

# (arm, exact filename inside AE_DIR) — timestamps disambiguate e15/e30 twins
AE_ARMS = [
    ("ae32",     "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat32_20261029_092859.json"),
    ("ae64",     "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat64_20261026_103423.json"),
    ("ae64e30",  "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat64_20261029_112552.json"),
    ("ae128",    "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat128_20261029_092746.json"),
    ("ae256",    "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat256_20261104_171421.json"),
    ("ae256e30", "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat256_20261108_035553.json"),
    ("ae512",    "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat512_20261104_173923.json"),
    ("ae512e30", "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat512_20261108_025159.json"),
    ("ae768",    "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261107_211430.json"),
    ("ae768e30", "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261108_043730.json"),
    ("ae960",    "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat960_20261108_214006.json"),
]
REF_ARMS = [
    ("var643", R / "is_encodings_helping_v2/models/vardims/"
                   "cv_results_xgb_emb_esmc300m_win_padtoken_vardims_ev80_var643_*.json"),
]
FAMILIES = {
    "ae_vs_ev80": [(a, "var643") for a, _ in AE_ARMS],
}


def find_ref(pat: Path):
    hits = sorted(pat.parent.glob(pat.name)) if pat.parent.exists() else []
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=None,
                    help="default: runs/autoencoder_pilot/results/"
                         "ae_delong_pairs.tsv")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        R / "autoencoder_pilot" / "results" / "ae_delong_pairs.tsv")

    print("Loading OOF vectors ...")
    data, y_ref = {}, None
    for name, fname in AE_ARMS:
        p = AE_DIR / fname
        if not p.exists():
            print(f"  WARN: no results for {name} — skipped")
            continue
        y, pr = load_oof(p)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise SystemExit(f"ABORT: {name} OOF rows misaligned ({p.name})")
        data[name] = pr
        print(f"  {name:<10} n={len(y)}")
    for name, pat in REF_ARMS:
        hit = find_ref(pat)
        if hit is None:
            print(f"  WARN: no results for {name} — skipped")
            continue
        y, pr = load_oof(hit)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise SystemExit(f"ABORT: {name} OOF rows misaligned ({hit.name})")
        data[name] = pr
        print(f"  {name:<10} n={len(y)} ({hit.parent.parent.name}/{hit.name})")

    rows = []
    for family, pairs in FAMILIES.items():
        avail = [(a, b) for a, b in pairs if a in data and b in data]
        for a, b in pairs:
            if (a, b) not in avail:
                print(f"  WARN [{family}]: skip {a}~{b} (missing arm)")
        if not avail:
            continue
        computed = []
        for a, b in avail:
            auc_a, auc_b, var_d = _p15.delong_test(y_ref, data[a], data[b])
            d = auc_a - auc_b
            se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
            p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
            computed.append((a, b, auc_a, auc_b, d, se, p))
        adjs = _p15.holm_adjust([c[6] for c in computed])
        for (a, b, auc_a, auc_b, d, se, p), ph in zip(computed, adjs):
            rows.append({
                "arm_a": a, "arm_b": b,
                "auc_a": f"{auc_a:.6f}", "auc_b": f"{auc_b:.6f}",
                "delta": f"{d:.6f}", "se": f"{se:.6f}",
                "z": f"{(d / se if se else 0.0):.4f}",
                "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                "family": family, "star": star(ph),
            })
            print(f"  {family:<10} {a:>9} ~ {b:<7} d={d:+.4f} "
                  f"holm={ph:.2e} [{star(ph)}]")

    if not rows:
        raise SystemExit("no pairs computed — are the AE runs finished?")

    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = list(rows[0])
    with open(out, "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r[c] for c in hdr) + "\n")
    print(f"  wrote {out}  ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
