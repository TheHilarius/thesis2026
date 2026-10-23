#!/usr/bin/env python3
"""
24_varfold_delong.py — pooled-OOF paired DeLong for the varfold pilot.

Arms (globs; missing arms are skipped with a warning so the script can be
re-run as campaigns land):

    varfold      runs/varfold_pilot/models/varfold/cv_results_*_varfold_top{N}_*.json
    global21     runs/simple21_pilot/models/flat_raw/*_d21_var_flat_raw_*.json
                 (global top-21 variance dims, physical reduced files)
    pca60/90     esmc: pca_sweep esmc300m k=291/5883 (60/90% EV)
                 esmif: esmif_eos_sweep k=632 (60% EV) / k=4308 (90% EV, searched)
    dimselect    recursive search: runs/**/cv_results_*dimselect*.json

Families (Holm within family):
    esmc   {vf21,vf50} x {global21, pca60, pca90, dimselect}
    esmif  {vf21,vf50} x {global21, pca60, pca90, dimselect}

Alignment: identical cross-campaign checks as 21_dim21_delong.py — OOF
fold-concat in numeric fold order, y_true must be byte-equal across arms;
mismatch aborts with the offending arm named (never forced).

KNOWN ARM MISMATCH (carried in the `note` column): ESM-IF PCA and dimselect
reference arms run on eos/bos-repeat, varfold/global21 on eosrepeat — same
rows/folds, different pad-mode embedding. ESM-C arms are all padtoken.

Output: runs/varfold_pilot/results/varfold_delong_pairs.tsv
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


# DeLong + Holm from the padmode significance script
_p15 = _load("p15", SRC_DIR / "15_padmode_significance.py")
# pooled-OOF loader + thesis star convention from the dim21 DeLong script
_d21 = _load("d21", SRC_DIR / "21_dim21_delong.py")
load_oof, star = _d21.load_oof, _d21.star

R = PROJECT_ROOT / "runs"
VF = "varfold_pilot/models/varfold/cv_results_xgb_emb_{}_varfold_top{}_*.json"

ARMS = {
    # ESM-C-300M padtoken
    "esmc_vf21": [VF.format("esmc300m_win_padtoken", 21)],
    "esmc_vf50": [VF.format("esmc300m_win_padtoken", 50)],
    "esmc_global21": ["simple21_pilot/models/flat_raw/"
                      "cv_results_xgb_emb_esmc300m_win_padtoken_d21_var_flat_raw_*.json"],
    "esmc_pca60": ["esmc300m_pca_sweep/models/flat/"
                   "cv_results_xgb_emb_esmc300m_win_padtoken_pca291_flat_*.json"],
    "esmc_pca90": ["esmc300m_pca_sweep/models/flat/"
                   "cv_results_xgb_emb_esmc300m_win_padtoken_pca5883_flat_*.json"],
    # ESM-IF (eos/bos-repeat for PCA refs — see note column)
    "esmif_vf21": [VF.format("esmif_eosrepeat", 21)],
    "esmif_vf50": [VF.format("esmif_eosrepeat", 50)],
    "esmif_global21": ["simple21_pilot/models/flat_raw/"
                       "cv_results_xgb_emb_esmif_eosrepeat_d21_var_flat_raw_*.json"],
    "esmif_pca60": ["esmif_eos_sweep/models/flat/"
                    "cv_results_xgb_emb_esmif_eos_bos_repeat_pca632_flat_*.json"],
    "esmif_pca90": ["esmif_eos_sweep/models/flat/"
                    "cv_results_xgb_emb_esmif_eos_bos_repeat_pca4308_flat_*.json",
                    "padmode_choice/models/flat/"
                    "cv_results_xgb_emb_esmif_eos_bos_repeat_pca4308_flat_*.json"],
    # dimselect — friend's campaign, location discovered recursively
    "esmc_dimselect": ["**/cv_results_*dimselect*esmc300m*.json",
                       "**/cv_results_*esmc300m*dimselect*.json"],
    "esmif_dimselect": ["**/cv_results_*dimselect*esmif*.json",
                        "**/cv_results_*esmif*dimselect*.json"],
}
REFS = ["global21", "pca60", "pca90", "dimselect"]
FAMILIES = {
    "esmc": [(f"esmc_vf{n}", f"esmc_{r}") for n in (21, 50) for r in REFS],
    "esmif": [(f"esmif_vf{n}", f"esmif_{r}") for n in (21, 50) for r in REFS],
}
NOTES = {
    "esmif_pca60": "eos_bos_repeat embedding (not eosrepeat)",
    "esmif_pca90": "eos_bos_repeat embedding (not eosrepeat)",
    "esmif_dimselect": "eos_bos_repeat embedding (not eosrepeat)",
}


def find_json(rel):
    for pat in rel:
        base = PROJECT_ROOT if pat.startswith("**") else R
        hits = sorted(base.glob(pat))
        if hits:
            return hits[-1]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=None,
                    help="default: runs/varfold_pilot/results/"
                         "varfold_delong_pairs.tsv")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        R / "varfold_pilot" / "results" / "varfold_delong_pairs.tsv")

    print("Loading OOF vectors ...")
    data, y_ref = {}, None
    for name, pats in ARMS.items():
        hit = None
        for pat in pats:
            hit = find_json([pat])
            if hit:
                break
        if hit is None:
            print(f"  WARN: no results for {name} yet — skipped")
            continue
        got = load_oof(hit)
        y, p = got
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise SystemExit(
                f"ABORT: {name} OOF rows misaligned with earlier arms "
                f"({hit.name}) — DeLong would compare different peptides; "
                f"not forcing the test")
        data[name] = p
        print(f"  {name:<16} n={len(y)} ({hit.parent.parent.parent.name}/"
              f"{hit.parent.name})")

    rows = []
    for family, pairs in FAMILIES.items():
        avail, missing = [], []
        for a, b in pairs:
            if a in data and b in data:
                avail.append((a, b))
            else:
                missing.append((a, b))
        for a, b in missing:
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
            note = "; ".join(n for n in (NOTES.get(a), NOTES.get(b)) if n)
            rows.append({
                "arm_a": a, "arm_b": b,
                "auc_a": f"{auc_a:.6f}", "auc_b": f"{auc_b:.6f}",
                "delta": f"{d:.6f}", "se": f"{se:.6f}",
                "z": f"{(d / se if se else 0.0):.4f}",
                "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                "family": family, "star": star(ph), "note": note,
            })
            print(f"  {family:<6} {a:>16} ~ {b:<16} d={d:+.4f} "
                  f"holm={ph:.2e} [{star(ph)}]")

    if not rows:
        raise SystemExit("no pairs computed — are the varfold runs finished?")

    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = list(rows[0])
    with open(out, "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(r[c] for c in hdr) + "\n")
    print(f"  wrote {out}  ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
