#!/usr/bin/env python3
"""
25_varfold_barplot.py — varfold comparison barplot with paired-DeLong
brackets (dim21/padmode style).

Bars (12, left to right), per embedding family:
    global21, varfold21, varfold50, dimselect50, flat-PCA60, flat-PCA90
Values read from the arms' cv_results JSONs (cv_summary.auc_roc.mean/std);
brackets + stars from runs/varfold_pilot/results/varfold_delong_pairs.tsv
(24_varfold_delong.py output, Holm-adjusted, thesis star convention).

NOTES baked into the terminal caption (not the PNG):
  - varfold21 == global21 by construction here: fold-nested ranking gave
    Jaccard 1.000 vs global in all 240 fold-rankings (selection stable).
  - ESM-IF dimselect/PCA arms run on eos/bos-repeat; varfold/global on
    eosrepeat (same rows/folds, different pad-mode embedding).

Style: ESM-C #EE6677 / ESM-IF #4477AA (pca-sweep colors), y-axis 0-1,
chance line at 0.5, short labels, value labels, DeLong brackets stacked.

Output: runs/varfold_pilot/plots/varfold_vs_refs_auc.{png,csv}
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

R = PROJECT_ROOT / "runs"
COLORS = {"esmc": "#EE6677", "esmif": "#4477AA"}

ARMS = {  # arm -> (glob, short label)
    "esmc_global21": ("simple21_pilot/models/flat_raw/"
                      "cv_results_*_esmc300m_win_padtoken_d21_var_flat_raw_*.json",
                      "global21"),
    "esmc_vf21": ("varfold_pilot/models/varfold/"
                  "cv_results_*_esmc300m_win_padtoken_varfold_top21_*.json",
                  "varfold21"),
    "esmc_vf50": ("varfold_pilot/models/varfold/"
                  "cv_results_*_esmc300m_win_padtoken_varfold_top50_*.json",
                  "varfold50"),
    "esmc_dimselect": ("dimselect_sweep/models/*/cv_results_*esmc300m*dimselect*"
                       "top50*xgb*.json", "dimselect50"),
    "esmc_pca60": ("esmc300m_pca_sweep/models/flat/"
                   "cv_results_*_esmc300m_win_padtoken_pca291_*.json",
                   "PCA60\n(k=291)"),
    "esmc_pca90": ("esmc300m_pca_sweep/models/flat/"
                   "cv_results_*_esmc300m_win_padtoken_pca5883_*.json",
                   "PCA90\n(k=5883)"),
    "esmif_global21": ("simple21_pilot/models/flat_raw/"
                       "cv_results_*_esmif_eosrepeat_d21_var_flat_raw_*.json",
                       "global21"),
    "esmif_vf21": ("varfold_pilot/models/varfold/"
                   "cv_results_*_esmif_eosrepeat_varfold_top21_*.json",
                   "varfold21"),
    "esmif_vf50": ("varfold_pilot/models/varfold/"
                   "cv_results_*_esmif_eosrepeat_varfold_top50_*.json",
                   "varfold50"),
    "esmif_dimselect": ("dimselect_sweep/models/*/cv_results_*esmif*dimselect*"
                        "top50*xgb*.json", "dimselect50"),
    "esmif_pca60": ("esmif_eos_sweep/models/flat/"
                    "cv_results_*_esmif_eos_bos_repeat_pca632_*.json",
                    "PCA60\n(k=632)"),
    "esmif_pca90": ("padmode_choice/models/flat/"
                    "cv_results_*_esmif_eos_bos_repeat_pca4308_*.json",
                    "PCA90\n(k=4308)"),
}
ORDER = ["esmc_global21", "esmc_vf21", "esmc_vf50", "esmc_dimselect",
         "esmc_pca60", "esmc_pca90",
         "esmif_global21", "esmif_vf21", "esmif_vf50", "esmif_dimselect",
         "esmif_pca60", "esmif_pca90"]


def load_arm(rel_glob):
    hits = sorted(R.glob(rel_glob))
    if not hits:
        return None
    d = json.loads(hits[-1].read_text())
    s = d["cv_summary"]["auc_roc"]
    return float(s["mean"]), float(s["std"]), hits[-1].name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pairs", default=None,
                    help="varfold_delong_pairs.tsv (default: runs/varfold_pilot/"
                         "results/varfold_delong_pairs.tsv)")
    ap.add_argument("--out-prefix", default=None,
                    help="default: runs/varfold_pilot/plots/varfold_vs_refs")
    args = ap.parse_args()

    pairs_tsv = Path(args.pairs) if args.pairs else (
        R / "varfold_pilot" / "results" / "varfold_delong_pairs.tsv")
    out_prefix = Path(args.out_prefix) if args.out_prefix else (
        R / "varfold_pilot" / "plots" / "varfold_vs_refs")

    print("Loading arms ...")
    vals, rows = {}, []
    for arm in ORDER:
        got = load_arm(ARMS[arm][0])
        if got is None:
            print(f"  WARN: missing {arm} ({ARMS[arm][0]}) — bar skipped")
            continue
        vals[arm] = got
        print(f"  {arm:<18} {got[0]:.4f} +/- {got[1]:.4f}")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    arm_x = {}
    for i, arm in enumerate(ORDER):
        if arm not in vals:
            continue
        mean, std, _ = vals[arm]
        fam = arm.split("_")[0]
        ax.bar(i, mean, yerr=std, capsize=4, width=0.62,
               color=COLORS[fam], edgecolor="black", linewidth=0.6)
        ax.text(i, mean + std + 0.006, f"{mean:.3f}",
                ha="center", fontsize=9, fontweight="bold")
        arm_x[arm] = i
        rows.append({"arm": arm, "auc_roc": f"{mean:.6f}",
                     "std": f"{std:.6f}"})

    xs = sorted(arm_x.values())
    ax.set_xticks(xs)
    ax.set_xticklabels([ARMS[a][1] for a in ORDER if a in arm_x], fontsize=9)
    ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
    ax.grid(axis="y", alpha=0.3)

    # ── DeLong brackets (stacked per family) ──
    if pairs_tsv.exists():
        with open(pairs_tsv) as fh:
            pairs = list(csv.DictReader(fh, delimiter="\t"))
        data_top = max(m + s for m, s, _ in vals.values())
        base, step, fam_lvl = data_top + 0.02, 0.016, {}
        for r in pairs:
            a, b = r["arm_a"], r["arm_b"]
            if a not in arm_x or b not in arm_x:
                print(f"  WARN: pair {a}~{b} has no bar — skipped")
                continue
            lvl = fam_lvl.get(r["family"], 0)
            fam_lvl[r["family"]] = lvl + 1
            y = base + lvl * step
            xa, xb = sorted((arm_x[a], arm_x[b]))
            ax.plot([xa, xa, xb, xb], [y - 0.004, y, y, y - 0.004],
                    lw=0.8, color="black")
            ax.text((xa + xb) / 2, y + 0.002, r["star"],
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.legend(handles=[
        Patch(facecolor=COLORS["esmc"], label="ESM-C-300M padtoken"),
        Patch(facecolor=COLORS["esmif"],
              label="ESM-IF (dimselect/PCA eos-bos, others eosrepeat)"),
    ], loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("varfold (fold-nested variance dims) vs global/PCA/dimselect",
                 fontsize=12)
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(out_prefix) + "_auc.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    csvp = Path(str(out_prefix) + "_auc.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "auc_roc", "std"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {png}")
    print(f"  wrote {csvp}")

    print("\n" + "=" * 60)
    print("  CAPTION (paste below the figure):")
    print("=" * 60)
    print("varfold = top-N dims by per-dim variance ranking recomputed on "
          "train rows only (fold-nested, leakage-free), gathered raw over all "
          "29 positions (29 x N features, no PCA). global21 = same ranking "
          "computed once on all rows (simple21 --dims var21). dimselect50 = "
          "fold-nested supervised per-dim XGB screening (top-50). PCA60/90 = "
          "flat window-PCA at 60/90% EV. ESM-IF dimselect/PCA arms run on "
          "eos/bos-repeat; varfold/global arms on eosrepeat. Brackets: paired "
          "DeLong on pooled OOF, Holm-adjusted within family "
          "(* p<.05, ** p<.01, *** p<.001). Fold-nested vs global selection "
          "was identical (Jaccard 1.000 across 240 fold-rankings).")
    print("=" * 60)


if __name__ == "__main__":
    main()
