#!/usr/bin/env python3
"""
30_ae_barplot.py — autoencoder capacity-sweep barplot with paired-DeLong
brackets (25_varfold_barplot style).

Bars (12, left to right, ESM-C esmc300m_win_padtoken only):
    ref:   EV80 var643 (18,647 feats, benchmark)
    AE:    lat32, lat64, lat64e30, lat128, lat256, lat256e30,
           lat512, lat512e30, lat768, lat768e30, lat960 (=D control)
Values from cv_results JSONs (cv_summary.auc_roc mean/std); brackets from
runs/autoencoder_pilot/results/ae_delong_pairs.tsv (30_ae_delong.py,
Holm-adjusted, thesis star convention).

Style: ESM-C #EE6677, refs grey, y 0-1, chance line 0.5, value labels.

Output: runs/autoencoder_pilot/plots/ae_capacity_auc.{png,csv}
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
AE_DIR = R / "autoencoder_pilot" / "models" / "autoencoder"
ESMC = "#EE6677"
REF_GRAY = "#999999"

# arm -> (glob/path, short label)
ARMS = {
    "var643": (R / "is_encodings_helping_v2/models/vardims/"
               "cv_results_xgb_emb_esmc300m_win_padtoken_vardims_ev80_var643_*.json",
               "EV80\nvar643 (18,647)"),
    "ae32": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat32_*.json",
             "AE32\n(928)"),
    "ae64": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat64_20261026_*.json",
             "AE64\n(1,856)"),
    "ae64e30": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat64_20261029_*.json",
                "AE64\n30ep\n(1,856)"),
    "ae128": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat128_*.json",
              "AE128\n(3,712)"),
    "ae256": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat256_20261104_*.json",
              "AE256\n(7,424)"),
    "ae256e30": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat256_20261108_*.json",
                 "AE256\n30ep\n(7,424)"),
    "ae512": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat512_20261104_*.json",
              "AE512\n(14,848)"),
    "ae512e30": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat512_20261108_*.json",
                 "AE512\n30ep\n(14,848)"),
    "ae768": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261107_*.json",
              "AE768\n(22,272)"),
    "ae768e30": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261108_*.json",
                 "AE768\n30ep\n(22,272)"),
    "ae960": (AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat960_*.json",
              "AE960=D\n(27,840)"),
}
ORDER = ["var643",
         "ae32", "ae64", "ae64e30", "ae128", "ae256", "ae256e30",
         "ae512", "ae512e30", "ae768", "ae768e30", "ae960"]
REFS = {"var643"}


def load_arm(pat):
    hits = sorted(pat.parent.glob(pat.name)) if pat.parent.exists() else []
    if not hits:
        return None
    d = json.loads(hits[-1].read_text())
    s = d["cv_summary"]["auc_roc"]
    return float(s["mean"]), float(s["std"]), hits[-1].name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pairs", default=None,
                    help="default: runs/autoencoder_pilot/results/"
                         "ae_delong_pairs.tsv")
    ap.add_argument("--out-prefix", default=None,
                    help="default: runs/autoencoder_pilot/plots/ae_capacity")
    args = ap.parse_args()
    pairs_tsv = Path(args.pairs) if args.pairs else (
        R / "autoencoder_pilot" / "results" / "ae_delong_pairs.tsv")
    out_prefix = Path(args.out_prefix) if args.out_prefix else (
        R / "autoencoder_pilot" / "plots" / "ae_capacity")

    print("Loading arms ...")
    vals, rows = {}, []
    for arm in ORDER:
        got = load_arm(ARMS[arm][0])
        if got is None:
            print(f"  WARN: missing {arm} — bar skipped")
            continue
        vals[arm] = got
        print(f"  {arm:<10} {got[0]:.4f} +/- {got[1]:.4f}")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    arm_x = {}
    for i, arm in enumerate(ORDER):
        if arm not in vals:
            continue
        mean, std, _ = vals[arm]
        color = REF_GRAY if arm in REFS else ESMC
        hatch = "//" if arm in REFS else None
        ax.bar(i, mean, yerr=std, capsize=4, width=0.62, color=color,
               edgecolor="black", linewidth=0.6, hatch=hatch)
        ax.text(i, mean + std + 0.006, f"{mean:.3f}",
                ha="center", fontsize=8.5, fontweight="bold")
        arm_x[arm] = i
        rows.append({"arm": arm, "auc_roc": f"{mean:.6f}",
                     "std": f"{std:.6f}"})

    xs = sorted(arm_x.values())
    ax.set_xticks(xs)
    ax.set_xticklabels([ARMS[a][1] for a in ORDER if a in arm_x], fontsize=8)
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
        Patch(facecolor=REF_GRAY, hatch="//", edgecolor="black",
              label="EV80 variance selection (benchmark)"),
        Patch(facecolor=ESMC, edgecolor="black",
              label="1x1-conv AE (unsupervised)"),
    ], loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("Autoencoder capacity sweep vs EV80 variance selection "
                 "(ESM-C-300M, XGB)", fontsize=12)
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
    print("AE = plain (non-variational) 1x1-conv bottleneck autoencoder: "
          "shared per-position Linear encoder D->hidden->latent, masked-MSE "
          "reconstruction (pad slots excluded), trained fold-nested on train "
          "rows only; 15 epochs unless marked 30ep. Features = 29 x latent "
          "(label). AE960=D is the no-compression control (latent equals "
          "embedding dim; only the non-linear transform remains). EV80 "
          "var643 = fold-nested top-variance dims "
          "(80% EV, 18,647 feats), the benchmark. Brackets: paired DeLong "
          "on pooled OOF "
          "(n=46,617), Holm-adjusted within family (* p<.05, ** p<.01, "
          "*** p<.001). No AE arm matches EV80 (best lat768-30ep: d=-0.008, "
          "***); even the no-compression control loses (d=-0.014, ***), "
          "showing the reconstruction objective — not compression — discards "
          "discriminative directions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
