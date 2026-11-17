#!/usr/bin/env python3
"""
30_ae_barplot.py — full autoencoder experiment results, two figures
(25_varfold_barplot style, ESM-C esmc300m_win_padtoken only):

Fig 1 — ae_capacity_auc.png: latent x epoch GROUPED bars.
    x = latent (32, 64, 128, 256, 512, 768, 960=D); bars per epoch regime
    where data exists (e15 all; e30 lat64/256/512/768/960; 45+ lat768 only
    — e45/60/90 byte-identical, one rung). EV80 (var643, 0.7018) is the
    dashed benchmark line. Per-arm DeLong stars (Holm, thesis convention)
    above each AE bar, taken from 30_ae_delong.py's TSV.

Fig 2 — ae_epoch_curve.png: training-time saturation curve.
    x = epochs (15, 30, 45+); line for lat768 (all 3 rungs) and lat960
    (2 rungs); EV80 dashed line. Annotates that e45=e60=e90: early
    stopping (patience 3) converged, more epochs cannot help.

Values from cv_results JSONs (cv_summary.auc_roc mean/std); stars from
runs/autoencoder_pilot/results/ae_delong_pairs.tsv (30_ae_delong.py,
Holm-adjusted).

Output: runs/autoencoder_pilot/plots/ae_capacity_auc.{png,csv}
        runs/autoencoder_pilot/plots/ae_epoch_curve.{png,csv}
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

R = PROJECT_ROOT / "runs"
AE_DIR = R / "autoencoder_pilot" / "models" / "autoencoder"

# ESM-C color family, darkening with epochs
COL = {15: "#F2A3AB", 30: "#EE6677", 45: "#B23A47"}
EP_LABEL = {15: "15ep", 30: "30ep", 45: "45+ep\n(early-stopped)"}
EV80 = 0.7018

# arm -> exact path (timestamp disambiguates e15/e30 twins)
ARM = {
    "ae32":    AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat32_*.json",
    "ae64":    AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat64_20261026_*.json",
    "ae64e30": AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat64_20261029_*.json",
    "ae128":   AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat128_*.json",
    "ae256":   AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat256_20261104_*.json",
    "ae256e30": AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat256_20261108_*.json",
    "ae512":   AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat512_20261104_*.json",
    "ae512e30": AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat512_20261108_*.json",
    "ae768":   AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261107_*.json",
    "ae768e30": AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261108_*.json",
    "ae768e45": AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat768_20261112_*.json",
    "ae960":   AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat960_20261108_*.json",
    "ae960e30": AE_DIR / "cv_results_xgb_emb_esmc300m_win_padtoken_autoencoder_lat960_20261114_*.json",
}
# latent -> {epoch_regime: arm}; 45 covers e45/60/90 (byte-identical)
GROUPS = [
    ("AE32\n(928)",     {15: "ae32"}),
    ("AE64\n(1,856)",   {15: "ae64", 30: "ae64e30"}),
    ("AE128\n(3,712)",  {15: "ae128"}),
    ("AE256\n(7,424)",  {15: "ae256", 30: "ae256e30"}),
    ("AE512\n(14,848)", {15: "ae512", 30: "ae512e30"}),
    ("AE768\n(22,272)", {15: "ae768", 30: "ae768e30", 45: "ae768e45"}),
    ("AE960=D\n(27,840)", {15: "ae960", 30: "ae960e30"}),
]


def load_arm(pat):
    hits = sorted(pat.parent.glob(pat.name)) if pat.parent.exists() else []
    if not hits:
        return None
    d = json.loads(hits[-1].read_text())
    s = d["cv_summary"]["auc_roc"]
    return float(s["mean"]), float(s["std"])


def load_stars():
    out = {}
    p = R / "autoencoder_pilot" / "results" / "ae_delong_pairs.tsv"
    if not p.exists():
        return out
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["arm_a"]] = r["star"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/autoencoder_pilot/plots")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        R / "autoencoder_pilot" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading arms ...")
    vals = {}
    for name, pat in ARM.items():
        got = load_arm(pat)
        if got is None:
            print(f"  WARN: missing {name} — skipped")
            continue
        vals[name] = got
    stars = load_stars()
    for (lbl, eps) in GROUPS:
        for ep, arm in eps.items():
            m, s = vals[arm]
            print(f"  {arm:<10} {lbl.splitlines()[0]:>8} {ep}ep "
                  f"{m:.4f} +/- {s:.4f} [{stars.get(arm, '?')}]")

    # ── Fig 1: latent x epoch grouped bars ──
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x0 = 0
    for lbl, eps_map in GROUPS:
        eps = sorted(eps_map)
        n = len(eps)
        width = 0.72 / n
        for j, ep in enumerate(eps):
            arm = eps_map[ep]
            m, s = vals[arm]
            xi = x0 + (j + 0.5) * width
            ax.bar(xi, m, width=width * 0.92, color=COL[ep],
                   edgecolor="black", linewidth=0.6)
            ax.errorbar(xi, m, yerr=s, color="black", capsize=3, lw=0.8)
            ax.text(xi, m + s + 0.006, f"{m:.3f}", ha="center",
                    fontsize=8, fontweight="bold")
            if arm in stars:
                ax.text(xi, m + s + 0.030, stars[arm], ha="center",
                        va="bottom", fontsize=8, color="#B23A47")
        x0 += 1
    ax.axhline(EV80, color="black", lw=1.2, ls="--")
    ax.text(-0.44, EV80 + 0.006, "EV80 var643  0.7018", fontsize=9,
            fontweight="bold")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[0] for g in GROUPS], fontsize=9)
    ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Autoencoder latent capacity sweep by epoch regime "
                 "(ESM-C-300M, XGB)", fontsize=12)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=COL[15], edgecolor="black",
                      label="15 epochs"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COL[30], edgecolor="black",
                      label="30 epochs"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COL[45], edgecolor="black",
                      label="45+ epochs (early-stopped)"),
        plt.Line2D([0], [0], color="black", ls="--", lw=1.2,
                   label="EV80 var643 (benchmark, 0.7018)"),
    ], loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    png1 = out_dir / "ae_capacity_auc.png"
    fig.savefig(png1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(out_dir / "ae_capacity_auc.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "latent", "epoch", "auc_roc", "std"])
        for lbl, eps in GROUPS:
            for ep, arm in eps.items():
                m, s = vals[arm]
                w.writerow([arm, lbl.splitlines()[0], ep, f"{m:.6f}",
                            f"{s:.6f}"])
    print(f"  wrote {png1}")

    # ── Fig 2: epoch saturation curve ──
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    series = [
        ("AE768 (22,272 feats)", ESMC := "#EE6677",
         [("ae768", 15), ("ae768e30", 30), ("ae768e45", 45)]),
        ("AE960=D (27,840 feats)", "#4477AA",
         [("ae960", 15), ("ae960e30", 30)]),
    ]
    for lbl, color, pts in series:
        xs = [p[1] for p in pts]
        ys = [vals[p[0]][0] for p in pts]
        ss = [vals[p[0]][1] for p in pts]
        ax.errorbar(xs, ys, yerr=ss, marker="o", ms=6, capsize=4,
                    lw=1.6, color=color, label=lbl)
        # lat960 hugs lat768 — label below-right so the texts don't collide
        dy = -13 if "960" in lbl else 6
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(8, dy), fontsize=8)
    ax.axhline(EV80, color="black", lw=1.2, ls="--")
    ax.text(30.2, EV80 + 0.004, "EV80 var643 0.7018", fontsize=9,
            fontweight="bold")
    ax.set_xticks([15, 30, 45])
    ax.set_xticklabels(["15", "30", "45 (=60=90,\nearly-stopped)"])
    ax.set_xlabel("Max AE training epochs (early stop patience 3)")
    ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
    ax.set_ylim(0.6, 0.75)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("AE training-time saturation at latent 768 / 960",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    png2 = out_dir / "ae_epoch_curve.png"
    fig.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(out_dir / "ae_epoch_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "latent", "epoch", "auc_roc", "std"])
        for lbl, _, pts in series:
            for arm, ep in pts:
                m, s = vals[arm]
                w.writerow([arm, lbl.split()[0], ep, f"{m:.6f}", f"{s:.6f}"])
    print(f"  wrote {png2}")

    print("\n" + "=" * 60)
    print("  CAPTIONS (paste below each figure):")
    print("=" * 60)
    print("FIG 1: AE = plain (non-variational) 1x1-conv bottleneck "
          "autoencoder: shared per-position Linear encoder "
          "D->hidden->latent, masked-MSE reconstruction (pad slots "
          "excluded), trained fold-nested on train rows only; one bar per "
          "epoch regime with data (features = 29 x latent, labels). "
          "AE960=D is the no-compression control. EV80 var643 = "
          "fold-nested top-variance dims (80% EV, 18,647 feats), dashed "
          "benchmark line. Stars above bars: paired DeLong on pooled OOF "
          "(n=46,617) vs EV80, Holm-adjusted within family (* p<.05, "
          "** p<.01, *** p<.001) — every arm ***. No arm reaches EV80: "
          "best lat768-45ep d=-0.0067 (p=1.3e-06); even the "
          "no-compression control loses, so the reconstruction objective "
          "- not compression, not capacity, not training time - discards "
          "discriminative directions.")
    print("FIG 2: Training-time saturation. Early stopping (patience 3 on "
          "val masked-MSE) converges at the same epoch regardless of the "
          "cap: lat768 at 45/60/90 epochs is byte-identical (AUC 0.6950), "
          "so 'train longer' cannot close the gap to EV80 (0.7018). "
          "15->30 epochs gained +0.0065/+0.0051 (lat768/lat960); beyond "
          "30 the curve is flat.")
    print("=" * 60)


if __name__ == "__main__":
    main()
