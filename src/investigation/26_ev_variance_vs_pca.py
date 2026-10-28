#!/usr/bin/env python3
"""
26_ev_variance_vs_pca.py — per-embedding bars: topX-variance vs flat-PCA at
each explained-variance threshold, with paired-DeLong brackets.

One figure per embedding (runs/ev_matched_topXvariance_sweep/plots/):
    ev_variance_vs_pca_auc_<emb_key>.png
x-axis = EV% ladder (30..90); two bars per tick:
    solid  = flat window-PCA (rotation) at that EV
    hatched= topX-variance (raw dims by cumulative dim-variance, no PCA)
Y-axis 0-1. Brackets: paired DeLong on pooled OOF per EV rung, Holm within
the 10-rung family per embedding, thesis star convention (* p<.05, ** p<.01,
*** p<.001, ns).

Sources:
    variance  runs/ev_matched_topXvariance_sweep/models/simple21/cv_results
              *_simple21_var{X}_*.json  (X = n_dims from dim_variance ev JSON)
    PCA       campaign rows in runs/pca_sweep_delong/pca_sweep_results.csv
              resolved to their cv_results JSONs (esmif 90% = padmode_choice)

NOTE on comparability (kept in terminal caption): the EV ladder matches
labels, not budgets — PCA's EV comes from the covariance spectrum (rotation,
optimal variance concentration), ours from the sum of per-dim variances
(diagonal only, no mixing), so equal EV% never means equal feature counts.

Outputs:
    .../plots/ev_variance_vs_pca_auc_<emb>.png   (2 figures)
    .../results/ev_variance_vs_pca_delong.tsv    (10 rungs x 2 embeddings)
    .../plots/ev_variance_vs_pca_auc.csv         (values incl. delta/star)
Caption printed to terminal (never burned into the PNG).
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
from config import PROJECT_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = PROJECT_ROOT / "runs"
EVS = [30, 40, 50, 60, 65, 70, 75, 80, 85, 90]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_p15 = _load("p15", SRC_DIR / "15_padmode_significance.py")
_d21 = _load("d21", SRC_DIR / "21_dim21_delong.py")
load_oof, star = _d21.load_oof, _d21.star

FAM = {  # emb_key -> (pca campaign, family color, pca feature token)
    "esmc300m_win_padtoken": ("esmc300m_pca_sweep", "#EE6677",
                              "esmc300m_win_padtoken"),
    "esmif_eosrepeat": ("esmif_eos_sweep", "#4477AA",
                        "esmif_eos_bos_repeat"),
}


def pca_k(campaign):
    """EV -> k from the DeLong campaign table."""
    out = {}
    for r in csv.DictReader(open(R / "pca_sweep_delong/pca_sweep_results.csv")):
        if r["campaign"] == campaign and int(r["ev_pct"]) in EVS:
            out[int(r["ev_pct"])] = int(r["k"])
    return out


def resolve_json(patterns):
    for pat in patterns:
        hits = sorted(R.glob(pat))
        if hits:
            return hits[-1]
    return None


def holm_p(pvals):
    return _p15.holm_adjust(pvals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/ev_matched_topXvariance_sweep")
    args = ap.parse_args()
    root = Path(args.out_dir) if args.out_dir else (
        R / "ev_matched_topXvariance_sweep")
    plot_dir = root / "plots"
    res_dir = root / "results"
    plot_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    delong_rows, csv_rows = [], []

    for key, (campaign, color, pca_tok) in FAM.items():
        print(f"\n=== {key} ===")
        # ── resolve arms: per EV, one variance JSON + one PCA JSON ──
        arms = {}  # ev -> {"var": (path, mean, std, n_dims), "pca": (...)}
        for ev in EVS:
            ej = PROJECT_ROOT / "results" / "tables" / f"dim_variance_{key}_ev{ev}.json"
            if not ej.exists():
                print(f"  WARN: missing {ej.name} — rung skipped")
                continue
            n_dims = json.loads(ej.read_text())["n_dims"]
            vp = resolve_json([f"ev_matched_topXvariance_sweep/models/simple21/"
                               f"cv_results_xgb_emb_{key}_simple21_var{n_dims}_*.json"])
            kk = pca_k(campaign).get(ev)
            pats = ([f"{campaign}/models/flat/"
                     f"cv_results_xgb_emb_{pca_tok}_pca{kk}_*.json"]
                    if ev != 90 or key != "esmif_eosrepeat"
                    else ["padmode_choice/models/flat/"
                          "cv_results_xgb_emb_esmif_eos_bos_repeat_pca4308_*.json",
                          f"{campaign}/models/flat/"
                          f"cv_results_xgb_emb_{pca_tok}_pca{kk}_*.json"])
            pp = resolve_json(pats)
            if vp is None or pp is None:
                print(f"  WARN: ev{ev}: var={'ok' if vp else 'MISSING'} "
                      f"pca={'ok' if pp else 'MISSING'} — rung skipped")
                continue
            vs = json.loads(vp.read_text())["cv_summary"]["auc_roc"]
            # PCA means/stds: prefer campaign table (canonical), OOF from JSON
            prow = next(r for r in csv.DictReader(
                open(R / "pca_sweep_delong" / "pca_sweep_results.csv"))
                if r["campaign"] == campaign and int(r["k"]) == kk)
            arms[ev] = {"var": (vp, float(vs["mean"]), float(vs["std"]), n_dims),
                        "pca": (pp, float(prow["auc_outer_mean"]),
                                float(prow["auc_outer_std"]), kk)}

        # ── pooled-OOF alignment across all 20 arms ──
        oof, y_ref = {}, None
        for ev, d in arms.items():
            for meth in ("var", "pca"):
                got = load_oof(d[meth][0])
                y, p = got
                if y_ref is None:
                    y_ref = y
                elif not np.array_equal(y, y_ref):
                    raise SystemExit(
                        f"ABORT [{key} ev{ev} {meth}]: OOF rows misaligned — "
                        f"DeLong would compare different peptides; not forcing")
                oof[(ev, meth)] = p

        # ── paired DeLong per EV rung, Holm over the 10-rung family ──
        evs_ok = sorted(arms)
        computed = []
        for ev in evs_ok:
            auc_a, auc_b, var_d = _p15.delong_test(
                y_ref, oof[(ev, "var")], oof[(ev, "pca")])
            d = auc_a - auc_b
            se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
            p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
            computed.append((ev, auc_a, auc_b, d, se, p))
        adjs = holm_p([c[5] for c in computed])
        stars = {}
        print(f"  {'EV':>4} {'var':>8} {'(X)':>5} {'pca':>8} {'(k)':>5} "
              f"{'delta':>8} {'holm p':>9} {'sig':>4}")
        for (ev, av, ap_, d, se, p), ph in zip(computed, adjs):
            stars[ev] = star(ph)
            xd = arms[ev]["var"][3]
            kk = arms[ev]["pca"][3]
            print(f"  {ev:>3}% {av:>8.4f} {xd:>5d} {ap_:>8.4f} {kk:>5d} "
                  f"{d:>+8.4f} {ph:>9.2e} {stars[ev]:>4}")
            delong_rows.append({
                "embedding": key, "ev_pct": ev, "auc_var": f"{av:.6f}",
                "auc_pca": f"{ap_:.6f}", "delta": f"{d:.6f}",
                "se": f"{se:.6f}", "z": f"{(d / se if se else 0.0):.4f}",
                "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                "star": stars[ev], "var_n_dims": xd, "pca_k": kk,
            })
            csv_rows.append({
                "embedding": key, "ev_pct": ev,
                "var_auc": f"{av:.6f}", "var_std": f"{arms[ev]['var'][2]:.6f}",
                "var_n_dims": xd,
                "pca_auc": f"{ap_:.6f}", "pca_std": f"{arms[ev]['pca'][2]:.6f}",
                "pca_k": kk, "delta": f"{d:.6f}", "star": stars[ev],
            })

        # ── figure: PCA solid, variance hatched, DeLong brackets ──
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        xs = range(len(evs_ok))
        w = 0.39
        ax.bar([x - w / 2 for x in xs], [arms[e]["pca"][1] for e in evs_ok],
               yerr=[arms[e]["pca"][2] for e in evs_ok], width=w, color=color,
               alpha=0.95, capsize=3, edgecolor="white", lw=0.5,
               label="flat-PCA (rotation)")
        ax.bar([x + w / 2 for x in xs], [arms[e]["var"][1] for e in evs_ok],
               yerr=[arms[e]["var"][2] for e in evs_ok], width=w, color=color,
               alpha=0.55, hatch="//", capsize=3, edgecolor="white", lw=0.5,
               label="topX-variance (raw dims, no PCA)")
        for i, e in enumerate(evs_ok):
            top = max(arms[e]["pca"][1] + arms[e]["pca"][2],
                      arms[e]["var"][1] + arms[e]["var"][2])
            xa, xb = i - w / 2 - 0.02, i + w / 2 + 0.02
            yb = top + 0.012
            ax.plot([xa, xa, xb, xb], [top + 0.004, yb, yb, top + 0.004],
                    lw=0.8, color="black")
            ax.text(i, yb + 0.002, stars[e], ha="center", va="bottom",
                    fontsize=9)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{arms[e]['var'][3]} ({e}%)" for e in evs_ok],
                           fontsize=9)
        ax.set_xlabel("raw dims selected at EV threshold")
        ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        title = ("ESM-C-300M padtoken" if key.startswith("esmc")
                 else "ESM-IF (PCA: eos/bos-repeat, variance: eosrepeat)")
        ax.set_title(f"{title} — dim selection vs rotation at matched EV",
                     fontsize=11)
        fig.tight_layout()
        png = plot_dir / f"ev_variance_vs_pca_auc_{key}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {png}")

    # ── tables ──
    tsv = res_dir / "ev_variance_vs_pca_delong.tsv"
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(delong_rows[0]),
                           delimiter="\t")
        w.writeheader()
        w.writerows(delong_rows)
    print(f"\n  wrote {tsv}  ({len(delong_rows)} rungs)")
    csvp = plot_dir / "ev_variance_vs_pca_auc.csv"
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  wrote {csvp}")

    print("\n" + "=" * 60)
    print("  CAPTION (paste below each figure):")
    print("=" * 60)
    print("At each explained-variance (EV) threshold on the flat-PCA "
          "ladder: flat-PCA (solid) keeps k rotated components capturing "
          "that share of the covariance-spectrum variance; topX-variance "
          "(hatched) instead selects the X raw embedding dims with the "
          "largest cumulative across-position variance (no rotation, all 29 "
          "positions preserved; 29*X features). Bars: nested-CV outer "
          "mean +/- std, XGB on 46,617 windows. Brackets: paired DeLong on "
          "pooled OOF per threshold, Holm-adjusted (* p<.05, ** p<.01, "
          "*** p<.001, ns). The EV labels are matched, the budgets are not: "
          "PCA's EV derives from the covariance spectrum (optimal variance "
          "concentration per component) while the selection EV sums "
          "individual dim variances (covariance ignored), so equal EV% "
          "never implies equal feature counts. Ticks: number of raw dims "
          "selected at that threshold (EV% in parentheses). ESM-IF pad "
          "mode: PCA arm on eos/bos-repeat, variance arm on eosrepeat "
          "(same rows/folds).")
    print("=" * 60)


if __name__ == "__main__":
    main()
