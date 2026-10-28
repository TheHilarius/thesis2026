#!/usr/bin/env python3
"""
27_variance_optimum.py — variance-dim sweep in the pca-sweep "withfull"
presentation style: locate the optimum dim count against the raw ceiling.

One figure per embedding (runs/ev_matched_topXvariance_sweep/plots/):
    variance_sweep_optimum_<emb_key>.png
Ladder of 10 topX-variance arms (family color, width .8) + the raw full
window embedding bar appended at the end (darkened family color, hatched,
black edge — styled like pca_sweep_auc_full_nobaselines_withfull.png).
Ticks: "<X> (<EV>%)" and "<29D> (Raw ESM-C/Raw ESM-IF)". Y starts at 0.

Brackets (withfull layout): paired DeLong on pooled OOF of EACH rung vs the
raw arm, Holm over the 10-rung family per embedding; brackets radiate from
significant rungs to the raw anchor, stacked nearest-first. ns rungs get no
bracket (reported in the terminal table + TSV).

Outputs:
    .../plots/variance_sweep_optimum_<emb_key>.png   (2 figures)
    .../results/variance_sweep_vs_raw_delong.tsv     (10 rungs x 2)
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

FAM = {  # emb_key -> (campaign, family color, raw token, raw tick label)
    "esmc300m_win_padtoken": ("esmc300m_pca_sweep", "#EE6677",
                              "esmc300m_win_padtoken", "Raw ESM-C"),
    "esmif_eosrepeat": ("esmif_eos_sweep", "#4477AA",
                        "esmif_eos_bos_repeat", "Raw ESM-IF"),
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_p15 = _load("p15", SRC_DIR / "15_padmode_significance.py")
_d21 = _load("d21", SRC_DIR / "21_dim21_delong.py")
_ps = _load("ps", SRC_DIR / "pca_sweep_AUC_delong_comparison.py")
load_oof = _d21.load_oof
draw_bracket, _shade, stars = _ps.draw_bracket, _ps._shade, _ps.stars
check_overlaps = _ps._check_label_overlaps


def delong_p(y, p_a, p_b):
    """Paired DeLong AUC difference -> (auc_a, auc_b, delta, se, p)."""
    auc_a, auc_b, var_d = _p15.delong_test(y, p_a, p_b)
    d = auc_a - auc_b
    se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
    p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
    return auc_a, auc_b, d, se, p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/ev_matched_topXvariance_sweep")
    args = ap.parse_args()
    root = Path(args.out_dir) if args.out_dir else (
        R / "ev_matched_topXvariance_sweep")
    plot_dir, res_dir = root / "plots", root / "results"
    plot_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    tsv_rows = []

    for key, (campaign, color, raw_tok, raw_label) in FAM.items():
        print(f"\n=== {key} ===")
        # ── variance arms: EV ladder -> X dims -> cv_results JSON ──
        arms = []  # (ev, x, mean, std, path)
        for ev in EVS:
            ej = (PROJECT_ROOT / "results" / "tables"
                  / f"dim_variance_{key}_ev{ev}.json")
            if not ej.exists():
                print(f"  WARN: missing {ej.name} — rung skipped")
                continue
            n_dims = json.loads(ej.read_text())["n_dims"]
            hits = sorted((root / "models" / "simple21").glob(
                f"cv_results_xgb_emb_{key}_simple21_var{n_dims}_*.json"))
            if not hits:
                print(f"  WARN: no cv_results for var{n_dims} (ev{ev})")
                continue
            s = json.loads(hits[-1].read_text())["cv_summary"]["auc_roc"]
            arms.append((ev, n_dims, float(s["mean"]), float(s["std"]),
                         hits[-1]))
        if not arms:
            raise SystemExit(f"ABORT [{key}]: no variance arms found")

        # ── raw ceiling arm (100% EV anchor) ──
        raw_hits = sorted(R.glob(
            f"{campaign}/models/flat_raw/"
            f"cv_results_xgb_emb_{raw_tok}_flat_raw_*.json"))
        if not raw_hits:
            raise SystemExit(f"ABORT [{key}]: raw flat_raw cv_results missing")
        rs = json.loads(raw_hits[-1].read_text())["cv_summary"]["auc_roc"]
        raw_mean, raw_std = float(rs["mean"]), float(rs["std"])
        raw_feats = json.loads(raw_hits[-1].read_text())["config"]["n_features"]

        # ── pooled-OOF alignment across all 11 arms (abort on misalign) ──
        oof, y_ref = {}, None
        for ev, x, m, sd, p in arms:
            y, pv = load_oof(p)
            if y_ref is None:
                y_ref = y
            elif not np.array_equal(y, y_ref):
                raise SystemExit(
                    f"ABORT [{key} ev{ev}]: OOF rows misaligned vs other "
                    f"arms — DeLong would compare different peptides; "
                    f"not forcing")
            oof[("var", ev)] = pv
        y_raw, p_raw = load_oof(raw_hits[0])
        if y_ref is None or not np.array_equal(y_raw, y_ref):
            raise SystemExit(
                f"ABORT [{key}]: raw-arm OOF rows misaligned — not forcing")
        oof[("raw", None)] = p_raw

        # ── DeLong: EVERY rung vs the raw anchor, Holm over the 10 ──
        computed = [(a, delong_p(y_ref, oof[("var", a[0])],
                                 oof[("raw", None)])) for a in arms]
        adjs = _p15.holm_adjust([c[1][4] for c in computed])
        stars_by_ev = {}
        best = max(arms, key=lambda a: a[2])
        print(f"  RAW anchor: {raw_mean:.4f} +/- {raw_std:.4f} "
              f"({raw_feats} feats)")
        print(f"  {'EV':>4} {'X dims':>6} {'feats':>6} {'auc':>8} "
              f"{'std':>6} {'delta':>8} {'holm p':>9} {'sig':>4}")
        for (a, (av, ar, d, se, p)), ph in zip(computed, adjs):
            stars_by_ev[a[0]] = stars(ph)
            print(f"  {a[0]:>3}% {a[1]:>6d} {29*a[1]:>6d} {a[2]:>8.4f} "
                  f"{a[3]:>6.4f} {d:>+8.4f} {ph:>9.2e} {stars_by_ev[a[0]]:>4}")
            tsv_rows.append({
                "embedding": key, "ev_pct": a[0], "var_n_dims": a[1],
                "n_features": 29 * a[1],
                "auc_var": f"{av:.6f}", "var_std": f"{a[3]:.6f}",
                "auc_raw": f"{ar:.6f}", "raw_std": f"{raw_std:.6f}",
                "delta": f"{d:.6f}", "se": f"{se:.6f}",
                "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                "star": stars_by_ev[a[0]], "is_best": a[0] == best[0],
            })

        # ── figure: withfull style ──
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        n = len(arms)
        ref_i = n  # raw anchor sits immediately after the ladder (no gap)
        w = 0.8
        pooled = [a[2] for a in arms]
        ostd = [a[3] for a in arms]

        for i, a in enumerate(arms):
            ax.bar(i, a[2], yerr=a[3], width=w, color=color, alpha=0.85,
                   capsize=3, edgecolor="white", lw=0.5, zorder=3)
        ax.bar(ref_i, raw_mean, yerr=raw_std, width=w, color=_shade(color),
               alpha=0.95, capsize=3, edgecolor="black", lw=0.8,
               hatch="//", zorder=3)

        # value labels (withfull white-bbox style)
        for i, a in enumerate(arms):
            ax.text(i, a[2] + a[3] + 0.015, f"{a[2]:.4f}",
                    ha="center", va="bottom", fontsize=8, zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
        ax.text(ref_i, raw_mean + raw_std + 0.015, f"{raw_mean:.4f}",
                ha="center", va="bottom", fontsize=8, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))

        # ── brackets: significant rungs -> raw anchor, stacked nearest-first
        sig = [(i, a) for i, a in enumerate(arms)
               if stars_by_ev[a[0]] != "ns"]
        sig.sort(key=lambda t: abs(ref_i - t[0]))  # nearest first
        h, step, gap = 0.012, 0.045, 0.06
        base = max((np.array(pooled) + np.array(ostd)).max() + 0.05,
                   raw_mean + raw_std + 0.04) + h
        for lvl, (i, a) in enumerate(sig):
            y = base + lvl * step
            draw_bracket(ax, float(i), ref_i - gap, y,
                         stars_by_ev[a[0]], h, fontsize=7)
        n_levels = max(len(sig), 1)
        y_top = max(1.0, base + (n_levels - 1) * step + 0.5 * step + 0.045)

        ax.set_ylim(0, y_top)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks(list(range(n + 1)))
        ax.set_xticklabels(
            [f"{a[1]} ({a[0]}%)" for a in arms]
            + [raw_label],
            fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel("raw dims selected by cumulative variance (EV threshold)")
        ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
        ax.grid(axis="y", alpha=0.3)
        title = ("ESM-C-300M padtoken" if key.startswith("esmc")
                 else "ESM-IF (variance: eosrepeat, raw: eos/bos-repeat)")
        ax.set_title(f"{title} — variance-dim sweep vs raw ceiling",
                     fontsize=11, pad=6)
        fig.tight_layout()
        check_overlaps(fig, tag=f"variance_sweep_optimum_{key}")
        png = plot_dir / f"variance_sweep_optimum_{key}.png"
        fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {png}")

    # ── TSV ──
    tsv = res_dir / "variance_sweep_vs_raw_delong.tsv"
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tsv_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(tsv_rows)
    print(f"\n  wrote {tsv}  ({len(tsv_rows)} rung-vs-raw comparisons)")

    print("\n" + "=" * 60)
    print("  CAPTION (paste below each figure):")
    print("=" * 60)
    print("Variance-dim sweep: X = fewest raw embedding dims whose "
          "cumulative across-position variance first reaches each EV "
          "threshold (30-90%); features = 29*X (all positions kept, no "
          "rotation). The hatched dark bar is the raw window embedding "
          "without reduction (29*D features = 100% EV, the ceiling arm; "
          "ESM-IF pad mode eos/bos-repeat vs eosrepeat for the variance "
          "arms - same rows and folds). Bars: nested-CV outer mean +/- "
          "std, XGB on 46,617 windows. Brackets: paired DeLong on pooled "
          "OOF of each rung vs the raw arm, Holm-adjusted over the "
          "10-rung family; only significant rungs are bracketed (ns "
          "rungs in the results table), ns / * p<.05 / ** p<.01 / "
          "*** p<.001.")
    print("=" * 60)


if __name__ == "__main__":
    main()
