#!/usr/bin/env python3
"""
30_dimselect_ev_bars.py — per-embedding bars: dimselect (supervised
per-dim screening) vs topX-variance (unsupervised) at each explained-
variance threshold, with paired-DeLong brackets.

Figures (runs/dimselect_matched/plots/, ESM-C only — ESM-IF
discontinued 2026-11-06):
    dimselect_ev_bars_<emb_key>.png
    dimselect_vs_vardims_ev80_<emb_key>.png   (leftmost hatched bar =
        the standalone vardims EV80 arm, N=643; dimselect rungs beside
        it; per-rung DeLong vs the EV80 bar)
x-axis = EV% ladder (30..90); two bars per tick:
    solid  = dimselect (top-N dims ranked by fold-nested small-predictor AUC)
    hatched= topX-variance (raw dims by cumulative dim-variance, no PCA)
Y-axis 0-1. Brackets: paired DeLong on pooled OOF per rung, Holm within
the available-rung family per embedding, thesis star convention.

Same style as 26_ev_variance_vs_pca.py (which compares PCA vs variance);
this one answers "does the supervised criterion beat the unsupervised one
at identical budgets". N per EV rung from
results/tables/dim_variance_<key>_ev<ev>.json (both bars share the N).

Sources:
    dimselect  runs/{dimselect_sweep,_legacy/dimselect_sweep,dimselect_matched}
               /models/dimselect/cv_results_xgb_<ds_key>_dimselect_top<N>_xgb_*.json
    variance   ev_matched_topXvariance_sweep/models/simple21/
                   cv_results_xgb_emb_{key}_simple21_var{N}_*.json
    EV80 ref   runs/xgb_emb_{key}_vardims_ev80_var<N>_*/
                   models/vardims/cv_results_*.json

Missing rungs (dimselect arm still queued) are skipped with a warning —
re-run once jobs land. Aborts if OOF rows misalign across campaigns.

Outputs:
    runs/dimselect_matched/plots/dimselect_ev_bars_<emb>.png
    runs/dimselect_matched/plots/dimselect_vs_vardims_ev80_<emb>.png
    runs/dimselect_matched/results/dimselect_ev_delong.tsv
    runs/dimselect_matched/results/dimselect_vs_vardims_ev80.tsv
    runs/dimselect_matched/plots/dimselect_ev_bars.csv
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

DIMSELECT_DIRS = [
    R / "dimselect_sweep/models/dimselect",
    R / "_legacy/dimselect_sweep/models/dimselect",
    R / "dimselect_matched/models/dimselect",
]

# emb_key -> (dimselect feature key, family color, variance glob template)
FAM = {
    "esmc300m_win_padtoken": (
        "emb_esmc300m_win_padtoken", "#EE6677",
        "ev_matched_topXvariance_sweep/models/simple21/"
        "cv_results_xgb_emb_{key}_simple21_var{X}_*.json"),
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_p15 = _load("p15", SRC_DIR / "15_padmode_significance.py")


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_p_pair(p, p_adj):
    """padmode_choice bracket text: 'p=0.51, p_adj=1.0'; star from p_adj."""
    a = "p_adj<0.001" if p_adj < 0.001 else f"p_adj={p_adj:.2g}"
    b = "p<0.001" if p < 0.001 else f"p={p:.2g}"
    return f"{b}, {a}"


def load_oof(path):
    res = json.loads(Path(path).read_text())
    folds = sorted(int(k) for k in res["outer_val_predictions"])
    y = np.concatenate([np.asarray(res["outer_val_predictions"][str(f)]["y_true"])
                        for f in folds])
    p = np.concatenate([np.asarray(res["outer_val_predictions"][str(f)]["y_prob"],
                                   dtype=float) for f in folds])
    return y, p


def mean_std(path):
    s = json.loads(Path(path).read_text())["cv_summary"]["auc_roc"]
    return float(s["mean"]), float(s["std"])


def resolve(patterns):
    for pat in patterns:
        hits = sorted(R.glob(pat))
        if hits:
            return hits[-1]
    return None


def dimselect_json(ds_key, n_dims):
    hits = []
    for d in DIMSELECT_DIRS:
        if d.exists():
            hits += list(d.glob(
                f"cv_results_xgb_{ds_key}_dimselect_top{n_dims}_xgb_*.json"))
    return sorted(hits)[-1] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/dimselect_matched")
    args = ap.parse_args()
    root = Path(args.out_dir) if args.out_dir else (R / "dimselect_matched")
    plot_dir, res_dir = root / "plots", root / "results"
    plot_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    delong_rows, csv_rows, ev80_rows = [], [], []

    for key, (ds_key, color, var_tmpl) in FAM.items():
        print(f"\n=== {key} ===")
        arms = {}  # ev -> {"ds": (path, mean, std), "var": (path, mean, std)}
        for ev in EVS:
            ej = PROJECT_ROOT / "results" / "tables" / f"dim_variance_{key}_ev{ev}.json"
            if not ej.exists():
                print(f"  WARN: missing {ej.name} — rung skipped")
                continue
            n_dims = json.loads(ej.read_text())["n_dims"]
            dp = dimselect_json(ds_key, n_dims)
            vp = resolve([var_tmpl.format(key=key, ev=ev, X=n_dims)])
            if dp is None or vp is None:
                print(f"  WARN: ev{ev} (N={n_dims}): "
                      f"dimselect={'ok' if dp else 'MISSING'} "
                      f"var={'ok' if vp else 'MISSING'} — rung skipped")
                continue
            dm, ds_ = mean_std(dp)
            vm, vs_ = mean_std(vp)
            arms[ev] = {"n": n_dims,
                        "ds": (dp, dm, ds_), "var": (vp, vm, vs_)}

        if not arms:
            print("  WARN: no rungs available yet")
            continue

        # ── pooled-OOF alignment across all arms ──
        oof, y_ref = {}, None
        for ev, d in arms.items():
            for meth in ("ds", "var"):
                y, p = load_oof(d[meth][0])
                if y_ref is None:
                    y_ref = y
                elif not np.array_equal(y, y_ref):
                    raise SystemExit(
                        f"ABORT [{key} ev{ev} {meth}]: OOF rows misaligned — "
                        f"DeLong would compare different peptides; not forcing")
                oof[(ev, meth)] = p

        # ── paired DeLong per rung, Holm within embedding ──
        evs_ok = sorted(arms)
        computed = []
        for ev in evs_ok:
            auc_a, auc_b, var_d = _p15.delong_test(
                y_ref, oof[(ev, "ds")], oof[(ev, "var")])
            d = auc_a - auc_b
            se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
            p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
            computed.append((ev, auc_a, auc_b, d, se, p))
        adjs = _p15.holm_adjust([c[5] for c in computed])
        stars = {}
        print(f"  {'EV':>4} {'N':>5} {'dimsel':>8} {'var':>8} "
              f"{'delta':>8} {'holm p':>9} {'sig':>4}")
        for (ev, ads, av, d, se, p), ph in zip(computed, adjs):
            stars[ev] = star(ph)
            n = arms[ev]["n"]
            print(f"  {ev:>3}% {n:>5d} {ads:>8.4f} {av:>8.4f} "
                  f"{d:>+8.4f} {ph:>9.2e} {stars[ev]:>4}")
            delong_rows.append({
                "embedding": key, "ev_pct": ev, "n_dims": n,
                "auc_dimselect": f"{ads:.6f}", "auc_variance": f"{av:.6f}",
                "delta": f"{d:.6f}", "se": f"{se:.6f}",
                "z": f"{(d / se if se else 0.0):.4f}",
                "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                "star": stars[ev],
            })
            csv_rows.append({
                "embedding": key, "ev_pct": ev, "n_dims": n,
                "dimselect_auc": f"{ads:.6f}",
                "dimselect_std": f"{arms[ev]['ds'][2]:.6f}",
                "variance_auc": f"{av:.6f}",
                "variance_std": f"{arms[ev]['var'][2]:.6f}",
                "delta": f"{d:.6f}", "star": stars[ev],
            })

        # ── figure: dimselect solid, variance hatched, DeLong brackets ──
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        xs = range(len(evs_ok))
        w = 0.39
        ax.bar([x - w / 2 for x in xs], [arms[e]["ds"][1] for e in evs_ok],
               yerr=[arms[e]["ds"][2] for e in evs_ok], width=w, color=color,
               alpha=0.95, capsize=3, edgecolor="white", lw=0.5,
               label="dimselect (supervised screening)")
        ax.bar([x + w / 2 for x in xs], [arms[e]["var"][1] for e in evs_ok],
               yerr=[arms[e]["var"][2] for e in evs_ok], width=w, color=color,
               alpha=0.55, hatch="//", capsize=3, edgecolor="white", lw=0.5,
               label="topX-variance (unsupervised)")
        for i, e in enumerate(evs_ok):
            top = max(arms[e]["ds"][1] + arms[e]["ds"][2],
                      arms[e]["var"][1] + arms[e]["var"][2])
            xa, xb = i - w / 2 - 0.02, i + w / 2 + 0.02
            yb = top + 0.012
            ax.plot([xa, xa, xb, xb], [top + 0.004, yb, yb, top + 0.004],
                    lw=0.8, color="black")
            ax.text(i, yb + 0.002, stars[e], ha="center", va="bottom",
                    fontsize=9)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{arms[e]['n']} ({e}%)" for e in evs_ok],
                           fontsize=9)
        ax.set_xlabel("raw dims kept at EV threshold")
        ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        title = ("ESM-C-300M padtoken" if key.startswith("esmc")
                 else "ESM-IF (eos/bos-repeat)")
        ax.set_title(f"{title} — supervised vs unsupervised selection "
                     "at matched EV", fontsize=11)
        fig.tight_layout()
        png = plot_dir / f"dimselect_ev_bars_{key}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {png}")

        # ── extra figure: dimselect rung-by-rung vs the standalone
        #    vardims EV80 arm (its own runs/ folder, run 2026-10-28) ──
        n80 = json.loads((PROJECT_ROOT / "results" / "tables" /
                          f"dim_variance_{key}_ev80.json").read_text())["n_dims"]
        ref_p = resolve([f"xgb_emb_{key}_vardims_ev80_var{n80}_*/"
                         "models/vardims/cv_results_*.json"])
        if ref_p is None:
            print(f"  WARN: vardims EV80 arm (N={n80}) not found — "
                  "extra figure skipped")
        else:
            ref_m, ref_s = mean_std(ref_p)
            ry, ref_oof = load_oof(ref_p)
            if not np.array_equal(ry, y_ref):
                raise SystemExit("ABORT: vardims EV80 OOF rows misaligned")
            print(f"  vs vardims EV80 arm (N={n80}, AUC={ref_m:.4f}):")
            computed80 = []
            for ev in evs_ok:
                auc_a, auc_b, var_d = _p15.delong_test(
                    y_ref, oof[(ev, "ds")], ref_oof)
                d = auc_a - auc_b
                se = float(np.sqrt(var_d)) if var_d > 0 else 0.0
                p = 1.0 if se == 0 else float(erfc(abs(d / se) / sqrt(2)))
                computed80.append((ev, auc_a, auc_b, d, se, p))
            adjs80 = _p15.holm_adjust([c[5] for c in computed80])
            stars80 = {}
            for (ev, ads, arb, d, se, p), ph in zip(computed80, adjs80):
                stars80[ev] = star(ph)
                print(f"    ev{ev:>2} (N={arms[ev]['n']:>3d}) {ads:.4f} vs "
                      f"{arb:.4f}: d={d:+.4f} holm={ph:.2e} [{stars80[ev]}]")
                ev80_rows.append({
                    "embedding": key, "ev_pct": ev,
                    "n_dims": arms[ev]["n"],
                    "auc_dimselect": f"{ads:.6f}",
                    "auc_vardims_ev80": f"{arb:.6f}",
                    "delta": f"{d:.6f}", "se": f"{se:.6f}",
                    "z": f"{(d / se if se else 0.0):.4f}",
                    "p_delong": f"{p:.3e}", "p_holm": f"{ph:.3e}",
                    "star": stars80[ev],
                })

            fig, ax = plt.subplots(figsize=(11.2, 6.2))
            # EV80 reference as the leftmost bar (between y-axis and the
            # ladder), hatched = unsupervised, matching the main figure
            ax.bar([-0.85] + list(xs), [ref_m] + [arms[e]["ds"][1]
                   for e in evs_ok],
                   yerr=[ref_s] + [arms[e]["ds"][2] for e in evs_ok],
                   width=0.6, color=color, capsize=3, edgecolor="white",
                   lw=0.5)
            # patches[0] = the EV80 reference bar; make it hatched and
            # label both handles for the legend
            ax.patches[0].set_alpha(0.55)
            ax.patches[0].set_hatch("//")
            ax.patches[0].set_label(
                f"topX-variance EV80 (N={n80}, {ref_m:.4f})")
            ax.patches[1].set_label("dimselect (supervised screening)")
            for b in ax.patches[1:]:
                b.set_alpha(0.95)
            # performance printed on top of every bar (4 decimals —
            # the deltas live at dp4)
            bar_x = [-0.85] + list(xs)
            bar_v = [ref_m] + [arms[e]["ds"][1] for e in evs_ok]
            for xi, v in zip(bar_x, bar_v):
                ax.text(xi, v + 0.005, f"{v:.4f}", ha="center",
                        va="bottom", fontsize=8)

            # significance-only brackets from the EV80 bar to each rung it
            # differs from (padmode_choice style: staggered, widest span
            # lowest; end-ticks; text = raw p + Holm p_adj, star from p_adj)
            sig_pairs = [(ev, p_, ph)
                         for (ev, _a, _b, _d, _se, p_), ph
                         in zip(computed80, adjs80) if ph < 0.05]
            sig_pairs.sort(key=lambda t: -evs_ok.index(t[0]))  # widest first
            values = bar_v
            y0 = max(values) * 1.08
            max_lvl = max(len(sig_pairs) + 1, 1)  # +1: 80-vs-85 bracket
            headroom = 0.97 - y0 - max(values) * 0.03
            step = min(max(values) * 0.085,
                       headroom / (max_lvl - 1 + 0.35))
            br_fontsize = 7.5 if step >= max(values) * 0.05 else 6.5
            xa = -0.85
            for lvl, (ev, p_, ph) in enumerate(sig_pairs):
                xb = float(evs_ok.index(ev))
                y_br = y0 + lvl * step
                ax.plot([xa, xa, xb, xb],
                        [y_br * 0.985, y_br, y_br, y_br * 0.985],
                        color="black", lw=0.9)
                txt = f"{fmt_p_pair(p_, ph)} ({stars80[ev]})"
                ax.text((xa + xb) / 2, y_br + step * 0.12, txt,
                        ha="center", va="bottom", fontsize=br_fontsize)

            # direct 80% vs 85% dimselect comparison (adjacent rungs;
            # single planned test — raw p only, no Holm)
            if 80 in evs_ok and 85 in evs_ok:
                i80, i85 = evs_ok.index(80), evs_ok.index(85)
                a1, b1, vd = _p15.delong_test(
                    y_ref, oof[(80, "ds")], oof[(85, "ds")])
                d1 = a1 - b1
                se1 = float(np.sqrt(vd)) if vd > 0 else 0.0
                p1 = 1.0 if se1 == 0 else float(erfc(abs(d1 / se1) / sqrt(2)))
                y_br = y0 + len(sig_pairs) * step
                xa8, xb8 = float(i80), float(i85)
                ax.plot([xa8, xa8, xb8, xb8],
                        [y_br * 0.985, y_br, y_br, y_br * 0.985],
                        color="black", lw=0.9)
                ax.text((xa8 + xb8) / 2, y_br + step * 0.12,
                        f"p={p1:.2g} ({star(p1)})",
                        ha="center", va="bottom", fontsize=br_fontsize)
                print(f"  dimselect ev80 vs ev85: d={d1:+.4f} "
                      f"p={p1:.2e} [{star(p1)}]")

            ax.set_xticks([-0.85] + list(xs))
            ax.set_xticklabels([f"EV80\n({n80})"] +
                               [f"{arms[e]['n']} ({e}%)" for e in evs_ok],
                               fontsize=9)
            ax.set_xlim(-1.5, len(evs_ok) - 0.4)
            ax.set_xlabel("raw dims kept at EV threshold (dimselect)")
            ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=8, frameon=False)
            ax.set_title("ESM-C-300M padtoken — dimselect vs topX-variance "
                         "EV80 operating point", fontsize=11)
            fig.tight_layout()
            png2 = plot_dir / f"dimselect_vs_vardims_ev80_{key}.png"
            fig.savefig(png2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {png2}")

    # ── tables ──
    if delong_rows:
        tsv = res_dir / "dimselect_ev_delong.tsv"
        with open(tsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(delong_rows[0]),
                               delimiter="\t")
            w.writeheader()
            w.writerows(delong_rows)
        print(f"\n  wrote {tsv}  ({len(delong_rows)} rungs)")
        csvp = plot_dir / "dimselect_ev_bars.csv"
        with open(csvp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  wrote {csvp}")
    else:
        print("\n  WARN: no rungs computed — dimselect arms not finished?")

    if ev80_rows:
        tsv80 = res_dir / "dimselect_vs_vardims_ev80.tsv"
        with open(tsv80, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ev80_rows[0]),
                               delimiter="\t")
            w.writeheader()
            w.writerows(ev80_rows)
        print(f"  wrote {tsv80}  ({len(ev80_rows)} rungs)")

    print("\n" + "=" * 60)
    print("  CAPTION (paste below each figure):")
    print("=" * 60)
    print("At each explained-variance (EV) threshold on the dim-variance "
          "ladder: dimselect (solid) keeps the N raw embedding dims ranked "
          "by a fold-nested supervised per-dim screener (small XGB per dim "
          "on the 29 window positions; top-N raw, no PCA, 29*N features); "
          "topX-variance (hatched) instead selects the N raw dims with the "
          "largest cumulative across-position variance (unsupervised, "
          "same N). Bars: nested-CV outer mean +/- std, XGB on 46,617 "
          "windows. Brackets: paired DeLong on pooled OOF per threshold, "
          "Holm-adjusted (* p<.05, ** p<.01, *** p<.001, ns). Both bars "
          "at a tick use the identical dim budget N (ticks: N, EV% in "
          "parentheses), so each bracket is a matched-budget comparison "
          "of the ranking criterion alone.")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  CAPTION 2 (dimselect_vs_vardims_ev80_<emb>.png):")
    print("=" * 60)
    print("The leftmost hatched bar is the topX-variance EV80 arm — "
          "the N=643 raw dims at the 80% cumulative dim-variance "
          "threshold, trained independently; it stands before the ladder "
          "as the comparison baseline. Each solid bar is the dimselect "
          "operating point (top-N raw embedding dims ranked by the "
          "fold-nested supervised per-dim screener) at that dim budget. "
          "AUC (4 decimals) is printed on every bar. Brackets run from "
          "the EV80 bar to each dimselect rung that differs from it "
          "significantly (paired DeLong on pooled OOF predictions, "
          "n=46,617, identical outer folds; Holm-adjusted over the rungs; "
          "ns pairs not bracketed; text shows raw p and adjusted p, star "
          "from p_adj: * p<.05, ** p<.01, *** p<.001). The top short "
          "bracket directly compares the adjacent dimselect rungs 643 "
          "vs 716 (unadjusted paired DeLong, raw p). Bars: nested-CV "
          "outer mean +/- std, XGB.")
    print("=" * 60)


if __name__ == "__main__":
    main()
