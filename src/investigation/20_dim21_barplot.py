#!/usr/bin/env python3
"""
20_dim21_barplot.py — AUC-ROC barplot: dim21 (29x21 reduced windows) vs
BLOSUM50 and the best flat-PCA points from the PCA sweeps.

Bars (left to right):
    BLOSUM50 (609 feat)                       — encodings_sweep
    ESM-C-300M padtoken flat-PCA 80% (k=2278) — pca_sweep_delong (optimal)
    ESM-C-300M dim21 first-21 / var-21        — runs/*/logs/flat_raw
    ESM-IF eos/bos flat-PCA 60% (k=632)       — pca_sweep_delong (optimal)
    ESM-IF eosrepeat dim21 first-21 / var-21  — runs/*/logs/flat_raw

Notes baked into the caption:
  * PCA bars are the DeLong-confirmed optimal picks (60% EV k=632 for
    ESM-IF, 80% EV k=2278 for ESM-C-300M), campaigns esmif_eos_sweep /
    esmc300m_pca_sweep.
  * the PCA bar is eos/bos-repeat while the dim21 bars are eosrepeat —
    different pad-mode embeddings, labels say so.

dim21 AUCs are the nested-CV "CROSS-VALIDATION SUMMARY" line
(mean +/- std over 6 outer iterations) parsed from the flat_raw logs.

Outputs (runs/<runtag>/ per convention, default runtag=simple21_pilot):
    runs/simple21_pilot/plots/dim21_vs_baselines_auc.png
    runs/simple21_pilot/plots/dim21_vs_baselines_auc.csv

With --stats-csv (21_dim21_delong.py output) significance brackets are
drawn above the bars: star per pair from Holm-adjusted paired DeLong
(* p<.05, ** p<.01, *** p<.001, **** p<.0001, ns otherwise).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fallback values if the sweep TSVs are absent (verified 2026-10-22)
FALLBACK = {
    "blosum50": (0.7633, 0.0044),
    "esmc300m_pca80": (0.6973, 0.0069),
    "esmif_pca60": (0.6448, 0.0041),
}

COLORS = {
    "encoding": "#BBBBBB",          # padmode palette: AA encoding grey
    "esmc_pca": "#85C1E9",
    "esmc_dim21": "#2E86C1",
    "esmif_pca": "#A9CCE3",
    "esmif_dim21": "#17A589",
}

AUC_RE = re.compile(r"auc_roc\s+([\d.]+)\s+\+/-\s+([\d.]+)")


def load_dim21(logs_dir: Path):
    """feature_set -> (auc, std) from the last CROSS-VALIDATION SUMMARY."""
    out = {}
    for f in sorted(logs_dir.glob("*d21*flat_raw*.txt")):
        text = f.read_text(errors="replace")
        hits = AUC_RE.findall(text)
        if not hits:
            print(f"  WARN: no AUC summary yet in {f.name}")
            continue
        fs = re.search(r"xgb_(emb_\S+?)_flat_raw", f.name)
        key = fs.group(1) if fs else f.stem
        out[key] = (float(hits[-1][0]), float(hits[-1][1]))
    return out


def load_delong_pick(path: Path, campaign: str, k: str):
    """(auc_outer_mean, auc_outer_std) from pca_sweep_results.csv."""
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("campaign") == campaign and r.get("k") == k:
                return (float(r["auc_outer_mean"]), float(r["auc_outer_std"]))
    return None


def load_blosum(path: Path):
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("feature_set") == "blosum50" and r.get("model_key") == "xgb":
                return (float(r["cv_mean_auc_roc"]), float(r["cv_std_auc_roc"]))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--logs-dir", default="runs/simple21_pilot/logs/flat_raw",
                    help="flat_raw log dir with the 4 d21 runs (project root)")
    ap.add_argument("--runtag", default="simple21_pilot",
                    help="run folder under runs/ that owns the outputs "
                         "(default: simple21_pilot)")
    ap.add_argument("--out-prefix", default=None,
                    help="explicit output prefix; default: "
                         "runs/<runtag>/plots/dim21_vs_baselines")
    ap.add_argument("--stats-csv", default=None,
                    help="dim21_delong_pairs.tsv from 21_dim21_delong.py — "
                         "draw significance brackets with stars. Default: "
                         "runs/<runtag>/results/dim21_delong_pairs.tsv if it "
                         "exists (brackets drawn automatically)")
    args = ap.parse_args()

    root = PROJECT_ROOT
    run_dir = root / "runs" / args.runtag
    out_prefix = Path(args.out_prefix) if args.out_prefix else (
        run_dir / "plots" / "dim21_vs_baselines")
    stats_csv = args.stats_csv
    if stats_csv is None:
        default_stats = run_dir / "results" / "dim21_delong_pairs.tsv"
        stats_csv = str(default_stats) if default_stats.exists() else None
        print(f"  stats: {stats_csv or 'none (no --stats-csv, no brackets)'}")

    print("Loading baselines ...")
    blosum = load_blosum(root / "runs/encodings_sweep/results/metrics.tsv") \
        or FALLBACK["blosum50"]
    delong = root / "runs/pca_sweep_delong/pca_sweep_results.csv"
    esmc_pca = load_delong_pick(delong, "esmc300m_pca_sweep", "2278") \
        or FALLBACK["esmc300m_pca80"]
    esmif_pca = load_delong_pick(delong, "esmif_eos_sweep", "632") \
        or FALLBACK["esmif_pca60"]
    print(f"  blosum50       {blosum[0]:.4f} +/- {blosum[1]:.4f}")
    print(f"  esmc300m PCA   {esmc_pca[0]:.4f} +/- {esmc_pca[1]:.4f}  (k=2278, 80%)")
    print(f"  esmif PCA      {esmif_pca[0]:.4f} +/- {esmif_pca[1]:.4f}  (k=632, 60%, optimal)")

    print(f"Loading dim21 logs from {args.logs_dir} ...")
    dim21 = load_dim21(root / args.logs_dir)
    for k, v in sorted(dim21.items()):
        print(f"  {k:<45} {v[0]:.4f} +/- {v[1]:.4f}")

    bars = [
        ("BLOSUM50\n(609 feat)", blosum, "encoding", "blosum50"),
        ("ESM-C-300M\nflat-PCA 80%\n(k=2278)", esmc_pca, "esmc_pca", "esmc_pca80"),
        ("ESM-C-300M\ndim21\nfirst-21",
         dim21.get("emb_esmc300m_win_padtoken_d21_first"), "esmc_dim21", "esmc_first"),
        ("ESM-C-300M\ndim21\nvar-21",
         dim21.get("emb_esmc300m_win_padtoken_d21_var"), "esmc_dim21", "esmc_var"),
        ("ESM-IF eos/bos\nflat-PCA 60%\n(k=632, optimal)", esmif_pca, "esmif_pca", "esmif_pca60"),
        ("ESM-IF eosrep\ndim21\nfirst-21",
         dim21.get("emb_esmif_eosrepeat_d21_first"), "esmif_dim21", "esmif_first"),
        ("ESM-IF eosrep\ndim21\nvar-21",
         dim21.get("emb_esmif_eosrepeat_d21_var"), "esmif_dim21", "esmif_var"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rows = []
    for i, (label, val, color, _) in enumerate(bars):
        if val is None:
            print(f"  WARN: missing result for bar {label!r} — skipped")
            continue
        ax.bar(i, val[0], yerr=val[1], capsize=4, width=0.62,
               color=COLORS[color], edgecolor="black", linewidth=0.6)
        ax.text(i, val[0] + val[1] + 0.006, f"{val[0]:.3f}",
                ha="center", fontsize=9, fontweight="bold")
        rows.append({"bar": label.replace("\n", " "),
                     "auc_roc": f"{val[0]:.6f}", "std": f"{val[1]:.6f}"})

    xs = [i for i, (_, v, _, _) in enumerate(bars) if v is not None]
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars if b[1] is not None], fontsize=8.5)
    ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
    ax.grid(axis="y", alpha=0.3)

    # ── significance brackets (--stats-csv from 21_dim21_delong.py) ──
    y_ceiling = 0.80
    arm_x = {arm: i for i, (_, v, _, arm) in enumerate(bars) if v is not None}
    if stats_csv:
        with open(stats_csv) as fh:
            stats_rows = list(csv.DictReader(fh, delimiter="\t"))
        data_top = max(v[0] + v[1] for _, v, _, _ in bars if v is not None)
        base, step, fam_lvl = data_top + 0.015, 0.016, {}
        for r in stats_rows:
            a, b = r["arm_a"], r["arm_b"]
            if a not in arm_x or b not in arm_x:
                print(f"  WARN: stats pair {a}~{b} has no bar — skipped")
                continue
            lvl = fam_lvl.get(r["family"], 0)
            fam_lvl[r["family"]] = lvl + 1
            if r["family"] == "vs_blosum":      # wide spans drawn highest
                lvl += 3
            y = base + lvl * step
            xa, xb = sorted((arm_x[a], arm_x[b]))
            ax.plot([xa, xa, xb, xb], [y - 0.004, y, y, y - 0.004],
                    lw=0.8, color="black")
            ax.text((xa + xb) / 2, y + 0.002, r["star"],
                    ha="center", va="bottom", fontsize=9.5)
            y_ceiling = max(y_ceiling, y + 0.022)

    ax.set_ylim(0.5, y_ceiling)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_title("dim21 (29x21 reduced windows) vs BLOSUM50 and flat-PCA", fontsize=12)
    fig.text(0.5, 0.005,
             "PCA bars = flat window-PCA at the DeLong-confirmed optimal picks (ESM-C 80% EV k=2278; ESM-IF 60% EV k=632). "
             "ESM-IF PCA = eos/bos-repeat; dim21 ESM-IF = eosrepeat.",
             ha="center", fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(out_prefix) + "_auc.png")
    fig.savefig(png, dpi=150)
    csvp = Path(str(out_prefix) + "_auc.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bar", "auc_roc", "std"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {png}")
    print(f"  wrote {csvp}")


if __name__ == "__main__":
    main()
