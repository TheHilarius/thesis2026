#!/usr/bin/env python3
"""
28_cnn_compare_barplot.py — AUC-ROC barplot: BLOSUM baselines vs ev-matched
top-variance selection vs CNN (simple + 2-layer) on the two decided embeddings.

Bars left to right (fixed):
    1. BLOSUM50 (xgb, encodings_sweep)
    2. handcrafted+blosum50 champion (xgb, handcrafted_sweep DeLong 0.7665)
    3. ESM-C-300M top-var select (ev-matched sweep best, e.g. var793 = 793*29 dims)
    4. ESM-IF top-var select (ev-matched sweep best, e.g. var392 = 392*29 dims)
    5. ESM-C CNN baseline (cnn_simple, runs/cnn_cv* on hub1)
    6. ESM-C CNN 2-layer (cnn_2layer, runs/cnn_2layer)
    7. ESM-IF CNN baseline (cnn_simple)
    8. ESM-IF CNN 2-layer (cnn_2layer)

IMPORTANT: bars 3-4 are NOT PCA. The ev_matched_topXvariance_sweep keeps the
top-N most variable raw embedding dims (variance-ranked feature SELECTION,
no rotation: features = N*29) and fits XGB on those raw values. PCA would
rotate into orthogonal components; this does not. Labels/hatches reflect that.

Inputs:
    runs/encodings_sweep/results/metrics.tsv          (blosum50 xgb row)
    runs/handcrafted_sweep/results/handcrafted_sweep_delong_stats.tsv  (champion)
    runs/ev_matched_topXvariance_sweep/results/model_matrix_*.tsv       (ev best)
    <cnn-dirs>/models/cnn/cv_results_*.json           (nested CNN, cv_summary.auc_roc)

Outputs (default runtag=cnn_compare):
    runs/cnn_compare/plots/cnn_compare_auc.png (+ .csv)
    runs/cnn_compare/results/cnn_compare_all.csv

Usage:
    python src/investigation/28_cnn_compare_barplot.py
    python src/investigation/28_cnn_compare_barplot.py --runtag cnn_compare \\
        --cnn-dirs runs/cnn_cv runs/cnn_cv_if runs/cnn_2layer runs/cnn_simple_test
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent / "pipeline" / "python"))
from config import PROJECT_ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FALLBACK = {
    "blosum50": (0.7633, 0.0044),
    "handcrafted_blosum50": (0.7665, 0.0045),
    "esmc_var": (0.7049, 0.0059),
    "esmif_var": (0.6489, 0.0056),
}

COLORS = {
    "encoding": "#BBBBBB",
    "champion": "#8A8A8A",
    "esmc": "#EE6677",
    "esmif": "#4477AA",
}

VAR_RE = re.compile(r"var(\d+)")


def load_blosum(path: Path):
    if path.exists():
        with open(path) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r.get("feature_set") == "blosum50" and r.get("model_key") == "xgb":
                    return (float(r["cv_mean_auc_roc"]), float(r["cv_std_auc_roc"]))
    return None


def load_champion(path: Path):
    """(pooled DeLong AUC, None) — delong stats carry no std; JSON std preferred."""
    if path.exists():
        with open(path) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r.get("explicit_arm") == "handcrafted_blosum50":
                    return (float(r["explicit_auc"]), None)
    return None


def load_ev_best(ev_results_dir: Path):
    """Best row per embedding from ev-matched model_matrix_*.tsv files.

    Returns {emb: (mean, std, features, varN, result_file)} with
    emb in {"esmc", "esmif"}. feature_set values on disk are
    emb_esmc300m_win_padtoken and emb_esmif_eosrepeat (note: no _win_padtoken
    / _bos_repeat suffix for the esmif ev arm).
    """
    import pandas as pd
    files = sorted(ev_results_dir.glob("model_matrix_*.tsv"))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    df = df.drop_duplicates("result_file")
    out = {}
    for emb, keys in (("esmc", ("emb_esmc300m_win_padtoken",)),
                      ("esmif", ("emb_esmif_eosrepeat", "emb_esmif_eos_bos_repeat"))):
        sub = df[df["feature_set"].isin(list(keys))]
        if sub.empty:
            continue
        best = sub.loc[sub["cv_mean_auc_roc"].idxmax()]
        m = VAR_RE.search(str(best["result_file"]))
        varn = int(m.group(1)) if m else -1
        out[emb] = (float(best["cv_mean_auc_roc"]), float(best["cv_std_auc_roc"]),
                    int(best["features"]), varn, str(best["result_file"]))
    return out


def load_cnn_jsons(cnn_dirs):
    """(model_key, features_key) -> (auc, std, path). Latest file wins per key."""
    out = {}
    for d in cnn_dirs:
        md = Path(d)
        cands = []
        if md.is_dir():
            cands += sorted(md.glob("models/cnn/cv_results_*.json"))
            cands += sorted(md.glob("models/*/cv_results_*.json"))
            cands += sorted(md.glob("cv_results_*.json"))
        for f in cands:
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                print(f"  WARN: unreadable {f} — skipped")
                continue
            summ = r.get("cv_summary", {}).get("auc_roc", {})
            if "mean" not in summ:
                continue
            key = (r.get("model_key", ""), r.get("features_key", ""))
            if not all(key):
                continue
            prev = out.get(key)
            stamp = r.get("timestamp", "")
            if prev is None or str(stamp) >= str(prev[3]):
                out[key] = (float(summ["mean"]), float(summ.get("std", 0.0)),
                            str(f), str(stamp))
    return {(k[0], k[1]): (v[0], v[1], v[2]) for k, v in out.items()}


def draw_figure(bars, png, csvp):
    fig, ax = plt.subplots(figsize=(15, 5.8))
    rows = []
    for i, (label, val, color, hatch, _arm) in enumerate(bars):
        if val is None or val[0] is None:
            print(f"  WARN: missing result for bar {label!r} — skipped")
            continue
        mean, std = val[0], val[1] or 0.0
        kw = {"hatch": hatch} if hatch else {}
        ax.bar(i, mean, yerr=std, capsize=3, width=0.62,
               color=color, edgecolor="black", linewidth=0.6, **kw)
        ax.text(i, mean + std + 0.006, f"{mean:.3f}",
                ha="center", fontsize=8, fontweight="bold")
        rows.append({"bar": label.replace("\n", " "),
                     "auc_roc": f"{mean:.6f}", "std": f"{std:.6f}"})
    present = [(i, b) for i, b in enumerate(bars) if b[1] is not None and b[1][0] is not None]
    ax.set_xticks([i for i, _ in present])
    ax.set_xticklabels([b[0] for _, b in present], fontsize=8)
    ax.set_ylabel("AUC-ROC (nested CV, mean +/- std)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.legend(handles=[
        Patch(facecolor=COLORS["encoding"], label="BLOSUM50"),
        Patch(facecolor=COLORS["champion"], label="handcrafted+blosum50 champ"),
        Patch(facecolor=COLORS["esmc"], label="ESM-C-300M padtoken"),
        Patch(facecolor=COLORS["esmif"], label="ESM-IF eos/bos-repeat"),
        Patch(facecolor="white", hatch="...",
              label="top-var select (no rotation)"),
        Patch(facecolor="white", hatch="",
              label="CNN raw (no PCA/select)"),
    ], loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("BLOSUM baselines vs top-var selection vs CNN (decided embeddings)",
                 fontsize=12)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bar", "auc_roc", "std"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {png}\n  wrote {csvp}")


def main():
    ap = argparse.ArgumentParser(description="CNN vs baseline comparison barplot")
    ap.add_argument("--runtag", default="cnn_compare")
    ap.add_argument("--cnn-dirs", nargs="*", default=[
        "runs/cnn_cv", "runs/cnn_cv_if", "runs/cnn_2layer", "runs/cnn_simple_test"],
        help="run dirs searched for models/cnn/cv_results_*.json (hub1 paths OK)")
    ap.add_argument("--ev-dir", default="runs/ev_matched_topXvariance_sweep/results")
    args = ap.parse_args()

    root = PROJECT_ROOT
    run_dir = root / "runs" / args.runtag

    blosum = load_blosum(root / "runs/encodings_sweep/results/metrics.tsv") \
        or FALLBACK["blosum50"]
    champ = load_champion(root / "runs/handcrafted_sweep/results/handcrafted_sweep_delong_stats.tsv") \
        or FALLBACK["handcrafted_blosum50"]
    print(f"  blosum50             {blosum[0]:.4f} +/- {blosum[1]:.4f}")
    print(f"  champ handcrafted+bl {champ[0]:.4f} (std {'n/a' if not champ[1] else f'{champ[1]:.4f}'})")

    ev = load_ev_best(root / args.ev_dir)
    for emb in ("esmc", "esmif"):
        if emb in ev:
            mean, std, feats, varn, src = ev[emb]
            print(f"  {emb} top-var best  {mean:.4f} +/- {std:.4f}  (var{varn} -> {feats} feats)")
            print(f"    src: {src}")
        else:
            print(f"  WARN: no ev-matched rows for {emb} — fallback used")

    cnn = load_cnn_jsons([root / d if not str(d).startswith("/") else Path(d)
                          for d in args.cnn_dirs])
    print(f"  {len(cnn)} CNN results loaded")
    for (m, f), (a, s, p) in sorted(cnn.items()):
        print(f"    {m:<12} {f:<32} {a:.4f} +/- {s:.4f}  {Path(p).name}")

    esmc_var = ev.get("esmc", (FALLBACK["esmc_var"],))[0:2] if "esmc" in ev else FALLBACK["esmc_var"]
    esmif_var = ev.get("esmif", (FALLBACK["esmif_var"],))[0:2] if "esmif" in ev else FALLBACK["esmif_var"]
    esmc_vn = ev["esmc"][3] if "esmc" in ev else -1
    esmif_vn = ev["esmif"][3] if "esmif" in ev else -1

    def cnn_val(model, feats_list):
        for f in feats_list:
            if (model, f) in cnn:
                return cnn[(model, f)][:2]
        return None

    ESMC_KEYS = ["emb_esmc300m_win_padtoken"]
    ESMIF_KEYS = ["emb_esmif_eos_bos_repeat", "emb_esmif_eosrepeat"]

    bars = [
        ("BLOSUM50", blosum, COLORS["encoding"], None, "blosum50"),
        ("handcrafted\n+blosum50\n(champ)", champ, COLORS["champion"], None, "champ"),
        (f"top-var\nESM-C\n(var{esmc_vn})", esmc_var, COLORS["esmc"], "...", "esmc_var"),
        (f"top-var\nESM-IF\n(var{esmif_vn})", esmif_var, COLORS["esmif"], "...", "esmif_var"),
        ("CNN-1L\nESM-C", cnn_val("cnn_simple", ESMC_KEYS), COLORS["esmc"], None, "cnn1_esmc"),
        ("CNN-2L\nESM-C", cnn_val("cnn_2layer", ESMC_KEYS), COLORS["esmc"], None, "cnn2_esmc"),
        ("CNN-1L\nESM-IF", cnn_val("cnn_simple", ESMIF_KEYS), COLORS["esmif"], None, "cnn1_esmif"),
        ("CNN-2L\nESM-IF", cnn_val("cnn_2layer", ESMIF_KEYS), COLORS["esmif"], None, "cnn2_esmif"),
    ]

    plots_dir = run_dir / "plots"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    draw_figure(bars, plots_dir / "cnn_compare_auc.png",
                plots_dir / "cnn_compare_auc.csv")

    agg = results_dir / "cnn_compare_all.csv"
    with open(agg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bar", "auc_roc", "std", "note"])
        w.writeheader()
        notes = {
            "blosum50": "xgb encodings_sweep",
            "champ": "xgb handcrafted_sweep DeLong pooled 0.7665",
            "esmc_var": "xgb top-var feature selection, no PCA rotation",
            "esmif_var": "xgb top-var feature selection, no PCA rotation",
        }
        for label, val, _c, _h, arm in bars:
            mean = f"{val[0]:.6f}" if val else ""
            std = f"{val[1]:.6f}" if val and val[1] else ""
            w.writerow({"bar": label.replace("\n", " "), "auc_roc": mean,
                        "std": std, "note": notes.get(arm, "cnn nested CV")})
    print(f"  wrote {agg}")

    print("\n" + "=" * 60 + "\n  CAPTION (paste below the figure):\n" + "=" * 60)
    print("AUC-ROC (nested CV, mean +/- std over 6 outer buckets). Grey: BLOSUM50 "
          "(0.763) and handcrafted+blosum50 champion (0.7665, DeLong pooled). Dotted: "
          "ev-matched top-variance SELECTION (XGB on raw dims ranked by variance, "
          "N*29 features, no PCA rotation — e.g. ESM-C var793 = 793 dims x 29 slots). "
          "Solid: CNN on raw windows (1-layer baseline vs 2-layer, naive GAP, no PCA/select). "
          "ESM-C-300M padtoken red; ESM-IF eos/bos-repeat blue.")
    print("=" * 60)


if __name__ == "__main__":
    main()
