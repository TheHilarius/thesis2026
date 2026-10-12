#!/usr/bin/env python3
"""
16_esmc300_vs_600.py
ESM-C-300M vs ESM-C-600M (pad-token) comparison figure.

Two panels over the same 5 bars:
    Sparse | BLOSUM50 | ESM-C-300M pad | ESM-C-600M pad | ESM-IF EOS/BOS-repeat
  1. held-out AUC-ROC (outer-fold mean), y fixed 0-1 — with a significance
     bracket between the two ESM-C bars: paired DeLong on pooled OOF
     predictions (same rows), Holm over the 1-pair family (p_adj == p;
     shown in the padmode_choice annotation format)
  2. nested-CV wall time, log-y — bars labeled wall (core-h); bracket =
     paired t-test on the 30 nested-CV fold durations (fold_seconds:
     6 outer x 5 inner, matched by (outer,inner) index across runs)

Both runs used 12 cores; memory left to a later task (peak RSS not
instrumented on the cluster — req_mem_gb recorded in the CSV only).

The 300M bar is skipped (with a note) until its JSON lands under --esmc-dir.
Sparse/BLOSUM/ESM-IF read from --padmode-dir; the 600M padtoken JSON is
looked up in --esmc-dir first, then --padmode-dir.

Usage:
    env_esmc/bin/python src/investigation/16_esmc300_vs_600.py

Outputs:
    runs/esmc300vs600/plots/esmc300_vs_600.{png,csv}
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_rel

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
for _p in (str(SRC_DIR), str(PIPELINE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import SPLIT_DATA_PATH  # noqa: E402


# ── Paired DeLong for two correlated AUCs (Sun & Xu midrank structure) ──
def delong_structure(y: np.ndarray, p: np.ndarray):
    pos = p[y == 1]
    neg = p[y == 0]
    m, n = len(pos), len(neg)
    neg_sorted = np.sort(neg)
    left = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (left + 0.5 * (right - left)) / n
    pos_sorted = np.sort(pos)
    leftp = np.searchsorted(pos_sorted, neg, side="left")
    rightp = np.searchsorted(pos_sorted, neg, side="right")
    v01 = 1.0 - (leftp + 0.5 * (rightp - leftp)) / m
    return v10, v01


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    v10a, v01a = delong_structure(y, p_a)
    v10b, v01b = delong_structure(y, p_b)
    auc_a, auc_b = float(v10a.mean()), float(v10b.mean())
    m, n = len(v10a), len(v01a)
    var_d = (np.var(v10a, ddof=1) / m + np.var(v01a, ddof=1) / n
             + np.var(v10b, ddof=1) / m + np.var(v01b, ddof=1) / n
             - 2.0 * (np.cov(v10a, v10b, ddof=1)[0, 1] / m
                      + np.cov(v01a, v01b, ddof=1)[0, 1] / n))
    return auc_a, auc_b, float(var_d)

CORES = 12          # both jobs ran --cpus-per-task=12
REQ_MEM_GB = 56     # both jobs --mem=56G (request ceiling, not measured)

# (bar key, label, features_key glob-part, source, color)
BARS = [
    ("sparse", "Sparse", "sparse", "padmode", "#BBBBBB"),
    ("blosum", "BLOSUM50", "blosum", "padmode", "#BBBBBB"),
    ("esmc300m", "ESM-C-300M\npad-token", "emb_esmc300m_win_padtoken",
     "esmc", "#F5A8A8"),
    ("esmc600m", "ESM-C-600M\npad-token", "emb_esmc_win_padtoken_pca6389",
     "either", "#EE6677"),
    ("esmif_eos", "ESM-IF\nEOS/BOS-repeat", "emb_esmif_eos_bos_repeat",
     "padmode", "#4477AA"),
]


def find_json(root: Path, part: str):
    hits = sorted(root.glob(f"cv_results_xgb_{part}*.json"))
    return hits[-1] if hits else None


def read_run(path: Path):
    d = json.load(open(path))
    ovm = d.get("outer_val_metrics", [])
    auc = (sum(o["auc_roc"] for o in ovm) / len(ovm)) if ovm else float("nan")
    runtime_h = d.get("runtime_seconds", 0.0) / 3600.0
    fold_s = np.asarray(d.get("fold_seconds", []), dtype=float)
    return d, auc, runtime_h, fold_s


def reconstruct_oof(res: dict) -> np.ndarray:
    """Full-length pooled OOF probability vector (fold scatter)."""
    import pandas as pd
    df = pd.read_csv(SPLIT_DATA_PATH, usecols=["label", "fold"])
    folds = df["fold"].to_numpy()
    n = len(df)
    probs = np.full(n, np.nan)
    for fk, entry in res["outer_val_predictions"].items():
        mask = folds == int(fk)
        probs[mask] = np.asarray(entry["y_prob"], dtype=float)
    if np.isnan(probs).any():
        raise ValueError("rows never predicted")
    return probs, df["label"].to_numpy()


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_p_pair(p, p_adj):
    a = "p_adj<0.001" if p_adj < 0.001 else f"p_adj={p_adj:.2g}"
    txt = "p<0.001" if p < 0.001 else f"p={p:.2g}"
    return f"{txt}, {a}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--padmode-dir", default="runs/padmode_choice/models/flat")
    ap.add_argument("--esmc-dir", default="runs/esmc300vs600/models/flat")
    ap.add_argument("--out-dir", default="runs/esmc300vs600/plots")
    args = ap.parse_args()

    pad_dir = Path(args.padmode_dir)
    esmc_dir = Path(args.esmc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, labels, colors, aucs, runtimes = [], [], [], [], []
    loaded = {}   # bar key -> (res, probs_y or None, fold_s)
    for key, label, part, src, color in BARS:
        path = None
        if src in ("esmc", "either"):
            path = find_json(esmc_dir, part)
        if path is None and src in ("padmode", "either"):
            path = find_json(pad_dir, part)
        if path is None:
            print(f"  [pending] {key}: no JSON matching '{part}'")
            rows.append({"bar": key, "label": label.replace("\n", " "),
                        "status": "pending"})
            continue
        res, auc, rt, fold_s = read_run(path)
        labels.append(label)
        colors.append(color)
        aucs.append(auc)
        runtimes.append(rt)
        rows.append({"bar": key, "label": label.replace("\n", " "),
                    "auc": auc, "runtime_hours": round(rt, 2),
                    "cores": CORES, "cpu_hours": round(rt * CORES, 1),
                    "req_mem_gb": REQ_MEM_GB, "status": "ok",
                    "result_file": path.name})
        # OOF + fold vectors only where later tests need them
        if key in ("esmc300m", "esmc600m"):
            probs, y = reconstruct_oof(res)
            loaded[key] = {"probs": probs, "y": y, "fold_s": fold_s,
                           "auc": auc}

    if not aucs:
        raise SystemExit("No run JSONs found — nothing to plot")

    # ── Tests: 300M vs 600M ──
    auc_p = auc_padj = rt_p = rt_padj = rt_t = None
    if "esmc300m" in loaded and "esmc600m" in loaded:
        a3, a6 = loaded["esmc300m"], loaded["esmc600m"]
        # invariant: DeLong AUC == pooled-OOF roc_auc (bars show the
        # mean-of-fold AUC — slightly different estimand, as everywhere
        # in this campaign)
        from sklearn.metrics import roc_auc_score
        assert abs(delong_test(a3["y"], a3["probs"], a6["probs"])[0]
                   - roc_auc_score(a3["y"], a3["probs"])) < 1e-12
        assert abs(delong_test(a3["y"], a3["probs"], a6["probs"])[1]
                   - roc_auc_score(a6["y"], a6["probs"])) < 1e-12
        _, _, var_d = delong_test(a3["y"], a3["probs"], a6["probs"])
        z = (a3["auc"] - a6["auc"]) / np.sqrt(var_d)
        auc_p = 2 * norm.sf(abs(z))
        auc_padj = auc_p  # Holm over a 1-pair family
        # runtime: paired t-test on the 30 nested-CV fold durations
        rt_t, rt_p = ttest_rel(a3["fold_s"], a6["fold_s"])
        rt_padj = rt_p
        print(f"  DeLong AUC 300-vs-600: z={z:.3f} p={auc_p:.4g}")
        print(f"  fold-seconds t-test: t={rt_t:.3f} p={rt_p:.4g}")

    for r in rows:
        r["p_delong"] = auc_p if r["bar"] == "esmc600m" else ""
        r["p_delong_holm"] = auc_padj if r["bar"] == "esmc600m" else ""
        r["t_runtime"] = rt_t if r["bar"] == "esmc600m" else ""
        r["p_runtime"] = rt_p if r["bar"] == "esmc600m" else ""
        r["p_runtime_holm"] = rt_padj if r["bar"] == "esmc600m" else ""

    csv_path = out_dir / "esmc300_vs_600.csv"
    fields = ["bar", "label", "auc", "runtime_hours", "cores", "cpu_hours",
              "req_mem_gb", "status", "result_file",
              "p_delong", "p_delong_holm",
              "t_runtime", "p_runtime", "p_runtime_holm"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved: {csv_path}")

    n = len(labels)
    x = np.arange(n)
    keys_present = [r["bar"] for r in rows if r.get("status") == "ok"]
    fig, (ax_auc, ax_rt) = plt.subplots(1, 2, figsize=(12.5, 5.5))

    # ── AUC panel (fixed 0-1) ──
    ax_auc.bar(x, aucs, color=colors, width=0.62, edgecolor="black",
               linewidth=0.5)
    for xi, v in zip(x, aucs):
        ax_auc.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5)
    ax_auc.set_ylim(0, 1.0)
    ax_auc.set_ylabel("AUC-ROC")
    ax_auc.set_title("Held-out AUC-ROC")
    if auc_p is not None:
        i3, i6 = keys_present.index("esmc300m"), keys_present.index("esmc600m")
        y_br = 0.86
        ax_auc.plot([i3, i3, i6, i6],
                    [y_br * 0.985, y_br, y_br, y_br * 0.985],
                    color="black", lw=0.9)
        star = f" ({stars(auc_padj)})"
        ax_auc.text((i3 + i6) / 2, y_br + 0.012,
                    f"{fmt_p_pair(auc_p, auc_padj)}{star}",
                    ha="center", va="bottom", fontsize=8.5)

    # ── Runtime panel (log-y) ──
    ax_rt.bar(x, runtimes, color=colors, width=0.62, edgecolor="black",
              linewidth=0.5)
    for xi, v, key in zip(x, runtimes, keys_present):
        ax_rt.text(xi, v * 1.12, f"{v:.1f}h ({v * CORES:.0f} core-h)",
                   ha="center", va="bottom", fontsize=8.5)
    ax_rt.set_yscale("log")
    ax_rt.set_ylim(min(runtimes) * 0.4, max(runtimes) * 6)
    ax_rt.set_ylabel("Hours (log scale)")
    ax_rt.set_title("Nested-CV wall time — 12 cores, ≤56G request")
    if rt_p is not None:
        i3, i6 = keys_present.index("esmc300m"), keys_present.index("esmc600m")
        y_br = max(runtimes) * 1.9
        ax_rt.plot([i3, i3, i6, i6],
                   [y_br * 0.985, y_br, y_br, y_br * 0.985],
                   color="black", lw=0.9)
        star = f" ({stars(rt_padj)})"
        ax_rt.text((i3 + i6) / 2, y_br * 1.08,
                   f"{fmt_p_pair(rt_p, rt_padj)}{star}",
                   ha="center", va="bottom", fontsize=8.5)

    for ax in (ax_auc, ax_rt):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ESM-C-300M vs 600M — pad-token embeddings @ 90% EV "
                 "(+ references)", fontsize=13, y=1.0)
    fig.tight_layout()
    png_path = out_dir / "esmc300_vs_600.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
