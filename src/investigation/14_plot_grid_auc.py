#!/usr/bin/env python3
"""
14_plot_grid_auc.py
Bar plots of the padfull_flat modelling grid (XGBoost, handcrafted_sparse + windows).

Figure sets:
  1. Overviews — full-data AUC-ROC and MCC per run (mean +/- std over the
     6 CV/held-out buckets), fixed bar order, handcrafted_sparse first.
  2. Saturation — AUC vs explained-variance ladder per toolkit (the total
     view): 4 padmode groups, one bar per PC level, dashed baseline line.
  3. Per-PC — one figure per (toolkit, k) in two sets: full data (per_pc/)
     and the informative padded SUBSET rows (per_pc_subset/; >=1 pad slot
     AND >=1 real slot; ~1443 ESM-C / ~1442 ESM-IF rows), where pad-mode
     signal actually lives. Fixed order: handcrafted_sparse -> zero ->
     eos_bos_repeat -> boundary -> pad_token/pad.

Titles always name the composite feature set (handcrafted_sparse_esmc /
handcrafted_sparse_esmif) — embeddings are never shown alone.

Plus grid_table.csv with every metric + wall time joined from the harness logs.

Inputs (relative to repo root):
    runs/padfull_flat_20261001_1645/models/{flat,flat_raw,slot}/cv_results_*.json
    logs/padfull_flat_*.out                      — "DONE in <s>s" per task
    data/processed/df_all_with_folds.csv         — label, fold (SPLIT_DATA_PATH)
    data/processed/embeddings/esmc_context_embeddings_zeropad.h5  — pad_mask
    data/processed/embeddings/esm-if_test_zero.h5                 — pad_mask

Output (under the campaign folder):
    runs/padfull_flat_20261001_1645/plots/grid_auc_full.png       — overviews
    runs/padfull_flat_20261001_1645/plots/grid_mcc_full.png
    runs/padfull_flat_20261001_1645/plots/grid_saturation_<toolkit>.png
        — TOTAL view: AUC vs explained-variance ladder, 4 padmode groups,
          one bar per PC level (EV% tick label), dashed handcrafted_sparse
          reference line; top panel full data, bottom panel subset rows.
    runs/padfull_flat_20261001_1645/plots/per_pc/grid_auc_<toolkit>_<k>.png
    runs/padfull_flat_20261001_1645/plots/per_pc_subset/grid_auc_<toolkit>_<k>.png
        — ONE FIGURE PER PC LEVEL (or raw): all padmodes at that k vs
          handcrafted_sparse, fixed order, full data / subset rows.
          This is the within-PC pick-best-padmode view.
    runs/padfull_flat_20261001_1645/results/grid_table.csv
    runs/padfull_flat_20261001_1645/results/grid_summary.tsv
        — one row per run (48 grid + sparse_raw/blosum_raw reference arms
          borrowed from runs/padmode_choice/): metrics + subset AUCs under
          both toolkit masks + runtime + n_features.
    runs/padfull_flat_20261001_1645/results/grid_comparisons.tsv
        — paired t-tests on the 6 fold-level SUBSET AUCs, BH-FDR corrected
          across every test: within-PC padmode pairs, each run vs the two
          raw-encoding arms, sparse_raw vs blosum_raw.

Usage:
    python src/investigation/14_plot_grid_auc.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SRC_DIR.parent / "pipeline" / "python"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

from config import PROJECT_ROOT, SPLIT_DATA_PATH, EMBEDDING_DIR

CAMPAIGN = PROJECT_ROOT / "runs" / "padfull_flat_20261001_1645"
MODELS_DIR = CAMPAIGN / "models"
PLOTS_DIR = CAMPAIGN / "plots"
OUT_CSV = CAMPAIGN / "results" / "grid_table.csv"
OUT_TSV = CAMPAIGN / "results" / "grid_summary.tsv"
OUT_PAIRS_TSV = CAMPAIGN / "results" / "grid_comparisons.tsv"
LOGS_DIR = PROJECT_ROOT / "logs"
# raw-encoding reference arms borrowed from Oliver's padmode-choice comparison
# (same config / folds / RANDOM_STATE -> fold-level pairing is valid)
PADMODE_CHOICE_DIR = PROJECT_ROOT / "runs" / "padmode_choice" / "models" / "flat"

PADMASK_FILES = {
    "ESM-C": EMBEDDING_DIR / "esmc_context_embeddings_zeropad.h5",
    "ESM-IF": EMBEDDING_DIR / "esm-if_test_zero.h5",
}

# filename/feature-key -> (toolkit, padtype label)
FEATURE_MAP = {
    "esmc_win_zeropad": ("ESM-C", "zero"),
    "esmc_win_impute_boundary": ("ESM-C", "boundary"),
    "esmc_win_impute_bos_eos": ("ESM-C", "eos_bos_repeat"),
    "esmc_win_padtoken": ("ESM-C", "pad_token"),
    "esmif_win": ("ESM-IF", "zero"),
    "esmif_boundary": ("ESM-IF", "boundary"),
    "esmif_eos_bos_repeat": ("ESM-IF", "eos_bos_repeat"),
    "esmif_pad": ("ESM-IF", "pad"),
    "handcrafted_sparse": ("baseline", "-"),
}

PADTYPE_COLOR = {
    "zero": "#4C72B0",
    "boundary": "#DD8452",
    "eos_bos_repeat": "#55A868",
    "pad_token": "#C44E52",
    "pad": "#C44E52",
    "-": "#7F7F7F",
}

# fixed bar order in EVERY figure, both toolkits:
#   handcrafted_sparse (baseline, "-") -> zero -> eos_bos_repeat -> boundary
#   -> pad_token (ESM-C) / pad (ESM-IF)
PADMODE_ORDER = ["-", "zero", "eos_bos_repeat", "boundary", "pad_token", "pad"]
_MODE_RANK = {p: i for i, p in enumerate(PADMODE_ORDER)}
_TOOLKIT_RANK = {"baseline": 0, "ESM-C": 1, "ESM-IF": 2}


def _order_overview(df: pd.DataFrame) -> pd.DataFrame:
    """baseline first, then ESM-C then ESM-IF; within a toolkit k asc
    (raw last), pad modes in PADMODE_ORDER inside each k."""
    d = df.copy()
    d["_tk"] = d["toolkit"].map(_TOOLKIT_RANK)
    d["_mode"] = d["padtype"].map(_MODE_RANK)
    d["_k"] = d["pca"].fillna(10**9)
    return d.sort_values(["_tk", "_k", "_mode"]).reset_index(drop=True)


def _order_pc(df: pd.DataFrame) -> pd.DataFrame:
    """One PC level: baseline first, then the pad modes in PADMODE_ORDER."""
    d = df.copy()
    d["_mode"] = d["padtype"].map(_MODE_RANK)
    return d.sort_values("_mode").reset_index(drop=True)


# exact explained-variance threshold each ladder k was built at
# (padmode-invariant by construction — the flat-full fit uses fully-real rows)
EV_BY_K = {
    "ESM-C": {"97": 50, "2346": 80, "3888": 85, "6389": 90, "10895": 95,
              "20142": 99, "raw": 100},
    "ESM-IF": {"345": 50, "2125": 80, "2974": 85, "4308": 90, "6700": 95,
               "11243": 99, "raw": 100},
}
PADMODES_PER_TOOLKIT = {
    "ESM-C": ["zero", "eos_bos_repeat", "boundary", "pad_token"],
    "ESM-IF": ["zero", "eos_bos_repeat", "boundary", "pad"],
}
FEATURE_SET_NAME = {"ESM-C": "handcrafted_sparse_esmc",
                    "ESM-IF": "handcrafted_sparse_esmif"}


def _ladder_keys(toolkit: str) -> list[str]:
    """k_labels for a toolkit in ladder order, raw last."""
    return sorted(EV_BY_K[toolkit],
                  key=lambda k: (10**9 if k == "raw" else int(k)))

JSON_RE = re.compile(
    r"^cv_results_xgb_handcrafted_sparse_"
    r"(?P<feat>.+?)"
    r"_pca(?P<k>\d+)"
    r"_(?P<mode>flat|flat_raw)"
    r"_(?P<ts>\d{8}_\d{6})\.json$"
)
# raw: pca_mode=flat_raw, run_tag carries no pca segment
JSON_RAW_RE = re.compile(
    r"^cv_results_xgb_handcrafted_sparse_"
    r"(?P<feat>.+?)_flat_raw_(?P<ts>\d{8}_\d{6})\.json$"
)
# baseline: run_tag has no pca/mode segment (slot mode, no embeddings)
JSON_SLOT_RE = re.compile(
    r"^cv_results_xgb_handcrafted_sparse_(?P<ts>\d{8}_\d{6})\.json$"
)
# raw-encoding arms borrowed from padmode_choice (sparse / blosum alone)
JSON_RAW_ENC_RE = re.compile(
    r"^cv_results_xgb_(?P<feat>sparse|blosum)_flat_(?P<ts>\d{8}_\d{6})\.json$"
)
LOG_DONE_RE = re.compile(r"DONE in ([0-9.]+)s")
LOG_RESULTS_RE = re.compile(r"Results file:\s+.*/([^/]+\.json)")


def discover_runs() -> list[dict]:
    """One entry per grid JSON in the campaign folder."""
    runs = []
    for mode in ("flat", "flat_raw", "slot"):
        for p in sorted((MODELS_DIR / mode).glob("cv_results_xgb_handcrafted_sparse_*.json")):
            m = (JSON_RE.match(p.name) or JSON_RAW_RE.match(p.name)
                 or JSON_SLOT_RE.match(p.name))
            if not m:
                print(f"  [skip] unrecognized: {p.name}")
                continue
            feat = m.group("feat") if "feat" in m.groupdict() else "handcrafted_sparse"
            if feat not in FEATURE_MAP:
                print(f"  [skip] unknown feature key: {feat}")
                continue
            toolkit, padtype = FEATURE_MAP[feat]
            k = int(m.group("k")) if m.groupdict().get("k") else None
            if mode == "flat_raw":
                k_label = "raw"
            elif mode == "slot" or k is None:
                k_label = "baseline"
            else:
                k_label = str(k)
            runs.append({
                "path": p, "basename": p.name, "mode": mode,
                "features_key": feat, "toolkit": toolkit, "padtype": padtype,
                "pca": k, "k_label": k_label,
            })
    return runs


def load_padded_masks() -> dict[str, np.ndarray]:
    """Per-toolkit informative padded flag in df order (see 09_padmode_subset_auc)."""
    masks = {}
    for toolkit, path in PADMASK_FILES.items():
        if not path.exists():
            raise SystemExit(f"Missing pad_mask source: {path}")
        with h5py.File(path, "r") as f:
            pm = f["pad_mask"][:].astype(bool)   # (rows, 29)
            ri = f["row_indices"][:]
        if not np.array_equal(np.sort(ri), np.arange(len(ri))):
            raise SystemExit(f"{path.name}: row_indices not a permutation of 0..n-1")
        informative = pm.any(axis=1) & ~pm.all(axis=1)
        by_df = np.zeros(len(ri), dtype=bool)
        by_df[ri] = informative
        masks[toolkit] = by_df
        print(f"  {toolkit}: informative padded rows = {int(by_df.sum())}")
    return masks


def parse_runtimes() -> dict[str, float]:
    """cv_results basename -> wall seconds, from the harness .out logs."""
    out = {}
    for log in sorted(LOGS_DIR.glob("padfull_flat_*.out")):
        text = log.read_text(errors="replace")
        m_res = LOG_RESULTS_RE.search(text)
        if not m_res:
            continue
        dones = LOG_DONE_RE.findall(text)
        if not dones:
            continue
        out[m_res.group(1)] = float(dones[-1])
    return out


def assemble_oof(path: Path, df: pd.DataFrame,
                 basename: str = "") -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Out-of-fold y_true/y_prob in df order — every row predicted exactly once."""
    with open(path) as fh:
        res = json.load(fh)
    preds = res["outer_val_predictions"]
    n = len(df)
    y_true = np.full(n, -1, dtype=int)
    y_prob = np.full(n, np.nan, dtype=float)
    for fold_key, entry in preds.items():
        fold = int(fold_key)
        bucket = df[df["fold"] == fold]
        pos = bucket.index.to_numpy()
        yt = np.asarray(entry["y_true"], dtype=int)
        yp = np.asarray(entry["y_prob"], dtype=float)
        if len(yt) != len(pos):
            raise SystemExit(f"{basename} fold {fold}: pred/bucket size mismatch")
        if not np.array_equal(yt, bucket["label"].to_numpy()):
            raise SystemExit(f"{basename} fold {fold}: y_true != df labels")
        y_true[pos] = yt
        y_prob[pos] = yp
    if (y_true < 0).any() or np.isnan(y_prob).any():
        raise SystemExit(f"{basename}: missing predictions")
    return y_true, y_prob, res, preds


def _fold_subset_aucs(preds: dict, df: pd.DataFrame, y_true: np.ndarray,
                      y_prob: np.ndarray, padded: np.ndarray) -> list[float]:
    """Per-fold subset AUCs (the paired-test units for the t-tests)."""
    out = []
    for fold in sorted(int(k) for k in preds):
        idx = df.index[df["fold"] == fold].to_numpy()
        sel = idx[padded[idx]]
        if len(sel) and y_true[sel].min() != y_true[sel].max():
            out.append(float(roc_auc_score(y_true[sel], y_prob[sel])))
    return out


def load_run(row: dict, df: pd.DataFrame, masks: dict[str, np.ndarray],
             runtimes: dict[str, float]) -> dict:
    """Metrics + per-fold subset AUCs for one campaign JSON."""
    y_true, y_prob, res, preds = assemble_oof(row["path"], df, row["basename"])
    cs = res["cv_summary"]
    metrics = {
        "auc": cs["auc_roc"]["mean"], "auc_std": cs["auc_roc"]["std"],
        "mcc": cs["mcc"]["mean"], "mcc_std": cs["mcc"]["std"],
        "n_features": res["config"]["n_features"],
        "heldout_auc": res["outer_val_metrics"][-1]["auc_roc"],
    }
    secs = runtimes.get(row["basename"])
    metrics["runtime_h"] = secs / 3600.0 if secs else np.nan
    metrics["auc_pooled"] = float(roc_auc_score(y_true, y_prob))

    toolkit = row["toolkit"]
    # subset AUC per toolkit mask; fold-level vectors kept for the paired
    # t-tests. Grid runs score their own mask (baseline: both, for the
    # per-PC comparisons); raw-encoding arms are toolkit-agnostic -> both.
    metrics["auc_padded"] = np.nan
    metrics["auc_padded_std"] = np.nan
    for tk in masks:
        padded = masks[tk]
        tk_key = tk.replace("-", "").lower()                # esmc / esmif
        fold_aucs = _fold_subset_aucs(preds, df, y_true, y_prob, padded)
        metrics[f"fold_auc_{tk_key}"] = fold_aucs
        auc = float(roc_auc_score(y_true[padded], y_prob[padded]))
        std = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else np.nan
        metrics[f"auc_padded_{tk_key}"] = auc
        metrics[f"auc_padded_std_{tk_key}"] = std
        if toolkit == tk:
            metrics["auc_padded"] = auc
            metrics["auc_padded_std"] = std
    metrics["n_padded"] = int(masks[toolkit].sum()) if toolkit in masks else 0
    return metrics


def discover_raw_arms() -> list[dict]:
    """sparse_raw / blosum_raw reference runs borrowed from padmode_choice."""
    arms = []
    if not PADMODE_CHOICE_DIR.is_dir():
        print(f"  [skip] no raw-encoding arms at {PADMODE_CHOICE_DIR}")
        return arms
    for p in sorted(PADMODE_CHOICE_DIR.glob("cv_results_xgb_*_flat_*.json")):
        m = JSON_RAW_ENC_RE.match(p.name)
        if not m:
            continue
        arms.append({
            "path": p, "basename": p.name, "mode": "flat",
            "features_key": m.group("feat"), "toolkit": "raw-encoding",
            "padtype": f"{m.group('feat')}_raw", "pca": None,
            "k_label": "raw",
        })
    return arms


def _paired_t(a: list, b: list) -> tuple[float, float]:
    """Two-sided paired t-test on the 6 fold-level subset AUCs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2:
        return np.nan, np.nan
    d = a - b
    if np.allclose(d, d[0]):                      # zero variance guard
        return float(d.mean()), (0.0 if d[0] != 0 else 1.0)
    t, p = ttest_rel(a, b)
    return float(d.mean()), float(p)


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment (NaN-safe)."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotone
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def _ev_pct(toolkit: str, k_label: str):
    if toolkit in EV_BY_K:
        return EV_BY_K[toolkit].get(k_label, "")
    return 100 if k_label == "raw" else ""       # raw arms / baseline


def build_summary_and_tests(df_grid: pd.DataFrame, df_raw: pd.DataFrame,
                            masks: dict[str, np.ndarray]):
    """grid_summary.tsv (one row per run) + grid_comparisons.tsv (paired
    t-tests, SUBSET rows only, one BH family across every p-value).

    Test families:
      within_pc  — every padmode pair inside one (toolkit, k) level
      vs_raw     — each grid run (incl. baseline) vs sparse_raw/blosum_raw
                   under the run's toolkit mask (baseline under both masks)
      raw_vs_raw — sparse_raw vs blosum_raw, once per mask

    Paired t-test on the 6 fold-level subset AUCs. Caveats: n=6 has low
    power; CV folds overlap in training data so p is mildly
    anti-conservative — standard for model comparison, and every run shares
    the identical fold structure so the comparisons themselves are fair.
    """
    all_runs = pd.concat([df_grid, df_raw], ignore_index=True)
    summary_rows = []
    for _, r in all_runs.iterrows():
        summary_rows.append({
            "toolkit": r["toolkit"], "padmode": r["padtype"],
            "k": r["k_label"], "ev_pct": _ev_pct(r["toolkit"], r["k_label"]),
            "n_features": r["n_features"], "runtime_h": r["runtime_h"],
            "auc_mean": r["auc"], "auc_std": r["auc_std"],
            "mcc_mean": r["mcc"], "heldout_auc": r["heldout_auc"],
            "auc_subset_esmc": r["auc_padded_esmc"],
            "auc_subset_std_esmc": r["auc_padded_std_esmc"],
            "auc_subset_esmif": r["auc_padded_esmif"],
            "auc_subset_std_esmif": r["auc_padded_std_esmif"],
            "n_padded_esmc": int(masks["ESM-C"].sum()),
            "n_padded_esmif": int(masks["ESM-IF"].sum()),
        })

    comps = []
    # 1) within-PC padmode pairs (grid runs only)
    for toolkit in ["ESM-C", "ESM-IF"]:
        tk_key = toolkit.replace("-", "").lower()
        sub = df_grid[df_grid["toolkit"] == toolkit]
        for k_label in _ladder_keys(toolkit):
            grp = sub[sub["k_label"] == k_label].copy()
            grp["_m"] = grp["padtype"].map(_MODE_RANK)
            grp = grp.sort_values("_m")
            modes = list(grp["padtype"])
            for i in range(len(modes)):
                for j in range(i + 1, len(modes)):
                    ra = grp.iloc[i]
                    rb = grp.iloc[j]
                    d, p = _paired_t(ra[f"fold_auc_{tk_key}"],
                                     rb[f"fold_auc_{tk_key}"])
                    comps.append({
                        "kind": "within_pc", "toolkit": toolkit, "k": k_label,
                        "mode_a": modes[i], "mode_b": modes[j],
                        "delta_subset": d, "p_ttest": p})

    # 2) every grid run (incl. baseline) vs both raw arms, own-mask rows
    raw_by_type = {r["padtype"]: r for _, r in df_raw.iterrows()}
    for _, r in df_grid.iterrows():
        tk_keys = (["esmc", "esmif"] if r["toolkit"] == "baseline"
                   else [r["toolkit"].replace("-", "").lower()])
        label = "handcrafted_sparse" if r["padtype"] == "-" else r["padtype"]
        for tk_key in tk_keys:
            for raw_name in ("sparse_raw", "blosum_raw"):
                if raw_name not in raw_by_type:
                    continue
                d, p = _paired_t(r[f"fold_auc_{tk_key}"],
                                 raw_by_type[raw_name][f"fold_auc_{tk_key}"])
                comps.append({
                    "kind": "vs_raw", "toolkit": tk_key, "k": r["k_label"],
                    "mode_a": label, "mode_b": raw_name,
                    "delta_subset": d, "p_ttest": p})

    # 3) sparse_raw vs blosum_raw, once per mask
    if len(raw_by_type) == 2:
        sr, br = raw_by_type["sparse_raw"], raw_by_type["blosum_raw"]
        for tk_key in ["esmc", "esmif"]:
            d, p = _paired_t(sr[f"fold_auc_{tk_key}"],
                             br[f"fold_auc_{tk_key}"])
            comps.append({
                "kind": "raw_vs_raw", "toolkit": tk_key, "k": "raw",
                "mode_a": "sparse_raw", "mode_b": "blosum_raw",
                "delta_subset": d, "p_ttest": p})

    comps = pd.DataFrame(comps)
    comps["q_bh"] = _bh(comps["p_ttest"].to_numpy())
    comps = comps.sort_values(["kind", "toolkit", "k", "mode_a", "mode_b"])

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_TSV, sep="\t", index=False)
    comps.to_csv(OUT_PAIRS_TSV, sep="\t", index=False)
    print(f"  saved: {OUT_TSV} ({len(summary_rows)} runs)")
    print(f"  saved: {OUT_PAIRS_TSV} ({len(comps)} paired t-tests, "
          f"subset rows, BH-corrected)")
    return pd.DataFrame(summary_rows), comps


def _bar_label(r: pd.Series) -> str:
    rt = f"  {r['runtime_h']:.0f}h" if pd.notna(r["runtime_h"]) else ""
    name = "handcrafted_sparse" if r["padtype"] == "-" else f"{r['padtype']} {r['k_label']}"
    return f"{name}{rt}"


def plot_metric(df: pd.DataFrame, value: str, std_col: str, title: str,
                outfile: Path, ylabel: str):
    """Vertical bars, 0-based axis, fixed order (baseline -> zero -> eos ->
    boundary -> pad), grouped by k ascending within each toolkit."""
    rows = _order_overview(df)
    n = len(rows)
    x = np.arange(n)
    colors = [PADTYPE_COLOR.get(p, "#7F7F7F") for p in rows["padtype"]]
    vals = rows[value].to_numpy(dtype=float)
    raw_err = rows[std_col].to_numpy(dtype=float)
    errs = np.where(np.isnan(raw_err), 0.0, raw_err)

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * n), 5))
    ax.bar(x, vals, yerr=errs, color=colors, edgecolor="black", linewidth=0.4,
           width=0.8, error_kw=dict(lw=0.8, capsize=2))
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            ax.text(xi, v + errs[int(xi)] + 0.004, f"{v:.3f}",
                    ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_bar_label(r) for _, r in rows.iterrows()],
                       rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, float(np.nanmax(vals + errs)) * 1.18)
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"  saved: {outfile}")


def plot_per_pc(df: pd.DataFrame, plots_root: Path):
    """Single-panel figures per PC level, in two sets.

    plots_root/per_pc/         — full-data AUC per (toolkit, k)
    plots_root/per_pc_subset/  — subset AUC per (toolkit, k), n=1443/1442

    Fixed bar order in every figure, both toolkits, vertical bars, 0-based:
    handcrafted_sparse -> zero -> eos_bos_repeat -> boundary -> pad_token/pad.
    Titles name the composite feature set — the runs are
    handcrafted_sparse_esmc/esmif(pc, padmode), never embeddings alone.
    """
    dir_full = plots_root / "per_pc"
    dir_subset = plots_root / "per_pc_subset"
    dir_full.mkdir(parents=True, exist_ok=True)
    dir_subset.mkdir(parents=True, exist_ok=True)
    base = df[df["padtype"] == "-"]
    for toolkit in ["ESM-C", "ESM-IF"]:
        sub = df[df["toolkit"] == toolkit].copy()
        if sub.empty or base.empty:
            continue
        tk_key = toolkit.replace("-", "").lower()          # esmc / esmif
        fs_name = FEATURE_SET_NAME[toolkit]
        n_pad = int(sub["n_padded"].iloc[0])
        for k_label in _ladder_keys(toolkit):
            group = sub[sub["k_label"] == k_label]
            if group.empty:
                continue
            rows = _order_pc(pd.concat([group, base], ignore_index=True))
            n = len(rows)
            x = np.arange(n)
            colors = [PADTYPE_COLOR.get(p, "#7F7F7F") for p in rows["padtype"]]
            specs = (
                (dir_full, "auc", "auc_std",
                 f"{fs_name} @ k={k_label} — pad modes vs handcrafted_sparse",
                 "AUC-ROC — full data"),
                (dir_subset, f"auc_padded_{tk_key}", f"auc_padded_std_{tk_key}",
                 f"{fs_name} @ k={k_label} — pad modes vs handcrafted_sparse "
                 f"(subset rows, n={n_pad})",
                 f"AUC-ROC — subset rows (n={n_pad})"),
            )
            for outdir, val, err, title, ylab in specs:
                vals = rows[val].to_numpy(dtype=float)
                raw_err = rows[err].to_numpy(dtype=float)
                errs = np.where(np.isnan(raw_err), 0.0, raw_err)
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.bar(x, vals, yerr=errs, color=colors, edgecolor="black",
                       linewidth=0.5, width=0.75,
                       error_kw=dict(lw=0.8, capsize=3))
                for xi, v in zip(x, vals):
                    if not np.isnan(v):
                        ax.text(xi, v + errs[int(xi)] + 0.006, f"{v:.3f}",
                                ha="center", fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels([_bar_label(r) for _, r in rows.iterrows()],
                                   rotation=35, ha="right", fontsize=8)
                ax.set_ylabel(ylab, fontsize=9)
                ax.set_title(title, fontsize=10)
                ax.set_ylim(0, float(np.nanmax(vals + errs)) * 1.18)
                ax.axhline(0, color="black", lw=0.5)
                fig.tight_layout()
                out = outdir / f"grid_auc_{toolkit}_{k_label}.png"
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"  saved: {out}")


def plot_saturation(df: pd.DataFrame, plots_root: Path):
    """AUC vs explained-variance ladder — one figure per toolkit, 2 panels.

    Top = full data, bottom = subset rows. X: 4 padmode groups; inside each,
    one bar per PC level at a fixed slot so the same EV% aligns across
    groups (sequential color by PC index). Baseline = one dashed line per
    panel (full AUC / subset AUC under that toolkit's padded mask).
    """
    base = df[df["padtype"] == "-"]
    if base.empty:
        return
    for toolkit in ["ESM-C", "ESM-IF"]:
        sub = df[df["toolkit"] == toolkit]
        if sub.empty:
            continue
        tk_key = toolkit.replace("-", "").lower()
        fs_name = FEATURE_SET_NAME[toolkit]
        ladder = EV_BY_K[toolkit]
        keys = _ladder_keys(toolkit)
        modes = PADMODES_PER_TOOLKIT[toolkit]
        base_full = float(base["auc"].iloc[0])
        base_sub = float(base[f"auc_padded_{tk_key}"].iloc[0])
        n_pad = int(sub["n_padded"].iloc[0])

        n_slots = len(keys)
        slot_w = 0.9 / n_slots
        group_w, group_gap = 1.0, 0.6
        cmap = plt.get_cmap("Blues")
        slot_colors = [cmap(0.35 + 0.55 * i / max(1, n_slots - 1))
                       for i in range(n_slots)]

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        panels = (
            (ax_top, "auc", "auc_std", base_full, "full data"),
            (ax_bot, f"auc_padded_{tk_key}", f"auc_padded_std_{tk_key}",
             base_sub, f"subset rows (n={n_pad})"),
        )
        for ax, val_col, std_col, base_val, ptitle in panels:
            ymax = base_val
            for gi, mode in enumerate(modes):
                g0 = gi * (group_w + group_gap)
                for si, k_label in enumerate(keys):
                    row = sub[(sub["padtype"] == mode)
                              & (sub["k_label"] == k_label)]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    v = r[val_col]
                    if pd.isna(v):
                        continue
                    e = r[std_col]
                    e = 0.0 if pd.isna(e) else float(e)
                    xpos = g0 + si * slot_w + slot_w / 2
                    ax.bar(xpos, v, yerr=e, width=slot_w * 0.85,
                           color=slot_colors[si], edgecolor="black",
                           linewidth=0.4,
                           error_kw=dict(lw=0.7, capsize=2))
                    # EV% inside the bar (white, vertical)
                    ax.text(xpos, 0.02, f"{ladder[k_label]}%",
                            rotation=90, va="bottom", ha="center",
                            color="white", fontsize=7)
                    # AUC above the bar (vertical so neighbours don't collide)
                    ax.text(xpos, v + e + 0.008, f"{v:.3f}",
                            rotation=90, va="bottom", ha="center", fontsize=6.5)
                    ymax = max(ymax, v + e)
            ax.axhline(base_val, color="black", linestyle="--", lw=1.0)
            ax.set_ylim(0, ymax * 1.28)
            ax.set_ylabel("AUC-ROC", fontsize=9)
            ax.set_title(f"{ptitle}   |   dashed: handcrafted_sparse "
                         f"{base_val:.3f}", fontsize=10)
            ax.axhline(0, color="black", lw=0.5)

        group_centers = [gi * (group_w + group_gap) + group_w / 2
                         for gi in range(len(modes))]
        ax_bot.set_xticks(group_centers)
        ax_bot.set_xticklabels(modes, fontsize=10)
        fig.suptitle(f"{fs_name} — AUC vs explained variance", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = plots_root / f"grid_saturation_{toolkit}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved: {out}")


def main():
    if not MODELS_DIR.is_dir():
        raise SystemExit(f"Missing {MODELS_DIR} — file the grid results first")

    df = pd.read_csv(SPLIT_DATA_PATH, usecols=["label", "fold"])
    print(f"  {df_path_name()}: {len(df)} rows, "
          f"pos={(df['label'] == 1).sum()}, neg={(df['label'] == 0).sum()}")

    masks = load_padded_masks()
    for toolkit, m in masks.items():
        if len(m) != len(df):
            raise SystemExit(f"{toolkit} pad_mask rows {len(m)} != df rows {len(df)}")

    runs = discover_runs()
    print(f"  {len(runs)} grid JSONs in {MODELS_DIR}")
    runtimes = parse_runtimes()
    print(f"  {len(runtimes)} runtimes parsed from {LOGS_DIR}/padfull_flat_*.out")

    rows = []
    for r in runs:
        rows.append({**r, **load_run(r, df, masks, runtimes)})
    out = pd.DataFrame(rows).drop(columns=["path"])
    out = out.sort_values(["toolkit", "padtype", "pca"], na_position="last")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_cols = [c for c in out.columns if c.startswith("fold_auc")]
    out.drop(columns=fold_cols).to_csv(OUT_CSV, index=False)
    print(f"  saved: {OUT_CSV}")

    # raw-encoding reference arms (sparse / blosum alone) from padmode_choice
    raw_arms = discover_raw_arms()
    raw_rows = []
    for r in raw_arms:
        raw_rows.append({**r, **load_run(r, df, masks, {})})
    raw_out = pd.DataFrame(raw_rows).drop(columns=["path"], errors="ignore")
    if not raw_out.empty:
        print(f"  {len(raw_out)} raw-encoding arms: "
              f"{', '.join(raw_out['padtype'])}")

    plot_metric(out, "auc", "auc_std",
                "handcrafted_sparse_esmc/esmif window runs — full-data "
                "AUC-ROC (mean ± std, 6 buckets)",
                PLOTS_DIR / "grid_auc_full.png", "AUC-ROC")
    plot_metric(out, "mcc", "mcc_std",
                "handcrafted_sparse_esmc/esmif window runs — full-data "
                "MCC (mean ± std, 6 buckets)",
                PLOTS_DIR / "grid_mcc_full.png", "MCC")

    # the padded overview is superseded by the saturation figures
    stale = PLOTS_DIR / "grid_auc_padded.png"
    if stale.exists():
        stale.unlink()
        print(f"  removed: {stale}")

    plot_per_pc(out, PLOTS_DIR)
    plot_saturation(out, PLOTS_DIR)

    # summary TSV + paired t-tests (subset rows, BH-corrected)
    summary, comps = build_summary_and_tests(out, raw_out, masks)

    # compact stdout table
    show = out[["toolkit", "padtype", "k_label", "auc", "auc_std", "mcc",
                "auc_padded", "runtime_h", "n_features"]]
    print("\n" + show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not comps.empty:
        sig = comps[comps["q_bh"] < 0.05]
        print(f"\n  paired t-tests (subset rows): {len(comps)} total, "
              f"{len(sig)} significant at q<0.05")
        print(comps.nsmallest(10, "p_ttest").to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))


def df_path_name() -> str:
    return SPLIT_DATA_PATH.name


if __name__ == "__main__":
    main()
