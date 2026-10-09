#!/usr/bin/env python3
"""
plot_padmode_data_compare.py — held-out AUC-ROC barplot for the
data-comparison grid.

Bars (fixed order):
  handcrafted | sparse (one-hot AA) | BLOSUM50 | BLOSUM50 + engineered
  then 4 ESM-C (600M) pad modes and 4 ESM-IF1 pad modes, embeddings only.

Inputs (one of):
  --tsv          aggregate TSV from extract_model_metrics.py (full-data
                 held-out AUC) → padmode_data_compare_auc_roc.png
  --subset-csv   padmode_subset_auc.csv from 09_padmode_subset_auc.py
                 (AUC on the informative padded subset) →
                 padmode_data_compare_auc_roc_subset.png. Baselines are
                 plotted under the ESM-C padded mask; both masks' values
                 stay in the source CSV. handcrafted-raw is skipped to
                 keep the 11-bar comparison scope.
  --runtime-tsv  runtime.tsv → padmode_data_compare_runtime.png (log-y)

With --stats-csv (15_'s paired_tests.csv), AUC figures draw significance
brackets: raw DeLong p + Holm p_adj, stars from p_adj. Bracket sets per
--layout: padmode = sparse-vs-BLOSUM + chosen padmode vs the other 3 per
toolkit; baselines = 5 "is-engineered-helping" pairs. Full-data figure
reads *_full test groups; subset figure reads padded-subset groups.

Usage:
    python src/investigation/plot_padmode_data_compare.py \
        --tsv runs/padmode_choice/results/model_matrix_padmode_11.tsv \
        --stats-csv runs/padmode_choice/results/paired_tests.csv \
        --out runs/padmode_choice/plots/
    python src/investigation/plot_padmode_data_compare.py \
        --subset-csv runs/padmode_choice/results/padmode_subset_auc.csv \
        --stats-csv runs/padmode_choice/results/paired_tests.csv \
        --out runs/padmode_choice/plots/
    python src/investigation/plot_padmode_data_compare.py \
        --runtime-tsv runs/padmode_choice/results/runtime.tsv \
        --out runs/padmode_choice/plots/

Outputs:
    padmode_data_compare_auc_roc[_subset].png/.csv — barplot + values
    padmode_data_compare_runtime.png/.csv          — log-y wall-time plot
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR_ORDER = [
    ("handcrafted", "Handcrafted (eng.)", "csv"),
    ("sparse", "Sparse", "csv"),
    ("blosum", "BLOSUM50", "csv"),
    ("handcrafted_sparse", "Sparse + engineered", "csv"),
    ("handcrafted_blosum", "BLOSUM50 +\nengineered", "csv"),
    ("emb_esmc_win_zeropad", "zero-pad", "esmc"),
    ("emb_esmc_win_padtoken", "pad-token", "esmc"),
    ("emb_esmc_win_impute_bos_eos", "EOS/BOS-repeat", "esmc"),
    ("emb_esmc_win_impute_boundary", "boundary", "esmc"),
    ("emb_esmif_win", "zero-pad", "esmif"),
    ("emb_esmif_pad", "pad-token", "esmif"),
    ("emb_esmif_eos_bos_repeat", "EOS/BOS-repeat", "esmif"),
    ("emb_esmif_boundary", "boundary", "esmif"),
]

# Paul Tol bright qualitative scheme (SRON/EPS/TN/09-002, colour-blind safe)
GROUP_COLORS = {
    "csv":   "#BBBBBB",
    "esmc":  "#EE6677",
    "esmif": "#4477AA",
}
GROUP_LABELS = {
    "csv": "AA encoding",
    "esmc": "ESM-C",
    "esmif": "ESM-IF",
}

BAR_BY_KEY = {k: (k, l, g) for k, l, g in BAR_ORDER}

# Which bars each figure shows (order preserved from BAR_ORDER)
LAYOUTS = {
    # padmode decision figure: AA encodings + 4+4 fill modes
    "padmode": ["sparse", "blosum",
                "emb_esmc_win_zeropad", "emb_esmc_win_padtoken",
                "emb_esmc_win_impute_bos_eos", "emb_esmc_win_impute_boundary",
                "emb_esmif_win", "emb_esmif_pad",
                "emb_esmif_eos_bos_repeat", "emb_esmif_boundary"],
    # is-engineered-helping figure: CSV feature sets only
    "baselines": ["handcrafted", "sparse", "blosum",
                  "handcrafted_sparse", "handcrafted_blosum"],
}

# (toolkit, padtype) in padmode_subset_auc.csv -> BAR_ORDER feature key
SUBSET_KEY_MAP = {
    ("ESM-C", "zero"): "emb_esmc_win_zeropad",
    ("ESM-C", "pad_token"): "emb_esmc_win_padtoken",
    ("ESM-C", "eos_bos_repeat"): "emb_esmc_win_impute_bos_eos",
    ("ESM-C", "boundary"): "emb_esmc_win_impute_boundary",
    ("ESM-IF", "zero"): "emb_esmif_win",
    ("ESM-IF", "pad"): "emb_esmif_pad",
    ("ESM-IF", "boundary"): "emb_esmif_boundary",
    ("ESM-IF", "eos_bos_repeat"): "emb_esmif_eos_bos_repeat",
}


def load_tsv(path: Path):
    """Return list of dict rows from the extract_model_metrics TSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pick_rows(rows, model, metric):
    """Index rows by feature_set for the requested model."""
    by_feat = {}
    for r in rows:
        if r.get("model_key", "").strip().lower() != model.lower():
            continue
        val = r.get(metric, "").strip()
        if val == "":
            continue
        by_feat[r["feature_set"].strip()] = float(val)
    return by_feat


def load_subset(path: Path, model: str):
    """Index auc_padded by BAR_ORDER feature key from 09_'s CSV.

    Baselines (pca==0): ESM-C padded mask only (1443 vs 1442 rows —
    masks agree to within the notfound rows; both stay in source CSV).
    """
    by_feat = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["learner"].strip().lower() != model.lower():
                continue
            val = r["auc_padded"].strip()
            if val == "":
                continue
            if int(float(r["pca"])) == 0:
                key = r["padtype"].strip()
                if key not in BAR_BY_KEY:
                    continue
                if r["toolkit"].strip() != "ESM-C":
                    continue  # one bar per baseline: ESM-C mask
                by_feat[key] = float(val)
            else:
                key = SUBSET_KEY_MAP.get(
                    (r["toolkit"].strip(), r["padtype"].strip()))
                if key:
                    by_feat[key] = float(val)
    return by_feat


def load_runtime(path: Path):
    """Index runtime_hours by features_key from runtime.tsv."""
    by_feat = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by_feat[r["features_key"].strip()] = float(r["runtime_hours"])
    return by_feat


# padmode-name -> BAR_ORDER feature key, per plot group
MODE_KEYS = {
    "esmc": {
        "zero": "emb_esmc_win_zeropad",
        "pad_token": "emb_esmc_win_padtoken",
        "boundary": "emb_esmc_win_impute_boundary",
        "eos_bos_repeat": "emb_esmc_win_impute_bos_eos",
    },
    "esmif": {
        "zero": "emb_esmif_win",
        "pad": "emb_esmif_pad",
        "boundary": "emb_esmif_boundary",
        "eos_bos_repeat": "emb_esmif_eos_bos_repeat",
    },
    "csv": {
        "sparse": "sparse",
        "blosum": "blosum",
        "handcrafted_blosum": "handcrafted_blosum",
        "handcrafted_sparse": "handcrafted_sparse",
        "handcrafted": "handcrafted",
    },
}

# Hardcoded significance brackets per layout: chosen padmode vs the other
# three; baselines-layout pairs answer "is engineered helping?".
HIGHLIGHT_PAIRS = {
    "padmode": {
        "csv": [("sparse", "blosum")],
        "esmc": [("pad_token", "zero"),
                 ("pad_token", "boundary"),
                 ("pad_token", "eos_bos_repeat")],
        "esmif": [("eos_bos_repeat", "zero"),
                  ("eos_bos_repeat", "pad"),
                  ("eos_bos_repeat", "boundary")],
    },
    "baselines": {
        "csv": [("handcrafted", "sparse"),
                ("handcrafted", "blosum"),
                ("handcrafted", "handcrafted_sparse"),
                ("handcrafted", "handcrafted_blosum"),
                ("sparse", "blosum"),
                ("sparse", "handcrafted_sparse"),
                ("sparse", "handcrafted_blosum"),
                ("blosum", "handcrafted_sparse"),
                ("blosum", "handcrafted_blosum"),
                ("handcrafted_sparse", "handcrafted_blosum")],
    },
}

# plot mode -> {bar group: 15_ test_group}
STATS_GROUPS = {
    "subset": {"csv": "baseline_subset_esmc",
               "esmc": "padmode_esmc", "esmif": "padmode_esmif"},
    "full": {"csv": "baseline_full",
             "esmc": "padmode_esmc_full", "esmif": "padmode_esmif_full"},
}


def load_stats(path: Path, p_col: str, mode: str, layout: str):
    """Resolve HIGHLIGHT_PAIRS[layout] against 15_'s CSV.

    Returns list of (group, key_a, key_b, delta, p, p_adj); p_adj is the
    Holm column (p_col + "_holm") when present, else None.
    """
    holm_col = f"{p_col}_holm"
    want = {}  # (test_group, frozenset pair) -> (group, ma, mb)
    for grp, pairs in HIGHLIGHT_PAIRS[layout].items():
        tg = STATS_GROUPS[mode][grp]
        for ma, mb in pairs:
            want[(tg, frozenset((ma, mb)))] = (grp, ma, mb)

    found = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tg = r["test_group"].strip()
            key = (tg, frozenset((r["model_a"].strip(),
                                  r["model_b"].strip())))
            if key in want:
                raw = (r.get(p_col) or "").strip()
                if raw == "":
                    continue  # NaN cell (e.g. bootstrap cols on *_full rows)
                raw_adj = (r.get(holm_col) or "").strip()
                p_adj = float(raw_adj) if raw_adj else None
                grp, ma, mb = want[key]
                found[key] = (grp, MODE_KEYS[grp][ma], MODE_KEYS[grp][mb],
                              float(r["delta_auc"]), float(raw), p_adj)
    out = list(found.values())
    missing = set(want) - set(found)
    if missing:
        print(f"WARNING: stats pairs not found in CSV for "
              f"mode={mode} layout={layout}: {sorted(missing)}")
    return out


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_p(p):
    """Bracket p-text; guards underflow/zero from norm.sf at large |z|."""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.2g}"


def fmt_p_pair(p, p_adj):
    """'p=0.51, p_adj=1.0' bracket text; stars come from p_adj."""
    if p_adj is None:
        return fmt_p(p)
    a = "p_adj<0.001" if p_adj < 0.001 else f"p_adj={p_adj:.2g}"
    return f"{fmt_p(p)}, {a}"


def main():
    ap = argparse.ArgumentParser(
        description="Barplot of held-out AUC-ROC for the pad-mode/data grid."
    )
    ap.add_argument("--tsv", help="Aggregate TSV (extract_model_metrics.py output)")
    ap.add_argument("--subset-csv",
                    help="09_padmode_subset_auc.py CSV → *_subset.png "
                         "(padded-subset AUC)")
    ap.add_argument("--runtime-tsv",
                    help="runtime.tsv → *_runtime.png (log-y wall time)")
    ap.add_argument("--stats-csv",
                    help="15_padmode_significance.py CSV; with --tsv or "
                         "--subset-csv adds significance brackets "
                         "(chosen padmodes vs others + CSV baselines)")
    ap.add_argument("--p-col", default="p_delong",
                    help="p-value column in --stats-csv (default: p_delong, "
                         "raw DeLong; use p_delong_holm for multiplicity-"
                         "adjusted annotations)")
    ap.add_argument("--layout", default="padmode",
                    choices=sorted(LAYOUTS),
                    help="bar set: padmode (sparse+BLOSUM+8 fill modes) or "
                         "baselines (5 CSV feature sets)")
    ap.add_argument("--out", default="results/figures/investigation",
                    help="Output directory for PNG + CSV")
    ap.add_argument("--model", default="xgb", help="model_key to plot (default: xgb)")
    ap.add_argument("--metric", default="heldout_auc_roc",
                    help="Metric column for --tsv mode (default: heldout_auc_roc)")
    args = ap.parse_args()

    n_modes = sum(bool(x) for x in (args.tsv, args.subset_csv, args.runtime_tsv))
    if n_modes != 1:
        raise SystemExit(
            "Provide exactly one of --tsv, --subset-csv, --runtime-tsv")
    if args.stats_csv and args.runtime_tsv:
        raise SystemExit("--stats-csv does not apply to --runtime-tsv")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = None

    def maybe_stats(mode):
        if not args.stats_csv:
            return None
        stats_path = Path(args.stats_csv)
        if not stats_path.exists():
            raise SystemExit(f"Stats CSV not found: {stats_path}")
        return load_stats(stats_path, args.p_col, mode, args.layout)

    if args.subset_csv:
        subset_path = Path(args.subset_csv)
        if not subset_path.exists():
            raise SystemExit(f"Subset CSV not found: {subset_path}")
        by_feat = load_subset(subset_path, args.model)
        if not by_feat:
            raise SystemExit(f"No rows for model='{args.model}' in {subset_path}")
        metric_name = "auc_padded"
        suffix = "_subset"
        ylabel = "AUC-ROC"
        title = "Held-out AUC, padded peptides only"
        stats = maybe_stats("subset")
    elif args.runtime_tsv:
        rt_path = Path(args.runtime_tsv)
        if not rt_path.exists():
            raise SystemExit(f"Runtime TSV not found: {rt_path}")
        by_feat = load_runtime(rt_path)
        if not by_feat:
            raise SystemExit(f"No rows in {rt_path}")
        metric_name = "runtime_hours"
        suffix = "_runtime"
        ylabel = "Runtime (hours, log scale)"
        title = (f"Data comparison grid — {args.model.upper()} "
                 f"(nested-CV wall time)")
    else:
        tsv_path = Path(args.tsv)
        if not tsv_path.exists():
            raise SystemExit(f"TSV not found: {tsv_path}")
        rows = load_tsv(tsv_path)
        by_feat = pick_rows(rows, args.model, args.metric)
        if not by_feat:
            raise SystemExit(
                f"No rows for model='{args.model}' / metric='{args.metric}' in {tsv_path}"
            )
        metric_name = args.metric
        suffix = ""
        ylabel = "AUC-ROC"
        title = "Held-out AUC, all peptides"
        stats = maybe_stats("full")

    layout_keys = LAYOUTS[args.layout]
    labels, groups, values = [], [], []
    missing = []
    for key in layout_keys:
        if key in by_feat:
            _, label, group = BAR_BY_KEY[key]
            labels.append(label)
            groups.append(group)
            values.append(by_feat[key])
        else:
            missing.append(key)

    if missing:
        print(f"WARNING: missing feature_set(s): {missing}")
    if not labels:
        raise SystemExit("No matching bars")

    n = len(labels)
    x = np.arange(n)
    colors = [GROUP_COLORS[g] for g in groups]
    is_runtime = args.runtime_tsv is not None

    # resolve bracket geometry before drawing so brackets fit under y=1
    # (fixed axis) — no labels spilling over the title or the axes box
    key_to_x, by_group, y0, step = {}, {}, 0.0, 0.0
    br_fontsize = 7.5
    if stats:
        xi = 0
        for key in layout_keys:
            if key in by_feat:
                key_to_x[key] = xi
                xi += 1
        for grp, ka, kb, delta, p, p_adj in stats:
            if ka in key_to_x and kb in key_to_x:
                by_group.setdefault(grp, []).append(
                    (ka, kb, delta, p, p_adj))
        if args.layout == "baselines":
            # significance-only brackets: drop ns pairs (p_adj >= 0.05)
            for grp, items in by_group.items():
                by_group[grp] = [it for it in items
                                 if (it[4] if it[4] is not None else it[3])
                                 < 0.05]
            # hub brackets: ONE bracket per left bar -> its rightmost
            # significant partner; stars from the min p_adj over the
            # covered pairs (exact per-pair p's stay in the CSV)
            keys_placed = [k for k in layout_keys if k in key_to_x]
            hubs = []
            for i, ka in enumerate(keys_placed):
                rightmost, best, partners = None, None, []
                for kb in keys_placed[i + 1:]:
                    for it in by_group.get("csv", []):
                        if {it[0], it[1]} != {ka, kb}:
                            continue
                        rightmost = kb  # ascending scan: last sig = rightmost
                        partners.append(kb)
                        src = it[4] if it[4] is not None else it[3]
                        best_src = (best[4] if best and best[4] is not None
                                    else (best[3] if best else np.inf))
                        if best is None or src < best_src:
                            best = it
                if rightmost is not None and best is not None:
                    # tick at the hub bar + every significant partner so
                    # "differs from all covered" is visually explicit
                    hubs.append((ka, rightmost, best[2], best[3], best[4],
                                 [ka] + partners))
            by_group = {"csv": hubs} if hubs else {}
        for items in by_group.values():
            items.sort(key=lambda it: -abs(key_to_x[it[0]] - key_to_x[it[1]]))
        y0 = max(values) * 1.08
        max_lvl = max((len(items) for items in by_group.values()),
                      default=1)
        # fit the highest bracket text under y=0.97 on the fixed 0-1 axis
        headroom = 0.97 - y0 - max(values) * 0.03
        step = min(max(values) * 0.085, headroom / (max_lvl - 1 + 0.35))
        if step < max(values) * 0.05:
            br_fontsize = 6.5

    fig_h = 5.5
    if stats:
        fig_h = 8.5 if (args.layout == "baselines" and max_lvl > 4) else 6.8
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.bar(x, values, color=colors, width=0.62, edgecolor="black", linewidth=0.5)

    if is_runtime:
        for xi, v in zip(x, values):
            ax.text(xi, v * 1.15, f"{v:.2f}h", ha="center", va="bottom",
                    fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(min(values) * 0.4, max(values) * 4)
    else:
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_ylim(0, 1.0)

    # significance brackets (15_ stats); staggered per group — widest
    # span on top. Text shows raw p + Holm p_adj; stars from p_adj.
    # Baselines hubs carry a 6th field: keys of every significant partner
    # inside the span — small down-ticks there make "differs from each"
    # explicit (ns bars inside a span get NO tick).
    if stats:
        for grp, items in by_group.items():
            for lvl, item in enumerate(items):
                ka, kb, delta, p, p_adj = item[:5]
                tick_keys = item[5] if len(item) > 5 else []
                xa, xb = key_to_x[ka], key_to_x[kb]
                y_br = y0 + lvl * step
                ax.plot([xa, xa, xb, xb],
                        [y_br * 0.985, y_br, y_br, y_br * 0.985],
                        color="black", lw=0.9)
                for tk_ in tick_keys:
                    xt = key_to_x[tk_]
                    if xt in (xa, xb):
                        continue  # endpoints already ticked
                    ax.plot([xt, xt], [y_br * 0.985, y_br],
                            color="black", lw=0.9)
                star_src = p_adj if p_adj is not None else p
                # dense all-sig strips: stars only (p/p_adj live in CSV)
                txt = (stars(star_src) if args.layout == "baselines"
                       else f"{fmt_p_pair(p, p_adj)} ({stars(star_src)})")
                ax.text((xa + xb) / 2, y_br + step * 0.12, txt,
                        ha="center", va="bottom", fontsize=br_fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(1, n):  # dividers between group transitions
        if groups[i] != groups[i - 1]:
            ax.axvline(i - 0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    present_groups = [g for g in GROUP_COLORS if g in groups]
    if len(present_groups) > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[g],
                                 edgecolor="black")
                   for g in present_groups]
        ax.legend(handles, [GROUP_LABELS[g] for g in present_groups],
                  fontsize=11, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False)

    fig.tight_layout()
    if is_runtime:
        stem = "padmode_data_compare_runtime"
    elif args.layout == "baselines":
        stem = f"baseline_compare_auc_roc{suffix}"
    else:
        stem = f"padmode_data_compare_auc_roc{suffix}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "feature_set", "label", "metric", "value"])
        for key in layout_keys:
            if key not in by_feat:
                continue
            _, label, group = BAR_BY_KEY[key]
            w.writerow([group, key, label, metric_name, by_feat[key]])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()