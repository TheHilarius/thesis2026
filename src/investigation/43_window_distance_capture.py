#!/usr/bin/env python3
"""
43_window_distance_capture.py
Window-pair Hamming-distance distribution and "capture curve" for the
robust homology split -- the data-driven leg of the window-cutoff choice.

Question answered: at which window-Hamming cutoff c have we essentially
captured all homologous window pairs, so that raising c further prevents
almost no additional leakage while coarsening the folds?

Pool: all unique full_context windows of the THESIS split
(df_all_with_folds_flank5_rmdup_updated_h6_pepcut1.csv), all pairs at
window-Hamming distance d <= 9 (pigeonhole index, provably complete).

Per cutoff c = 1..9:
  pairs_cum(c)      # pairs with d <= c  (raw edge-set-A capture)
  marginal(c)       # pairs with d <= c NOT already grouped by the fixed
                    # rule arms (pep-ham<=1 via edge set B, identical-flank
                    # via edge set C) -- the NEW grouping that raising c adds
  crossfold_old(c)  # pairs with d <= c that straddle folds under the
                    # canonical pep-only rmdup split (the leakage the
                    # robust rule fixes on top of it)

Outputs (run tree runs/r19_rmdup):
    plots/window_distance_capture.png / .csv
Usage (repo root):
    python src/investigation/43_window_distance_capture.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline" / "python"))

RUN_ROOT = PROJECT_ROOT / "runs" / "r19_rmdup"
THESIS_SPLIT = (PROJECT_ROOT / "data" / "processed" /
                "df_all_with_folds_flank5_rmdup_updated_h6_pepcut1.csv")
CANON_SPLIT = (PROJECT_ROOT / "data" / "processed" /
               "df_all_with_folds_flank5_rmdup.csv")
MAX_D = 9

spec = importlib.util.spec_from_file_location(
    "datasplit", PROJECT_ROOT / "src" / "pipeline" / "python" / "01_datasplit.py")
datasplit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datasplit)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--thesis-split", default=str(THESIS_SPLIT))
    ap.add_argument("--canon-split", default=str(CANON_SPLIT))
    args = ap.parse_args()

    print(f"Loading {args.thesis_split} ...")
    df = pd.read_csv(args.thesis_split, usecols=["full_context", "peptide"])
    ctx_pep = df.drop_duplicates("full_context").set_index("full_context")["peptide"]
    contexts = ctx_pep.index.tolist()
    n_ctx = len(contexts)
    ctx_to_peptide = {c: ctx_pep[c] for c in contexts}
    idx_of = {c: i for i, c in enumerate(contexts)}
    print(f"  {n_ctx} unique windows")

    # ── exact-distance pair census (d <= 9, deduped) ──────────────────────
    print(f"Pigeonhole pass at max_dist={MAX_D} over {n_ctx} windows ...")
    t0 = time.time()
    pair_d: dict[tuple[int, int], int] = {}
    for i, j in datasplit.iter_close_pairs(contexts, MAX_D):
        key = (i, j) if i < j else (j, i)
        if key not in pair_d:
            pair_d[key] = hamming(contexts[i], contexts[j])
    print(f"  {len(pair_d)} close pairs in {time.time() - t0:.1f}s")
    hist = defaultdict(int)
    for d in pair_d.values():
        hist[d] += 1

    # ── fixed-rule union-find: pep-ham<=1 (B) + identical flank (C) ───────
    print("Building fixed-rule grouping (B: pep-ham<=1, C: identical flank) ...")
    unique_peptides = sorted(set(ctx_to_peptide.values()))
    pep_clusters = datasplit.cluster_by_hamming(unique_peptides, max_dist=1)
    pep_root = {}
    for root, peps in pep_clusters.items():
        for p in peps:
            pep_root[p] = root
    uf = datasplit.UnionFind()
    for c in contexts:
        uf.find(c)
    by_pep = defaultdict(list)
    for c in contexts:
        by_pep[pep_root[ctx_to_peptide[c]]].append(c)
    L = (len(contexts[0]) - 9) // 2
    by_nflank, by_cflank = defaultdict(list), defaultdict(list)
    for c in contexts:
        by_nflank[c[:L]].append(c)
        by_cflank[c[len(c) - L:]].append(c)
    for groups in (by_pep, by_nflank, by_cflank):
        for members in groups.values():
            for m in members[1:]:
                uf.union(members[0], m)
    root_of = {c: uf.find(c) for c in contexts}

    # ── canonical (pep-only) split folds per window ───────────────────────
    print(f"Loading canonical split folds from {args.canon_split} ...")
    canon = pd.read_csv(args.canon_split, usecols=["full_context", "fold"])
    canon_fold = canon.drop_duplicates("full_context").set_index("full_context")["fold"]
    fold_of = [int(canon_fold[c]) for c in contexts]

    # ── per-cutoff metrics ────────────────────────────────────────────────
    rows = []
    cum_pairs = cum_marg = cum_cross = 0
    for c in range(1, MAX_D + 1):
        d_pairs = [(k, d) for k, d in pair_d.items() if d == c]
        cum_pairs += len(d_pairs)
        marg = sum(1 for (i, j), _ in d_pairs
                   if root_of[contexts[i]] != root_of[contexts[j]])
        cross = sum(1 for (i, j), _ in d_pairs if fold_of[i] != fold_of[j])
        cum_marg += marg
        cum_cross += cross
        rows.append(dict(cutoff=c, pairs_at_d=len(d_pairs),
                         pairs_cum=cum_pairs, marginal_new=cum_marg,
                         crossfold_old_cum=cum_cross))
        print(f"  c={c}: +{len(d_pairs):>6} pairs (cum {cum_pairs:>6})  "
              f"marginal {marg:>5} (cum {cum_marg:>6})  "
              f"crossfold+{cross:>5} (cum {cum_cross:>5})")
    tab = pd.DataFrame(rows)
    total_pool = cum_pairs
    tab["capture_pct"] = 100.0 * tab["pairs_cum"] / total_pool

    # ── figure ────────────────────────────────────────────────────────────
    outdir = RUN_ROOT / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ds = tab["cutoff"]

    ax1.bar(ds - 0.18, tab["pairs_at_d"], width=0.36, color="#4477AA",
            label="pairs at exact distance d")
    marg_d = tab["marginal_new"] - tab["marginal_new"].shift(1, fill_value=0)
    ax1.bar(ds + 0.18, marg_d, width=0.36, color="#EE6677",
            label="NEW grouping at d (not already pep-ham<=1 / identical-flank)")
    ax1b = ax1.twinx()
    ax1b.plot(ds, tab["capture_pct"], "o-", color="#EE6677",
              label="cumulative capture %")
    ax1b.set_ylabel("cumulative capture (% of pool)", color="#EE6677")
    ax1b.set_ylim(0, 105)
    ax1b.tick_params(axis="y", labelcolor="#EE6677")
    ax1.set_yscale("log")
    ax1.set_xlabel("window Hamming distance d")
    ax1.set_ylabel("window pairs (log)")
    ax1.set_title(f"Pair-distance distribution (pool = {total_pool:,} pairs, d<=9)")
    ax1.set_xticks(range(1, MAX_D + 1))

    ax2.bar(ds - 0.18, tab["crossfold_old_cum"] - tab["crossfold_old_cum"].shift(1, fill_value=0),
            width=0.36, color="#228833", label="straddling pairs at d (canonical split)")
    ax2b = ax2.twinx()
    ax2b.plot(ds, tab["crossfold_old_cum"], "o-", color="#CCBB44",
              label="cumulative cross-fold pairs fixed by c")
    ax2b.set_ylabel("cumulative cross-fold pairs", color="#CCBB44")
    ax2b.tick_params(axis="y", labelcolor="#CCBB44")
    ax2.set_yscale("log")
    ax2.set_xlabel("window Hamming distance d")
    ax2.set_ylabel("window pairs (log)")
    ax2.set_title("Leakage the robust rule fixes vs the canonical split")
    ax2.set_xticks(range(1, MAX_D + 1))

    for ax in (ax1, ax2):
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Window-cutoff capture curve (pepcut1 fixed, flank5_rmdup)", y=1.00)
    fig.tight_layout()
    out_png = outdir / "window_distance_capture.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    tab.to_csv(outdir / "window_distance_capture.csv", index=False)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {outdir / 'window_distance_capture.csv'}")
    print(f"\nPool total: {total_pool} pairs at d<=9")
    c_chosen = 6
    print("Chosen c={}: capture {:.1f}% of pool | marginal cum {} pairs | "
          "crossfold fixed cum {}".format(
              c_chosen,
              100 * tab.loc[tab.cutoff == c_chosen, "pairs_cum"].iloc[0] / total_pool,
              int(tab.loc[tab.cutoff == c_chosen, "marginal_new"].iloc[0]),
              int(tab.loc[tab.cutoff == c_chosen, "crossfold_old_cum"].iloc[0])))


if __name__ == "__main__":
    main()
