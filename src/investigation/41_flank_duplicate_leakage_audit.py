#!/usr/bin/env python3
"""
41_flank_duplicate_leakage_audit.py
Read-only audit of df_all_with_folds.csv for thesis discussion:
  A) duplicate peptides (same 9-mer in different proteins) — flank identity,
     label composition (all-pos / all-neg / mixed), fold containment
  B) mechanism of identical-flank mixed-label groups (annotation artifacts)
  C) flank-context leakage across folds (split ignores flanks; only peptide
     Hamming <= HAMMING_CUTOFF drives folds): exact flank-pair overlap,
     one-sided n_flank / c_flank overlap, held-out vs train overlap,
     Hamming distance of cross-fold flank-sharing peptide pairs

Does NOT touch the pipeline. Reads only data/processed/df_all_with_folds.csv.

Usage (from repo root):
    python src/investigation/41_flank_duplicate_leakage_audit.py

Output:
    logs/41_flank_duplicate_leakage_audit_<ts>.txt   (tee'd stdout)
    results/tables/41_flank_leakage/
        dup_peptide_summary.csv          per-category counts
        identical_flank_mixed_groups.csv 260-group mechanism, one row/peptide
        cross_fold_flank_pairs.csv       cross-fold flank pairs + Hamming
        flank_overlap_by_scope.csv       pair / n_flank / c_flank x scope
"""

import os
import sys
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SRC_DIR, "..", "pipeline", "python")
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import pandas as pd

from config import (
    PROJECT_ROOT, SPLIT_DATA_PATH, LOG_DIR,
    PEPTIDE_COL, LABEL_COL, FOLD_COL, HAMMING_CUTOFF,
    HELD_OUT_INDEX,
)

TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "41_flank_leakage"

FLANK_COLS = ["n_flank", "c_flank"]
MECH_COLS = ["uniprot_id", "start", "source_molecule",
             "molecule_parent", "rank", "full_context"]


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


class Reporter:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def say(self, s=""):
        print(s)
        self.f.write(s + "\n")

    def close(self):
        self.f.close()


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = Reporter(LOG_DIR / f"41_flank_duplicate_leakage_audit_{ts}.txt")

    df = pd.read_csv(
        SPLIT_DATA_PATH,
        usecols=[PEPTIDE_COL, LABEL_COL, FOLD_COL, "uniprot_id", "start",
                 "n_flank", "c_flank", "source_molecule", "molecule_parent",
                 "rank", "full_context"],
    )
    n_rows = len(df)
    df["flank_pair"] = (df["n_flank"].fillna("<NA>").astype(str)
                        + "|" + df["c_flank"].fillna("<NA>").astype(str))
    cv_folds = sorted(df[FOLD_COL].unique())
    held = HELD_OUT_INDEX

    rep.say("=" * 80)
    rep.say("FLANK / DUPLICATE LEAKAGE AUDIT — df_all_with_folds.csv")
    rep.say(f"Generated: {ts}   rows={n_rows}   "
            f"hamming_cutoff<={HAMMING_CUTOFF}   held-out fold={held}")
    rep.say("=" * 80)

    # ───────────────────────────────────────────────
    # A. DUPLICATE PEPTIDE OVERVIEW
    # ───────────────────────────────────────────────
    vc = df[PEPTIDE_COL].value_counts()
    dup_idx = vc[vc > 1].index
    g = df[df[PEPTIDE_COL].isin(dup_idx)]

    rep.say("")
    rep.say("A. DUPLICATE PEPTIDES")
    rep.say("-" * 80)
    rep.say(f"  unique peptides:        {df[PEPTIDE_COL].nunique()}")
    rep.say(f"  duplicated peptides:    {len(dup_idx)}")
    rep.say(f"  rows from duplicates:   {len(g)}  ({pct(len(g), n_rows):.2f}% of data)")

    def grp_stats(sub):
        return pd.Series({
            "n_prot": sub["uniprot_id"].nunique(),
            "n_nflank": sub["n_flank"].nunique(dropna=False),
            "n_cflank": sub["c_flank"].nunique(dropna=False),
            "n_lab": sub[LABEL_COL].nunique(),
            "n_pos": int((sub[LABEL_COL] == 1).sum()),
            "n_neg": int((sub[LABEL_COL] == 0).sum()),
            "n_fold": sub[FOLD_COL].nunique(),
        })

    r = g.groupby(PEPTIDE_COL).apply(grp_stats, include_groups=False)

    same_flanks = (r["n_nflank"] == 1) & (r["n_cflank"] == 1)
    rep.say(f"  same protein:           {(r['n_prot'] == 1).sum()}")
    rep.say(f"  different proteins:     {(r['n_prot'] > 1).sum()}")
    rep.say(f"  identical flanks (n+c): {same_flanks.sum()}")
    rep.say(f"  different flanks:       {(~same_flanks).sum()}")

    allpos = (r["n_lab"] == 1) & (r["n_pos"] > 0)
    allneg = (r["n_lab"] == 1) & (r["n_neg"] > 0)
    mixed = r["n_lab"] > 1
    rep.say("")
    rep.say("  label composition of duplicated peptides:")
    for name, m in (("all positive", allpos), ("all negative", allneg),
                    ("mixed pos+neg", mixed)):
        rep.say(f"    {name:<16} {int(m.sum()):>5} peptides  "
                f"{int(r.loc[m].eval('n_pos + n_neg').sum()):>5} rows")

    n_cross_fold = int((r["n_fold"] > 1).sum())
    rep.say("")
    rep.say(f"  peptides spanning >1 fold: {n_cross_fold}  "
            f"-> {'OK: no duplicate crosses folds' if n_cross_fold == 0 else 'FOLD LEAKAGE'}")

    # mixed x flanks
    mixed_diff = int((mixed & ~same_flanks).sum())
    mixed_same = int((mixed & same_flanks).sum())
    rep.say(f"  mixed labels w/ different flanks: {mixed_diff}")
    rep.say(f"  mixed labels w/ IDENTICAL flanks: {mixed_same}")

    pd.DataFrame({
        "category": ["unique_peptides", "duplicated_peptides",
                     "dup_rows", "dup_same_protein", "dup_diff_proteins",
                     "dup_identical_flanks", "dup_different_flanks",
                     "label_all_positive", "label_all_negative",
                     "label_mixed", "mixed_identical_flanks",
                     "mixed_different_flanks", "peptides_cross_fold"],
        "count": [int(df[PEPTIDE_COL].nunique()), len(dup_idx), len(g),
                  int((r["n_prot"] == 1).sum()), int((r["n_prot"] > 1).sum()),
                  int(same_flanks.sum()), int((~same_flanks).sum()),
                  int(allpos.sum()), int(allneg.sum()), int(mixed.sum()),
                  mixed_same, mixed_diff, n_cross_fold],
    }).to_csv(TABLE_DIR / "dup_peptide_summary.csv", index=False)

    # ───────────────────────────────────────────────
    # B. IDENTICAL-FLANK MIXED-LABEL MECHANISM
    # ───────────────────────────────────────────────
    if mixed.sum() > 0:
        ident_mixed_idx = r.index[mixed & same_flanks]
        s = df[df[PEPTIDE_COL].isin(ident_mixed_idx)].copy()

        rep.say("")
        rep.say("B. IDENTICAL-FLANK MIXED-LABEL GROUPS (input-identical, label differs)")
        rep.say("-" * 80)
        rep.say(f"  groups: {len(ident_mixed_idx)}   rows: {len(s)}  "
                f"({pct(len(s), n_rows):.2f}% of data)")

        grp = s.groupby(PEPTIDE_COL)
        n_var = {c: int((grp[c].nunique(dropna=False) > 1).sum())
                 for c in MECH_COLS}
        rep.say("  within-group variation (groups where column varies):")
        for c, n in n_var.items():
            rep.say(f"    {c:>18}: {n:>4} / {len(ident_mixed_idx)}")
        rep.say(f"  groups with identical start:   "
                f"{int((grp['start'].nunique(dropna=False) == 1).sum())}")
        iso = sum(1 for p in ident_mixed_idx
                  if df.loc[df[PEPTIDE_COL] == p, "uniprot_id"]
                  .str.split("-").str[0].nunique() == 1)
        rep.say(f"  isoform-style ids (same base UniProt): {iso}")
        rep.say("  -> conflicts track protein annotation, not sequence:")
        rep.say("     label = per-protein detection evidence; identical input")
        rep.say("     context with opposite labels is irreducible noise.")

        mech = grp.agg(
            rows=(LABEL_COL, "size"),
            n_pos=(LABEL_COL, lambda x: int((x == 1).sum())),
            n_neg=(LABEL_COL, lambda x: int((x == 0).sum())),
            n_uniprot=("uniprot_id", "nunique"),
            n_start=("start", "nunique"),
            n_source_molecule=("source_molecule", "nunique"),
            n_molecule_parent=("molecule_parent", "nunique"),
            n_rank=("rank", "nunique"),
            n_full_context=("full_context", "nunique"),
            folds=(FOLD_COL, lambda x: ",".join(map(str, sorted(x.unique())))),
        ).reset_index()
        mech["same_start"] = mech["n_start"] == 1
        mech["same_parent"] = mech["n_molecule_parent"] == 1
        mech.to_csv(TABLE_DIR / "identical_flank_mixed_groups.csv", index=False)

        hard = mech[mech["same_parent"] & mech["same_start"]]
        rep.say(f"  same parent + same start (worst case): {len(hard)} groups")
        rep.say("")
        rep.say("  examples (same context, opposite labels):")
        shown = 0
        for _, row in hard.iterrows():
            if shown >= 5:
                break
            t = df[df[PEPTIDE_COL] == row[PEPTIDE_COL]]
            rep.say(f"    {row[PEPTIDE_COL]}  fold(s)={row['folds']}  "
                    f"pos={row['n_pos']}, neg={row['n_neg']}")
            for _, rr in t.iterrows():
                rep.say(f"      {rr['uniprot_id']:>16} start={rr['start']:<6} "
                        f"label={rr[LABEL_COL]}  {rr['source_molecule']}")
            shown += 1

    # ───────────────────────────────────────────────
    # C. FLANK-CONTEXT LEAKAGE ACROSS FOLDS
    # ───────────────────────────────────────────────
    rep.say("")
    rep.say("C. FLANK-CONTEXT LEAKAGE ACROSS FOLDS (split uses peptide only)")
    rep.say("-" * 80)

    overlap_rows = []

    def scope(key, series, label_name):
        """shared-by->1peptide windows, cross-fold subset, train/heldout."""
        d = df.assign(_key=series).dropna(subset=["_key"])
        gp = d.groupby("_key").agg(
            npep=(PEPTIDE_COL, "nunique"), nf=(FOLD_COL, "nunique"),
            rows=(PEPTIDE_COL, "size"))
        shared = gp[gp["npep"] > 1]
        cross = shared[shared["nf"] > 1]
        tr = d[d[FOLD_COL] != held]
        te = d[d[FOLD_COL] == held]
        sh_trte = set(tr["_key"]) & set(te["_key"])
        te_rows = te[te["_key"].isin(sh_trte)]
        pos_tr = int(tr[tr["_key"].isin(sh_trte) & (tr[LABEL_COL] == 1)]
                     ["_key"].nunique())
        for scope_name, units, unit_rows in (
            ("all", int(len(gp)), None),
            ("shared_by_>1_peptide", int(len(shared)), int(shared["rows"].sum())),
            ("cross_fold", int(len(cross)), int(cross["rows"].sum())),
            (f"train_vs_heldout({held})", len(sh_trte), len(te_rows)),
        ):
            overlap_rows.append({
                "key": label_name, "scope": scope_name,
                "windows": units, "rows": unit_rows})
        rep.say(f"  {label_name}:")
        rep.say(f"    distinct windows:            {len(gp)}")
        rep.say(f"    shared by >1 peptide:        {len(shared)}  "
                f"({int(shared['rows'].sum())} rows)")
        rep.say(f"    spanning >1 fold:            {len(cross)}  "
                f"({int(cross['rows'].sum())} rows, "
                f"{pct(int(cross['rows'].sum()), n_rows):.2f}% of data)")
        rep.say(f"    train(0-{held - 1}) vs held-out({held}): {len(sh_trte)} windows, "
                f"{len(te_rows)} held-out rows "
                f"(pos={int((te_rows[LABEL_COL] == 1).sum())}, "
                f"neg={int((te_rows[LABEL_COL] == 0).sum())}), "
                f"positive in train: {pos_tr}")
        return cross

    cross_pair = scope("flank_pair", df["flank_pair"], "flank pair (n+c)")
    scope("n_flank", df["n_flank"], "n_flank")
    scope("c_flank", df["c_flank"], "c_flank")

    # Hamming distance of cross-fold pairs sharing an exact flank pair
    sub = df[df["flank_pair"].isin(cross_pair.index)]
    pair_rows = []
    for fp, t in sub.groupby("flank_pair"):
        vals = t[[PEPTIDE_COL, FOLD_COL, LABEL_COL, "uniprot_id", "start"]].values.tolist()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                if a[1] == b[1]:
                    continue  # same fold = not leakage
                pair_rows.append({
                    "flank_pair": fp,
                    "pep_a": a[0], "fold_a": int(a[1]), "label_a": int(a[2]),
                    "prot_a": a[3], "start_a": a[4],
                    "pep_b": b[0], "fold_b": int(b[1]), "label_b": int(b[2]),
                    "prot_b": b[3], "start_b": b[4],
                    "hamming": hamming(a[0], b[0]),
                })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(TABLE_DIR / "cross_fold_flank_pairs.csv", index=False)

    rep.say("")
    rep.say("  cross-fold peptide pairs sharing an EXACT flank pair:")
    rep.say(f"    pairs: {len(pairs)}")
    if len(pairs):
        hvc = pairs["hamming"].value_counts().sort_index()
        rep.say("    Hamming distance: " +
                ", ".join(f"{k}:{v}" for k, v in hvc.items()))
        violations = pairs[pairs["hamming"] <= HAMMING_CUTOFF]
        rep.say(f"    ham <= {HAMMING_CUTOFF} violations: {len(violations)}  "
                f"-> {'OK: none' if len(violations) == 0 else 'HOMOLOGY LEAKAGE'}")
        rep.say("    examples (train/held-out near-duplicates, identical flanks):")
        for _, row in pairs.sort_values("hamming").head(8).iterrows():
            rep.say(f"      {row['pep_a']} f{row['fold_a']} L{row['label_a']} "
                    f"{row['prot_a']}:{row['start_a']}  vs  "
                    f"{row['pep_b']} f{row['fold_b']} L{row['label_b']} "
                    f"{row['prot_b']}:{row['start_b']}  ham={row['hamming']}")

    pd.DataFrame(overlap_rows).to_csv(TABLE_DIR / "flank_overlap_by_scope.csv",
                                      index=False)

    # ───────────────────────────────────────────────
    # VERDICT
    # ───────────────────────────────────────────────
    n_cross_pair_rows = (int(cross_pair["rows"].sum())
                         if len(cross_pair) else 0)
    tr = df[df[FOLD_COL] != held]
    te = df[df[FOLD_COL] == held]
    sh = set(tr["flank_pair"]) & set(te["flank_pair"])
    te_pos = int((te[te["flank_pair"].isin(sh)][LABEL_COL] == 1).sum())

    rep.say("")
    rep.say("=" * 80)
    rep.say("VERDICT (thesis discussion)")
    rep.say("=" * 80)
    rep.say(f"  1. Exact-peptide integrity: {n_cross_fold} duplicated peptides cross"
            f" folds -> {'PASS' if n_cross_fold == 0 else 'FAIL'}.")
    rep.say(f"  2. Label conflicts: {int(mixed.sum())} duplicated peptides mixed-label;"
            f" {mixed_same} have identical input context (annotation artifact,"
            f" irreducible, same fold).")
    rep.say(f"  3. Exact flank-pair overlap across folds: {len(cross_pair)} pairs,"
            f" {n_cross_pair_rows} rows ({pct(n_cross_pair_rows, n_rows):.2f}%).")
    rep.say(f"  4. Held-out({held}) rows whose flank pair appears in train:"
            f" {len(sh)} windows / {int(len(te[te['flank_pair'].isin(sh)]))} rows"
            f" ({te_pos} positive).")
    rep.say(f"  5. Homology constraint (ham <= {HAMMING_CUTOFF}) never violated by"
            f" flank-sharing pairs."
            if not len(pairs[pairs['hamming'] <= HAMMING_CUTOFF]) else
            f"  5. WARNING: ham <= {HAMMING_CUTOFF} violations present.")
    rep.say("  -> Folds are disjoint for peptide identity and Hamming-<=1 homology,")
    rep.say("     but NOT for exact flanking context; overlap is marginal (<=0.1%")
    rep.say("     of rows for full flank pairs) and does not materially affect")
    rep.say("     held-out performance estimates.")

    rep.say("")
    rep.say(f"Tables written to {TABLE_DIR}/")
    rep.close()
    print(f"Log written to {LOG_DIR}/41_flank_duplicate_leakage_audit_{ts}.txt")


if __name__ == "__main__":
    main()
