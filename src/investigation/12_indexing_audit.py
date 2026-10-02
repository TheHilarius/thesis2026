#!/usr/bin/env python3
"""
12_indexing_audit.py — Full indexing/coordinate audit of df_all.csv.

Validates every row (not a 200-row sample) of the coordinate, flank and
full_context columns against the protein sequence column, mirroring the
indexing logic that used to live in the ESM-C embedding script
(src/tools/esm/embed_peptides_esmc.py).

Dataset is locked to a single coordinate convention; the audit asserts that
convention (1idx_inclusive) and fails loudly if the data ever drifts.

Checks:
  1. Coordinate convention detection (all rows)
  2. Coordinate validation: prot[s0:e0] == peptide for every row
  3. Raw-flank validation vs protein sequence (terminus-aware)
  4. Padded-flank validation: n_flank/c_flank are X-padded to length 10
  5. full_context validation: len == 29, == n_flank+peptide+c_flank,
     X-tolerant match against prot[s0-10 : s0+19]
  6. Per-protein pad counts: n_pad_len/c_pad_len == max(10 - flank_len_real)

Output:
  Console summary
  data/processed/sequence_audit/indexing_audit_summary.txt
  data/processed/sequence_audit/indexing_audit_mismatches.tsv

Usage:
    python src/investigation/12_indexing_audit.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DF_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "df_all.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sequence_audit"

WINDOW_LEN = 29
FLANK_SIZE = 10
PEPTIDE_LEN = 9  # dataset is all 9-mers
EXPECTED_CONVENTION = "1idx_inclusive"

OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SHOWN = 5  # mismatches printed per check


# ── Helpers ───────────────────────────────────────────────────────────────────
def span_0idx(row, conv: str) -> tuple[int, int]:
    """Return (start_0, end_0_exclusive) for the peptide in the protein."""
    s, e = int(row["start"]), int(row["end"])
    return (s - 1, e) if conv == "1idx_inclusive" else (s, e)


def get_flank_raw(row, col: str) -> str:
    """Return raw flank string, or '' when NaN (terminus -> no flank)."""
    v = row[col]
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


# ── Check runners ─────────────────────────────────────────────────────────────
def check_convention(df, t0):
    print("=" * 65)
    print("  CHECK 1: COORDINATE CONVENTION (all rows)")
    print("=" * 65)
    n = len(df)
    a = b = 0
    for i, row in df.iterrows():
        prot, pep = str(row["sequence"]), str(row["peptide"])
        s, e = int(row["start"]), int(row["end"])
        if 0 <= s < e <= len(prot) and prot[s:e] == pep:
            a += 1
        if s >= 1 and e <= len(prot) and prot[s - 1 : e] == pep:
            b += 1
    print(f"  rows checked            : {n}")
    print(f"  0idx half-open  [s,e)  : {a} ({a/n*100:.2f}%)")
    print(f"  1idx inclusive [s,e]   : {b} ({b/n*100:.2f}%)")

    if b >= a:
        detected = "1idx_inclusive"
    elif a > b:
        detected = "0idx_halfopen"
    else:
        detected = None

    if detected != EXPECTED_CONVENTION:
        raise RuntimeError(
            f"Convention drift! Detected '{detected}' but expected "
            f"'{EXPECTED_CONVENTION}'. Data has changed — update the "
            f"embedding script constant AND this audit before continuing."
        )
    print(f"  detected convention     : {detected}")
    print(f"  expected convention     : {EXPECTED_CONVENTION}  [OK]")
    return detected, (a, b, n)


def check_coordinates(df, conv, t0):
    print("=" * 65)
    print("  CHECK 2: COORDINATE VALIDATION (prot[s0:e0] == peptide)")
    print("=" * 65)
    bad = []
    for i, row in df.iterrows():
        prot, pep = str(row["sequence"]), str(row["peptide"])
        s0, e0 = span_0idx(row, conv)
        if s0 < 0 or e0 > len(prot) or prot[s0:e0] != pep:
            bad.append((i, row["uniprot_id"], row["start"], row["end"],
                        prot[s0:e0], pep))
    print(f"  rows checked            : {len(df)}")
    print(f"  mismatches              : {len(bad)}")
    for i, uid, s, e, got, pep in bad[:MAX_SHOWN]:
        print(f"    row {i} {uid}: prot[{s}:{e}]='{got}' != '{pep}'")
    return bad


def check_raw_flanks(df, conv, t0):
    print("=" * 65)
    print("  CHECK 3: RAW FLANK VALIDATION (vs protein)")
    print("=" * 65)
    bad = []
    n_terminus = 0
    for i, row in df.iterrows():
        prot = str(row["sequence"])
        s0, e0 = span_0idx(row, conv)
        nfr = get_flank_raw(row, "n_flank_raw")
        cfr = get_flank_raw(row, "c_flank_raw")

        n_real = int(row["n_flank_len_real"]) if not pd.isna(row["n_flank_len_real"]) else 0
        c_real = int(row["c_flank_len_real"]) if not pd.isna(row["c_flank_len_real"]) else 0

        nf_ok = prot[s0 - len(nfr) : s0] == nfr if nfr else n_real == 0
        cf_ok = prot[e0 : e0 + len(cfr)] == cfr if cfr else c_real == 0

        if nfr == "":
            n_terminus += 1

        if not (nf_ok and cf_ok):
            bad.append((i, row["uniprot_id"], row["start"], row["end"],
                        repr(nfr), repr(cfr)))
    print(f"  rows checked            : {len(df)}")
    print(f"  rows w/ N-terminus (no raw n_flank): {n_terminus}")
    print(f"  mismatches              : {len(bad)}")
    for i, uid, s, e, nfr, cfr in bad[:MAX_SHOWN]:
        print(f"    row {i} {uid}: start={s} end={e} nfr={nfr} cfr={cfr}")
    return bad


def check_padded_flanks(df, t0):
    print("=" * 65)
    print("  CHECK 4: PADDED FLANK VALIDATION (X-pad to 10)")
    print("=" * 65)
    bad = []
    for i, row in df.iterrows():
        nf, cf = str(row["n_flank"]), str(row["c_flank"])
        nfr = get_flank_raw(row, "n_flank_raw")
        cfr = get_flank_raw(row, "c_flank_raw")

        exp_nf = "X" * FLANK_SIZE if not nfr else "X" * (FLANK_SIZE - len(nfr)) + nfr
        exp_cf = "X" * FLANK_SIZE if not cfr else cfr + "X" * (FLANK_SIZE - len(cfr))

        if len(nf) != FLANK_SIZE or len(cf) != FLANK_SIZE:
            bad.append((i, "bad_length", repr(nf), repr(cf)))
            continue
        if nf != exp_nf or cf != exp_cf:
            bad.append((i, "bad_padding", repr(nf), repr(cf),
                        repr(exp_nf), repr(exp_cf)))
    print(f"  rows checked            : {len(df)}")
    print(f"  mismatches              : {len(bad)}")
    for b in bad[:MAX_SHOWN]:
        print(f"    row {b[0]}: {b[1]}  nf={b[2]} cf={b[3]}")
        if len(b) > 4:
            print(f"        expected nf={b[4]} cf={b[5]}")
    return bad


def check_full_context(df, conv, t0):
    print("=" * 65)
    print("  CHECK 5: FULL_CONTEXT VALIDATION (29-mer)")
    print("=" * 65)
    bad_len = 0
    bad_comp = []
    bad_prot = []
    for i, row in df.iterrows():
        fc = str(row["full_context"])
        pep = str(row["peptide"])
        nf, cf = str(row["n_flank"]), str(row["c_flank"])

        if len(fc) != WINDOW_LEN:
            bad_len += 1
            continue
        if fc != nf + pep + cf:
            bad_comp.append((i, row["uniprot_id"], repr(fc),
                             repr(nf + pep + cf)))
            continue

        prot = str(row["sequence"])
        s0, _e0 = span_0idx(row, conv)
        # Reconstruct expected 29-mer from the protein, X-padding at termini
        # the same way R's extract_flanks_safe does:
        #   n_flank: real part prot[max(0,s0-10):s0], X-left-padded to 10
        #   c_flank: real part prot[s0+9:min(len,s0+19)], X-right-padded to 10
        nf_real = prot[max(0, s0 - FLANK_SIZE):s0]
        nf_exp = "X" * (FLANK_SIZE - len(nf_real)) + nf_real
        cf_real = prot[s0 + PEPTIDE_LEN:min(len(prot), s0 + PEPTIDE_LEN + FLANK_SIZE)]
        cf_exp = cf_real + "X" * (FLANK_SIZE - len(cf_real))
        exp_fc = nf_exp + prot[s0:s0 + PEPTIDE_LEN] + cf_exp
        if exp_fc != fc:
            bad_prot.append((i, "protein_window_mismatch", repr(fc),
                             repr(exp_fc)))
    print(f"  rows checked            : {len(df)}")
    print(f"  len != {WINDOW_LEN}             : {bad_len}")
    print(f"  != n_flank+pep+c_flank : {len(bad_comp)}")
    print(f"  protein window mismatch: {len(bad_prot)}")
    for b in bad_comp[:MAX_SHOWN]:
        print(f"    row {b[0]} {b[1]}: fc={b[2]}  expected={b[3]}")
    for b in bad_prot[:MAX_SHOWN]:
        print(f"    row {b[0]}: {b[1]}")
        if len(b) > 2:
            print(f"        fc={b[2]}  prot_win={b[3]}")
    return bad_len, bad_comp, bad_prot


def check_pad_counts(df, t0):
    print("=" * 65)
    print("  CHECK 6: PER-PROTEIN PAD COUNTS (n_pad_len/c_pad_len)")
    print("=" * 65)
    bad = []
    n_proteins = df["uniprot_id"].nunique()
    for uid, grp in df.groupby("uniprot_id"):
        n_fl = grp["n_flank_len_real"].astype(int)
        c_fl = grp["c_flank_len_real"].astype(int)
        exp_n = (FLANK_SIZE - n_fl).clip(lower=0).max()
        exp_c = (FLANK_SIZE - c_fl).clip(lower=0).max()
        got_n = int(grp["n_pad_len"].iloc[0])
        got_c = int(grp["c_pad_len"].iloc[0])
        if got_n != exp_n or got_c != exp_c:
            bad.append((uid, got_n, got_c, int(exp_n), int(exp_c)))
    print(f"  proteins checked        : {n_proteins}")
    print(f"  mismatches              : {len(bad)}")
    for uid, gn, gc, en, ec in bad[:MAX_SHOWN]:
        print(f"    {uid}: got n={gn} c={gc}  expected n={en} c={ec}")
    return bad


def main():
    print("=" * 65)
    print("  INDEXING AUDIT — df_all.csv")
    print("=" * 65)
    t0 = time.time()

    print(f"\nLoading {DF_ALL_PATH}...")
    df = pd.read_csv(DF_ALL_PATH)
    print(f"  rows: {len(df)}")
    print(f"  proteins: {df['uniprot_id'].nunique()}")
    print(f"  pep_lengths: {df['pep_length'].value_counts().to_dict()}")

    results = {}

    conv, conv_counts = check_convention(df, t0)
    results["convention"] = conv_counts

    bad_coords = check_coordinates(df, conv, t0)
    results["coordinates"] = len(bad_coords)

    bad_raw = check_raw_flanks(df, conv, t0)
    results["raw_flanks"] = len(bad_raw)

    bad_pad = check_padded_flanks(df, t0)
    results["padded_flanks"] = len(bad_pad)

    bad_len, bad_comp, bad_prot = check_full_context(df, conv, t0)
    results["fc_length"] = bad_len
    results["fc_composition"] = len(bad_comp)
    results["fc_protein"] = len(bad_prot)

    bad_padcounts = check_pad_counts(df, t0)
    results["pad_counts"] = len(bad_padcounts)

    # ── Save mismatches ───────────────────────────────────────────────────────
    mismatch_rows = []
    for i, uid, s, e, got, pep in bad_coords:
        mismatch_rows.append({"check": "coordinates", "row": i,
                              "uniprot_id": uid, "start": s, "end": e,
                              "got": got, "expected": pep})
    for i, uid, s, e, nfr, cfr in bad_raw:
        mismatch_rows.append({"check": "raw_flanks", "row": i,
                              "uniprot_id": uid, "start": s, "end": e,
                              "got": nfr, "expected": cfr})
    for b in bad_pad:
        rec = {"check": "padded_flanks", "row": b[0], "detail": b[1],
               "got": f"nf={b[2]} cf={b[3]}"}
        if len(b) > 4:
            rec["expected"] = f"nf={b[4]} cf={b[5]}"
        mismatch_rows.append(rec)
    for b in bad_comp:
        mismatch_rows.append({"check": "fc_composition", "row": b[0],
                              "uniprot_id": b[1], "got": b[2],
                              "expected": b[3]})
    for b in bad_prot:
        rec = {"check": "fc_protein", "row": b[0], "detail": b[1]}
        if len(b) > 2:
            rec["got"] = b[2]
            rec["expected"] = b[3]
        mismatch_rows.append(rec)
    for uid, gn, gc, en, ec in bad_padcounts:
        mismatch_rows.append({"check": "pad_counts", "uniprot_id": uid,
                              "got": f"n={gn} c={gc}",
                              "expected": f"n={en} c={ec}"})

    mismatch_path = OUT_DIR / "indexing_audit_mismatches.tsv"
    if mismatch_rows:
        pd.DataFrame(mismatch_rows).to_csv(mismatch_path, sep="\t", index=False)
    else:
        # Write a header-only file so the artifact always exists
        pd.DataFrame(columns=["check", "row", "uniprot_id", "start", "end",
                              "got", "expected", "detail"]).to_csv(
            mismatch_path, sep="\t", index=False)
    print(f"\nSaved mismatches: {mismatch_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    a, b, n = conv_counts
    summary = [
        "=" * 50,
        "  INDEXING AUDIT SUMMARY",
        "=" * 50,
        "",
        f"Dataset: {DF_ALL_PATH}",
        f"  rows: {len(df)}, proteins: {df['uniprot_id'].nunique()}",
        "",
        f"1. Convention:        {conv}",
        f"   0idx half-open:    {a} ({a/n*100:.2f}%)",
        f"   1idx inclusive:    {b} ({b/n*100:.2f}%)",
        "",
        f"2. Coordinate mismatches:   {results['coordinates']}",
        f"3. Raw-flank mismatches:    {results['raw_flanks']}",
        f"4. Padded-flank mismatches: {results['padded_flanks']}",
        f"5. full_context:",
        f"   len != {WINDOW_LEN}:      {results['fc_length']}",
        f"   composition:             {results['fc_composition']}",
        f"   protein window:          {results['fc_protein']}",
        f"6. Pad-count mismatches:    {results['pad_counts']}",
        "",
        f"Total mismatch rows:        {len(mismatch_rows)}",
        f"Runtime: {time.time() - t0:.1f}s",
        "=" * 50,
    ]
    summary_text = "\n".join(summary)

    summary_path = OUT_DIR / "indexing_audit_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved summary: {summary_path}")

    print(f"\n{summary_text}")
    print("\nDone.")


if __name__ == "__main__":
    main()